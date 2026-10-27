#!/usr/bin/env python3
"""Catalogue VBI / TBC captures: which programme was on air while the tape was recorded.

usage: catalog.py [-o catalog.csv] [--survey-root DIR ...] [--work DIR] [-j N]
                  [--schedule FILE ...] [--details] [--gaps FILE] INPUT ...

INPUT is a capture (.flac, .vbi, .tbc, .tbcy; a chroma file is skipped when its luma file
is present), a folder of captures, or a survey folder written by survey.py.

For every capture the survey output (packets.bin, vps.txt, l16.txt) is taken from an
existing survey folder <survey-root>/<name>/ (same naming as run_all.sh) or, if there is
none, written by survey.py into <work>/<name>/ first. From it the tool reads
  * the teletext header clock and date of every header row (row 0) and of the early
    packet 8/30, as many samples over the whole capture; a recording consists of one or
    more segments in which clock minus capture position stays constant (a jump means the
    recorder was paused or an older recording on the tape shows through);
  * the channel: VPS CNI, data line 16 word 3 (4-bit source code) and word 4 (plain-text
    source, docs/FINDINGS.md 3.2/3.3), the East German programme number on line 16
    (FINDINGS 5), the teletext service in the header and, as a last resort, the file name;
  * the programme schedule from the teletext pages: pages with at least three
    "hh.mm title" rows (ARD/ZDF: 160 ff, 170 ff, 180-185 in the early years, 300 ff
    later; other services wherever they are), majority-voted per segment;
  * the VPS label (announced start time) of the programme on line 16;
  * for Pro 7 recordings without teletext clock: the clock of the Channel VideoDat data
    service, decoded with channel-videodat-decoder (found next to this repository, in
    ~/git/repos_itewreed/ or via $VIDEODAT_DECODER): date and time packets every 5 s in the
    1993 variant (sampled), the plain-text clock in the 1990 variant (whole capture).
Every programme of the schedule that overlaps a segment is written as one CSV row. Where
the teletext has no schedule (the third programmes often list only from 18:00), entries
from --schedule files are used (default docs/schedules.csv; Sender;Datum;Uhrzeit;Sendung
with the channel names of this tool, e.g. "N3", "West 3", "Datum" as dd.mm.yyyy, taken
from historical TV listings). Historical listings are not searchable online by date and
channel, so --gaps writes the recordings that are still unknown (channel, date, time,
VPS label) as a list to look up.

CSV (semicolon separated, UTF-8):
  Collection;Tape;Sendung;Sender;Datum;Uhrzeit;Aufnahme_von;Aufnahme_bis;Datei
  Collection  the number after "coll" in the file name, 0 if there is none
  Tape        the three-digit number after "tape" or "tape_"
  Uhrzeit     start of the programme as listed (hh:mm), or the start of the recording
              (hh:mm:ss) when the programme is unknown
  Aufnahme_von, Aufnahme_bis  clock time of the recorded part of the programme
  Datei       the capture file
--details adds Position (h:mm:ss in the capture) and Quelle (where the programme and channel
came from).
"""
import argparse, csv, datetime as dt, glob, json, os, re, subprocess, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vps, dataline
from wst2 import HAM, HERR

FPS = 25
TOOLS = os.path.dirname(os.path.abspath(__file__))
CAPTURE_EXT = ('.flac', '.vbi', '.tbc', '.tbcy', '.tbcc')
PAR = np.array([bin(i).count('1') & 1 for i in range(256)], bool)     # odd parity = valid
G0 = {0x23: '#', 0x24: '$', 0x40: '§', 0x5b: 'Ä', 0x5c: 'Ö', 0x5d: 'Ü', 0x5e: '^', 0x5f: '_', 0x60: '°',
      0x7b: 'ä', 0x7c: 'ö', 0x7d: 'ü', 0x7e: 'ß', 0x7f: ' '}
BAD = '~'                                                             # character with a parity error

# ---------------------------------------------------------------- channels
# VPS CNI (12 bit) of the German networks, from ETSI TR 101 231 (as in zvbi's network table),
# with the names the channels had at the time of the tapes
CNI = {
    0xDC1: 'ARD', 0xDC2: 'ZDF', 0xDC3: 'ARD/ZDF', 0xDC4: 'ARD', 0xDC5: 'ARD',
    0xDC7: '3sat', 0xDC8: 'Phoenix', 0xDC9: 'Kinderkanal', 0xDCA: 'ARD (BR regional)',
    0xDCB: 'Bayern 3', 0xDCC: 'Bayern 3', 0xDCD: 'Bayern 3', 0xDCE: 'ARD (HR regional)', 0xDCF: 'Hessen 3',
    0xDD0: 'ARD (NDR regional)', 0xDD1: 'ARD (NDR regional)', 0xDD2: 'ARD (NDR regional)',
    0xDD3: 'ARD (NDR regional)', 0xDD4: 'N3', 0xDD5: 'N3', 0xDD6: 'N3', 0xDD7: 'N3', 0xDD8: 'N3',
    0xDD9: 'ARD (RB regional)', 0xDDA: 'N3', 0xDDB: 'ARD (SFB regional)', 0xDDC: 'SFB 3',
    0xDDD: 'ARD (SDR/SWF regional)', 0xDDE: 'ARD (SWF regional)', 0xDDF: 'ARD (SR regional)',
    0xDE0: 'Südwest 3', 0xDE1: 'Südwest 3', 0xDE2: 'Südwest 3', 0xDE3: 'Südwest 3', 0xDE4: 'Südwest 3',
    0xDE5: 'ARD (WDR regional)', 0xDE6: 'West 3', 0xDE7: 'West 3', 0xDE8: 'West 3', 0xDE9: 'West 3',
    0xDEA: 'West 3', 0xDEB: 'West 3', 0xDEC: 'Südwest 3', 0xDED: 'Südwest 3', 0xDEE: 'Südwest 3',
    0xDEF: 'Südwest 3', 0xD85: 'arte', 0xD8E: 'VOX', 0xD8A: 'Super RTL', 0xD8C: 'n-tv', 0xD8D: 'DSF',
    0xD8F: 'RTL 2', 0xD91: 'Eurosport', 0xD92: 'Kabel 1', 0xD94: 'Pro 7', 0xDAB: 'RTL', 0xDAC: 'Premiere',
    0xDB9: 'SAT.1', 0xDBB: 'Deutsche Welle TV', 0xD78: 'Tele 5', 0xD82: 'ORB', 0xD81: 'ARD (ORB regional)',
}
for c in range(0xD9A, 0xDAB):
    CNI[c] = 'RTL'
for c in list(range(0xD95, 0xD9A)) + list(range(0xDAD, 0xDB9)):
    CNI[c] = 'SAT.1'
# data line word 3 (low nibble) -> ARD station; 5 = ZDF, 0 = private / cable (FINDINGS 3.2)
W3_STATION = {0x1: 'SWF', 0x4: 'BR', 0x5: 'ZDF', 0x6: 'SFB', 0x8: 'HR', 0x9: 'WDR', 0xA: 'SR', 0xC: 'NDR',
              0xE: 'SDR'}
# third programme of each ARD station
DRITTE = {'BR': 'Bayern 3', 'HR': 'Hessen 3', 'SWF': 'Südwest 3', 'SDR': 'Südwest 3', 'SR': 'Südwest 3',
          'WDR': 'West 3', 'NDR': 'N3', 'SFB': 'SFB 3'}
# data line word 4 texts that name a channel directly
W4_CHANNEL = {'ZDF': 'ZDF', 'SAT1*LU1': 'SAT.1', 'RIAS-TV': 'RIAS-TV'}
# teletext services in the header row: (regex, channel or None if the service is shared)
FAMILIES = [
    ('ARD/ZDF', r'A.D.ZDF|AR./ZDF|ARD/Z.F|ARD.Z.F', None),
    ('NORDTEXT', r'NORDTEXT|NORDT.XT|N.RDTEXT', 'N3'),
    ('BAYERNTEXT', r'BAYERNTEXT|BAYERNT.XT|BAY.RNTEXT', 'Bayern 3'),
    ('WDR-Text', r'WDR-Text|WDR-T.xt|W.R-Text', 'West 3'),
    ('HESSENTEXT', r'HESSENTEXT|HR-TEXT|HR-Text', 'Hessen 3'),
    ('SÜDWESTTEXT', r'S.DWEST.?TEXT|SWF.?TEXT|S3-TEXT', 'Südwest 3'),
    ('TSVT', r'TSVT', None),
    ('SAT.1-text', r'SAT.1.?[Tt]ext|SAT1TEXT|SAT.1', 'SAT.1'),
    ('RTL 2', r'RTL ?2|RTL ?II|RTL2', 'RTL 2'),
    ('Super RTL', r'SUPER ?RTL|Super ?RTL', 'Super RTL'),
    ('RTL-Text', r'RTL-?Text|RTL.?plus|RTLplus|RTL WORLD|RTL W.RLD', 'RTL'),
    ('PRO 7', r'PRO ?7|ProSieben', 'Pro 7'),
    ('TELE 5', r'TELE ?5', 'Tele 5'),
    ('PK Berlin', r'PK B.rl.n|P. Berlin', None),
    ('BerlinText', r'Berl.n ?Text|B.rlinText|BerlinT.xt', 'B1'),
    ('SFB', r'SFB/ZENTRALE', 'SFB 3'),
    ('DFF', r'\bDFF\b', None),
    ('DDR-Test', r'TECHNISCHER VERSUCH', None),
    ('ORB/MDR', r'ORB.MDR', None),                  # shared teletext of ORB and MDR in the mid-1990s
    ('MDR', r'\bMDR\b', 'MDR'),
    ('ORB', r'\bORB\b', 'ORB'),
    ('3sat', r'3sat|3SAT', '3sat'),
    ('Kabel 1', r'KABEL ?1|Kabel ?1', 'Kabel 1'),
    ('INFOTEXT', r'INFOTEXT', None),                 # Telekom info service on Kabelkanal, no programme
    ('VOX', r'\bVOX\b|\bvox\b', 'VOX'),
    ('n-tv', r'\bn-tv\b|\bN-TV\b', 'n-tv'),
    ('DSF', r'\bDSF\b', 'DSF'),
    ('Eurosport', r'EUROSPORT|Eurosport', 'Eurosport'),
    ('arte', r'\barte\b|\bARTE\b', 'arte'),
    ('Phoenix', r'PHOENIX|phoenix', 'Phoenix'),
    ('Kinderkanal', r'KI\.KA|KIKA|Kinderkanal', 'Kinderkanal'),
    ('9Live', r'9 ?LIVE|9 ?Live|NEUN ?LIVE', '9Live'),
    ('Premiere', r'PREMIERE|Premiere', 'Premiere'),
]
FAMILY_RE = [(n, re.compile(r), c) for n, r, c in FAMILIES]
FAMILY_CHANNEL = {n: c for n, r, c in FAMILIES}
FAMILY_CHANNEL['VideoDat'] = 'Pro 7'                 # clock of the Pro 7 data service, see videodat_clock()
# file name channel words
FILE_CHANNEL = {
    'ard': 'ARD', 'zdf': 'ZDF', 'ndr': 'N3', 'n3': 'N3', 'br': 'Bayern 3', 'wdr': 'West 3', 'hr': 'Hessen 3',
    'swf': 'Südwest 3', 'sdr': 'Südwest 3', 'sfb': 'SFB 3', 'b1': 'B1', 'rtl': 'RTL', 'rtlplus': 'RTL',
    'sat1': 'SAT.1', 'pro7': 'Pro 7', 'tele5': 'Tele 5', 'kabel1': 'Kabel 1', 'ddr1': 'DDR 1',
    'ddr2': 'DDR 2', 'dff': 'DFF', 'dff1': 'DFF 1', 'dff2': 'DFF 2', 'mdr': 'MDR', 'premiere': 'Premiere',
    'teleclub': 'Teleclub', '3sat': '3sat', 'vox': 'VOX', 'mkb': 'Kabel Berlin', 'pkberlin': 'Kabel Berlin',
    'rias': 'RIAS-TV', 'tsvt': None, 'ardzdf': None, 'orb': 'ORB', 'orbmdr': None, 'kabelkanal': 'Kabel 1',
    'rtl2': 'RTL 2', 'superrtl': 'Super RTL', 'ntv': 'n-tv', 'n-tv': 'n-tv', 'dsf': 'DSF', 'arte': 'arte',
    'kika': 'Kinderkanal', 'phoenix': 'Phoenix', '9-live': '9Live', '9live': '9Live', 'eurosport': 'Eurosport',
}
DRITTE_OF_FILE = {'ndr': 'NDR', 'n3': 'NDR', 'br': 'BR', 'wdr': 'WDR', 'hr': 'HR', 'swf': 'SWF', 'sdr': 'SDR',
                  'sfb': 'SFB'}

MONTHS = {'jan': 1, 'feb': 2, 'mrz': 3, 'mär': 3, 'mar': 3, 'apr': 4, 'mai': 5, 'may': 5, 'jun': 6, 'jul': 7,
          'aug': 8, 'sep': 9, 'okt': 10, 'oct': 10, 'nov': 11, 'dez': 12, 'dec': 12}
WEEKDAYS = {'mo': 0, 'mon': 0, 'di': 1, 'die': 1, 'tue': 1, 'mi': 2, 'mit': 2, 'wed': 2, 'do': 3, 'don': 3,
            'thu': 3, 'fr': 4, 'fre': 4, 'fri': 4, 'sa': 5, 'sam': 5, 'sat': 5, 'so': 6, 'son': 6, 'sun': 6}


# ---------------------------------------------------------------- helpers
def chars(bs):
    """Teletext bytes as text; characters with a parity error become '~'."""
    return ''.join((G0.get(c & 0x7f, chr(c & 0x7f) if 0x20 <= (c & 0x7f) < 0x7f else ' ') if PAR[c] else BAD)
                   for c in bs)


def hms(s, sec=True):
    s = int(round(s)) % 86400
    return '%02d:%02d:%02d' % (s // 3600, s // 60 % 60, s % 60) if sec else '%02d:%02d' % (s // 3600, s // 60 % 60)


def pos(f):
    s = int(f / FPS)
    return '%d:%02d:%02d' % (s // 3600, s // 60 % 60, s % 60)


def name_of(path):
    """Survey folder name of a capture, as in run_all.sh."""
    b = os.path.basename(path)
    for suf in ('_vbi.flac', '.flac', '.tbcy', '.tbcc', '.tbc', '.vbi'):
        if b.endswith(suf):
            return b[:-len(suf)]
    return b


def from_filename(path):
    b = os.path.basename(path)
    m = re.search(r'coll(\d+)', b, re.I)
    coll = m.group(1) if m else '0'
    m = re.search(r'tape_?(\d{3})', b, re.I)
    tape = m.group(1) if m else ''
    rest = re.sub(r'^.*?tape_?\d{3}(_\d+)?_?', '', name_of(b))
    tok = rest.split('_')
    chan = tok[0].lower() if tok else ''
    if chan not in FILE_CHANNEL:
        known = [c for c in re.split(r'[-()]', chan) if c in FILE_CHANNEL and FILE_CHANNEL[c]]
        chan = known[0] if known else chan
    m = re.search(r'((?:19|20)\d[\dx])-(\d\d|xx)-(\d\d|xx)', b)
    date = m.groups() if m else None
    words = [t for t in tok[1:] if not re.match(r'^((19|20)\d[\dx](-.*)?|\d{4}\.|sp|lp|ep|fs\d+|hd\d+|hs\d+|take\d|vbi|nc|'
                                               r'hifi|a|chroma|sweep|tests?|montag|dienstag|mittwoch|donnerstag|'
                                               r'freitag|samstag|sonntag|xx.*|\d{4}-.*)$', t, re.I)]
    title = ' '.join(w.replace('-', ' ') for w in words if re.search('[a-z]', w, re.I))
    return dict(coll=coll, tape=tape, chan=chan, date=date, title=title)


# ---------------------------------------------------------------- header rows
CLOCK = re.compile(r'([0-2]\d)[:.]([0-5]\d)[:.]([0-5]\d)')
DATE_NUM = re.compile(r'(?<![\d:])(\d{1,2})[.](\d{2})(?:[.](\d{2})?)?(?![:\d])')
DATE_MON = re.compile(r'(?<!\d)(\d{1,2})\.? ?(Jan|Feb|Mrz|Mär|Mar|Apr|Mai|May|Jun|Jul|Aug|Sep|Okt|Oct|Nov|Dez|Dec)')
WDAY = re.compile(r'(?<![A-Za-z])(Mo|Di|Mi|Do|Fr|Sa|So|Mon|Die|Mit|Don|Fre|Sam|Son|Tue|Wed|Thu|Fri|Sat|Sun)'
                  r'(?![a-z])')


def parse_header(s):
    """Clock (seconds of day), day, month, 2-digit year, weekday and service of a header text."""
    r = dict(sec=None, day=None, mon=None, yy=None, wd=None, fam=None)
    cl = list(CLOCK.finditer(s))
    if cl:
        m = cl[-1]
        h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if h < 24:
            r['sec'] = h * 3600 + mi * 60 + se
        s_wo = s[:m.start()]
    else:
        s_wo = s
    m = DATE_MON.search(s_wo)
    if m:
        r['day'], r['mon'] = int(m.group(1)), MONTHS[m.group(2).lower()]
    else:
        for m in DATE_NUM.finditer(s_wo):
            d, mo = int(m.group(1)), int(m.group(2))
            if 1 <= d <= 31 and 1 <= mo <= 12:
                r['day'], r['mon'] = d, mo
                if m.group(3):
                    r['yy'] = int(m.group(3))
                break
    m = WDAY.search(s_wo[3:])
    if m:
        r['wd'] = WEEKDAYS[m.group(1).lower()]
    for n, rx, c in FAMILY_RE:
        if rx.search(s):
            r['fam'] = n
            break
    else:
        # unknown service: the name between page number and date, e.g. "KABEL 1"
        m = re.match(r'^.{3,4}\s+([A-Za-zÄÖÜäöü][\w./ -]{1,14}?)\s{2,}', s)
        if m and BAD not in m.group(1):
            r['fam'] = m.group(1).strip().upper().replace(' ', '')
    return r


# ---------------------------------------------------------------- reading a survey folder
def load_packets(d):
    raw = np.fromfile(os.path.join(d, 'packets.bin'), np.uint8)
    raw = raw[:len(raw) // 51 * 51].reshape(-1, 51)
    fr = raw[:, :4].copy().view('<u4').ravel()
    rows = raw[:, 4]
    P = raw[:, 6:]
    ok = (HERR[P[:, 3]] == 0) & (HERR[P[:, 4]] == 0)
    # rows that carry teletext (not stray decodes on the data line or other signals)
    cnt = Counter(rows[ok].tolist())
    good = [r for r, c in cnt.items() if c >= max(50, 0.02 * ok.sum())]
    ok &= np.isin(rows, good)
    fr, P = fr[ok], P[ok]
    mag = HAM[P[:, 3]] & 7
    mag[mag == 0] = 8
    pkt = (HAM[P[:, 3]] >> 3) | (HAM[P[:, 4]] << 1)
    return fr, mag, pkt, P


def read_teletext(d):
    """Header samples [(frame, parsed header, page)] and page rows {(mag,page,sub): [(frame,row,bytes)]}."""
    fr, mag, pkt, P = load_packets(d)
    samples, pages, cur = [], defaultdict(list), {}
    for i in range(len(P)):
        m, n, p = int(mag[i]), int(pkt[i]), P[i]
        if n == 0:
            u, t = HAM[p[5]], HAM[p[6]]
            s = HAM[p[7:13]]
            if min(u, t, *s) < 0:
                cur.pop(m, None)
                continue
            page = t << 4 | u
            sub = int(s[0] | (s[1] & 7) << 4 | s[2] << 8 | (s[3] & 3) << 12)
            if page == 0xFF:
                cur.pop(m, None)
                continue
            cur[m] = (m, page, sub)
            pages[cur[m]].append((int(fr[i]), 0, p[13:45]))
            h = parse_header(chars(p[13:45]))
            if h['sec'] is not None or h['day'] is not None or h['fam']:
                samples.append((int(fr[i]), h, m * 256 + page))
        elif n <= 25:
            if m in cur:
                pages[cur[m]].append((int(fr[i]), n, p[5:45]))
        elif n == 30 and m == 8 and HAM[p[5]] in (0, 1):
            # early packet 8/30 carries the header text with date and time (FINDINGS 7)
            h = parse_header(chars(p[6:45]))
            if h['sec'] is not None:
                samples.append((int(fr[i]), h, -1))
    return samples, pages


def read_dataline(d):
    """Stretches of line 16 [(f0, f1, word3, word4 text, cni)] and VPS labels [(f0, f1, (day,mon,h,m))]."""
    src = os.path.join(d, 'vps.txt')
    recs = []
    if os.path.exists(src):
        for l in open(src):
            p = l.split()
            if len(p) >= 3 and p[1] == '9':
                recs.append((int(p[0]), bytes.fromhex(p[2])))
    segs = dataline.segments(recs) if recs else []
    labels = []
    for f, b in recs:
        q = vps.parse(b)
        if 1 <= q['day'] <= 31 and 1 <= q['mon'] <= 12 and q['hour'] < 24 and q['min'] < 60:
            labels.append((f, (q['day'], q['mon'], q['hour'], q['min'])))
    runs = []
    for k in range(0, len(labels), 25):              # vote per second, join equal seconds
        grp = labels[k:k + 25]
        v, c = Counter(x[1] for x in grp).most_common(1)[0]
        if c < len(grp) * 0.6:
            continue
        if runs and runs[-1][2] == v and grp[0][0] - runs[-1][1] < 750:
            runs[-1][1] = grp[-1][0]
        else:
            runs.append([grp[0][0], grp[-1][0], v])
    runs = [r for r in runs if r[1] - r[0] >= 5 * FPS]
    return segs, runs


def read_ddr(d):
    """East German programme number on line 16 [(frame, 1|2)] (FINDINGS 5)."""
    src = os.path.join(d, 'l16.txt')
    out = []
    if os.path.exists(src):
        for l in open(src):
            p = l.split()
            if len(p) == 3 and p[1] == '9' and len(p[2]) >= 35 and p[2].startswith('11111111' + '1010'):
                n = {'100': 1, '010': 2}.get(p[2][32:35])
                if n:
                    out.append((int(p[0]), n))
    return out


# ---------------------------------------------------------------- segments
def unwrap(off, ref):
    while off - ref > 43200:
        off -= 86400
    while off - ref < -43200:
        off += 86400
    return off


def segments(samples, nframes, tol=4.0, need=20):
    if samples and all(pg == -2 for f, h, pg in samples):
        tol, need = 15.0, 2                          # VideoDat: few samples, clock drifts against the frames
    """Split the capture where clock minus position jumps or the teletext service changes.
    A segment: dict(f0, f1, off, samples) with clock = off + frame / 25 (seconds, may pass 86400)."""
    timed = [(f, h) for f, h, pg in samples if h['sec'] is not None]
    segs, cur, pend = [], None, []
    def same(a, b):
        return abs(unwrap(a[0], b[0]) - b[0]) <= tol and (a[1] is None or b[1] is None or a[1] == b[1])
    for f, h in timed:
        o = (h['sec'] - f / FPS, h['fam'])
        if cur is not None and same(o, (cur['off'], cur['fam'])):
            cur['offs'].append(unwrap(o[0], cur['off']))
            cur['f1'] = f
            cur['fam_c'][h['fam']] += h['fam'] is not None
            if len(cur['offs']) % 16 == 0:
                cur['off'] = float(np.median(cur['offs'][-64:]))
            pend = []
            continue
        pend = [x for x in pend if same(o, x[1])] + [(f, o, h)]
        if len(pend) >= need:
            fam = Counter(x[1][1] for x in pend if x[1][1]).most_common(1)
            cur = dict(f0=pend[0][0], f1=f, off=float(np.median([x[1][0] for x in pend])),
                       fam=fam[0][0] if fam else None, offs=[], fam_c=Counter())
            for x in pend:
                cur['offs'].append(unwrap(x[1][0], cur['off']))
            segs.append(cur)
            pend = []
    for s in segs:
        s['off'] = float(np.median(s['offs']))
        if s['fam'] is None and s['fam_c']:
            s['fam'] = s['fam_c'].most_common(1)[0][0]
    # join neighbours that are the same recording (an interruption of the teletext only)
    out = []
    for s in segs:
        if out and abs(unwrap(s['off'], out[-1]['off']) - out[-1]['off']) <= tol and \
                (s['fam'] == out[-1]['fam'] or None in (s['fam'], out[-1]['fam'])):
            out[-1]['f1'] = s['f1']
            out[-1]['fam'] = out[-1]['fam'] or s['fam']
        else:
            out.append(s)
    # extend each segment to the start of the next one; drop segments shorter than 10 s
    out = [s for s in out if s['f1'] - s['f0'] >= 10 * FPS]
    for k, s in enumerate(out):
        s['f0'] = 0 if k == 0 else s['f0']
        s['f1'] = out[k + 1]['f0'] - 1 if k + 1 < len(out) else nframes - 1
    if not out:
        fam = Counter(h['fam'] for f, h, pg in samples if h['fam']).most_common(1)
        out = [dict(f0=0, f1=nframes - 1, off=None, fam=fam[0][0] if fam else None)]
    for s in out:
        s.pop('offs', None); s.pop('fam_c', None)
    return out


def segment_date(seg, samples, fn):
    """Calendar date of clock second 0 of the segment, from the headers (day, month, year,
    weekday), the file name and the weekday. Returns (date or None, note)."""
    days, years, wds = Counter(), Counter(), Counter()
    for f, h, pg in samples:
        if not seg['f0'] <= f <= seg['f1'] or h['day'] is None:
            continue
        k = 0
        if seg['off'] is not None and h['sec'] is not None:
            t = seg['off'] + f / FPS
            if abs(unwrap(h['sec'] - f / FPS, seg['off']) - seg['off']) > (15 if pg == -2 else 4):
                continue                         # header from another recording
            k = int(t // 86400)
        days[(h['day'], h['mon'], k)] += 1
        if h['yy'] is not None:
            years[(h['day'], h['mon'], k, h['yy'])] += 1
        if h['wd'] is not None:
            wds[(h['wd'] - k) % 7] += 1
    if not days:
        return None, ''
    (d, m, k), _ = days.most_common(1)[0]
    wd = wds.most_common(1)[0][0] if wds else None
    cands = []
    years = Counter({key[3]: n for key, n in years.items() if key[:3] == (d, m, k)})
    if years:
        yy = years.most_common(1)[0][0]
        cands = [(1900 + yy if yy >= 50 else 2000 + yy)]
    else:
        fy = fn['date'][0] if fn['date'] else None
        if fy and 'x' not in fy:
            cands = [int(fy)]
        else:
            lo, hi = (int(fy[:3]) * 10, int(fy[:3]) * 10 + 9) if fy else (1979, 2009)
            cands = list(range(lo, hi + 1))
    for y in cands:
        try:
            date = dt.date(y, m, d) - dt.timedelta(days=k)
        except ValueError:
            continue
        if len(cands) == 1 or (wd is not None and date.weekday() == wd):
            return date, '' if len(cands) == 1 else 'Jahr aus Wochentag'
        if wd is None:
            return date, 'Jahr unsicher'
    return None, ''


def page_year(pages, ranges, date):
    """Year of the schedule pages that carry the day and month of `date` (for headers without year)."""
    years = Counter()
    for key, ents in pages.items():
        e = [x for x in ents if x[1] <= 6 and any(f0 <= x[0] <= f1 for f0, f1 in ranges)]
        if len(e) < 3:
            continue
        rows = vote_rows(e)
        head = ' '.join(rows.get(n, '') for n in range(1, 7))
        for m in PAGE_DATE.finditer(head):
            y = m.group(3)
            if y and len(y) == 4 and int(m.group(1)) == date.day and month_of(m.group(2)) == date.month:
                years[int(y)] += 1
    return years.most_common(1)[0][0] if years else None


# ---------------------------------------------------------------- pages and schedules
def vote_rows(entries):
    """Majority vote per character over all copies of each row; parity errors do not vote."""
    rows = defaultdict(list)
    for f, n, b in entries:
        rows[n].append(b)
    out = {}
    for n, bs in rows.items():
        a = np.array(bs, np.uint8)
        ok = PAR[a]
        txt = ''
        for j in range(a.shape[1]):
            v = a[ok[:, j], j] & 0x7f
            c = int(np.bincount(v, minlength=128).argmax()) if len(v) else 0x20
            txt += G0.get(c, chr(c) if 0x20 <= c < 0x7f else ' ')
        out[n] = txt
    return out


TIME_ROW = re.compile(r'^ {0,4}(?:[^\w\s] {0,2})?([0-2Oo][0-9Oo])[.:,ß ]([0-5Oo][0-9Oo])\b(?: ?Uhr)?[!:]? +(?:[0-2]\d[0-5]\d +)?(\S.*)$')
CONT_ROW = re.compile(r'^ {6,}(\S.*)$')
PAGE_DATE = re.compile(r'(\d{1,2})\. ?([A-Za-zÄäÖöÜü~]{3,10}|\d{1,2}(?=[\s.]|$))\.?\s?(\d{4}|\d{2})?(?!\d)')
CODE_DATE = re.compile(r'1D[0-9A-F]{3} +([0-3]\d)([01]\d)(\d\d)\b')
DAY_WORDS = {'MONTAG': 0, 'DIENSTAG': 1, 'MITTWOCH': 2, 'DONNERSTAG': 3, 'FREITAG': 4, 'SAMSTAG': 5,
             'SONNABEND': 5, 'SONNTAG': 6}
MONTH_NAMES = ['januar', 'februar', 'märz', 'april', 'mai', 'juni', 'juli', 'august', 'september', 'oktober',
               'november', 'dezember']
PAGE_CODE = {'101': 'ARD', '102': 'ZDF', '103': 'ARD+ZDF', '110': 'Bayern 3', '111': 'Bayern 3', '120': 'N3',
             '202': 'ORB', '207': 'SAT.1', '217': 'SAT.1', '237': 'SAT.1', '256': 'SAT.1', '257': 'SAT.1',
             '267': 'SAT.1', '220': 'Pro 7', '243': 'RTL'}
PAGE_CHANNEL = [(r'ZDF|ZWEITES', 'ZDF'), (r'ERSTES|ERSTEN|(?<!WEST)DEUTSCHESFERNSEHEN|ARD-PROGRAMM', 'ARD'),
                (r'EINSPLUS|EINS PLUS', 'Eins Plus'), (r'3SAT', '3sat'), (r'BAYER', 'Bayern 3'),
                (r'HESSEN', 'Hessen 3'), (r'S.DWEST|SWF|SDR|SAAR', 'Südwest 3'), (r'WEST|WDR', 'West 3'),
                (r'NORD|NDR|N3', 'N3'), (r'SFB', 'SFB 3')]


RADIO = re.compile(r'^(HR|SFB|NDR|WDR|BR|SWF|SDR|SR|RB|DLF|DW|RIAS|DS) ?\d\b|^(DLF|Deutschlandfunk)\b')


def clean_title(t):
    t = re.sub(r'\(?VPS[ :]?\d\d[.:]\d\d\)?', ' ', t)               # VPS time of the entry
    t = re.sub(r'^(Uhr\b|\d{3}\s+)', '', t.strip()).strip()        # "Uhr", page reference at the start
    t = re.sub(r'[\s.ß>_,^/~:*-]*\d{3}$', '', t.strip())         # page reference at the end
    t = re.sub(r'\.{3,}.*$', '', t)                               # dot leader
    t = re.sub(r'[ .ß>_^~]+$', '', t)
    t = re.sub(r'^[!~ ]+', '', t)
    return re.sub(r'\s{2,}', ' ', t).strip()


def month_of(w):
    w = w.lower().replace('~', '').rstrip('.')
    if w.isdigit():
        return int(w) if 1 <= int(w) <= 12 else None
    for i, n in enumerate(MONTH_NAMES):
        if len(w) >= 3 and (n.startswith(w[:3]) or sum(a == b for a, b in zip(w, n)) >= max(3, len(n) - 2)):
            return i + 1
    return None


def parse_schedule(rows, key, fam, seg_date):
    """Programme entries of a page: [(minutes from 0:00 of the page date, title)], channel, date."""
    entries, last = [], None
    for n in range(1, 24):
        s = rows.get(n)
        if s is None:
            continue
        m = TIME_ROW.match(s)
        if m:
            h = int(m.group(1).replace('O', '0').replace('o', '0'))
            mi = int(m.group(2).replace('O', '0').replace('o', '0'))
            if h > 29:
                last = None
                continue
            title = m.group(3)
            if re.match(r'^.{1,3} {2,}\S', title):     # damaged "Uhr" column of the 1980s pages
                title = title[3:]
            title = clean_title(title)
            if re.match(r'^(oo|\*)?\s*(=\s*stereo|bis\b|Ende\b|Sendeschlu)|^oo$', title, re.I):
                entries.append([h * 60 + mi, '', 2])     # end of the last programme ("12.30 oo = stereo")
                last = None
                continue
            if re.match(r'^-\s*\d', title) or len(re.sub(r'[^A-Za-zÄÖÜäöüß]', '', title)) < 2 or \
                    (len(title) >= 12 and sum(c.isdigit() for c in title) > 0.25 * len(title)):
                last = None                          # time range ("19.00 - 23.00 ...") or no title
                continue
            entries.append([h * 60 + mi, title, 0])
            last = entries[-1]
        elif last is not None and CONT_ROW.match(s) and last[2] < 2 and not re.search(r'\d{3}\s*$', last[1]):
            c = clean_title(CONT_ROW.match(s).group(1))
            if c and not re.match(r'^(VPS|anschl|bis )', c):
                last[1] = last[1] + ' ' + c if not last[1].endswith('-') else last[1][:-1] + c
                last[2] += 1
            else:
                last = None
        else:
            last = None
    if len(entries) < 3:
        return None
    # times after midnight on the same page belong to the next day
    out, add, prev = [], 0, None
    for t, title, _ in entries:
        if prev is not None and t + add < prev - 120:
            add += 1440
        out.append((t + add, title))
        prev = t + add
    if sum(b[0] > a[0] for a, b in zip(out, out[1:])) < 0.6 * (len(out) - 1):
        return None
    head = ' '.join(rows.get(n, '') for n in range(1, 7))
    date = None
    m = CODE_DATE.search(head)                       # "1D120 101088": programme code and date
    if m:
        try:
            date = dt.date(1900 + int(m.group(3)) if int(m.group(3)) >= 50 else 2000 + int(m.group(3)),
                           int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    for m in ([] if date else PAGE_DATE.finditer(head)):
        mo = month_of(m.group(2))
        d = int(m.group(1))
        if mo and 1 <= d <= 31:
            y = m.group(3)
            y = (int(y) if len(y) == 4 else 1900 + int(y)) if y else (seg_date.year if seg_date else None)
            if y and 1970 < y < 2010:
                try:
                    date = dt.date(y, mo, d)
                except ValueError:
                    pass
                break
    flat = re.sub(r'[\s~]', '', head.upper())
    if date is None and seg_date is not None:
        # undated page: only for the day of the recording ("Heute") or a matching weekday
        wd = [v for k, v in DAY_WORDS.items() if k in flat]
        if (wd and seg_date.weekday() not in wd) or re.search(r'MORGEN|[ÜU]BERMORGEN', flat):
            return None
    if re.search(r'H.RFUNK|RADIO', flat) or sum(bool(RADIO.match(t)) for _, t in out) * 2 >= len(out):
        return None                                  # radio programme
    letters = ''.join(re.sub(r'[^A-Za-zÄÖÜäöü]', '', t) for _, t in out)
    if letters and sum(c.isupper() for c in letters) > 0.7 * len(letters):
        return None                                  # radio pages and charts are in capitals
    if sum(bool(re.search(r'\d+,\d+\s*%?\)?\s*$|\d+,\d+\s+\d+\s*$', t)) for _, t in out) * 3 >= len(out) or \
            re.search(r'MARKTANTEIL|QUOTEN|ZUSCHAUER', flat):
        return None                                  # audience ratings (title, million, share)
    if sum(bool(re.search(r'\b(DM|ÜF|HP|AI|DZ|EZ|Abflug)\b|\bTuS\b|\bFC\b|\bSV\b|\d+:\d+\s*$', t))
           for _, t in out) * 3 >= len(out):
        return None                                  # travel offers, sports results
    chans = set()
    for _, t in out:
        chans.update(re.findall(r'\b(ARD|ZDF|RTL|SAT\.?1|PRO ?7?|DSF|MTV|KABK?|VOX|3SAT|N3|B1|ORB|MDR|EURO|TELE ?5|'
                                r'RTL ?2|n-tv|WDR|NDR|BR|HR|SWF)\s*\d*\s*$', t))
    if len(chans) >= 3:
        return None                                  # "now on TV" across several channels
    trains = sum(bool(re.match(r'^ ?(IC|ICE|IR|RE|RB|EC|EN|D|E|SE|RE-BUS)\w* {1,4}\d{2,5}\b', rows.get(n, '')))
                 for n in range(1, 24))
    if re.search(r'DEUTSCHEBAHN|BAHNHOF|ABFAHRT|ANKUNFT|GLEIS|FLUGHAFEN|ABFLUG', flat) or trains >= 2 or \
            sum(bool(re.search(r'\b(Hbf|SBB|Bf|Hauptbahnhof)\b|\d+ ?Min\b', t)) for _, t in out) * 3 >= len(out):
        return None                                  # train and flight departures (station, platform)
    out = [e for e in out if not RADIO.match(e[1])]
    chan = None
    # programme code on the schedule pages from about 1988 ("1D101 240792"); the codes were
    # matched to the channels by the VPS CNI of the recordings
    code = re.search(r'1D([0-9A-F]{3})\b', head)
    if code:
        chan = PAGE_CODE.get(code.group(1))
        if chan is None:
            code = None
    shared = fam in ('ARD/ZDF', 'ORB/MDR', 'TSVT', 'PK Berlin', None) or FAMILY_CHANNEL.get(fam) is None
    for rx, c in PAGE_CHANNEL:
        if chan is None and shared and re.search(rx, flat):
            chan = c
            break
    if chan is None and fam == 'ORB/MDR':
        o = sum(bool(re.search(r'Brandenburg|ORB', t)) for _, t in out)
        d = sum(bool(re.search(r'MDR|Sachsen|Thüringen|Anhalt|Riverboat', t)) for _, t in out)
        chan = 'ORB' if o > d else 'MDR' if d > o else None
    page = key[1]
    pnum = key[0] * 100 + (page >> 4) * 10 + (page & 15)
    if fam == 'ARD/ZDF':
        # ARD/ZDF teletext: 120 ff ZDF today, 160 ff ARD, 170 ff ZDF, 180-189 third programmes in the
        # early years; later day pages 300 ff for both (with the programme code 1D101/1D102) and the third
        # programmes around 410 ff. Other pages are overviews, news or test pages; the radio pages
        # (300 ff in the 1980s) are dropped above.
        if not (120 <= pnum <= 129 or 160 <= pnum <= 189 or 300 <= pnum <= 499):
            return None
        dritte = chan if chan not in (None, 'ARD', 'ZDF', 'Eins Plus', '3sat') else None
        if code is None and dritte is None:
            if 120 <= pnum <= 129 or 170 <= pnum <= 179:
                chan = 'ZDF'
            elif 160 <= pnum <= 169:
                chan = 'ARD'
            elif 180 <= pnum <= 189:
                chan = None
            elif chan is None:
                z = sum(bool(re.search(r'\bheute\b|ZDF', t)) for _, t in out)
                a = sum(bool(re.search(r'Tagesschau|Tagesthemen|Sportschau|im Ersten', t)) for _, t in out)
                chan = 'ZDF' if z > a else 'ARD' if a > z else None
    elif chan is None:
        chan = FAMILY_CHANNEL.get(fam)
    return dict(entries=out, chan=chan, date=date, page=pnum)


def schedules(pages, ranges, seg_date, fam, fam_chan=None):
    """All schedule pages within the frame ranges: {channel: {start datetime: Counter(title)}}.
    fam_chan: channel of each teletext service found in the capture, for services this tool
    does not know by name."""
    sched = defaultdict(lambda: defaultdict(Counter))
    for key, ents in pages.items():
        e = [x for x in ents if any(f0 <= x[0] <= f1 for f0, f1 in ranges)]
        if len(e) < 3:
            continue
        rows = vote_rows(e)
        pfam = parse_header(rows.get(0, ''))['fam'] or fam
        sc = parse_schedule(rows, key, pfam, seg_date)
        if sc and not sc['chan'] and pfam not in FAMILY_CHANNEL and fam_chan:
            sc['chan'] = fam_chan.get(pfam)
        if not sc or not sc['chan']:
            continue
        date = sc['date'] or seg_date
        if date is None:
            continue
        for ch in (('ARD', 'ZDF') if sc['chan'] == 'ARD+ZDF' else (sc['chan'],)):
            for t, title in sc['entries']:
                start = dt.datetime.combine(date, dt.time()) + dt.timedelta(minutes=t)
                sched[ch][start][title] += len(e)
    return sched


def load_schedule_files(paths):
    sched = defaultdict(lambda: defaultdict(Counter))
    for p in paths:
        for r in csv.DictReader(open(p, encoding='utf-8'), delimiter=';'):
            try:
                d = dt.datetime.strptime(r['Datum'].strip() + ' ' + r['Uhrzeit'].strip()[:5], '%d.%m.%Y %H:%M')
            except (ValueError, KeyError):
                continue
            sched[r['Sender'].strip()][d][r['Sendung'].strip()] += 1
    return sched


def merge_titles(starts):
    """Entries from several pages: [(start, best title)]."""
    return [(t, best_title(starts[t])) for t in sorted(starts)]


def best_title(c):
    """Character vote over the readings of the most common length (each reading weighted by the
    number of copies it was voted from); with a single reading, that reading."""
    lens = Counter()
    for t, n in c.items():
        lens[len(t)] += n
    L = lens.most_common(1)[0][0]
    same = [(t, n) for t, n in c.items() if len(t) == L]
    if len(same) > 1:
        out = ''
        for j in range(L):
            v = Counter()
            for t, n in same:
                if t[j] != BAD:
                    v[t[j]] += n
            out += v.most_common(1)[0][0] if v else ' '
        return re.sub(r'\s{2,}', ' ', out).strip()
    # the most frequent reading; ties go to the reading with fewer damaged characters
    return max(c.items(), key=lambda kv: (kv[1], -sum(ch in '~ß^' for ch in kv[0])))[0]


def programmes(entries, a, b):
    """Entries (start, title) that overlap the recorded span [a, b] noticeably."""
    out = []
    for k, (t, title) in enumerate(entries):
        end = entries[k + 1][0] if k + 1 < len(entries) else t + dt.timedelta(hours=2)
        if (end - t).total_seconds() > 6 * 3600:
            end = t + dt.timedelta(hours=2)
        ov = (min(b, end) - max(a, t)).total_seconds()
        if title and ov >= min(180, 0.5 * (b - a).total_seconds()) and ov > 0:
            out.append((t, title, end))
    return out


# ---------------------------------------------------------------- channel of a segment
def channel(seg, dl, ddr, fn, year):
    """Channel of a segment and how it was found."""
    f0, f1 = seg['f0'], seg['f1']
    cnis, w3s, w4s = Counter(), Counter(), Counter()
    for a, b, w3, txt, cni in dl:
        n = min(b, f1) - max(a, f0)
        if n <= 0:
            continue
        if cni not in (0, 0xFFF):
            cnis[cni] += n
        w3s[w3 & 0xF] += n
        if txt:
            w4s[txt.strip()] += n
    fam = seg['fam']
    fc = FILE_CHANNEL.get(fn['chan'])
    dd = Counter(n for f, n in ddr if f0 <= f <= f1)
    if dd and sum(dd.values()) > 100:
        n = dd.most_common(1)[0][0]
        return ('DFF %d' if year and year >= 1990 else 'DDR %d') % n, 'Zeile 16 (DDR-Programmnummer)'
    if cnis:
        c = cnis.most_common(1)[0][0]
        if c in CNI:
            return CNI[c], 'VPS CNI %03X' % c
    if w4s:
        t = w4s.most_common(1)[0][0]
        if t in W4_CHANNEL:
            return W4_CHANNEL[t], 'Datenzeile Wort 4 %s' % t
    if FAMILY_CHANNEL.get(fam):
        return FAMILY_CHANNEL[fam], 'Videotext-Kopfzeile %s' % fam
    if w3s:
        w3 = w3s.most_common(1)[0][0]
        st = W3_STATION.get(w3)
        how = 'Datenzeile Wort 3 %X (%s)%s' % (w3, st or 'privat/Kabel', ', Wort 4 ' + w4s.most_common(1)[0][0]
                                               if w4s else '')
        if st == 'ZDF':
            return 'ZDF', how
        if st:
            if DRITTE_OF_FILE.get(fn['chan']) == st or (fn['chan'] in DRITTE_OF_FILE and
                                                         DRITTE.get(DRITTE_OF_FILE[fn['chan']]) == DRITTE[st]):
                return DRITTE[st], how + ', Dateiname'
            return 'ARD', how
        if fc:
            return fc, how + ', Dateiname'
    if fam == 'TSVT' and fc in ('RTL', 'SAT.1'):
        return fc, 'Videotext TSVT, Dateiname'
    if fc:
        return fc, 'Dateiname'
    if fam:
        return {'TSVT': 'RTL plus oder SAT.1', 'PK Berlin': 'Kabel Berlin', 'DFF': 'DFF', 'DDR-Test': 'DDR'}.get(
            fam, fam), 'Videotext-Kopfzeile ' + fam
    return '', ''


# ---------------------------------------------------------------- one capture
# ---------------------------------------------------------------- Pro 7 VideoDat clock
VIDEODAT = os.environ.get('VIDEODAT_DECODER') or next(
    (p for p in (os.path.join(TOOLS, '..', '..', 'channel-videodat-decoder'),
                 os.path.expanduser('~/git/repos_itewreed/channel-videodat-decoder'))
     if os.path.exists(os.path.join(p, 'videodat_decoder.py'))), None)
VD_WINDOW, VD_STEP = 15, 60                          # seconds decoded / distance of the samples


def videodat_clock(capture, tmpdir):
    """Clock samples from Pro 7 VideoDat (channel-videodat-decoder) for recordings without teletext:
    1993 variant: clock packets (stream 01 type 06, date and time every 5 s); 1990 variant: the
    plain-text clock between the packets (time only). A window of VD_WINDOW s is decoded every
    VD_STEP s. Returns header-like samples [(frame, parsed, -2)]."""
    if not VIDEODAT or os.path.isdir(capture):
        return []
    sys.path.insert(0, os.path.abspath(VIDEODAT))
    import videodat_decoder as vd, videodat1990 as vd90
    from vbisource import iter_frames
    quiet = lambda *a, **k: None
    out, variant = [], 'auto'
    step = [VD_STEP]                                 # the 1990 variant sends its clock rarely: read all
    raw = os.path.join(tmpdir, 'videodat_window_%d.vbi' % os.getpid())

    def windows():
        # one pass over the capture (a FLAC can only be read from the start)
        buf, f0, pos = [], None, 0
        for a in iter_frames(capture, 250):
            for k in range(len(a)):
                if step[0] == VD_WINDOW or (pos + k) % (step[0] * FPS) < VD_WINDOW * FPS:
                    if f0 is None:
                        f0 = pos + k
                    buf.append(a[k])
                elif buf:
                    yield f0, buf
                    buf, f0 = [], None
            pos += len(a)
        if buf:
            yield f0, buf

    for f0, buf in windows():
        with open(raw, 'wb') as fh:
            fh.write(np.ascontiguousarray(np.array(buf)).tobytes())
        try:
            cap = vd.read_vbi(raw, 2048, 32, None, quiet, VD_WINDOW, True, variant)
        except SystemExit:
            continue                                 # no data lines in this window
        variant = cap.fmt.variant
        if variant == '1990':
            step[0] = VD_WINDOW
        nb = cap.fmt.nbytes
        if variant == '1993':
            st = cap.stream()
            items = cap.items
            for x in vd.find_packets(st):
                p = x.pay
                if x.ok and p and x.A == 0x01 and p[0] == 0x06 and len(p) == 8:
                    try:
                        t = dt.datetime(1900 + p[1], p[2], p[3], p[4], p[5], p[6])
                    except ValueError:
                        continue
                    f = f0 + items[min(x.pos // nb, len(items) - 1)][0]
                    out.append((f, dict(sec=t.hour * 3600 + t.minute * 60 + t.second, day=t.day,
                                        mon=t.month, yy=t.year % 100, wd=t.weekday(), fam='VideoDat'), -2))
        else:
            for label, st in vd90.split_channels(cap, quiet):
                pk = vd90.find_packets(st)
                for pos, txt in vd90.zwischentexte(st, pk):
                    m = re.match(r'Uhrzeit (\d\d):(\d\d):(\d\d)$', txt)
                    if m and int(m.group(1)) < 24:
                        # position within the window from the share of bytes before it
                        f = f0 + int(pos / max(len(st.data), 1) * len(buf))
                        out.append((f, dict(sec=int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)),
                                            day=None, mon=None, yy=None, wd=None, fam='VideoDat'), -2))
    try:
        os.remove(raw)
    except OSError:
        pass
    return out


def catalog_one(capture, survey):
    fn = from_filename(capture)
    S = json.load(open(os.path.join(survey, 'summary.json')))
    nframes = S['frames']
    samples, pages = read_teletext(survey)
    dl, labels = read_dataline(survey)
    ddr = read_ddr(survey)
    segs = segments(samples, nframes)
    vdclock = False
    if all(s['off'] is None for s in segs) and (fn['chan'] == 'pro7' or any(c == 0xD94 for *_, c in dl)):
        # Pro 7 without teletext clock: clock of the VideoDat data service
        vds = videodat_clock(capture, os.path.dirname(os.path.abspath(survey)))
        if vds:
            samples, vdclock = samples + vds, True
            segs = segments(vds, nframes)
    for seg in segs:
        seg['date'], seg['dnote'] = segment_date(seg, samples, fn)
        if vdclock:
            seg['dnote'] = ('; '.join(x for x in (seg['dnote'], 'Uhrzeit aus VideoDat') if x))
        if seg['date'] is None and fn['date'] and 'x' not in ''.join(fn['date']):
            seg['date'], seg['dnote'] = dt.date(*map(int, fn['date'])), 'Datum aus Dateiname'
    for seg in segs:
        if seg['dnote'] == 'Jahr unsicher':
            y = page_year(pages, [(seg['f0'], seg['f1'])], seg['date'])
            if y:
                seg['date'], seg['dnote'] = seg['date'].replace(year=y), 'Jahr aus Programmseite'
    # the schedule pages of all segments of the same day are voted together
    groups = defaultdict(list)
    for seg in segs:
        groups[seg['date']].append((seg['f0'], seg['f1']))
    for seg in segs:
        d = seg['date']
        seg['chan'], seg['how'] = channel(seg, dl, ddr, fn, d.year if d else None)
        if seg['chan'] == 'SFB 3' and d and d.year >= 1992:
            seg['chan'] = 'B1'                       # SFB 3 became B1 in 1992
    fam_chan = {s['fam']: s['chan'] for s in segs if s['fam'] and FAMILY_CHANNEL.get(s['fam']) is None and
                s['fam'] not in ('ARD/ZDF', 'ORB/MDR', 'TSVT', 'PK Berlin', 'INFOTEXT')}
    cache = {}
    rows = []
    for seg in segs:
        date, dnote, chan, how = seg['date'], seg['dnote'], seg['chan'], seg['how']
        base = dict(coll=fn['coll'], tape=fn['tape'], chan=chan, file=os.path.basename(capture),
                    position='%s-%s' % (pos(seg['f0']), pos(seg['f1'])))
        if seg['off'] is not None and date is not None:
            day0 = dt.datetime.combine(date, dt.time())
            a = day0 + dt.timedelta(seconds=seg['off'] + seg['f0'] / FPS)
            b = day0 + dt.timedelta(seconds=seg['off'] + seg['f1'] / FPS)
        else:
            a = b = None
        found = []
        if a is not None:
            g = date
            if g not in cache:
                cache[g] = schedules(pages, groups[g], date, seg['fam'], fam_chan)
            sched = cache[g]
            if chan == 'ARD/ZDF':
                # shared ARD/ZDF teletext and no data line: both programmes are candidates
                alt = {c: programmes(merge_titles(sched.get(c, {})), a, b) for c in ('ARD', 'ZDF')}
                if alt['ARD'] and alt['ZDF']:
                    chan = base['chan'] = 'ARD oder ZDF'
                    for c in ('ARD', 'ZDF'):
                        for t, ti, e in alt[c]:
                            rows.append(dict(base, title='%s (%s)' % (ti, c), date=t.strftime('%d.%m.%Y'),
                                             time=t.strftime('%H:%M'), rec_a=max(a, t).strftime('%H:%M:%S'),
                                             rec_b=min(b, e).strftime('%H:%M:%S'),
                                             how='Videotext S. Programm; Sender unbestimmt (ARD/ZDF-Videotext, '
                                                 'keine Datenzeile)' + ('; ' + dnote if dnote else '')))
                    continue
            entries = merge_titles(sched.get(chan, {}))
            found = [(t, ti, e, 'Videotext S. Programm') for t, ti, e in programmes(entries, a, b)]
            if not found and chan in EXTERNAL:
                found = [(t, ti, e, 'Programmliste') for t, ti, e in
                         programmes(merge_titles(EXTERNAL[chan]), a, b)]
        if not found:
            # programme unknown: one row for the recording, or one per VPS label
            lab = [r for r in labels if seg['f0'] <= r[0] <= seg['f1'] and r[1] - r[0] >= 20 * FPS and
                   date is not None and plausible_label(r[2], date)]
            if lab and date is not None:
                for f0, f1, (d, m, h, mi) in lab:
                    rows.append(dict(base, title=fn['title'] or 'unbekannt',
                                     date=date.replace(month=m, day=d).strftime('%d.%m.%Y'),
                                     time='%02d:%02d' % (h, mi), rec_a=hms(seg['off'] + f0 / FPS) if a else '',
                                     rec_b=hms(seg['off'] + f1 / FPS) if a else '',
                                     how=('Dateiname, ' if fn['title'] else '') + 'VPS-Label; Sender: ' + how))
                continue
            rows.append(dict(base, title=fn['title'] or 'unbekannt',
                             date=date.strftime('%d.%m.%Y') if date else '',
                             time=a.strftime('%H:%M:%S') if a else '',
                             rec_a=a.strftime('%H:%M:%S') if a else '', rec_b=b.strftime('%H:%M:%S') if b else '',
                             how=('Titel aus Dateiname; ' if fn['title'] else '') + 'Sender: ' + how +
                             ('; ' + dnote if dnote else '')))
            continue
        for t, title, e, src in found:
            rows.append(dict(base, title=title, date=t.strftime('%d.%m.%Y'), time=t.strftime('%H:%M'),
                             rec_a=max(a, t).strftime('%H:%M:%S'), rec_b=min(b, e).strftime('%H:%M:%S'),
                             how=src + '; Sender: ' + how + ('; ' + dnote if dnote else '')))
    for r in rows:
        if 'Jahr unsicher' in r['how'] and r['date'] and fn['date']:
            r['date'] = r['date'][:6] + fn['date'][0]          # e.g. 30.11.198x
        try:
            d = dt.datetime.strptime(r['date'], '%d.%m.%Y').date()
        except ValueError:
            d = None
        if d and r['chan'] == 'RTL' and d.year < 1993:
            r['chan'] = 'RTL plus'                   # the name until 1992
        if d and r['chan'] == 'Kabel 1' and d < dt.date(1995, 2, 28):
            r['chan'] = 'Kabelkanal'                 # renamed Kabel 1 on 28 February 1995
    return merge_rows(rows)


def merge_rows(rows):
    """Join rows of neighbouring segments that name the same programme."""
    out = []
    for r in rows:
        p = out[-1] if out else None
        if p and (p['title'], p['chan'], p['date'], p['time']) == (r['title'], r['chan'], r['date'], r['time']) \
                and p['title'] != 'unbekannt':
            p['rec_b'] = r['rec_b']
            p['position'] = p['position'].split('-')[0] + '-' + r['position'].split('-')[1]
        else:
            out.append(r)
    return out


EXTERNAL = {}


def plausible_label(lab, date):
    """VPS label within a day of the recording (the 1983-85 test cycle sends fake dates, FINDINGS 4)."""
    d, m, h, mi = lab
    for y in (date.year - 1, date.year, date.year + 1):
        try:
            if abs((dt.date(y, m, d) - date).days) <= 1:
                return True
        except ValueError:
            pass
    return False


# ---------------------------------------------------------------- inputs
def captures(inputs):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, 'packets.bin')):
                out.append(p)
                continue
            fs = sorted(f for f in glob.glob(os.path.join(p, '*')) if f.lower().endswith(CAPTURE_EXT))
            out += [f for f in fs if not chroma_with_luma(f) and (not f.lower().endswith('.flac') or is_vbi_flac(f))]
        elif not chroma_with_luma(p):
            out.append(p)
    return out


def is_vbi_flac(f):
    """bttv VBI in FLAC is mono 8 bit; the HiFi / linear audio FLACs next to TBC files are not."""
    try:
        with open(f, 'rb') as fh:
            h = fh.read(42)
    except OSError:
        return False
    if h[:4] != b'fLaC' or len(h) < 42:
        return True                                  # let the reader report it
    x = int.from_bytes(h[18:21], 'big')
    return ((x >> 1) & 7) + 1 == 1 and (((x & 1) << 4) | (h[21] >> 4)) + 1 == 8


def chroma_with_luma(f):
    if f.endswith('.tbcc'):
        b = f[:-5]
        return os.path.exists(b + '.tbcy') or os.path.exists(b + '.tbc')
    if f.endswith('_chroma.tbc'):
        return os.path.exists(f[:-len('_chroma.tbc')] + '.tbc')
    return False


def find_survey(capture, roots):
    if os.path.isdir(capture):
        return capture
    n = name_of(capture)
    for r in roots:
        for c in (n, n[:-4] if n.endswith('_vbi') else None):
            if c and os.path.exists(os.path.join(r, c, 'summary.json')) and \
                    os.path.exists(os.path.join(r, c, 'packets.bin')):
                return os.path.join(r, c)
    return None


def run_survey(args):
    capture, out = args
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'log.txt'), 'w') as log:
        r = subprocess.run([sys.executable, os.path.join(TOOLS, 'survey.py'), capture, out], stdout=log,
                           stderr=subprocess.STDOUT)
    return capture, r.returncode


def work_one(args):
    capture, survey = args
    try:
        return capture, catalog_one(capture, survey), None
    except Exception as e:                       # one broken capture must not stop the catalogue
        return capture, [], '%s: %s' % (type(e).__name__, e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--csvout', default='catalog.csv', help='CSV file to write (default catalog.csv)')
    ap.add_argument('--survey-root', action='append', default=[],
                    help='folder with existing survey output (<root>/<name>/), may be given several times')
    ap.add_argument('--work', default=None, help='folder for new survey output (default: next to the CSV, '
                                                 'catalog_survey/)')
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--schedule', action='append', default=[],
                    help='CSV with programme listings (Sender;Datum;Uhrzeit;Sendung), used where the '
                         'teletext has none')
    ap.add_argument('--details', action='store_true', help='add position in the capture and source')
    ap.add_argument('--gaps', help='also write the recordings whose programme is unknown (channel, date and '
                                   'time to look up in historical listings) to this CSV')
    a = ap.parse_args()
    work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.csvout)), 'catalog_survey')
    roots = a.survey_root + [work]
    sch = a.schedule or [p for p in [os.path.join(TOOLS, '..', 'docs', 'schedules.csv')] if os.path.exists(p)]
    EXTERNAL.update(load_schedule_files(sch))
    caps = captures(a.inputs)
    todo, jobs = [], []
    for c in caps:
        s = find_survey(c, roots)
        if s:
            jobs.append((c, s))
        else:
            todo.append((c, os.path.join(work, name_of(c))))
    if todo:
        print('surveying %d captures into %s ...' % (len(todo), work), file=sys.stderr)
        with ProcessPoolExecutor(a.jobs) as ex:
            for c, rc in ex.map(run_survey, todo):
                print('  %s %s' % ('done' if rc == 0 else 'FAILED (see log.txt)', os.path.basename(c)),
                      file=sys.stderr)
                if rc == 0:
                    jobs.append((c, os.path.join(work, name_of(c))))
    results = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for c, rows, err in ex.map(work_one, jobs):
            if err:
                print('error in %s: %s' % (c, err), file=sys.stderr)
            results[c] = rows
    head = ['Collection', 'Tape', 'Sendung', 'Sender', 'Datum', 'Uhrzeit', 'Aufnahme_von', 'Aufnahme_bis', 'Datei']
    if a.details:
        head += ['Position', 'Quelle']
    n = 0
    with open(a.csvout, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(head)
        for c in caps:
            for r in results.get(c, []):
                row = [r['coll'], r['tape'], r['title'], r['chan'], r['date'], r['time'], r['rec_a'], r['rec_b'],
                       r['file']]
                if a.details:
                    row += [r['position'], r['how']]
                w.writerow(row)
                n += 1
    print('%s: %d rows from %d captures' % (a.csvout, n, len(results)), file=sys.stderr)
    if a.gaps:
        k = 0
        with open(a.gaps, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(['Collection', 'Tape', 'Sender', 'Datum', 'Aufnahme_von', 'Aufnahme_bis', 'VPS', 'Datei'])
            for c in caps:
                for r in results.get(c, []):
                    if r['how'].startswith('Videotext S.') or r['how'].startswith('Programmliste') or \
                            not r['date'] or not r['rec_a']:
                        continue
                    w.writerow([r['coll'], r['tape'], r['chan'], r['date'], r['rec_a'], r['rec_b'],
                                r['time'] if 'VPS-Label' in r['how'] else '', r['file']])
                    k += 1
        print('%s: %d recordings without programme' % (a.gaps, k), file=sys.stderr)


if __name__ == '__main__':
    main()
