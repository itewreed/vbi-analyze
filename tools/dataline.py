#!/usr/bin/env python3
"""Readable summary of the data line: line 16 (source data line) and line 329 (section data line).

Reads vps.txt of a survey folder (frame, row, words 3-15 as hex) and writes dataline.txt:
one row per stretch of frames in which the line keeps the same content, with
  word 3   4-bit source code (low nibble) and the source it stands for (docs/FINDINGS.md 3.2)
  word 4   plain-text source / section identification, 8 characters, one per frame;
           the character with bit 8 set marks the start of the text
  VPS CNI  from words 11-14 (line 16 only)
  meaning  for known texts (docs/FINDINGS.md 3.3)

usage: dataline.py <survey-dir> [...]     (survey.py also calls it at the end)
"""
import os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vps

FPS = 25
WIN = 48          # frames per analysis window (a multiple of the 8-character cycle)
GAP = 250         # frames without data line that do not end a stretch
LINE = {9: 16, 25: 329}

# Word 3, low nibble; inferred from the tapes (docs/FINDINGS.md 3.2)
SOURCE = {
    0x0: 'private / cable feed (Sat.1, RIAS-TV, Berlin cable head end, MKB, RTL)',
    0x1: 'SWF (Suedwestfunk)',
    0x3: 'Deutsche Bundespost',
    0x4: 'BR (Bayerischer Rundfunk)',
    0x5: 'ZDF',
    0x6: 'SFB (Sender Freies Berlin)',
    0x8: 'HR (Hessischer Rundfunk)',
    0x9: 'WDR (Westdeutscher Rundfunk)',
    0xA: 'SR (Saarlaendischer Rundfunk); 1983-84 also ZDF test phase',
    0xB: 'Deutsche Bundespost',
    0xC: 'NDR (Norddeutscher Rundfunk)',
    0xE: 'SDR (Sueddeutscher Rundfunk)',
}

# Word 4 texts seen on the tapes (docs/FINDINGS.md 3.3 and 3.4)
MEANING = {
    'WDR-KLN': 'WDR Cologne', 'WDR*KLN*': 'WDR Cologne', 'NDR HMBG': 'NDR Hamburg',
    'SFB-SCHO': 'SFB Schoeneberg (Berlin)', 'SFB1.REG': 'SFB 1 regional', 'SWF*BDN*': 'SWF Baden-Baden',
    'BR*MCHN*': 'BR Munich', 'BR-MCHN': 'BR Munich', 'SAARL*RF': 'Saarlaendischer Rundfunk',
    'HR*FFTM1': 'HR Frankfurt', 'BRM685/1': 'unknown (BR?)', 'WDR*ZUBR': 'WDR feed line (Zubringer)',
    'SDR*STGT': 'SDR Stuttgart', 'WDR-SAT': 'WDR satellite feed', 'ZDF': 'ZDF', '*ZDF*': 'ZDF',
    'SAT1*LU1': 'Sat.1 Ludwigshafen', 'RIAS-TV': 'RIAS-TV Berlin', 'PK-BLN-2': 'Berlin cable head end',
    'DBP-1': 'Deutsche Bundespost', 'DBP 2': 'Deutsche Bundespost',
    'DBP-FTZ': 'Deutsche Bundespost, Fernmeldetechnisches Zentralamt',
}


def rev(b):
    return int(f'{b:08b}'[::-1], 2)


def text_of(chars):
    """Assemble word 4 from (frame, byte) pairs: one character per frame in an
    8-frame cycle, the character with bit 8 set is the first one."""
    pos = [Counter() for _ in range(8)]
    for f, c in chars:
        pos[f % 8][c] += 1
    best = [p.most_common(1)[0][0] if p else None for p in pos]
    if None in best:
        return None, 0.0
    q = sum(p.most_common(1)[0][1] for p in pos) / len(chars)
    marks = [i for i, c in enumerate(best) if c & 0x80]
    start = marks[0] if len(marks) == 1 else 0
    order = best[start:] + best[:start]
    t = ''.join(chr(c & 0x7f) if 0x20 <= (c & 0x7f) < 0x7f else '.' for c in order)
    return t, q


def windows(recs):
    """Per window of WIN frames: [first frame, last frame, word 3, text or None, cni]."""
    out, i = [], 0
    while i < len(recs):
        w0 = recs[i][0] // WIN
        j = i
        while j < len(recs) and recs[j][0] // WIN == w0:
            j += 1
        grp = recs[i:j]; i = j
        if len(grp) < WIN // 3:
            continue
        w3 = Counter(rev(b[0]) for _, b in grp).most_common(1)[0][0]
        txt, q = text_of([(f, rev(b[1])) for f, b in grp])
        cni = Counter(vps.parse(b)['cni'] for _, b in grp).most_common(1)[0][0]
        out.append([grp[0][0], grp[-1][0], w3, txt if q >= 0.8 else None, cni])
    return out


def segments(recs):
    """Join windows with the same word 3, VPS CNI and text. Windows whose text is
    unreadable, gaps of up to GAP frames and single deviating windows do not split a
    stretch; its text is the most frequent window text. (The text is read per window
    because dropped frames shift the 8-frame cycle.)"""
    ws = windows(recs)
    for k in range(1, len(ws) - 1):        # a lone deviating window between two equal ones
        a, b, c = ws[k - 1], ws[k], ws[k + 1]
        if a[2:] == c[2:] and b[2:] != a[2:] and b[2] == a[2] and b[4] == a[4]:
            b[3] = None
    seg = []
    for w in ws:
        last = seg[-1] if seg else None
        if last and w[2] == last[2] and w[4] == last[4] and w[0] - last[1] <= GAP and \
                (w[3] is None or last[3] is None or w[3] == last[3]):
            last[1] = w[1]
            last[5][w[3]] += w[3] is not None
            last[3] = last[3] or w[3]
        else:
            seg.append(list(w) + [Counter({w[3]: 1} if w[3] else {})])
    for sg in seg:
        sg[3] = sg[5].most_common(1)[0][0] if sg[5] else None
    # short stretches (up to 4 windows) whose text differs from the previous one in at
    # most two characters are read errors: fold them into the previous stretch
    out = []
    for sg in seg:
        last = out[-1] if out else None
        if last and sg[2] == last[2] and sg[4] == last[4] and sg[0] - last[1] <= GAP and (
                sg[1] - sg[0] < 4 * WIN and close(sg[3], last[3]) or sg[3] == last[3]):
            last[1] = sg[1]
        else:
            out.append(sg)
    return [sg[:5] for sg in out]


def close(a, b):
    if a is None or b is None:
        return True
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 2


def tc(f):
    s = f / FPS
    return '%d:%02d:%02d' % (s // 3600, s // 60 % 60, s % 60)


def write(survey_dir):
    src = os.path.join(survey_dir, 'vps.txt')
    if not os.path.exists(src):
        return None
    recs = {9: [], 25: []}
    for l in open(src):
        p = l.split()
        if len(p) >= 3 and int(p[1]) in recs:
            recs[int(p[1])].append((int(p[0]), bytes.fromhex(p[2])))
    lines = ['# Data line words per stretch of frames (line 16: source data line, line 329: section data line)',
             '# frames            time               line  word3  source                                                                 '
             'word4 text   VPS CNI  meaning']
    for row in (9, 25):
        for f0, f1, w3, txt, cni in segments(recs[row]):
            src_name = SOURCE.get(w3 & 0xF, 'unknown')
            t = '|%-8s|' % txt if txt is not None else '(unreadable)'
            c = '%03X' % cni if row == 9 and cni not in (0, 0xFFF) else '-'
            key = (txt or '').strip()
            lines.append('%7d-%-7d  %s-%s  %4d   %02x/%X  %-70s %-12s %-8s %s' % (
                f0, f1, tc(f0), tc(f1), LINE[row], w3, w3 & 0xF, src_name, t, c, MEANING.get(key, '')))
    out = os.path.join(survey_dir, 'dataline.txt')
    open(out, 'w').write('\n'.join(lines) + '\n')
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        print(write(d) or '%s: no vps.txt' % d)
