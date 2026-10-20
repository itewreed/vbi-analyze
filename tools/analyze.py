"""Analyse survey output: writes report.txt and pages/ (teletext pages, majority-voted)."""
import sys, os, json, numpy as np, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vps
from wst2 import HAM, HERR
d = sys.argv[1]
S = json.load(open(os.path.join(d, 'summary.json')))
R = []
def out(*a): R.append(' '.join(str(x) for x in a))
G0 = {0x23: '#', 0x24: '$', 0x40: '§', 0x5b: 'Ä', 0x5c: 'Ö', 0x5d: 'Ü', 0x5e: '^', 0x5f: '_', 0x60: '°',
      0x7b: 'ä', 0x7c: 'ö', 0x7d: 'ü', 0x7e: 'ß', 0x7f: '■'}
def txt(bs):
    s = ''
    for c in bs:
        c &= 0x7f
        s += G0.get(c, chr(c) if 0x20 <= c < 0x7f else ' ')
    return s
def par_ok(bs): return np.mean([bin(int(c)).count('1') & 1 for c in bs])
fr_total = S['frames']
out(f"# {os.path.basename(d)}")
out(f"Frames: {fr_total} ({fr_total/25:.0f} s)")
# ---- rows
out("\n## Row usage (row: active / classes)")
st = defaultdict(dict)
for k, v in S['stat'].items():
    r, c = k.split(':'); st[int(r)][c] = v
for r in range(32):
    a = S['active'][str(r)]
    if a < 0.02 * fr_total: continue
    tv = 7 + r if r < 16 else 320 + r - 16
    out(f"row {r:2d} (TV {tv:3d}) active {a/fr_total*100:5.1f}%  " + ', '.join(f'{c}={v}' for c, v in sorted(st[r].items())))
# ---- VPS
vp = [l.split() for l in open(os.path.join(d, 'vps.txt'))]
if vp:
    out("\n## VPS")
    C = Counter(); B4 = Counter(); seq = []
    for fr, r, h, conf in vp:
        b = bytes.fromhex(h); p = vps.parse(b)
        C[(r, p['cni'], p['day'], p['mon'], p['hour'], p['min'], p['pcs'], p['pty'])] += 1
        B4[b[1]] += 1; seq.append((int(fr), b[1]))
    for k, v in C.most_common(15):
        if v < 3: break
        out(f"row {k[0]} CNI 0x{k[1]:03X} PIL {k[2]:02d}.{k[3]:02d}. {k[4]:02d}:{k[5]:02d} PCS {k[6]} PTY 0x{k[7]:02X}  x{v}")
    out("VPS-Byte 4 Werte:", ' '.join(f'{k:02x}:{v}' for k, v in B4.most_common(20)))
# ---- line 16 biphase
l16 = [l.split() for l in open(os.path.join(d, 'l16.txt'))]
if l16:
    out("\n## Line 16/329 biphase signal (1 Mbit/s)")
    C = Counter((r, s) for fr, r, s in l16)
    for (r, s), v in C.most_common(12):
        if v < 3: break
        out(f"row {r} x{v}: {s}")
# ---- WST
raw = np.fromfile(os.path.join(d, 'packets.bin'), np.uint8)
raw = raw[: len(raw) // 51 * 51].reshape(-1, 51)
frames = raw[:, :4].copy().view('<u4').ravel(); rows = raw[:, 4]; meth = raw[:, 5]; P = raw[:, 6:]
out(f"\n## Teletext (WST): {len(P)} packets")
mag = np.array([HAM[p[3]] & 7 for p in P]); mag[mag == 0] = 8
pkt = np.array([(HAM[p[3]] >> 3) | (HAM[p[4]] << 1) for p in P])
for r in sorted(set(rows)):
    m = rows == r
    if m.sum() < 20: continue
    out(f"row {r:2d}: {m.sum():6d} packets, magazines {dict(Counter(mag[m]).most_common())} packet no. {dict(sorted(Counter(pkt[m]).items()))}")
# valid rows: those where packets look like teletext (more than 1 distinct packet number)
def _is_tt(r):
    m = rows == r; n = m.sum()
    if n < max(20, 0.05 * fr_total) or n < 0.15 * S['active'][str(r)]: return False
    pk_share = [v / n for v in Counter(pkt[m]).values()]
    mg_share = [v / n for v in Counter(mag[m]).values()]
    return sum(s >= 0.10 for s in pk_share) >= 2 or sum(s >= 0.03 for s in mg_share) >= 3
good_rows = [r for r in set(rows) if _is_tt(r)]
out('Teletext rows:', sorted(int(r) for r in good_rows))
sel = np.isin(rows, good_rows)
out("Packet number distribution (all):", dict(sorted(Counter(pkt[sel]).items())))
# page tracking
cur = {}
pages = defaultdict(lambda: defaultdict(Counter))  # (mag,page,sub) -> row -> Counter(bytes)
hdrs = defaultdict(Counter); pagecount = Counter(); special = defaultdict(list)
first_hdr = {}; last_hdr = {}
for i in range(len(P)):
    if not sel[i]:
        n = int(pkt[i])
        if n >= 26 and HERR[P[i][3]] == 0 and HERR[P[i][4]] == 0:
            special[n].append((int(mag[i]), int(frames[i]), int(rows[i]), bytes(P[i][5:45]).hex()))
        continue
    p = P[i]; m = int(mag[i]); n = int(pkt[i])
    if n == 0:
        u, t = HAM[p[5]], HAM[p[6]]
        s = [HAM[x] for x in p[7:13]]
        if min(u, t, *s) < 0: cur.pop(m, None); continue
        page = t << 4 | u
        sub = (s[0] | (s[1] & 7) << 4 | s[2] << 8 | (s[3] & 3) << 12)
        ctrl = dict(C4=s[1] >> 3 & 1, C5=s[3] >> 2 & 1, C6=s[3] >> 3 & 1, C7_10=s[4], C11_14=s[5])
        key = (m, page, sub); cur[m] = key; pagecount[(m, page)] += 1
        pages[key][0][bytes(p[13:45])] += 1
        if page != 0xFF:
            h = txt(p[13:45]); hdrs[m][h[:24]] += 1
            first_hdr.setdefault(m, (int(frames[i]), h)); last_hdr[m] = (int(frames[i]), h)
        if (page & 0xF) > 9 or (page >> 4) > 9:
            special['hexpage'].append((m, page, sub, int(frames[i])))
    elif n <= 25:
        if m in cur: pages[cur[m]][n][bytes(p[5:45])] += 1
    elif HERR[p[3]] == 0 and HERR[p[4]] == 0:
        special[n].append((m, int(frames[i]), int(rows[i]), bytes(p[5:45]).hex()))
out("\n### Header rows (most frequent per magazine)")
for m in sorted(hdrs):
    for h, v in hdrs[m].most_common(3): out(f"M{m} x{v}: |{h}|")
    out(f"   first: |{first_hdr[m][1]}|  last: |{last_hdr[m][1]}|")
out("\n### Pages per magazine")
for m in range(1, 9):
    pl = sorted(pg for (mm, pg) in pagecount if mm == m)
    if pl: out(f"M{m}: {len(pl)} pages: " + ' '.join(f'{m}{pg:02X}' for pg in pl))
hp = Counter((m, pg) for m, pg, s, f in special['hexpage'])
if hp: out("Hex page numbers:", ' '.join(f'{m}{pg:02X}x{v}' for (m, pg), v in sorted(hp.items())))
for n in sorted(k for k in special if k != 'hexpage'):
    L = special[n]
    out(f"\n### Packet X/{n}: {len(L)} packets, magazines {dict(Counter(x[0] for x in L))}, rows {dict(Counter(x[2] for x in L))}")
    dc = Counter(HAM[bytes.fromhex(x[3])[0]] for x in L)
    out("  Designation codes:", dict(dc))
    for x in L[:6]: out(f"  M{x[0]} fr{x[1]} row{x[2]} {x[3]}  |{txt(bytes.fromhex(x[3]))}|")
    if n == 30:
        for x in L:
            b = bytes.fromhex(x[3])
            if x[0] == 8 and HAM[b[0]] in (0, 1):
                ni = b[7] << 8 | b[8]
                try:
                    mjd = int(''.join(str(((b[10 + k // 2] >> (4 * (1 - k % 2))) & 15) - 1) for k in range(1, 6)))
                    utc = ''.join(str(((b[13 + k // 2] >> (4 * (1 - k % 2))) & 15) - 1) for k in range(6))
                except Exception: mjd = None; utc = ''
                out(f"  8/30 Fmt1: NI 0x{ni:04X} MJD {mjd} UTC {utc} Status |{txt(b[20:40])}|"); break
# write pages
pd = os.path.join(d, 'pages'); os.makedirs(pd, exist_ok=True)
for key, rws in pages.items():
    m, page, sub = key
    lines = []
    for r in range(26):
        if r not in rws: continue
        arr = np.array([np.frombuffer(b, np.uint8) for b, c in rws[r].items() for _ in range(c)])
        maj = np.array([Counter(arr[:, j] & 0x7f).most_common(1)[0][0] for j in range(arr.shape[1])])
        lines.append(f"{r:2d} |{txt(maj)}|")
    nm = f"{m}{page:02X}-{sub:04X}.txt"
    open(os.path.join(pd, nm), 'w').write('\n'.join(lines) + '\n')
# unknown rows
pass
unk_rows = [r for r in range(32) if st[r].get('unknown', 0) > 0.05 * fr_total and r not in good_rows]
out("\n## Rows with unclassified signal:", unk_rows)
open(os.path.join(d, 'report.txt'), 'w').write('\n'.join(R) + '\n')
print('\n'.join(R))
