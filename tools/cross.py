import sys, os, json, numpy as np, glob
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vps
from wst2 import HAM, HERR
from analyze_helpers import txt
root = sys.argv[1]
rev = lambda b: int(f'{b:08b}'[::-1], 2)
for d in sorted(glob.glob(root + '/*/')):
    name = os.path.basename(d.rstrip('/')).replace('_fs200', '')
    S = json.load(open(d + 'summary.json')); F = S['frames']
    st = defaultdict(dict)
    for k, v in S['stat'].items():
        r, c = k.split(':'); st[int(r)][c] = v
    out = [f"## {name}  ({F} Frames, {F/25/60:.1f} min)"]
    # unknown rows
    unk = [(r, st[r].get('unknown', 0) / F) for r in range(32) if st[r].get('unknown', 0) > 0.3 * F]
    if unk: out.append("  unclassified rows (>30% of frames): " + ', '.join(f"r{r}({p*100:.0f}%)" for r, p in unk))
    # VPS
    vl = [l.split() for l in open(d + 'vps.txt')] if os.path.exists(d + 'vps.txt') else []
    if len(vl) > 50:
        C = Counter(); ch = Counter(); seqs = []
        for fr, r, h, c in vl:
            b = bytes.fromhex(h); p = vps.parse(b)
            C[(p['cni'], p['day'], p['mon'], p['hour'], p['min'])] += 1
            seqs.append(rev(b[1]) & 0x7f)
        # find period-8 text by majority over positions using frame index
        best = None
        for per in (8, 10, 12, 16):
            pos = defaultdict(Counter)
            for (fr, r, h, c), v in zip(vl, seqs): pos[int(fr) % per][v] += 1
            s = ''.join(chr(pos[i].most_common(1)[0][0]) if pos[i] else '?' for i in range(per))
            q = sum(pos[i].most_common(1)[0][1] for i in range(per) if pos[i]) / len(vl)
            if best is None or q > best[0] + 0.05: best = (q, s)
        top = C.most_common(4)
        out.append(f"  VPS ({len(vl)} lines): " + '; '.join(f"CNI {k[0]:03X} {k[1]:02d}.{k[2]:02d}. {k[3]:02d}:{k[4]:02d} x{v}" for k, v in top))
        out.append(f"  data word 4 text (period {len(best[1])}, consistency {best[0]*100:.0f}%): |{best[1]}|")
    ll = [l.split() for l in open(d + 'l16.txt')] if os.path.exists(d + 'l16.txt') else []
    if len(ll) > 50:
        C = Counter((r, s) for fr, r, s in ll)
        out.append(f"  L16 1-Mbit/s biphase ({len(ll)} lines): " + '; '.join(f"r{r} {s} x{v}" for (r, s), v in C.most_common(4)))
    raw = np.fromfile(d + 'packets.bin', np.uint8); raw = raw[:len(raw) // 51 * 51].reshape(-1, 51)
    P = raw[:, 6:]; rows = raw[:, 4]
    ok = (HERR[P[:, 3]] == 0) & (HERR[P[:, 4]] == 0)
    mag = HAM[P[:, 3]] & 7; mag[mag == 0] = 8
    pkt = (HAM[P[:, 3]] >> 3) | (HAM[P[:, 4]] << 1)
    for n in (26, 27, 28, 29, 30, 31):
        m = ok & (pkt == n)
        if m.sum() < 30: continue
        C = Counter(zip(mag[m], rows[m]))
        dom = [(k, v) for k, v in C.most_common(6) if v >= 10]
        if not dom: continue
        out.append(f"  X/{n}: {m.sum()} packets; (mag,row): " + ', '.join(f"M{k[0]}r{k[1]}:{v}" for k, v in dom))
        mm = m & (mag == dom[0][0][0])
        heads = Counter(bytes(p[5:9]).hex() for p in P[mm])
        out.append(f"     most frequent head bytes M{dom[0][0][0]}: " + ', '.join(f"{k}:{v}" for k, v in heads.most_common(4)))
    print('\n'.join(out))
