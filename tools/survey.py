"""Survey a bttv PAL VBI capture (2048 samples x 32 rows per frame, unsigned 8 bit).

usage: survey.py <file.vbi | file.flac | file.tbc | file.tbcy | file.tbcc | -> <outdir>
Writes packets.bin (WST packets), vps.txt, l16.txt, dataline.txt (data line words 3/4 of lines 16/329,
see dataline.py), summary.json, extra.npz (unknown lines, templates)."""
import sys, os, json, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wst2, vps, dataline
from wstslice import slice_line
from l16bits import bits_of
from vbisource import iter_frames
HERR = wst2.HERR
src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
pk_out = open(os.path.join(outdir, 'packets.bin'), 'wb')   # u32 frame, u8 row, u8 method, 45 bytes
vps_out = open(os.path.join(outdir, 'vps.txt'), 'w')
l16_out = open(os.path.join(outdir, 'l16.txt'), 'w')
unk = defaultdict(list); unk_frames = defaultdict(list)
act = np.zeros(32, int); stat = Counter(); fcount = 0
templates = {}; tmpl_buf = defaultdict(list); static_rows = set()
amp_hist = []
taps, off = wst2.DEFAULT_TAPS, wst2.DEFAULT_OFF
fit_y, fit_b, refits = [], [], 0
row_try = Counter(); row_ok = Counter()   # skip Viterbi on rows that never yield WST
def emit(fr, r, method, b):
    pk_out.write(np.array([fr], '<u4').tobytes() + bytes([r, method]) + bytes(b[:45]))
for a in iter_frames(src, 64):
    n = len(a)
    black = np.median(a[:, :, 20:100], axis=2); top = np.percentile(a[:, :, 150:1950], 98, axis=2)
    amp = top - black
    for i in range(n):
        fr = fcount + i
        for r in range(32):
            if amp[i, r] < 35: continue
            act[r] += 1
            line = a[i, r]
            if r in (9, 25):
                v = vps.decode(line)
                if v is not None and v[1] > 0.12 and v[0][9] != 0 or (v is not None and v[1] > 0.2):
                    vps_out.write(f'{fr} {r} {v[0].tobytes().hex()} {v[1]:.2f}\n'); stat[(r, 'vps')] += 1; continue
                s = bits_of(line)
                if s and len(s) >= 30 and s.startswith('1111111'):
                    l16_out.write(f'{fr} {r} {s}\n'); stat[(r, 'l16')] += 1; continue
            if r not in templates:
                tmpl_buf[r].append(line.astype(np.float32))
                if len(tmpl_buf[r]) == 60:
                    T = np.median(np.array(tmpl_buf[r]), 0)
                    d = [np.sqrt(np.mean((x - T) ** 2)) for x in tmpl_buf[r]]
                    templates[r] = T
                    if np.median(d) < 0.10 * (T.max() - T.min()): static_rows.add(r)
                    tmpl_buf[r] = None
            if r in static_rows:
                d = np.sqrt(np.mean((line.astype(np.float32) - templates[r]) ** 2))
                if d < 0.15 * amp[i, r]:
                    stat[(r, 'static')] += 1; continue
            b = slice_line(line)
            if b is not None and b[2] == 0x27 and HERR[b[3]] == 0 and HERR[b[4]] == 0:
                emit(fr, r, 0, b); stat[(r, 'wst')] += 1; row_ok[r] += 1; continue
            row_try[r] += 1
            if row_try[r] > 400 and row_ok[r] < 0.02 * row_try[r]:
                pass   # row does not carry WST: skip expensive Viterbi
            else:
                d = wst2.decode(line, taps, off)
                if d is not None and d[1] <= 1:
                    emit(fr, r, 1 + d[1], d[0]); stat[(r, 'wst')] += 1; row_ok[r] += 1
                    if d[1] == 0 and refits < 2:
                        fit_y.append(d[3]); fit_b.append(d[4])
                        if len(fit_y) >= 300:
                            taps, off = wst2.fit(fit_y, fit_b); fit_y, fit_b = [], []; refits += 1
                    continue
            stat[(r, 'unknown')] += 1
            if len(unk[r]) < 300 and (stat[(r, 'unknown')] % 5 == 1):
                unk[r].append(line.copy()); unk_frames[r].append(fr)
    fcount += n
    amp_hist.append(((amp > 35).mean(0) * 100).astype(np.uint8))
summary = dict(frames=fcount, active={r: int(act[r]) for r in range(32)},
               stat={f'{k[0]}:{k[1]}': v for k, v in sorted(stat.items())}, static_rows=sorted(static_rows),
               taps=[round(float(t), 2) for t in taps] + [round(float(off), 2)])
json.dump(summary, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(outdir, 'extra.npz'), amp_hist=np.array(amp_hist),
                    **{f'unk{r}': np.array(unk[r]) for r in unk}, **{f'unkf{r}': np.array(unk_frames[r]) for r in unk},
                    **{f'tmpl{r}': templates[r] for r in templates})
vps_out.close()
dataline.write(outdir)
