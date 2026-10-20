#!/usr/bin/env python3
"""Equalising teletext slicer for VHS recordings with heavy low-pass filtering.

Input: a numpy array (frames, rows, 2048) of uint8 bttv raw VBI lines (35.46895 MHz),
rows in transmission order. Steps:
  train   fit a linear equaliser (decision directed), then a decision-feedback
          equaliser (DFE), then refine the DFE on page headers whose fixed text
          (e.g. "ARD/ZDF Sa  7 Jun") is learned from the majority of headers.
  decode  slice every line: timing search with the linear equaliser, then DFE
          with odd-parity correction (flip the least reliable bit of a bad byte).
  pages   assemble pages per magazine and vote each character over all repeats;
          write a text dump (mosaics as Unicode sextants) and optional PNGs.

Example:
  vt_equalise.py train lines.npy w.npz --start 88 118
  vt_equalise.py decode lines.npy w.npz pk.pkl --start 88 118
  vt_equalise.py pages pk.pkl pages.txt --png-dir png/
"""
import argparse, collections, pickle, sys
import numpy as np

H = [0x15,0x02,0x49,0x5e,0x64,0x73,0x38,0x2f,0xd0,0xc7,0x8c,0x9b,0xa1,0xb6,0xfd,0xea]
HAM = np.full(256, -1)
for i, b in enumerate(H): HAM[b] = i
HC = np.full(256, -1)          # Hamming 8/4 with single-error correction
for v in range(256):
    d = [bin(v ^ c).count('1') for c in H]
    if min(d) <= 1: HC[v] = d.index(min(d))
PAR = np.array([bin(v).count('1') & 1 for v in range(256)])
W8 = 1 << np.arange(8)
FS = 35.46895; SPB = FS / 6.9375
NB = 8 * 45
FRAME = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0, 1,1,1,0,0,1,0,0])
M = np.arange(-20, 21)          # equaliser taps (samples)
KB = 6                          # feedback taps (bits)

def enc_par(c): return c | (0x80 if bin(c).count('1') % 2 == 0 else 0)
def sample(x, pos):
    i = pos.astype(int); fr = pos - i
    return x[..., i] * (1 - fr) + x[..., i + 1] * fr
def grid(lo, hi, rates=(0.998, 1.0, 1.002)):
    return np.array([s + np.arange(NB) * SPB * r for r in rates for s in np.arange(lo, hi, SPB / 16)])
def filt(x, w):
    if w is None: return x - np.convolve(x, np.ones(100) / 100, 'same')
    return np.convolve(np.pad(x, 20, mode='edge'), w[:-1][::-1], 'valid') + w[-1]
def best_start(x, P, w):
    B = (sample(filt(x, w), P) > 0).astype(int)
    fm = (B[:, 16:24] == FRAME[16:]).sum(1)
    by = (B[:, 24:].reshape(len(B), -1, 8) * W8).sum(2)
    sc = np.where(fm == 8, PAR[by[:, 2:]].mean(1) + 0.5 * ((HAM[by[:, 0]] >= 0).astype(int) + (HAM[by[:, 1]] >= 0)), 0)
    k = sc.argmax(); return k, sc[k], by[k], B[k]
def feat(x, pos):
    xp = np.pad(x, 21, mode='edge'); fl = np.floor(pos).astype(int); fr = pos - fl
    i = fl[:, None] + M[None, :] + 21
    return np.hstack([xp[i] * (1 - fr[:, None]) + xp[i + 1] * fr[:, None], np.ones((len(pos), 1))])
def dfe_feats(x, pos, bits):
    b = 2 * np.asarray(bits) - 1; FB = np.zeros((len(pos), KB))
    for k in range(1, KB + 1): FB[k:, k - 1] = b[:-k]
    return np.hstack([feat(x, pos), FB])
def dfe_decode(x, pos, wf, wb, fix=True):
    ff = feat(x, pos) @ wf; out = np.zeros(NB, int); soft = np.zeros(NB); out[:24] = FRAME
    for n in range(24, NB):
        v = ff[n] + wb @ (2 * out[n - KB:n][::-1] - 1); soft[n] = v; out[n] = v > 0
        if fix and n >= 40 and (n - 24) % 8 == 7 and out[n - 7:n + 1].sum() % 2 == 0:
            out[n - 7 + np.argmin(np.abs(soft[n - 7:n + 1]))] ^= 1
    return out, soft
def bytes_of(out): return (out[24:].reshape(-1, 8) * W8).sum(1)
def trusted(by):
    m = np.zeros(NB, bool); m[16:24] = True
    if HAM[by[0]] >= 0: m[24:32] = True
    if HAM[by[1]] >= 0: m[32:40] = True
    for i, g in enumerate(PAR[by[2:]] == 1):
        if g: m[40 + 8 * i:48 + 8 * i] = True
    return m
def lstsq(X, T): return np.linalg.lstsq(np.vstack(X), np.concatenate(T), rcond=None)[0]
def lines_of(a, step):
    for f in range(0, a.shape[0], step):
        for j in range(a.shape[1]): yield f, j, a[f, j].astype(float)

def train(a, lo, hi, step=9, log=print):
    P = grid(lo, hi); w = None
    for it in range(5):                                   # linear, decision directed
        X = []; T = []; scs = []
        for f, j, x in lines_of(a, step):
            k, sc, by, B = best_start(x, P, w); scs.append(sc)
            if sc > 1.5: m = trusted(by); X.append(feat(x, P[k][m])); T.append(2 * B[m] - 1)
        w = lstsq(X, T); log('linear %d: median score %.3f' % (it, np.median(scs)))
    cache = []
    for f, j, x in lines_of(a, step):
        k, sc, by, B = best_start(x, P, w)
        if sc > 0: cache.append((x, P[k], B, by))
    wf, wb = w.copy(), np.zeros(KB)
    for it in range(4):                                   # DFE, decision directed
        X = []; T = []; pars = []
        for x, pos, B, by in cache:
            if it: B, _ = dfe_decode(x, pos, wf, wb, fix=False); by = bytes_of(B)
            pars.append(PAR[by[2:]].mean())
            if pars[-1] > 0.7: m = trusted(by); X.append(dfe_feats(x, pos, B)[m]); T.append(2 * B[m] - 1)
        ww = lstsq(X, T); wf, wb = ww[:-KB], ww[-KB:]
        log('dfe %d: median parity %.3f' % (it, np.median(pars)))
    hdr = []; C = [collections.Counter() for _ in range(42)]   # header template
    for x, pos, _, _ in cache:
        B, _ = dfe_decode(x, pos, wf, wb); by = bytes_of(B)
        m, p = HC[by[0]], HC[by[1]]
        if m < 0 or p < 0 or (m >> 3) | (p << 1): continue
        u, t = HC[by[2]], HC[by[3]]
        if not (0 <= u <= 9 and 0 <= t <= 9): continue
        hdr.append((x, pos, B, m, p, u, t))
        for i in range(10, 42): C[i][by[i] & 0x7f] += 1
    tmpl = {i: enc_par(C[i].most_common(1)[0][0]) for i in range(10, 42) if C[i] and C[i].most_common(1)[0][1] > 0.6 * len(hdr)}
    log('header template: ' + ''.join(chr(tmpl[i] & 0x7f) if i in tmpl and tmpl[i] & 0x7f >= 32 else '_' for i in range(10, 42)))
    X = []; T = []
    for x, pos, B, m, p, u, t in hdr:
        known = {0: H[m], 1: H[p], 2: H[u], 3: H[t]}; known.update(tmpl)
        lab = B.copy(); msk = np.zeros(NB, bool); msk[16:24] = True
        for i, v in known.items(): lab[24 + 8 * i:32 + 8 * i] = (v >> np.arange(8)) & 1; msk[24 + 8 * i:32 + 8 * i] = True
        msk[:KB + 16] = False; X.append(dfe_feats(x, pos, lab)[msk]); T.append(2 * lab[msk] - 1)
    if X:
        ww = lstsq(X, T); wf, wb = ww[:-KB], ww[-KB:]
    return w, wf, wb

def decode(a, w, wf, wb, lo, hi):
    P = grid(lo, hi); res = []
    for f, j, x in lines_of(a, 1):
        k, sc, by, B = best_start(x, P, w)
        if sc == 0: continue
        B1, _ = dfe_decode(x, P[k], wf, wb)
        res.append((f, j, bytes(bytes_of(B1).astype(np.uint8))))
    return res

G0 = {0x5b:'Ä',0x5c:'Ö',0x5d:'Ü',0x5e:'^',0x60:'°',0x7b:'ä',0x7c:'ö',0x7d:'ü',0x7e:'ß',0x40:'§'}
def assemble(res):
    cur = {}; votes = collections.defaultdict(lambda: [collections.Counter() for _ in range(40)])
    hdrs = collections.Counter(); ht = collections.defaultdict(lambda: [collections.Counter() for _ in range(32)])
    for f, j, by in res:
        m, p = HC[by[0]], HC[by[1]]
        if m < 0 or p < 0: continue
        mag = (m & 7) or 8; row = (m >> 3) | (p << 1)
        if row == 0:
            u, t = HC[by[2]], HC[by[3]]
            if not (0 <= u <= 9 and 0 <= t <= 9): cur[mag] = None; continue
            pg = mag * 100 + t * 10 + u; cur[mag] = pg; hdrs[pg] += 1
            for i in range(32):
                if PAR[by[10 + i]]: ht[pg][i][by[10 + i]] += 1
        elif 1 <= row <= 24 and cur.get(mag):
            for i in range(40):
                if PAR[by[2 + i]]: votes[(cur[mag], row)][i][by[2 + i]] += 1
    return hdrs, votes, ht
def best_codes(cols): return [c.most_common(1)[0][0] if c else 0x20 for c in cols]
def sext(c):
    b = (c & 0x1f) | ((c & 0x40) >> 1)
    if b == 0: return ' '
    if b == 63: return '█'
    if b == 21: return '▌'
    if b == 42: return '▐'
    return chr(0x1FB00 + b - 1 - (b > 21) - (b > 42))
def row_text(codes):
    out = []; gfx = False
    for c in codes:
        c &= 0x7f
        if c < 32:
            if 0x10 <= c <= 0x17: gfx = True
            elif c <= 7: gfx = False
            out.append(' ')
        elif gfx and not 0x40 <= c <= 0x5f: out.append(sext(c))
        else: out.append(G0.get(c, chr(c)))
    return ''.join(out)
def page_text(v, ht, pg):
    L = ['P%d  %s' % (pg, row_text(best_codes(ht[pg])))]
    for row in range(1, 25):
        L.append('%2d %s' % (row, row_text(best_codes(v[(pg, row)])) if (pg, row) in v else ''))
    return '\n'.join(L)
def render_png(v, ht, pg, path):
    from PIL import Image, ImageDraw, ImageFont
    COL = [(0,0,0),(255,0,0),(0,255,0),(255,255,0),(0,0,255),(255,0,255),(0,255,255),(255,255,255)]
    CW, CH = 12, 20
    F = ImageFont.truetype('DejaVuSansMono.ttf', 16)
    im = Image.new('RGB', (40 * CW, 25 * CH)); d = ImageDraw.Draw(im)
    rows = [(0, [0x20] * 8 + best_codes(ht[pg]))] + [(r, best_codes(v[(pg, r)])) for r in range(1, 25) if (pg, r) in v]
    for y, codes in rows:
        fg = 7; bg = 0; gfx = False; sep = False
        for x, c in enumerate(codes):
            c &= 0x7f; X, Y = x * CW, y * CH
            if c < 32:
                if c <= 7: fg, gfx = c, False
                elif 0x10 <= c <= 0x17: fg, gfx = c - 0x10, True
                elif c == 0x1c: bg = 0
                elif c == 0x1d: bg = fg
                elif c == 0x1a: sep = True
                elif c == 0x19: sep = False
            d.rectangle([X, Y, X + CW - 1, Y + CH - 1], fill=COL[bg])
            if c < 32: continue
            if gfx and not 0x40 <= c <= 0x5f:
                b = (c & 0x1f) | ((c & 0x40) >> 1); g = 1 if sep else 0
                for k in range(6):
                    if b >> k & 1:
                        cx, cy = X + (k % 2) * CW // 2, Y + (k // 2) * CH // 3
                        d.rectangle([cx + g, cy + g, cx + CW // 2 - 1, cy + CH // 3 - 1 + (k // 2 == 2)], fill=COL[fg])
            else: d.text((X, Y + 1), G0.get(c, chr(c)), font=F, fill=COL[fg])
    im.save(path)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('train'); t.add_argument('lines'); t.add_argument('weights')
    d = sp.add_parser('decode'); d.add_argument('lines'); d.add_argument('weights'); d.add_argument('out')
    for q in (t, d): q.add_argument('--start', nargs=2, type=float, default=(88, 118), help='sample window of the first run-in bit')
    p = sp.add_parser('pages'); p.add_argument('packets'); p.add_argument('txt'); p.add_argument('--png-dir'); p.add_argument('--min-headers', type=int, default=2)
    a = ap.parse_args()
    if a.cmd == 'train':
        w, wf, wb = train(np.load(a.lines), *a.start); np.savez(a.weights, w=w, wf=wf, wb=wb)
    elif a.cmd == 'decode':
        W = np.load(a.weights); res = decode(np.load(a.lines), W['w'], W['wf'], W['wb'], *a.start)
        pickle.dump(res, open(a.out, 'wb')); print(len(res), 'packets')
    else:
        import os
        hd, v, ht = assemble(pickle.load(open(a.packets, 'rb')))
        pgs = [pg for pg in sorted(hd) if hd[pg] >= a.min_headers]
        open(a.txt, 'w').write('\n'.join(page_text(v, ht, pg) + '\n' + '-' * 44 for pg in pgs) + '\n')
        if a.png_dir:
            os.makedirs(a.png_dir, exist_ok=True)
            for pg in pgs: render_png(v, ht, pg, os.path.join(a.png_dir, 'p%d.png' % pg))
        print(len(pgs), 'pages')
if __name__ == '__main__': main()
