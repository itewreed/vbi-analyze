#!/usr/bin/env python3
"""Descramble a Nagravision Syster recording (vhs-decode / ld-decode TBC) using the
Syster VBI data of the recording itself.

The Syster lines 14/15/327/328 are sliced from the TBC (see syster_vbi.py), the ECMs
are decrypted with the given key, and the per-field line order is rebuilt with the
hacktv PRBS and line-order model. The control word timing was measured on a
Teleclub recording of March 1991 (BERICHT.md section 7.7):

  * every 25 frames a block with an odd ECM index c and ap = f announces a new CW,
    25 frames later the same c is sent with ap = 0 (takeover);
  * the PRBS is reset with that CW at field 2F+3 (F = frame of the ap = 0 block,
    counted from a first field) and again 50 fields later;
  * between resets the PRBS advances once per field.

Fields whose CW is unknown can be descrambled without any key with --search: all
32768 (s, r) pairs are tried and the order giving the smoothest picture wins. A new
order is only used when it makes the field clearly smoother, so clear video is kept.
apply also writes <out>.fields.txt (field, cw/search/kept, s, r, roughness before/after).

  syster_descramble.py vbi    in.tbcy cws.txt --hacktv DIR [--key 0000000000001234]
  syster_descramble.py apply  in.tbcy cws.txt out.tbcy --hacktv DIR [--chroma in.tbcc out.tbcc]
                              [--search] [--fields A:B]
  syster_descramble.py search in.tbcy out.txt --hacktv DIR --fields A:B

The output TBC has the same layout as the input; copy the .tbc.json next to it.

Chroma (--chroma): the Syster scrambler keeps each line's colour burst and PAL switch
in place, so a moved line's chroma has to be re-referenced: it is demodulated against
the burst it was transmitted with and remodulated with the burst of its new line,
mirrored in V when the PAL switch differs (--chroma-mode burst). This was checked on
synthetic data; --chroma-mode move copies the samples unchanged.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import syster_vbi as sv

LINE0 = 22        # TBC row of the first scrambled line (field lines 23 / 336)
NL = 287          # scrambled lines per field
X0 = 140          # first sample moved: everything after the colour burst
SR2 = 0x1FFFFFFF


class Tbc:
    def __init__(self, path):
        meta = path.rsplit('.', 1)[0] + '.tbc.json'
        self.json = json.load(open(meta)) if os.path.exists(meta) else None
        vp = self.json['videoParameters'] if self.json else {}
        self.w, self.h = vp.get('fieldWidth', 1135), vp.get('fieldHeight', 313)
        self.a = np.memmap(path, np.uint16, 'r')
        self.n = len(self.a) // (self.w * self.h)
        self.a = self.a[:self.n * self.w * self.h].reshape(self.n, self.h, self.w)
        ff = [f.get('isFirstField') for f in self.json['fields'][:4]] if self.json else [True]
        self.first = 0 if ff[0] else 1        # index of the first "first field"


def vbi_rows(t, f0, f1):
    """Syster rows 13/14 of both fields of frames f0..f1, resampled to the bttv grid."""
    x, q = np.arange(t.w), np.arange(2048) / 2 + 125.5
    out = np.zeros((f1 - f0, 4, 2048), np.float32)
    for f in range(f0, f1):
        n = t.first + 2 * f
        if n + 1 >= t.n:
            break
        for k, (d, r) in enumerate([(0, 13), (0, 14), (1, 13), (1, 14)]):
            out[f - f0, k] = np.interp(q, x, t.a[n + d, r].astype(np.float32)) / 256
    return out


def cmd_vbi(a):
    t = Tbc(a.tbc); sy = sv.Syster(a.hacktv); key = bytes.fromhex(a.key)
    log = lambda s: print(s, file=sys.stderr)
    nfr = (t.n - t.first) // 2
    w = sv.fit([sv.feats(r, sv.timing(r)) for fr in vbi_rows(t, 0, min(nfr, 300)) for r in fr.astype(float)], log=log)
    first, odd, events, good, tot = None, [], [], 0, 0
    for f0 in range(0, nfr, 500):
        for i, fr in enumerate(vbi_rows(t, f0, min(nfr, f0 + 500))):
            for r in fr.astype(float):
                _, _, by, ok = sv.slice_line(r, w); tot += 1; good += ok
                if not ok or by[4] != sv.SEQ[8]:
                    continue
                m = by[5:26]; c = ((m[1] & 7) << 4) | (m[2] >> 4); ap = m[2] & 15
                if not c & 1:
                    first = m[3:11]
                elif first is not None:
                    odd.append((f0 + i, c, first + m[3:11]))
                    if ap in (0, 15):
                        events.append((f0 + i + (25 if ap == 15 else 0), c))
        log('frame %d: %d of %d lines pass the CRC, %d takeovers' % (min(nfr, f0 + 500), good, tot, len(events)))
    # Takeovers come every 50 frames with the odd ECM index c rising by 2; fill the ones
    # whose ap 0 / ap f block was lost, then take the latest ECM with that c before each.
    seen = dict(events)
    if not seen:
        sys.exit('no CW takeover found')
    f0, c0 = min(seen.items())
    out = open(a.out, 'w')
    for ft in range(f0 % 50, nfr, 50):
        c = (c0 + 2 * ((ft - f0) // 50)) % 128
        ecm = [e for f, cc, e in odd if cc == c and ft - 100 <= f <= ft]
        if not ecm:
            continue
        p = sy.decrypt_ecm(ecm[-1], key)
        out.write('%d %d %s %016x %s %s\n' % (ft, c, 'seen' if ft in seen else 'filled', sy.control_word(p),
                  sv.date_word(p[10] | p[11] << 8), p.hex()))


def resets(path, first):
    """(reset field, cw) from the takeover list, each CW used for two resets."""
    rs = {}
    for l in open(path):
        ft, cw = int(l.split()[0]), int(l.split()[3], 16)
        for d in (3, 53):
            rs[first + 2 * ft + d] = cw
    return sorted(rs.items())


def all_orders(sy):
    tab = np.array(sy.t['_key_table1'])
    S, R = np.meshgrid(np.arange(128), np.arange(256), indexing='ij'); S = S.ravel(); R = R.ravel()
    N = len(S); b = np.tile(np.arange(-32, 0), (N, 1)); O = np.zeros((N, NL), np.int16); idx = np.arange(N)
    for i in range(NL):
        j = tab[(R + (2 * S + 1) * i) & 0xFF] if i <= 254 else np.full(N, i - 255)
        O[idx, b[idx, j] + 32] = i; b[idx, j] = i
    return O


BURST = slice(100, 136)   # colour burst samples (4fsc PAL)


def analytic(rows):
    """Analytic signal of 4fsc chroma rows, with the column carrier removed (baseband)."""
    x = rows.astype(np.float64); dc = x.mean(1)
    X = np.fft.fft(x - dc[:, None], axis=1); n = x.shape[1]
    X[:, n // 2 + 1:] = 0; X[:, 1:n // 2] *= 2
    return np.fft.ifft(X, axis=1) * np.exp(-0.5j * np.pi * np.arange(n)), dc


def burst_state(m):
    """Unit burst phasor of every row and the PAL switch (+1/-1) by row parity."""
    q = m[:, BURST].mean(1); q /= np.abs(q) + 1e-9
    d = np.angle(q[1:] * np.conj(q[:-1]))              # about +-90 deg, alternating
    s0 = 1 if (np.sin(d) * (-1.) ** np.arange(len(d))).sum() > 0 else -1
    return q, s0 * (-1) ** np.arange(len(q))


def rephase(t, k, o):
    """Chroma of field k in picture order, each line re-referenced from the burst it was
    transmitted with to the burst of the line it is moved to (PAL switch included)."""
    mp, dcp = analytic(t.a[k - 1, LINE0:LINE0 + NL]); mk, dck = analytic(t.a[k, LINE0:LINE0 + NL])
    qp, sp = burst_state(mp); qk, sk = burst_state(mk)
    m = np.concatenate([mp[255:], mk[:255]]); q = np.concatenate([qp[255:], qk[:255]])
    sw = np.concatenate([sp[255:], sk[:255]]); dc = np.concatenate([dcp[255:], dck[:255]])
    w = m * np.conj(q)[:, None]
    out = np.empty_like(m); out[o] = w; sw_o = np.empty_like(sw); sw_o[o] = sw; dc_o = np.empty_like(dc); dc_o[o] = dc
    flip = sw_o != sk
    out[flip] = np.conj(out[flip])
    out *= qk[:, None]
    c = (out * np.exp(0.5j * np.pi * np.arange(m.shape[1]))).real + dc_o[:, None]
    return np.clip(np.round(c), 0, 65535).astype(np.uint16)


def transmitted(t, k):
    """The 287 transmitted lines that carry the picture of field k (32 from field k-1)."""
    return np.concatenate([t.a[k - 1, LINE0 + 255:LINE0 + NL], t.a[k, LINE0:LINE0 + 255]])


def search(t, k, INV, cols=np.arange(200, 1100, 9)):
    tx = transmitted(t, k)[:, cols].astype(np.float32); sc = np.empty(len(INV))
    for i in range(0, len(INV), 2048):
        sc[i:i + 2048] = np.abs(np.diff(tx[INV[i:i + 2048]], axis=1)).mean(axis=(1, 2))
    i = int(np.argmin(sc))
    return i >> 8, i & 255, sc[i]


def cmd_search(a):
    t = Tbc(a.tbc); sy = sv.Syster(a.hacktv); INV = np.argsort(all_orders(sy), axis=1)
    k0, k1 = map(int, a.fields.split(':')); out = open(a.out, 'w')
    for k in range(max(k0, 1), min(k1, t.n)):
        s, r, sc = search(t, k, INV); out.write('%d %d %d %.0f\n' % (k, s, r, sc)); out.flush()


COLS = np.arange(200, 1100, 9)


def roughness(img):
    return np.abs(np.diff(img[:, COLS].astype(np.float32), axis=0)).mean()


def cmd_apply(a):
    """Reorder every field that has a CW (or, with --search, a clear best order).
    A field is left as it is when the new order is not clearly smoother, so clear
    video before, after and between scrambled parts is never shuffled."""
    t = Tbc(a.tbc); sy = sv.Syster(a.hacktv)
    rs = resets(a.cws, t.first); RK = [r for r, _ in rs]
    O = all_orders(sy) if a.search else None; INV = np.argsort(O, axis=1) if a.search else None
    files = [(t, open(a.out, 'wb'))]
    if a.chroma:
        files.append((Tbc(a.chroma[0]), open(a.chroma[1], 'wb')))
    k0, k1 = map(int, a.fields.split(':')) if a.fields else (0, t.n)
    seeds = {}; stat = {'cw': 0, 'search': 0, 'kept': 0, 'none': 0}
    log = open(a.out + '.fields.txt', 'w')
    for k in range(k0, min(k1, t.n)):
        j = np.searchsorted(RK, k, 'right') - 1
        sr, how = None, 'none'
        if k > 0 and j >= 0 and k - RK[j] < 50:
            if RK[j] not in seeds:
                seeds = {RK[j]: sy.field_seeds(rs[j][1], 50)}
            sr, how = seeds[RK[j]][k - RK[j]], 'cw'
        elif k > 0 and a.search and roughness(t.a[k, LINE0:LINE0 + NL]) > a.min_rough:
            sr, how = search(t, k, INV)[:2], 'search'
        o = None
        if sr is not None:
            o = np.array(sy.field_order(*sr)) if O is None else O[sr[0] * 256 + sr[1]]
            before = roughness(t.a[k, LINE0:LINE0 + NL]); tx = transmitted(t, k)
            img = np.empty_like(tx); img[o] = tx
            after = roughness(img)
            if after > 0.75 * before:
                o, how = None, 'kept'
            log.write('%d %s %d %d %.0f %.0f\n' % (k, how, sr[0], sr[1], before, after))
        stat[how] += 1
        for tt, fo in files:
            fld = np.array(tt.a[k])
            if o is not None and tt is not t and a.chroma_mode == 'burst':
                fld[LINE0:LINE0 + NL, X0:] = rephase(tt, k, o)[:, X0:]
            elif o is not None:
                img = np.empty((NL, tt.w - X0), np.uint16)
                img[o] = transmitted(tt, k)[:, X0:]
                fld[LINE0:LINE0 + NL, X0:] = img
            fo.write(fld.tobytes())
        if k % 1000 == 0:
            print('field %d %s' % (k, stat), file=sys.stderr)
    print('fields: %s' % stat)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('vbi'); p.add_argument('tbc'); p.add_argument('out')
    p.add_argument('--hacktv', required=True); p.add_argument('--key', default='0000000000001234')
    p.set_defaults(fn=cmd_vbi)
    p = sub.add_parser('apply'); p.add_argument('tbc'); p.add_argument('cws'); p.add_argument('out')
    p.add_argument('--hacktv', required=True); p.add_argument('--chroma', nargs=2)
    p.add_argument('--chroma-mode', choices=['burst', 'move'], default='burst',
                   help='burst: re-reference each moved line to the burst of its new line (default); move: copy samples')
    p.add_argument('--search', action='store_true'); p.add_argument('--fields')
    p.add_argument('--min-rough', type=float, default=3000, help='search only fields rougher than this (16-bit TBC units)')
    p.set_defaults(fn=cmd_apply)
    p = sub.add_parser('search'); p.add_argument('tbc'); p.add_argument('out')
    p.add_argument('--hacktv', required=True); p.add_argument('--fields', required=True)
    p.set_defaults(fn=cmd_search)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
