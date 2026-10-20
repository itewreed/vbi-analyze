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
32768 (s, r) pairs are tried and the order giving the smoothest picture wins.

  syster_descramble.py vbi    in.tbcy cws.txt --hacktv DIR [--key 0000000000001234]
  syster_descramble.py apply  in.tbcy cws.txt out.tbcy --hacktv DIR [--chroma in.tbcc out.tbcc]
                              [--search] [--fields A:B]
  syster_descramble.py search in.tbcy out.txt --hacktv DIR --fields A:B

The output TBC has the same layout as the input; copy the .tbc.json next to it.
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


def cmd_apply(a):
    t = Tbc(a.tbc); sy = sv.Syster(a.hacktv)
    rs = resets(a.cws, t.first); RK = [r for r, _ in rs]
    O = all_orders(sy) if a.search else None; INV = np.argsort(O, axis=1) if a.search else None
    files = [(t, open(a.out, 'wb'))]
    if a.chroma:
        files.append((Tbc(a.chroma[0]), open(a.chroma[1], 'wb')))
    k0, k1 = map(int, a.fields.split(':')) if a.fields else (0, t.n)
    seeds = {}; stat = {'cw': 0, 'search': 0, 'none': 0}
    for k in range(k0, min(k1, t.n)):
        j = np.searchsorted(RK, k, 'right') - 1
        sr = None
        if j >= 0 and k - RK[j] < 50:
            if RK[j] not in seeds:
                seeds = {RK[j]: sy.field_seeds(rs[j][1], 50)}
            sr = seeds[RK[j]][k - RK[j]]; stat['cw'] += 1
        elif a.search and k > 0:
            sr = search(t, k, INV)[:2]; stat['search'] += 1
        else:
            stat['none'] += 1
        for tt, fo in files:
            fld = np.array(tt.a[k])
            if sr is not None and k > 0:
                o = np.array(sy.field_order(*sr)) if O is None else O[sr[0] * 256 + sr[1]]
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
    p.add_argument('--search', action='store_true'); p.add_argument('--fields')
    p.set_defaults(fn=cmd_apply)
    p = sub.add_parser('search'); p.add_argument('tbc'); p.add_argument('out')
    p.add_argument('--hacktv', required=True); p.add_argument('--fields', required=True)
    p.set_defaults(fn=cmd_search)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
