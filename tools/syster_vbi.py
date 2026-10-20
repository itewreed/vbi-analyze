#!/usr/bin/env python3
"""Decode Nagravision Syster VBI data (Premiere, lines 14/15/327/328) from bttv raw VBI.

The line format, block layout, ECM cipher, PRBS and line-shuffle model follow
hacktv (src/syster.c, src/syster-ca.c). The cipher and key tables are read from
a local hacktv checkout at run time (--hacktv), they are not copied here.

Input: .npy array (frames, rows, 2048) uint8 holding the four Syster rows of
each frame in transmission order (bttv rows 7, 8, 23, 24 = lines 14, 15, 327, 328).

  syster_vbi.py train  lines.npy w.npy                    # fit the slicer equaliser
  syster_vbi.py decode lines.npy w.npy lines.pkl          # slice, check sync and CRC
  syster_vbi.py blocks lines.pkl blocks.txt --hacktv DIR --key 0000000000001234
                                                          # blocks, parity, ECM, CW, per-field s/r
"""
import argparse, os, pickle, re, sys, collections
import numpy as np

SPB = 35.46895 / 4.4375                     # samples per bit, 284 fH NRZ
SYNC = bytes([0x55, 0xD0, 0x18, 0x6C])
SEQ = [0x73, 0x9B, 0x5E, 0xB6, 0x49, 0xA1, 0x02, 0xEA, 0x15, 0xFD]
SYNCBITS = np.array([(SYNC[i // 8] >> (i % 8)) & 1 for i in range(32)])
SEQBITS = {s: np.array([(s >> i) & 1 for i in range(8)]) for s in SEQ}
TAPS = np.arange(-16, 17)


def crc16(d):
    c = 0
    for b in d:
        c ^= b
        for _ in range(8):
            c = (c >> 1) ^ 0xC003 if c & 1 else c >> 1
    return c


def bits2bytes(b):
    return bytes(int(sum(int(x) << i for i, x in enumerate(r))) for r in b.reshape(-1, 8))


def line_ok(by):
    return by[:4] == SYNC and crc16(by[4:26]) == (by[26] | by[27] << 8)


def timing(r):
    best = None
    for off in np.arange(100, 150, 0.25):
        v = np.interp(off + SPB * np.arange(32), np.arange(2048), r)
        c = np.corrcoef(v, SYNCBITS)[0, 1]
        if best is None or c > best[0]:
            best = (c, off)
    return best[1]


def feats(r, off, n=224):
    idx = off + SPB * np.arange(n)
    X = np.stack([np.interp(idx + t, np.arange(2048), r) for t in TAPS], 1)
    return np.hstack([X, np.ones((n, 1))])


def slice_line(r, w):
    """Equalise and slice one line; on a CRC error flip up to one weak bit."""
    X = feats(r, timing(r))
    y = X @ w
    bits = (y > 0).astype(int)
    by = bits2bytes(bits)
    if not line_ok(by):
        for k in np.argsort(np.abs(y[32:]))[:10] + 32:
            b2 = bits.copy(); b2[k] ^= 1
            by2 = bits2bytes(b2)
            if line_ok(by2):
                return X, b2, by2, True
    return X, bits, by, line_ok(by)


def cmd_train(a):
    arr = np.load(a.lines, mmap_mode='r')
    L = [feats(r, timing(r)) for f in range(min(a.frames, arr.shape[0]))
         for r in np.array(arr[f]).astype(float)]
    w = np.linalg.lstsq(np.vstack([X[:32] for X in L]),
                        np.concatenate([SYNCBITS * 2. - 1] * len(L)), rcond=None)[0]
    for it in range(4):
        XS, YS, good = [], [], 0
        for X in L:
            y = X @ w; bits = (y > 0).astype(int)
            s = min(SEQ, key=lambda s: (SEQBITS[s] != bits[32:40]).sum())
            bits[:32] = SYNCBITS; bits[32:40] = SEQBITS[s]
            if line_ok(bits2bytes(bits)):
                good += 1; XS.append(X); YS.append(bits * 2. - 1)
            else:
                XS.append(X[:40]); YS.append(bits[:40] * 2. - 1)
        w = np.linalg.lstsq(np.vstack(XS), np.concatenate(YS), rcond=None)[0]
        print('iteration %d: %d of %d lines pass the CRC' % (it, good, len(L)))
    np.save(a.weights, w)


def cmd_decode(a):
    arr = np.load(a.lines, mmap_mode='r'); w = np.load(a.weights)
    out, good = [], 0
    for f in range(arr.shape[0]):
        fr = np.array(arr[f]).astype(float)
        for ri in range(fr.shape[0]):
            _, _, by, ok = slice_line(fr[ri], w)
            good += ok; out.append((f, ri, bool(ok), by))
    print('%d of %d lines pass sync and CRC' % (good, len(out)))
    pickle.dump(out, open(a.out, 'wb'))


# --- hacktv cipher, PRBS and line order (tables read from the hacktv sources) ---

def c_tables(path, names):
    src = open(path).read()
    t = {}
    for n in names:
        m = re.search(r'\b' + n + r'\s*\[[^\]]*\]\s*=\s*\{(.*?)\};', src, re.S)
        body = re.sub(r'/\*.*?\*/', '', m.group(1), flags=re.S)
        t[n] = [int(x, 0) for x in body.replace('\n', ' ').split(',') if x.strip()]
    return t


class Syster:
    def __init__(self, hacktv):
        t = c_tables(os.path.join(hacktv, 'src/syster-ca.c'), ['LS', 'S', 'C', 'E', 'P', 'kp', 'ip', 'fp'])
        t.update(c_tables(os.path.join(hacktv, 'src/syster.c'), ['_key_table1', '_key_table2']))
        self.t = t

    def _permute(self, inp, p):
        t = list(inp); b = [0] * 8
        for j in range(7, -1, -1):
            for i in range(8):
                if p[0] & 3:
                    b[j] = ((b[j] << 1) | (t[p[i]] & 1)) & 0xFF; t[p[i]] >>= 1
                else:
                    b[p[i]] = ((b[p[i]] >> 1) | (0x80 if t[j] & 1 else 0)) & 0xFF; t[j] >>= 1
        return b

    @staticmethod
    def _expand(e, data):
        res = [0] * 8; mask = 0x1F if e[0] == 0x1F else 0xFF
        for j in range(8):
            for i in range(6, 0, -1):
                res[j] = (res[j] << 1) & 0xFF
                d = e[(7 - j) * 6 + (i - 1)] & mask
                if data[d >> 3] & (1 << (d & 7)):
                    res[j] |= 1
        return res

    def _rotate(self, r, k):
        for _ in range(self.t['LS'][r]):
            for j in range(3):
                k[j] = (k[j] >> 1 | (k[j + 1] & 1) << 7) & 0xFF
                k[j + 4] = (k[j + 4] >> 1 | (k[j + 5] & 1) << 7) & 0xFF
            k[3] = (k[3] >> 1 | ((k[0] >> 3) & 1) << 7) & 0xFF
            k[7] = (k[7] >> 1 | ((k[4] >> 3) & 1) << 7) & 0xFF

    def _des(self, k, cw, m):
        S, P, C, E = self.t['S'], self.t['P'], self.t['C'], self.t['E']
        k = list(k); kr = []
        for i in range(16):
            kr.append(list(k)); self._rotate(i, k)
        for i in range(16):
            ek = self._expand(C, kr[15 - i if m else i]); ecw = self._expand(E, cw)
            r = [0] * 4; j = 31
            for c in range(8):
                x = (ek[c] ^ ecw[c]) & 0x3F
                sb = S[x >> 1 | ((0x20 * (8 - c)) & 0xFF)]
                if x & 1:
                    sb = (sb << 4) & 0xF0
                for _ in range(4):
                    b = P[j] & 3; mm = 1 << ((P[j] >> 4) & 7)
                    r[b] = r[b] & (mm ^ 0xFF) if sb & 0x80 else r[b] | mm
                    sb = (sb << 1) & 0xFF; j -= 1
            for l in range(4):
                r[l] ^= cw[l + 4]; cw[l + 4] = cw[l]; cw[l] = r[l]
        return cw

    def _half(self, half, key, m):
        k = self._permute(key, self.t['kp']); k[0] = (k[4] << 4) & 0xFF
        return bytes(self._permute(self._des(k, self._permute(half, self.t['ip']), m), self.t['fp']))

    def decrypt_ecm(self, ecm, key):
        return self._half(ecm[:8], key, 0) + self._half(ecm[8:], key, 0)

    def encrypt_ecm(self, ecm, key):
        return self._half(ecm[:8], key, 1) + self._half(ecm[8:], key, 1)

    @staticmethod
    def control_word(plain):
        b1 = plain[0:4] + plain[12:16]; cw = [0] * 8
        for i in range(4):
            cw[i] = b1[i + 4] & (0x7F if i == 3 else 0xFF)
            cw[i + 4] = ((b1[i] << 1) & (0x1F if i == 3 else 0xFF)) | (b1[7 if i == 0 else i - 1] >> 7 & 1)
        return sum(c << (8 * i) for i, c in enumerate(cw))

    def field_seeds(self, cw, n=50):
        """(s, r) for n successive fields after a PRBS reset with control word cw."""
        sr1 = cw & ((1 << 31) - 1); sr2 = (cw >> 32) & ((1 << 29) - 1); out = []
        for _ in range(n):
            code = 0
            for _ in range(16):
                sr1 = (sr1 >> 1) ^ (0x7BB88888 if sr1 & 1 else 0)
                sr2 = (sr2 >> 1) ^ (0x17A2C100 if sr2 & 1 else 0)
                a = min((sr2 >> 24) & 0x1F, 30)
                code = (code << 1) | ((sr1 >> a) & 1)
            code >>= 1
            out.append((code & 0x7F, code >> 7))
        return out

    def field_order(self, s, r, table=1):
        """Transmitted position of each of the 287 decoder-delayed field lines (hacktv model)."""
        tab = self.t['_key_table1' if table == 1 else '_key_table2']
        b = [-32 + i for i in range(32)]; order = [0] * 287
        for i in range(287):
            j = tab[(r + (2 * s + 1) * i) & 0xFF] if i <= 254 else i - 255
            order[b[j] + 32] = i; b[j] = i
        return order


def assemble(lines):
    blocks, cur = [], None
    for f, ri, ok, by in lines:
        if not ok or by[4] not in SEQ:
            cur = None; continue
        i = SEQ.index(by[4])
        if i == 0:
            cur = {'frame': f, 'row': ri, 0: by}
        elif cur is not None and i - 1 in cur:
            cur[i] = by
            if i == 9:
                blocks.append(cur); cur = None
        else:
            cur = None
    return blocks


def parity_ok(b):
    for p in (0, 1):
        for x in range(5, 26):
            v = b[p][x]
            for i in range(2, 10):
                if i & 1 == p:
                    v ^= b[i][x]
            if v:
                return False
    return True


def messages(b):
    m2 = b[4][5:26] + b[5][5:26] + b[2][5:26] + b[3][5:26]
    m1 = b[8][5:26] + b[9][5:26] + b[6][5:26] + b[7][5:26]
    return m1, m2


def date_word(d):
    """Inverse of hacktv _get_date: 0x8000 | (y-1990)<<9 | h<<8 | m'<<5 | day."""
    y = 1990 + ((d >> 9) & 0x3F); h = (d >> 8) & 1; m = (d >> 5) & 7; day = d & 31
    return '%04d-%02d-%02d' % (y, m + 6 * h if h else m, day)


def cmd_blocks(a):
    sy = Syster(a.hacktv); key = bytes.fromhex(a.key)
    lines = pickle.load(open(a.lines, 'rb'))
    blocks = assemble(lines)
    rows, halves = [], collections.defaultdict(collections.Counter)
    npar = 0
    for b in blocks:
        if not parity_ok(b):
            continue
        npar += 1
        m1, m2 = messages(b)
        c = ((m1[1] & 7) << 4) | (m1[2] >> 4); ap = m1[2] & 15
        rows.append((b['frame'], m1, m2, c, ap))
        halves[c][m1[3:11]] += 1
    out = open(a.out, 'w')
    out.write('# %d lines, %d pass CRC, %d complete blocks, %d pass both XOR parity lines\n'
              % (len(lines), sum(l[2] for l in lines), len(blocks), npar))
    out.write('# frame flags aud c ap ecm_half | msg2[0:5] | msg1 EMM (72 bytes)\n')
    for f, m1, m2, c, ap in rows:
        out.write('%5d %02x %2d %02x %x %s | %s | %s\n' % (f, m1[0], m1[1] >> 3, c, ap,
                  m1[3:11].hex(), m2[:5].hex(), m1[12:].hex()))
    out.write('\n# ECM pairs, decrypted with key %s\n# c ecm plain date cw\n' % a.key)
    cws = {}
    for c in sorted(halves):
        if c & 1 or c + 1 not in halves:
            continue
        e = halves[c].most_common(1)[0][0] + halves[c + 1].most_common(1)[0][0]
        p = sy.decrypt_ecm(e, key)
        cw = sy.control_word(p); cws[c] = cw
        out.write('%02x %s %s %s %016x\n' % (c, e.hex(), p.hex(), date_word(p[10] | p[11] << 8), cw))
    out.write('\n# Control word changes (ap 0 or f) and the (s, r) of the first fields after the reset\n')
    for f, m1, m2, c, ap in rows:
        if ap in (0, 0xF) and (c & ~1) in cws:
            cw = cws[c & ~1]
            seeds = sy.field_seeds(cw, 6)
            out.write('frame %5d c %02x ap %x cw %016x  s,r: %s\n'
                      % (f, c & ~1, ap, cw, ' '.join('%d,%d' % x for x in seeds)))
    print(open(a.out).read().split('\n')[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('train'); p.add_argument('lines'); p.add_argument('weights')
    p.add_argument('--frames', type=int, default=300); p.set_defaults(fn=cmd_train)
    p = sub.add_parser('decode'); p.add_argument('lines'); p.add_argument('weights'); p.add_argument('out')
    p.set_defaults(fn=cmd_decode)
    p = sub.add_parser('blocks'); p.add_argument('lines'); p.add_argument('out')
    p.add_argument('--hacktv', required=True); p.add_argument('--key', default='0000000000001234')
    p.set_defaults(fn=cmd_blocks)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
