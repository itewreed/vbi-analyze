"""WST demodulator for band-limited VHS recordings: lowpass, correlation sync, numba Viterbi."""
import numpy as np, numba
FS = 35.46895e6; SPB = FS / 6.9375e6; NB = 360
PREFIX = np.array([(b >> i) & 1 for b in (0x55, 0x55, 0x27) for i in range(8)], np.int64)
_n = np.arange(-24, 25); LP = np.sinc(2 * 3.5e6 / FS * _n) * np.hamming(len(_n)); LP /= LP.sum()
def _tm():
    bits = PREFIX.astype(float) - .5
    m = int(24 * SPB) + 1; t = np.arange(m) / SPB
    w = bits[np.clip(t.astype(int), 0, 23)]
    k = np.arange(-8, 9); ker = np.exp(-.5 * (k / (0.8 * SPB)) ** 2); w = np.convolve(w, ker / ker.sum(), 'same')
    return w - w.mean()
TM = _tm()
DEFAULT_TAPS = np.array([0.0, 4.2, 0.5, 23.4, 52.2, 23.4, -3.5, -1.2, 0.0]); DEFAULT_OFF = -5.0

def ham_tables():
    t = np.full(256, -1, np.int64); e = np.full(256, 2, np.int64)
    for d in range(16):
        d1, d2, d3, d4 = [(d >> i) & 1 for i in range(4)]
        p1 = 1 ^ d1 ^ d3 ^ d4; p2 = 1 ^ d1 ^ d2 ^ d4; p3 = 1 ^ d1 ^ d2 ^ d3
        b = p1 | d1 << 1 | p2 << 2 | d2 << 3 | p3 << 4 | d3 << 5 | d4 << 7
        p4 = 1 ^ (bin(b).count('1') & 1); b |= p4 << 6
        t[b] = d; e[b] = 0
    for b in range(256):
        if e[b] == 0: continue
        for i in range(8):
            c = b ^ (1 << i)
            if e[c] == 0 and t[c] >= 0:
                t[b] = t[c]; e[b] = 1; break
    return t, e
HAM, HERR = ham_tables()

@numba.njit(cache=True)
def viterbi(y, taps, off, prefix):
    L = taps.shape[0]; M = L // 2; S = 1 << (L - 1); n = y.shape[0]
    INF = 1e18
    cost = np.full(S, INF); cost[0] = 0.0
    back = np.zeros((n + M, S), np.int32)
    pred0 = np.empty(S); pred1 = np.empty(S)
    for s in range(S):
        acc = off
        for j in range(L - 1):
            if (s >> j) & 1: acc += taps[j]
        pred0[s] = acc; pred1[s] = acc + taps[L - 1]
    for t in range(n + M):
        k = t - M
        new = np.full(S, INF * 10); ptr = np.zeros(S, np.int32)
        for s in range(S):
            c = cost[s]
            if c >= INF: continue
            for bit in range(2):
                if t < prefix.shape[0] and prefix[t] != bit: continue
                cc = c
                if k >= 0:
                    p = pred1[s] if bit else pred0[s]
                    cc += (y[k] - p) ** 2
                ns = (s >> 1) | (bit << (L - 2))
                if cc < new[ns]:
                    new[ns] = cc; ptr[ns] = s
        cost = new; back[t] = ptr
    s = 0; best = INF * 100
    for i in range(S):
        if cost[i] < best: best = cost[i]; s = i
    out = np.zeros(n + M, np.int64)
    for t in range(n + M - 1, -1, -1):
        out[t] = (s >> (L - 2)) & 1
        s = back[t, s]
    return out[:n], best

def prepare(line):
    x = np.convolve(line.astype(float), LP, 'same')
    black = np.median(x[20:100]); top = np.percentile(x[150:1950], 99)
    if top - black < 25: return None
    x = (x - black) * (100.0 / (top - black))
    sm = np.convolve(x, np.ones(9) / 9, 'same')
    above = np.nonzero(sm[40:700] > 30)[0]
    if not len(above): return None
    on = 40 + above[0]
    lo = max(0, on - 40)
    seg = x[lo:on + 40 + len(TM)]
    c = np.correlate(seg - seg.mean(), TM, 'valid')
    st = lo + int(np.argmax(c)) + 0.5 * SPB
    # bit phase from crossings
    ts = []
    for thr in (25.0, 75.0):
        ab = x > thr; idx = np.nonzero(ab[1:] != ab[:-1])[0]
        idx = idx[(idx > st + 80) & (idx < st + 1830)]
        if len(idx): ts.append(idx + (thr - x[idx]) / (x[idx + 1] - x[idx]))
    if ts:
        t = np.concatenate(ts); z = np.exp(2j * np.pi * t / SPB).sum()
        b0 = (np.angle(z) / (2 * np.pi)) % 1 * SPB
        st = b0 + np.round((st - b0) / SPB) * SPB
    return x, st

def sample(x, st, shift=0.0):
    t = st + (np.arange(NB) + shift) * SPB
    if t[-1] >= len(x) - 1: return None
    return np.interp(t, np.arange(len(x)), x)

def bits_to_bytes(bits):
    return np.packbits(bits.reshape(-1, 8)[:, ::-1].astype(np.uint8), axis=1).ravel()

def decode(line, taps=DEFAULT_TAPS, off=DEFAULT_OFF, shifts=(0.0, -1.0, 1.0)):
    """Returns (bytes45, mrag_err, rms residual, y, bits) or None."""
    pr = prepare(line)
    if pr is None: return None
    x, st = pr; best = None
    for sh in shifts:
        y = sample(x, st, sh)
        if y is None: continue
        bits, cost = viterbi(y, taps, off, PREFIX)
        b = bits_to_bytes(bits)
        me = HERR[b[3]] + HERR[b[4]]
        rms = np.sqrt(cost / NB)
        cand = (b, me, rms, y, bits)
        if me == 0 and rms < 15: return cand
        if best is None or (me, rms) < (best[1], best[2]): best = cand
    return best

def design(bits, M=4):
    p = np.concatenate([np.zeros(M, int), bits, np.zeros(M, int)])
    X = np.lib.stride_tricks.sliding_window_view(p, 2 * M + 1)[:len(bits)]
    return np.concatenate([X, np.ones((len(bits), 1))], 1)

def fit(ys, bitss, M=4):
    X = np.concatenate([design(b, M) for b in bitss]); Y = np.concatenate(ys)
    sol = np.linalg.lstsq(X, Y, rcond=None)[0]
    return sol[:-1], sol[-1]
