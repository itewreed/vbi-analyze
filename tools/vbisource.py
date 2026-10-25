#!/usr/bin/env python3
"""Read VBI captures in the bttv layout used by all analysis tools.

Every source is delivered as uint8 frames (n, 32, 2048): rows 0-15 = TV lines 7-22,
rows 16-31 = TV lines 320-335, 2048 samples per line at 35.46895 MHz (8 x PAL fsc).

Supported inputs:
  *.flac             bttv raw VBI compressed with flac (decoded with the flac tool)
  *.vbi, *.raw, -    bttv raw VBI (65,536 bytes per frame); - reads stdin
  *.tbc, *.tbcy      vhs-decode / ld-decode TBC (16 bit, 4 fsc, PAL 1135 x 313 per field);
                     lines 7-22 of each first field and 320-335 of the following field are
                     resampled to the bttv grid (2 bttv samples per TBC sample) and the levels
                     mapped from the .tbc.json (blanking -> 58, white -> 168)
  *.tbcc, *_chroma.tbc  TBC chroma; read the same way, with a warning: these files hold
                     only the chroma signal, the VBI data is in the luma (.tbcy / .tbc)

  vbisource.py info    FILE                         # format, frames, levels
  vbisource.py raw     FILE > out.vbi               # write bttv raw VBI to stdout
  vbisource.py extract FILE out.npy --rows 7,8,23,24 [--frames 0:12000]
"""
import argparse, json, os, subprocess, sys
import numpy as np

ROWS, SPL = 32, 2048
FB = ROWS * SPL
TBC_X0 = 128.0          # TBC sample of bttv sample 0: 7.17 us after 0H, measured from the sync edges
TBC_ROW0 = 6            # TBC row of TV line 7 (field 1) and of line 320 (field 2)
BLANK8, WHITE8 = 58.0, 168.0   # typical bttv levels of blanking and white
TBC_EXT = ('.tbc', '.tbcy', '.tbcc')


def warn(msg):
    print('warning: ' + msg, file=sys.stderr)


def kind(path):
    if path == '-':
        return 'raw'
    p = path.lower()
    if p.endswith('.flac'):
        return 'flac'
    if p.endswith('.tbcc') or p.endswith('_chroma.tbc'):
        return 'tbcc'
    if p.endswith('.tbc') or p.endswith('.tbcy'):
        return 'tbc'
    return 'raw'


def tbc_json(path):
    base = path.rsplit('.', 1)[0]
    for b in (base, base[:-len('_chroma')] if base.endswith('_chroma') else None):
        if b and os.path.exists(b + '.tbc.json'):
            return b + '.tbc.json'
    return None


class TbcSource:
    def __init__(self, path):
        if kind(path) == 'tbcc':
            warn('%s is a TBC chroma file: it holds only the chroma signal, so VBI data '
                          '(teletext, VPS, data line) cannot be decoded from it. Use the matching '
                          '.tbcy or .tbc file.' % path)
        meta = tbc_json(path)
        self.json = json.load(open(meta)) if meta else None
        vp = self.json['videoParameters'] if self.json else {}
        if vp.get('system', 'PAL') != 'PAL':
            raise ValueError('%s: only PAL TBC files are supported (system %s)' % (path, vp['system']))
        self.w, self.h = int(vp.get('fieldWidth', 1135)), int(vp.get('fieldHeight', 313))
        blank = vp.get('blanking16bIre', vp.get('black16bIre', 16384.0))
        white = vp.get('white16bIre', 59417.6)
        self.gain = (WHITE8 - BLANK8) / (white - blank); self.blank = blank
        if not meta:
            warn('%s: no .tbc.json found, assuming PAL 1135 x 313, first field first' % path)
        self.a = np.memmap(path, np.uint16, 'r')
        fs = self.w * self.h
        if len(self.a) % fs:
            warn('%s: size is not a whole number of %d x %d fields; the last %d samples are ignored'
                          % (path, self.w, self.h, len(self.a) % fs))
        self.nf = len(self.a) // fs
        self.a = self.a[:self.nf * fs].reshape(self.nf, self.h, self.w)
        self.pairs = self._pairs()
        q = TBC_X0 + np.arange(SPL) / 2.0
        self.i0 = np.clip(np.floor(q).astype(int), 0, self.w - 1)
        self.i1 = np.clip(self.i0 + 1, 0, self.w - 1)
        self.fr = (q - np.floor(q)).astype(np.float32)
        self.pad = q > self.w - 1            # beyond the TBC line: next line's sync

    def _pairs(self):
        """Indices of (first field, second field) for each frame."""
        if not self.json:
            return [(i, i + 1) for i in range(0, self.nf - 1, 2)]
        ff = [f.get('isFirstField', True) for f in self.json['fields'][:self.nf]]
        out, i = [], 0
        while i + 1 < len(ff):
            if ff[i] and not ff[i + 1]:
                out.append((i, i + 1)); i += 2
            else:
                i += 1
        return out

    def __len__(self):
        return len(self.pairs)

    def frames(self, f0, f1, rows=None):
        rows = list(range(ROWS)) if rows is None else list(rows)
        f1 = min(f1, len(self))
        out = np.empty((max(f1 - f0, 0), len(rows), SPL), np.uint8)
        for k, f in enumerate(range(f0, f1)):
            a, b = self.pairs[f]
            src = np.stack([self.a[a if r < 16 else b, TBC_ROW0 + r % 16] for r in rows]).astype(np.float32)
            x = src[:, self.i0] * (1 - self.fr) + src[:, self.i1] * self.fr
            x = BLANK8 + (x - self.blank) * self.gain
            x[:, self.pad] = 12
            out[k] = np.clip(np.rint(x), 0, 255)
        return out


class RawSource:
    """bttv raw VBI from a file, stdin or a flac pipe (sequential reading)."""
    def __init__(self, path):
        self.proc = None
        if path == '-':
            self.f = sys.stdin.buffer
        elif kind(path) == 'flac':
            self.proc = subprocess.Popen(['flac', '-s', '-c', '-d', '--force-raw-format', '--sign=unsigned',
                                          '--endian=little', path], stdout=subprocess.PIPE)
            self.f = self.proc.stdout
        else:
            self.f = open(path, 'rb')
        self.pos = 0

    def read(self, n):
        buf = self.f.read(FB * n)
        k = len(buf) // FB
        self.pos += k
        return np.frombuffer(buf[:k * FB], np.uint8).reshape(k, ROWS, SPL)


def open_source(path):
    return TbcSource(path) if kind(path) in ('tbc', 'tbcc') else RawSource(path)


def iter_frames(path, chunk=64, f0=0, f1=None, rows=None):
    """Yield uint8 blocks (n, rows, 2048) of consecutive frames from any supported source."""
    s = open_source(path)
    if isinstance(s, TbcSource):
        end = len(s) if f1 is None else min(f1, len(s))
        for i in range(f0, end, chunk):
            yield s.frames(i, min(i + chunk, end), rows)
        return
    while True:
        if f1 is not None and s.pos >= f1:
            break
        n = chunk if f1 is None else min(chunk, f1 - s.pos)
        a = s.read(n)
        if not len(a):
            break
        lo = max(0, f0 - (s.pos - len(a)))
        if lo < len(a):
            a = a[lo:]
            yield a if rows is None else a[:, list(rows)]
    if s.proc:
        s.proc.stdout.close(); s.proc.wait()


def load_rows(path, rows, frames=None):
    """(frames, len(rows), 2048) uint8 array; a .npy file is returned as stored."""
    if path.lower().endswith('.npy'):
        return np.load(path, mmap_mode='r')
    f0, f1 = frames if frames else (0, None)
    parts = list(iter_frames(path, 256, f0, f1, rows))
    return np.concatenate(parts) if parts else np.zeros((0, len(rows), SPL), np.uint8)


def frame_range(s):
    if not s:
        return None
    a, b = s.split(':')
    return int(a or 0), int(b) if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('info'); p.add_argument('file')
    p = sub.add_parser('raw'); p.add_argument('file'); p.add_argument('--frames')
    p = sub.add_parser('extract'); p.add_argument('file'); p.add_argument('out')
    p.add_argument('--rows', required=True); p.add_argument('--frames')
    a = ap.parse_args()
    if a.cmd == 'info':
        s = open_source(a.file)
        if isinstance(s, TbcSource):
            print('%s: TBC (%s), %d fields of %d x %d, %d frames, json %s' % (
                a.file, 'chroma' if kind(a.file) == 'tbcc' else 'luma/composite', s.nf, s.w, s.h,
                len(s), 'yes' if s.json else 'no'))
        else:
            n = sum(len(b) for b in iter_frames(a.file, 256))
            print('%s: bttv raw VBI (%s), %d frames' % (a.file, kind(a.file), n))
    elif a.cmd == 'raw':
        fr = frame_range(a.frames) or (0, None)
        for b in iter_frames(a.file, 64, *fr):
            sys.stdout.buffer.write(np.ascontiguousarray(b).tobytes())
    else:
        rows = [int(r) for r in a.rows.split(',')]
        x = load_rows(a.file, rows, frame_range(a.frames))
        np.save(a.out, x)
        print('%s: %s' % (a.out, x.shape))


if __name__ == '__main__':
    main()
