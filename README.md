# vbi-analyze

Analysis of the vertical blanking interval (VBI) of German TV broadcasts from 1979 to 1998, recovered from VHS tapes.

Besides ordinary teletext, the VBI of that era carried data that viewers never saw:

- the inter-broadcaster **data line** on lines 16 and 329, which includes VPS;
- **data channels hidden inside teletext**;
- early, pre-standard teletext packets;
- several **measurement signals**.

This repository documents what was found in 376 captures (a first batch of 110 and a second batch, "collection 04", of 266) and contains the tools used to find it.

**→ Read the findings: [docs/FINDINGS.md](docs/FINDINGS.md)**

## Highlights

- **Line 16 and line 329 are a full data line, not just VPS.** Every broadcaster sent its name in plain ASCII (`WDR-KLN`, `NDR HMBG`, `SFB-SCHO`, `RIAS-TV`, `PK-BLN-2` …) plus a 4-bit source code. On line 329 the Deutsche Bundespost identified its own network section (`DBP-FTZ`, `DBP-1`).
- **A VPS test cycle (1983–85):** before VPS officially launched, ARD/ZDF looped a 30-minute sequence of fake programme labels ("11 May, 10:xx") and special codes.
- **A second, unidentified 1 Mbit/s biphase signal** on lines 16/329. It appears on an early-1980s West German recording and on East German DDR1/DFF in 1989/90.
- **Data channels in teletext:** the GIS financial feed on ProSieben 1990 (see [gis-financefeed-vbi-decoder](https://github.com/itewreed/gis-financefeed-vbi-decoder)). Apparent second channels (M5/31) on ProSieben and RTL turned out to be bit-slip artefacts of the demodulator.
- **Early packet 8/30** with date and time in plain text. Together with teletext headers it helps date undated tapes.
- **Static test signals** on lines 323 and 328 that flip polarity every frame; line 328 already on ZDF in 1979.
- **The data line predates VPS:** it first appears in March 1983 with plain-text sources such as `HR1 FFTM`, `BR-MCHN1` and `SDR-TVS3`.
- **Pre-standard teletext from the 1980 Videotext trial:** packet 8/30 with date and time from June 1980, a recurring M3 X/29 packet and a data channel (M7/31) in 1981.
- **1990s data channels:** a common M8/31 channel on ARD and ZDF 1996–98, and an RTL data channel on line 329 in 1998.

## Capture format

| Property | Value |
|---|---|
| Hardware | bt848/bt878 card, Linux `bttv` raw VBI |
| Sampling | 2048 samples per line, unsigned 8 bit, 35.46895 MHz (8 × PAL fsc) |
| Frame | 32 rows (65,536 bytes): rows 0–15 = TV lines 7–22, rows 16–31 = TV lines 320–335 |
| Storage | FLAC; decode with `flac -d --force-raw-format --sign=unsigned --endian=little file.flac` |

## Tools

Python 3 with `numpy` and `numba` (plus `matplotlib` for plots). Install them with `pip install -r requirements.txt`.

| Script | Purpose |
|---|---|
| [tools/run_all.sh](tools/run_all.sh) | `run_all.sh <flac-dir> <out-dir> [jobs]`: decodes every FLAC capture on the fly and runs `survey.py` and `analyze.py` per tape |
| [tools/survey.py](tools/survey.py) | `survey.py <file.vbi \| -> <outdir>`: classifies every active line as teletext, VPS/data line, 1 Mbit/s biphase, static or unknown, and writes `packets.bin`, `vps.txt`, `l16.txt`, `summary.json` and `extra.npz` |
| [tools/analyze.py](tools/analyze.py) | `analyze.py <outdir>`: writes `report.txt` (row usage, magazines, headers, special packets) and majority-voted teletext pages |
| [tools/cross.py](tools/cross.py) | `cross.py <survey-root>`: one-paragraph summary per tape (see [docs/tape-overview.txt](docs/tape-overview.txt)) |
| [tools/dz.py](tools/dz.py) | `dz.py <survey-root>`: decodes all 15 words of data lines 16/329 per tape (see [docs/data-line-words.txt](docs/data-line-words.txt)) |
| [tools/p830.py](tools/p830.py) | `p830.py <survey-root>`: majority-vote text of the early packet 8/30 |
| [tools/slipcheck.py](tools/slipcheck.py) | `slipcheck.py <survey-dir> <mag> <packet> [n]`: re-frames packets by ±1–2 bits to expose bit-slip artefacts |
| [tools/special.py](tools/special.py) | `special.py <survey-dir> <mag> <packet> [n]`: dumps teletext packets of one type |
| [tools/plotunk.py](tools/plotunk.py) | plots sample waveforms of unclassified rows |

Library modules:

- `wst2.py`: a WST demodulator for band-limited VHS recordings. It applies a lowpass filter, finds the packet start from the signal onset and runs a 256-state Viterbi detector with a channel model refitted per file.
- `wstslice.py`: a fast threshold slicer.
- `vps.py`: decoder for the 2.5 Mbit/s biphase data line.
- `l16bits.py`: decoder for the 1 Mbit/s biphase signal.

### Output record format (`packets.bin`)

Each record is 51 bytes: `u32 frame (LE)`, `u8 row`, `u8 method`, then 45 bytes of the teletext packet (run-in `55 55`, framing code `27`, two MRAG bytes, 40 data bytes).

## Status

This is an ongoing analysis. Open questions are listed at the end of [docs/FINDINGS.md](docs/FINDINGS.md#10-open-questions). Information about the original ARD/ZDF/IRT data line specification is very welcome.
