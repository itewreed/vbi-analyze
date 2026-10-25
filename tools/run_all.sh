#!/bin/bash
# usage: run_all.sh <capture-dir> <out-dir> [parallel]
# Surveys the VBI content of every capture in <capture-dir>: *.flac (bttv raw VBI,
# decompressed on the fly) and vhs-decode / ld-decode TBC files (*.tbc, *.tbcy, *.tbcc).
# A chroma file (*.tbcc, *_chroma.tbc) is skipped when its luma file is present; on its
# own it is surveyed with a warning, since it holds no VBI data.
IN="$1"; OUT="$2"; P="${3:-6}"; T="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
shopt -s nullglob
for f in "$IN"/*.flac "$IN"/*.tbc "$IN"/*.tbcy "$IN"/*.tbcc; do
  case "$f" in
    *.tbcc)       l="${f%.tbcc}"; [ -f "$l.tbcy" ] || [ -f "$l.tbc" ] && { echo "skipping chroma file $f (luma file present)" >&2; continue; } ;;
    *_chroma.tbc) [ -f "${f%_chroma.tbc}.tbc" ] && { echo "skipping chroma file $f (luma file present)" >&2; continue; } ;;
  esac
  printf '%s\0' "$f"
done | xargs -0 -P "$P" -I{} bash -c '
  f="{}"; b=$(basename "$f"); b=${b%_vbi.flac}; b=${b%.flac}; b=${b%.tbcy}; b=${b%.tbcc}; b=${b%.tbc}
  o="'"$OUT"'/$b"
  [ -f "$o/report.txt" ] && exit 0
  mkdir -p "$o"
  python3 "'"$T"'/survey.py" "$f" "$o" > "$o/log.txt" 2>&1 \
   && python3 "'"$T"'/analyze.py" "$o" > /dev/null 2>> "$o/log.txt"
  echo "$(date +%T) done: $b"'
