#!/bin/bash
# usage: run_all.sh <flac-dir> <out-dir> [parallel]
# Decompresses each *.flac on the fly and surveys the VBI content.
IN="$1"; OUT="$2"; P="${3:-6}"; T="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
ls "$IN"/*.flac | xargs -P "$P" -I{} bash -c '
  f="{}"; b=$(basename "$f" _vbi.flac); o="'"$OUT"'/$b"
  [ -f "$o/report.txt" ] && exit 0
  mkdir -p "$o"
  flac -s -c -d --force-raw-format --sign=unsigned --endian=little "$f" | python3 "'"$T"'/survey.py" - "$o" > "$o/log.txt" 2>&1 \
   && python3 "'"$T"'/analyze.py" "$o" > /dev/null 2>> "$o/log.txt"
  echo "$(date +%T) done: $b"'
