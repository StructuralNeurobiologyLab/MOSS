#!/usr/bin/env bash
# Convert a screen recording (.mov/.mp4) into a compact, repo-friendly GIF.
#
# Usage:
#   scripts/mov_to_gif.sh recording.mov media/01_load_and_convert.gif [fps] [width]
#
# Defaults: 12 fps, 900 px wide. Uses a two-pass palette for good quality at
# small size. Aim to keep each GIF well under ~8 MB so clones stay light.
set -euo pipefail

SRC="${1:?source .mov/.mp4 required}"
OUT="${2:?output .gif path required}"
FPS="${3:-12}"
WIDTH="${4:-900}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found (try: conda install -c conda-forge ffmpeg)"; exit 1; }
mkdir -p "$(dirname "$OUT")"

PALETTE="$(mktemp -t palette).png"
FILTERS="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"

ffmpeg -y -i "$SRC" -vf "${FILTERS},palettegen=stats_mode=diff" "$PALETTE"
ffmpeg -y -i "$SRC" -i "$PALETTE" \
  -lavfi "${FILTERS} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3" "$OUT"
rm -f "$PALETTE"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
