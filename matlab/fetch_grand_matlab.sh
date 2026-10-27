#!/usr/bin/env bash
# Clone Ken Duffy's GRAND-MATLAB reference code next to this script (gitignored).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/GRAND-MATLAB"
if [ -d "$DEST/.git" ]; then
  echo "already present: $DEST"; git -C "$DEST" pull --ff-only || true
else
  git clone --depth 1 https://github.com/kenrduffy/GRAND-MATLAB "$DEST"
fi
echo "files:"; find "$DEST" -name "bin_ORBGRAND*.m" -o -name "landslide.m" | sed "s|$DEST/||"
echo "license: GRAND Codebase Non-Commercial Academic Research Use License (see repo); cite Duffy ICASSP 2021, IEEE TSP 2022"
