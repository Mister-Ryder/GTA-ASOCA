#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/ASOCA_GTAirway_20261013
PY=$ROOT/env/bin/python
cd "$ROOT/code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$PY" losses.py
"$PY" model.py --device cuda --size 32 --batch-size 2 --equivalence-only
"$PY" model.py --device cuda --size 48 --batch-size 2
for METHOD in gta unet unet_cldice swinunetr; do
    "$PY" train.py --manifest "$ROOT/pilot_cache/manifest.json" --output "$ROOT/results/smoke96_pinned/$METHOD" --method "$METHOD" --epochs 1 --steps 3 --patch-size 96 --batch-size 2 --workers 0 --smoke
done
date -u +%FT%TZ > "$ROOT/manifests/smoke_passed.txt"
