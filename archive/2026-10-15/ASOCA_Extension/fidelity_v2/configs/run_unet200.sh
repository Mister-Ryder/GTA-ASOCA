#!/usr/bin/env bash
# Independent single-fold development calibration; same segmentation recipe.
source "$(dirname "${BASH_SOURCE[0]}")/preflight.sh"
GTA_OUTPUT=${GTA_OUTPUT:-$GTA_OUTPUT_ROOT/unet_recipe200x125/training}
"$GTA_PYTHON" -u "$GTA_ROOT/code/train.py" \
  --manifest "$GTA_MANIFEST" --output "$GTA_OUTPUT" --method unet --fold 0 \
  --epochs 200 --steps 125 --patch-size 128 --batch-size 2 --workers 4 \
  --seed 20261013 --split-seed 20261014 --cohort internal28 \
  --lr 0.0003 --weight-decay 0.0001 --warmup-epochs 5 --backbone-warmup-epochs 0 \
  --seg-loss dice_ce --sampling standard --sampling-fov image-body \
  --save-every 10 --gradient-probe-every 250 --device cuda "${GTA_RESUME_ARGS[@]}"
