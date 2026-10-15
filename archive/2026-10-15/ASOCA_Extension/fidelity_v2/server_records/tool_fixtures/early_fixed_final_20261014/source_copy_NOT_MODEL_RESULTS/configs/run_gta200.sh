#!/usr/bin/env bash
# One fold, 25,000 segmentation updates, plus 5,120 classifier updates.
# Classifier fixed-final readiness is checked; accuracy is diagnostic only.
source "$(dirname "${BASH_SOURCE[0]}")/preflight.sh"
GTA_OUTPUT=${GTA_OUTPUT:-$GTA_OUTPUT_ROOT/gta_enhanced200x125/training}
GTA_CLASSIFIER=${CLS_CHECKPOINT:-${GTA_CLASSIFIER:-$GTA_ROOT/results/classifier80x64/training/final.pt}}
GTA_CLASSIFIER_TRAINING=${GTA_CLASSIFIER_TRAINING:-$(dirname "$GTA_CLASSIFIER")}
"$GTA_PYTHON" "$GTA_ROOT/configs/verify_classifier_quality.py" \
  --training "$GTA_CLASSIFIER_TRAINING" --checkpoint "$GTA_CLASSIFIER" --manifest "$GTA_MANIFEST"
"$GTA_PYTHON" -u "$GTA_ROOT/code/train.py" \
  --manifest "$GTA_MANIFEST" --output "$GTA_OUTPUT" --method gta --fold 0 \
  --epochs 200 --steps 125 --patch-size 128 --batch-size 2 --workers 4 \
  --seed 20261013 --split-seed 20261014 --cohort internal28 \
  --lr 0.0003 --weight-decay 0.0001 --warmup-epochs 5 --backbone-warmup-epochs 0 \
  --freeze-epochs 50 --partial-until 70 --classifier-lr-scale 0.1 \
  --seg-loss dice_ce --sampling standard --sampling-fov image-body \
  --classifier-weight 0.1 --guidance-weight 0.1 --gaussian-weight 10 \
  --classifier-init "$GTA_CLASSIFIER" --classifier-domain asoca --classifier-head-policy load \
  --backbone-init "$GTA_ROOT/initialization/airway_backbone_epoch599.pkl" --backbone-domain airway \
  --save-every 10 --gradient-probe-every 250 --device cuda "${GTA_RESUME_ARGS[@]}"
