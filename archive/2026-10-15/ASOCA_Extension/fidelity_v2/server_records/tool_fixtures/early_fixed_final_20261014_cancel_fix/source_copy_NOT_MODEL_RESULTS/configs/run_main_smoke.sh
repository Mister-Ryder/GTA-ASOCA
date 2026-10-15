#!/usr/bin/env bash
# Six updates, three real-gradient stages. These weights cannot select a model.
source "$(dirname "${BASH_SOURCE[0]}")/preflight.sh"
GTA_OUTPUT=${GTA_OUTPUT:-$GTA_OUTPUT_ROOT/gta_mechanical_smoke128/training}
GTA_CLASSIFIER=${CLS_CHECKPOINT:-${GTA_CLASSIFIER:-$GTA_ROOT/results/classifier_smoke128/training/final.pt}}
test -f "$GTA_CLASSIFIER"
"$GTA_PYTHON" -u "$GTA_ROOT/code/train.py" \
  --manifest "$GTA_MANIFEST" --output "$GTA_OUTPUT" --method gta --fold 0 \
  --epochs 3 --steps 2 --patch-size 128 --batch-size 2 --workers 2 \
  --seed 20261013 --split-seed 20261014 --cohort internal28 \
  --lr 0.0003 --weight-decay 0.0001 --warmup-epochs 0 --backbone-warmup-epochs 0 \
  --freeze-epochs 1 --partial-until 2 --classifier-lr-scale 0.1 \
  --seg-loss dice_ce --sampling standard --sampling-fov image-body \
  --classifier-weight 0.1 --guidance-weight 0.1 --gaussian-weight 10 \
  --classifier-init "$GTA_CLASSIFIER" --classifier-domain asoca --classifier-head-policy load \
  --backbone-init "$GTA_ROOT/initialization/airway_backbone_epoch599.pkl" --backbone-domain airway \
  --save-every 1 --gradient-probe-every 1 --device cuda --smoke "${GTA_RESUME_ARGS[@]}"
