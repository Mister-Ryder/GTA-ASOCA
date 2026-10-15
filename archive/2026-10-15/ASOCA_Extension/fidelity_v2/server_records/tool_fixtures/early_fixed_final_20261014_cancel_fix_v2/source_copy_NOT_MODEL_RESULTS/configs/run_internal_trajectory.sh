#!/usr/bin/env bash
# Run separately after the selected training has finished; one GPU process only.
source "$(dirname "${BASH_SOURCE[0]}")/preflight.sh"
: "${GTA_TRAINING:?Set GTA_TRAINING to the completed gta or unet training folder}"
GTA_TRAJECTORY=${GTA_TRAJECTORY:-$(dirname "$GTA_TRAINING")/internal4_trajectory}
"$GTA_PYTHON" -u "$GTA_ROOT/code/evaluate_trajectory.py" \
  --training "$GTA_TRAINING" --manifest "$GTA_MANIFEST" --raw-root "$GTA_RAW_ROOT" \
  --output "$GTA_TRAJECTORY" --sw-batch-size 1
"$GTA_PYTHON" "$GTA_ROOT/ops/plot_learning_curves.py" \
  --run "development=$GTA_TRAINING" --validation "development=$GTA_TRAJECTORY/validation_history.jsonl" \
  --output "$(dirname "$GTA_TRAINING")/learning_curves"
