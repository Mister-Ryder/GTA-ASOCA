#!/usr/bin/env bash
set -euo pipefail
GTA_OLD_ROOT=${GTA_OLD_ROOT:-/root/autodl-tmp/ASOCA_GTAirway_20261013}
GTA_FIDELITY_ROOT=${GTA_FIDELITY_ROOT:-$GTA_OLD_ROOT/fidelity_v2}
GTA_PYTHON=${GTA_PYTHON:-$GTA_OLD_ROOT/env/bin/python}
"$GTA_PYTHON" -u "$GTA_FIDELITY_ROOT/ops/start_unet_calibration.py" \
  --root "$GTA_FIDELITY_ROOT" --python "$GTA_PYTHON" \
  --manifest "$GTA_OLD_ROOT/cache/manifest.json" --raw-root "$GTA_OLD_ROOT/raw/CompressedVersion" \
  --output-root "${OUTPUT_ROOT:-$GTA_FIDELITY_ROOT/results}" "$@"
