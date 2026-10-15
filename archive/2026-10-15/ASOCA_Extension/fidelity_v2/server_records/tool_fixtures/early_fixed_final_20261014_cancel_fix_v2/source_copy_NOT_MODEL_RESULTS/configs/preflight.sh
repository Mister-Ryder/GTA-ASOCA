#!/usr/bin/env bash
# Source this file only from one saved single-run recipe. Nothing launches here.
set -euo pipefail
GTA_ROOT=${GTA_ROOT:-/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014}
GTA_DATA=${GTA_DATA:-/root/autodl-fs/datasets/ASOCA_GTA_20261014}
GTA_MANIFEST=${GTA_MANIFEST:-$GTA_ROOT/dataset/cache/manifest.json}
GTA_RAW_ROOT=${GTA_RAW_ROOT:-$GTA_DATA/raw}
GTA_PYTHON=${GTA_PYTHON:-/root/miniconda3/envs/gta-asoca311/bin/python}
GTA_OUTPUT_ROOT=${OUTPUT_ROOT:-${GTA_OUTPUT_ROOT:-/root/autodl-fs/experiments/GTA_ASOCA_Fidelity_20261014/results}}
"$GTA_PYTHON" "$GTA_ROOT/configs/verify_main_source.py" --root "$GTA_ROOT"
test -f "$GTA_MANIFEST"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU occupied; run this recipe when the current job finishes.' >&2
  exit 2
fi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
GTA_RESUME_ARGS=()
if [[ "${GTA_RESUME:-0}" == 1 ]]; then GTA_RESUME_ARGS+=(--resume); fi
