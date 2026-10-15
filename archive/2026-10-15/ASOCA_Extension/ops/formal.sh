#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/ASOCA_GTAirway_20261013
if test -f "$ROOT/development/scope_change.json"; then
  echo "Fivefold launcher is deferred by the current method-development scope. Use development/coordinator.py."
  exit 2
fi
PY=$ROOT/env/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$ROOT/code"
test -f "$ROOT/manifests/environment_ready.txt"
test -f "$ROOT/manifests/smoke_passed.txt"
test -f "$ROOT/raw/CompressedVersion.zip"
"$PY" - <<'PY'
from pathlib import Path
import zipfile
root=Path('/root/autodl-tmp/ASOCA_GTAirway_20261013')
archive=root/'raw/CompressedVersion.zip'
target=root/'raw/CompressedVersion'
target.mkdir(exist_ok=True)
marker=root/'manifests/raw_extracted.txt'
if not marker.exists():
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            destination=(target/name).resolve()
            if target.resolve() not in destination.parents and destination!=target.resolve():
                raise ValueError('Unsafe ZIP member')
        z.extractall(target)
    marker.write_text('181-member ASOCA original archive extracted\n')
PY
"$PY" prepare_data.py --raw-root "$ROOT/raw/CompressedVersion" --output "$ROOT/cache" --workers 2
"$PY" run_queue.py --root "$ROOT" --raw-root "$ROOT/raw/CompressedVersion" --manifest "$ROOT/cache/manifest.json"
