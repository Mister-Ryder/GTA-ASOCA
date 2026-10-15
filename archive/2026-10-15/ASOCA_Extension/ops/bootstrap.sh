#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/ASOCA_GTAirway_20261013
ENV=$ROOT/env
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/manifests"
if [ ! -x "$ENV/bin/python" ]; then
    /root/miniconda3/bin/conda create -p "$ENV" python=3.11 pip -y
fi
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
printf 'torch==2.7.1+cu128\ntorchvision==0.22.1+cu128\nnumpy==1.26.4\n' > "$ROOT/manifests/core_constraints.txt"
"$ENV/bin/python" -m pip install -c "$ROOT/manifests/core_constraints.txt" numpy==1.26.4 scipy==1.14.1 SimpleITK==2.5.2 scikit-image==0.25.2 monai==1.5.1 nnunetv2==2.6.4 einops==0.8.1
"$ENV/bin/python" -m pip freeze > "$ROOT/manifests/environment.lock.txt"
"$ENV/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.randn(2,1,32,32,32,device="cuda"); m=torch.nn.Conv3d(1,4,3,padding=1).cuda(); m(x).sum().backward(); print("CUDA backward OK")'
date -u +%FT%TZ > "$ROOT/manifests/environment_ready.txt"
