#!/usr/bin/env bash
# Creates the `prima` conda env and downloads PRIMA + its weights (~6.4 GB).
# Differences from upstream's install, all deliberate:
#   * torch 2.7.1+cu128 instead of 2.2.1: RTX PRO Blackwell GPUs (sm_120) need CUDA >= 12.8.
#   * deeplabcut / pytorch3d / open3d / mmcv / xtcocotools omitted: only TTA, the Gradio app
#     and evaluation use them, not this video pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-prima}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PRIMA_DIR="$ROOT/third_party/PRIMA"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

if [[ ! -d "$PRIMA_DIR/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/AdaptiveMotorControlLab/PRIMA "$PRIMA_DIR"
fi

python -m pip install -U pip wheel "setuptools<81" "packaging<25" "Cython<3" "numpy==1.26.1"
python -m pip install --index-url "$TORCH_INDEX" torch==2.7.1 torchvision==0.22.1
python -m pip install -r "$ROOT/requirements.txt"
python -m pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
python -m pip install --no-deps -e "$PRIMA_DIR"

[[ "${SKIP_DATA:-0}" == "1" ]] || (cd "$PRIMA_DIR" && python scripts/setup_demo_data.py)
python -c "import torch, detectron2, prima; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "[setup] done. Activate with: conda activate $ENV_NAME"
