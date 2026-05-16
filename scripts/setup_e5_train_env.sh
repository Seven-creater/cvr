#!/usr/bin/env bash
set -euo pipefail

CONDA_SH=${CONDA_SH:-/data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-e5_train}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_VERSION=${TORCH_VERSION:-2.5.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.20.1}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.5.1}
TORCH_CUDA_INDEX=${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}

if [ ! -f "$CONDA_SH" ]; then
  echo "[e5-train-env] missing conda profile: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[e5-train-env] reuse conda env: $ENV_NAME"
else
  echo "[e5-train-env] create conda env: $ENV_NAME python=$PYTHON_VERSION"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
python -m pip install -U pip

echo "[e5-train-env] install torch stack from $TORCH_CUDA_INDEX"
python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
python -m pip install --force-reinstall --index-url "$TORCH_CUDA_INDEX" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

python -m pip install -U --upgrade-strategy only-if-needed \
  "sentence-transformers[image,audio,video]>=5.4" \
  "transformers" \
  "accelerate" \
  "peft" \
  "datasets" \
  "safetensors" \
  "tqdm" \
  "numpy"

# Keep the PyTorch stack pinned to CUDA 12.x wheels. The training server
# currently exposes a CUDA 12.2 driver API, so newer CUDA 13 wheels fail before
# the smoke can run. Reinstalling after optional extras prevents pip from
# leaving torchvision/torchaudio on incompatible wheels.
python -m pip install --force-reinstall --index-url "$TORCH_CUDA_INDEX" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

python - <<'PY'
import torch
import sentence_transformers
import transformers
import peft
print("[e5-train-env] python ok")
print("[e5-train-env] torch", torch.__version__)
print("[e5-train-env] torch cuda", torch.version.cuda)
try:
    import torchvision
    print("[e5-train-env] torchvision", torchvision.__version__)
except Exception as exc:
    print("[e5-train-env] torchvision import failed", repr(exc))
try:
    import torchaudio
    print("[e5-train-env] torchaudio", torchaudio.__version__)
except Exception as exc:
    print("[e5-train-env] torchaudio import failed", repr(exc))
print("[e5-train-env] sentence_transformers", sentence_transformers.__version__)
print("[e5-train-env] transformers", transformers.__version__)
print("[e5-train-env] peft", peft.__version__)
PY

echo "[e5-train-env] done env=$ENV_NAME"
