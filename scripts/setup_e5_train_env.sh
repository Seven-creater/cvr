#!/usr/bin/env bash
set -euo pipefail

CONDA_SH=${CONDA_SH:-/data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-e5_train}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

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
python -m pip install -U \
  "torch" \
  "sentence-transformers[image,audio,video]>=5.4" \
  "transformers" \
  "accelerate" \
  "peft" \
  "datasets" \
  "safetensors" \
  "tqdm" \
  "numpy"

python - <<'PY'
import torch
import sentence_transformers
import transformers
import peft
print("[e5-train-env] python ok")
print("[e5-train-env] torch", torch.__version__)
print("[e5-train-env] sentence_transformers", sentence_transformers.__version__)
print("[e5-train-env] transformers", transformers.__version__)
print("[e5-train-env] peft", peft.__version__)
PY

echo "[e5-train-env] done env=$ENV_NAME"
