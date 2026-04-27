#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
MASK_ROOT=${MASK_ROOT:-$MODEL_ROOT/MaskEdit}
CONDA_ENV=${CONDA_ENV:-grounded_sam2}
FLORENCE_MODEL=${FLORENCE_MODEL:-$MASK_ROOT/Florence-2/Florence-2-large}

echo "[florence2-repair] start $(date)"
echo "[florence2-repair] conda_env=$CONDA_ENV"
echo "[florence2-repair] florence_model=$FLORENCE_MODEL"

if [[ ! -d "$FLORENCE_MODEL" ]]; then
  echo "[florence2-repair] missing Florence-2 model dir: $FLORENCE_MODEL" >&2
  exit 1
fi

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export FLORENCE_MODEL

python -m pip install --upgrade pip setuptools wheel
python -m pip install "transformers>=4.45,<4.50" "tokenizers<0.22" accelerate safetensors

python - <<'PY'
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor

model_dir = Path(__import__("os").environ["FLORENCE_MODEL"])
print("[florence2-repair] python", sys.executable)
print("[florence2-repair] torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("[florence2-repair] transformers", transformers.__version__)
processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("[florence2-repair] processor", type(processor).__name__)
print("[florence2-repair] model", type(model).__name__)
PY

echo "[florence2-repair] done $(date)"
