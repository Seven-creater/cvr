#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
MASK_ROOT=${MASK_ROOT:-$MODEL_ROOT/MaskEdit}
CONDA_ENV=${CONDA_ENV:-grounded_sam2}
GROUNDING_DINO_CODE=${GROUNDING_DINO_CODE:-$MASK_ROOT/GroundingDINO/code}

echo "[groundingdino-repair] start $(date)"
echo "[groundingdino-repair] conda_env=$CONDA_ENV"
echo "[groundingdino-repair] grounding_dino_code=$GROUNDING_DINO_CODE"

if [[ ! -d "$GROUNDING_DINO_CODE" ]]; then
  echo "[groundingdino-repair] missing GroundingDINO code: $GROUNDING_DINO_CODE" >&2
  exit 1
fi

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install cython

cd "$GROUNDING_DINO_CODE"
export GROUNDINGDINO_USE_CUDA=${GROUNDINGDINO_USE_CUDA:-1}
export FORCE_CUDA=${FORCE_CUDA:-1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}

echo "[groundingdino-repair] install editable without build isolation"
python -m pip install --no-build-isolation -e .

python - <<'PY'
import importlib.util
import sys
import torch

print("[groundingdino-repair] python", sys.executable)
print("[groundingdino-repair] torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for name in ("groundingdino", "groundingdino.groundingdino.util.inference"):
    print(f"[groundingdino-repair] {name}", "ok" if importlib.util.find_spec(name) else "missing")
if importlib.util.find_spec("groundingdino") is None:
    raise SystemExit("groundingdino import still missing")
PY

echo "[groundingdino-repair] done $(date)"
