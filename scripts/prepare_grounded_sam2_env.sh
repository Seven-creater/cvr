#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
MASK_ROOT=${MASK_ROOT:-$MODEL_ROOT/MaskEdit}
CONDA_ENV=${CONDA_ENV:-grounded_sam2}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

echo "[grounded-sam2-env] start $(date)"
echo "[grounded-sam2-env] model_root=$MODEL_ROOT"
echo "[grounded-sam2-env] mask_root=$MASK_ROOT"
echo "[grounded-sam2-env] conda_env=$CONDA_ENV"

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "[grounded-sam2-env] creating env $CONDA_ENV"
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
else
  echo "[grounded-sam2-env] env exists: $CONDA_ENV"
fi

conda activate "$CONDA_ENV"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
python -m pip install \
  opencv-python pillow matplotlib tqdm numpy scipy einops hydra-core iopath \
  transformers accelerate addict yapf supervision pycocotools timm

if [[ -d "$MASK_ROOT/SAM2.1/code" ]]; then
  echo "[grounded-sam2-env] install SAM2.1 editable"
  python -m pip install -e "$MASK_ROOT/SAM2.1/code"
else
  echo "[grounded-sam2-env] missing SAM2.1 code: $MASK_ROOT/SAM2.1/code"
fi

if [[ -d "$MASK_ROOT/GroundingDINO/code" ]]; then
  echo "[grounded-sam2-env] install GroundingDINO editable"
  python -m pip install -e "$MASK_ROOT/GroundingDINO/code"
else
  echo "[grounded-sam2-env] missing GroundingDINO code: $MASK_ROOT/GroundingDINO/code"
fi

python - <<'PY'
import importlib.util
import sys
import torch

print("[grounded-sam2-env] python", sys.executable)
print("[grounded-sam2-env] torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for name in ("cv2", "transformers", "groundingdino", "sam2"):
    print(f"[grounded-sam2-env] {name}", "ok" if importlib.util.find_spec(name) else "missing")
PY

echo "[grounded-sam2-env] done $(date)"
