#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
WAN_ROOT=${WAN_ROOT:-$MODEL_ROOT/Wan2.1}
CONDA_ENV=${CONDA_ENV:-omni_src}

echo "[video-edit-env] start $(date)"
echo "[video-edit-env] model_root=$MODEL_ROOT"
echo "[video-edit-env] wan_root=$WAN_ROOT"
echo "[video-edit-env] conda_env=$CONDA_ENV"

echo "[video-edit-env] disk"
df -h "$MODEL_ROOT" || df -h "$(dirname "$MODEL_ROOT")"

echo "[video-edit-env] gpu"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits || true

echo "[video-edit-env] conda envs"
/data02/usr/wangqihao/miniconda3/bin/conda env list || true

echo "[video-edit-env] python tools"
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
python - <<'PY'
import importlib.util
import shutil
import sys

print("python", sys.executable)
for name in ("huggingface_hub", "modelscope", "torch", "diffusers", "transformers"):
    print(f"{name}:", "ok" if importlib.util.find_spec(name) else "missing")
for command in ("hf", "modelscope", "git", "ffprobe", "ffmpeg"):
    print(f"{command}:", shutil.which(command) or "missing")
PY

echo "[video-edit-env] existing model dirs"
for path in \
  "$WAN_ROOT/code" \
  "$WAN_ROOT/Wan2.1-VACE-1.3B" \
  "$WAN_ROOT/Wan2.1-VACE-14B" \
  "$MODEL_ROOT/LTX-Video/code" \
  "$MODEL_ROOT/LTX-Video/weights"; do
  if [[ -e "$path" ]]; then
    du -sh "$path" || true
  else
    echo "missing $path"
  fi
done

echo "[video-edit-env] note: this script only checks environment and never downloads or runs generation."
echo "[video-edit-env] done $(date)"
