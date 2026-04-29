#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
IMAGE_ROOT=${IMAGE_ROOT:-$MODEL_ROOT/ImageGen}
LOG_ROOT=${LOG_ROOT:-$MODEL_ROOT/model_download_logs}
CONDA_ENV=${CONDA_ENV:-omni_src}

mkdir -p "$IMAGE_ROOT" "$LOG_ROOT"

nohup bash -lc '
set -u
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "'"$CONDA_ENV"'"

echo "[image-download] start $(date)"
echo "[image-download] model_root='"$MODEL_ROOT"'"
mkdir -p "'"$IMAGE_ROOT"'/Qwen-Image-2512" \
         "'"$IMAGE_ROOT"'/Qwen-Image-Edit-2511" \
         "'"$IMAGE_ROOT"'/Qwen-Image-Edit-2509"

download_ms() {
  local repo="$1"
  local out="$2"
  echo "[image-download] try modelscope repo=$repo -> $out"
  mkdir -p "$out"
  modelscope download --model "$repo" --local_dir "$out" \
    && echo "[image-download] OK $repo" \
    || echo "[image-download] FAIL $repo"
}

download_ms Qwen/Qwen-Image-2512 "'"$IMAGE_ROOT"'/Qwen-Image-2512"
download_ms Qwen/Qwen-Image-Edit-2511 "'"$IMAGE_ROOT"'/Qwen-Image-Edit-2511"
download_ms Qwen/Qwen-Image-Edit-2509 "'"$IMAGE_ROOT"'/Qwen-Image-Edit-2509"

echo "[image-download] disk usage"
du -sh "'"$IMAGE_ROOT"'"/* || true
echo "[image-download] done $(date)"
' > "$LOG_ROOT/image_generation_modelscope_download.log" 2>&1 &

echo "DOWNLOAD_PID=$!"
echo "LOG=$LOG_ROOT/image_generation_modelscope_download.log"
