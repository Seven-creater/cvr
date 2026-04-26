#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
MASK_ROOT=${MASK_ROOT:-$MODEL_ROOT/MaskEdit}
LOG_ROOT=${LOG_ROOT:-$MODEL_ROOT/model_download_logs}
CONDA_ENV=${CONDA_ENV:-omni_src}
mkdir -p "$MASK_ROOT" "$LOG_ROOT"

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "[mask-download] start $(date)"
echo "[mask-download] model_root=$MODEL_ROOT"
echo "[mask-download] mask_root=$MASK_ROOT"
echo "[mask-download] conda_env=$CONDA_ENV"

mkdir -p \
  "$MASK_ROOT/Grounded-SAM-2" \
  "$MASK_ROOT/GroundingDINO" \
  "$MASK_ROOT/SAM2.1" \
  "$MASK_ROOT/Florence-2"

clone_or_pull() {
  local repo="$1"
  local out="$2"
  if [ ! -d "$out/.git" ]; then
    git clone "$repo" "$out" || echo "[mask-download] clone failed $repo"
  else
    git -C "$out" pull --ff-only || true
  fi
}

download_ms() {
  local repo="$1"
  local out="$2"
  echo "[mask-download] try modelscope repo=$repo -> $out"
  mkdir -p "$out"
  modelscope download --model "$repo" --local_dir "$out" \
    && echo "[mask-download] OK $repo" \
    || echo "[mask-download] FAIL $repo"
}

clone_or_pull https://github.com/IDEA-Research/Grounded-SAM-2.git "$MASK_ROOT/Grounded-SAM-2/code"
clone_or_pull https://github.com/IDEA-Research/GroundingDINO.git "$MASK_ROOT/GroundingDINO/code"
clone_or_pull https://github.com/facebookresearch/sam2.git "$MASK_ROOT/SAM2.1/code"

download_ms facebook/sam2.1-hiera-large "$MASK_ROOT/SAM2.1/checkpoints/sam2.1-hiera-large"
download_ms facebook/sam2.1-hiera-base-plus "$MASK_ROOT/SAM2.1/checkpoints/sam2.1-hiera-base-plus"
download_ms IDEA-Research/grounding-dino-base "$MASK_ROOT/GroundingDINO/checkpoints"
download_ms IDEA-Research/grounding-dino-tiny "$MASK_ROOT/GroundingDINO/checkpoints"
download_ms IDEA-Research/GroundingDINO "$MASK_ROOT/GroundingDINO/checkpoints"
download_ms microsoft/Florence-2-large "$MASK_ROOT/Florence-2/Florence-2-large"
download_ms microsoft/Florence-2-base "$MASK_ROOT/Florence-2/Florence-2-base"

echo "[mask-download] disk usage"
du -sh "$MASK_ROOT"/* || true
echo "[mask-download] done $(date)"
