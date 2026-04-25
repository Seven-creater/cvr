#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
LOG_ROOT=${LOG_ROOT:-$MODEL_ROOT/model_download_logs}
CONDA_ENV=${CONDA_ENV:-omni_src}
mkdir -p "$LOG_ROOT"

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "[download] start $(date)"
echo "[download] model_root=$MODEL_ROOT"
echo "[download] conda_env=$CONDA_ENV"

mkdir -p \
  "$MODEL_ROOT/LTX-2" \
  "$MODEL_ROOT/LTX-Video" \
  "$MODEL_ROOT/AudioEdit/FoleyCrafter" \
  "$MODEL_ROOT/AudioEdit/Frieren-V2A"

clone_or_pull() {
  local repo="$1"
  local out="$2"
  if [ ! -d "$out/.git" ]; then
    git clone "$repo" "$out" || echo "[download] clone failed $repo"
  else
    git -C "$out" pull --ff-only || true
  fi
}

download_ms() {
  local repo="$1"
  local out="$2"
  echo "[download] try modelscope repo=$repo -> $out"
  mkdir -p "$out"
  modelscope download --model "$repo" --local_dir "$out" \
    && echo "[download] OK $repo" \
    || echo "[download] FAIL $repo"
}

clone_or_pull https://github.com/Lightricks/LTX-2.git "$MODEL_ROOT/LTX-2/code"
clone_or_pull https://github.com/Lightricks/LTX-Video.git "$MODEL_ROOT/LTX-Video/code"
clone_or_pull https://github.com/open-mmlab/FoleyCrafter.git "$MODEL_ROOT/AudioEdit/FoleyCrafter/code"
clone_or_pull https://github.com/cyanbx/Frieren-V2A.git "$MODEL_ROOT/AudioEdit/Frieren-V2A/code"

download_ms Lightricks/LTX-2 "$MODEL_ROOT/LTX-2/weights"
download_ms Lightricks/LTX-2.3 "$MODEL_ROOT/LTX-2/weights"
download_ms Lightricks/LTX-Video "$MODEL_ROOT/LTX-Video/weights"
download_ms Lightricks/LTX-Video-0.9.7-distilled "$MODEL_ROOT/LTX-Video/weights"

download_ms ymzhang319/FoleyCrafter "$MODEL_ROOT/AudioEdit/FoleyCrafter/checkpoints"
download_ms open-mmlab/FoleyCrafter "$MODEL_ROOT/AudioEdit/FoleyCrafter/checkpoints"
download_ms auffusion/auffusion-full-no-adapter "$MODEL_ROOT/AudioEdit/FoleyCrafter/checkpoints/auffusion"

download_ms cyanbx/Frieren-V2A "$MODEL_ROOT/AudioEdit/Frieren-V2A/checkpoints"
download_ms Frieren/Frieren-V2A "$MODEL_ROOT/AudioEdit/Frieren-V2A/checkpoints"

echo "[download] disk usage"
du -sh "$MODEL_ROOT/LTX-2" "$MODEL_ROOT/LTX-Video" "$MODEL_ROOT/AudioEdit" || true
echo "[download] done $(date)"
