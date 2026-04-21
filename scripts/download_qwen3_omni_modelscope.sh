#!/usr/bin/env bash
set -euo pipefail

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

BASE=/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone
LOG_ROOT=/data02/usr/wangqihao/Demo/test/cvr/runs/model_download_20260421

mkdir -p "$BASE"
mkdir -p "$LOG_ROOT"

python -m pip install -U modelscope

echo "[check] modelscope version"
python - <<'PY'
import modelscope
print(modelscope.__version__)
PY

echo "[download] Qwen3-Omni-30B-A3B-Instruct start $(date)"
modelscope download \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local_dir "$BASE/qwen3-omni-30b-a3b-instruct"

echo "[download] Qwen3-Omni-30B-A3B-Captioner start $(date)"
modelscope download \
  --model Qwen/Qwen3-Omni-30B-A3B-Captioner \
  --local_dir "$BASE/qwen3-omni-30b-a3b-captioner"

echo "[verify] disk usage"
du -sh "$BASE/qwen3-omni-30b-a3b-instruct" || true
du -sh "$BASE/qwen3-omni-30b-a3b-captioner" || true

echo "[verify] required files"
test -f "$BASE/qwen3-omni-30b-a3b-instruct/config.json" && echo "instruct config.json OK" || echo "instruct config.json MISSING"
test -f "$BASE/qwen3-omni-30b-a3b-captioner/config.json" && echo "captioner config.json OK" || echo "captioner config.json MISSING"

echo "[verify] safetensors count"
find "$BASE/qwen3-omni-30b-a3b-instruct" -name "*.safetensors" | wc -l
find "$BASE/qwen3-omni-30b-a3b-captioner" -name "*.safetensors" | wc -l

echo "[verify] top-level files instruct"
find "$BASE/qwen3-omni-30b-a3b-instruct" -maxdepth 1 -type f | sort | head -80

echo "[verify] top-level files captioner"
find "$BASE/qwen3-omni-30b-a3b-captioner" -maxdepth 1 -type f | sort | head -80

echo "[done] $(date)"
