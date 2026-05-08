#!/usr/bin/env bash
set -euo pipefail

if [ -f /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-omni_src}"
fi

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATASET_ROOT=${DATASET_ROOT:-/data02/usr/wangqihao/Demo/test/data}
RUNS_ROOT=${RUNS_ROOT:-/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs}
RUN_ROOT=${RUN_ROOT:-$RUNS_ROOT/composed_avigate_smoke20_$(date +%Y%m%d_%H%M%S)}
STAGED_ROOT=${STAGED_ROOT:-$RUNS_ROOT/composed_avigate_smoke20_staged}
CACHE_DIR=${CACHE_DIR:-$RUNS_ROOT/composed_avigate_runtime_cache}

MODEL_DIR=${MODEL_DIR:-/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/avigate/ckpt_msrvtt_paper_like_4gpu_stable}
CHECKPOINT=${CHECKPOINT:-$MODEL_DIR/pytorch_model.bin.4}
CLIP_WEIGHT=${CLIP_WEIGHT:-/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/clip/ViT-B-32.pt}

CHECKER_BASE_URL=${CHECKER_BASE_URL:-http://127.0.0.1:8092/v1}
CHECKER_API_KEY=${CHECKER_API_KEY:-EMPTY}
CHECKER_MODEL=${CHECKER_MODEL:-}
GPU_ID=${GPU_ID:-4}
SAMPLE_SIZE=${SAMPLE_SIZE:-20}
OMNI_CONCURRENCY=${OMNI_CONCURRENCY:-2}
RERANK_WINDOW=${RERANK_WINDOW:-5}
TOPK=${TOPK:-1,5,10}
TOPK_VALUE=${TOPK_VALUE:-10}
MAX_WORDS=${MAX_WORDS:-32}
LINK_MODE=${LINK_MODE:-symlink}
FFMPEG=${FFMPEG:-ffmpeg}

usage() {
  cat <<'EOF'
Usage: run_composed_avigate_smoke20.sh [options]

Options:
  --dataset-root PATH
  --run-root PATH
  --staged-root PATH
  --cache-dir PATH
  --model-dir PATH
  --checkpoint PATH
  --clip-weight PATH
  --checker-base-url URL
  --checker-api-key KEY
  --checker-model NAME
  --gpu-id ID
  --sample-size N
  --omni-concurrency N
  --rerank-window N
  --topk 1,5,10
  --topk-value N
  --max-words N
  --link-mode symlink|hardlink|copy
  --ffmpeg PATH
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --staged-root) STAGED_ROOT="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --clip-weight) CLIP_WEIGHT="$2"; shift 2 ;;
    --checker-base-url) CHECKER_BASE_URL="$2"; shift 2 ;;
    --checker-api-key) CHECKER_API_KEY="$2"; shift 2 ;;
    --checker-model) CHECKER_MODEL="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --omni-concurrency) OMNI_CONCURRENCY="$2"; shift 2 ;;
    --rerank-window) RERANK_WINDOW="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --topk-value) TOPK_VALUE="$2"; shift 2 ;;
    --max-words) MAX_WORDS="$2"; shift 2 ;;
    --link-mode) LINK_MODE="$2"; shift 2 ;;
    --ffmpeg) FFMPEG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[composed-avigate-smoke20] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_path() {
  local label="$1"
  local path="$2"
  if [ ! -e "$path" ]; then
    echo "[composed-avigate-smoke20] missing $label: $path" >&2
    exit 1
  fi
}

require_path "dataset root" "$DATASET_ROOT"
require_path "AVIGATE model dir" "$MODEL_DIR"
require_path "AVIGATE checkpoint" "$CHECKPOINT"
require_path "CLIP weight" "$CLIP_WEIGHT"
command -v "$FFMPEG" >/dev/null 2>&1 || { echo "[composed-avigate-smoke20] ffmpeg not found: $FFMPEG" >&2; exit 1; }

if [ -z "$CHECKER_MODEL" ]; then
  CHECKER_MODEL=$(python3 - "$CHECKER_BASE_URL" "$CHECKER_API_KEY" <<'PY'
import json
import sys
from urllib import request

base_url, api_key = sys.argv[1], sys.argv[2]
req = request.Request(base_url.rstrip("/") + "/models")
if api_key:
    req.add_header("Authorization", f"Bearer {api_key}")
with request.urlopen(req, timeout=20) as response:
    payload = json.loads(response.read().decode("utf-8"))
for item in payload.get("data", []):
    if isinstance(item, dict) and item.get("id"):
        print(item["id"])
        raise SystemExit(0)
raise SystemExit("no model id found in /models response")
PY
)
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[composed-avigate-smoke20] repo=$REPO_ROOT"
echo "[composed-avigate-smoke20] dataset_root=$DATASET_ROOT sample_size=$SAMPLE_SIZE"
echo "[composed-avigate-smoke20] run_root=$RUN_ROOT"
echo "[composed-avigate-smoke20] staged_root=$STAGED_ROOT"
echo "[composed-avigate-smoke20] cache_dir=$CACHE_DIR"
echo "[composed-avigate-smoke20] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[composed-avigate-smoke20] checker_base_url=$CHECKER_BASE_URL checker_model=$CHECKER_MODEL"

python3 -m app.composed_avigate_smoke \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$RUN_ROOT" \
  --staged-root "$STAGED_ROOT" \
  --sample-size "$SAMPLE_SIZE" \
  --model-dir "$MODEL_DIR" \
  --checkpoint "$CHECKPOINT" \
  --clip-weight "$CLIP_WEIGHT" \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --max-words "$MAX_WORDS" \
  --topk "$TOPK" \
  --topk-value "$TOPK_VALUE" \
  --checker-base-url "$CHECKER_BASE_URL" \
  --checker-api-key "$CHECKER_API_KEY" \
  --checker-model "$CHECKER_MODEL" \
  --omni-concurrency "$OMNI_CONCURRENCY" \
  --rerank-window "$RERANK_WINDOW" \
  --link-mode "$LINK_MODE" \
  --ffmpeg "$FFMPEG"

echo "[composed-avigate-smoke20] comparison: $RUN_ROOT/comparison.md"
