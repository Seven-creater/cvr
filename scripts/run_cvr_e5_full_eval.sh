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

RUNS_ROOT=${RUNS_ROOT:-/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs}
RUN_ROOT=${RUN_ROOT:-$RUNS_ROOT/cvr_e5_full_$(date +%Y%m%d_%H%M%S)}
STAGED_ROOT=${STAGED_ROOT:-}
CACHE_DIR=${CACHE_DIR:-$RUNS_ROOT/composed_avigate_runtime_cache}

MODEL_DIR=${MODEL_DIR:-/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/avigate/ckpt_msrvtt_paper_like_4gpu_stable}
CHECKPOINT=${CHECKPOINT:-$MODEL_DIR/pytorch_model.bin.4}
CLIP_WEIGHT=${CLIP_WEIGHT:-/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/clip/ViT-B-32.pt}
E5_MODEL=${E5_MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B}
E5_INDEX_DIR=${E5_INDEX_DIR:-$RUNS_ROOT/e5_omni_target_index}

CHECKER_BASE_URL=${CHECKER_BASE_URL:-http://127.0.0.1:8092/v1}
CHECKER_API_KEY=${CHECKER_API_KEY:-EMPTY}
CHECKER_MODEL=${CHECKER_MODEL:-qwen2.5-omni}
GPU_ID=${GPU_ID:-4}
SAMPLE_SIZE=${SAMPLE_SIZE:-}
TOPK=${TOPK:-1,5,10}
TOPK_VALUE=${TOPK_VALUE:-10}
AVIGATE_TOPK=${AVIGATE_TOPK:-50}
E5_TOPK=${E5_TOPK:-50}
FUSED_TOPK=${FUSED_TOPK:-20}
OMNI_CONCURRENCY=${OMNI_CONCURRENCY:-2}
RERANK_WINDOW=${RERANK_WINDOW:-5}
SKIP_AGENT=${SKIP_AGENT:-0}

usage() {
  cat <<'EOF'
Usage: run_cvr_e5_full_eval.sh [options]

Runs the full CVR evaluation on an existing composed AVIGATE staging
directory. This script does not download e5, does not start Omni, and
does not rebuild the dataset.

Options:
  --run-root PATH
  --staged-root PATH
  --cache-dir PATH
  --model-dir PATH
  --checkpoint PATH
  --clip-weight PATH
  --e5-model PATH
  --e5-index-dir PATH
  --gpu-id ID
  --sample-size N
  --skip-agent
  --checker-base-url URL
  --checker-api-key KEY
  --checker-model NAME
  --topk 1,5,10
  --topk-value N
  --avigate-topk N
  --e5-topk N
  --fused-topk N
  --omni-concurrency N
  --rerank-window N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --staged-root) STAGED_ROOT="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --clip-weight) CLIP_WEIGHT="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --e5-index-dir) E5_INDEX_DIR="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --skip-agent) SKIP_AGENT=1; shift ;;
    --checker-base-url) CHECKER_BASE_URL="$2"; shift 2 ;;
    --checker-api-key) CHECKER_API_KEY="$2"; shift 2 ;;
    --checker-model) CHECKER_MODEL="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --topk-value) TOPK_VALUE="$2"; shift 2 ;;
    --avigate-topk) AVIGATE_TOPK="$2"; shift 2 ;;
    --e5-topk) E5_TOPK="$2"; shift 2 ;;
    --fused-topk) FUSED_TOPK="$2"; shift 2 ;;
    --omni-concurrency) OMNI_CONCURRENCY="$2"; shift 2 ;;
    --rerank-window) RERANK_WINDOW="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[cvr-e5] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$STAGED_ROOT" ]; then
  STAGED_ROOT=$(find "$RUNS_ROOT" -path "*/staged/triplets.jsonl" -print 2>/dev/null | sort | tail -1 | xargs -r dirname)
fi

require_path() {
  local label="$1"
  local path="$2"
  if [ ! -e "$path" ]; then
    echo "[cvr-e5] missing $label: $path" >&2
    exit 1
  fi
}

require_path "staged root" "$STAGED_ROOT"
require_path "staged split.csv" "$STAGED_ROOT/split.csv"
require_path "staged data.json" "$STAGED_ROOT/data.json"
require_path "staged video_root" "$STAGED_ROOT/video_root"
require_path "staged audio_root" "$STAGED_ROOT/audio_root"
require_path "staged triplets.jsonl" "$STAGED_ROOT/triplets.jsonl"
require_path "AVIGATE model dir" "$MODEL_DIR"
require_path "AVIGATE checkpoint" "$CHECKPOINT"
require_path "CLIP weight" "$CLIP_WEIGHT"
require_path "e5-omni model dir" "$E5_MODEL"
require_path "e5-omni config" "$E5_MODEL/config.json"

mkdir -p "$RUN_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[cvr-e5] repo=$REPO_ROOT"
echo "[cvr-e5] run_root=$RUN_ROOT"
echo "[cvr-e5] staged_root=$STAGED_ROOT"
echo "[cvr-e5] e5_model=$E5_MODEL"
echo "[cvr-e5] e5_index_dir=$E5_INDEX_DIR"
echo "[cvr-e5] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[cvr-e5] skip_agent=$SKIP_AGENT checker_model=$CHECKER_MODEL"

ARGS=(
  -m app.eval cvr-full-eval
  --model-dir "$MODEL_DIR"
  --checkpoint "$CHECKPOINT"
  --data-json "$STAGED_ROOT/data.json"
  --split-csv "$STAGED_ROOT/split.csv"
  --video-root "$STAGED_ROOT/video_root"
  --audio-root "$STAGED_ROOT/audio_root"
  --clip-weight "$CLIP_WEIGHT"
  --cache-dir "$CACHE_DIR"
  --device cuda
  --triplets-jsonl "$STAGED_ROOT/triplets.jsonl"
  --output-dir "$RUN_ROOT"
  --e5-model "$E5_MODEL"
  --e5-index-dir "$E5_INDEX_DIR"
  --topk "$TOPK"
  --topk-value "$TOPK_VALUE"
  --avigate-topk "$AVIGATE_TOPK"
  --e5-topk "$E5_TOPK"
  --fused-topk "$FUSED_TOPK"
  --omni-concurrency "$OMNI_CONCURRENCY"
  --rerank-window "$RERANK_WINDOW"
  --checker-base-url "$CHECKER_BASE_URL"
  --checker-api-key "$CHECKER_API_KEY"
  --checker-model "$CHECKER_MODEL"
)

if [ -n "$SAMPLE_SIZE" ]; then
  ARGS+=(--sample-size "$SAMPLE_SIZE")
fi
if [ "$SKIP_AGENT" = "1" ]; then
  ARGS+=(--skip-agent)
fi

python3 "${ARGS[@]}"
echo "[cvr-e5] comparison: $RUN_ROOT/comparison.md"
