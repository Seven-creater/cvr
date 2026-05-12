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
RUN_ROOT=${RUN_ROOT:-}
TRIPLETS_JSONL=${TRIPLETS_JSONL:-/data02/usr/wangqihao/Demo/test/three_data/merged_all/triplets.jsonl}
E5_MODEL=${E5_MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B}
GPU_ID=${GPU_ID:-4}
EXPECTED_COUNT=${EXPECTED_COUNT:-1697}
SMOKE_SIZE=${SMOKE_SIZE:-20}
TOPK=${TOPK:-1,5,10}
TOPK_TRACE=${TOPK_TRACE:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
VIDEO_FPS=${VIDEO_FPS:-1}
FORCE_REBUILD_INDEX=${FORCE_REBUILD_INDEX:-0}

usage() {
  cat <<'EOF'
Usage: run_e5_three_data_mixed_eval.sh [options]

Runs e5-omni over the merged three-data CVR manifest, then reports metrics
by dataset (cvr_943, a_line, b_line). This script does not start Omni/vLLM,
does not start AVIGATE, and does not download models.

Options:
  --triplets-jsonl PATH
  --run-root PATH
  --runs-root PATH
  --e5-model PATH
  --gpu-id ID
  --expected-count N
  --smoke-size N
  --topk 1,5,10
  --topk-trace N
  --batch-size N
  --torch-dtype NAME
  --attn-implementation NAME
  --video-max-pixels N
  --video-fps N
  --force-rebuild-index
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --triplets-jsonl) TRIPLETS_JSONL="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --expected-count) EXPECTED_COUNT="$2"; shift 2 ;;
    --smoke-size) SMOKE_SIZE="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --topk-trace) TOPK_TRACE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --attn-implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --video-max-pixels) VIDEO_MAX_PIXELS="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --force-rebuild-index) FORCE_REBUILD_INDEX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[e5-three-data] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_path() {
  local label="$1"
  local path="$2"
  if [ ! -e "$path" ]; then
    echo "[e5-three-data] missing $label: $path" >&2
    exit 1
  fi
}

require_path "triplets jsonl" "$TRIPLETS_JSONL"
require_path "e5 model dir" "$E5_MODEL"
require_path "e5 config" "$E5_MODEL/config.json"

if [ -z "$RUN_ROOT" ]; then
  RUN_ROOT=$RUNS_ROOT/e5_three_data_mixed_$(date +%Y%m%d_%H%M%S)
fi

TRIPLET_COUNT=$(wc -l < "$TRIPLETS_JSONL")
if [ "$TRIPLET_COUNT" -ne "$EXPECTED_COUNT" ]; then
  echo "[e5-three-data] expected $EXPECTED_COUNT triplets, got $TRIPLET_COUNT: $TRIPLETS_JSONL" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"

common_args=(
  --triplets-jsonl "$TRIPLETS_JSONL"
  --runs-root "$RUNS_ROOT"
  --e5-model "$E5_MODEL"
  --gpu-id "$GPU_ID"
  --expected-count "$EXPECTED_COUNT"
  --smoke-size "$SMOKE_SIZE"
  --topk "$TOPK"
  --topk-trace "$TOPK_TRACE"
  --batch-size "$BATCH_SIZE"
  --torch-dtype "$TORCH_DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --video-max-pixels "$VIDEO_MAX_PIXELS"
  --video-fps "$VIDEO_FPS"
  --reference-audio-mode original
)

force_args=()
if [ "$FORCE_REBUILD_INDEX" = "1" ]; then
  force_args=(--force-rebuild-index)
fi

echo "[e5-three-data] repo=$REPO_ROOT"
echo "[e5-three-data] run_root=$RUN_ROOT"
echo "[e5-three-data] triplets_jsonl=$TRIPLETS_JSONL"
echo "[e5-three-data] e5_model=$E5_MODEL"
echo "[e5-three-data] gpu_id=$GPU_ID"
echo "[e5-three-data] expected_count=$EXPECTED_COUNT"
echo "[e5-three-data] topk=$TOPK"
echo "[e5-three-data] mode 1/3 V+T+A audio-on start $(date)"

bash scripts/run_e5_cvr_eval.sh \
  "${common_args[@]}" \
  --run-root "$RUN_ROOT/vta_audio_on" \
  --query-mode composed \
  --video-audio-mode on \
  "${force_args[@]}"

echo "[e5-three-data] mode 2/3 V+T audio-off start $(date)"
bash scripts/run_e5_cvr_eval.sh \
  "${common_args[@]}" \
  --run-root "$RUN_ROOT/vt_audio_off" \
  --query-mode composed \
  --video-audio-mode off \
  "${force_args[@]}"

echo "[e5-three-data] mode 3/3 V+A video-only audio-on start $(date)"
bash scripts/run_e5_cvr_eval.sh \
  "${common_args[@]}" \
  --run-root "$RUN_ROOT/va_video_only_audio_on" \
  --query-mode video-only \
  --video-audio-mode on \
  --target-index-dir "$RUN_ROOT/vta_audio_on/target_index"

echo "[e5-three-data] grouped summary start $(date)"
python3 -m app.e5_three_data_eval \
  --triplets-jsonl "$TRIPLETS_JSONL" \
  --run-root "$RUN_ROOT" \
  --topk "$TOPK"

echo "[e5-three-data] comparison: $RUN_ROOT/comparison_by_dataset.md"
echo "[e5-three-data] done $(date)"
