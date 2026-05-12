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
GPU_IDS=${GPU_IDS:-}
PARALLEL_MODES=${PARALLEL_MODES:-0}
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
  --gpu-ids ID1,ID2,ID3
  --parallel-modes
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
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --parallel-modes) PARALLEL_MODES=1; shift ;;
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
echo "[e5-three-data] gpu_ids=${GPU_IDS:-}"
echo "[e5-three-data] parallel_modes=$PARALLEL_MODES"
echo "[e5-three-data] expected_count=$EXPECTED_COUNT"
echo "[e5-three-data] topk=$TOPK"

run_mode() {
  local label="$1"
  local mode_root="$2"
  local gpu_id="$3"
  local query_mode="$4"
  local video_audio_mode="$5"
  shift 5

  echo "[e5-three-data] $label start $(date) gpu=$gpu_id run_root=$mode_root"
  bash scripts/run_e5_cvr_eval.sh \
    "${common_args[@]}" \
    --gpu-id "$gpu_id" \
    --run-root "$mode_root" \
    --query-mode "$query_mode" \
    --video-audio-mode "$video_audio_mode" \
    "$@" \
    "${force_args[@]}"
  echo "[e5-three-data] $label done $(date)"
}

if [ "$PARALLEL_MODES" = "1" ]; then
  if [ -z "$GPU_IDS" ]; then
    echo "[e5-three-data] --parallel-modes requires --gpu-ids ID1,ID2,ID3" >&2
    exit 2
  fi
  IFS=',' read -r -a MODE_GPUS <<< "$GPU_IDS"
  if [ "${#MODE_GPUS[@]}" -lt 3 ]; then
    echo "[e5-three-data] --parallel-modes requires at least 3 GPU ids, got: $GPU_IDS" >&2
    exit 2
  fi

  mkdir -p "$RUN_ROOT/mode_logs"
  pids=()
  labels=()

  (
    run_mode "mode 1/3 V+T+A audio-on" "$RUN_ROOT/vta_audio_on" "${MODE_GPUS[0]}" composed on
  ) > "$RUN_ROOT/mode_logs/vta_audio_on.log" 2>&1 &
  pids+=($!)
  labels+=("V+T+A")

  (
    run_mode "mode 2/3 V+T audio-off" "$RUN_ROOT/vt_audio_off" "${MODE_GPUS[1]}" composed off
  ) > "$RUN_ROOT/mode_logs/vt_audio_off.log" 2>&1 &
  pids+=($!)
  labels+=("V+T")

  # In parallel mode this builds its own audio-on target index instead of waiting
  # for V+T+A, trading some duplicate work for lower wall-clock time.
  (
    run_mode "mode 3/3 V+A video-only audio-on" "$RUN_ROOT/va_video_only_audio_on" "${MODE_GPUS[2]}" video-only on
  ) > "$RUN_ROOT/mode_logs/va_video_only_audio_on.log" 2>&1 &
  pids+=($!)
  labels+=("V+A")

  failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[e5-three-data] parallel mode done label=${labels[$i]} log=$RUN_ROOT/mode_logs"
    else
      echo "[e5-three-data] parallel mode FAILED label=${labels[$i]} log=$RUN_ROOT/mode_logs" >&2
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "[e5-three-data] one or more parallel modes failed; skip grouped summary" >&2
    exit 1
  fi
else
  echo "[e5-three-data] serial mode 1/3 V+T+A audio-on start $(date)"
  run_mode "mode 1/3 V+T+A audio-on" "$RUN_ROOT/vta_audio_on" "$GPU_ID" composed on

  echo "[e5-three-data] serial mode 2/3 V+T audio-off start $(date)"
  run_mode "mode 2/3 V+T audio-off" "$RUN_ROOT/vt_audio_off" "$GPU_ID" composed off

  echo "[e5-three-data] serial mode 3/3 V+A video-only audio-on start $(date)"
  run_mode \
    "mode 3/3 V+A video-only audio-on" \
    "$RUN_ROOT/va_video_only_audio_on" \
    "$GPU_ID" \
    video-only \
    on \
    --target-index-dir "$RUN_ROOT/vta_audio_on/target_index"
fi

echo "[e5-three-data] grouped summary start $(date)"
python3 -m app.e5_three_data_eval \
  --triplets-jsonl "$TRIPLETS_JSONL" \
  --run-root "$RUN_ROOT" \
  --topk "$TOPK"

echo "[e5-three-data] comparison: $RUN_ROOT/comparison_by_dataset.md"
echo "[e5-three-data] done $(date)"
