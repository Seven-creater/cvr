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
TARGET_INDEX_DIR=${TARGET_INDEX_DIR:-}
TRIPLETS_JSONL=${TRIPLETS_JSONL:-}
E5_MODEL=${E5_MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B}
GPU_ID=${GPU_ID:-4}
EXPECTED_COUNT=${EXPECTED_COUNT:-943}
SMOKE_SIZE=${SMOKE_SIZE:-20}
QUERY_MODE=${QUERY_MODE:-composed}
REFERENCE_AUDIO_MODE=${REFERENCE_AUDIO_MODE:-original}
REFERENCE_AUDIO_CACHE_DIR=${REFERENCE_AUDIO_CACHE_DIR:-$RUNS_ROOT/e5_reference_audio_media_cache}
VIDEO_AUDIO_MODE=${VIDEO_AUDIO_MODE:-on}
TOPK=${TOPK:-1,5,10}
TOPK_TRACE=${TOPK_TRACE:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
VIDEO_FPS=${VIDEO_FPS:-1}
FFMPEG=${FFMPEG:-ffmpeg}
FFPROBE=${FFPROBE:-ffprobe}
FORCE_REBUILD_INDEX=${FORCE_REBUILD_INDEX:-0}

usage() {
  cat <<'EOF'
Usage: run_e5_cvr_eval.sh [options]

Runs e5-omni only composed video retrieval. This script does not download
models, does not start AVIGATE, does not start vLLM, and does not contact
any Omni service.

Options:
  --triplets-jsonl PATH
  --run-root PATH
  --runs-root PATH
  --target-index-dir PATH
  --e5-model PATH
  --gpu-id ID
  --expected-count N
  --smoke-size N
  --query-mode composed|video-only
  --reference-audio-mode original|muted|silent
  --reference-audio-cache-dir PATH
  --video-audio-mode on|off
  --topk 1,5,10
  --topk-trace N
  --batch-size N
  --torch-dtype NAME
  --attn-implementation NAME
  --video-max-pixels N
  --video-fps N
  --ffmpeg PATH
  --ffprobe PATH
  --force-rebuild-index
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --triplets-jsonl) TRIPLETS_JSONL="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --target-index-dir) TARGET_INDEX_DIR="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --expected-count) EXPECTED_COUNT="$2"; shift 2 ;;
    --smoke-size) SMOKE_SIZE="$2"; shift 2 ;;
    --query-mode) QUERY_MODE="$2"; shift 2 ;;
    --reference-audio-mode) REFERENCE_AUDIO_MODE="$2"; shift 2 ;;
    --reference-audio-cache-dir) REFERENCE_AUDIO_CACHE_DIR="$2"; shift 2 ;;
    --video-audio-mode) VIDEO_AUDIO_MODE="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --topk-trace) TOPK_TRACE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --attn-implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --video-max-pixels) VIDEO_MAX_PIXELS="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --ffmpeg) FFMPEG="$2"; shift 2 ;;
    --ffprobe) FFPROBE="$2"; shift 2 ;;
    --force-rebuild-index) FORCE_REBUILD_INDEX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[e5-cvr] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$TRIPLETS_JSONL" ]; then
  LATEST_TRIPLETS_DIR=$(ls -td "$RUNS_ROOT"/composed_triplets_full_* 2>/dev/null | head -1 || true)
  if [ -n "$LATEST_TRIPLETS_DIR" ]; then
    TRIPLETS_JSONL="$LATEST_TRIPLETS_DIR/triplets.jsonl"
  fi
fi

if [ -z "$RUN_ROOT" ]; then
  if [ "$VIDEO_AUDIO_MODE" = "on" ]; then
    AUDIO_PREFIX=e5_audio_on
    LOAD_AUDIO_FROM_VIDEO=true
  elif [ "$VIDEO_AUDIO_MODE" = "off" ]; then
    AUDIO_PREFIX=e5_audio_off
    LOAD_AUDIO_FROM_VIDEO=false
  else
    echo "[e5-cvr] invalid video audio mode: $VIDEO_AUDIO_MODE" >&2
    exit 2
  fi
  if [ "$REFERENCE_AUDIO_MODE" = "original" ] && [ "$QUERY_MODE" = "composed" ]; then
    RUN_ROOT=$RUNS_ROOT/${AUDIO_PREFIX}_composed_eval_$(date +%Y%m%d_%H%M%S)
  elif [ "$REFERENCE_AUDIO_MODE" = "original" ] && [ "$QUERY_MODE" = "video-only" ]; then
    RUN_ROOT=$RUNS_ROOT/${AUDIO_PREFIX}_video_only_eval_$(date +%Y%m%d_%H%M%S)
  elif [ "$QUERY_MODE" = "video-only" ]; then
    RUN_ROOT=$RUNS_ROOT/${AUDIO_PREFIX}_video_only_ref_${REFERENCE_AUDIO_MODE}_eval_$(date +%Y%m%d_%H%M%S)
  else
    RUN_ROOT=$RUNS_ROOT/${AUDIO_PREFIX}_composed_ref_${REFERENCE_AUDIO_MODE}_eval_$(date +%Y%m%d_%H%M%S)
  fi
fi

if [ "$VIDEO_AUDIO_MODE" = "on" ]; then
  LOAD_AUDIO_FROM_VIDEO=true
elif [ "$VIDEO_AUDIO_MODE" = "off" ]; then
  LOAD_AUDIO_FROM_VIDEO=false
else
  echo "[e5-cvr] invalid video audio mode: $VIDEO_AUDIO_MODE" >&2
  exit 2
fi

require_path() {
  local label="$1"
  local path="$2"
  if [ ! -e "$path" ]; then
    echo "[e5-cvr] missing $label: $path" >&2
    exit 1
  fi
}

require_path "e5 model dir" "$E5_MODEL"
require_path "e5 config" "$E5_MODEL/config.json"
require_path "triplets jsonl" "$TRIPLETS_JSONL"
if [ -n "$TARGET_INDEX_DIR" ]; then
  require_path "target index dir" "$TARGET_INDEX_DIR"
fi
if [ "$REFERENCE_AUDIO_MODE" != "original" ]; then
  command -v "$FFMPEG" >/dev/null || { echo "[e5-cvr] missing ffmpeg: $FFMPEG" >&2; exit 1; }
  command -v "$FFPROBE" >/dev/null || { echo "[e5-cvr] missing ffprobe: $FFPROBE" >&2; exit 1; }
fi

python3 - <<'PY'
import importlib.util
missing = [name for name in ("numpy", "torch", "sentence_transformers") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("[e5-cvr] missing python packages: " + ", ".join(missing))
PY

TRIPLET_COUNT=$(wc -l < "$TRIPLETS_JSONL")
if [ "$TRIPLET_COUNT" -ne "$EXPECTED_COUNT" ]; then
  echo "[e5-cvr] expected $EXPECTED_COUNT triplets, got $TRIPLET_COUNT: $TRIPLETS_JSONL" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[e5-cvr] repo=$REPO_ROOT"
echo "[e5-cvr] run_root=$RUN_ROOT"
echo "[e5-cvr] triplets_jsonl=$TRIPLETS_JSONL"
echo "[e5-cvr] e5_model=$E5_MODEL"
echo "[e5-cvr] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[e5-cvr] query_mode=$QUERY_MODE"
echo "[e5-cvr] reference_audio_mode=$REFERENCE_AUDIO_MODE"
echo "[e5-cvr] target_audio_mode=original"
echo "[e5-cvr] video_audio_mode=$VIDEO_AUDIO_MODE"
echo "[e5-cvr] load_audio_from_video=$LOAD_AUDIO_FROM_VIDEO"
echo "[e5-cvr] use_audio_in_video=$LOAD_AUDIO_FROM_VIDEO"
echo "[e5-cvr] reference_audio_cache_dir=$REFERENCE_AUDIO_CACHE_DIR"
echo "[e5-cvr] target_index_dir=${TARGET_INDEX_DIR:-$RUN_ROOT/target_index}"
echo "[e5-cvr] smoke_size=$SMOKE_SIZE expected_count=$EXPECTED_COUNT"
echo "[e5-cvr] video_max_pixels=$VIDEO_MAX_PIXELS video_fps=$VIDEO_FPS"

ARGS=(
  -m app.e5_cvr_eval
  --triplets-jsonl "$TRIPLETS_JSONL"
  --run-root "$RUN_ROOT"
  --runs-root "$RUNS_ROOT"
  --expected-count "$EXPECTED_COUNT"
  --query-mode "$QUERY_MODE"
  --reference-audio-mode "$REFERENCE_AUDIO_MODE"
  --reference-audio-cache-dir "$REFERENCE_AUDIO_CACHE_DIR"
  --video-audio-mode "$VIDEO_AUDIO_MODE"
  --e5-model "$E5_MODEL"
  --device cuda
  --torch-dtype "$TORCH_DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --batch-size "$BATCH_SIZE"
  --video-max-pixels "$VIDEO_MAX_PIXELS"
  --video-fps "$VIDEO_FPS"
  --ffmpeg "$FFMPEG"
  --ffprobe "$FFPROBE"
  --smoke-size "$SMOKE_SIZE"
  --topk "$TOPK"
  --topk-trace "$TOPK_TRACE"
)

if [ -n "$TARGET_INDEX_DIR" ]; then
  ARGS+=(--target-index-dir "$TARGET_INDEX_DIR")
fi

if [ "$FORCE_REBUILD_INDEX" = "1" ]; then
  ARGS+=(--force-rebuild-index)
fi

python3 "${ARGS[@]}"
echo "[e5-cvr] comparison: $RUN_ROOT/comparison.md"
