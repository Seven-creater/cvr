#!/usr/bin/env bash
set -euo pipefail

if [ -f /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
  conda activate omni_src
fi

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/clips/audio_cvr_8_12s}
CLIP_SECONDS=${CLIP_SECONDS:-10}
MIN_CLIP_SECONDS=${MIN_CLIP_SECONDS:-8}
MAX_CLIP_SECONDS=${MAX_CLIP_SECONDS:-12}
STRIDE_SECONDS=${STRIDE_SECONDS:-}
MIN_CLIPS_PER_SOURCE=${MIN_CLIPS_PER_SOURCE:-2}
MAX_CLIPS_PER_SOURCE=${MAX_CLIPS_PER_SOURCE:-0}
MAX_SOURCE_VIDEOS=${MAX_SOURCE_VIDEOS:-0}
MAX_SOURCE_VIDEOS_PER_DATASET=${MAX_SOURCE_VIDEOS_PER_DATASET:-0}
DATASETS=${DATASETS:-}
DRY_RUN=${DRY_RUN:-0}
OVERWRITE=${OVERWRITE:-0}

usage() {
  cat <<'EOF'
Usage: build_audio_cvr_8_12s_clips.sh [options]

Options:
  --root PATH
  --output-root PATH
  --dataset NAME[,NAME]
  --clip-seconds N          default: 10
  --min-clip-seconds N      default: 8
  --max-clip-seconds N      default: 12
  --stride-seconds N        default: same as clip seconds
  --min-clips-per-source N  default: 2
  --max-clips-per-source N  default: 0 (no cap)
  --max-source-videos N     default: 0 (all)
  --max-source-videos-per-dataset N default: 0 (all)
  --dry-run
  --overwrite
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset) DATASETS="${DATASETS:+$DATASETS,}$2"; shift 2 ;;
    --clip-seconds) CLIP_SECONDS="$2"; shift 2 ;;
    --min-clip-seconds) MIN_CLIP_SECONDS="$2"; shift 2 ;;
    --max-clip-seconds) MAX_CLIP_SECONDS="$2"; shift 2 ;;
    --stride-seconds) STRIDE_SECONDS="$2"; shift 2 ;;
    --min-clips-per-source) MIN_CLIPS_PER_SOURCE="$2"; shift 2 ;;
    --max-clips-per-source) MAX_CLIPS_PER_SOURCE="$2"; shift 2 ;;
    --max-source-videos) MAX_SOURCE_VIDEOS="$2"; shift 2 ;;
    --max-source-videos-per-dataset) MAX_SOURCE_VIDEOS_PER_DATASET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[audio-cvr-clips] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

args=(
  python3 -m app.audio_cvr_clips
  --root "$ROOT"
  --output-root "$OUTPUT_ROOT"
  --clip-seconds "$CLIP_SECONDS"
  --min-clip-seconds "$MIN_CLIP_SECONDS"
  --max-clip-seconds "$MAX_CLIP_SECONDS"
  --min-clips-per-source "$MIN_CLIPS_PER_SOURCE"
  --max-clips-per-source "$MAX_CLIPS_PER_SOURCE"
  --max-source-videos "$MAX_SOURCE_VIDEOS"
  --max-source-videos-per-dataset "$MAX_SOURCE_VIDEOS_PER_DATASET"
)

if [ -n "$STRIDE_SECONDS" ]; then
  args+=(--stride-seconds "$STRIDE_SECONDS")
fi
if [ -n "$DATASETS" ]; then
  IFS=',' read -r -a dataset_items <<< "$DATASETS"
  for dataset in "${dataset_items[@]}"; do
    dataset="${dataset#"${dataset%%[![:space:]]*}"}"
    dataset="${dataset%"${dataset##*[![:space:]]}"}"
    test -n "$dataset" && args+=(--dataset "$dataset")
  done
fi
if [ "$DRY_RUN" = "1" ]; then
  args+=(--dry-run)
fi
if [ "$OVERWRITE" = "1" ]; then
  args+=(--overwrite)
fi

echo "[audio-cvr-clips] root=$ROOT output_root=$OUTPUT_ROOT clip_seconds=$CLIP_SECONDS min=$MIN_CLIP_SECONDS max=$MAX_CLIP_SECONDS datasets=${DATASETS:-all}"
"${args[@]}"
