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
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/clips/audio_cvr_6_9s}
CLIP_SECONDS=${CLIP_SECONDS:-8}
MIN_CLIP_SECONDS=${MIN_CLIP_SECONDS:-6}
MAX_CLIP_SECONDS=${MAX_CLIP_SECONDS:-9}
STRIDE_SECONDS=${STRIDE_SECONDS:-}
MIN_CLIPS_PER_SOURCE=${MIN_CLIPS_PER_SOURCE:-2}
MAX_CLIPS_PER_SOURCE=${MAX_CLIPS_PER_SOURCE:-0}
MAX_SOURCE_VIDEOS=${MAX_SOURCE_VIDEOS:-0}
MAX_SOURCE_VIDEOS_PER_DATASET=${MAX_SOURCE_VIDEOS_PER_DATASET:-0}
DATASETS=${DATASETS:-}
EXCLUDE_DATASETS=${EXCLUDE_DATASETS:-voxceleb_seed}
INCLUDE_TAIL_SEGMENT=${INCLUDE_TAIL_SEGMENT:-1}
DRY_RUN=${DRY_RUN:-0}
OVERWRITE=${OVERWRITE:-0}

usage() {
  cat <<'EOF'
Usage: build_audio_cvr_6_9s_clips.sh [options]

Build 6-9 second Audio-CVR clips, defaulting to 8s, into clips/audio_cvr_6_9s.
This script is for the large-scale B-line audio-primary CVR run.

Options:
  --root PATH
  --output-root PATH
  --dataset NAME[,NAME]
  --exclude-dataset NAME[,NAME] default: voxceleb_seed
  --clip-seconds N          default: 8
  --min-clip-seconds N      default: 6
  --max-clip-seconds N      default: 9
  --stride-seconds N        default: same as clip seconds
  --min-clips-per-source N  default: 2
  --max-clips-per-source N  default: 0 (no cap)
  --max-source-videos N     default: 0 (all)
  --max-source-videos-per-dataset N default: 0 (all)
  --include-tail-segment    default: enabled
  --no-include-tail-segment
  --dry-run
  --overwrite
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset) DATASETS="${DATASETS:+$DATASETS,}$2"; shift 2 ;;
    --exclude-dataset) EXCLUDE_DATASETS="${EXCLUDE_DATASETS:+$EXCLUDE_DATASETS,}$2"; shift 2 ;;
    --clip-seconds) CLIP_SECONDS="$2"; shift 2 ;;
    --min-clip-seconds) MIN_CLIP_SECONDS="$2"; shift 2 ;;
    --max-clip-seconds) MAX_CLIP_SECONDS="$2"; shift 2 ;;
    --stride-seconds) STRIDE_SECONDS="$2"; shift 2 ;;
    --min-clips-per-source) MIN_CLIPS_PER_SOURCE="$2"; shift 2 ;;
    --max-clips-per-source) MAX_CLIPS_PER_SOURCE="$2"; shift 2 ;;
    --max-source-videos) MAX_SOURCE_VIDEOS="$2"; shift 2 ;;
    --max-source-videos-per-dataset) MAX_SOURCE_VIDEOS_PER_DATASET="$2"; shift 2 ;;
    --include-tail-segment) INCLUDE_TAIL_SEGMENT=1; shift ;;
    --no-include-tail-segment) INCLUDE_TAIL_SEGMENT=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[audio-cvr-6-9s] unknown argument: $1" >&2; usage >&2; exit 2 ;;
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
  --dataset-video-root daily_omni=video
  --dataset-video-root worldsense=videos
  --dataset-video-root hdtf=videos
  --dataset-video-root avatar=.,video
  --dataset-video-root vggsound=scratch
  --dataset-video-root vgg_monoaudio=inter_class/mixed
  --dataset-video-root voxceleb=.
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
if [ -n "$EXCLUDE_DATASETS" ]; then
  IFS=',' read -r -a excluded_dataset_items <<< "$EXCLUDE_DATASETS"
  for dataset in "${excluded_dataset_items[@]}"; do
    dataset="${dataset#"${dataset%%[![:space:]]*}"}"
    dataset="${dataset%"${dataset##*[![:space:]]}"}"
    test -n "$dataset" && args+=(--exclude-dataset "$dataset")
  done
fi
if [ "$DRY_RUN" = "1" ]; then
  args+=(--dry-run)
fi
if [ "$INCLUDE_TAIL_SEGMENT" = "1" ]; then
  args+=(--include-tail-segment)
fi
if [ "$OVERWRITE" = "1" ]; then
  args+=(--overwrite)
fi

echo "[audio-cvr-6-9s] root=$ROOT output_root=$OUTPUT_ROOT clip_seconds=$CLIP_SECONDS min=$MIN_CLIP_SECONDS max=$MAX_CLIP_SECONDS include_tail_segment=$INCLUDE_TAIL_SEGMENT datasets=${DATASETS:-all} exclude_datasets=${EXCLUDE_DATASETS:-none}"
"${args[@]}"
