#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/synthetic_video_edit_validation}
PAIRS_PATH=${PAIRS_PATH:-$RUN_ROOT/accepted_synthetic_pairs.jsonl}
CLIP_ANNOTATIONS=${CLIP_ANNOTATIONS:-$RUN_ROOT/detective_annotations.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/manual_review_bundle}
LIMIT=${LIMIT:-}
COPY_VIDEOS=${COPY_VIDEOS:-1}

usage() {
  cat <<'EOF'
Usage: build_synthetic_manual_review_bundle.sh [options]

Options:
  --root PATH
  --run-root PATH
  --pairs-path PATH
  --clip-annotations PATH
  --output-dir PATH
  --limit N
  --copy-videos 0|1
  -h, --help

Builds a manual review folder with one subdirectory per pair:
reference.mp4, target.mp4, review.md, metadata.json, optional src_ref_images,
and optional mask.mp4.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --pairs-path) PAIRS_PATH="$2"; shift 2 ;;
    --clip-annotations) CLIP_ANNOTATIONS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --copy-videos) COPY_VIDEOS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[manual-review-bundle] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ARGS=(
  -m app.composed_data build-review-bundle
  --root "$ROOT"
  --pairs-path "$PAIRS_PATH"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$CLIP_ANNOTATIONS" && -s "$CLIP_ANNOTATIONS" ]]; then
  ARGS+=(--clip-annotations-path "$CLIP_ANNOTATIONS")
fi
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "$COPY_VIDEOS" != "1" ]]; then
  ARGS+=(--no-copy-videos)
fi

echo "[manual-review-bundle] root=$ROOT"
echo "[manual-review-bundle] pairs=$PAIRS_PATH"
echo "[manual-review-bundle] clip_annotations=$CLIP_ANNOTATIONS"
echo "[manual-review-bundle] output_dir=$OUTPUT_DIR"
python "${ARGS[@]}"
