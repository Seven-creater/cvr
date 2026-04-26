#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/vace_masked_visual_batch}
VIDEO_EDIT_PLAN=${VIDEO_EDIT_PLAN:-$RUN_ROOT/video_edit_plan.jsonl}
MASK_MANIFEST=${MASK_MANIFEST:-$RUN_ROOT/video_mask_manifest.jsonl}

usage() {
  cat <<'EOF'
Usage: run_vace_masked_visual_batch_from_plan.sh [options]

Options:
  --data-root PATH
  --run-root PATH
  --video-edit-plan PATH
  --mask-manifest PATH
  plus any option accepted by run_vace_visual_batch_from_plan.sh

Runs VACE only for plans that already have a generated mask manifest.
This script does not start Omni and does not generate masks by itself.
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --video-edit-plan) VIDEO_EDIT_PLAN="$2"; shift 2 ;;
    --mask-manifest) MASK_MANIFEST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -s "$VIDEO_EDIT_PLAN" ]]; then
  echo "[vace-masked-batch] missing video edit plan: $VIDEO_EDIT_PLAN" >&2
  exit 1
fi
if [[ ! -s "$MASK_MANIFEST" ]]; then
  echo "[vace-masked-batch] missing mask manifest: $MASK_MANIFEST" >&2
  exit 1
fi

echo "[vace-masked-batch] run_root=$RUN_ROOT"
echo "[vace-masked-batch] video_edit_plan=$VIDEO_EDIT_PLAN"
echo "[vace-masked-batch] mask_manifest=$MASK_MANIFEST"

scripts/run_vace_visual_batch_from_plan.sh \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --video-edit-plan "$VIDEO_EDIT_PLAN" \
  --mask-manifest "$MASK_MANIFEST" \
  "${EXTRA_ARGS[@]}"
