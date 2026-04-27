#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT=${DATA_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT:-$REPO_ROOT/runs/omni_detective_prompt_gate_fix_20260424}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/masked_vace_pipeline_queue_$(date +%Y%m%d)}

PAIR_CANDIDATES=${PAIR_CANDIDATES:-$SOURCE_RUN_ROOT/judged_pair_proposals.jsonl}
CLIP_ANNOTATIONS=${CLIP_ANNOTATIONS:-$SOURCE_RUN_ROOT/detective_annotations.jsonl}
VIDEO_EDIT_PLAN=${VIDEO_EDIT_PLAN:-$RUN_ROOT/video_edit_plan.jsonl}
VIDEO_EDIT_PLANNER_CACHE=${VIDEO_EDIT_PLANNER_CACHE:-$RUN_ROOT/video_edit_planner_cache.jsonl}
VIDEO_MASK_PLAN=${VIDEO_MASK_PLAN:-$RUN_ROOT/video_mask_plan.jsonl}
VIDEO_MASK_MANIFEST=${VIDEO_MASK_MANIFEST:-$RUN_ROOT/video_mask_manifest.jsonl}
GENERATED_MASK_MANIFEST=${GENERATED_MASK_MANIFEST:-$RUN_ROOT/video_mask_manifest.generated.jsonl}
VACE_RUN_ROOT=${VACE_RUN_ROOT:-$RUN_ROOT/vace_batch}
TARGET_ANNOTATIONS=${TARGET_ANNOTATIONS:-$RUN_ROOT/synthetic_target_annotations.jsonl}
ALL_ANNOTATIONS=${ALL_ANNOTATIONS:-$RUN_ROOT/synthetic_all_annotations.jsonl}
VALIDATION_RUN_ROOT=${VALIDATION_RUN_ROOT:-$RUN_ROOT/validation}
REVIEW_BUNDLE=${REVIEW_BUNDLE:-$RUN_ROOT/manual_review_bundle}

BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
API_KEY=${API_KEY:-EMPTY}
MODEL=${MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}
MAX_PLANS=${MAX_PLANS:-30}
MAX_MASKS=${MAX_MASKS:-}
VACE_TOP_K=${VACE_TOP_K:-5}
MAX_ACCEPTED_PAIRS=${MAX_ACCEPTED_PAIRS:-20}
PLANNING_MODE=${PLANNING_MODE:-production}

MASK_GPU_IDS=${MASK_GPU_IDS:-6}
VACE_GPU_IDS=${VACE_GPU_IDS:-2,3,4,5}
VACE_MAX_GPUS=${VACE_MAX_GPUS:-4}
VACE_ULYSSES_SIZE=${VACE_ULYSSES_SIZE:-4}
VACE_RING_SIZE=${VACE_RING_SIZE:-0}
WAN_CKPT=${WAN_CKPT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/Wan2.1/Wan2.1-VACE-14B}
VACE_TASK=${VACE_TASK:-vace-14B}
VACE_CONDA_ENV=${VACE_CONDA_ENV:-wan_vace}
GROUNDED_SAM2_CONDA_ENV=${GROUNDED_SAM2_CONDA_ENV:-grounded_sam2}
GROUNDER=${GROUNDER:-florence2}
STAGE=${STAGE:-all}

usage() {
  cat <<'EOF'
Usage: run_masked_vace_pipeline_queue.sh [options]

Stages:
  plan      Run Omni prompt planning and mask planning.
  mask      Generate video masks from video_mask_plan.jsonl.
  vace      Run masked VACE batch generation.
  annotate  Annotate generated target videos with existing Omni service.
  validate  Validate generated known pairs with existing Omni service.
  bundle    Build manual review bundle from accepted pairs.
  all       Run plan, mask, vace, annotate, validate, bundle sequentially.

Options:
  --stage STAGE
  --data-root PATH
  --source-run-root PATH
  --run-root PATH
  --pair-candidates PATH
  --clip-annotations PATH
  --base-url URL
  --api-key KEY
  --model MODEL_ID
  --max-plans N
  --max-masks N
  --vace-top-k N
  --planning-mode production|exploration
  --mask-gpu-ids IDS
  --vace-gpu-ids IDS
  --timeout-seconds N
  -h, --help

This is a queue orchestrator. It never starts or stops Omni.
Recommended resource split:
  Omni service: GPU 0,1
  Grounded-SAM2/FLORENCE/SAM2 masks: GPU 6
  VACE-14B: GPU 2,3,4,5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --source-run-root) SOURCE_RUN_ROOT="$2"; PAIR_CANDIDATES="$2/judged_pair_proposals.jsonl"; CLIP_ANNOTATIONS="$2/detective_annotations.jsonl"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --pair-candidates) PAIR_CANDIDATES="$2"; shift 2 ;;
    --clip-annotations) CLIP_ANNOTATIONS="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-plans) MAX_PLANS="$2"; shift 2 ;;
    --max-masks) MAX_MASKS="$2"; shift 2 ;;
    --vace-top-k) VACE_TOP_K="$2"; shift 2 ;;
    --planning-mode) PLANNING_MODE="$2"; shift 2 ;;
    --mask-gpu-ids) MASK_GPU_IDS="$2"; shift 2 ;;
    --vace-gpu-ids) VACE_GPU_IDS="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[masked-vace-queue] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Recompute derived paths after CLI parsing so --run-root is authoritative.
VIDEO_EDIT_PLAN="$RUN_ROOT/video_edit_plan.jsonl"
VIDEO_EDIT_PLANNER_CACHE="$RUN_ROOT/video_edit_planner_cache.jsonl"
VIDEO_MASK_PLAN="$RUN_ROOT/video_mask_plan.jsonl"
VIDEO_MASK_MANIFEST="$RUN_ROOT/video_mask_manifest.jsonl"
GENERATED_MASK_MANIFEST="$RUN_ROOT/video_mask_manifest.generated.jsonl"
VACE_RUN_ROOT="$RUN_ROOT/vace_batch"
TARGET_ANNOTATIONS="$RUN_ROOT/synthetic_target_annotations.jsonl"
ALL_ANNOTATIONS="$RUN_ROOT/synthetic_all_annotations.jsonl"
VALIDATION_RUN_ROOT="$RUN_ROOT/validation"
REVIEW_BUNDLE="$RUN_ROOT/manual_review_bundle"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$VALIDATION_RUN_ROOT"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "[masked-vace-queue] missing $label: $path" >&2
    exit 1
  fi
}

run_plan() {
  require_file "$PAIR_CANDIDATES" "pair candidates"
  require_file "$CLIP_ANNOTATIONS" "clip annotations"
  echo "[masked-vace-queue] stage=plan run_root=$RUN_ROOT"
  python -m app.composed_data plan-video-edits \
    --root "$DATA_ROOT" \
    --pair-candidates-path "$PAIR_CANDIDATES" \
    --clip-annotations-path "$CLIP_ANNOTATIONS" \
    --output-path "$VIDEO_EDIT_PLAN" \
    --max-plans "$MAX_PLANS" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --planning-mode "$PLANNING_MODE" \
    --planner-cache-path "$VIDEO_EDIT_PLANNER_CACHE" \
    | tee "$RUN_ROOT/logs/plan_video_edits.json"

  local mask_limit_args=()
  if [[ -n "$MAX_MASKS" ]]; then
    mask_limit_args=(--max-masks "$MAX_MASKS")
  fi
  python -m app.composed_data plan-video-masks \
    --root "$DATA_ROOT" \
    --video-edit-plan-path "$VIDEO_EDIT_PLAN" \
    --output-path "$VIDEO_MASK_PLAN" \
    --mask-manifest-path "$VIDEO_MASK_MANIFEST" \
    "${mask_limit_args[@]}" \
    | tee "$RUN_ROOT/logs/plan_video_masks.json"
}

run_mask() {
  require_file "$VIDEO_MASK_PLAN" "video mask plan"
  require_file "$VIDEO_MASK_MANIFEST" "video mask manifest"
  echo "[masked-vace-queue] stage=mask gpu_ids=$MASK_GPU_IDS"
  local mask_limit_args=()
  if [[ -n "$MAX_MASKS" ]]; then
    mask_limit_args=(--max-masks "$MAX_MASKS")
  fi
  scripts/run_grounded_sam2_video_masks.sh \
    --data-root "$DATA_ROOT" \
    --run-root "$RUN_ROOT" \
    --mask-plan "$VIDEO_MASK_PLAN" \
    --mask-manifest "$VIDEO_MASK_MANIFEST" \
    --output-manifest "$GENERATED_MASK_MANIFEST" \
    --report-path "$RUN_ROOT/grounded_sam2_mask_report.md" \
    --grounder "$GROUNDER" \
    --conda-env "$GROUNDED_SAM2_CONDA_ENV" \
    --gpu-ids "$MASK_GPU_IDS" \
    "${mask_limit_args[@]}"
}

run_vace() {
  require_file "$VIDEO_EDIT_PLAN" "video edit plan"
  require_file "$GENERATED_MASK_MANIFEST" "generated mask manifest"
  echo "[masked-vace-queue] stage=vace gpu_ids=$VACE_GPU_IDS top_k=$VACE_TOP_K"
  scripts/run_vace_masked_visual_batch_from_plan.sh \
    --data-root "$DATA_ROOT" \
    --run-root "$VACE_RUN_ROOT" \
    --video-edit-plan "$VIDEO_EDIT_PLAN" \
    --mask-manifest "$GENERATED_MASK_MANIFEST" \
    --top-k "$VACE_TOP_K" \
    --gpu-ids "$VACE_GPU_IDS" \
    --max-gpus "$VACE_MAX_GPUS" \
    --wan-ckpt "$WAN_CKPT" \
    --vace-task "$VACE_TASK" \
    --conda-env "$VACE_CONDA_ENV" \
    --use-torchrun 1 \
    --ulysses-size "$VACE_ULYSSES_SIZE" \
    --ring-size "$VACE_RING_SIZE"
}

run_annotate() {
  local target_manifest="$VACE_RUN_ROOT/synthetic_visual_target_manifest.jsonl"
  require_file "$target_manifest" "synthetic target manifest"
  echo "[masked-vace-queue] stage=annotate target_manifest=$target_manifest"
  python -m app.composed_data detective-annotate-clips \
    --root "$DATA_ROOT" \
    --clips-manifest-path "$target_manifest" \
    --output-path "$TARGET_ANNOTATIONS" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --overwrite \
    | tee "$RUN_ROOT/logs/annotate_targets.json"

  cat "$CLIP_ANNOTATIONS" "$TARGET_ANNOTATIONS" > "$ALL_ANNOTATIONS"
  echo "[masked-vace-queue] all_annotations=$ALL_ANNOTATIONS"
}

run_validate() {
  local known_pairs="$VACE_RUN_ROOT/synthetic_visual_candidate_pairs.jsonl"
  require_file "$known_pairs" "synthetic known pairs"
  require_file "$ALL_ANNOTATIONS" "combined annotations"
  echo "[masked-vace-queue] stage=validate known_pairs=$known_pairs"
  python -m app.composed_data validate-known-pairs \
    --root "$DATA_ROOT" \
    --known-pairs-path "$known_pairs" \
    --clip-annotations-path "$ALL_ANNOTATIONS" \
    --output-path "$VALIDATION_RUN_ROOT/judged_synthetic_pair_proposals.jsonl" \
    --accepted-output-path "$VALIDATION_RUN_ROOT/accepted_synthetic_pairs.jsonl" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --max-accepted-pairs "$MAX_ACCEPTED_PAIRS" \
    --overwrite \
    | tee "$RUN_ROOT/logs/validate_known_pairs.json"

  if [[ -s "$VALIDATION_RUN_ROOT/accepted_synthetic_pairs.jsonl" ]]; then
    python -m app.composed_data validate-pilot \
      --root "$DATA_ROOT" \
      --pilot-jsonl-path "$VALIDATION_RUN_ROOT/accepted_synthetic_pairs.jsonl" \
      --gallery-output-path "$VALIDATION_RUN_ROOT/synthetic_gallery.jsonl" \
      --report-output-path "$VALIDATION_RUN_ROOT/synthetic_pilot_review.md" \
      | tee "$RUN_ROOT/logs/validate_pilot.json"
  else
    echo "[masked-vace-queue] accepted_synthetic_pairs.jsonl is empty; skip validate-pilot"
  fi
}

run_bundle() {
  require_file "$VALIDATION_RUN_ROOT/accepted_synthetic_pairs.jsonl" "accepted synthetic pairs"
  echo "[masked-vace-queue] stage=bundle output=$REVIEW_BUNDLE"
  scripts/build_synthetic_manual_review_bundle.sh \
    --root "$DATA_ROOT" \
    --pairs-path "$VALIDATION_RUN_ROOT/accepted_synthetic_pairs.jsonl" \
    --clip-annotations "$ALL_ANNOTATIONS" \
    --output-dir "$REVIEW_BUNDLE" \
    --copy-videos 1
}

echo "[masked-vace-queue] stage=$STAGE"
echo "[masked-vace-queue] run_root=$RUN_ROOT"
echo "[masked-vace-queue] source_run_root=$SOURCE_RUN_ROOT"
echo "[masked-vace-queue] pair_candidates=$PAIR_CANDIDATES"
echo "[masked-vace-queue] clip_annotations=$CLIP_ANNOTATIONS"
echo "[masked-vace-queue] note: this script never starts or stops Omni"

case "$STAGE" in
  plan) run_plan ;;
  mask) run_mask ;;
  vace) run_vace ;;
  annotate) run_annotate ;;
  validate) run_validate ;;
  bundle) run_bundle ;;
  all)
    run_plan
    run_mask
    run_vace
    run_annotate
    run_validate
    run_bundle
    ;;
  *) echo "[masked-vace-queue] unknown stage: $STAGE" >&2; usage >&2; exit 2 ;;
esac

echo "[masked-vace-queue] done stage=$STAGE $(date)"
