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
STABLE_CLIP_PLAN=${STABLE_CLIP_PLAN:-$RUN_ROOT/omni_stable_clip_plan.jsonl}
STABLE_CLIP_MANIFEST=${STABLE_CLIP_MANIFEST:-$RUN_ROOT/omni_stable_clips.jsonl}
STABLE_CLIP_SELECTION_CACHE=${STABLE_CLIP_SELECTION_CACHE:-$RUN_ROOT/omni_stable_clip_selection_cache.jsonl}
STABLE_CLIP_ANNOTATIONS=${STABLE_CLIP_ANNOTATIONS:-$RUN_ROOT/omni_stable_clip_annotations.jsonl}
REFERENCE_UNDERSTANDING_CACHE=${REFERENCE_UNDERSTANDING_CACHE:-$RUN_ROOT/reference_understanding_cache.jsonl}
SRC_REF_IMAGE_PLAN=${SRC_REF_IMAGE_PLAN:-$RUN_ROOT/src_ref_image_plan.jsonl}
SRC_REF_IMAGE_SELECTION=${SRC_REF_IMAGE_SELECTION:-$RUN_ROOT/src_ref_image_selection.jsonl}
SRC_REF_IMAGE_GENERATION_MANIFEST=${SRC_REF_IMAGE_GENERATION_MANIFEST:-$RUN_ROOT/src_ref_image_generation_manifest.jsonl}
SRC_REF_IMAGE_ROOT=${SRC_REF_IMAGE_ROOT:-$RUN_ROOT/src_ref_images}
SRC_REF_IMAGE_MODEL_DIR=${SRC_REF_IMAGE_MODEL_DIR:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/ImageGen/Qwen-Image-2512}
IMAGE_GEN_CONDA_ENV=${IMAGE_GEN_CONDA_ENV:-omni_src}
IMAGE_GEN_GPU_IDS=${IMAGE_GEN_GPU_IDS:-6}
IMAGE_GEN_DEVICE_MAP=${IMAGE_GEN_DEVICE_MAP:-}
IMAGE_GEN_LOW_CPU_MEM_USAGE=${IMAGE_GEN_LOW_CPU_MEM_USAGE:-0}
VACE_RUN_ROOT=${VACE_RUN_ROOT:-$RUN_ROOT/vace_batch}
TARGET_ANNOTATIONS=${TARGET_ANNOTATIONS:-$RUN_ROOT/synthetic_target_annotations.jsonl}
ALL_ANNOTATIONS=${ALL_ANNOTATIONS:-$RUN_ROOT/synthetic_all_annotations.jsonl}
VALIDATION_RUN_ROOT=${VALIDATION_RUN_ROOT:-$RUN_ROOT/validation}
REVIEW_BUNDLE=${REVIEW_BUNDLE:-$RUN_ROOT/manual_review_bundle}

BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
API_KEY=${API_KEY:-EMPTY}
MODEL=${MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}
ANNOTATE_CONCURRENCY=${ANNOTATE_CONCURRENCY:-1}
MAX_PLANS=${MAX_PLANS:-30}
MAX_MASKS=${MAX_MASKS:-}
VACE_TOP_K=${VACE_TOP_K:-5}
MAX_ACCEPTED_PAIRS=${MAX_ACCEPTED_PAIRS:-20}
PLANNING_MODE=${PLANNING_MODE:-production}
MAX_SOURCE_VIDEOS=${MAX_SOURCE_VIDEOS:-50}
MIN_STABLE_CLIP_SECONDS=${MIN_STABLE_CLIP_SECONDS:-5}
MAX_STABLE_CLIP_SECONDS=${MAX_STABLE_CLIP_SECONDS:-8}
SRC_REF_NUM_CANDIDATES=${SRC_REF_NUM_CANDIDATES:-4}

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
  select-clips  Ask Omni to select stable 5-8 second windows from raw assets.
  extract-clips  Extract the selected windows.
  understand  Annotate selected stable clips and write reference understanding cache.
  plan      Run Omni prompt planning and mask planning.
  refplan   Plan src_ref_images requirements for VACE plans.
  refgen    Generate src_ref_images from src_ref_image_plan.jsonl.
  refselect Select generated src_ref_images from candidate folders.
  mask      Generate video masks from video_mask_plan.jsonl.
  vace      Run masked VACE batch generation.
  annotate  Annotate generated target videos with existing Omni service.
  validate  Validate generated known pairs with existing Omni service.
  bundle    Build manual review bundle from accepted pairs.
  all       Run plan, mask, vace, annotate, validate, bundle sequentially.
  overnight Run select-clips, extract-clips, understand, plan, refplan.

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
  --max-source-videos N
  --max-masks N
  --vace-top-k N
  --planning-mode production|exploration
  --image-gpu-ids IDS
  --image-device-map balanced|auto
  --mask-gpu-ids IDS
  --vace-gpu-ids IDS
  --timeout-seconds N
  --annotate-concurrency N
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
    --max-source-videos) MAX_SOURCE_VIDEOS="$2"; shift 2 ;;
    --max-masks) MAX_MASKS="$2"; shift 2 ;;
    --vace-top-k) VACE_TOP_K="$2"; shift 2 ;;
    --planning-mode) PLANNING_MODE="$2"; shift 2 ;;
    --image-gpu-ids) IMAGE_GEN_GPU_IDS="$2"; shift 2 ;;
    --image-device-map) IMAGE_GEN_DEVICE_MAP="$2"; shift 2 ;;
    --mask-gpu-ids) MASK_GPU_IDS="$2"; shift 2 ;;
    --vace-gpu-ids) VACE_GPU_IDS="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --annotate-concurrency) ANNOTATE_CONCURRENCY="$2"; shift 2 ;;
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
STABLE_CLIP_PLAN="$RUN_ROOT/omni_stable_clip_plan.jsonl"
STABLE_CLIP_MANIFEST="$RUN_ROOT/omni_stable_clips.jsonl"
STABLE_CLIP_SELECTION_CACHE="$RUN_ROOT/omni_stable_clip_selection_cache.jsonl"
STABLE_CLIP_ANNOTATIONS="$RUN_ROOT/omni_stable_clip_annotations.jsonl"
REFERENCE_UNDERSTANDING_CACHE="$RUN_ROOT/reference_understanding_cache.jsonl"
SRC_REF_IMAGE_PLAN="$RUN_ROOT/src_ref_image_plan.jsonl"
SRC_REF_IMAGE_SELECTION="$RUN_ROOT/src_ref_image_selection.jsonl"
SRC_REF_IMAGE_GENERATION_MANIFEST="$RUN_ROOT/src_ref_image_generation_manifest.jsonl"
SRC_REF_IMAGE_ROOT="$RUN_ROOT/src_ref_images"
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

run_select_clips() {
  echo "[masked-vace-queue] stage=select-clips max_source_videos=$MAX_SOURCE_VIDEOS"
  python -m app.composed_data plan-stable-omni-clips \
    --root "$DATA_ROOT" \
    --output-path "$STABLE_CLIP_PLAN" \
    --cache-path "$STABLE_CLIP_SELECTION_CACHE" \
    --max-source-videos "$MAX_SOURCE_VIDEOS" \
    --min-clip-seconds "$MIN_STABLE_CLIP_SECONDS" \
    --max-clip-seconds "$MAX_STABLE_CLIP_SECONDS" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    | tee "$RUN_ROOT/logs/plan_stable_omni_clips.json"
}

run_extract_clips() {
  require_file "$STABLE_CLIP_PLAN" "stable clip plan"
  echo "[masked-vace-queue] stage=extract-clips"
  python -m app.composed_data extract-clips \
    --root "$DATA_ROOT" \
    --plan-path "$STABLE_CLIP_PLAN" \
    --output-manifest-path "$STABLE_CLIP_MANIFEST" \
    --overwrite \
    | tee "$RUN_ROOT/logs/extract_stable_clips.json"
}

run_understand() {
  require_file "$STABLE_CLIP_MANIFEST" "stable clip manifest"
  echo "[masked-vace-queue] stage=understand"
  python -m app.composed_data detective-annotate-clips \
    --root "$DATA_ROOT" \
    --clips-manifest-path "$STABLE_CLIP_MANIFEST" \
    --output-path "$STABLE_CLIP_ANNOTATIONS" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --concurrency "$ANNOTATE_CONCURRENCY" \
    --overwrite \
    | tee "$RUN_ROOT/logs/annotate_stable_clips.json"
  python -m app.composed_data cache-reference-understandings \
    --root "$DATA_ROOT" \
    --clip-annotations-path "$STABLE_CLIP_ANNOTATIONS" \
    --output-path "$REFERENCE_UNDERSTANDING_CACHE" \
    | tee "$RUN_ROOT/logs/cache_reference_understandings.json"
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

run_refplan() {
  require_file "$VIDEO_EDIT_PLAN" "video edit plan"
  echo "[masked-vace-queue] stage=refplan"
  python -m app.composed_data plan-src-ref-images \
    --root "$DATA_ROOT" \
    --video-edit-plan-path "$VIDEO_EDIT_PLAN" \
    --output-path "$SRC_REF_IMAGE_PLAN" \
    --image-root "$SRC_REF_IMAGE_ROOT" \
    --num-candidates "$SRC_REF_NUM_CANDIDATES" \
    | tee "$RUN_ROOT/logs/plan_src_ref_images.json"
}

run_refselect() {
  if [[ ! -s "$SRC_REF_IMAGE_PLAN" ]]; then
    echo "[masked-vace-queue] stage=refselect no src_ref image plan; writing empty selection"
    : > "$SRC_REF_IMAGE_SELECTION"
    return
  fi
  echo "[masked-vace-queue] stage=refselect"
  python -m app.composed_data select-src-ref-images \
    --root "$DATA_ROOT" \
    --src-ref-image-plan-path "$SRC_REF_IMAGE_PLAN" \
    --output-path "$SRC_REF_IMAGE_SELECTION" \
    | tee "$RUN_ROOT/logs/select_src_ref_images.json"
}

run_refgen() {
  if [[ ! -s "$SRC_REF_IMAGE_PLAN" ]]; then
    echo "[masked-vace-queue] stage=refgen no src_ref image plan; skip"
    return
  fi
  echo "[masked-vace-queue] stage=refgen model_dir=$SRC_REF_IMAGE_MODEL_DIR gpu_ids=$IMAGE_GEN_GPU_IDS device_map=${IMAGE_GEN_DEVICE_MAP:-none}"
  local image_device_map_args=()
  if [[ -n "$IMAGE_GEN_DEVICE_MAP" ]]; then
    image_device_map_args=(--device-map "$IMAGE_GEN_DEVICE_MAP" --low-cpu-mem-usage "$IMAGE_GEN_LOW_CPU_MEM_USAGE")
  fi
  scripts/run_src_ref_image_generation_from_plan.sh \
    --src-ref-image-plan "$SRC_REF_IMAGE_PLAN" \
    --model-dir "$SRC_REF_IMAGE_MODEL_DIR" \
    --output-manifest "$SRC_REF_IMAGE_GENERATION_MANIFEST" \
    --conda-env "$IMAGE_GEN_CONDA_ENV" \
    --gpu-ids "$IMAGE_GEN_GPU_IDS" \
    "${image_device_map_args[@]}" \
    | tee "$RUN_ROOT/logs/generate_src_ref_images.json"
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
  local src_ref_args=()
  if [[ -s "$SRC_REF_IMAGE_SELECTION" ]]; then
    src_ref_args=(--src-ref-selection "$SRC_REF_IMAGE_SELECTION")
  fi
  scripts/run_vace_masked_visual_batch_from_plan.sh \
    --data-root "$DATA_ROOT" \
    --run-root "$VACE_RUN_ROOT" \
    --video-edit-plan "$VIDEO_EDIT_PLAN" \
    --mask-manifest "$GENERATED_MASK_MANIFEST" \
    "${src_ref_args[@]}" \
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
    --concurrency "$ANNOTATE_CONCURRENCY" \
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
  select-clips) run_select_clips ;;
  extract-clips) run_extract_clips ;;
  understand) run_understand ;;
  plan) run_plan ;;
  refplan) run_refplan ;;
  refgen) run_refgen ;;
  refselect) run_refselect ;;
  mask) run_mask ;;
  vace) run_vace ;;
  annotate) run_annotate ;;
  validate) run_validate ;;
  bundle) run_bundle ;;
  all)
    run_plan
    run_refplan
    if [[ -s "$SRC_REF_IMAGE_PLAN" ]]; then
      run_refgen
    fi
    if [[ -s "$SRC_REF_IMAGE_SELECTION" ]]; then
      echo "[masked-vace-queue] using existing src_ref selection: $SRC_REF_IMAGE_SELECTION"
    else
      run_refselect
    fi
    run_mask
    run_vace
    run_annotate
    run_validate
    run_bundle
    ;;
  overnight)
    run_select_clips
    run_extract_clips
    run_understand
    run_plan
    run_refplan
    ;;
  *) echo "[masked-vace-queue] unknown stage: $STAGE" >&2; usage >&2; exit 2 ;;
esac

echo "[masked-vace-queue] done stage=$STAGE $(date)"
