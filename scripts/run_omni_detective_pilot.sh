#!/usr/bin/env bash
set -euo pipefail

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/omni_detective_pilot}
MODEL=${MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
SOURCE_CLIPS=${SOURCE_CLIPS:-$ROOT/metadata/source_clips_all.jsonl}
MAX_SOURCE_VIDEOS=${MAX_SOURCE_VIDEOS:-80}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-8}
CONCURRENCY=${CONCURRENCY:-1}
MAX_ACCEPTED_PAIRS=${MAX_ACCEPTED_PAIRS:-10}
MAX_PROPOSALS=${MAX_PROPOSALS:-40}
ANNOTATION_MAX_PASSES=${ANNOTATION_MAX_PASSES:-3}
START_STAGE=${START_STAGE:-plan}
MODEL_STAGE=${MODEL_STAGE:-instruct}
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
MAX_GPUS=${MAX_GPUS:-6}

usage() {
  cat <<'EOF'
Usage: run_omni_detective_pilot.sh [options]

Options:
  --root PATH
  --run-root PATH
  --model PATH
  --base-url URL
  --source-clips PATH
  --max-source-videos N
  --segment-seconds N
  --concurrency N
  --max-accepted-pairs N
  --max-proposals N
  --annotation-max-passes N
  --start-stage plan|extract|annotate|propose|validate|review
  --model-stage VALUE
  --gpu-ids IDS
  --max-gpus N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --source-clips)
      SOURCE_CLIPS="$2"
      shift 2
      ;;
    --max-source-videos)
      MAX_SOURCE_VIDEOS="$2"
      shift 2
      ;;
    --segment-seconds)
      SEGMENT_SECONDS="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --max-accepted-pairs)
      MAX_ACCEPTED_PAIRS="$2"
      shift 2
      ;;
    --max-proposals)
      MAX_PROPOSALS="$2"
      shift 2
      ;;
    --annotation-max-passes)
      ANNOTATION_MAX_PASSES="$2"
      shift 2
      ;;
    --start-stage)
      START_STAGE="$2"
      shift 2
      ;;
    --model-stage)
      MODEL_STAGE="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --max-gpus)
      MAX_GPUS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[omni-detective] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

count_gpu_ids() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo 0
    return
  fi
  python - "$value" <<'PY'
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(items))
PY
}

stage_rank() {
  case "$1" in
    plan) echo 1 ;;
    extract) echo 2 ;;
    annotate) echo 3 ;;
    propose) echo 4 ;;
    validate) echo 5 ;;
    review) echo 6 ;;
    *)
      echo "[omni-detective] unsupported start stage: $1" >&2
      exit 2
      ;;
  esac
}

stage_enabled() {
  local stage="$1"
  [ "$(stage_rank "$stage")" -ge "$(stage_rank "$START_STAGE")" ]
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -s "$path" ]; then
    echo "[omni-detective] missing required $label: $path" >&2
    exit 2
  fi
}

GPU_COUNT=$(count_gpu_ids "$GPU_IDS")
if (( GPU_COUNT > MAX_GPUS )); then
  echo "[resource-policy] refusing to run with GPU_COUNT=$GPU_COUNT > MAX_GPUS=$MAX_GPUS" >&2
  exit 2
fi

case "$MODEL_STAGE" in
  instruct)
    ;;
  captioner|thinking)
    echo "[resource-policy] this script runs the Instruct pair-construction stage only; stop the current service and run a dedicated $MODEL_STAGE stage separately" >&2
    exit 2
    ;;
  *)
    echo "[resource-policy] unsupported MODEL_STAGE=$MODEL_STAGE; expected instruct, captioner, or thinking" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"

echo "[omni-detective] start $(date)"
echo "[omni-detective] root=$ROOT"
echo "[omni-detective] run_root=$RUN_ROOT"
echo "[omni-detective] source_clips=$SOURCE_CLIPS"
echo "[omni-detective] base_url=$BASE_URL"
echo "[omni-detective] model=$MODEL"
echo "[resource-policy] one Omni model per run; do not keep Captioner/Instruct/Thinking loaded together"
echo "[resource-policy] model_stage=$MODEL_STAGE gpu_ids=${GPU_IDS:-unset} gpu_count=$GPU_COUNT max_gpus=$MAX_GPUS"
echo "[omni-detective] start_stage=$START_STAGE"
echo "[omni-detective] max_source_videos=$MAX_SOURCE_VIDEOS segment_seconds=$SEGMENT_SECONDS concurrency=$CONCURRENCY max_accepted_pairs=$MAX_ACCEPTED_PAIRS max_proposals=$MAX_PROPOSALS annotation_max_passes=$ANNOTATION_MAX_PASSES"
curl -fsS "$BASE_URL/models"
echo

if stage_enabled "plan"; then
  python -m app.composed_data plan-detective-clips \
    --root "$ROOT" \
    --source-clips-path "$SOURCE_CLIPS" \
    --clip-plan-output-path "$RUN_ROOT/clip_plan_detective.jsonl" \
    --clip-groups-output-path "$RUN_ROOT/clip_groups.jsonl" \
    --max-source-videos "$MAX_SOURCE_VIDEOS" \
    --segment-seconds "$SEGMENT_SECONDS" \
    --min-clip-seconds 3 \
    --max-clip-seconds 15

  echo "[omni-detective] planning done $(date)"
else
  require_file "$RUN_ROOT/clip_plan_detective.jsonl" "clip plan"
  require_file "$RUN_ROOT/clip_groups.jsonl" "clip groups"
  echo "[omni-detective] skip planning start_stage=$START_STAGE"
fi

if stage_enabled "extract"; then
  python -m app.composed_data extract-clips \
    --root "$ROOT" \
    --plan-path "$RUN_ROOT/clip_plan_detective.jsonl" \
    --output-manifest-path "$RUN_ROOT/extracted_event_clips.jsonl" \
    --overwrite

  echo "[omni-detective] extraction done $(date)"
else
  require_file "$RUN_ROOT/extracted_event_clips.jsonl" "extracted clips manifest"
  echo "[omni-detective] skip extraction start_stage=$START_STAGE"
fi

if stage_enabled "annotate"; then
  ANNOTATION_TARGET_COUNT=$(grep -cve '^[[:space:]]*$' "$RUN_ROOT/extracted_event_clips.jsonl" || true)
  ANNOTATION_DONE_COUNT=0
  if [ -f "$RUN_ROOT/detective_annotations.jsonl" ]; then
    ANNOTATION_DONE_COUNT=$(grep -cve '^[[:space:]]*$' "$RUN_ROOT/detective_annotations.jsonl" || true)
  fi
  if [ "$ANNOTATION_DONE_COUNT" -ge "$ANNOTATION_TARGET_COUNT" ]; then
    echo "[omni-detective] annotation already complete done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT"
  else
    ANNOTATION_PASS=1
    while [ "$ANNOTATION_PASS" -le "$ANNOTATION_MAX_PASSES" ]; do
      echo "[omni-detective] annotation pass $ANNOTATION_PASS/$ANNOTATION_MAX_PASSES target_clips=$ANNOTATION_TARGET_COUNT $(date)"
      set +e
      python -m app.composed_data detective-annotate-clips \
        --root "$ROOT" \
        --clips-manifest-path "$RUN_ROOT/extracted_event_clips.jsonl" \
        --output-path "$RUN_ROOT/detective_annotations.jsonl" \
        --base-url "$BASE_URL" \
        --api-key EMPTY \
        --model "$MODEL" \
        --timeout-seconds 300 \
        --concurrency "$CONCURRENCY" \
        --overwrite
      ANNOTATION_STATUS=$?
      set -e
      ANNOTATION_DONE_COUNT=0
      if [ -f "$RUN_ROOT/detective_annotations.jsonl" ]; then
        ANNOTATION_DONE_COUNT=$(grep -cve '^[[:space:]]*$' "$RUN_ROOT/detective_annotations.jsonl" || true)
      fi
      echo "[omni-detective] annotation pass $ANNOTATION_PASS exit=$ANNOTATION_STATUS done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT"
      if [ "$ANNOTATION_DONE_COUNT" -ge "$ANNOTATION_TARGET_COUNT" ]; then
        break
      fi
      if [ "$ANNOTATION_PASS" -ge "$ANNOTATION_MAX_PASSES" ]; then
        echo "[omni-detective] annotation incomplete after $ANNOTATION_MAX_PASSES passes; inspect $RUN_ROOT/detective_annotations.jsonl and $RUN_ROOT/logs" >&2
        exit 3
      fi
      ANNOTATION_PASS=$((ANNOTATION_PASS + 1))
      sleep 10
    done
  fi
  echo "[omni-detective] annotation done $(date)"
else
  require_file "$RUN_ROOT/detective_annotations.jsonl" "detective annotations"
  echo "[omni-detective] skip annotation start_stage=$START_STAGE"
fi

if stage_enabled "propose"; then
  python -m app.composed_data propose-group-pairs \
    --root "$ROOT" \
    --clip-annotations-path "$RUN_ROOT/detective_annotations.jsonl" \
    --clip-groups-path "$RUN_ROOT/clip_groups.jsonl" \
    --output-path "$RUN_ROOT/judged_pair_proposals.jsonl" \
    --accepted-output-path "$RUN_ROOT/accepted_pairs.jsonl" \
    --base-url "$BASE_URL" \
    --api-key EMPTY \
    --model "$MODEL" \
    --timeout-seconds 300 \
    --max-accepted-pairs "$MAX_ACCEPTED_PAIRS" \
    --max-proposals "$MAX_PROPOSALS" \
    --overwrite

  echo "[omni-detective] group proposal and judge done $(date)"
else
  require_file "$RUN_ROOT/judged_pair_proposals.jsonl" "judged pair proposals"
  echo "[omni-detective] skip propose start_stage=$START_STAGE"
fi

if stage_enabled "validate"; then
  if [ -s "$RUN_ROOT/accepted_pairs.jsonl" ]; then
    python -m app.composed_data validate-pilot \
      --root "$ROOT" \
      --pilot-jsonl-path "$RUN_ROOT/accepted_pairs.jsonl" \
      --gallery-output-path "$RUN_ROOT/gallery.jsonl" \
      --report-output-path "$RUN_ROOT/pilot_review.md"
  else
    echo "[omni-detective] no accepted pairs; skip validate-pilot"
  fi
else
  echo "[omni-detective] skip validate start_stage=$START_STAGE"
fi

if stage_enabled "review"; then
  if [ -s "$RUN_ROOT/accepted_pairs.jsonl" ]; then
    python -m app.composed_data build-review-bundle \
      --root "$ROOT" \
      --pairs-path "$RUN_ROOT/accepted_pairs.jsonl" \
      --clip-annotations-path "$RUN_ROOT/detective_annotations.jsonl" \
      --output-dir "$RUN_ROOT/manual_review_bundle"
  else
    echo "[omni-detective] no accepted pairs; skip manual review bundle"
  fi
else
  echo "[omni-detective] skip review start_stage=$START_STAGE"
fi

echo "[verify] outputs"
ls -lh "$RUN_ROOT/clip_plan_detective.jsonl" || true
ls -lh "$RUN_ROOT/clip_groups.jsonl" || true
ls -lh "$RUN_ROOT/extracted_event_clips.jsonl" || true
ls -lh "$RUN_ROOT/detective_annotations.jsonl" || true
ls -lh "$RUN_ROOT/judged_pair_proposals.jsonl" || true
ls -lh "$RUN_ROOT/accepted_pairs.jsonl" || true
ls -lh "$RUN_ROOT/gallery.jsonl" || true
ls -ld "$RUN_ROOT/manual_review_bundle" || true
cat "$RUN_ROOT/pilot_review.md" || true

echo "[omni-detective] done $(date)"
