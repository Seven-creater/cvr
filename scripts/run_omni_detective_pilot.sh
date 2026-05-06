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
ANNOTATION_PASS_TIMEOUT_SECONDS=${ANNOTATION_PASS_TIMEOUT_SECONDS:-900}
PROPOSE_TIMEOUT_SECONDS=${PROPOSE_TIMEOUT_SECONDS:-900}
PAIR_REQUEST_TIMEOUT_SECONDS=${PAIR_REQUEST_TIMEOUT_SECONDS:-90}
START_STAGE=${START_STAGE:-plan}
ALLOW_PARTIAL_ANNOTATIONS=${ALLOW_PARTIAL_ANNOTATIONS:-0}
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
  --annotation-pass-timeout-seconds N
  --propose-timeout-seconds N
  --pair-request-timeout-seconds N
  --start-stage plan|extract|annotate|propose|validate|review
  --allow-partial-annotations
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
    --annotation-pass-timeout-seconds)
      ANNOTATION_PASS_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --propose-timeout-seconds)
      PROPOSE_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --pair-request-timeout-seconds)
      PAIR_REQUEST_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --start-stage)
      START_STAGE="$2"
      shift 2
      ;;
    --allow-partial-annotations)
      ALLOW_PARTIAL_ANNOTATIONS=1
      shift
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

jsonl_row_count() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo 0
    return
  fi
  grep -cve '^[[:space:]]*$' "$path" || true
}

jsonl_unique_clip_count() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo 0
    return
  fi
  python - "$path" <<'PY'
import json
import sys
from pathlib import Path

seen = set()
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        clip_id = str(json.loads(line).get("clip_id", "")).strip()
    except json.JSONDecodeError:
        continue
    if clip_id:
        seen.add(clip_id)
print(len(seen))
PY
}

run_with_timeout() {
  local label="$1"
  local timeout_seconds="$2"
  shift 2
  "$@" &
  local child_pid=$!
  local elapsed=0
  while kill -0 "$child_pid" 2>/dev/null; do
    if [ "$elapsed" -ge "$timeout_seconds" ]; then
      echo "[omni-detective] ERROR: $label timed out after ${timeout_seconds}s; killing child pid=$child_pid" >&2
      kill "$child_pid" 2>/dev/null || true
      sleep 2
      kill -9 "$child_pid" 2>/dev/null || true
      wait "$child_pid" 2>/dev/null || true
      return 124
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  wait "$child_pid"
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
echo "[omni-detective] max_source_videos=$MAX_SOURCE_VIDEOS segment_seconds=$SEGMENT_SECONDS concurrency=$CONCURRENCY max_accepted_pairs=$MAX_ACCEPTED_PAIRS max_proposals=$MAX_PROPOSALS annotation_max_passes=$ANNOTATION_MAX_PASSES annotation_pass_timeout_seconds=$ANNOTATION_PASS_TIMEOUT_SECONDS propose_timeout_seconds=$PROPOSE_TIMEOUT_SECONDS pair_request_timeout_seconds=$PAIR_REQUEST_TIMEOUT_SECONDS allow_partial_annotations=$ALLOW_PARTIAL_ANNOTATIONS"
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
  ANNOTATION_TARGET_COUNT=$(jsonl_row_count "$RUN_ROOT/extracted_event_clips.jsonl")
  ANNOTATION_DONE_COUNT=$(jsonl_unique_clip_count "$RUN_ROOT/detective_annotations.jsonl")
  ANNOTATION_ROW_COUNT=$(jsonl_row_count "$RUN_ROOT/detective_annotations.jsonl")
  if [ "$ANNOTATION_DONE_COUNT" -ge "$ANNOTATION_TARGET_COUNT" ]; then
    echo "[omni-detective] annotation already complete unique_done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT rows=$ANNOTATION_ROW_COUNT"
  else
    ANNOTATION_PASS=1
    while [ "$ANNOTATION_PASS" -le "$ANNOTATION_MAX_PASSES" ]; do
      echo "[omni-detective] annotation pass $ANNOTATION_PASS/$ANNOTATION_MAX_PASSES target_clips=$ANNOTATION_TARGET_COUNT $(date)"
      set +e
      run_with_timeout "detective-annotate-clips pass $ANNOTATION_PASS" "$ANNOTATION_PASS_TIMEOUT_SECONDS" \
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
      ANNOTATION_DONE_COUNT=$(jsonl_unique_clip_count "$RUN_ROOT/detective_annotations.jsonl")
      ANNOTATION_ROW_COUNT=$(jsonl_row_count "$RUN_ROOT/detective_annotations.jsonl")
      echo "[omni-detective] annotation pass $ANNOTATION_PASS exit=$ANNOTATION_STATUS unique_done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT rows=$ANNOTATION_ROW_COUNT"
      if [ "$ANNOTATION_STATUS" -eq 124 ]; then
        echo "[omni-detective] annotation pass timed out; report this status immediately instead of waiting" >&2
        exit 124
      fi
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

ANNOTATION_TARGET_COUNT=$(jsonl_row_count "$RUN_ROOT/extracted_event_clips.jsonl")
ANNOTATION_DONE_COUNT=$(jsonl_unique_clip_count "$RUN_ROOT/detective_annotations.jsonl")
ANNOTATION_ROW_COUNT=$(jsonl_row_count "$RUN_ROOT/detective_annotations.jsonl")
echo "[omni-detective] annotation coverage unique_done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT rows=$ANNOTATION_ROW_COUNT"
if [ "$ANNOTATION_DONE_COUNT" -lt "$ANNOTATION_TARGET_COUNT" ] && [ "$ALLOW_PARTIAL_ANNOTATIONS" != "1" ]; then
  echo "[omni-detective] annotation incomplete by unique clip_id count; rerun with --start-stage annotate before propose, or pass --allow-partial-annotations for diagnostics" >&2
  exit 3
fi

if stage_enabled "propose"; then
  set +e
  run_with_timeout "propose-group-pairs" "$PROPOSE_TIMEOUT_SECONDS" \
  python -m app.composed_data propose-group-pairs \
    --root "$ROOT" \
    --clip-annotations-path "$RUN_ROOT/detective_annotations.jsonl" \
    --clip-groups-path "$RUN_ROOT/clip_groups.jsonl" \
    --output-path "$RUN_ROOT/judged_pair_proposals.jsonl" \
    --accepted-output-path "$RUN_ROOT/accepted_pairs.jsonl" \
    --base-url "$BASE_URL" \
    --api-key EMPTY \
    --model "$MODEL" \
    --timeout-seconds "$PAIR_REQUEST_TIMEOUT_SECONDS" \
    --max-accepted-pairs "$MAX_ACCEPTED_PAIRS" \
    --max-proposals "$MAX_PROPOSALS" \
    --overwrite
  PROPOSE_STATUS=$?
  set -e
  PROPOSAL_ROW_COUNT=$(jsonl_row_count "$RUN_ROOT/judged_pair_proposals.jsonl")
  ACCEPTED_ROW_COUNT=$(jsonl_row_count "$RUN_ROOT/accepted_pairs.jsonl")
  echo "[omni-detective] propose exit=$PROPOSE_STATUS judged_rows=$PROPOSAL_ROW_COUNT accepted_rows=$ACCEPTED_ROW_COUNT"
  if [ "$PROPOSE_STATUS" -ne 0 ]; then
    echo "[omni-detective] propose failed or timed out; report this status immediately instead of waiting" >&2
    exit "$PROPOSE_STATUS"
  fi

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
