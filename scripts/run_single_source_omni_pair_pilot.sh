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
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/single_source_omni_pair_pilot}
MODEL=${MODEL:-qwen3-omni}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
SOURCE_CLIPS=${SOURCE_CLIPS:-$ROOT/metadata/source_clips_all.jsonl}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-5}
CONCURRENCY=${CONCURRENCY:-1}
MAX_ACCEPTED_PAIRS=${MAX_ACCEPTED_PAIRS:-5}
MAX_PROPOSALS=${MAX_PROPOSALS:-15}
ZERO_ACCEPTED_STOP_AFTER=${ZERO_ACCEPTED_STOP_AFTER:-10}
SELECT_TIMEOUT_SECONDS=${SELECT_TIMEOUT_SECONDS:-120}
SOURCE_SELECTION_TOP_K=${SOURCE_SELECTION_TOP_K:-3}
SOURCE_SELECTION_SCAN_LIMIT=${SOURCE_SELECTION_SCAN_LIMIT:-500}
SOURCE_SELECTION_MAX_ELIGIBLE=${SOURCE_SELECTION_MAX_ELIGIBLE:-24}
OMNI_SOURCE_SELECTION=${OMNI_SOURCE_SELECTION:-0}
ANNOTATION_TIMEOUT_SECONDS=${ANNOTATION_TIMEOUT_SECONDS:-240}
PROPOSE_TIMEOUT_SECONDS=${PROPOSE_TIMEOUT_SECONDS:-900}
PAIR_REQUEST_TIMEOUT_SECONDS=${PAIR_REQUEST_TIMEOUT_SECONDS:-90}
ACCEPTANCE_PROFILE=${ACCEPTANCE_PROFILE:-exploration}
START_STAGE=${START_STAGE:-select}

usage() {
  cat <<'EOF'
Usage: run_single_source_omni_pair_pilot.sh [options]

Options:
  --root PATH
  --run-root PATH
  --model NAME
  --base-url URL
  --source-clips PATH
  --segment-seconds N
  --concurrency N
  --max-accepted-pairs N
  --max-proposals N
  --zero-accepted-stop-after N
  --select-timeout-seconds N
  --source-selection-top-k N
  --source-selection-scan-limit N
  --source-selection-max-eligible N
  --omni-source-selection
  --annotation-timeout-seconds N
  --propose-timeout-seconds N
  --pair-request-timeout-seconds N
  --acceptance-profile exploration|final
  --start-stage select|plan|extract|annotate|mine|propose|validate|review
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --source-clips) SOURCE_CLIPS="$2"; shift 2 ;;
    --segment-seconds) SEGMENT_SECONDS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --max-accepted-pairs) MAX_ACCEPTED_PAIRS="$2"; shift 2 ;;
    --max-proposals) MAX_PROPOSALS="$2"; shift 2 ;;
    --zero-accepted-stop-after) ZERO_ACCEPTED_STOP_AFTER="$2"; shift 2 ;;
    --select-timeout-seconds) SELECT_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --source-selection-top-k) SOURCE_SELECTION_TOP_K="$2"; shift 2 ;;
    --source-selection-scan-limit) SOURCE_SELECTION_SCAN_LIMIT="$2"; shift 2 ;;
    --source-selection-max-eligible) SOURCE_SELECTION_MAX_ELIGIBLE="$2"; shift 2 ;;
    --omni-source-selection) OMNI_SOURCE_SELECTION=1; shift ;;
    --annotation-timeout-seconds) ANNOTATION_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --propose-timeout-seconds) PROPOSE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --pair-request-timeout-seconds) PAIR_REQUEST_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --acceptance-profile) ACCEPTANCE_PROFILE="$2"; shift 2 ;;
    --start-stage) START_STAGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[single-source-omni] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

stage_rank() {
  case "$1" in
    select) echo 1 ;;
    plan) echo 2 ;;
    extract) echo 3 ;;
    annotate) echo 4 ;;
    mine) echo 5 ;;
    propose) echo 6 ;;
    validate) echo 7 ;;
    review) echo 8 ;;
    *) echo "[single-source-omni] unsupported start stage: $1" >&2; exit 2 ;;
  esac
}

stage_enabled() {
  local stage="$1"
  [ "$(stage_rank "$stage")" -ge "$(stage_rank "$START_STAGE")" ]
}

jsonl_row_count() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo 0
    return
  fi
  grep -cve '^[[:space:]]*$' "$path" || true
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -s "$path" ]; then
    echo "[single-source-omni] missing required $label: $path" >&2
    exit 2
  fi
}

probe_omni_model() {
  local models_json
  models_json=$(curl -fsS "$BASE_URL/models")
  OMNI_MODELS_JSON="$models_json" python3 - "$MODEL" <<'PY'
import json
import os
import sys

wanted = sys.argv[1]
payload = json.loads(os.environ["OMNI_MODELS_JSON"])
served = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
print("[single-source-omni] served_models=" + ",".join(served))
if wanted not in served:
    raise SystemExit(f"[single-source-omni] model {wanted!r} is not served by {served}; use vLLM registered name, not checkpoint path")
PY
}

run_with_timeout() {
  local label="$1"
  local seconds="$2"
  shift 2
  echo "[single-source-omni] $label start $(date)"
  if timeout "$seconds" "$@"; then
    echo "[single-source-omni] $label done $(date)"
  else
    local status=$?
    echo "[single-source-omni] ERROR: $label failed or timed out status=$status after ${seconds}s" >&2
    exit "$status"
  fi
}

mkdir -p "$RUN_ROOT/logs"
echo "[single-source-omni] start $(date)"
echo "[single-source-omni] run_root=$RUN_ROOT start_stage=$START_STAGE model=$MODEL base_url=$BASE_URL acceptance_profile=$ACCEPTANCE_PROFILE"
probe_omni_model

SELECTED_SOURCE="$RUN_ROOT/selected_source_video.json"
SOURCE_CANDIDATES="$RUN_ROOT/selected_source_candidates.jsonl"
SOURCE_SELECTION_ANNOTATIONS="$RUN_ROOT/selected_source_selection_annotations.jsonl"
WHOLE_MANIFEST="$RUN_ROOT/selected_source_manifest.jsonl"
CLIP_PLAN="$RUN_ROOT/single_source_clip_plan.jsonl"
CLIP_GROUPS="$RUN_ROOT/single_source_clip_groups.jsonl"
SEGMENTS_MANIFEST="$RUN_ROOT/extracted_single_source_clips.jsonl"
WHOLE_ANNOTATION="$RUN_ROOT/single_source_whole_annotation.jsonl"
SEGMENT_ANNOTATIONS="$RUN_ROOT/single_source_annotations.jsonl"
PAIR_CANDIDATES="$RUN_ROOT/single_source_pair_candidates.jsonl"
PAIR_REPORT="$RUN_ROOT/single_source_pair_report.md"
RANKED_PAIRS="$RUN_ROOT/ranked_single_source_pairs.jsonl"
ACCEPTED_PAIRS="$RUN_ROOT/accepted_pairs.jsonl"
PILOT_REVIEW="$RUN_ROOT/single_source_pilot_review.md"
GALLERY="$RUN_ROOT/gallery.jsonl"
REVIEW_BUNDLE="$RUN_ROOT/single_source_review_bundle"

if stage_enabled select; then
  SELECT_CMD=(
    python3 -m app.composed_data select-single-source-video
    --root "$ROOT"
    --source-clips-path "$SOURCE_CLIPS"
    --output-path "$SELECTED_SOURCE"
    --candidates-output-path "$SOURCE_CANDIDATES"
    --selection-annotations-path "$SOURCE_SELECTION_ANNOTATIONS"
    --top-k "$SOURCE_SELECTION_TOP_K"
    --max-source-videos-scan "$SOURCE_SELECTION_SCAN_LIMIT"
    --max-eligible-candidates "$SOURCE_SELECTION_MAX_ELIGIBLE"
  )
  if [ "$OMNI_SOURCE_SELECTION" = "1" ]; then
    SELECT_CMD+=(
      --base-url "$BASE_URL"
      --api-key EMPTY
      --model "$MODEL"
      --timeout-seconds "$SELECT_TIMEOUT_SECONDS"
    )
  else
    echo "[single-source-omni] fast local source selection; Omni source selection disabled (pass --omni-source-selection to enable)"
  fi
  run_with_timeout "select-single-source-video" "$SELECT_TIMEOUT_SECONDS" \
    "${SELECT_CMD[@]}"
else
  echo "[single-source-omni] skip select start_stage=$START_STAGE"
fi

require_file "$SELECTED_SOURCE" "selected source"
if stage_enabled plan; then
  python3 -m app.composed_data plan-single-source-clips \
    --root "$ROOT" \
    --selected-source-path "$SELECTED_SOURCE" \
    --clip-plan-output-path "$CLIP_PLAN" \
    --clip-groups-output-path "$CLIP_GROUPS" \
    --whole-manifest-output-path "$WHOLE_MANIFEST" \
    --segment-seconds "$SEGMENT_SECONDS"
else
  echo "[single-source-omni] skip plan start_stage=$START_STAGE"
fi

require_file "$CLIP_PLAN" "single source clip plan"
require_file "$CLIP_GROUPS" "single source clip groups"
require_file "$WHOLE_MANIFEST" "whole source manifest"
if stage_enabled extract; then
  python3 -m app.composed_data extract-clips \
    --root "$ROOT" \
    --plan-path "$CLIP_PLAN" \
    --output-manifest-path "$SEGMENTS_MANIFEST" \
    --overwrite
else
  echo "[single-source-omni] skip extract start_stage=$START_STAGE"
fi

require_file "$SEGMENTS_MANIFEST" "extracted single source clips"
if stage_enabled annotate; then
  run_with_timeout "annotate-whole-source" "$ANNOTATION_TIMEOUT_SECONDS" \
    python3 -m app.composed_data detective-annotate-clips \
      --root "$ROOT" \
      --clips-manifest-path "$WHOLE_MANIFEST" \
      --output-path "$WHOLE_ANNOTATION" \
      --base-url "$BASE_URL" \
      --api-key EMPTY \
      --model "$MODEL" \
      --timeout-seconds "$ANNOTATION_TIMEOUT_SECONDS" \
      --concurrency 1
  run_with_timeout "annotate-single-source-segments" "$ANNOTATION_TIMEOUT_SECONDS" \
    python3 -m app.composed_data detective-annotate-clips \
      --root "$ROOT" \
      --clips-manifest-path "$SEGMENTS_MANIFEST" \
      --output-path "$SEGMENT_ANNOTATIONS" \
      --base-url "$BASE_URL" \
      --api-key EMPTY \
      --model "$MODEL" \
      --timeout-seconds "$ANNOTATION_TIMEOUT_SECONDS" \
      --concurrency "$CONCURRENCY"
else
  echo "[single-source-omni] skip annotate start_stage=$START_STAGE"
fi

require_file "$SEGMENT_ANNOTATIONS" "single source annotations"
if stage_enabled mine; then
  python3 -m app.composed_data mine-single-source-pairs \
    --root "$ROOT" \
    --clip-annotations-path "$SEGMENT_ANNOTATIONS" \
    --clip-groups-path "$CLIP_GROUPS" \
    --output-path "$PAIR_CANDIDATES" \
    --report-path "$PAIR_REPORT" \
    --acceptance-profile "$ACCEPTANCE_PROFILE"
else
  echo "[single-source-omni] skip mine start_stage=$START_STAGE"
fi

require_file "$PAIR_CANDIDATES" "single source pair candidates"
if stage_enabled propose; then
  run_with_timeout "propose-single-source-pairs" "$PROPOSE_TIMEOUT_SECONDS" \
    python3 -m app.composed_data propose-group-pairs \
      --root "$ROOT" \
      --clip-annotations-path "$SEGMENT_ANNOTATIONS" \
      --clip-groups-path "$CLIP_GROUPS" \
      --mined-candidates-path "$PAIR_CANDIDATES" \
      --output-path "$RANKED_PAIRS" \
      --accepted-output-path "$ACCEPTED_PAIRS" \
      --base-url "$BASE_URL" \
      --api-key EMPTY \
      --model "$MODEL" \
      --timeout-seconds "$PAIR_REQUEST_TIMEOUT_SECONDS" \
      --max-accepted-pairs "$MAX_ACCEPTED_PAIRS" \
      --max-proposals "$MAX_PROPOSALS" \
      --zero-accepted-stop-after "$ZERO_ACCEPTED_STOP_AFTER" \
      --acceptance-profile "$ACCEPTANCE_PROFILE"
else
  echo "[single-source-omni] skip propose start_stage=$START_STAGE"
fi

JUDGED_COUNT=$(jsonl_row_count "$RANKED_PAIRS")
ACCEPTED_COUNT=$(jsonl_row_count "$ACCEPTED_PAIRS")
echo "[single-source-omni] judged_pairs=$JUDGED_COUNT accepted_pairs=$ACCEPTED_COUNT"
if [ "$JUDGED_COUNT" -ge "$ZERO_ACCEPTED_STOP_AFTER" ] && [ "$ZERO_ACCEPTED_STOP_AFTER" -gt 0 ] && [ "$ACCEPTED_COUNT" -eq 0 ]; then
  echo "[single-source-omni] ERROR: zero accepted after $JUDGED_COUNT judged pairs; selected source or segment annotations need inspection" >&2
  exit 10
fi

if stage_enabled validate; then
  python3 -m app.composed_data validate-pilot \
    --root "$ROOT" \
    --pilot-jsonl-path "$ACCEPTED_PAIRS" \
    --gallery-output-path "$GALLERY" \
    --report-output-path "$PILOT_REVIEW"
else
  echo "[single-source-omni] skip validate start_stage=$START_STAGE"
fi

if stage_enabled review; then
  python3 -m app.composed_data build-single-source-review-bundle \
    --root "$ROOT" \
    --selected-source-path "$SELECTED_SOURCE" \
    --segments-manifest-path "$SEGMENTS_MANIFEST" \
    --clip-annotations-path "$SEGMENT_ANNOTATIONS" \
    --ranked-pairs-path "$RANKED_PAIRS" \
    --accepted-pairs-path "$ACCEPTED_PAIRS" \
    --output-dir "$REVIEW_BUNDLE"
else
  echo "[single-source-omni] skip review start_stage=$START_STAGE"
fi

echo "[single-source-omni] done $(date)"
echo "[single-source-omni] review_bundle=$REVIEW_BUNDLE"
