#!/usr/bin/env bash
set -euo pipefail

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/audio_matters_natural_omni_$(date +%Y%m%d_%H%M%S)}
MODEL=${MODEL:-qwen3-omni}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
SOURCE_CLIPS=${SOURCE_CLIPS:-$ROOT/metadata/source_clips_all.jsonl}
PREPARE_START_STAGE=${PREPARE_START_STAGE:-plan}
MAX_SOURCE_VIDEOS=${MAX_SOURCE_VIDEOS:-80}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-8}
CONCURRENCY=${CONCURRENCY:-1}
ANNOTATION_MAX_PASSES=${ANNOTATION_MAX_PASSES:-3}
ANNOTATION_PASS_TIMEOUT_SECONDS=${ANNOTATION_PASS_TIMEOUT_SECONDS:-900}
MINE_AUDIO_TIMEOUT_SECONDS=${MINE_AUDIO_TIMEOUT_SECONDS:-600}
PROPOSE_TIMEOUT_SECONDS=${PROPOSE_TIMEOUT_SECONDS:-1800}
PAIR_REQUEST_TIMEOUT_SECONDS=${PAIR_REQUEST_TIMEOUT_SECONDS:-90}
ZERO_ACCEPTED_STOP_AFTER=${ZERO_ACCEPTED_STOP_AFTER:-0}
MAX_AUDIO_CANDIDATES=${MAX_AUDIO_CANDIDATES:-240}
MAX_ACCEPTED_PAIRS=${MAX_ACCEPTED_PAIRS:-80}
MAX_PROPOSALS=${MAX_PROPOSALS:-160}
MIN_AUDIO_ANCHOR_SCORE=${MIN_AUDIO_ANCHOR_SCORE:-0.86}
MIN_AUDIO_RMS=${MIN_AUDIO_RMS:-0.001}
MIN_DIFFERENCE_STRENGTH=${MIN_DIFFERENCE_STRENGTH:-0.60}
MAX_LOCAL_COMPARISONS=${MAX_LOCAL_COMPARISONS:-20000}
ACCEPTANCE_PROFILE=${ACCEPTANCE_PROFILE:-final}
ALLOW_PARTIAL_ANNOTATIONS=${ALLOW_PARTIAL_ANNOTATIONS:-0}
MODEL_STAGE=${MODEL_STAGE:-instruct}
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
MAX_GPUS=${MAX_GPUS:-6}

usage() {
  cat <<'EOF'
Usage: run_audio_matters_natural_omni.sh [options]

Options:
  --root PATH
  --run-root PATH
  --model NAME
  --base-url URL
  --source-clips PATH
  --prepare-start-stage plan|extract|annotate|mine-candidates|none
  --max-source-videos N
  --segment-seconds N
  --concurrency N
  --annotation-max-passes N
  --annotation-pass-timeout-seconds N
  --mine-audio-timeout-seconds N
  --propose-timeout-seconds N
  --pair-request-timeout-seconds N
  --zero-accepted-stop-after N
  --max-audio-candidates N
  --max-accepted-pairs N
  --max-proposals N
  --min-audio-anchor-score FLOAT
  --min-audio-rms FLOAT
  --min-difference-strength FLOAT
  --max-local-comparisons N
  --acceptance-profile exploration|final
  --allow-partial-annotations
  --model-stage VALUE
  --gpu-ids IDS
  --max-gpus N
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
    --prepare-start-stage) PREPARE_START_STAGE="$2"; shift 2 ;;
    --max-source-videos) MAX_SOURCE_VIDEOS="$2"; shift 2 ;;
    --segment-seconds) SEGMENT_SECONDS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --annotation-max-passes) ANNOTATION_MAX_PASSES="$2"; shift 2 ;;
    --annotation-pass-timeout-seconds) ANNOTATION_PASS_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --mine-audio-timeout-seconds) MINE_AUDIO_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --propose-timeout-seconds) PROPOSE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --pair-request-timeout-seconds) PAIR_REQUEST_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --zero-accepted-stop-after) ZERO_ACCEPTED_STOP_AFTER="$2"; shift 2 ;;
    --max-audio-candidates) MAX_AUDIO_CANDIDATES="$2"; shift 2 ;;
    --max-accepted-pairs) MAX_ACCEPTED_PAIRS="$2"; shift 2 ;;
    --max-proposals) MAX_PROPOSALS="$2"; shift 2 ;;
    --min-audio-anchor-score) MIN_AUDIO_ANCHOR_SCORE="$2"; shift 2 ;;
    --min-audio-rms) MIN_AUDIO_RMS="$2"; shift 2 ;;
    --min-difference-strength) MIN_DIFFERENCE_STRENGTH="$2"; shift 2 ;;
    --max-local-comparisons) MAX_LOCAL_COMPARISONS="$2"; shift 2 ;;
    --acceptance-profile) ACCEPTANCE_PROFILE="$2"; shift 2 ;;
    --allow-partial-annotations) ALLOW_PARTIAL_ANNOTATIONS=1; shift ;;
    --model-stage) MODEL_STAGE="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[audio-matters-natural] unknown argument: $1" >&2
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

prepare_stage_rank() {
  case "$1" in
    plan) echo 1 ;;
    extract) echo 2 ;;
    annotate) echo 3 ;;
    none) echo 99 ;;
    *)
      echo "[audio-matters-natural] unsupported prepare stage: $1" >&2
      exit 2
      ;;
  esac
}

prepare_stage_enabled() {
  local stage="$1"
  [ "$(prepare_stage_rank "$stage")" -ge "$(prepare_stage_rank "$PREPARE_START_STAGE")" ]
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -s "$path" ]; then
    echo "[audio-matters-natural] missing required $label: $path" >&2
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
      echo "[audio-matters-natural] ERROR: $label timed out after ${timeout_seconds}s; killing child pid=$child_pid" >&2
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

probe_omni_model() {
  local models_json
  models_json=$(curl -fsS "$BASE_URL/models")
  printf '%s\n' "$models_json"
  BASE_URL="$BASE_URL" OMNI_MODELS_JSON="$models_json" python - "$MODEL" <<'PY'
import json
import os
import sys
wanted = sys.argv[1]
payload = json.loads(os.environ["OMNI_MODELS_JSON"])
served = [
    str(item.get("id", "")).strip()
    for item in payload.get("data", [])
    if isinstance(item, dict) and str(item.get("id", "")).strip()
]
print("[audio-matters-natural] served_models=" + ",".join(served))
if wanted not in served:
    choices = ", ".join(served) if served else "<none>"
    print(
        f"[audio-matters-natural] ERROR: model={wanted!r} is not served by {os.environ['BASE_URL']}; use one of: {choices}",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
}

GPU_COUNT=$(count_gpu_ids "$GPU_IDS")
if (( GPU_COUNT > MAX_GPUS )); then
  echo "[resource-policy] refusing to run with GPU_COUNT=$GPU_COUNT > MAX_GPUS=$MAX_GPUS" >&2
  exit 2
fi

case "$MODEL_STAGE" in
  instruct) ;;
  captioner|thinking)
    echo "[resource-policy] this script runs the Instruct pair-construction stage only; stop other model stages first" >&2
    exit 2
    ;;
  *)
    echo "[resource-policy] unsupported MODEL_STAGE=$MODEL_STAGE; expected instruct, captioner, or thinking" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"

echo "[audio-matters-natural] start $(date)"
echo "[audio-matters-natural] root=$ROOT"
echo "[audio-matters-natural] run_root=$RUN_ROOT"
echo "[audio-matters-natural] source_clips=$SOURCE_CLIPS"
echo "[audio-matters-natural] base_url=$BASE_URL"
echo "[audio-matters-natural] model=$MODEL"
echo "[audio-matters-natural] prepare_start_stage=$PREPARE_START_STAGE"
echo "[audio-matters-natural] max_source_videos=$MAX_SOURCE_VIDEOS segment_seconds=$SEGMENT_SECONDS concurrency=$CONCURRENCY"
echo "[audio-matters-natural] max_audio_candidates=$MAX_AUDIO_CANDIDATES max_accepted_pairs=$MAX_ACCEPTED_PAIRS max_proposals=$MAX_PROPOSALS"
echo "[audio-matters-natural] min_audio_anchor_score=$MIN_AUDIO_ANCHOR_SCORE min_audio_rms=$MIN_AUDIO_RMS min_difference_strength=$MIN_DIFFERENCE_STRENGTH"

if [ "$PREPARE_START_STAGE" != "none" ]; then
  probe_omni_model

  if prepare_stage_enabled "plan"; then
    python -m app.composed_data plan-detective-clips \
      --root "$ROOT" \
      --source-clips-path "$SOURCE_CLIPS" \
      --clip-plan-output-path "$RUN_ROOT/clip_plan_detective.jsonl" \
      --clip-groups-output-path "$RUN_ROOT/clip_groups.jsonl" \
      --max-source-videos "$MAX_SOURCE_VIDEOS" \
      --segment-seconds "$SEGMENT_SECONDS" \
      --min-clip-seconds 3 \
      --max-clip-seconds 15
    echo "[audio-matters-natural] planning done $(date)"
  else
    require_file "$RUN_ROOT/clip_plan_detective.jsonl" "clip plan"
    require_file "$RUN_ROOT/clip_groups.jsonl" "clip groups"
    echo "[audio-matters-natural] skip planning prepare_start_stage=$PREPARE_START_STAGE"
  fi

  if prepare_stage_enabled "extract"; then
    python -m app.composed_data extract-clips \
      --root "$ROOT" \
      --plan-path "$RUN_ROOT/clip_plan_detective.jsonl" \
      --output-manifest-path "$RUN_ROOT/extracted_event_clips.jsonl" \
      --overwrite
    echo "[audio-matters-natural] extraction done $(date)"
  else
    require_file "$RUN_ROOT/extracted_event_clips.jsonl" "extracted clips manifest"
    echo "[audio-matters-natural] skip extraction prepare_start_stage=$PREPARE_START_STAGE"
  fi

  if prepare_stage_enabled "annotate"; then
    ANNOTATION_TARGET_COUNT=$(jsonl_row_count "$RUN_ROOT/extracted_event_clips.jsonl")
    ANNOTATION_DONE_COUNT=$(jsonl_unique_clip_count "$RUN_ROOT/detective_annotations.jsonl")
    ANNOTATION_ROW_COUNT=$(jsonl_row_count "$RUN_ROOT/detective_annotations.jsonl")
    if [ "$ANNOTATION_DONE_COUNT" -ge "$ANNOTATION_TARGET_COUNT" ]; then
      echo "[audio-matters-natural] annotation already complete unique_done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT rows=$ANNOTATION_ROW_COUNT"
    else
      ANNOTATION_PASS=1
      while [ "$ANNOTATION_PASS" -le "$ANNOTATION_MAX_PASSES" ]; do
        echo "[audio-matters-natural] annotation pass $ANNOTATION_PASS/$ANNOTATION_MAX_PASSES target_clips=$ANNOTATION_TARGET_COUNT $(date)"
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
        echo "[audio-matters-natural] annotation pass $ANNOTATION_PASS exit=$ANNOTATION_STATUS unique_done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT rows=$ANNOTATION_ROW_COUNT"
        if [ "$ANNOTATION_STATUS" -eq 124 ]; then
          echo "[audio-matters-natural] annotation pass timed out; report this status immediately instead of waiting" >&2
          exit 124
        fi
        if [ "$ANNOTATION_DONE_COUNT" -ge "$ANNOTATION_TARGET_COUNT" ]; then
          break
        fi
        if [ "$ANNOTATION_PASS" -ge "$ANNOTATION_MAX_PASSES" ]; then
          echo "[audio-matters-natural] annotation incomplete after $ANNOTATION_MAX_PASSES passes" >&2
          exit 3
        fi
        ANNOTATION_PASS=$((ANNOTATION_PASS + 1))
        sleep 10
      done
    fi
    echo "[audio-matters-natural] annotation done $(date)"
  else
    require_file "$RUN_ROOT/detective_annotations.jsonl" "detective annotations"
    echo "[audio-matters-natural] skip annotation prepare_start_stage=$PREPARE_START_STAGE"
  fi
else
  echo "[audio-matters-natural] skip natural preparation; using existing annotations in run_root"
fi

require_file "$RUN_ROOT/clip_groups.jsonl" "clip groups"
require_file "$RUN_ROOT/extracted_event_clips.jsonl" "extracted clips manifest"
require_file "$RUN_ROOT/detective_annotations.jsonl" "detective annotations"

ANNOTATION_TARGET_COUNT=$(jsonl_row_count "$RUN_ROOT/extracted_event_clips.jsonl")
ANNOTATION_DONE_COUNT=$(jsonl_unique_clip_count "$RUN_ROOT/detective_annotations.jsonl")
echo "[audio-matters-natural] annotation coverage unique_done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT"
if [ "$ANNOTATION_DONE_COUNT" -lt "$ANNOTATION_TARGET_COUNT" ] && [ "$ALLOW_PARTIAL_ANNOTATIONS" != "1" ]; then
  echo "[audio-matters-natural] annotation incomplete; rerun preparation or pass --allow-partial-annotations for diagnostics" >&2
  exit 3
fi

run_with_timeout "mine-audio-matters-candidates" "$MINE_AUDIO_TIMEOUT_SECONDS" \
python -m app.audio_matters_natural mine-candidates \
  --root "$ROOT" \
  --clip-annotations-path "$RUN_ROOT/detective_annotations.jsonl" \
  --clip-groups-path "$RUN_ROOT/clip_groups.jsonl" \
  --output-path "$RUN_ROOT/audio_matters_mined_candidates.jsonl" \
  --report-path "$RUN_ROOT/audio_matters_mining_report.md" \
  --summary-path "$RUN_ROOT/audio_matters_mining_summary.json" \
  --max-candidates "$MAX_AUDIO_CANDIDATES" \
  --min-audio-anchor-score "$MIN_AUDIO_ANCHOR_SCORE" \
  --min-audio-rms "$MIN_AUDIO_RMS" \
  --min-difference-strength "$MIN_DIFFERENCE_STRENGTH" \
  --max-local-comparisons "$MAX_LOCAL_COMPARISONS" \
  --acceptance-profile "$ACCEPTANCE_PROFILE"

MINED_ROW_COUNT=$(jsonl_row_count "$RUN_ROOT/audio_matters_mined_candidates.jsonl")
echo "[audio-matters-natural] mined audio candidates rows=$MINED_ROW_COUNT"
if [ "$MINED_ROW_COUNT" -eq 0 ]; then
  echo "[audio-matters-natural] no audio-matters candidates mined; inspect $RUN_ROOT/audio_matters_mining_report.md" >&2
  exit 4
fi

probe_omni_model
run_with_timeout "propose-audio-matters-pairs" "$PROPOSE_TIMEOUT_SECONDS" \
python -m app.composed_data propose-group-pairs \
  --root "$ROOT" \
  --clip-annotations-path "$RUN_ROOT/detective_annotations.jsonl" \
  --clip-groups-path "$RUN_ROOT/clip_groups.jsonl" \
  --mined-candidates-path "$RUN_ROOT/audio_matters_mined_candidates.jsonl" \
  --output-path "$RUN_ROOT/judged_audio_matters_pair_proposals.jsonl" \
  --accepted-output-path "$RUN_ROOT/accepted_audio_matters_pairs.jsonl" \
  --base-url "$BASE_URL" \
  --api-key EMPTY \
  --model "$MODEL" \
  --timeout-seconds "$PAIR_REQUEST_TIMEOUT_SECONDS" \
  --max-accepted-pairs "$MAX_ACCEPTED_PAIRS" \
  --max-proposals "$MAX_PROPOSALS" \
  --zero-accepted-stop-after "$ZERO_ACCEPTED_STOP_AFTER" \
  --acceptance-profile "$ACCEPTANCE_PROFILE" \
  --overwrite

ACCEPTED_ROW_COUNT=$(jsonl_row_count "$RUN_ROOT/accepted_audio_matters_pairs.jsonl")
echo "[audio-matters-natural] accepted audio pairs rows=$ACCEPTED_ROW_COUNT"

if [ "$ACCEPTED_ROW_COUNT" -gt 0 ]; then
  python -m app.audio_matters_natural export-triplets \
    --root "$ROOT" \
    --accepted-pairs-path "$RUN_ROOT/accepted_audio_matters_pairs.jsonl" \
    --output-path "$RUN_ROOT/audio_matters_triplets.jsonl" \
    --summary-path "$RUN_ROOT/audio_matters_triplets_summary.json"

  python -m app.composed_data validate-pilot \
    --root "$ROOT" \
    --pilot-jsonl-path "$RUN_ROOT/accepted_audio_matters_pairs.jsonl" \
    --gallery-output-path "$RUN_ROOT/audio_matters_gallery.jsonl" \
    --report-output-path "$RUN_ROOT/audio_matters_pilot_review.md"

  python -m app.composed_data build-review-bundle \
    --root "$ROOT" \
    --pairs-path "$RUN_ROOT/accepted_audio_matters_pairs.jsonl" \
    --clip-annotations-path "$RUN_ROOT/detective_annotations.jsonl" \
    --output-dir "$RUN_ROOT/audio_matters_manual_review_bundle"
else
  echo "[audio-matters-natural] no accepted audio pairs; skip export and review bundle"
fi

echo "[audio-matters-natural] done $(date)"
echo "[audio-matters-natural] run_root=$RUN_ROOT"
