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
SINGLE_SOURCE_ROOT=${SINGLE_SOURCE_ROOT:-$ROOT/clips/single_source}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/audio_lines_single_source_reuse_$(date +%Y%m%d_%H%M%S)}
MODEL=${MODEL:-qwen3-omni-30b-a3b-instruct}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
AUDIO_DATASET_LINE=${AUDIO_DATASET_LINE:-both}
TARGET_A_COUNT=${TARGET_A_COUNT:-8}
TARGET_B_COUNT=${TARGET_B_COUNT:-8}
MAX_SOURCE_FOLDERS=${MAX_SOURCE_FOLDERS:-80}
MAX_CLIPS=${MAX_CLIPS:-0}
MAX_A_CANDIDATES=${MAX_A_CANDIDATES:-320}
MAX_B_CANDIDATES=${MAX_B_CANDIDATES:-320}
PROPOSE_SHARDS=${PROPOSE_SHARDS:-16}
PROPOSE_PARALLEL_JOBS=${PROPOSE_PARALLEL_JOBS:-8}
CONCURRENCY=${CONCURRENCY:-4}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-90}
SHARD_TIMEOUT_SECONDS=${SHARD_TIMEOUT_SECONDS:-3600}
ANNOTATION_TIMEOUT_SECONDS=${ANNOTATION_TIMEOUT_SECONDS:-900}
MIN_AUDIO_ANCHOR_SCORE=${MIN_AUDIO_ANCHOR_SCORE:-0.86}
AUDIO_LINE_QUALITY_PROFILE=${AUDIO_LINE_QUALITY_PROFILE:-default}
B_ACCEPTANCE_PROFILE=${B_ACCEPTANCE_PROFILE:-exploration}
FORCE_AUDIO_FOCUSED_REFRESH=${FORCE_AUDIO_FOCUSED_REFRESH:-0}
FRESH_ANNOTATIONS=${FRESH_ANNOTATIONS:-1}
REUSE_RUN_ROOT=${REUSE_RUN_ROOT:-}
SKIP_ANNOTATION_REFRESH=${SKIP_ANNOTATION_REFRESH:-0}
ANNOTATION_SEARCH_ROOTS=${ANNOTATION_SEARCH_ROOTS:-}
A_CANDIDATE_MODE=${A_CANDIDATE_MODE:-hybrid}
B_CANDIDATE_MODE=${B_CANDIDATE_MODE:-hybrid}
OMNI_TRANSIENT_RETRIES=${OMNI_TRANSIENT_RETRIES:-2}
FAIL_ON_TRANSIENT_OMNI_ERRORS=${FAIL_ON_TRANSIENT_OMNI_ERRORS:-1}

usage() {
  cat <<'EOF'
Usage: run_audio_lines_single_source_reuse.sh [options]

Options:
  --root PATH
  --single-source-root PATH
  --run-root PATH
  --base-url URL
  --model NAME
  --audio-dataset-line visual_audio_anchor|speech_audio_content|both
  --target-a-count N
  --target-b-count N
  --max-source-folders N
  --max-clips N
  --propose-shards N
  --propose-parallel-jobs N
  --request-timeout-seconds N
  --shard-timeout-seconds N
  --audio-line-quality-profile default|v4_strict|v5_audio_primary|b_audio_context_cvr|b_audio_blind_review|b_audio_blind_review_v2
  --acceptance-profile exploration|b_audio_review|b_audio_context_cvr|b_audio_blind_review|b_audio_blind_review_v2
  --a-candidate-mode hybrid|omni_first
  --b-candidate-mode hybrid|audio_first
  --reuse-run-root PATH
  --skip-annotation-refresh
  --run-b-only
  --fresh-annotations
  --force-audio-focused-refresh
  --annotation-search-root PATH
  --omni-transient-retries N
  --allow-transient-omni-fallback
  --concurrency N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --single-source-root) SINGLE_SOURCE_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --audio-dataset-line) AUDIO_DATASET_LINE="$2"; shift 2 ;;
    --target-a-count) TARGET_A_COUNT="$2"; shift 2 ;;
    --target-b-count) TARGET_B_COUNT="$2"; shift 2 ;;
    --max-source-folders) MAX_SOURCE_FOLDERS="$2"; shift 2 ;;
    --max-clips) MAX_CLIPS="$2"; shift 2 ;;
    --propose-shards) PROPOSE_SHARDS="$2"; shift 2 ;;
    --propose-parallel-jobs) PROPOSE_PARALLEL_JOBS="$2"; shift 2 ;;
    --request-timeout-seconds) REQUEST_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --shard-timeout-seconds) SHARD_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --audio-line-quality-profile) AUDIO_LINE_QUALITY_PROFILE="$2"; shift 2 ;;
    --acceptance-profile) B_ACCEPTANCE_PROFILE="$2"; shift 2 ;;
    --a-candidate-mode) A_CANDIDATE_MODE="$2"; shift 2 ;;
    --b-candidate-mode) B_CANDIDATE_MODE="$2"; shift 2 ;;
    --reuse-run-root) REUSE_RUN_ROOT="$2"; RUN_ROOT="$2"; FRESH_ANNOTATIONS=0; shift 2 ;;
    --skip-annotation-refresh) SKIP_ANNOTATION_REFRESH=1; shift ;;
    --run-b-only) AUDIO_DATASET_LINE=speech_audio_content; shift ;;
    --fresh-annotations) FRESH_ANNOTATIONS=1; shift ;;
    --force-audio-focused-refresh) FORCE_AUDIO_FOCUSED_REFRESH=1; shift ;;
    --annotation-search-root)
      ANNOTATION_SEARCH_ROOTS="${ANNOTATION_SEARCH_ROOTS:+$ANNOTATION_SEARCH_ROOTS,}$2"
      shift 2
      ;;
    --omni-transient-retries) OMNI_TRANSIENT_RETRIES="$2"; shift 2 ;;
    --allow-transient-omni-fallback) FAIL_ON_TRANSIENT_OMNI_ERRORS=0; shift ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[audio-lines] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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
    echo "[audio-lines] missing required $label: $path" >&2
    exit 2
  fi
}

resolve_omni_model() {
  local models_json
  local resolved_model
  models_json=$(curl -fsS "$BASE_URL/models")
  resolved_model=$(OMNI_MODELS_JSON="$models_json" python3 - "$MODEL" <<'PY'
import json
import os
import sys

wanted = sys.argv[1]
payload = json.loads(os.environ["OMNI_MODELS_JSON"])
served = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
print("[audio-lines] served_models=" + ",".join(served), file=sys.stderr)
if wanted in served:
    print(wanted)
    raise SystemExit(0)

candidates = []
if wanted == "qwen3-omni":
    candidates = [item for item in served if "qwen3-omni" in item]
if not candidates:
    candidates = [item for item in served if wanted and (wanted in item or item in wanted)]
if len(candidates) == 1:
    print(f"[audio-lines] resolved_model_alias={wanted}->{candidates[0]}", file=sys.stderr)
    print(candidates[0])
    raise SystemExit(0)

raise SystemExit(f"[audio-lines] model {wanted!r} is not served by {served}; use the registered service name")
PY
)
  MODEL="$resolved_model"
  echo "[audio-lines] model=$MODEL"
}

run_line_shards() {
  local line_name="$1"
  local shard_dir="$2"
  local ranked_prefix="$3"
  local accepted_prefix="$4"
  local acceptance_profile="$5"
  local line_mode="$6"
  local max_accepted="$7"
  mkdir -p "$shard_dir/logs"
  local active=0
  local failed=0
  shopt -s nullglob
  for shard in "$shard_dir"/"${ranked_prefix}"_shard_*.jsonl; do
    local rows
    rows=$(jsonl_row_count "$shard")
    if [ "$rows" -eq 0 ]; then
      continue
    fi
    local shard_id
    shard_id=$(basename "$shard" .jsonl | sed "s/${ranked_prefix}_shard_//")
    (
      echo "[audio-lines] $line_name shard=$shard_id rows=$rows start $(date)"
      set +e
      timeout "$SHARD_TIMEOUT_SECONDS" python3 -m app.composed_data propose-single-source-pairs \
        --root "$ROOT" \
        --clip-annotations-path "$SEGMENT_ANNOTATIONS" \
        --pair-candidates-path "$shard" \
        --whole-annotation-path "$WHOLE_ANNOTATION" \
        --output-path "$shard_dir/ranked_${shard_id}.jsonl" \
        --accepted-output-path "$shard_dir/accepted_${shard_id}.jsonl" \
        --accepted-progress-path "$shard_dir/accepted_progress_${shard_id}.jsonl" \
        --rejected-progress-path "$shard_dir/rejected_progress_${shard_id}.jsonl" \
        --base-url "$BASE_URL" \
        --api-key EMPTY \
        --model "$MODEL" \
        --timeout-seconds "$REQUEST_TIMEOUT_SECONDS" \
        --max-accepted-pairs "$max_accepted" \
        --zero-accepted-stop-after 0 \
        --acceptance-profile "$acceptance_profile" \
        --audio-dataset-line "$line_mode" \
        --omni-retries "$OMNI_TRANSIENT_RETRIES" \
        $(if [ "$FAIL_ON_TRANSIENT_OMNI_ERRORS" = "1" ]; then printf '%s' '--fail-on-transient-omni-errors'; fi)
      status=$?
      set -e
      if [ "$status" -eq 0 ]; then
        echo "[audio-lines] $line_name shard=$shard_id done $(date)"
      elif [ "$status" -eq 124 ]; then
        echo "[audio-lines] WARN $line_name shard=$shard_id timed out after ${SHARD_TIMEOUT_SECONDS}s $(date)" >&2
      else
        echo "[audio-lines] WARN $line_name shard=$shard_id failed status=$status $(date)" >&2
      fi
      exit "$status"
    ) > "$shard_dir/logs/${line_name}_${shard_id}.log" 2>&1 &
    active=$((active + 1))
    if [ "$active" -ge "$PROPOSE_PARALLEL_JOBS" ]; then
      if ! wait -n; then
        failed=$((failed + 1))
      fi
      active=$((active - 1))
    fi
  done
  shopt -u nullglob
  while [ "$active" -gt 0 ]; do
    if ! wait -n; then
      failed=$((failed + 1))
    fi
    active=$((active - 1))
  done
  if [ "$failed" -gt 0 ]; then
    echo "[audio-lines] WARN $line_name completed with failed_or_timed_out_shards=$failed; continuing with progress files" >&2
  fi
}

mkdir -p "$RUN_ROOT" "$REPO_ROOT/logs"
echo "[audio-lines] start $(date)"
echo "[audio-lines] run_root=$RUN_ROOT root=$ROOT single_source_root=$SINGLE_SOURCE_ROOT line=$AUDIO_DATASET_LINE"
echo "[audio-lines] max_source_folders=$MAX_SOURCE_FOLDERS max_clips=$MAX_CLIPS propose_shards=$PROPOSE_SHARDS propose_parallel_jobs=$PROPOSE_PARALLEL_JOBS shard_timeout_seconds=$SHARD_TIMEOUT_SECONDS annotation_search_roots=$ANNOTATION_SEARCH_ROOTS audio_line_quality_profile=$AUDIO_LINE_QUALITY_PROFILE b_acceptance_profile=$B_ACCEPTANCE_PROFILE a_candidate_mode=$A_CANDIDATE_MODE b_candidate_mode=$B_CANDIDATE_MODE fresh_annotations=$FRESH_ANNOTATIONS force_audio_focused_refresh=$FORCE_AUDIO_FOCUSED_REFRESH reuse_run_root=$REUSE_RUN_ROOT skip_annotation_refresh=$SKIP_ANNOTATION_REFRESH omni_transient_retries=$OMNI_TRANSIENT_RETRIES fail_on_transient_omni_errors=$FAIL_ON_TRANSIENT_OMNI_ERRORS"
resolve_omni_model

SEGMENTS_MANIFEST="$RUN_ROOT/extracted_single_source_clips.jsonl"
CLIP_GROUPS="$RUN_ROOT/single_source_clip_groups.jsonl"
CLIPS_TO_ANNOTATE="$RUN_ROOT/clips_to_annotate.jsonl"
AUDIO_REFRESH_MANIFEST="$RUN_ROOT/audio_refresh_clips.jsonl"
SEGMENT_ANNOTATIONS="$RUN_ROOT/single_source_annotations.jsonl"
AUDIO_REFRESH_ANNOTATIONS="$RUN_ROOT/audio_refresh_annotations.jsonl"
WHOLE_ANNOTATION="$RUN_ROOT/single_source_whole_annotation.jsonl"
PAIR_CANDIDATES="$RUN_ROOT/single_source_pair_candidates.jsonl"
A_CANDIDATES="$RUN_ROOT/a_candidates.jsonl"
B_CANDIDATES="$RUN_ROOT/b_candidates.jsonl"

annotation_search_args=()
if [ -n "$ANNOTATION_SEARCH_ROOTS" ]; then
  IFS=',' read -r -a annotation_search_roots <<< "$ANNOTATION_SEARCH_ROOTS"
  for annotation_search_root in "${annotation_search_roots[@]}"; do
    annotation_search_root="${annotation_search_root#"${annotation_search_root%%[![:space:]]*}"}"
    annotation_search_root="${annotation_search_root%"${annotation_search_root##*[![:space:]]}"}"
    if [ -n "$annotation_search_root" ]; then
      annotation_search_args+=(--annotation-search-root "$annotation_search_root")
    fi
  done
fi
if [ "$FRESH_ANNOTATIONS" != "1" ] && [ "${#annotation_search_args[@]}" -eq 0 ]; then
  annotation_search_args+=(--annotation-search-root "$REPO_ROOT/runs" --annotation-search-root "$ROOT")
fi

if [ "$SKIP_ANNOTATION_REFRESH" = "1" ]; then
  echo "[audio-lines] skip annotation refresh; reusing existing manifests from $RUN_ROOT"
  require_file "$SEGMENT_ANNOTATIONS" "segment annotations"
  require_file "$CLIP_GROUPS" "clip groups"
else
  prepare_existing_args=(
    python3
    -m
    app.audio_lines_single_source
    prepare-existing
    --root "$ROOT"
    --single-source-root "$SINGLE_SOURCE_ROOT"
    --run-root "$RUN_ROOT"
    --max-source-folders "$MAX_SOURCE_FOLDERS"
  )
  if [ "$MAX_CLIPS" != "0" ]; then
    prepare_existing_args+=(--max-clips "$MAX_CLIPS")
  fi
  if [ "${#annotation_search_args[@]}" -gt 0 ]; then
    prepare_existing_args+=("${annotation_search_args[@]}")
  elif [ "$FRESH_ANNOTATIONS" != "1" ]; then
    prepare_existing_args+=(--annotation-search-root "$REPO_ROOT/runs" --annotation-search-root "$ROOT")
  elif [ "$FRESH_ANNOTATIONS" = "1" ]; then
    prepare_existing_args+=(--no-annotation-reuse)
  fi
  if [ "$FORCE_AUDIO_FOCUSED_REFRESH" = "1" ]; then
    prepare_existing_args+=(--force-audio-focused-refresh)
  fi
  "${prepare_existing_args[@]}"

  require_file "$CLIPS_TO_ANNOTATE" "clips manifest"
  python3 -m app.composed_data detective-annotate-clips \
    --root "$ROOT" \
    --clips-manifest-path "$CLIPS_TO_ANNOTATE" \
    --output-path "$SEGMENT_ANNOTATIONS" \
    --base-url "$BASE_URL" \
    --api-key EMPTY \
    --model "$MODEL" \
    --timeout-seconds "$ANNOTATION_TIMEOUT_SECONDS" \
    --concurrency "$CONCURRENCY" \
    $(if [ "$AUDIO_LINE_QUALITY_PROFILE" = "v4_strict" ] || [ "$AUDIO_LINE_QUALITY_PROFILE" = "v5_audio_primary" ] || [ "$AUDIO_LINE_QUALITY_PROFILE" = "b_audio_context_cvr" ] || [ "$AUDIO_LINE_QUALITY_PROFILE" = "b_audio_blind_review" ] || [ "$AUDIO_LINE_QUALITY_PROFILE" = "b_audio_blind_review_v2" ]; then printf '%s' '--audio-focused'; fi)
fi

if [ "$SKIP_ANNOTATION_REFRESH" != "1" ] && [ "$(jsonl_row_count "$AUDIO_REFRESH_MANIFEST")" -gt 0 ]; then
  echo "[audio-lines] audio refresh annotation start rows=$(jsonl_row_count "$AUDIO_REFRESH_MANIFEST")"
  python3 -m app.composed_data detective-annotate-clips \
    --root "$ROOT" \
    --clips-manifest-path "$AUDIO_REFRESH_MANIFEST" \
    --output-path "$AUDIO_REFRESH_ANNOTATIONS" \
    --base-url "$BASE_URL" \
    --api-key EMPTY \
    --model "$MODEL" \
    --timeout-seconds "$ANNOTATION_TIMEOUT_SECONDS" \
    --concurrency "$CONCURRENCY" \
    $(if [ "$AUDIO_LINE_QUALITY_PROFILE" = "v4_strict" ] || [ "$AUDIO_LINE_QUALITY_PROFILE" = "v5_audio_primary" ] || [ "$AUDIO_LINE_QUALITY_PROFILE" = "b_audio_context_cvr" ] || [ "$AUDIO_LINE_QUALITY_PROFILE" = "b_audio_blind_review" ] || [ "$AUDIO_LINE_QUALITY_PROFILE" = "b_audio_blind_review_v2" ]; then printf '%s' '--audio-focused'; fi)
  python3 -m app.audio_lines_single_source merge-annotations \
    --base-annotations-path "$SEGMENT_ANNOTATIONS" \
    --refresh-annotations-path "$AUDIO_REFRESH_ANNOTATIONS" \
    --output-path "$SEGMENT_ANNOTATIONS"
fi

if [ ! -s "$WHOLE_ANNOTATION" ]; then
  : > "$WHOLE_ANNOTATION"
fi

python3 -m app.composed_data mine-single-source-pairs \
  --root "$ROOT" \
  --clip-annotations-path "$SEGMENT_ANNOTATIONS" \
  --clip-groups-path "$CLIP_GROUPS" \
  --output-path "$PAIR_CANDIDATES" \
  --report-path "$RUN_ROOT/single_source_pair_report.md" \
  --acceptance-profile exploration

python3 -m app.audio_lines_single_source split-candidates \
  --root "$ROOT" \
  --clip-annotations-path "$SEGMENT_ANNOTATIONS" \
  --pair-candidates-path "$PAIR_CANDIDATES" \
  --a-output-path "$A_CANDIDATES" \
  --b-output-path "$B_CANDIDATES" \
  --summary-path "$RUN_ROOT/audio_line_candidate_summary.json" \
  --min-audio-anchor-score "$MIN_AUDIO_ANCHOR_SCORE" \
  --max-a-candidates "$MAX_A_CANDIDATES" \
  --max-b-candidates "$MAX_B_CANDIDATES" \
  --audio-line-quality-profile "$AUDIO_LINE_QUALITY_PROFILE" \
  --a-candidate-mode "$A_CANDIDATE_MODE" \
  --b-candidate-mode "$B_CANDIDATE_MODE"

if [ "$AUDIO_DATASET_LINE" = "both" ] || [ "$AUDIO_DATASET_LINE" = "visual_audio_anchor" ]; then
  test -d "$RUN_ROOT/a_shards" && mv "$RUN_ROOT/a_shards" "$RUN_ROOT/a_shards_before_$(date +%Y%m%d_%H%M%S)"
  python3 -m app.audio_lines_single_source shard-jsonl \
    --input-path "$A_CANDIDATES" \
    --output-dir "$RUN_ROOT/a_shards" \
    --shards "$PROPOSE_SHARDS" \
    --prefix a
  run_line_shards "a_visual_audio_anchor" "$RUN_ROOT/a_shards" "a" "a" "audio_matters" "visual_audio_anchor" "$TARGET_A_COUNT"
fi

if [ "$AUDIO_DATASET_LINE" = "both" ] || [ "$AUDIO_DATASET_LINE" = "speech_audio_content" ]; then
  test -d "$RUN_ROOT/b_shards" && mv "$RUN_ROOT/b_shards" "$RUN_ROOT/b_shards_before_$(date +%Y%m%d_%H%M%S)"
  python3 -m app.audio_lines_single_source shard-jsonl \
    --input-path "$B_CANDIDATES" \
    --output-dir "$RUN_ROOT/b_shards" \
    --shards "$PROPOSE_SHARDS" \
    --prefix b
  run_line_shards "b_speech_audio_content" "$RUN_ROOT/b_shards" "b" "b" "$B_ACCEPTANCE_PROFILE" "speech_audio_content" "$TARGET_B_COUNT"
fi

python3 -m app.audio_lines_single_source merge-line-results \
  --run-root "$RUN_ROOT" \
  --target-a-count "$TARGET_A_COUNT" \
  --target-b-count "$TARGET_B_COUNT"

mkdir -p "$RUN_ROOT/manual_review"
python3 -m app.composed_data build-review-bundle \
  --root "$ROOT" \
  --pairs-path "$RUN_ROOT/a_visual_audio_anchor_triplets.jsonl" \
  --output-dir "$RUN_ROOT/manual_review/A" \
  --clip-annotations-path "$SEGMENT_ANNOTATIONS"
python3 -m app.composed_data build-review-bundle \
  --root "$ROOT" \
  --pairs-path "$RUN_ROOT/b_speech_audio_content_triplets.jsonl" \
  --output-dir "$RUN_ROOT/manual_review/B" \
  --clip-annotations-path "$SEGMENT_ANNOTATIONS"

cat "$RUN_ROOT/summary.json"
echo "[audio-lines] done $(date)"
