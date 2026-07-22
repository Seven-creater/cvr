#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
SINGLE_SOURCE_ROOT=${SINGLE_SOURCE_ROOT:-$ROOT/clips/audio_cvr_8_12s}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/audio_cvr_v1_b_first_$(date +%Y%m%d_%H%M%S)}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
BASE_URL_POOL=${BASE_URL_POOL:-}
MODEL=${MODEL:-qwen3-omni-30b-a3b-instruct}
MAX_SOURCE_FOLDERS=${MAX_SOURCE_FOLDERS:-0}
MAX_CLIPS=${MAX_CLIPS:-0}
MAX_B_CANDIDATES=${MAX_B_CANDIDATES:-0}
PROPOSE_SHARDS=${PROPOSE_SHARDS:-32}
PROPOSE_PARALLEL_JOBS=${PROPOSE_PARALLEL_JOBS:-4}
CONCURRENCY=${CONCURRENCY:-4}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-180}
SHARD_TIMEOUT_SECONDS=${SHARD_TIMEOUT_SECONDS:-7200}
ANNOTATION_TIMEOUT_SECONDS=${ANNOTATION_TIMEOUT_SECONDS:-900}
TARGET_B_COUNT=${TARGET_B_COUNT:-1000000}
OMNI_TRANSIENT_RETRIES=${OMNI_TRANSIENT_RETRIES:-2}
FAIL_ON_TRANSIENT_OMNI_ERRORS=${FAIL_ON_TRANSIENT_OMNI_ERRORS:-1}
RESUME=${RESUME:-0}
QUALITY_PROFILE=${QUALITY_PROFILE:-b_audio_blind_review_v2}
CLIPS_MANIFEST_OVERRIDE=${CLIPS_MANIFEST_OVERRIDE:-}
CLIP_GROUPS_OVERRIDE=${CLIP_GROUPS_OVERRIDE:-}

usage() {
  cat <<'EOF'
Usage: run_audio_cvr_v1_b_first.sh [options]

Options:
  --root PATH
  --single-source-root PATH
  --run-root PATH
  --base-url URL
  --base-url-pool URL[,URL] weighted endpoint pool used by proposal shards
  --model NAME
  --max-source-folders N
  --max-clips N
  --max-b-candidates N       default: 0 (all B candidates)
  --propose-shards N
  --propose-parallel-jobs N
  --concurrency N
  --request-timeout-seconds N
  --shard-timeout-seconds N
  --target-b-count N
  --quality-profile b_audio_blind_review_v2|b_audio_blind_review_v2_volume
  --clips-manifest-override PATH
  --clip-groups-override PATH
  --resume
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --single-source-root) SINGLE_SOURCE_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --base-url-pool) BASE_URL_POOL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-source-folders) MAX_SOURCE_FOLDERS="$2"; shift 2 ;;
    --max-clips) MAX_CLIPS="$2"; shift 2 ;;
    --max-b-candidates) MAX_B_CANDIDATES="$2"; shift 2 ;;
    --propose-shards) PROPOSE_SHARDS="$2"; shift 2 ;;
    --propose-parallel-jobs) PROPOSE_PARALLEL_JOBS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --request-timeout-seconds) REQUEST_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --shard-timeout-seconds) SHARD_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --target-b-count) TARGET_B_COUNT="$2"; shift 2 ;;
    --quality-profile) QUALITY_PROFILE="$2"; shift 2 ;;
    --clips-manifest-override) CLIPS_MANIFEST_OVERRIDE="$2"; shift 2 ;;
    --clip-groups-override) CLIP_GROUPS_OVERRIDE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[audio-cvr-v1-b] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

export ROOT
export SINGLE_SOURCE_ROOT
export RUN_ROOT
export BASE_URL
export BASE_URL_POOL
export MODEL
export MAX_SOURCE_FOLDERS
export MAX_CLIPS
export MAX_B_CANDIDATES
export PROPOSE_SHARDS
export PROPOSE_PARALLEL_JOBS
export CONCURRENCY
export REQUEST_TIMEOUT_SECONDS
export SHARD_TIMEOUT_SECONDS
export ANNOTATION_TIMEOUT_SECONDS
export TARGET_B_COUNT
export AUDIO_DATASET_LINE=speech_audio_content
if [ "$QUALITY_PROFILE" != "b_audio_blind_review_v2" ] && [ "$QUALITY_PROFILE" != "b_audio_blind_review_v2_volume" ]; then
  echo "[audio-cvr-v1-b] unsupported quality profile: $QUALITY_PROFILE" >&2
  exit 2
fi
export AUDIO_LINE_QUALITY_PROFILE="$QUALITY_PROFILE"
export B_ACCEPTANCE_PROFILE="$QUALITY_PROFILE"
export B_CANDIDATE_MODE=audio_first
export A_CANDIDATE_MODE=omni_first
export MIN_CLIPS_PER_FOLDER=2
export MIN_GROUP_CLIPS=2
export KEEP_ALL_B=1
export FORCE_AUDIO_FOCUSED_REFRESH=1
export FRESH_ANNOTATIONS=1
export OMNI_TRANSIENT_RETRIES
export FAIL_ON_TRANSIENT_OMNI_ERRORS
export RESUME

echo "[audio-cvr-v1-b] run_root=$RUN_ROOT"
echo "[audio-cvr-v1-b] single_source_root=$SINGLE_SOURCE_ROOT"
echo "[audio-cvr-v1-b] B first, keep all accepted B samples, no A-line run in this script"

runner_args=(
  bash scripts/run_audio_lines_single_source_reuse.sh
  --root "$ROOT"
  --single-source-root "$SINGLE_SOURCE_ROOT"
  --run-root "$RUN_ROOT"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --run-b-only
  --target-b-count "$TARGET_B_COUNT"
  --max-source-folders "$MAX_SOURCE_FOLDERS"
  --max-clips "$MAX_CLIPS"
  --propose-shards "$PROPOSE_SHARDS"
  --propose-parallel-jobs "$PROPOSE_PARALLEL_JOBS"
  --concurrency "$CONCURRENCY"
  --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS"
  --shard-timeout-seconds "$SHARD_TIMEOUT_SECONDS"
  --audio-line-quality-profile "$AUDIO_LINE_QUALITY_PROFILE"
  --acceptance-profile "$B_ACCEPTANCE_PROFILE"
  --b-candidate-mode "$B_CANDIDATE_MODE"
  --min-clips-per-folder "$MIN_CLIPS_PER_FOLDER"
  --min-group-clips "$MIN_GROUP_CLIPS"
  --keep-all-b
  --fresh-annotations
  --force-audio-focused-refresh
  --omni-transient-retries "$OMNI_TRANSIENT_RETRIES"
)
if [ -n "$BASE_URL_POOL" ]; then
  runner_args+=(--base-url-pool "$BASE_URL_POOL")
fi
if [ -n "$CLIPS_MANIFEST_OVERRIDE" ] || [ -n "$CLIP_GROUPS_OVERRIDE" ]; then
  runner_args+=(
    --clips-manifest-override "$CLIPS_MANIFEST_OVERRIDE"
    --clip-groups-override "$CLIP_GROUPS_OVERRIDE"
  )
fi
if [ "$RESUME" = "1" ]; then
  runner_args+=(--resume)
fi
if [ "$FAIL_ON_TRANSIENT_OMNI_ERRORS" = "0" ]; then
  runner_args+=(--allow-transient-omni-fallback)
fi
"${runner_args[@]}"
