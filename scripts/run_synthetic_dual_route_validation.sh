#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/synthetic_dual_route_validation}
KNOWN_PAIRS=${KNOWN_PAIRS:-$RUN_ROOT/synthetic_candidate_pairs.jsonl}
CLIP_ANNOTATIONS=${CLIP_ANNOTATIONS:-$RUN_ROOT/synthetic_annotations.jsonl}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
API_KEY=${API_KEY:-EMPTY}
MODEL=${MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct}
GPU_IDS=${GPU_IDS:-0,1}
MAX_GPUS=${MAX_GPUS:-2}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}
MAX_ACCEPTED_PAIRS=${MAX_ACCEPTED_PAIRS:-10}
OVERWRITE=${OVERWRITE:-1}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --known-pairs) KNOWN_PAIRS="$2"; shift 2 ;;
    --clip-annotations) CLIP_ANNOTATIONS="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-accepted-pairs) MAX_ACCEPTED_PAIRS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

GPU_COUNT=$(python3 - <<PY
ids = [item for item in "$GPU_IDS".split(",") if item.strip()]
print(len(ids))
PY
)
if [[ "$GPU_COUNT" -gt "$MAX_GPUS" ]]; then
  echo "[synthetic-dual] refusing to run with GPU_COUNT=$GPU_COUNT > MAX_GPUS=$MAX_GPUS" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
echo "[synthetic-dual] one Omni model per run; validation expects existing service at $BASE_URL"
echo "[synthetic-dual] run_root=$RUN_ROOT"
echo "[synthetic-dual] known_pairs=$KNOWN_PAIRS"
echo "[synthetic-dual] clip_annotations=$CLIP_ANNOTATIONS"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
VALIDATE_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  VALIDATE_ARGS+=(--overwrite)
fi

python -m app.composed_data validate-known-pairs \
  --root "$RUN_ROOT" \
  --known-pairs-path "$KNOWN_PAIRS" \
  --clip-annotations-path "$CLIP_ANNOTATIONS" \
  --output-path "$RUN_ROOT/judged_synthetic_pair_proposals.jsonl" \
  --accepted-output-path "$RUN_ROOT/accepted_synthetic_pairs.jsonl" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --model "$MODEL" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --max-accepted-pairs "$MAX_ACCEPTED_PAIRS" \
  "${VALIDATE_ARGS[@]}"

if [[ -s "$RUN_ROOT/accepted_synthetic_pairs.jsonl" ]]; then
  python -m app.composed_data validate-pilot \
    --root "$RUN_ROOT" \
    --pilot-jsonl-path "$RUN_ROOT/accepted_synthetic_pairs.jsonl" \
    --gallery-output-path "$RUN_ROOT/synthetic_gallery.jsonl" \
    --report-output-path "$RUN_ROOT/synthetic_pilot_review.md"
else
  echo "[synthetic-dual] accepted_synthetic_pairs.jsonl is empty; skip validate-pilot"
fi
