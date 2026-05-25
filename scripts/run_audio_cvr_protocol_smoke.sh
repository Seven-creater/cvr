#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_audio_cvr_protocol_smoke.sh \
    --run-root RUN_ROOT \
    --output-dir OUTPUT_DIR \
    --adapter-dir ADAPTER_DIR \
    [--gallery-size 1000] [--seed 13] [--max-train-records 64] [--max-eval-records 30] \
    [--protocols random,reference,local_same_source,typed_hardneg] \
    [--mine-local-same-source] [--local-same-source-candidates PATH] \
    [--video-audio-mode on|off] [--query-input-mode composed|text_only|video_only] [--mock-encoder]

This is a pilot convenience wrapper around the reusable protocol eval logic.
It does not train a new adapter and does not claim final benchmark numbers. It
prepares galleries, caches embeddings, runs eval with top-k diagnostics, and
writes summary tables.
EOF
}

RUN_ROOT=""
OUTPUT_DIR=""
ADAPTER_DIR=""
GALLERY_SIZE=1000
SEED=13
MAX_TRAIN_RECORDS=64
MAX_EVAL_RECORDS=30
PROTOCOLS="random,reference,local_same_source,typed_hardneg"
VIDEO_AUDIO_MODE="on"
QUERY_INPUT_MODE="composed"
MOCK_ENCODER=0
DEVICE="cuda"
LOCAL_SEGMENTS=0
REUSE_CACHE_FROM=""
MINE_LOCAL_SAME_SOURCE=0
LOCAL_SAME_SOURCE_CANDIDATES=""
MAX_LOCAL_SAME_SOURCE_PER_QUERY=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --adapter-dir) ADAPTER_DIR="$2"; shift 2 ;;
    --gallery-size) GALLERY_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-train-records) MAX_TRAIN_RECORDS="$2"; shift 2 ;;
    --max-eval-records) MAX_EVAL_RECORDS="$2"; shift 2 ;;
    --protocols) PROTOCOLS="$2"; shift 2 ;;
    --video-audio-mode) VIDEO_AUDIO_MODE="$2"; shift 2 ;;
    --query-input-mode) QUERY_INPUT_MODE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --local-segments) LOCAL_SEGMENTS="$2"; shift 2 ;;
    --reuse-cache-from) REUSE_CACHE_FROM="$2"; shift 2 ;;
    --mine-local-same-source) MINE_LOCAL_SAME_SOURCE=1; shift ;;
    --local-same-source-candidates) LOCAL_SAME_SOURCE_CANDIDATES="$2"; shift 2 ;;
    --max-local-same-source-per-query) MAX_LOCAL_SAME_SOURCE_PER_QUERY="$2"; shift 2 ;;
    --mock-encoder) MOCK_ENCODER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_ROOT" || -z "$OUTPUT_DIR" || -z "$ADAPTER_DIR" ]]; then
  echo "ERROR: --run-root, --output-dir, and --adapter-dir are required." >&2
  usage
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

python3 -m app.audio_cvr_protocol_eval summarize-data \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --run-label "1% Audio-CVR Protocol Smoke"

if [[ "$MINE_LOCAL_SAME_SOURCE" -eq 1 ]]; then
  if [[ -z "$LOCAL_SAME_SOURCE_CANDIDATES" ]]; then
    LOCAL_SAME_SOURCE_CANDIDATES="$OUTPUT_DIR/b_main_local_same_source_candidates.jsonl"
  fi
  python3 -m app.audio_cvr_protocol_eval mine-local-same-source \
    --run-root "$RUN_ROOT" \
    --input "$RUN_ROOT/b_main_audio_cvr_triplets.jsonl" \
    --output "$LOCAL_SAME_SOURCE_CANDIDATES" \
    --max-per-query "$MAX_LOCAL_SAME_SOURCE_PER_QUERY"
fi

IFS=',' read -ra PROTOCOL_ARRAY <<< "$PROTOCOLS"
EVAL_ARGS=()
for protocol in "${PROTOCOL_ARRAY[@]}"; do
  protocol="$(echo "$protocol" | xargs)"
  [[ -z "$protocol" ]] && continue
  records_dir="$OUTPUT_DIR/records_${protocol}"
  cache_dir="$OUTPUT_DIR/cache_${protocol}"
  eval_dir="$OUTPUT_DIR/eval_${protocol}"

  prepare_cmd=(
    python3 -m app.e5_audio_delta_train prepare
    --dataset-run-root "$RUN_ROOT" \
    --output-dir "$records_dir" \
    --max-train-records "$MAX_TRAIN_RECORDS" \
    --max-eval-records "$MAX_EVAL_RECORDS" \
    --eval-gallery-size "$GALLERY_SIZE" \
    --eval-gallery-protocol "$protocol" \
    --distractor-seed "$SEED"
  )
  if [[ -n "$LOCAL_SAME_SOURCE_CANDIDATES" ]]; then
    prepare_cmd+=(--local-same-source-candidates "$LOCAL_SAME_SOURCE_CANDIDATES")
  fi
  "${prepare_cmd[@]}"

  cache_cmd=(
    python3 -m app.e5_audio_delta_train cache-embeddings
    --records-dir "$records_dir"
    --output-dir "$cache_dir"
    --device "$DEVICE"
    --video-audio-mode "$VIDEO_AUDIO_MODE"
    --query-input-mode "$QUERY_INPUT_MODE"
    --local-segments "$LOCAL_SEGMENTS"
  )
  if [[ "$MOCK_ENCODER" -eq 1 ]]; then
    cache_cmd+=(--mock-encoder)
  fi
  if [[ -n "$REUSE_CACHE_FROM" ]]; then
    cache_cmd+=(--reuse-cache-from "$REUSE_CACHE_FROM")
  fi
  "${cache_cmd[@]}"

  python3 -m app.e5_audio_delta_train eval \
    --cache-dir "$cache_dir" \
    --adapter-dir "$ADAPTER_DIR" \
    --output-dir "$eval_dir" \
    --save-topk 10

  EVAL_ARGS+=(--eval "${protocol}=${eval_dir}")
done

python3 -m app.audio_cvr_protocol_eval summarize-evals \
  --output-dir "$OUTPUT_DIR" \
  --run-label "1% Audio-CVR Protocol Smoke" \
  "${EVAL_ARGS[@]}"

cat <<EOF
[audio-cvr-smoke] done
output_dir=$OUTPUT_DIR
data_quality=$OUTPUT_DIR/data_quality_summary.md
gallery_results=$OUTPUT_DIR/gallery_protocol_results.md
hard_negative_breakdown=$OUTPUT_DIR/hard_negative_breakdown.md
topk_errors=$OUTPUT_DIR/topk_errors.md
advisor_brief=$OUTPUT_DIR/advisor_brief.md
EOF
