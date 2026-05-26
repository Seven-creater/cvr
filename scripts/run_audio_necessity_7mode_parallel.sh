#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_audio_necessity_7mode_parallel.sh \
    --run-root RUN_ROOT \
    --output-dir OUTPUT_DIR \
    [--gpu-ids 1,2,3,4,5,6,7] \
    [--gallery-protocol typed_hardneg] [--gallery-size 1000] [--seed 26] \
    [--max-train-records 192] [--max-eval-records 64] \
    [--steps 120] [--batch-size 8] [--learning-rate 0.0003] \
    [--local-segments 0]

Runs the 7 Audio Necessity ablations in parallel across 7 GPUs:
  T-only-fullAV, V-only, A-only, V+T, A+T, V+A, V+A+T.

The adapter is trained once from the V+A+T cache and reused by all modes.
EOF
}

RUN_ROOT=""
OUTPUT_DIR=""
GPU_IDS="1,2,3,4,5,6,7"
GALLERY_PROTOCOL="typed_hardneg"
GALLERY_SIZE=1000
SEED=26
MAX_TRAIN_RECORDS=192
MAX_EVAL_RECORDS=64
STEPS=120
BATCH_SIZE=8
LEARNING_RATE=0.0003
LOCAL_SEGMENTS=0
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --gallery-protocol) GALLERY_PROTOCOL="$2"; shift 2 ;;
    --gallery-size) GALLERY_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-train-records) MAX_TRAIN_RECORDS="$2"; shift 2 ;;
    --max-eval-records) MAX_EVAL_RECORDS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --local-segments) LOCAL_SEGMENTS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_ROOT" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: --run-root and --output-dir are required." >&2
  usage
  exit 2
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -lt 7 ]]; then
  echo "ERROR: --gpu-ids must provide at least 7 GPU ids for the 7 modes." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"

RECORDS_DIR="$OUTPUT_DIR/records_${GALLERY_PROTOCOL}"
ADAPTER_DIR="$OUTPUT_DIR/adapter"

echo "[audio-necessity-7mode] run_root=$RUN_ROOT"
echo "[audio-necessity-7mode] output_dir=$OUTPUT_DIR"
echo "[audio-necessity-7mode] gpu_ids=$GPU_IDS"
echo "[audio-necessity-7mode] gallery_protocol=$GALLERY_PROTOCOL"

python3 -m app.audio_cvr_protocol_eval summarize-data \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --run-label "Audio Necessity 7-mode ${GALLERY_PROTOCOL}"

python3 -m app.e5_audio_delta_train prepare \
  --dataset-run-root "$RUN_ROOT" \
  --output-dir "$RECORDS_DIR" \
  --max-train-records "$MAX_TRAIN_RECORDS" \
  --max-eval-records "$MAX_EVAL_RECORDS" \
  --eval-gallery-size "$GALLERY_SIZE" \
  --eval-gallery-protocol "$GALLERY_PROTOCOL" \
  --distractor-seed "$SEED"

MODE_NAMES=(
  "T-only-fullAV"
  "V-only"
  "A-only"
  "V+T"
  "A+T"
  "V+A"
  "V+A+T"
)
MODE_SAFE=(
  "T_only_fullAV"
  "V_only"
  "A_only"
  "V_T"
  "A_T"
  "V_A"
  "V_A_T"
)
QUERY_MODES=(
  "text_only"
  "video_only"
  "audio_only"
  "composed"
  "audio_text"
  "video_only"
  "composed"
)
DOCUMENT_MODES=(
  "video"
  "video"
  "audio"
  "video"
  "audio"
  "video"
  "video"
)
VIDEO_AUDIO_MODES=(
  "on"
  "off"
  "off"
  "off"
  "off"
  "on"
  "on"
)

declare -a CACHE_PIDS=()
declare -a CACHE_LOGS=()

for index in "${!MODE_NAMES[@]}"; do
  mode="${MODE_NAMES[$index]}"
  safe="${MODE_SAFE[$index]}"
  gpu="${GPU_ARRAY[$index]}"
  cache_dir="$OUTPUT_DIR/cache_${safe}"
  log_path="$OUTPUT_DIR/logs/cache_${safe}.log"
  CACHE_LOGS+=("$log_path")
  echo "[audio-necessity-7mode] cache start mode=$mode gpu=$gpu log=$log_path"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 -m app.e5_audio_delta_train cache-embeddings \
      --records-dir "$RECORDS_DIR" \
      --output-dir "$cache_dir" \
      --device "$DEVICE" \
      --video-audio-mode "${VIDEO_AUDIO_MODES[$index]}" \
      --query-input-mode "${QUERY_MODES[$index]}" \
      --document-input-mode "${DOCUMENT_MODES[$index]}" \
      --audio-media-cache-dir "$cache_dir/audio_media_cache" \
      --local-segments "$LOCAL_SEGMENTS"
  ) > "$log_path" 2>&1 &
  CACHE_PIDS+=("$!")
done

for index in "${!CACHE_PIDS[@]}"; do
  pid="${CACHE_PIDS[$index]}"
  mode="${MODE_NAMES[$index]}"
  log_path="${CACHE_LOGS[$index]}"
  if ! wait "$pid"; then
    echo "[audio-necessity-7mode] cache failed mode=$mode log=$log_path" >&2
    tail -120 "$log_path" >&2 || true
    exit 1
  fi
  echo "[audio-necessity-7mode] cache done mode=$mode"
done

TRAIN_GPU="${GPU_ARRAY[6]}"
echo "[audio-necessity-7mode] train adapter gpu=$TRAIN_GPU"
(
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
  python3 -m app.e5_audio_delta_train train-adapter \
    --cache-dir "$OUTPUT_DIR/cache_V_A_T" \
    --output-dir "$ADAPTER_DIR" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --training-profile e5_omni_recipe
) > "$OUTPUT_DIR/logs/train_adapter.log" 2>&1

declare -a EVAL_PIDS=()
declare -a EVAL_ARGS=()
declare -a EVAL_LOGS=()

for index in "${!MODE_NAMES[@]}"; do
  mode="${MODE_NAMES[$index]}"
  safe="${MODE_SAFE[$index]}"
  gpu="${GPU_ARRAY[$index]}"
  eval_dir="$OUTPUT_DIR/eval_${safe}"
  log_path="$OUTPUT_DIR/logs/eval_${safe}.log"
  EVAL_ARGS+=(--eval "${mode}=${eval_dir}")
  EVAL_LOGS+=("$log_path")
  echo "[audio-necessity-7mode] eval start mode=$mode gpu=$gpu log=$log_path"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 -m app.e5_audio_delta_train eval \
      --cache-dir "$OUTPUT_DIR/cache_${safe}" \
      --adapter-dir "$ADAPTER_DIR" \
      --output-dir "$eval_dir" \
      --device "$DEVICE" \
      --save-topk 10
  ) > "$log_path" 2>&1 &
  EVAL_PIDS+=("$!")
done

for index in "${!EVAL_PIDS[@]}"; do
  pid="${EVAL_PIDS[$index]}"
  mode="${MODE_NAMES[$index]}"
  log_path="${EVAL_LOGS[$index]}"
  if ! wait "$pid"; then
    echo "[audio-necessity-7mode] eval failed mode=$mode log=$log_path" >&2
    tail -120 "$log_path" >&2 || true
    exit 1
  fi
  echo "[audio-necessity-7mode] eval done mode=$mode"
done

python3 -m app.audio_cvr_protocol_eval summarize-evals \
  --output-dir "$OUTPUT_DIR" \
  --run-label "Audio Necessity 7-mode ${GALLERY_PROTOCOL}" \
  "${EVAL_ARGS[@]}"

find "$OUTPUT_DIR" -maxdepth 3 \( -name "summary.json" -o -name "audio_necessity_results.md" -o -name "gallery_protocol_results.md" -o -name "hard_negative_breakdown.md" \) | sort > "$OUTPUT_DIR/result_files.txt"

cat <<EOF
[audio-necessity-7mode] done
output_dir=$OUTPUT_DIR
adapter_dir=$ADAPTER_DIR
audio_necessity=$OUTPUT_DIR/audio_necessity_results.md
gallery_results=$OUTPUT_DIR/gallery_protocol_results.md
hard_negative_breakdown=$OUTPUT_DIR/hard_negative_breakdown.md
result_files=$OUTPUT_DIR/result_files.txt
EOF
