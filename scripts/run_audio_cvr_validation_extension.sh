#!/usr/bin/env bash
set -euo pipefail

# Validation-only extension for an already encoded V+A+T train/validation cache.
# It never reads a test cache and is safe to run while a new benchmark is built.

CACHE_DIR=""
OUTPUT_DIR=""
GPU_IDS="0"
STEPS_GRID="700,1000,1300,1600"
SEEDS="13,23,42"
LEARNING_RATE="0.001"
BATCH_SIZE="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --steps-grid) STEPS_GRID="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$CACHE_DIR" && -n "$OUTPUT_DIR" ]] || {
  echo "usage: $0 --cache-dir <TRAIN_VAL_CACHE> --output-dir <OUT> [options]" >&2
  exit 2
}
[[ -s "$CACHE_DIR/train_embeddings.npz" && -s "$CACHE_DIR/eval_embeddings.npz" ]] || {
  echo "cache must contain train_embeddings.npz and eval_embeddings.npz: $CACHE_DIR" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR/logs"
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
IFS=',' read -ra STEP_ARRAY <<< "$STEPS_GRID"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
[[ "${#GPU_ARRAY[@]}" -gt 0 ]] || { echo "no GPU ids supplied" >&2; exit 2; }

ACTIVE_PIDS=()
ACTIVE_LOGS=()

wait_active_jobs() {
  local failed=0
  for index in "${!ACTIVE_PIDS[@]}"; do
    if ! wait "${ACTIVE_PIDS[$index]}"; then
      echo "validation job failed; see ${ACTIVE_LOGS[$index]}" >&2
      failed=1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LOGS=()
  [[ "$failed" -eq 0 ]]
}

gpu_cursor=0
for steps in "${STEP_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    run_dir="$OUTPUT_DIR/grid/steps_${steps}_lr_${LEARNING_RATE}_batch_${BATCH_SIZE}/seed_${seed}"
    adapter_dir="$run_dir/adapter"
    eval_dir="$run_dir/eval"
    log_path="$OUTPUT_DIR/logs/steps${steps}_seed${seed}.log"
    if [[ -s "$eval_dir/summary.json" && -s "$eval_dir/per_query_scores.jsonl" ]]; then
      continue
    fi
    gpu="${GPU_ARRAY[$gpu_cursor]}"
    gpu_cursor=$(( (gpu_cursor + 1) % ${#GPU_ARRAY[@]} ))
    mkdir -p "$adapter_dir" "$eval_dir"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python3 -m app.e5_audio_delta_train train-adapter \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$adapter_dir" \
        --steps "$steps" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --seed "$seed" \
        --device cuda \
        --training-profile e5_omni_recipe
      python3 -m app.e5_audio_delta_train eval \
        --cache-dir "$CACHE_DIR" \
        --adapter-dir "$adapter_dir" \
        --output-dir "$eval_dir" \
        --device cuda \
        --topk 1,5,10 \
        --save-topk 10
    ) > "$log_path" 2>&1 &
    ACTIVE_PIDS+=("$!")
    ACTIVE_LOGS+=("$log_path")
    if [[ "${#ACTIVE_PIDS[@]}" -ge "${#GPU_ARRAY[@]}" ]]; then
      wait_active_jobs
    fi
  done
done
wait_active_jobs

python3 -m app.audio_cvr_paper_experiment summarize-validation \
  --input-root "$OUTPUT_DIR/grid" \
  --output-dir "$OUTPUT_DIR/selection" \
  --required-seeds "$SEEDS" \
  --selection-rule one_se_earliest \
  --top-n 4 \
  > "$OUTPUT_DIR/logs/summarize_validation.log" 2>&1

echo "validation extension complete"
echo "selection=$OUTPUT_DIR/selection/validation_model_selection.md"
