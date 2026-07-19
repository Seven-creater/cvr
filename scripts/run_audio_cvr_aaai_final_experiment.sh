#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_audio_cvr_aaai_final_experiment.sh \
    --run-root RUN_ROOT \
    --output-dir OUTPUT_DIR \
    [--split-root RUN_ROOT/b_splits] \
    [--gpu-ids 1,2,3,4,5,6,7]

This is the paper-grade Audio-CVR experiment:
  1. Preserve the existing source-disjoint assignment and filter test to B-main.
  2. Encode one real E5 V+A+T train/validation cache.
  3. Select steps/LR/batch using validation only (coarse seed 13, then 3-seed refinement).
  4. Lock the selected configuration and train five final adapter seeds.
  5. Encode and evaluate all seven audio-necessity modes on one shared test gallery.
  6. Produce mean/std, paired bootstrap CI, randomization tests, McNemar tests, and error analysis.

The script is resumable: completed caches, adapters, and eval outputs are reused.
It does not modify data, enable LoRA, or enable AudioDelta task-specific losses.
EOF
}

RUN_ROOT=""
SPLIT_ROOT=""
OUTPUT_DIR=""
GPU_IDS="1,2,3,4,5,6,7"
GALLERY_SIZE=1000
GALLERY_SEED=13
E5_MODEL="/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B"
COARSE_STEPS="60,120,240,450,700,1000"
COARSE_LRS="0.0001,0.0003,0.001"
COARSE_BATCHES="4,8,16"
COARSE_SEED=13
REFINE_SEEDS="13,23,42"
FINAL_SEEDS="13,23,42,71,101"
TOP_CONFIGS=6
BOOTSTRAP_SAMPLES=20000
PERMUTATION_SAMPLES=20000
VIDEO_FPS=1
VIDEO_MAX_PIXELS=401408

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --split-root) SPLIT_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --gallery-size) GALLERY_SIZE="$2"; shift 2 ;;
    --gallery-seed) GALLERY_SEED="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --coarse-steps) COARSE_STEPS="$2"; shift 2 ;;
    --coarse-learning-rates) COARSE_LRS="$2"; shift 2 ;;
    --coarse-batch-sizes) COARSE_BATCHES="$2"; shift 2 ;;
    --coarse-seed) COARSE_SEED="$2"; shift 2 ;;
    --refine-seeds) REFINE_SEEDS="$2"; shift 2 ;;
    --final-seeds) FINAL_SEEDS="$2"; shift 2 ;;
    --top-configs) TOP_CONFIGS="$2"; shift 2 ;;
    --bootstrap-samples) BOOTSTRAP_SAMPLES="$2"; shift 2 ;;
    --permutation-samples) PERMUTATION_SAMPLES="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --video-max-pixels) VIDEO_MAX_PIXELS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_ROOT" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: --run-root and --output-dir are required." >&2
  usage
  exit 2
fi
SPLIT_ROOT="${SPLIT_ROOT:-$RUN_ROOT/b_splits}"

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -lt 7 ]]; then
  echo "ERROR: this launcher requires seven GPU ids so each real-E5 ablation owns one GPU." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"
STATUS_PATH="$OUTPUT_DIR/status.json"
STARTED_AT="$(date -Iseconds)"

write_status() {
  local state="$1"
  local message="$2"
  python3 - "$STATUS_PATH" "$state" "$message" "$STARTED_AT" <<'PY'
import json, pathlib, sys
path, state, message, started = sys.argv[1:]
payload = {"state": state, "message": message, "started_at": started}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

on_error() {
  local code=$?
  write_status "FAILED" "command failed at line ${BASH_LINENO[0]} with exit code $code"
  echo "[aaai-final] FAILED exit=$code line=${BASH_LINENO[0]}" >&2
  exit "$code"
}
trap on_error ERR
write_status "RUNNING" "paper experiment started"

echo "[aaai-final] commit=$(git rev-parse HEAD)"
echo "[aaai-final] run_root=$RUN_ROOT"
echo "[aaai-final] split_root=$SPLIT_ROOT"
echo "[aaai-final] output_dir=$OUTPUT_DIR"
echo "[aaai-final] gpu_ids=$GPU_IDS"

PAPER_SPLITS="$OUTPUT_DIR/paper_splits"
if [[ ! -s "$PAPER_SPLITS/split_verification.json" ]]; then
  python3 -m app.audio_cvr_paper_experiment prepare-splits \
    --split-root "$SPLIT_ROOT" \
    --output-dir "$PAPER_SPLITS" \
    > "$OUTPUT_DIR/logs/prepare_splits.log" 2>&1
fi

TRAIN_VAL_RECORDS="$OUTPUT_DIR/records_train_val"
if [[ ! -s "$TRAIN_VAL_RECORDS/summary.json" ]]; then
  python3 -m app.e5_audio_delta_train prepare \
    --dataset-run-root "$RUN_ROOT" \
    --output-dir "$TRAIN_VAL_RECORDS" \
    --train-path "$PAPER_SPLITS/train.jsonl" \
    --eval-path "$PAPER_SPLITS/val.jsonl" \
    --max-train-records 0 \
    --max-eval-records 0 \
    --eval-gallery-size "$GALLERY_SIZE" \
    --eval-gallery-protocol typed_hardneg \
    --distractor-seed "$GALLERY_SEED" \
    > "$OUTPUT_DIR/logs/prepare_train_val.log" 2>&1
fi

TRAIN_VAL_CACHE="$OUTPUT_DIR/cache_train_val_V_A_T"
if [[ ! -s "$TRAIN_VAL_CACHE/train_embeddings.npz" || ! -s "$TRAIN_VAL_CACHE/eval_embeddings.npz" ]]; then
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}"
    python3 -m app.e5_audio_delta_train cache-embeddings \
      --records-dir "$TRAIN_VAL_RECORDS" \
      --output-dir "$TRAIN_VAL_CACHE" \
      --e5-model "$E5_MODEL" \
      --device cuda \
      --torch-dtype bfloat16 \
      --video-fps "$VIDEO_FPS" \
      --video-max-pixels "$VIDEO_MAX_PIXELS" \
      --video-audio-mode on \
      --query-input-mode composed \
      --document-input-mode video \
      --local-segments 0
  ) > "$OUTPUT_DIR/logs/cache_train_val_V_A_T.log" 2>&1
fi

# CoVA reports that explicit audio/visual fusion is an important baseline.
# Encode validation V+T and A+T once so the late-fusion weight is selected
# without looking at test data.
declare -a FUSION_CACHE_PIDS=()
declare -a FUSION_CACHE_LOGS=()
VAL_FUSION_MODES=("V_T" "A_T")
VAL_FUSION_QUERY=("composed" "audio_text")
VAL_FUSION_DOCUMENT=("video" "audio")
VAL_FUSION_AUDIO=("off" "off")
for index in "${!VAL_FUSION_MODES[@]}"; do
  mode="${VAL_FUSION_MODES[$index]}"
  cache_dir="$OUTPUT_DIR/cache_val_${mode}"
  log_path="$OUTPUT_DIR/logs/cache_val_${mode}.log"
  if [[ -s "$cache_dir/eval_embeddings.npz" ]]; then
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
    python3 -m app.e5_audio_delta_train cache-embeddings \
      --records-dir "$TRAIN_VAL_RECORDS" \
      --output-dir "$cache_dir" \
      --e5-model "$E5_MODEL" \
      --device cuda \
      --torch-dtype bfloat16 \
      --video-fps "$VIDEO_FPS" \
      --video-max-pixels "$VIDEO_MAX_PIXELS" \
      --video-audio-mode "${VAL_FUSION_AUDIO[$index]}" \
      --query-input-mode "${VAL_FUSION_QUERY[$index]}" \
      --document-input-mode "${VAL_FUSION_DOCUMENT[$index]}" \
      --audio-media-cache-dir "$cache_dir/audio_media_cache" \
      --local-segments 0 \
      --skip-train
  ) > "$log_path" 2>&1 &
  FUSION_CACHE_PIDS+=("$!")
  FUSION_CACHE_LOGS+=("$log_path")
done
for index in "${!FUSION_CACHE_PIDS[@]}"; do
  if ! wait "${FUSION_CACHE_PIDS[$index]}"; then
    echo "[aaai-final] validation fusion cache failed log=${FUSION_CACHE_LOGS[$index]}" >&2
    tail -120 "${FUSION_CACHE_LOGS[$index]}" >&2 || true
    exit 1
  fi
done

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LOGS=()

wait_active_jobs() {
  local index
  for index in "${!ACTIVE_PIDS[@]}"; do
    if ! wait "${ACTIVE_PIDS[$index]}"; then
      echo "[aaai-final] parallel job failed log=${ACTIVE_LOGS[$index]}" >&2
      tail -120 "${ACTIVE_LOGS[$index]}" >&2 || true
      return 1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LOGS=()
}

launch_validation_run() {
  local stage="$1"
  local steps="$2"
  local lr="$3"
  local batch="$4"
  local seed="$5"
  local gpu="$6"
  local lr_safe="${lr//./p}"
  local run_dir="$OUTPUT_DIR/validation/$stage/steps_${steps}_lr_${lr_safe}_bs_${batch}/seed_${seed}"
  local log_path="$OUTPUT_DIR/logs/validation_${stage}_s${steps}_lr${lr_safe}_b${batch}_seed${seed}.log"
  if [[ -s "$run_dir/eval/summary.json" && -s "$run_dir/adapter/adapter.pt" ]]; then
    return 0
  fi
  mkdir -p "$run_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 -m app.e5_audio_delta_train train-adapter \
      --cache-dir "$TRAIN_VAL_CACHE" \
      --output-dir "$run_dir/adapter" \
      --steps "$steps" \
      --batch-size "$batch" \
      --learning-rate "$lr" \
      --seed "$seed" \
      --device cuda \
      --training-profile e5_omni_recipe
    python3 -m app.e5_audio_delta_train eval \
      --cache-dir "$TRAIN_VAL_CACHE" \
      --adapter-dir "$run_dir/adapter" \
      --output-dir "$run_dir/eval" \
      --device cuda \
      --topk 1,5,10
  ) > "$log_path" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("$log_path")
  if [[ "${#ACTIVE_PIDS[@]}" -ge "${#GPU_ARRAY[@]}" ]]; then
    wait_active_jobs
  fi
}

IFS=',' read -ra STEP_ARRAY <<< "$COARSE_STEPS"
IFS=',' read -ra LR_ARRAY <<< "$COARSE_LRS"
IFS=',' read -ra BATCH_ARRAY <<< "$COARSE_BATCHES"
gpu_cursor=0
for steps in "${STEP_ARRAY[@]}"; do
  for lr in "${LR_ARRAY[@]}"; do
    for batch in "${BATCH_ARRAY[@]}"; do
      launch_validation_run "coarse" "$steps" "$lr" "$batch" "$COARSE_SEED" "${GPU_ARRAY[$gpu_cursor]}"
      gpu_cursor=$(( (gpu_cursor + 1) % ${#GPU_ARRAY[@]} ))
    done
  done
done
wait_active_jobs

COARSE_SELECTION="$OUTPUT_DIR/validation/coarse_selection"
python3 -m app.audio_cvr_paper_experiment summarize-validation \
  --input-root "$OUTPUT_DIR/validation/coarse" \
  --output-dir "$COARSE_SELECTION" \
  --required-seeds "$COARSE_SEED" \
  --top-n "$TOP_CONFIGS" \
  > "$OUTPUT_DIR/logs/summarize_validation_coarse.log" 2>&1

IFS=',' read -ra REFINE_SEED_ARRAY <<< "$REFINE_SEEDS"
gpu_cursor=0
while IFS=$'\t' read -r steps lr batch; do
  [[ -z "$steps" ]] && continue
  for seed in "${REFINE_SEED_ARRAY[@]}"; do
    launch_validation_run "refine" "$steps" "$lr" "$batch" "$seed" "${GPU_ARRAY[$gpu_cursor]}"
    gpu_cursor=$(( (gpu_cursor + 1) % ${#GPU_ARRAY[@]} ))
  done
done < "$COARSE_SELECTION/top_configs.tsv"
wait_active_jobs

FINAL_SELECTION="$OUTPUT_DIR/validation/final_selection"
python3 -m app.audio_cvr_paper_experiment summarize-validation \
  --input-root "$OUTPUT_DIR/validation/refine" \
  --output-dir "$FINAL_SELECTION" \
  --required-seeds "$REFINE_SEEDS" \
  --selection-rule one_se_earliest \
  --top-n 1 \
  > "$OUTPUT_DIR/logs/summarize_validation_final.log" 2>&1

IFS=$'\t' read -r SELECTED_STEPS SELECTED_LR SELECTED_BATCH < "$FINAL_SELECTION/selected_config.tsv"
echo "[aaai-final] selected steps=$SELECTED_STEPS lr=$SELECTED_LR batch=$SELECTED_BATCH"

TEST_RECORDS="$OUTPUT_DIR/records_test_main"
if [[ ! -s "$TEST_RECORDS/summary.json" ]]; then
  python3 -m app.e5_audio_delta_train prepare \
    --dataset-run-root "$RUN_ROOT" \
    --output-dir "$TEST_RECORDS" \
    --train-path "$PAPER_SPLITS/train.jsonl" \
    --eval-path "$PAPER_SPLITS/test_main.jsonl" \
    --max-train-records 0 \
    --max-eval-records 0 \
    --eval-gallery-size "$GALLERY_SIZE" \
    --eval-gallery-protocol typed_hardneg \
    --distractor-seed "$GALLERY_SEED" \
    > "$OUTPUT_DIR/logs/prepare_test_main.log" 2>&1
fi

MODE_NAMES=("T_only_fullAV" "V_only" "A_only" "V_T" "A_T" "V_A" "V_A_T")
QUERY_MODES=("text_only" "video_only" "audio_only" "composed" "audio_text" "video_only" "composed")
DOCUMENT_MODES=("video" "video" "audio" "video" "audio" "video" "video")
VIDEO_AUDIO_MODES=("on" "off" "off" "off" "off" "on" "on")

ACTIVE_PIDS=()
ACTIVE_LOGS=()
for index in "${!MODE_NAMES[@]}"; do
  mode="${MODE_NAMES[$index]}"
  gpu="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  cache_dir="$OUTPUT_DIR/cache_test_${mode}"
  log_path="$OUTPUT_DIR/logs/cache_test_${mode}.log"
  if [[ -s "$cache_dir/eval_embeddings.npz" ]]; then
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 -m app.e5_audio_delta_train cache-embeddings \
      --records-dir "$TEST_RECORDS" \
      --output-dir "$cache_dir" \
      --e5-model "$E5_MODEL" \
      --device cuda \
      --torch-dtype bfloat16 \
      --video-fps "$VIDEO_FPS" \
      --video-max-pixels "$VIDEO_MAX_PIXELS" \
      --video-audio-mode "${VIDEO_AUDIO_MODES[$index]}" \
      --query-input-mode "${QUERY_MODES[$index]}" \
      --document-input-mode "${DOCUMENT_MODES[$index]}" \
      --audio-media-cache-dir "$cache_dir/audio_media_cache" \
      --local-segments 0 \
      --skip-train
  ) > "$log_path" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("$log_path")
done
wait_active_jobs

IFS=',' read -ra FINAL_SEED_ARRAY <<< "$FINAL_SEEDS"
ACTIVE_PIDS=()
ACTIVE_LOGS=()
for index in "${!FINAL_SEED_ARRAY[@]}"; do
  seed="${FINAL_SEED_ARRAY[$index]}"
  gpu="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  adapter_dir="$OUTPUT_DIR/final/seed_${seed}/adapter"
  log_path="$OUTPUT_DIR/logs/final_train_seed${seed}.log"
  if [[ -s "$adapter_dir/adapter.pt" ]]; then
    continue
  fi
  mkdir -p "$adapter_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 -m app.e5_audio_delta_train train-adapter \
      --cache-dir "$TRAIN_VAL_CACHE" \
      --output-dir "$adapter_dir" \
      --steps "$SELECTED_STEPS" \
      --batch-size "$SELECTED_BATCH" \
      --learning-rate "$SELECTED_LR" \
      --seed "$seed" \
      --device cuda \
      --training-profile e5_omni_recipe
  ) > "$log_path" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("$log_path")
done
wait_active_jobs

for seed in "${FINAL_SEED_ARRAY[@]}"; do
  fusion_selection="$OUTPUT_DIR/final/seed_${seed}/fusion_selection"
  if [[ ! -s "$fusion_selection/selected_alpha.json" ]]; then
    gpu="${GPU_ARRAY[0]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python3 -m app.audio_cvr_paper_experiment score-fusion \
        --cache-a "$OUTPUT_DIR/cache_val_V_T" \
        --cache-b "$OUTPUT_DIR/cache_val_A_T" \
        --adapter-dir "$OUTPUT_DIR/final/seed_${seed}/adapter" \
        --output-dir "$fusion_selection" \
        --alpha-grid 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
        --device cuda \
        --save-topk 1
    ) > "$OUTPUT_DIR/logs/fusion_select_seed${seed}.log" 2>&1
  fi
done

ACTIVE_PIDS=()
ACTIVE_LOGS=()
gpu_cursor=0
for seed in "${FINAL_SEED_ARRAY[@]}"; do
  for mode in "${MODE_NAMES[@]}"; do
    gpu="${GPU_ARRAY[$gpu_cursor]}"
    eval_dir="$OUTPUT_DIR/final/seed_${seed}/eval_${mode}"
    log_path="$OUTPUT_DIR/logs/final_eval_${mode}_seed${seed}.log"
    gpu_cursor=$(( (gpu_cursor + 1) % ${#GPU_ARRAY[@]} ))
    if [[ -s "$eval_dir/summary.json" && -s "$eval_dir/per_query_scores.jsonl" ]]; then
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python3 -m app.e5_audio_delta_train eval \
        --cache-dir "$OUTPUT_DIR/cache_test_${mode}" \
        --adapter-dir "$OUTPUT_DIR/final/seed_${seed}/adapter" \
        --output-dir "$eval_dir" \
        --device cuda \
        --topk 1,5,10 \
        --save-topk 20
    ) > "$log_path" 2>&1 &
    ACTIVE_PIDS+=("$!")
    ACTIVE_LOGS+=("$log_path")
    if [[ "${#ACTIVE_PIDS[@]}" -ge "${#GPU_ARRAY[@]}" ]]; then
      wait_active_jobs
    fi
  done
done
wait_active_jobs

ACTIVE_PIDS=()
ACTIVE_LOGS=()
gpu_cursor=0
for seed in "${FINAL_SEED_ARRAY[@]}"; do
  eval_dir="$OUTPUT_DIR/final/seed_${seed}/eval_late_fusion"
  log_path="$OUTPUT_DIR/logs/final_eval_late_fusion_seed${seed}.log"
  if [[ -s "$eval_dir/summary.json" && -s "$eval_dir/per_query_scores.jsonl" ]]; then
    continue
  fi
  alpha="$(python3 - "$OUTPUT_DIR/final/seed_${seed}/fusion_selection/selected_alpha.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["alpha_cache_a"])
PY
)"
  gpu="${GPU_ARRAY[$gpu_cursor]}"
  gpu_cursor=$(( (gpu_cursor + 1) % ${#GPU_ARRAY[@]} ))
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 -m app.audio_cvr_paper_experiment score-fusion \
      --cache-a "$OUTPUT_DIR/cache_test_V_T" \
      --cache-b "$OUTPUT_DIR/cache_test_A_T" \
      --adapter-dir "$OUTPUT_DIR/final/seed_${seed}/adapter" \
      --output-dir "$eval_dir" \
      --alpha "$alpha" \
      --device cuda \
      --save-topk 20
  ) > "$log_path" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("$log_path")
done
wait_active_jobs

python3 -m app.audio_cvr_paper_experiment aggregate-final \
  --input-root "$OUTPUT_DIR/final" \
  --output-dir "$OUTPUT_DIR/statistics" \
  --required-seeds "$FINAL_SEEDS" \
  --primary-mode V_A_T \
  --reference-mode V_T \
  --comparison V_A_T:V_T \
  --comparison V_A_T:V_A \
  --comparison V_A_T:V_only \
  --comparison V_T:V_only \
  --comparison A_T:A_only \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --permutation-samples "$PERMUTATION_SAMPLES" \
  > "$OUTPUT_DIR/logs/aggregate_final.log" 2>&1

find "$OUTPUT_DIR" -maxdepth 5 \( -name "summary.json" -o -name "*.md" -o -name "audit.json" -o -name "*selection.json" \) \
  | sort > "$OUTPUT_DIR/result_files.txt"

write_status "COMPLETE" "paper experiment completed successfully"
trap - ERR

cat <<EOF
[aaai-final] COMPLETE
output_dir=$OUTPUT_DIR
selected_steps=$SELECTED_STEPS
selected_learning_rate=$SELECTED_LR
selected_batch_size=$SELECTED_BATCH
validation=$FINAL_SELECTION/validation_model_selection.md
final_results=$OUTPUT_DIR/statistics/test_main_comparison.md
audio_gain=$OUTPUT_DIR/statistics/audio_gain_summary.md
audit=$OUTPUT_DIR/statistics/audit.json
review_manifest=$PAPER_SPLITS/test_main_human_review.jsonl
EOF
