#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  setsid nohup bash scripts/run_audio_cvr_fewshot_bidir_final.sh \
    --benchmark-root BENCHMARK_V1_FINAL150_VAL28 \
    --distractor-pool-path SINGLE_SOURCE_ANNOTATIONS_JSONL \
    --media-root CVR_MEDIA_ROOT \
    --output-dir OUTPUT_DIR \
    --expected-head GIT_COMMIT \
    --omni-model OMNI_MODEL_NAME \
    --omni-start-command 'python3 -m vllm.entrypoints.openai.api_server ...' \
    --e5-model /path/to/e5-omni-7B \
    > logs/audio_cvr_fewshot_bidir_final.log 2>&1 < /dev/null &

The launcher is resumable and implements the frozen non-speech Audio-CVR experiment:
  data audit -> verified train-only inverse augmentation -> exact Omni cleanup
  -> multi-GPU E5 encoding -> validation-only low-rank selection
  -> five-seed forward/bidirectional tests -> paired statistics.

The server must pull a clean GitHub commit before running this script. The launcher
never edits source code, never changes the frozen test, and never uses test metrics
for model selection.
EOF
}

BENCHMARK_ROOT=""
DISTRACTOR_POOL_PATH=""
MEDIA_ROOT=""
OUTPUT_DIR=""
EXPECTED_HEAD=""
EXPECTED_TEST_SHA256="f4b22e25e1f1262d488ff5474fdae9511301919611b42b9cc89f55c3aa633fd6"
EXPECTED_FORWARD_COUNT=65
OMNI_MODEL=""
OMNI_START_COMMAND=""
OMNI_BASE_URL="http://127.0.0.1:8093/v1"
OMNI_API_KEY="EMPTY"
OMNI_GPU_IDS="0,1,2,3"
E5_GPU_IDS="0,1,2,3,4,5"
E5_MODEL="/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B"
GALLERY_SIZE=1000
GALLERY_SEED=13
ADAPTER_RANKS="16,32"
COARSE_STEPS="50,100,200,400"
COARSE_LRS="0.0003,0.001"
COARSE_SEED=13
REFINE_SEEDS="13,23,42"
FINAL_SEEDS="13,23,42,71,101"
TOP_CONFIGS=4
BATCH_SIZE=8
BOOTSTRAP_SAMPLES=20000
PERMUTATION_SAMPLES=20000
VIDEO_FPS=1
VIDEO_MAX_PIXELS=401408

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --distractor-pool-path) DISTRACTOR_POOL_PATH="$2"; shift 2 ;;
    --media-root) MEDIA_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --expected-head) EXPECTED_HEAD="$2"; shift 2 ;;
    --expected-test-sha256) EXPECTED_TEST_SHA256="$2"; shift 2 ;;
    --expected-forward-count) EXPECTED_FORWARD_COUNT="$2"; shift 2 ;;
    --omni-model) OMNI_MODEL="$2"; shift 2 ;;
    --omni-start-command) OMNI_START_COMMAND="$2"; shift 2 ;;
    --omni-base-url) OMNI_BASE_URL="$2"; shift 2 ;;
    --omni-api-key) OMNI_API_KEY="$2"; shift 2 ;;
    --omni-gpu-ids) OMNI_GPU_IDS="$2"; shift 2 ;;
    --e5-gpu-ids) E5_GPU_IDS="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --gallery-size) GALLERY_SIZE="$2"; shift 2 ;;
    --gallery-seed) GALLERY_SEED="$2"; shift 2 ;;
    --adapter-ranks) ADAPTER_RANKS="$2"; shift 2 ;;
    --coarse-steps) COARSE_STEPS="$2"; shift 2 ;;
    --coarse-learning-rates) COARSE_LRS="$2"; shift 2 ;;
    --coarse-seed) COARSE_SEED="$2"; shift 2 ;;
    --refine-seeds) REFINE_SEEDS="$2"; shift 2 ;;
    --final-seeds) FINAL_SEEDS="$2"; shift 2 ;;
    --top-configs) TOP_CONFIGS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --bootstrap-samples) BOOTSTRAP_SAMPLES="$2"; shift 2 ;;
    --permutation-samples) PERMUTATION_SAMPLES="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --video-max-pixels) VIDEO_MAX_PIXELS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for value in BENCHMARK_ROOT DISTRACTOR_POOL_PATH MEDIA_ROOT OUTPUT_DIR EXPECTED_HEAD OMNI_MODEL OMNI_START_COMMAND; do
  if [[ -z "${!value}" ]]; then
    echo "ERROR: --$(tr '_' '-' <<< "${value,,}") is required" >&2
    exit 2
  fi
done
for path in "$BENCHMARK_ROOT/train.jsonl" "$BENCHMARK_ROOT/val.jsonl" \
  "$BENCHMARK_ROOT/test_main_150.jsonl" "$DISTRACTOR_POOL_PATH"; do
  [[ -s "$path" ]] || { echo "ERROR: required input is missing or empty: $path" >&2; exit 2; }
done

IFS=',' read -ra E5_GPU_ARRAY <<< "$E5_GPU_IDS"
if [[ "${#E5_GPU_ARRAY[@]}" -lt 6 ]]; then
  echo "ERROR: at least six E5 GPU ids are required" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"
STATUS_PATH="$OUTPUT_DIR/status.json"
STARTED_AT="$(date -Iseconds)"
RUN_STATE="RUNNING"
OMNI_PID=""
OMNI_PGID=""

write_status() {
  local state="$1" stage="$2" message="$3"
  python3 - "$STATUS_PATH" "$state" "$stage" "$message" "$STARTED_AT" <<'PY'
import json, pathlib, sys
path, state, stage, message, started = sys.argv[1:]
payload = {"state": state, "stage": stage, "message": message, "started_at": started}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

stop_omni() {
  [[ -n "$OMNI_PGID" ]] || return 0
  local own_pgid
  own_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  if [[ "$OMNI_PGID" == "$own_pgid" ]]; then
    echo "ERROR: refusing to kill launcher process group $OMNI_PGID" >&2
    return 1
  fi
  if kill -0 -- "-$OMNI_PGID" 2>/dev/null; then
    kill -TERM -- "-$OMNI_PGID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 -- "-$OMNI_PGID" 2>/dev/null || break
      sleep 2
    done
    if kill -0 -- "-$OMNI_PGID" 2>/dev/null; then
      kill -KILL -- "-$OMNI_PGID" 2>/dev/null || true
    fi
  fi
  wait "$OMNI_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! curl -fsS --max-time 2 "$OMNI_BASE_URL/models" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  if curl -fsS --max-time 2 "$OMNI_BASE_URL/models" >/dev/null 2>&1; then
    echo "ERROR: Omni API still responds after exact process-group cleanup" >&2
    return 1
  fi
  OMNI_PID=""
  OMNI_PGID=""
}

on_exit() {
  local code=$?
  stop_omni || true
  if [[ "$RUN_STATE" != "COMPLETE" ]]; then
    write_status "FAILED" "aborted" "launcher exited with code $code"
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_status "RUNNING" "git_and_data_audit" "checking GitHub code and frozen benchmark"
ACTUAL_HEAD="$(git rev-parse HEAD)"
[[ "$ACTUAL_HEAD" == "$EXPECTED_HEAD" ]] || { echo "ERROR: HEAD $ACTUAL_HEAD != $EXPECTED_HEAD" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: git status is not clean" >&2; git status --short >&2; exit 1; }
python3 -m py_compile app/e5_audio_delta_train.py app/audio_cvr_paper_experiment.py app/audio_lines_single_source.py
python3 -m unittest tests.test_e5_audio_delta_train tests.test_audio_cvr_paper_experiment -v \
  > "$OUTPUT_DIR/logs/unit_tests.log" 2>&1

DATA_DIR="$OUTPUT_DIR/data"
if [[ ! -s "$DATA_DIR/data_audit.json" ]]; then
  python3 -m app.audio_cvr_paper_experiment prepare-training-subset \
    --train-path "$BENCHMARK_ROOT/train.jsonl" \
    --val-path "$BENCHMARK_ROOT/val.jsonl" \
    --test-path "$BENCHMARK_ROOT/test_main_150.jsonl" \
    --output-dir "$DATA_DIR" \
    --eligible-subtypes sound_event,music \
    --expected-count "$EXPECTED_FORWARD_COUNT" \
    --expected-test-sha256 "$EXPECTED_TEST_SHA256" \
    --media-root "$MEDIA_ROOT" \
    --require-existing-media \
    > "$OUTPUT_DIR/logs/prepare_training_subset.log" 2>&1
fi
FORWARD_TRAIN="$DATA_DIR/train_non_speech_forward.jsonl"
cp "$DATA_DIR/data_audit.json" "$OUTPUT_DIR/data_audit.json"

INVERSE_ROOT="$OUTPUT_DIR/inverse_review"
if [[ ! -s "$INVERSE_ROOT/b_inverse_summary.json" ]]; then
  write_status "RUNNING" "inverse_review" "starting exact-process-group Omni inverse verification"
  IFS=',' read -ra OMNI_GPU_ARRAY <<< "$OMNI_GPU_IDS"
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${OMNI_GPU_ARRAY[*]}")"
  setsid bash -lc "exec $OMNI_START_COMMAND" > "$OUTPUT_DIR/logs/omni_server.log" 2>&1 &
  OMNI_PID=$!
  OMNI_PGID="$(ps -o pgid= -p "$OMNI_PID" | tr -d ' ')"
  [[ -n "$OMNI_PGID" ]] || { echo "ERROR: could not determine Omni process group" >&2; exit 1; }
  printf '%s\n' "$OMNI_PID" > "$INVERSE_ROOT.omni.pid"
  printf '%s\n' "$OMNI_PGID" > "$INVERSE_ROOT.omni.pgid"
  for _ in $(seq 1 180); do
    curl -fsS --max-time 3 "$OMNI_BASE_URL/models" >/dev/null 2>&1 && break
    kill -0 "$OMNI_PID" 2>/dev/null || { echo "ERROR: Omni process exited during startup" >&2; exit 1; }
    sleep 5
  done
  curl -fsS --max-time 3 "$OMNI_BASE_URL/models" >/dev/null 2>&1 || { echo "ERROR: Omni API did not become ready" >&2; exit 1; }
  python3 -m app.audio_lines_single_source augment-b-inverse \
    --run-root "$INVERSE_ROOT" \
    --input-path "$FORWARD_TRAIN" \
    --root "$MEDIA_ROOT" \
    --base-url "$OMNI_BASE_URL" \
    --api-key "$OMNI_API_KEY" \
    --model "$OMNI_MODEL" \
    --timeout-seconds 180 \
    --omni-retries 2 \
    --fail-on-transient-omni-errors \
    --resume \
    > "$OUTPUT_DIR/logs/augment_inverse.log" 2>&1
fi

write_status "RUNNING" "omni_cleanup" "stopping only the recorded Omni process group"
stop_omni
unset CUDA_VISIBLE_DEVICES
IFS=',' read -ra OMNI_GPU_ARRAY <<< "$OMNI_GPU_IDS"
for gpu in "${OMNI_GPU_ARRAY[@]}"; do
  residual="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "$residual" ]]; then
    echo "ERROR: GPU $gpu still has compute processes after Omni cleanup: $residual" >&2
    exit 1
  fi
done
cp "$INVERSE_ROOT/b_inverse_summary.json" "$OUTPUT_DIR/inverse_summary.json"
BIDIR_TRAIN="$INVERSE_ROOT/b_train_bidirectional_triplets.jsonl"
python3 -m app.audio_cvr_paper_experiment audit-training-splits \
  --train-path "$BIDIR_TRAIN" \
  --val-path "$BENCHMARK_ROOT/val.jsonl" \
  --test-path "$BENCHMARK_ROOT/test_main_150.jsonl" \
  --output-dir "$DATA_DIR/bidirectional_split_audit" \
  > "$OUTPUT_DIR/logs/audit_bidirectional_splits.log" 2>&1

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LOGS=()

wait_jobs() {
  local index
  for index in "${!ACTIVE_PIDS[@]}"; do
    if ! wait "${ACTIVE_PIDS[$index]}"; then
      echo "ERROR: parallel job failed; log=${ACTIVE_LOGS[$index]}" >&2
      tail -120 "${ACTIVE_LOGS[$index]}" >&2 || true
      return 1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LOGS=()
}

PREPARE_COMMON=(
  --dataset-run-root "$BENCHMARK_ROOT"
  --max-train-records 0
  --max-eval-records 0
  --eval-gallery-size "$GALLERY_SIZE"
  --eval-gallery-protocol typed_hardneg
  --distractor-pool-path "$DISTRACTOR_POOL_PATH"
  --distractor-seed "$GALLERY_SEED"
)

prepare_records() {
  local name="$1" train_path="$2" eval_path="$3"
  local records="$OUTPUT_DIR/records_$name"
  if [[ ! -s "$records/summary.json" ]]; then
    python3 -m app.e5_audio_delta_train prepare "${PREPARE_COMMON[@]}" \
      --output-dir "$records" --train-path "$train_path" --eval-path "$eval_path" \
      > "$OUTPUT_DIR/logs/prepare_${name}.log" 2>&1
  fi
}

write_status "RUNNING" "prepare_records" "building fixed train/validation/test records"
prepare_records forward_val "$FORWARD_TRAIN" "$BENCHMARK_ROOT/val.jsonl"
prepare_records bidir_val "$BIDIR_TRAIN" "$BENCHMARK_ROOT/val.jsonl"
prepare_records test "$FORWARD_TRAIN" "$BENCHMARK_ROOT/test_main_150.jsonl"

run_cache_sync() {
  local name="$1" records="$2" query_mode="$3" doc_mode="$4" audio_mode="$5" skip_train="$6" gpu="$7"
  local cache="$OUTPUT_DIR/cache_$name" log="$OUTPUT_DIR/logs/cache_${name}.log"
  if [[ -s "$cache/eval_embeddings.npz" && ( "$skip_train" == "yes" || -s "$cache/train_embeddings.npz" ) ]]; then
    return 0
  fi
  {
    export CUDA_VISIBLE_DEVICES="$gpu"
    args=(python3 -m app.e5_audio_delta_train cache-embeddings
      --records-dir "$records" --output-dir "$cache" --e5-model "$E5_MODEL"
      --device cuda --torch-dtype bfloat16 --video-fps "$VIDEO_FPS"
      --video-max-pixels "$VIDEO_MAX_PIXELS" --video-audio-mode "$audio_mode"
      --query-input-mode "$query_mode" --document-input-mode "$doc_mode"
      --audio-media-cache-dir "$cache/audio_media_cache" --local-segments 0)
    [[ "$skip_train" == "yes" ]] && args+=(--skip-train)
    "${args[@]}"
  } > "$log" 2>&1
}

launch_cache() {
  local name="$1" records="$2" query_mode="$3" doc_mode="$4" audio_mode="$5" skip_train="$6" gpu="$7"
  local log="$OUTPUT_DIR/logs/cache_${name}.log"
  (run_cache_sync "$name" "$records" "$query_mode" "$doc_mode" "$audio_mode" "$skip_train" "$gpu") &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("$log")
  if [[ "${#ACTIVE_PIDS[@]}" -ge "${#E5_GPU_ARRAY[@]}" ]]; then
    wait_jobs
  fi
}

write_status "RUNNING" "e5_encoding" "encoding train, validation, and seven frozen test modes"
gpu_cursor=0
launch_cache train_forward "$OUTPUT_DIR/records_forward_val" composed video on no "${E5_GPU_ARRAY[$gpu_cursor]}"; gpu_cursor=$((gpu_cursor+1))
launch_cache train_bidir "$OUTPUT_DIR/records_bidir_val" composed video on no "${E5_GPU_ARRAY[$gpu_cursor]}"; gpu_cursor=$((gpu_cursor+1))
launch_cache val_V_T "$OUTPUT_DIR/records_forward_val" composed video off yes "${E5_GPU_ARRAY[$gpu_cursor]}"; gpu_cursor=$((gpu_cursor+1))
launch_cache val_A_T "$OUTPUT_DIR/records_forward_val" audio_text audio off yes "${E5_GPU_ARRAY[$gpu_cursor]}"; gpu_cursor=$((gpu_cursor+1))
wait_jobs

MODE_NAMES=(T_only_fullAV V_only A_only V_T A_T V_A V_A_T)
QUERY_MODES=(text_only video_only audio_only composed audio_text video_only composed)
DOCUMENT_MODES=(video video audio video audio video video)
VIDEO_AUDIO_MODES=(on off off off off on on)
gpu_cursor=0
for index in "${!MODE_NAMES[@]}"; do
  gpu="${E5_GPU_ARRAY[$((gpu_cursor % ${#E5_GPU_ARRAY[@]}))]}"
  launch_cache "test_${MODE_NAMES[$index]}" "$OUTPUT_DIR/records_test" \
    "${QUERY_MODES[$index]}" "${DOCUMENT_MODES[$index]}" "${VIDEO_AUDIO_MODES[$index]}" yes "$gpu"
  gpu_cursor=$((gpu_cursor+1))
done
wait_jobs

launch_validation() {
  local stage="$1" architecture="$2" rank="$3" steps="$4" lr="$5" seed="$6" gpu="$7"
  local lr_safe="${lr//./p}"
  local run="$OUTPUT_DIR/validation/$stage/rank_${rank}_steps_${steps}_lr_${lr_safe}/seed_${seed}"
  local log="$OUTPUT_DIR/logs/validation_${stage}_r${rank}_s${steps}_lr${lr_safe}_seed${seed}.log"
  if [[ -s "$run/eval/summary.json" && -s "$run/adapter/adapter.pt" ]]; then
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 -m app.e5_audio_delta_train train-adapter \
      --cache-dir "$OUTPUT_DIR/cache_train_bidir" --output-dir "$run/adapter" \
      --steps "$steps" --batch-size "$BATCH_SIZE" --learning-rate "$lr" --seed "$seed" \
      --device cuda --training-profile e5_omni_recipe \
      --adapter-architecture "$architecture" --adapter-rank "$rank"
    python3 -m app.e5_audio_delta_train eval \
      --cache-dir "$OUTPUT_DIR/cache_train_bidir" --adapter-dir "$run/adapter" \
      --output-dir "$run/eval" --device cuda --topk 1,5,10
  ) > "$log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("$log")
  if [[ "${#ACTIVE_PIDS[@]}" -ge "${#E5_GPU_ARRAY[@]}" ]]; then
    wait_jobs
  fi
}

write_status "RUNNING" "validation_coarse" "running 16 low-rank configurations on val28 only"
IFS=',' read -ra RANK_ARRAY <<< "$ADAPTER_RANKS"
IFS=',' read -ra STEP_ARRAY <<< "$COARSE_STEPS"
IFS=',' read -ra LR_ARRAY <<< "$COARSE_LRS"
gpu_cursor=0
for rank in "${RANK_ARRAY[@]}"; do
  for steps in "${STEP_ARRAY[@]}"; do
    for lr in "${LR_ARRAY[@]}"; do
      launch_validation coarse low_rank_residual "$rank" "$steps" "$lr" "$COARSE_SEED" \
        "${E5_GPU_ARRAY[$((gpu_cursor % ${#E5_GPU_ARRAY[@]}))]}"
      gpu_cursor=$((gpu_cursor+1))
    done
  done
done
wait_jobs

COARSE_SELECTION="$OUTPUT_DIR/validation/coarse_selection"
python3 -m app.audio_cvr_paper_experiment summarize-validation \
  --input-root "$OUTPUT_DIR/validation/coarse" --output-dir "$COARSE_SELECTION" \
  --required-seeds "$COARSE_SEED" --top-n "$TOP_CONFIGS" \
  > "$OUTPUT_DIR/logs/summarize_validation_coarse.log" 2>&1

write_status "RUNNING" "validation_refine" "rechecking the top four configurations across three val seeds"
IFS=',' read -ra REFINE_SEED_ARRAY <<< "$REFINE_SEEDS"
gpu_cursor=0
while IFS=$'\t' read -r architecture rank steps lr batch; do
  [[ -n "$architecture" ]] || continue
  [[ "$batch" == "$BATCH_SIZE" ]] || { echo "ERROR: validation TSV batch changed unexpectedly" >&2; exit 1; }
  for seed in "${REFINE_SEED_ARRAY[@]}"; do
    launch_validation refine "$architecture" "$rank" "$steps" "$lr" "$seed" \
      "${E5_GPU_ARRAY[$((gpu_cursor % ${#E5_GPU_ARRAY[@]}))]}"
    gpu_cursor=$((gpu_cursor+1))
  done
done < "$COARSE_SELECTION/top_adapter_configs.tsv"
wait_jobs

FINAL_SELECTION="$OUTPUT_DIR/validation/final_selection"
python3 -m app.audio_cvr_paper_experiment summarize-validation \
  --input-root "$OUTPUT_DIR/validation/refine" --output-dir "$FINAL_SELECTION" \
  --required-seeds "$REFINE_SEEDS" --selection-rule one_se_earliest --top-n 1 \
  > "$OUTPUT_DIR/logs/summarize_validation_final.log" 2>&1
cp "$FINAL_SELECTION/validation_model_selection.json" "$OUTPUT_DIR/validation_selection.json"
IFS=$'\t' read -r SELECTED_ARCH SELECTED_RANK SELECTED_STEPS SELECTED_LR SELECTED_BATCH \
  < "$FINAL_SELECTION/selected_adapter_config.tsv"

launch_final_train() {
  local variant="$1" cache="$2" seed="$3" gpu="$4"
  local adapter="$OUTPUT_DIR/final_$variant/seed_${seed}/adapter"
  local log="$OUTPUT_DIR/logs/final_train_${variant}_seed${seed}.log"
  [[ -s "$adapter/adapter.pt" ]] && return 0
  mkdir -p "$adapter"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 -m app.e5_audio_delta_train train-adapter \
      --cache-dir "$cache" --output-dir "$adapter" --steps "$SELECTED_STEPS" \
      --batch-size "$SELECTED_BATCH" --learning-rate "$SELECTED_LR" --seed "$seed" \
      --device cuda --training-profile e5_omni_recipe \
      --adapter-architecture "$SELECTED_ARCH" --adapter-rank "$SELECTED_RANK"
  ) > "$log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("$log")
  if [[ "${#ACTIVE_PIDS[@]}" -ge "${#E5_GPU_ARRAY[@]}" ]]; then
    wait_jobs
  fi
}

write_status "RUNNING" "final_training" "training forward-only and verified-bidirectional adapters with five fixed seeds"
IFS=',' read -ra FINAL_SEED_ARRAY <<< "$FINAL_SEEDS"
gpu_cursor=0
for variant in forward_only forward_bidir; do
  cache="$OUTPUT_DIR/cache_train_forward"
  [[ "$variant" == "forward_bidir" ]] && cache="$OUTPUT_DIR/cache_train_bidir"
  for seed in "${FINAL_SEED_ARRAY[@]}"; do
    launch_final_train "$variant" "$cache" "$seed" "${E5_GPU_ARRAY[$((gpu_cursor % ${#E5_GPU_ARRAY[@]}))]}"
    gpu_cursor=$((gpu_cursor+1))
  done
done
wait_jobs

for variant in forward_only forward_bidir; do
  for seed in "${FINAL_SEED_ARRAY[@]}"; do
    fusion_dir="$OUTPUT_DIR/final_$variant/seed_${seed}/fusion_selection"
    if [[ ! -s "$fusion_dir/selected_alpha.json" ]]; then
      export CUDA_VISIBLE_DEVICES="${E5_GPU_ARRAY[0]}"
      python3 -m app.audio_cvr_paper_experiment score-fusion \
        --cache-a "$OUTPUT_DIR/cache_val_V_T" --cache-b "$OUTPUT_DIR/cache_val_A_T" \
        --adapter-dir "$OUTPUT_DIR/final_$variant/seed_${seed}/adapter" \
        --output-dir "$fusion_dir" --alpha-grid 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
        --device cuda --save-topk 1 \
        > "$OUTPUT_DIR/logs/fusion_select_${variant}_seed${seed}.log" 2>&1
      unset CUDA_VISIBLE_DEVICES
    fi
  done
done

launch_eval() {
  local variant="$1" seed="$2" mode="$3" cache="$4" suffix="$5" gpu="$6"
  local eval_dir="$OUTPUT_DIR/final_$variant/seed_${seed}/eval_${mode}${suffix}"
  local log="$OUTPUT_DIR/logs/eval_${variant}_${mode}${suffix}_seed${seed}.log"
  [[ -s "$eval_dir/summary.json" && -s "$eval_dir/per_query_scores.jsonl" ]] && return 0
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    args=(python3 -m app.e5_audio_delta_train eval --cache-dir "$cache"
      --adapter-dir "$OUTPUT_DIR/final_$variant/seed_${seed}/adapter"
      --output-dir "$eval_dir" --device cuda --topk 1,5,10 --save-topk 20)
    [[ "$suffix" == "_no_ref" ]] && args+=(--exclude-gallery-kind reference_negative)
    "${args[@]}"
  ) > "$log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("$log")
  if [[ "${#ACTIVE_PIDS[@]}" -ge "${#E5_GPU_ARRAY[@]}" ]]; then
    wait_jobs
  fi
}

write_status "RUNNING" "final_evaluation" "evaluating seven modes and exact with/without-reference pairs"
gpu_cursor=0
for variant in forward_only forward_bidir; do
  for seed in "${FINAL_SEED_ARRAY[@]}"; do
    for mode in "${MODE_NAMES[@]}"; do
      launch_eval "$variant" "$seed" "$mode" "$OUTPUT_DIR/cache_test_${mode}" "" \
        "${E5_GPU_ARRAY[$((gpu_cursor % ${#E5_GPU_ARRAY[@]}))]}"
      gpu_cursor=$((gpu_cursor+1))
    done
    for mode in V_T V_A_T; do
      launch_eval "$variant" "$seed" "$mode" "$OUTPUT_DIR/cache_test_${mode}" "_no_ref" \
        "${E5_GPU_ARRAY[$((gpu_cursor % ${#E5_GPU_ARRAY[@]}))]}"
      gpu_cursor=$((gpu_cursor+1))
    done
  done
done
wait_jobs

python3 - "$OUTPUT_DIR" "$FINAL_SEEDS" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2].split(",") if value]
violations = []
rows = []
for variant in ("forward_only", "forward_bidir"):
    for seed in seeds:
        for mode in ("V_T", "V_A_T"):
            with_path = root / f"final_{variant}" / f"seed_{seed}" / f"eval_{mode}" / "summary.json"
            without_path = root / f"final_{variant}" / f"seed_{seed}" / f"eval_{mode}_no_ref" / "summary.json"
            with_ref = json.loads(with_path.read_text(encoding="utf-8"))
            without_ref = json.loads(without_path.read_text(encoding="utf-8"))
            checks = {
                "same_cache": with_ref.get("cache_dir") == without_ref.get("cache_dir"),
                "same_raw_gallery_count": with_ref.get("gallery_count") == without_ref.get("gallery_count"),
                "with_reference": bool(with_ref.get("reference_in_gallery")),
                "without_reference": not bool(without_ref.get("reference_in_gallery")),
                "only_reference_excluded": without_ref.get("excluded_gallery_kinds") == ["reference_negative"],
                "effective_count_exact": int(without_ref.get("effective_gallery_count", -1))
                == int(with_ref.get("gallery_count", 0)) - int(without_ref.get("excluded_gallery_count", 0)),
            }
            if not all(checks.values()):
                violations.append({"variant": variant, "seed": seed, "mode": mode, "checks": checks})
            rows.append({"variant": variant, "seed": seed, "mode": mode, **checks})
audit = {"protocol": "exact_reference_removal_same_cache_v1", "rows": rows, "violation_count": len(violations), "violations": violations}
(root / "reference_exclusion_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
if violations:
    raise SystemExit(f"reference exclusion audit failed: {violations[:2]}")
PY

for variant in forward_only forward_bidir; do
  for seed in "${FINAL_SEED_ARRAY[@]}"; do
    eval_dir="$OUTPUT_DIR/final_$variant/seed_${seed}/eval_late_fusion"
    [[ -s "$eval_dir/summary.json" ]] && continue
    alpha="$(python3 - "$OUTPUT_DIR/final_$variant/seed_${seed}/fusion_selection/selected_alpha.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["alpha_cache_a"])
PY
)"
    export CUDA_VISIBLE_DEVICES="${E5_GPU_ARRAY[0]}"
    python3 -m app.audio_cvr_paper_experiment score-fusion \
      --cache-a "$OUTPUT_DIR/cache_test_V_T" --cache-b "$OUTPUT_DIR/cache_test_A_T" \
      --adapter-dir "$OUTPUT_DIR/final_$variant/seed_${seed}/adapter" \
      --output-dir "$eval_dir" --alpha "$alpha" --device cuda --save-topk 20 \
      > "$OUTPUT_DIR/logs/fusion_test_${variant}_seed${seed}.log" 2>&1
    unset CUDA_VISIBLE_DEVICES
  done
done

write_status "RUNNING" "statistics" "running paired bootstrap, randomization, McNemar, and Holm correction"
for variant in forward_only forward_bidir; do
  python3 -m app.audio_cvr_paper_experiment aggregate-final \
    --input-root "$OUTPUT_DIR/final_$variant" --output-dir "$OUTPUT_DIR/statistics_$variant" \
    --required-seeds "$FINAL_SEEDS" --primary-mode V_A_T --reference-mode V_T \
    --comparison V_A_T:V_T --comparison V_A_T_no_ref:V_A_T --comparison V_T_no_ref:V_T \
    --comparison late_fusion:V_A_T --comparison V_A_T:V_A --comparison A_T:A_only \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" --permutation-samples "$PERMUTATION_SAMPLES" \
    > "$OUTPUT_DIR/logs/aggregate_${variant}.log" 2>&1
done

VARIANT_ROOT="$OUTPUT_DIR/variant_comparison_matrix"
for seed in "${FINAL_SEED_ARRAY[@]}"; do
  mkdir -p "$VARIANT_ROOT/seed_${seed}"
  ln -sfn "$(realpath "$OUTPUT_DIR/final_forward_only/seed_${seed}/eval_V_A_T")" \
    "$VARIANT_ROOT/seed_${seed}/eval_Forward_only"
  ln -sfn "$(realpath "$OUTPUT_DIR/final_forward_bidir/seed_${seed}/eval_V_A_T")" \
    "$VARIANT_ROOT/seed_${seed}/eval_Forward_Bidir"
done
python3 -m app.audio_cvr_paper_experiment aggregate-final \
  --input-root "$VARIANT_ROOT" --output-dir "$OUTPUT_DIR/statistics_variant_comparison" \
  --required-seeds "$FINAL_SEEDS" --primary-mode Forward_Bidir --reference-mode Forward_only \
  --comparison Forward_Bidir:Forward_only --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --permutation-samples "$PERMUTATION_SAMPLES" \
  > "$OUTPUT_DIR/logs/aggregate_variant_comparison.log" 2>&1

python3 - "$OUTPUT_DIR" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
bad = []
curve_count = 0
for path in root.glob("**/loss_curve.jsonl"):
    curve_count += 1
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
            bad.append({"path": str(path), "line": line_no})
audit = {"loss_curve_count": curve_count, "non_finite_count": len(bad), "violations": bad}
(root / "loss_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
if bad:
    raise SystemExit(f"non-finite loss values found: {bad[:5]}")

stats = root / "statistics_forward_only"
for source, target in (
    (stats / "per_seed_results.json", root / "per_seed_results.json"),
    (stats / "test_main_mean_std.json", root / "final_mean_std.json"),
    (stats / "paired_comparisons.json", root / "paired_comparisons.json"),
    (stats / "error_breakdown.json", root / "error_breakdown.json"),
):
    target.write_bytes(source.read_bytes())

selection = json.loads((root / "validation_selection.json").read_text(encoding="utf-8"))["selected_config"]
inverse = json.loads((root / "inverse_summary.json").read_text(encoding="utf-8"))
lines = [
    "# Audio-CVR Few-Shot Adapter Final Results",
    "",
    f"- Adapter: `{selection['adapter_architecture']}`, rank `{selection['adapter_rank']}`.",
    f"- Validation-selected steps/LR/batch: `{selection['steps']}` / `{selection['learning_rate']}` / `{selection['batch_size']}`.",
    f"- Independent forward source pairs: `{inverse['input_count']}`.",
    f"- Omni-accepted inverse records: `{inverse['accepted_count']}`.",
    "- Test selection: frozen before tuning; all final statistics use five seeds.",
    "",
    "## Result Files",
    "",
    "- `statistics_forward_only/test_main_comparison.md`: primary seven-mode table.",
    "- `statistics_forward_only/paired_comparisons.md`: primary audio and reference counterfactual tests.",
    "- `statistics_variant_comparison/paired_comparisons.md`: verified-bidirectional augmentation versus forward-only.",
    "- `statistics_forward_bidir/test_main_comparison.md`: verified-bidirectional augmentation ablation.",
]
(root / "paper_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

find "$OUTPUT_DIR" -maxdepth 7 \( -name '*.json' -o -name '*.md' -o -name '*.jsonl' \) | sort \
  > "$OUTPUT_DIR/result_files.txt"
RUN_STATE="COMPLETE"
write_status "COMPLETE" "complete" "all frozen few-shot bidirectional experiments completed"

echo "[audiocvr-fewshot] COMPLETE"
echo "output_dir=$OUTPUT_DIR"
echo "selected_architecture=$SELECTED_ARCH"
echo "selected_rank=$SELECTED_RANK"
echo "selected_steps=$SELECTED_STEPS"
echo "selected_learning_rate=$SELECTED_LR"
echo "paper_results=$OUTPUT_DIR/paper_results.md"
