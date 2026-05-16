#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
RUNS_ROOT=${RUNS_ROOT:-/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs}
DATASET_RUN_ROOT=${DATASET_RUN_ROOT:-}
RUN_ROOT=${RUN_ROOT:-$RUNS_ROOT/e5_audio_delta_smoke_$(date +%Y%m%d_%H%M%S)}
E5_MODEL=${E5_MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MAX_TRAIN_RECORDS=${MAX_TRAIN_RECORDS:-8}
MAX_EVAL_RECORDS=${MAX_EVAL_RECORDS:-4}
TRAIN_STEPS=${TRAIN_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-0.001}
DEVICE=${DEVICE:-cuda}
MOCK_ENCODER=${MOCK_ENCODER:-0}

usage() {
  cat <<'USAGE'
Usage: run_e5_audio_delta_smoke.sh [options]

Options:
  --dataset-run-root PATH   B-line run root with b_splits or b_main/b_extended outputs.
  --run-root PATH           Output run directory.
  --e5-model PATH           e5-omni model path.
  --gpu-ids IDS             CUDA_VISIBLE_DEVICES value, default 0,1,2,3.
  --max-train-records N     Few-shot train record count, default 8.
  --max-eval-records N      Few-shot eval record count, default 4.
  --train-steps N           Adapter steps, default 20.
  --batch-size N            Adapter batch size, default 4.
  --mock-encoder            Use deterministic fake embeddings for code smoke only.
USAGE
}

has_audio_delta_records() {
  local root="$1"
  test -f "$root/b_splits/train.jsonl" && return 0
  test -f "$root/b_train_bidirectional_triplets.jsonl" && return 0
  test -f "$root/b_main_audio_cvr_triplets.jsonl" && return 0
  test -f "$root/b_extended_audio_cvr_triplets.jsonl" && return 0
  test -f "$root/b_all_audio_cvr_triplets.jsonl" && return 0
  return 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset-run-root) DATASET_RUN_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max-train-records) MAX_TRAIN_RECORDS="$2"; shift 2 ;;
    --max-eval-records) MAX_EVAL_RECORDS="$2"; shift 2 ;;
    --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --mock-encoder) MOCK_ENCODER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[e5-audio-delta-smoke] unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cd "$REPO_ROOT"

if [ -z "$DATASET_RUN_ROOT" ]; then
  DATASET_RUN_ROOT=""
  shopt -s nullglob
  candidates=( "$RUNS_ROOT"/audio_cvr_bline_6_9s_full_* "$RUNS_ROOT"/audio_cvr_ab_6_9s_minimal_* )
  shopt -u nullglob
  if [ "${#candidates[@]}" -gt 0 ]; then
    while IFS= read -r candidate; do
      if has_audio_delta_records "$candidate"; then
        DATASET_RUN_ROOT="$candidate"
        break
      fi
    done < <(printf '%s\n' "${candidates[@]}" | xargs -r ls -td 2>/dev/null)
  fi
fi
if [ -z "$DATASET_RUN_ROOT" ] || [ ! -d "$DATASET_RUN_ROOT" ]; then
  echo "[e5-audio-delta-smoke] missing dataset run root; pass --dataset-run-root" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "[e5-audio-delta-smoke] repo=$REPO_ROOT"
echo "[e5-audio-delta-smoke] dataset_run_root=$DATASET_RUN_ROOT"
echo "[e5-audio-delta-smoke] run_root=$RUN_ROOT"
echo "[e5-audio-delta-smoke] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[e5-audio-delta-smoke] discovered training files:"
find "$DATASET_RUN_ROOT" -maxdepth 2 -type f \( \
  -name 'train.jsonl' -o \
  -name 'test_main.jsonl' -o \
  -name 'val.jsonl' -o \
  -name 'b_train_bidirectional_triplets.jsonl' -o \
  -name 'b_main_audio_cvr_triplets.jsonl' -o \
  -name 'b_extended_audio_cvr_triplets.jsonl' -o \
  -name 'b_all_audio_cvr_triplets.jsonl' \
\) -print | sort

python3 -m app.e5_audio_delta_train prepare \
  --dataset-run-root "$DATASET_RUN_ROOT" \
  --output-dir "$RUN_ROOT/records" \
  --max-train-records "$MAX_TRAIN_RECORDS" \
  --max-eval-records "$MAX_EVAL_RECORDS"

cache_args=()
if [ "$MOCK_ENCODER" = "1" ]; then
  cache_args+=(--mock-encoder)
else
  test -f "$E5_MODEL/config.json" || { echo "[e5-audio-delta-smoke] missing e5 config: $E5_MODEL/config.json" >&2; exit 1; }
  cache_args+=(--e5-model "$E5_MODEL")
fi

python3 -m app.e5_audio_delta_train cache-embeddings \
  --records-dir "$RUN_ROOT/records" \
  --output-dir "$RUN_ROOT/embedding_cache" \
  --device "$DEVICE" \
  "${cache_args[@]}"

python3 -m app.e5_audio_delta_train train-adapter \
  --cache-dir "$RUN_ROOT/embedding_cache" \
  --output-dir "$RUN_ROOT/adapter" \
  --steps "$TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --device "$DEVICE"

python3 -m app.e5_audio_delta_train eval \
  --cache-dir "$RUN_ROOT/embedding_cache" \
  --adapter-dir "$RUN_ROOT/adapter" \
  --output-dir "$RUN_ROOT/eval" \
  --device "$DEVICE"

cat "$RUN_ROOT/eval/comparison.md"
echo "[e5-audio-delta-smoke] done run_root=$RUN_ROOT"
