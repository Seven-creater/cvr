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
SYNTHETIC_SMOKE=${SYNTHETIC_SMOKE:-0}
LOCAL_SEGMENTS=${LOCAL_SEGMENTS:-0}

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
  --device cpu|cuda         Training device, default cuda.
  --local-segments N        Cache N temporal local views per video, default 0.
  --mock-encoder            Use deterministic fake embeddings for code smoke only.
  --synthetic-smoke         Create tiny synthetic records and force mock encoder.
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
    --device) DEVICE="$2"; shift 2 ;;
    --local-segments) LOCAL_SEGMENTS="$2"; shift 2 ;;
    --mock-encoder) MOCK_ENCODER=1; shift ;;
    --synthetic-smoke) SYNTHETIC_SMOKE=1; MOCK_ENCODER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[e5-audio-delta-smoke] unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cd "$REPO_ROOT"

if [ "$SYNTHETIC_SMOKE" = "1" ]; then
  DATASET_RUN_ROOT="$RUN_ROOT/synthetic_dataset_run"
  mkdir -p "$DATASET_RUN_ROOT"
  python3 - "$DATASET_RUN_ROOT/b_all_audio_cvr_triplets.jsonl" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for index in range(1, 5):
    old_audio = f"reference topic {index}"
    new_audio = f"target topic {index}"
    rows.append({
        "sample_id": f"synthetic_b_{index:03d}",
        "reference_video": f"/synthetic/reference_{index:03d}.mp4",
        "target_video": f"/synthetic/target_{index:03d}.mp4",
        "edit_text": f"change the speech from discussing {old_audio} to discussing {new_audio}",
        "edit_type": "replace",
        "audio_delta_type": "speech_topic",
        "old_audio": old_audio,
        "new_audio": new_audio,
        "direction": "forward",
        "split_tier": "extended",
        "raw_source_id": f"synthetic_source_{index:03d}",
        "pair_group_id": f"synthetic_pair_{index:03d}",
        "inverse_pair_group_id": f"synthetic_pair_{index:03d}",
        "shortcut_label": "synthetic_code_smoke",
        "audio_delta_strength": 0.8,
        "video_context_strength": 0.7,
        "asr_degeneracy_risk": 0.2,
        "visual_shortcut_risk": 0.1,
        "full_av_required": True,
        "audio_delta_hard_negatives": [
            {"type": "reference_negative", "video": f"/synthetic/reference_{index:03d}.mp4"},
            {"type": "visual_hard", "video": f"/synthetic/visual_hard_{index:03d}.mp4"},
            {"type": "audio_hard", "video": f"/synthetic/audio_hard_{index:03d}.mp4"},
            {"type": "asr_hard", "video": f"/synthetic/asr_hard_{index:03d}.mp4"},
        ],
    })
out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
print(f"[e5-audio-delta-smoke] wrote synthetic records: {out} rows={len(rows)}")
PY
elif [ -z "$DATASET_RUN_ROOT" ]; then
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
echo "[e5-audio-delta-smoke] synthetic_smoke=$SYNTHETIC_SMOKE"
echo "[e5-audio-delta-smoke] mock_encoder=$MOCK_ENCODER"
echo "[e5-audio-delta-smoke] local_segments=$LOCAL_SEGMENTS"
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
  --local-segments "$LOCAL_SEGMENTS" \
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
