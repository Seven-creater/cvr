#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/finalize_audio_cvr_review_and_prepare_training.sh \
    --work-root WORK_ROOT \
    [--output-dir WORK_ROOT/benchmark_v1] \
    [--test-targets sound_event=120,music=30,speech_topic_in_video_context=0] \
    [--validation-targets sound_event=30,music=10,speech_topic_in_video_context=0]

WORK_ROOT must contain:
  review_pool/combined_pool_deduplicated.jsonl
  review_pool/automatic_review_candidates.jsonl
  reviews/pass1/shard_*.jsonl
  reviews/pass2/shard_*.jsonl

The script never reruns Omni review. It freezes the benchmark from completed
reviews, rebuilds train from the full deduplicated pool, and hard-fails on any
source/pair leakage. Inverse augmentation is intentionally a later train-only
step, after this audit succeeds.
EOF
}

WORK_ROOT=""
OUTPUT_DIR=""
TEST_TARGETS="sound_event=120,music=30,speech_topic_in_video_context=0"
VALIDATION_TARGETS="sound_event=30,music=10,speech_topic_in_video_context=0"
RANDOM_SEED=20260720

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --test-targets) TEST_TARGETS="$2"; shift 2 ;;
    --validation-targets) VALIDATION_TARGETS="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$WORK_ROOT" ]]; then
  echo "ERROR: --work-root is required" >&2
  usage
  exit 2
fi
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/benchmark_v1}"
POOL_DIR="$WORK_ROOT/review_pool"
CANDIDATE_PATH="$POOL_DIR/automatic_review_candidates.jsonl"
COMBINED_POOL_PATH="$POOL_DIR/combined_pool_deduplicated.jsonl"

for path in "$CANDIDATE_PATH" "$COMBINED_POOL_PATH"; do
  [[ -s "$path" ]] || { echo "ERROR: missing non-empty input: $path" >&2; exit 1; }
done

mapfile -t PASS1_PATHS < <(find "$WORK_ROOT/reviews/pass1" -maxdepth 1 -type f -name 'shard_*.jsonl' -size +0c | sort)
mapfile -t PASS2_PATHS < <(find "$WORK_ROOT/reviews/pass2" -maxdepth 1 -type f -name 'shard_*.jsonl' -size +0c | sort)
[[ "${#PASS1_PATHS[@]}" -gt 0 ]] || { echo "ERROR: no pass1 shard outputs found" >&2; exit 1; }
[[ "${#PASS2_PATHS[@]}" -gt 0 ]] || { echo "ERROR: no pass2 shard outputs found" >&2; exit 1; }

PASS1_ARGS=()
PASS2_ARGS=()
for path in "${PASS1_PATHS[@]}"; do PASS1_ARGS+=(--pass1-review-path "$path"); done
for path in "${PASS2_PATHS[@]}"; do PASS2_ARGS+=(--pass2-review-path "$path"); done

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/audit"
python3 -m app.audio_cvr_paper_experiment finalize-automatic-benchmark \
  --combined-pool-path "$COMBINED_POOL_PATH" \
  --candidate-path "$CANDIDATE_PATH" \
  "${PASS1_ARGS[@]}" \
  "${PASS2_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --subtype-targets "$TEST_TARGETS" \
  --validation-targets "$VALIDATION_TARGETS" \
  --repeat-review-fraction 0.20 \
  --max-dataset-ratio 0.50 \
  --relaxed-dataset-ratio 0.55 \
  --max-hdtf-ratio 0.15 \
  --max-voxceleb-ratio 0.05 \
  --max-per-source 1 \
  --random-seed "$RANDOM_SEED" \
  > "$OUTPUT_DIR/logs/finalize_automatic_benchmark.log" 2>&1

python3 -m app.audio_cvr_paper_experiment audit-training-splits \
  --train-path "$OUTPUT_DIR/train.jsonl" \
  --val-path "$OUTPUT_DIR/val.jsonl" \
  --test-path "$OUTPUT_DIR/test_main_150.jsonl" \
  --output-dir "$OUTPUT_DIR/audit" \
  > "$OUTPUT_DIR/logs/audit_training_splits.log" 2>&1

python3 - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "frozen_benchmark_manifest.json").read_text(encoding="utf-8"))
audit = json.loads((root / "audit" / "training_split_audit.json").read_text(encoding="utf-8"))
print("[post-review] benchmark freeze complete")
print("[post-review] test", manifest["test_final_count"], manifest["test_subtype_distribution"])
print("[post-review] val", audit["splits"]["val"]["count"], audit["splits"]["val"]["subtype_distribution"])
print("[post-review] train", audit["splits"]["train"]["count"], "sources", audit["splits"]["train"]["unique_source_count"])
print("[post-review] leakage violations", audit["leakage"]["violation_count"])
print("[post-review] next: run train-only inverse augmentation, then the E5 paper experiment")
PY
