#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VGG_RUN=""
FIXED_TEST=""
LEGACY_POOL=""
FRESH_POOL=""
VGG_MEDIA_ROOT=""
LEGACY_MEDIA_ROOT=""
FRESH_MEDIA_ROOT=""
OUT_ROOT=""
MODEL="qwen3-omni-30b-a3b-instruct"
ENDPOINTS="http://127.0.0.1:8093/v1,http://127.0.0.1:8094/v1,http://127.0.0.1:8095/v1,http://127.0.0.1:8096/v1"
SHARD_COUNT=64
PARALLEL_JOBS=32
WAIT_SECONDS=30
RANDOM_SEED=20260720
TARGET_COUNT=1000
MAX_SPEECH_COUNT=50
EXCLUDE_PATHS=()

usage() {
  cat <<'EOF'
Usage: run_audio_cvr_fixed_test1000_fill.sh \
  --vgg-run PATH --fixed-test PATH --legacy-pool PATH --fresh-pool PATH \
  --vgg-media-root PATH --legacy-media-root PATH --fresh-media-root PATH \
  --exclude-path TRAIN_OR_VAL_JSONL [--exclude-path ...] --out-root PATH

The launcher waits for all VGG candidate shards, snapshots final accepted rows,
reviews only source-disjoint additions on four existing Omni endpoints, and
freezes an immutable Test1000. It never starts or stops Omni services.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vgg-run) VGG_RUN="$2"; shift 2 ;;
    --fixed-test) FIXED_TEST="$2"; shift 2 ;;
    --legacy-pool) LEGACY_POOL="$2"; shift 2 ;;
    --fresh-pool) FRESH_POOL="$2"; shift 2 ;;
    --vgg-media-root) VGG_MEDIA_ROOT="$2"; shift 2 ;;
    --legacy-media-root) LEGACY_MEDIA_ROOT="$2"; shift 2 ;;
    --fresh-media-root) FRESH_MEDIA_ROOT="$2"; shift 2 ;;
    --exclude-path) EXCLUDE_PATHS+=("$2"); shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --endpoints) ENDPOINTS="$2"; shift 2 ;;
    --shard-count) SHARD_COUNT="$2"; shift 2 ;;
    --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --target-count) TARGET_COUNT="$2"; shift 2 ;;
    --max-speech-count) MAX_SPEECH_COUNT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for name in VGG_RUN FIXED_TEST LEGACY_POOL FRESH_POOL VGG_MEDIA_ROOT LEGACY_MEDIA_ROOT FRESH_MEDIA_ROOT OUT_ROOT; do
  [[ -n "${!name}" ]] || { echo "Missing required option: $name" >&2; usage >&2; exit 2; }
done
for path in "$FIXED_TEST" "$LEGACY_POOL" "$FRESH_POOL"; do
  [[ -s "$path" ]] || { echo "Missing non-empty input: $path" >&2; exit 2; }
done
[[ ${#EXCLUDE_PATHS[@]} -gt 0 ]] || { echo "At least one --exclude-path is required" >&2; exit 2; }
for path in "${EXCLUDE_PATHS[@]}"; do
  [[ -s "$path" ]] || { echo "Missing protected train/val input: $path" >&2; exit 2; }
done

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/input_snapshots" \
  "$OUT_ROOT/review_pool" "$OUT_ROOT/reviews/pass1" "$OUT_ROOT/reviews/pass2"
STATUS="$OUT_ROOT/status.json"
RUN_STATE="FAILED"

write_status() {
  python3 - "$STATUS" "$1" "$2" "$3" <<'PY'
import json, os, sys, tempfile, time
path, state, stage, detail = sys.argv[1:]
payload = {"state": state, "stage": stage, "detail": detail, "launcher_pid": os.getppid(), "updated_unix": time.time()}
fd, tmp = tempfile.mkstemp(prefix=".status.", dir=os.path.dirname(path))
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp, path)
PY
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ "$RUN_STATE" != "COMPLETE" ]]; then
    write_status "FAILED" "launcher" "exit_code=$code; all review shards and snapshots are preserved"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

progress_counts() {
  python3 - "$VGG_RUN" <<'PY'
from pathlib import Path
import json
import sys
root = Path(sys.argv[1]) / "b_shards"

def proposal_ids(paths):
    ids = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("proposal_id") or row.get("candidate_id") or "").strip()
                if key:
                    ids.add(key)
    return ids

candidates = proposal_ids(root.glob("b_shard_*.jsonl"))
ranked = proposal_ids(root.glob("ranked_*.jsonl"))
accepted = proposal_ids(root.glob("accepted_[0-9]*.jsonl"))
rejected = proposal_ids(root.glob("rejected_[0-9]*.jsonl"))
print(len(candidates), len(ranked), len(accepted), len(rejected), len(accepted & rejected))
PY
}

write_status "RUNNING" "waiting_vgg" "waiting for all VGG candidate progress to be durable"
while true; do
  read -r candidate ranked accepted rejected overlap < <(progress_counts)
  write_status "RUNNING" "waiting_vgg" "candidate_unique=$candidate ranked_unique=$ranked final_accepted_unique=$accepted final_rejected_unique=$rejected terminal_overlap=$overlap"
  if (( candidate > 0 && ranked >= candidate )); then break; fi
  sleep "$WAIT_SECONDS"
done

VGG_SNAPSHOT="$OUT_ROOT/input_snapshots/vgg_accepted_snapshot.jsonl"
python3 - "$VGG_RUN" "$VGG_SNAPSHOT" <<'PY'
from pathlib import Path
import json, os, sys, tempfile
root, output = Path(sys.argv[1]) / "b_shards", Path(sys.argv[2])
rows = {}
for path in sorted(root.glob("accepted_[0-9]*.jsonl")):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        key = str(row.get("proposal_id") or row.get("candidate_id") or line)
        rows[key] = row
fd, tmp = tempfile.mkstemp(prefix=".vgg_snapshot.", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    for key in sorted(rows): handle.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    handle.flush(); os.fsync(handle.fileno())
os.replace(tmp, output)
PY

EXCLUDE_ARGS=()
for path in "${EXCLUDE_PATHS[@]}"; do EXCLUDE_ARGS+=(--exclude-path "$path"); done

write_status "RUNNING" "prepare_review_pool" "deduplicating fixed-test additions without retrieval scores"
python3 -m app.audio_cvr_paper_experiment prepare-fixed-test-fill-review \
  --fixed-test-path "$FIXED_TEST" \
  --input-path "$VGG_SNAPSHOT" --input-media-root "$VGG_MEDIA_ROOT" \
  --input-path "$LEGACY_POOL" --input-media-root "$LEGACY_MEDIA_ROOT" \
  --input-path "$FRESH_POOL" --input-media-root "$FRESH_MEDIA_ROOT" \
  "${EXCLUDE_ARGS[@]}" \
  --output-dir "$OUT_ROOT/review_pool" --max-per-source 1 --random-seed "$RANDOM_SEED" \
  > "$OUT_ROOT/logs/prepare_review_pool.log" 2>&1

CANDIDATES="$OUT_ROOT/review_pool/automatic_review_candidates.jsonl"
IFS=',' read -r -a URLS <<< "$ENDPOINTS"
[[ ${#URLS[@]} -eq 4 ]] || { echo "Exactly four Omni endpoints are required" >&2; exit 2; }

run_review_pass() {
  local pass_id="$1" output_dir="$2"
  local -a pass1_args=()
  if [[ "$pass_id" == "2" ]]; then
    for path in "$OUT_ROOT"/reviews/pass1/shard_*.jsonl; do
      [[ -f "$path" ]] && pass1_args+=(--pass1-review-path "$path")
    done
  fi
  local running=0
  for ((shard=0; shard<SHARD_COUNT; shard++)); do
    endpoint="${URLS[$((shard % ${#URLS[@]}))]}"
    python3 -m app.audio_cvr_paper_experiment review-benchmark-omni \
      --candidate-path "$CANDIDATES" \
      --output-path "$output_dir/shard_$(printf '%02d' "$shard").jsonl" \
      --media-root / --cache-dir "$OUT_ROOT/omni_cache" \
      --base-url "$endpoint" --api-key EMPTY --model "$MODEL" \
      --review-pass-id "$pass_id" "${pass1_args[@]}" \
      --repeat-review-fraction 0.20 --random-seed "$RANDOM_SEED" \
      --shard-index "$shard" --shard-count "$SHARD_COUNT" \
      --timeout-seconds 240 --omni-retries 2 --resume \
      > "$OUT_ROOT/logs/pass${pass_id}_shard_$(printf '%02d' "$shard").log" 2>&1 &
    running=$((running + 1))
    if (( running >= PARALLEL_JOBS )); then wait -n; running=$((running - 1)); fi
  done
  wait
}

write_status "RUNNING" "omni_review_pass1" "running $SHARD_COUNT shards across four endpoints"
run_review_pass 1 "$OUT_ROOT/reviews/pass1"
write_status "RUNNING" "omni_review_pass2" "repeating deterministic 20 percent audit"
run_review_pass 2 "$OUT_ROOT/reviews/pass2"

PASS1_ARGS=(); PASS2_ARGS=()
for path in "$OUT_ROOT"/reviews/pass1/shard_*.jsonl; do [[ -f "$path" ]] && PASS1_ARGS+=(--pass1-review-path "$path"); done
for path in "$OUT_ROOT"/reviews/pass2/shard_*.jsonl; do [[ -f "$path" ]] && PASS2_ARGS+=(--pass2-review-path "$path"); done
[[ ${#PASS1_ARGS[@]} -gt 0 && ${#PASS2_ARGS[@]} -gt 0 ]] || { echo "Review outputs are incomplete" >&2; exit 1; }

write_status "RUNNING" "finalizing_test1000" "freezing fixed 516 plus model-verified additions"
python3 -m app.audio_cvr_paper_experiment finalize-fixed-test-fill \
  --fixed-test-path "$FIXED_TEST" --candidate-path "$CANDIDATES" \
  "${PASS1_ARGS[@]}" "${PASS2_ARGS[@]}" "${EXCLUDE_ARGS[@]}" \
  --output-dir "$OUT_ROOT/final_test1000" --target-count "$TARGET_COUNT" \
  --max-speech-count "$MAX_SPEECH_COUNT" --sound-event-ratio 0.80 \
  --repeat-review-fraction 0.20 --random-seed "$RANDOM_SEED" \
  > "$OUT_ROOT/logs/finalize_test1000.log" 2>&1

RUN_STATE="COMPLETE"
write_status "COMPLETE" "test1000_frozen" "Test1000 frozen; Omni services remain running for the recorded handoff"
echo "[fixed-fill] COMPLETE: $OUT_ROOT/final_test1000/test_main_1000.jsonl"
