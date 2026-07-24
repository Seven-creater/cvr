#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONO_ROOT=""
FIXED_TEST=""
TRAIN_EXCLUDE=""
VAL_EXCLUDE=""
AVE_RUN=""
OUT_ROOT=""
MAX_COMPONENT_USES=2
MAX_CANDIDATES=120
SHARD_COUNT=64
PARALLEL_JOBS=16
RANDOM_SEED=20260723
MODEL="qwen3-omni-30b-a3b-instruct"
ENDPOINTS="http://127.0.0.1:8095/v1,http://127.0.0.1:8096/v1,http://127.0.0.1:8095/v1,http://127.0.0.1:8096/v1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mono-root) MONO_ROOT="$2"; shift 2 ;;
    --fixed-test) FIXED_TEST="$2"; shift 2 ;;
    --train-exclude) TRAIN_EXCLUDE="$2"; shift 2 ;;
    --val-exclude) VAL_EXCLUDE="$2"; shift 2 ;;
    --ave-run) AVE_RUN="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --max-component-uses) MAX_COMPONENT_USES="$2"; shift 2 ;;
    --max-candidates) MAX_CANDIDATES="$2"; shift 2 ;;
    --shard-count) SHARD_COUNT="$2"; shift 2 ;;
    --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --endpoints) ENDPOINTS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for name in MONO_ROOT FIXED_TEST TRAIN_EXCLUDE VAL_EXCLUDE AVE_RUN OUT_ROOT; do
  [[ -n "${!name}" ]] || { echo "Missing required option: $name" >&2; exit 2; }
done
for path in "$FIXED_TEST" "$TRAIN_EXCLUDE" "$VAL_EXCLUDE"; do
  [[ -s "$path" ]] || { echo "Missing non-empty input: $path" >&2; exit 2; }
done

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"/{candidate_pool,review_pool,reviews/pass1,reviews/pass2,logs}
printf '%s\n' "$$" > "$OUT_ROOT/launcher.pid"
ps -o pgid= -p "$$" | tr -d ' ' > "$OUT_ROOT/launcher.pgid"
STATUS="$OUT_ROOT/status.json"
RUN_STATE="FAILED"

write_status() {
  python3 - "$STATUS" "$1" "$2" "$3" <<'PY'
import json, os, sys, tempfile, time
path, state, stage, detail = sys.argv[1:]
payload = {
    "state": state,
    "stage": stage,
    "detail": detail,
    "launcher_pid": os.getppid(),
    "updated_unix": time.time(),
}
fd, temp = tempfile.mkstemp(prefix=".status.", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temp, path)
PY
}

cleanup() {
  code=$?
  trap - EXIT INT TERM
  if [[ "$RUN_STATE" != "COMPLETE" ]]; then
    write_status "FAILED" "launcher" "exit_code=$code; all candidates and reviews are preserved"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

write_status "RUNNING" "prepare_candidates" "building reversible same-layout candidates on CPU"
python3 -m app.audio_cvr_vgg_monoaudio \
  --root "$MONO_ROOT" --output-dir "$OUT_ROOT/candidate_pool" \
  --exclude-jsonl "$FIXED_TEST" --exclude-jsonl "$TRAIN_EXCLUDE" --exclude-jsonl "$VAL_EXCLUDE" \
  --max-component-uses "$MAX_COMPONENT_USES" --max-candidates "$MAX_CANDIDATES" \
  --random-seed "$RANDOM_SEED" \
  > "$OUT_ROOT/logs/prepare_candidates.log" 2>&1

write_status "WAITING" "waiting_ave" "candidate pool is durable; waiting for AVE review to release Omni capacity"
AVE_PID=""
[[ -s "$AVE_RUN/launcher.pid" ]] && AVE_PID="$(tr -dc '0-9' < "$AVE_RUN/launcher.pid")"
while [[ -n "$AVE_PID" ]] && kill -0 "$AVE_PID" 2>/dev/null; do
  sleep 20
done

for port in 8095 8096; do
  curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null \
    || { echo "Omni endpoint $port is not healthy" >&2; exit 1; }
done

write_status "RUNNING" "prepare_review_pool" "deduplicating candidates against fixed test, train, and validation"
python3 -m app.audio_cvr_paper_experiment prepare-fixed-test-fill-review \
  --fixed-test-path "$FIXED_TEST" \
  --input-path "$OUT_ROOT/candidate_pool/review_candidates.jsonl" --input-media-root / \
  --exclude-path "$TRAIN_EXCLUDE" --exclude-path "$VAL_EXCLUDE" \
  --output-dir "$OUT_ROOT/review_pool" --max-per-source 1 --random-seed "$RANDOM_SEED" \
  > "$OUT_ROOT/logs/prepare_review_pool.log" 2>&1

CANDIDATES="$OUT_ROOT/review_pool/automatic_review_candidates.jsonl"
IFS=',' read -r -a URLS <<< "$ENDPOINTS"
[[ ${#URLS[@]} -eq 4 ]] || { echo "Exactly four endpoint entries are required" >&2; exit 2; }

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
    endpoint="${URLS[$((shard % 4))]}"
    python3 -m app.audio_cvr_paper_experiment review-benchmark-omni \
      --candidate-path "$CANDIDATES" \
      --output-path "$output_dir/shard_$(printf '%02d' "$shard").jsonl" \
      --media-root / --cache-dir "$OUT_ROOT/omni_cache" \
      --base-url "$endpoint" --api-key EMPTY --model "$MODEL" \
      --review-pass-id "$pass_id" "${pass1_args[@]}" \
      --repeat-review-fraction 0.20 --random-seed "$RANDOM_SEED" \
      --shard-index "$shard" --shard-count "$SHARD_COUNT" \
      --timeout-seconds 240 --omni-retries 2 \
      --audio-review-max-seconds 3 \
      --video-review-max-dimension 320 --video-review-fps 2 \
      --skip-review-errors --retry-terminal-review-errors --resume \
      > "$OUT_ROOT/logs/pass${pass_id}_shard_$(printf '%02d' "$shard").log" 2>&1 &
    running=$((running + 1))
    if (( running >= PARALLEL_JOBS )); then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

write_status "RUNNING" "omni_review_pass1" "reviewing VGG-MonoAudio candidates"
run_review_pass 1 "$OUT_ROOT/reviews/pass1"
write_status "RUNNING" "omni_review_pass2" "repeating deterministic 20 percent of passes"
run_review_pass 2 "$OUT_ROOT/reviews/pass2"

RUN_STATE="COMPLETE"
write_status "COMPLETE" "review_complete" "VGG-MonoAudio pass1 and pass2 are ready for unified finalization"
