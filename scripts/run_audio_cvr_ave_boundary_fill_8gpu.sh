#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AVE_ROOT=""
FIXED_TEST=""
PRIOR_RUN=""
TRAIN_EXCLUDE=""
VAL_EXCLUDE=""
OUT_ROOT=""
TARGET_COUNT=1000
MAX_SPEECH_COUNT=50
MAX_AVE_CANDIDATES=1000
CLIP_WORKERS=24
SHARD_COUNT=64
PARALLEL_JOBS=32
RANDOM_SEED=20260723
MODEL="qwen3-omni-30b-a3b-instruct"
ENDPOINTS="http://127.0.0.1:8093/v1,http://127.0.0.1:8094/v1,http://127.0.0.1:8095/v1,http://127.0.0.1:8096/v1"

usage() {
  cat <<'EOF'
Usage: run_audio_cvr_ave_boundary_fill_8gpu.sh \
  --ave-root PATH --fixed-test PATH --prior-run PATH \
  --train-exclude PATH --val-exclude PATH --out-root PATH

Prepares AVE boundary-directed clips immediately on CPU, waits for the prior
fixed-fill review to finish, then reviews only AVE additions on four existing
Omni endpoints. Existing reviews are reused during final Test1000 freezing.
The launcher never starts or stops an Omni service.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ave-root) AVE_ROOT="$2"; shift 2 ;;
    --fixed-test) FIXED_TEST="$2"; shift 2 ;;
    --prior-run) PRIOR_RUN="$2"; shift 2 ;;
    --train-exclude) TRAIN_EXCLUDE="$2"; shift 2 ;;
    --val-exclude) VAL_EXCLUDE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --target-count) TARGET_COUNT="$2"; shift 2 ;;
    --max-speech-count) MAX_SPEECH_COUNT="$2"; shift 2 ;;
    --max-ave-candidates) MAX_AVE_CANDIDATES="$2"; shift 2 ;;
    --clip-workers) CLIP_WORKERS="$2"; shift 2 ;;
    --shard-count) SHARD_COUNT="$2"; shift 2 ;;
    --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --endpoints) ENDPOINTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for name in AVE_ROOT FIXED_TEST PRIOR_RUN TRAIN_EXCLUDE VAL_EXCLUDE OUT_ROOT; do
  [[ -n "${!name}" ]] || { echo "Missing required option: $name" >&2; exit 2; }
done
for path in "$FIXED_TEST" "$TRAIN_EXCLUDE" "$VAL_EXCLUDE"; do
  [[ -s "$path" ]] || { echo "Missing non-empty input: $path" >&2; exit 2; }
done
[[ -s "$AVE_ROOT/Annotations.txt" ]] || { echo "Missing AVE annotations" >&2; exit 2; }
[[ -d "$AVE_ROOT/extracted/videos" ]] || { echo "Missing AVE extracted videos" >&2; exit 2; }

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"/{ave_pool,review_pool,reviews/pass1,reviews/pass2,logs,final_test1000}
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
fd, tmp = tempfile.mkstemp(prefix=".status.", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, path)
PY
}

cleanup() {
  code=$?
  trap - EXIT INT TERM
  if [[ "$RUN_STATE" != "COMPLETE" ]]; then
    write_status "FAILED" "launcher" "exit_code=$code; AVE clips and all review progress are preserved"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

write_status "RUNNING" "prepare_ave_boundary_pool" "cutting boundary-directed AVE clips on CPU"
python3 -m app.audio_cvr_ave prepare \
  --ave-root "$AVE_ROOT" \
  --output-dir "$OUT_ROOT/ave_pool" \
  --exclude-jsonl "$FIXED_TEST" \
  --exclude-jsonl "$TRAIN_EXCLUDE" \
  --exclude-jsonl "$VAL_EXCLUDE" \
  --clip-seconds 6 \
  --max-candidates "$MAX_AVE_CANDIDATES" \
  --workers "$CLIP_WORKERS" \
  --random-seed "$RANDOM_SEED" \
  > "$OUT_ROOT/logs/prepare_ave_boundary_pool.log" 2>&1

AVE_RAW="$OUT_ROOT/ave_pool/ave_boundary_candidates.jsonl"
[[ -s "$AVE_RAW" ]] || { echo "AVE boundary pool is empty" >&2; exit 1; }

write_status "RUNNING" "waiting_prior_review" "AVE clips ready; waiting for prior fixed-fill launcher"
PRIOR_PID=""
[[ -s "$PRIOR_RUN/launcher.pid" ]] && PRIOR_PID="$(tr -dc '0-9' < "$PRIOR_RUN/launcher.pid")"
while [[ -n "$PRIOR_PID" ]] && kill -0 "$PRIOR_PID" 2>/dev/null; do
  sleep 20
done

for port in 8093 8094 8095 8096; do
  curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null \
    || { echo "Omni endpoint $port is not healthy" >&2; exit 1; }
done

write_status "RUNNING" "prepare_ave_review_pool" "deduplicating AVE against immutable test/train/val"
python3 -m app.audio_cvr_paper_experiment prepare-fixed-test-fill-review \
  --fixed-test-path "$FIXED_TEST" \
  --input-path "$AVE_RAW" --input-media-root / \
  --exclude-path "$TRAIN_EXCLUDE" --exclude-path "$VAL_EXCLUDE" \
  --output-dir "$OUT_ROOT/review_pool" \
  --max-per-source 1 --random-seed "$RANDOM_SEED" \
  > "$OUT_ROOT/logs/prepare_ave_review_pool.log" 2>&1

AVE_CANDIDATES="$OUT_ROOT/review_pool/automatic_review_candidates.jsonl"
IFS=',' read -r -a URLS <<< "$ENDPOINTS"
[[ ${#URLS[@]} -eq 4 ]] || { echo "Exactly four endpoints are required" >&2; exit 2; }

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
      --candidate-path "$AVE_CANDIDATES" \
      --output-path "$output_dir/shard_$(printf '%02d' "$shard").jsonl" \
      --media-root / --cache-dir "$OUT_ROOT/omni_cache" \
      --base-url "$endpoint" --api-key EMPTY --model "$MODEL" \
      --review-pass-id "$pass_id" "${pass1_args[@]}" \
      --repeat-review-fraction 0.20 --random-seed "$RANDOM_SEED" \
      --shard-index "$shard" --shard-count "$SHARD_COUNT" \
      --timeout-seconds 240 --omni-retries 2 --resume \
      > "$OUT_ROOT/logs/pass${pass_id}_shard_$(printf '%02d' "$shard").log" 2>&1 &
    running=$((running + 1))
    if (( running >= PARALLEL_JOBS )); then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

write_status "RUNNING" "omni_review_pass1" "reviewing AVE boundary candidates on four endpoints"
run_review_pass 1 "$OUT_ROOT/reviews/pass1"
write_status "RUNNING" "omni_review_pass2" "repeating deterministic 20 percent of AVE passes"
run_review_pass 2 "$OUT_ROOT/reviews/pass2"

PRIOR_CANDIDATES="$PRIOR_RUN/review_pool/automatic_review_candidates.jsonl"
[[ -s "$PRIOR_CANDIDATES" ]] || { echo "Prior candidate pool is missing" >&2; exit 1; }
COMBINED="$OUT_ROOT/combined_candidates.jsonl"
python3 - "$PRIOR_CANDIDATES" "$AVE_CANDIDATES" "$COMBINED" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
inputs = [Path(value) for value in sys.argv[1:-1]]
output = Path(sys.argv[-1])
rows = {}
for path in inputs:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("sample_id") or row.get("proposal_id") or "").strip()
        if key:
            rows.setdefault(key, row)
fd, tmp = tempfile.mkstemp(prefix=".combined.", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    for key in sorted(rows):
        handle.write(json.dumps(rows[key], ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, output)
PY

PASS1_ARGS=()
PASS2_ARGS=()
for path in "$PRIOR_RUN"/reviews/pass1/shard_*.jsonl "$OUT_ROOT"/reviews/pass1/shard_*.jsonl; do
  [[ -f "$path" ]] && PASS1_ARGS+=(--pass1-review-path "$path")
done
for path in "$PRIOR_RUN"/reviews/pass2/shard_*.jsonl "$OUT_ROOT"/reviews/pass2/shard_*.jsonl; do
  [[ -f "$path" ]] && PASS2_ARGS+=(--pass2-review-path "$path")
done
[[ ${#PASS1_ARGS[@]} -gt 0 && ${#PASS2_ARGS[@]} -gt 0 ]] \
  || { echo "Combined review outputs are incomplete" >&2; exit 1; }

write_status "RUNNING" "finalizing_test1000" "combining prior and AVE Omni-consensus candidates"
python3 -m app.audio_cvr_paper_experiment finalize-fixed-test-fill \
  --fixed-test-path "$FIXED_TEST" --candidate-path "$COMBINED" \
  "${PASS1_ARGS[@]}" "${PASS2_ARGS[@]}" \
  --exclude-path "$TRAIN_EXCLUDE" --exclude-path "$VAL_EXCLUDE" \
  --output-dir "$OUT_ROOT/final_test1000" \
  --target-count "$TARGET_COUNT" --max-speech-count "$MAX_SPEECH_COUNT" \
  --sound-event-ratio 0.80 --repeat-review-fraction 0.20 \
  --random-seed "$RANDOM_SEED" \
  > "$OUT_ROOT/logs/finalize_test1000.log" 2>&1

RUN_STATE="COMPLETE"
write_status "COMPLETE" "test1000_frozen" "Prior plus AVE Test1000 frozen; Omni services remain running"
echo "[ave-boundary-fill] COMPLETE: $OUT_ROOT/final_test1000/test_main_1000.jsonl"
