#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRE_RECORDS=""
FINAL_RECORDS=""
CORE_RECORDS=""
OUT_ROOT=""
MODEL_DIR="/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/imagebind"
PYTHON_BIN="/data02/usr/wangqihao/miniconda3/envs/omni_src/bin/python"
GPU_IDS="4,5,6,7"
MEDIA_ROOTS=()
EXPECTED_PRE_SHA256="6f4f17aada3967a72ee0eaf5305c9f3a8fd5dc3ab76f6b867e94f37766977db1"
WAIT_SECONDS=60
MIN_START_FREE_MIB=28000
MIN_RUNTIME_FREE_MIB=14000
BATCH_SIZE=2
RETRIES=4

usage() {
  cat <<'EOF'
Usage: run_audio_cvr_imagebind_incremental.sh \
  --pre-records PATH --final-records PATH --core-records PATH --out-root PATH \
  [--media-root PATH ...] [--gpu-ids 4,5,6,7]

Stage A encodes the existing 516-query pool and then waits for final Test1000.
Stage B reuses every content-addressed cache item and encodes only the delta.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pre-records) PRE_RECORDS="$2"; shift 2 ;;
    --final-records) FINAL_RECORDS="$2"; shift 2 ;;
    --core-records) CORE_RECORDS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --media-root) MEDIA_ROOTS+=("$2"); shift 2 ;;
    --expected-pre-sha256) EXPECTED_PRE_SHA256="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    --min-start-free-mib) MIN_START_FREE_MIB="$2"; shift 2 ;;
    --min-runtime-free-mib) MIN_RUNTIME_FREE_MIB="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --encoding-retries) RETRIES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$PRE_RECORDS" && -n "$FINAL_RECORDS" && -n "$OUT_ROOT" ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$PRE_RECORDS" ]] || { echo "Pre-516 records missing: $PRE_RECORDS" >&2; exit 2; }
[[ -f "$MODEL_DIR/model.safetensors" && -f "$MODEL_DIR/config.json" ]] || { echo "ImageBind model incomplete: $MODEL_DIR" >&2; exit 2; }

if [[ ${#MEDIA_ROOTS[@]} -eq 0 ]]; then
  PRE_PARENT="$(cd "$(dirname "$PRE_RECORDS")" && pwd)"
  FINAL_PARENT="$(cd "$(dirname "$FINAL_RECORDS")" 2>/dev/null && pwd || dirname "$FINAL_RECORDS")"
  MEDIA_ROOTS=(
    "$REPO_ROOT"
    "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval"
    "$PRE_PARENT/construction_data"
    "$FINAL_PARENT/construction_data"
    "$(dirname "$FINAL_PARENT")/construction_data"
  )
  for construction_root in "$REPO_ROOT"/runs/audio_cvr_*test1000*/construction_data; do
    [[ -d "$construction_root" ]] && MEDIA_ROOTS+=("$construction_root")
  done
fi

mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/pids"
STATUS="$OUT_ROOT/status.json"
CACHE_ROOT="$OUT_ROOT/content_cache"
PRE_INVENTORY="$OUT_ROOT/pre516_inventory"
FINAL_INVENTORY="$OUT_ROOT/final1000_inventory"
DELTA_DIR="$OUT_ROOT/final_delta"
ASSEMBLY_DIR="$OUT_ROOT/final_assembly"
EVAL_DIR="$OUT_ROOT/evaluation"
STAT_DIR="$OUT_ROOT/statistics"
OWN_PIDS=()
MONITOR_PIDS=()

write_status() {
  local state="$1" stage="$2" message="$3"
  "$PYTHON_BIN" - "$STATUS" "$state" "$stage" "$message" <<'PY'
import json, os, sys, tempfile, time
path, state, stage, message = sys.argv[1:]
payload = {
    "state": state,
    "stage": stage,
    "message": message,
    "launcher_pid": os.getppid(),
    "updated_unix": time.time(),
}
os.makedirs(os.path.dirname(path), exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=".status.", dir=os.path.dirname(path))
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, path)
PY
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  for pid in "${MONITOR_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for pid in "${OWN_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  if [[ $code -ne 0 ]]; then
    write_status "FAILED" "launcher" "exit_code=$code; all completed cache items are preserved"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "No GPUs configured" >&2; exit 2; }

gpu_free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

preflight_gpus() {
  for gpu in "${GPUS[@]}"; do
    local free
    free="$(gpu_free_mib "$gpu")"
    if (( free < MIN_START_FREE_MIB )); then
      echo "GPU $gpu has only ${free} MiB free; need ${MIN_START_FREE_MIB}" >&2
      exit 3
    fi
  done
  ps -eo pid,pgid,args | grep -E '[f]airseq|[r]unner\.emb\.co\.start' > "$OUT_ROOT/foreign_processes_before.txt" || true
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT_ROOT/gpu_before.csv"
}

monitor_own_worker() {
  local pid="$1" gpu="$2" marker="$3"
  while kill -0 "$pid" 2>/dev/null; do
    local free
    free="$(gpu_free_mib "$gpu" 2>/dev/null || echo 99999)"
    if (( free < MIN_RUNTIME_FREE_MIB )); then
      echo "$(date -Is) GPU $gpu free=${free}MiB below threshold; stopping only ImageBind PID $pid" >> "$marker"
      kill -TERM "$pid" 2>/dev/null || true
      return
    fi
    sleep 30
  done
}

media_root_args=()
for root in "${MEDIA_ROOTS[@]}"; do
  media_root_args+=(--media-root "$root")
done

run_workers() {
  local media_inventory="$1" text_inventory="$2" label="$3"
  local shard_count="${#GPUS[@]}"
  OWN_PIDS=()
  MONITOR_PIDS=()
  for index in "${!GPUS[@]}"; do
    local gpu="${GPUS[$index]}"
    local log="$OUT_ROOT/logs/${label}_both_shard_${index}_of_${shard_count}.log"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$gpu" \
      nice -n 10 "$PYTHON_BIN" -m app.audio_cvr_external_baseline cache-imagebind \
        --inventory-kind both \
        --media-inventory "$media_inventory" \
        --text-inventory "$text_inventory" \
        --cache-root "$CACHE_ROOT" \
        --model-dir "$MODEL_DIR" \
        --shard-index "$index" \
        --shard-count "$shard_count" \
        --device cuda:0 \
        --batch-size "$BATCH_SIZE" \
        --encoding-retries "$RETRIES" \
        > "$log" 2>&1 &
    local pid=$!
    OWN_PIDS+=("$pid")
    echo "$pid" > "$OUT_ROOT/pids/${label}_both_${index}.pid"
    monitor_own_worker "$pid" "$gpu" "$OUT_ROOT/logs/gpu_safety.log" &
    MONITOR_PIDS+=("$!")
  done
  local failed=0
  for pid in "${OWN_PIDS[@]}"; do
    wait "$pid" || failed=1
  done
  for pid in "${MONITOR_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  OWN_PIDS=()
  MONITOR_PIDS=()
  (( failed == 0 )) || return 1
}

run_inventory_cache() {
  local inventory_dir="$1" label="$2" audit_inventory_dir="${3:-$1}"
  local media_name="media_inventory.jsonl" text_name="text_inventory.jsonl"
  if [[ "$label" == "final1000_delta" ]]; then
    media_name="delta_media_inventory.jsonl"
    text_name="delta_text_inventory.jsonl"
  fi
  run_workers "$inventory_dir/$media_name" "$inventory_dir/$text_name" "$label"
  "$PYTHON_BIN" -m app.audio_cvr_external_baseline audit-cache \
    --inventory-dir "$audit_inventory_dir" \
    --cache-root "$CACHE_ROOT" \
    --output "$OUT_ROOT/${label}_cache_audit.json" \
    > "$OUT_ROOT/logs/${label}_cache_audit.log" 2>&1
}

cd "$REPO_ROOT"
preflight_gpus
write_status "RUNNING" "pre516_inventory" "preparing deterministic 516-query inventory"
"$PYTHON_BIN" -m app.audio_cvr_external_baseline prepare-inventory \
  --records "$PRE_RECORDS" \
  --output-dir "$PRE_INVENTORY" \
  "${media_root_args[@]}" \
  --expected-count 516 \
  --expected-sha256 "$EXPECTED_PRE_SHA256" \
  > "$OUT_ROOT/logs/pre516_inventory.log" 2>&1

cp "$PRE_INVENTORY/media_inventory.jsonl" "$OUT_ROOT/pre516_media_inventory.jsonl"
cp "$PRE_INVENTORY/text_inventory.jsonl" "$OUT_ROOT/pre516_text_inventory.jsonl"
cp "$PRE_INVENTORY/inventory_summary.json" "$OUT_ROOT/pre516_inventory_summary.json"

write_status "RUNNING" "pre516_cache" "four GPUs encode disjoint content-addressed shards"
run_inventory_cache "$PRE_INVENTORY" "pre516"
write_status "PREENCODE_COMPLETE" "waiting_final_test1000" "pre516 cache is durable; waiting for frozen Test1000"

while [[ ! -s "$FINAL_RECORDS" ]]; do
  sleep "$WAIT_SECONDS"
done

write_status "RUNNING" "final1000_inventory" "validating Test1000 inheritance and building delta"
final_args=(
  --records "$FINAL_RECORDS"
  --output-dir "$FINAL_INVENTORY"
  --expected-count 1000
  --expected-subtypes sound_event=800,music=200
  --inherited-records "$PRE_RECORDS"
  --require-unique-source-pair
)
final_args+=("${media_root_args[@]}")
"$PYTHON_BIN" -m app.audio_cvr_external_baseline prepare-inventory "${final_args[@]}" \
  > "$OUT_ROOT/logs/final1000_inventory.log" 2>&1
"$PYTHON_BIN" -m app.audio_cvr_external_baseline prepare-delta \
  --pre-inventory-dir "$PRE_INVENTORY" \
  --final-inventory-dir "$FINAL_INVENTORY" \
  --output-dir "$DELTA_DIR" \
  > "$OUT_ROOT/logs/final_delta.log" 2>&1

write_status "RUNNING" "final1000_delta_cache" "reusing pre516 cache and encoding only delta inventories"
run_inventory_cache "$DELTA_DIR" "final1000_delta" "$FINAL_INVENTORY"

write_status "RUNNING" "assemble" "assembling one common valid-query set and 2N target/reference gallery"
assemble_args=(
  --records "$FINAL_RECORDS"
  --inventory-dir "$FINAL_INVENTORY"
  --cache-root "$CACHE_ROOT"
  --output-dir "$ASSEMBLY_DIR"
  --pre-records "$PRE_RECORDS"
  --max-exclusion-rate 0.01
)
if [[ -n "$CORE_RECORDS" && -s "$CORE_RECORDS" ]]; then
  assemble_args+=(--core-records "$CORE_RECORDS")
fi
"$PYTHON_BIN" -m app.audio_cvr_external_baseline assemble "${assemble_args[@]}" \
  > "$OUT_ROOT/logs/assemble.log" 2>&1

write_status "RUNNING" "evaluate" "deriving seven modes and exact per-query source masking"
"$PYTHON_BIN" -m app.audio_cvr_external_baseline evaluate \
  --assembly-dir "$ASSEMBLY_DIR" \
  --output-dir "$EVAL_DIR" \
  --save-topk 20 \
  > "$OUT_ROOT/logs/evaluate.log" 2>&1

write_status "RUNNING" "statistics" "running paired bootstrap, randomization, McNemar, and Holm correction"
"$PYTHON_BIN" -m app.audio_cvr_external_baseline summarize \
  --evaluation-dir "$EVAL_DIR" \
  --output-dir "$STAT_DIR" \
  --iterations 20000 \
  --seed 20260723 \
  > "$OUT_ROOT/logs/statistics.log" 2>&1

ps -eo pid,pgid,args | grep -E '[f]airseq|[r]unner\.emb\.co\.start' > "$OUT_ROOT/foreign_processes_after.txt" || true
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT_ROOT/gpu_after.csv"
write_status "COMPLETE" "complete" "pre516 reuse, final delta, seven modes, reference masking, and statistics complete"
trap - EXIT INT TERM
exit 0
