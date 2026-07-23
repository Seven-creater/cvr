#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINAL_RECORDS=""
PRE_RECORDS=""
TRAIN_RECORDS=""
VAL_RECORDS=""
CORE_RECORDS=""
DATASET_RUN_ROOT=""
CONSTRUCTION_STATUS=""
E5_ROOT=""
IMAGEBIND_ROOT=""
ADAPTER_ROOT=""
OUT_ROOT=""
E5_MODEL="/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/e5_omni_7b"
IMAGEBIND_MODEL="/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/imagebind"
PYTHON_BIN="python3"
GPU_IDS="0,1,2,3,4,5,6,7"
FINAL_SEEDS="13,23,42,71,101"
EXPECTED_PRE_SHA256="6f4f17aada3967a72ee0eaf5305c9f3a8fd5dc3ab76f6b867e94f37766977db1"
EXPECTED_FINAL_SHA256=""
EXPECTED_FINAL_SUBTYPES=""
EXPECTED_PRE_MEDIA=1044
EXPECTED_PRE_TEXT=469
WAIT_SECONDS=60
E5_LOAD_WAIT_SECONDS=90
MIN_E5_START_FREE_MIB=38000
MIN_IMAGEBIND_START_FREE_MIB=12000
MIN_IMAGEBIND_RUNTIME_FREE_MIB=6000
E5_BATCH_SIZE=2
IMAGEBIND_BATCH_SIZE=2
ENCODING_RETRIES=4
MEDIA_ROOTS=()
OMNI_PID_FILES=()
OMNI_PORTS=()

usage() {
  cat <<'EOF'
Usage: run_audio_cvr_e5_imagebind_final_8gpu.sh \
  --final-records PATH --pre-records PATH --train-records PATH --val-records PATH \
  --dataset-run-root PATH --construction-status PATH \
  --e5-root PATH --imagebind-root PATH --adapter-root PATH --out-root PATH \
  [--core-records PATH] [--media-root PATH ...] \
  [--omni-pid-file PATH ...] [--omni-port PORT ...]

By default, final subtype proportions are recorded but not enforced. Pass
--expected-final-subtypes sound_event=N,music=N only for an explicit audit.

The script waits for frozen Test1000, audits it, precisely stops recorded Omni
services, reuses the existing pre-516 E5/ImageBind caches, and runs seven E5
modes plus eight ImageBind delta shards concurrently on GPUs 0-7.

Launch with:
  setsid nohup bash scripts/run_audio_cvr_e5_imagebind_final_8gpu.sh ... \
    > logs/e5_imagebind_final_8gpu.log 2>&1 < /dev/null &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --final-records) FINAL_RECORDS="$2"; shift 2 ;;
    --pre-records) PRE_RECORDS="$2"; shift 2 ;;
    --train-records) TRAIN_RECORDS="$2"; shift 2 ;;
    --val-records) VAL_RECORDS="$2"; shift 2 ;;
    --core-records) CORE_RECORDS="$2"; shift 2 ;;
    --dataset-run-root) DATASET_RUN_ROOT="$2"; shift 2 ;;
    --construction-status) CONSTRUCTION_STATUS="$2"; shift 2 ;;
    --e5-root) E5_ROOT="$2"; shift 2 ;;
    --imagebind-root) IMAGEBIND_ROOT="$2"; shift 2 ;;
    --adapter-root) ADAPTER_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --imagebind-model) IMAGEBIND_MODEL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --final-seeds) FINAL_SEEDS="$2"; shift 2 ;;
    --expected-pre-sha256) EXPECTED_PRE_SHA256="$2"; shift 2 ;;
    --expected-final-sha256) EXPECTED_FINAL_SHA256="$2"; shift 2 ;;
    --expected-final-subtypes) EXPECTED_FINAL_SUBTYPES="$2"; shift 2 ;;
    --expected-pre-media) EXPECTED_PRE_MEDIA="$2"; shift 2 ;;
    --expected-pre-text) EXPECTED_PRE_TEXT="$2"; shift 2 ;;
    --media-root) MEDIA_ROOTS+=("$2"); shift 2 ;;
    --omni-pid-file) OMNI_PID_FILES+=("$2"); shift 2 ;;
    --omni-port) OMNI_PORTS+=("$2"); shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    --e5-load-wait-seconds) E5_LOAD_WAIT_SECONDS="$2"; shift 2 ;;
    --e5-batch-size) E5_BATCH_SIZE="$2"; shift 2 ;;
    --imagebind-batch-size) IMAGEBIND_BATCH_SIZE="$2"; shift 2 ;;
    --encoding-retries) ENCODING_RETRIES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in FINAL_RECORDS PRE_RECORDS TRAIN_RECORDS VAL_RECORDS DATASET_RUN_ROOT CONSTRUCTION_STATUS E5_ROOT IMAGEBIND_ROOT ADAPTER_ROOT OUT_ROOT; do
  [[ -n "${!value}" ]] || { echo "Missing required option for $value" >&2; usage >&2; exit 2; }
done
[[ -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" || -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -s "$PRE_RECORDS" && -s "$TRAIN_RECORDS" && -s "$VAL_RECORDS" ]] || { echo "Pre/train/val records are missing" >&2; exit 2; }
[[ -f "$IMAGEBIND_MODEL/model.safetensors" && -f "$IMAGEBIND_MODEL/config.json" ]] || { echo "ImageBind model is incomplete" >&2; exit 2; }
[[ ${#OMNI_PID_FILES[@]} -eq ${#OMNI_PORTS[@]} ]] || { echo "Omni PID file and port counts differ" >&2; exit 2; }
[[ ${#OMNI_PID_FILES[@]} -eq 4 ]] || { echo "Exactly four recorded Omni services are required for a safe handoff" >&2; exit 2; }

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ ${#GPUS[@]} -eq 8 ]] || { echo "Exactly eight GPU IDs are required" >&2; exit 2; }
IFS=',' read -r -a SEEDS <<< "$FINAL_SEEDS"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/pids" "$OUT_ROOT/workers"
STATUS="$OUT_ROOT/status.json"
E5_RECORDS="$OUT_ROOT/e5_records_final1000"
E5_FAILURE_ROOT="$OUT_ROOT/e5_encoding_failures"
E5_EVAL_ROOT="$OUT_ROOT/e5_evaluation"
E5_STAT_ROOT="$OUT_ROOT/e5_statistics"
FINAL_INVENTORY="$IMAGEBIND_ROOT/final1000_inventory"
PRE_INVENTORY="$IMAGEBIND_ROOT/pre516_inventory"
DELTA_DIR="$IMAGEBIND_ROOT/final_delta"
IMAGEBIND_CACHE="$IMAGEBIND_ROOT/content_cache"
IMAGEBIND_ASSEMBLY="$IMAGEBIND_ROOT/final_assembly"
IMAGEBIND_EVAL="$IMAGEBIND_ROOT/evaluation"
IMAGEBIND_STATS="$IMAGEBIND_ROOT/statistics"
CHILD_PIDS=()
MONITOR_PIDS=()
RUN_STATE="FAILED"

write_status() {
  local state="$1" stage="$2" message="$3"
  "$PYTHON_BIN" - "$STATUS" "$state" "$stage" "$message" <<'PY'
import json, os, sys, tempfile, time
path, state, stage, message = sys.argv[1:]
payload = {"state": state, "stage": stage, "message": message, "launcher_pid": os.getppid(), "updated_unix": time.time()}
os.makedirs(os.path.dirname(path), exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=".status.", dir=os.path.dirname(path))
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp, path)
PY
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  for pid in "${MONITOR_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${CHILD_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
  if [[ "$RUN_STATE" != "COMPLETE" ]]; then
    write_status "FAILED" "launcher" "exit_code=$code; all item checkpoints and content caches are preserved"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

gpu_free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

wait_for_frozen_test() {
  write_status "WAITING_TEST1000" "waiting_test1000" "waiting for frozen Test1000 and completed construction status"
  while true; do
    if [[ -s "$FINAL_RECORDS" && -s "$CONSTRUCTION_STATUS" ]]; then
      state="$($PYTHON_BIN - "$CONSTRUCTION_STATUS" <<'PY'
import json, pathlib, sys
try: print(str(json.loads(pathlib.Path(sys.argv[1]).read_text())["state"]).upper())
except Exception: print("")
PY
)"
      [[ "$state" == "COMPLETE" ]] && return
      [[ "$state" == "FAILED" ]] && { echo "Construction status is FAILED" >&2; exit 3; }
    fi
    sleep "$WAIT_SECONDS"
  done
}

stop_recorded_omni() {
  local shell_pgid
  shell_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  for index in "${!OMNI_PID_FILES[@]}"; do
    local pid_file="${OMNI_PID_FILES[$index]}" port="${OMNI_PORTS[$index]}" pid pgid cmd
    [[ -s "$pid_file" ]] || { echo "Omni PID file missing: $pid_file" >&2; exit 4; }
    pid="$(tr -dc '0-9' < "$pid_file")"
    [[ -n "$pid" ]] || { echo "Invalid Omni PID file: $pid_file" >&2; exit 4; }
    if kill -0 "$pid" 2>/dev/null; then
      pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
      cmd="$(ps -o args= -p "$pid")"
      [[ -n "$pgid" && "$pgid" != "$shell_pgid" ]] || { echo "Unsafe Omni PGID: $pgid" >&2; exit 4; }
      [[ "$cmd" =~ vllm|api_server|omni ]] || { echo "PID $pid is not an Omni/vLLM service: $cmd" >&2; exit 4; }
      echo "$pid $pgid $port $cmd" >> "$OUT_ROOT/stopped_omni_services.tsv"
      kill -TERM -- "-$pgid"
      for _ in {1..60}; do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
      kill -0 "$pid" 2>/dev/null && { echo "Omni PID $pid did not stop" >&2; exit 4; }
    fi
    for _ in {1..30}; do
      if ! curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then break; fi
      sleep 2
    done
    curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { echo "Omni port $port remains open" >&2; exit 4; }
  done
  return 0
}

media_root_args=()
for root in "${MEDIA_ROOTS[@]}"; do media_root_args+=(--media-root "$root"); done
if [[ ${#MEDIA_ROOTS[@]} -eq 0 ]]; then
  media_root_args=(--media-root "$REPO_ROOT" --media-root "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval")
fi

prepare_and_audit() {
  write_status "PREPARING_DELTA" "preparing_delta" "auditing Test1000, split leakage, inventories, and reusable caches"
  [[ -s "$PRE_INVENTORY/media_inventory.jsonl" && -s "$PRE_INVENTORY/text_inventory.jsonl" ]] || {
    echo "Existing ImageBind pre516 inventory is missing: $PRE_INVENTORY" >&2
    exit 3
  }
  for index in {0..6}; do
    mode="${MODE_NAMES[$index]}"
    cache="$E5_ROOT/cache_$mode"
    [[ -d "$cache/item_embedding_cache" ]] || {
      echo "Existing E5 pre516 checkpoint cache is missing for mode=$mode: $cache" >&2
      exit 3
    }
    "$PYTHON_BIN" - "$cache" "$E5_MODEL" "${QUERY_MODES[$index]}" "${DOCUMENT_MODES[$index]}" "${VIDEO_AUDIO_MODES[$index]}" <<'PY'
import json, pathlib, sys
cache, model = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
expected_query, expected_document, expected_audio = sys.argv[3:]
summaries = []
for path in cache.glob("checkpoint_prefill_shard_*.json"):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        continue
    runtime = payload.get("runtime") or {}
    raw_model = pathlib.Path(str(runtime.get("model_path") or ""))
    same_model = raw_model.resolve() == model.resolve() if raw_model.exists() and model.exists() else str(raw_model) == str(model)
    if (same_model and runtime.get("query_input_mode") == expected_query
            and runtime.get("document_input_mode") == expected_document
            and runtime.get("video_audio_mode") == expected_audio):
        summaries.append(payload)
if not summaries:
    raise SystemExit(
        f"no reusable pre516 checkpoint summary for cache={cache} model={model} "
        f"query={expected_query} document={expected_document} audio={expected_audio}"
    )
root = pathlib.Path(str(summaries[-1]["checkpoint_root"]))
if not root.is_dir() or not next(root.rglob("*.npy"), None):
    raise SystemExit(f"reusable checkpoint root is empty: {root}")
PY
  done
  "$PYTHON_BIN" -m app.audio_cvr_paper_experiment audit-training-splits \
    --train-path "$TRAIN_RECORDS" --val-path "$VAL_RECORDS" --test-path "$FINAL_RECORDS" \
    --output-dir "$OUT_ROOT/split_audit" > "$OUT_ROOT/logs/split_audit.log" 2>&1

  final_args=(--records "$FINAL_RECORDS" --output-dir "$FINAL_INVENTORY" "${media_root_args[@]}" \
    --expected-count 1000 --inherited-records "$PRE_RECORDS" --require-unique-source-pair)
  [[ -n "$EXPECTED_FINAL_SUBTYPES" && "$EXPECTED_FINAL_SUBTYPES" != "none" ]] \
    && final_args+=(--expected-subtypes "$EXPECTED_FINAL_SUBTYPES")
  [[ -n "$EXPECTED_FINAL_SHA256" ]] && final_args+=(--expected-sha256 "$EXPECTED_FINAL_SHA256")
  "$PYTHON_BIN" -m app.audio_cvr_external_baseline prepare-inventory "${final_args[@]}" \
    > "$OUT_ROOT/logs/final_inventory.log" 2>&1
  "$PYTHON_BIN" -m app.audio_cvr_external_baseline prepare-delta \
    --pre-inventory-dir "$PRE_INVENTORY" --final-inventory-dir "$FINAL_INVENTORY" --output-dir "$DELTA_DIR" \
    > "$OUT_ROOT/logs/final_delta.log" 2>&1
  "$PYTHON_BIN" -m app.audio_cvr_external_baseline audit-cache \
    --inventory-dir "$PRE_INVENTORY" --cache-root "$IMAGEBIND_CACHE" --output "$OUT_ROOT/pre516_cache_audit.json" \
    > "$OUT_ROOT/logs/pre516_cache_audit.log" 2>&1

  "$PYTHON_BIN" - "$OUT_ROOT/pre516_cache_audit.json" "$EXPECTED_PRE_MEDIA" "$EXPECTED_PRE_TEXT" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected_media, expected_text = map(int, sys.argv[2:])
if not p.get("complete") or p["media"]["complete_count"] != expected_media or p["text"]["complete_count"] != expected_text:
    raise SystemExit(f"pre516 ImageBind cache mismatch: {p}")
PY

  "$PYTHON_BIN" -m app.e5_audio_delta_train prepare \
    --dataset-run-root "$DATASET_RUN_ROOT" --output-dir "$E5_RECORDS" \
    --train-path "$TRAIN_RECORDS" --eval-path "$FINAL_RECORDS" \
    "${media_root_args[@]}" --require-existing-media \
    --max-train-records 0 --max-eval-records 0 \
    --eval-gallery-size 2000 --eval-gallery-protocol reference --distractor-seed 20260723 \
    > "$OUT_ROOT/logs/e5_prepare.log" 2>&1
}

# The audit writer is kept separate so shell quoting remains readable.
write_final_audit() {
  "$PYTHON_BIN" - "$FINAL_RECORDS" "$PRE_RECORDS" "$FINAL_INVENTORY/inventory_summary.json" "$E5_RECORDS/summary.json" "$OUT_ROOT/final_test1000_audit.json" "$EXPECTED_PRE_SHA256" <<'PY'
import hashlib, json, pathlib, sys
final, pre, inv_path, e5_path, output = map(pathlib.Path, sys.argv[1:6])
expected_pre = sys.argv[6].strip().lower()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if expected_pre and sha(pre) != expected_pre:
    raise SystemExit(f"pre516 SHA mismatch: {sha(pre)}")
inventory = json.loads(inv_path.read_text())
e5 = json.loads(e5_path.read_text())
gallery = e5.get("eval_gallery") or {}
if inventory.get("record_count") != 1000:
    raise SystemExit(f"final inventory count mismatch: {inventory}")
if gallery.get("gallery_count") != 2000 or gallery.get("positive_count") != 1000 or gallery.get("reference_negative_count") != 1000:
    raise SystemExit(f"E5 reference gallery mismatch: {gallery}")
payload = {
    "test_sha256": sha(final), "pre516_sha256": sha(pre),
    "test_count": inventory["record_count"], "media_count": inventory["media_count"],
    "text_count": inventory["text_count"], "e5_gallery": e5.get("eval_gallery"),
    "pre516_inherited": True, "selection_uses_test_metrics": False,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

progress_monitor() {
  while true; do
    {
      printf '%s\t' "$(date -Is)"
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | tr '\n' ';'
      printf '\te5_npy=%s\tib_media=%s\tib_text=%s\n' \
        "$(find "$E5_ROOT" -path '*/item_embedding_cache/*' -name '*.npy' 2>/dev/null | wc -l)" \
        "$(find "$IMAGEBIND_CACHE/indexes/media" -name '*.json' 2>/dev/null | wc -l)" \
        "$(find "$IMAGEBIND_CACHE/indexes/text" -name '*.json' 2>/dev/null | wc -l)"
    } >> "$OUT_ROOT/progress_30s.tsv"
    sleep 30
  done
}

MODE_NAMES=(V_A_T V_T V_A T_only_fullAV V_only A_T A_only)
QUERY_MODES=(composed composed video_only text_only video_only audio_text audio_only)
DOCUMENT_MODES=(video video video video video audio audio)
VIDEO_AUDIO_MODES=(on off on on off off off)
E5_PIDS=()
E5_FAILED=()
IB_PIDS=()
IB_FAILED=()
IB_DEFERRED=()

e5_cache_dir() { echo "$E5_ROOT/cache_$1"; }

start_e5_prefill() {
  local index="$1" batch="$2" mode="${MODE_NAMES[$index]}" gpu="${GPUS[$index]}" cache log pid
  cache="$(e5_cache_dir "$mode")"
  log="$OUT_ROOT/logs/e5_prefill_${mode}_batch${batch}.log"
  mkdir -p "$cache" "$E5_FAILURE_ROOT/$mode"
  (
    export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
    exec "$PYTHON_BIN" -m app.e5_audio_delta_train cache-embeddings \
      --records-dir "$E5_RECORDS" --output-dir "$cache" --e5-model "$E5_MODEL" \
      --device cuda --torch-dtype bfloat16 --batch-size "$batch" --video-fps 1 \
      --video-audio-mode "${VIDEO_AUDIO_MODES[$index]}" \
      --query-input-mode "${QUERY_MODES[$index]}" --document-input-mode "${DOCUMENT_MODES[$index]}" \
      --audio-media-cache-dir "$cache/audio_media_cache" --local-segments 0 --skip-train \
      --checkpoint-embeddings --checkpoint-prefill-only --checkpoint-shard-index 0 --checkpoint-shard-count 1 \
      --encoding-item-batch-size "$batch" --encoding-retries "$ENCODING_RETRIES" \
      --skip-persistent-encoding-failures --encoding-failure-dir "$E5_FAILURE_ROOT/$mode"
  ) > "$log" 2>&1 &
  pid=$!; CHILD_PIDS+=("$pid"); E5_PIDS[$index]="$pid"; echo "$pid" > "$OUT_ROOT/pids/e5_${mode}.pid"
}

monitor_imagebind_worker() {
  local pid="$1" gpu="$2" shard="$3"
  while kill -0 "$pid" 2>/dev/null; do
    free="$(gpu_free_mib "$gpu" 2>/dev/null || echo 99999)"
    if (( free < MIN_IMAGEBIND_RUNTIME_FREE_MIB )); then
      echo "$(date -Is) stopping ImageBind shard=$shard pid=$pid gpu=$gpu free_mib=$free" >> "$OUT_ROOT/logs/gpu_safety.log"
      kill -TERM "$pid" 2>/dev/null || true
      return
    fi
    sleep 30
  done
}

start_imagebind_shard() {
  local shard="$1" batch="$2" gpu="${GPUS[$shard]}" log pid
  log="$OUT_ROOT/logs/imagebind_delta_shard_${shard}_of_8_batch${batch}.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
    exec nice -n 10 "$PYTHON_BIN" -m app.audio_cvr_external_baseline cache-imagebind \
      --inventory-kind both \
      --media-inventory "$DELTA_DIR/delta_media_inventory.jsonl" \
      --text-inventory "$DELTA_DIR/delta_text_inventory.jsonl" \
      --cache-root "$IMAGEBIND_CACHE" --model-dir "$IMAGEBIND_MODEL" \
      --shard-index "$shard" --shard-count 8 --device cuda:0 \
      --batch-size "$batch" --encoding-retries "$ENCODING_RETRIES"
  ) > "$log" 2>&1 &
  pid=$!; CHILD_PIDS+=("$pid"); IB_PIDS[$shard]="$pid"; echo "$pid" > "$OUT_ROOT/pids/imagebind_shard_${shard}.pid"
  monitor_imagebind_worker "$pid" "$gpu" "$shard" & MONITOR_PIDS+=("$!")
}

wait_for_e5_load_or_exit() {
  local waited=0 ready active free
  while (( waited < E5_LOAD_WAIT_SECONDS )); do
    ready=1
    for index in {0..6}; do
      active=0
      kill -0 "${E5_PIDS[$index]}" 2>/dev/null && active=1
      if (( active == 1 )); then
        free="$(gpu_free_mib "${GPUS[$index]}")"
        (( free < MIN_E5_START_FREE_MIB )) || ready=0
      fi
    done
    (( ready == 1 )) && return
    sleep 5
    waited=$((waited + 5))
  done
  echo "E5 load wait reached ${E5_LOAD_WAIT_SECONDS}s; applying per-GPU ImageBind free-memory gates" >> "$OUT_ROOT/logs/gpu_safety.log"
}

run_concurrent_encoding() {
  write_status "LOADING_E5" "loading_e5" "starting one frozen E5 mode on GPUs 0-6"
  for index in {0..6}; do start_e5_prefill "$index" "$E5_BATCH_SIZE"; done
  wait_for_e5_load_or_exit

  write_status "ENCODING_CONCURRENT" "encoding_concurrent" "seven E5 modes and safe ImageBind delta shards run concurrently"
  for shard in {0..7}; do
    free="$(gpu_free_mib "${GPUS[$shard]}")"
    if (( free >= MIN_IMAGEBIND_START_FREE_MIB )); then start_imagebind_shard "$shard" "$IMAGEBIND_BATCH_SIZE"; else IB_DEFERRED+=("$shard"); fi
  done

  for index in {0..6}; do wait "${E5_PIDS[$index]}" || E5_FAILED+=("$index"); done
  for shard in "${!IB_PIDS[@]}"; do wait "${IB_PIDS[$shard]}" || IB_FAILED+=("$shard"); done
  for pid in "${MONITOR_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  MONITOR_PIDS=()

  if [[ ${#E5_FAILED[@]} -gt 0 ]]; then
    E5_PIDS=()
    for index in "${E5_FAILED[@]}"; do start_e5_prefill "$index" 1; done
    for index in "${E5_FAILED[@]}"; do wait "${E5_PIDS[$index]}"; done
  fi
  retry_shards=("${IB_DEFERRED[@]:-}" "${IB_FAILED[@]:-}")
  if [[ ${#retry_shards[@]} -gt 0 ]]; then
    IB_PIDS=()
    for shard in "${retry_shards[@]}"; do [[ -n "$shard" ]] && start_imagebind_shard "$shard" 1; done
    for shard in "${!IB_PIDS[@]}"; do wait "${IB_PIDS[$shard]}"; done
    for pid in "${MONITOR_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    MONITOR_PIDS=()
  fi
}

assemble_all() {
  write_status "ASSEMBLING" "assembling" "assembling seven E5 caches without model reload and one ImageBind base cache"
  assembly_pids=()
  for index in {0..6}; do
    mode="${MODE_NAMES[$index]}"; cache="$(e5_cache_dir "$mode")"; log="$OUT_ROOT/logs/e5_assemble_${mode}.log"
    "$PYTHON_BIN" -m app.e5_audio_delta_train cache-embeddings \
      --records-dir "$E5_RECORDS" --output-dir "$cache" --skip-train --local-segments 0 \
      --checkpoint-embeddings --assemble-from-checkpoints-only \
      --skip-persistent-encoding-failures --encoding-failure-dir "$E5_FAILURE_ROOT/$mode" \
      > "$log" 2>&1 &
    assembly_pids+=("$!"); CHILD_PIDS+=("$!")
  done
  (
    assemble_args=("$PYTHON_BIN" -m app.audio_cvr_external_baseline assemble \
      --records "$FINAL_RECORDS" --inventory-dir "$FINAL_INVENTORY" --cache-root "$IMAGEBIND_CACHE" \
      --output-dir "$IMAGEBIND_ASSEMBLY" --pre-records "$PRE_RECORDS" --max-exclusion-rate 0.01)
    [[ -n "$CORE_RECORDS" && -s "$CORE_RECORDS" ]] && assemble_args+=(--core-records "$CORE_RECORDS")
    "$PYTHON_BIN" -m app.audio_cvr_external_baseline audit-cache \
      --inventory-dir "$FINAL_INVENTORY" --cache-root "$IMAGEBIND_CACHE" --output "$IMAGEBIND_ROOT/final1000_cache_audit.json"
    "${assemble_args[@]}"
    "$PYTHON_BIN" -m app.audio_cvr_external_baseline evaluate \
      --assembly-dir "$IMAGEBIND_ASSEMBLY" --output-dir "$IMAGEBIND_EVAL" --save-topk 20
    "$PYTHON_BIN" -m app.audio_cvr_external_baseline summarize \
      --evaluation-dir "$IMAGEBIND_EVAL" --output-dir "$IMAGEBIND_STATS" --iterations 20000 --seed 20260723
  ) > "$OUT_ROOT/logs/imagebind_assemble_evaluate.log" 2>&1 &
  assembly_pids+=("$!"); CHILD_PIDS+=("$!")
  for pid in "${assembly_pids[@]}"; do wait "$pid"; done
}

run_e5_evaluation() {
  write_status "EVALUATING" "e5_evaluation" "evaluating five frozen adapters, seven modes, and exact source masking"
  active=()
  active_logs=()
  eval_output_matches_cache() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import pathlib
import sys

cache_dir, eval_dir = map(pathlib.Path, sys.argv[1:])
cache_records = cache_dir / "eval_records.jsonl"
summary_path = eval_dir / "summary.json"
score_path = eval_dir / "per_query_scores.jsonl"
if not all(path.is_file() and path.stat().st_size > 0 for path in (cache_records, summary_path, score_path)):
    raise SystemExit(1)

load = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
cache_rows = load(cache_records)
score_rows = load(score_path)
summary = json.loads(summary_path.read_text())
sample_id = lambda row: str(row.get("sample_id") or row.get("proposal_id") or "")
cache_ids = [sample_id(row) for row in cache_rows]
score_ids = [sample_id(row) for row in score_rows]
valid = (
    len(cache_rows) == len(score_rows)
    and int(summary.get("eval_count", -1)) == len(cache_rows)
    and len(set(cache_ids)) == len(cache_ids)
    and len(set(score_ids)) == len(score_ids)
    and set(cache_ids) == set(score_ids)
    and all(cache_ids)
)
raise SystemExit(0 if valid else 1)
PY
  }
  for seed in "${SEEDS[@]}"; do
    adapter="$ADAPTER_ROOT/seed_${seed}/adapter"
    [[ -s "$adapter/adapter.pt" && -s "$adapter/adapter_config.json" ]] || { echo "Missing frozen adapter: $adapter" >&2; exit 6; }
    for mode in T_only_fullAV V_only A_only V_T A_T V_A V_A_T; do
      suffixes=("")
      [[ "$mode" == "V_T" || "$mode" == "V_A_T" ]] && suffixes+=("_no_ref")
      for suffix in "${suffixes[@]}"; do
        eval_dir="$E5_EVAL_ROOT/seed_${seed}/eval_${mode}${suffix}"
        cache_dir="$(e5_cache_dir "$mode")"
        if eval_output_matches_cache "$cache_dir" "$eval_dir"; then
          continue
        fi
        if [[ -d "$eval_dir" ]]; then
          stale_dir="${eval_dir}.stale_$(date +%Y%m%d_%H%M%S)"
          mv "$eval_dir" "$stale_dir"
          echo "Archived stale E5 evaluation: $eval_dir -> $stale_dir"
        fi
        gpu="${GPUS[$(( ${#active[@]} % 8 ))]}"; log="$OUT_ROOT/logs/eval_seed${seed}_${mode}${suffix}.log"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          args=("$PYTHON_BIN" -m app.e5_audio_delta_train eval --cache-dir "$cache_dir" \
            --adapter-dir "$adapter" --output-dir "$eval_dir" --device cuda --topk 1,5,10 --save-topk 20)
          [[ "$suffix" == "_no_ref" ]] && args+=(--exclude-query-reference)
          exec "${args[@]}"
        ) > "$log" 2>&1 &
        active+=("$!"); active_logs+=("$log"); CHILD_PIDS+=("$!")
        if [[ ${#active[@]} -ge 8 ]]; then
          for i in "${!active[@]}"; do wait "${active[$i]}" || { tail -100 "${active_logs[$i]}" >&2; exit 7; }; done
          active=(); active_logs=()
        fi
      done
    done
  done
  for i in "${!active[@]}"; do wait "${active[$i]}" || { tail -100 "${active_logs[$i]}" >&2; exit 7; }; done

  "$PYTHON_BIN" -m app.audio_cvr_paper_experiment aggregate-final \
    --input-root "$E5_EVAL_ROOT" --output-dir "$E5_STAT_ROOT" --required-seeds "$FINAL_SEEDS" \
    --primary-mode V_A_T --reference-mode V_T \
    --comparison V_A_T:V_T --comparison V_A_T_no_ref:V_A_T --comparison V_T_no_ref:V_T \
    --comparison V_A_T:V_A --comparison A_T:A_only --bootstrap-samples 20000 --permutation-samples 20000 \
    > "$OUT_ROOT/logs/e5_statistics.log" 2>&1
}

final_audit() {
  "$PYTHON_BIN" - "$FINAL_RECORDS" "$E5_ROOT" "$E5_EVAL_ROOT" "$IMAGEBIND_ASSEMBLY/records.jsonl" "$OUT_ROOT/common_query_audit.json" <<'PY'
import json, pathlib, sys
final_path, e5_root, e5_eval_root, imagebind_records, output = map(pathlib.Path, sys.argv[1:])
load = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
sid = lambda row: str(row.get("sample_id") or row.get("proposal_id") or "")
final_ids = {sid(row) for row in load(final_path)}
sets = {"imagebind": {sid(row) for row in load(imagebind_records)}}
for mode in ("T_only_fullAV", "V_only", "A_only", "V_T", "A_T", "V_A", "V_A_T"):
    sets[f"e5_{mode}"] = {sid(row) for row in load(e5_root / f"cache_{mode}" / "eval_records.jsonl")}
for seed in (13, 23, 42, 71, 101):
    for mode in ("T_only_fullAV", "V_only", "A_only", "V_T", "A_T", "V_A", "V_A_T"):
        suffixes = ("", "_no_ref") if mode in {"V_T", "V_A_T"} else ("",)
        for suffix in suffixes:
            name = f"{mode}{suffix}"
            path = e5_eval_root / f"seed_{seed}" / f"eval_{name}" / "per_query_scores.jsonl"
            sets[f"e5_eval_seed{seed}_{name}"] = {sid(row) for row in load(path)}
common = set.intersection(*sets.values())
payload = {
    "final_query_count": len(final_ids), "common_query_count": len(common),
    "common_exclusion_rate": 1.0 - len(common) / max(1, len(final_ids)),
    "sample_sets_identical": all(values == common for values in sets.values()),
    "per_model_counts": {key: len(value) for key, value in sets.items()},
    "common_sample_ids": sorted(common),
    "selection_uses_test_metrics": False,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if payload["common_exclusion_rate"] > 0.01:
    raise SystemExit(f"common-query exclusion rate exceeds 1%: {payload}")
if not payload["sample_sets_identical"]:
    raise SystemExit("model/mode query sets differ; caches are preserved but paper statistics require aligned query sets")
PY
  "$PYTHON_BIN" - "$E5_ROOT" "$OUT_ROOT/e5_delta_payload_summary.json" <<'PY'
import json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
rows = {}
for mode in ("T_only_fullAV", "V_only", "A_only", "V_T", "A_T", "V_A", "V_A_T"):
    path = root / f"cache_{mode}" / "checkpoint_prefill_shard_000_of_001.json"
    rows[mode] = json.loads(path.read_text())
output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
PY
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT_ROOT/gpu_after.csv"
}

cd "$REPO_ROOT"
wait_for_frozen_test
prepare_and_audit
write_final_audit
write_status "PREPARING_DELTA" "omni_cleanup" "stopping only recorded Omni PID/PGID services"
stop_recorded_omni
for gpu in "${GPUS[@]}"; do
  free="$(gpu_free_mib "$gpu")"; (( free >= MIN_E5_START_FREE_MIB )) || { echo "GPU $gpu has only ${free}MiB free" >&2; exit 5; }
done
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT_ROOT/gpu_before.csv"
progress_monitor & MONITOR_PIDS+=("$!")
run_concurrent_encoding
assemble_all
run_e5_evaluation
final_audit
for pid in "${MONITOR_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
MONITOR_PIDS=()
RUN_STATE="COMPLETE"
write_status "COMPLETE" "complete" "Test1000 delta reuse, seven E5 modes, ImageBind, reference masking, and statistics are complete"
trap - EXIT INT TERM
exit 0
