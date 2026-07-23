#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  setsid nohup bash scripts/run_audio_cvr_audiovlm2vec_8gpu.sh \
    --output-dir runs/audiovlm2vec_reference_diagnostic_TAG \
    --expected-head GIT_SHA \
    --python /path/to/python \
    --audio-test /path/to/test_main_1000.jsonl \
    --audio-train /path/to/b_train_bidirectional_triplets.jsonl \
    --audio-val /path/to/val.jsonl \
    --audio-test-sha256 SHA256 \
    --omnicvr-annotations /path/to/omnicvr.jsonl \
    --omnicvr-videos /path/to/videos \
    --media-root /path/to/media/root \
    --qwen2-audio /path/to/Qwen2-Audio \
    --qwen2-vl /path/to/Qwen2-VL-7B-Instruct \
    --vlm2vec-adapter /path/to/VLM2Vec-Qwen2VL-7B \
    --gpu-ids 0,1,2,3,4,5,6,7

The launcher is resumable. Captions and embeddings are written atomically per
media/item. It never modifies datasets, E5/ImageBind outputs, or foreign GPU
processes.
EOF
}

OUTPUT_DIR=""
EXPECTED_HEAD=""
PYTHON="python3"
AUDIO_TEST=""
AUDIO_TRAIN=""
AUDIO_VAL=""
AUDIO_TEST_SHA256=""
OMNICVR_ANNOTATIONS=""
OMNICVR_VIDEOS=""
QWEN2_AUDIO=""
QWEN2_VL=""
VLM2VEC_ADAPTER=""
GPU_IDS="0,1,2,3,4,5,6,7"
CAPTION_RETRIES=4
ENCODING_RETRIES=4
ENCODING_BATCH_SIZE=2
STAT_ITERATIONS=20000
MEDIA_ROOTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --expected-head) EXPECTED_HEAD="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --audio-test) AUDIO_TEST="$2"; shift 2 ;;
    --audio-train) AUDIO_TRAIN="$2"; shift 2 ;;
    --audio-val) AUDIO_VAL="$2"; shift 2 ;;
    --audio-test-sha256) AUDIO_TEST_SHA256="$2"; shift 2 ;;
    --omnicvr-annotations) OMNICVR_ANNOTATIONS="$2"; shift 2 ;;
    --omnicvr-videos) OMNICVR_VIDEOS="$2"; shift 2 ;;
    --media-root) MEDIA_ROOTS+=("$2"); shift 2 ;;
    --qwen2-audio) QWEN2_AUDIO="$2"; shift 2 ;;
    --qwen2-vl) QWEN2_VL="$2"; shift 2 ;;
    --vlm2vec-adapter) VLM2VEC_ADAPTER="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --caption-retries) CAPTION_RETRIES="$2"; shift 2 ;;
    --encoding-retries) ENCODING_RETRIES="$2"; shift 2 ;;
    --encoding-batch-size) ENCODING_BATCH_SIZE="$2"; shift 2 ;;
    --stat-iterations) STAT_ITERATIONS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for name in OUTPUT_DIR EXPECTED_HEAD AUDIO_TEST AUDIO_TRAIN AUDIO_VAL AUDIO_TEST_SHA256 \
  OMNICVR_ANNOTATIONS OMNICVR_VIDEOS QWEN2_AUDIO QWEN2_VL VLM2VEC_ADAPTER; do
  [[ -n "${!name}" ]] || { echo "ERROR: $name is required" >&2; exit 2; }
done

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
[[ "${#GPU_ARRAY[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs are required" >&2; exit 2; }
[[ "${#MEDIA_ROOTS[@]}" -gt 0 ]] || MEDIA_ROOTS+=("$(pwd)")

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/workers"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
STATUS_PATH="$OUTPUT_DIR/status.json"
STARTED_AT="$(date -Iseconds)"
CHILD_PIDS=()

write_status() {
  local state="$1" stage="$2" message="$3"
  "$PYTHON" - "$STATUS_PATH" "$state" "$stage" "$message" "$STARTED_AT" <<'PY'
import json, os, pathlib, sys, tempfile
path, state, stage, message, started = sys.argv[1:]
payload = {
    "state": state,
    "stage": stage,
    "message": message,
    "started_at": started,
    "launcher_pid": os.getppid(),
}
target = pathlib.Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
    temporary = pathlib.Path(handle.name)
temporary.replace(target)
PY
}

cleanup_children() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

on_exit() {
  local code=$?
  cleanup_children
  if [[ "$code" -ne 0 ]]; then
    write_status "FAILED" "launcher" "exit_code=$code; all item caches retained"
  fi
}
trap on_exit EXIT INT TERM

printf '%s\n' "$$" > "$OUTPUT_DIR/launcher.pid"
ps -o pgid= -p "$$" | tr -d ' ' > "$OUTPUT_DIR/launcher.pgid"

write_status "RUNNING" "git_audit" "checking immutable GitHub revision"
ACTUAL_HEAD="$(git rev-parse HEAD)"
[[ "$ACTUAL_HEAD" == "$EXPECTED_HEAD" ]] || {
  echo "ERROR: HEAD=$ACTUAL_HEAD expected=$EXPECTED_HEAD" >&2
  exit 3
}
[[ -z "$(git status --short)" ]] || {
  echo "ERROR: tracked or untracked repository changes are present" >&2
  git status --short
  exit 3
}
"$PYTHON" -m py_compile app/audio_cvr_audiovlm2vec.py
"$PYTHON" -m unittest tests.test_audio_cvr_audiovlm2vec -v

for path in "$AUDIO_TEST" "$AUDIO_TRAIN" "$AUDIO_VAL" "$OMNICVR_ANNOTATIONS" \
  "$OMNICVR_VIDEOS" "$QWEN2_AUDIO" "$QWEN2_VL"; do
  [[ -e "$path" ]] || { echo "ERROR: required path missing: $path" >&2; exit 4; }
done

write_status "RUNNING" "model_adapter" "validating public VLM2Vec-Qwen2VL adapter"
if [[ ! -s "$VLM2VEC_ADAPTER/adapter_config.json" ]] || \
   { [[ ! -s "$VLM2VEC_ADAPTER/adapter_model.bin" ]] && [[ ! -s "$VLM2VEC_ADAPTER/adapter_model.safetensors" ]]; }; then
  mkdir -p "$VLM2VEC_ADAPTER"
  export HF_ENDPOINT="https://hf-mirror.com"
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_DISABLE_XET=1
  "$PYTHON" - "$VLM2VEC_ADAPTER" <<'PY'
from huggingface_hub import snapshot_download
import sys
snapshot_download(
    repo_id="TIGER-Lab/VLM2Vec-Qwen2VL-7B",
    local_dir=sys.argv[1],
    allow_patterns=["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors", "README.md"],
)
PY
fi
[[ -s "$VLM2VEC_ADAPTER/adapter_config.json" ]] || {
  echo "ERROR: VLM2Vec adapter download is incomplete" >&2
  exit 4
}

RECORDS_DIR="$OUTPUT_DIR/records"
MEDIA_INVENTORY="$RECORDS_DIR/media_inventory.jsonl"
CAPTION_CACHE="$OUTPUT_DIR/audio_caption_cache"
EMBEDDING_INVENTORY="$OUTPUT_DIR/vlm2vec_embedding_inventory.jsonl"
EMBEDDING_CACHE="$OUTPUT_DIR/vlm2vec_embedding_cache"
ZERO_SHOT_DIR="$OUTPUT_DIR/zero_shot"
ADAPTER_ROOT="$OUTPUT_DIR/task_adapters"
FINAL_DIR="$OUTPUT_DIR/final_results"

if [[ ! -s "$RECORDS_DIR/prepare_summary.json" ]]; then
  write_status "RUNNING" "prepare" "normalizing Audio-CVR1000, train89, val28, and OmniCVR1000"
  prepare_args=(
    "$PYTHON" -m app.audio_cvr_audiovlm2vec prepare
    --audio-test "$AUDIO_TEST"
    --audio-train "$AUDIO_TRAIN"
    --audio-val "$AUDIO_VAL"
    --audio-test-sha256 "$AUDIO_TEST_SHA256"
    --omnicvr-annotations "$OMNICVR_ANNOTATIONS"
    --omnicvr-videos "$OMNICVR_VIDEOS"
    --omnicvr-query-count 1000
    --omnicvr-gallery-size 2000
    --output-dir "$RECORDS_DIR"
  )
  for root in "${MEDIA_ROOTS[@]}"; do prepare_args+=(--media-root "$root"); done
  "${prepare_args[@]}" > "$OUTPUT_DIR/logs/prepare.log" 2>&1
fi

run_caption_workers() {
  local index pid failed=0
  CHILD_PIDS=()
  for index in "${!GPU_ARRAY[@]}"; do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$index]}" \
      nice -n 5 "$PYTHON" -m app.audio_cvr_audiovlm2vec caption-audio \
        --inventory "$MEDIA_INVENTORY" \
        --cache-dir "$CAPTION_CACHE" \
        --model "$QWEN2_AUDIO" \
        --shard-index "$index" \
        --shard-count 8 \
        --device cuda \
        --retries "$CAPTION_RETRIES" \
        > "$OUTPUT_DIR/logs/caption_shard_${index}.log" 2>&1 &
    pid=$!
    CHILD_PIDS+=("$pid")
    printf '%s\n' "$pid" > "$OUTPUT_DIR/workers/caption_shard_${index}.pid"
  done
  for pid in "${CHILD_PIDS[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  CHILD_PIDS=()
  return "$failed"
}

write_status "RUNNING" "caption_audio" "eight GPUs caption unique audio; every item is atomic"
run_caption_workers || {
  echo "ERROR: one or more caption workers failed; rerun the same launcher to resume" >&2
  exit 5
}
"$PYTHON" -m app.audio_cvr_audiovlm2vec audit-captions \
  --inventory "$MEDIA_INVENTORY" \
  --cache-dir "$CAPTION_CACHE" \
  --output "$OUTPUT_DIR/caption_audit.json" \
  > "$OUTPUT_DIR/logs/caption_audit.log" 2>&1
"$PYTHON" - "$OUTPUT_DIR/caption_audit.json" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["complete"], value
PY

if [[ ! -s "$EMBEDDING_INVENTORY" ]]; then
  write_status "RUNNING" "embedding_inventory" "building fixed V+T and audio-as-text inventories"
  "$PYTHON" -m app.audio_cvr_audiovlm2vec prepare-embedding-inventory \
    --records-dir "$RECORDS_DIR" \
    --caption-cache "$CAPTION_CACHE" \
    --output "$EMBEDDING_INVENTORY" \
    > "$OUTPUT_DIR/logs/embedding_inventory.log" 2>&1
fi

run_embedding_workers() {
  local index pid failed=0
  CHILD_PIDS=()
  for index in "${!GPU_ARRAY[@]}"; do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$index]}" \
      nice -n 5 "$PYTHON" -m app.audio_cvr_audiovlm2vec encode \
        --inventory "$EMBEDDING_INVENTORY" \
        --cache-dir "$EMBEDDING_CACHE" \
        --base-model "$QWEN2_VL" \
        --adapter-model "$VLM2VEC_ADAPTER" \
        --shard-index "$index" \
        --shard-count 8 \
        --device cuda \
        --batch-size "$ENCODING_BATCH_SIZE" \
        --retries "$ENCODING_RETRIES" \
        > "$OUTPUT_DIR/logs/encode_shard_${index}.log" 2>&1 &
    pid=$!
    CHILD_PIDS+=("$pid")
    printf '%s\n' "$pid" > "$OUTPUT_DIR/workers/encode_shard_${index}.pid"
  done
  for pid in "${CHILD_PIDS[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  CHILD_PIDS=()
  return "$failed"
}

write_status "RUNNING" "encode_vlm2vec" "eight GPUs encode VLM2Vec items; every embedding is atomic"
run_embedding_workers || {
  echo "ERROR: one or more embedding workers failed; rerun the same launcher to resume" >&2
  exit 6
}
"$PYTHON" -m app.audio_cvr_audiovlm2vec audit-embeddings \
  --inventory "$EMBEDDING_INVENTORY" \
  --cache-dir "$EMBEDDING_CACHE" \
  --output "$OUTPUT_DIR/embedding_audit.json" \
  > "$OUTPUT_DIR/logs/embedding_audit.log" 2>&1
"$PYTHON" - "$OUTPUT_DIR/embedding_audit.json" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["complete"], value
assert len(value["dimensions"]) == 1, value
PY

write_status "RUNNING" "zero_shot_evaluation" "evaluating Audio-CVR and OmniCVR with exact source masking"
"$PYTHON" -m app.audio_cvr_audiovlm2vec evaluate-zero-shot \
  --records-dir "$RECORDS_DIR" \
  --inventory "$EMBEDDING_INVENTORY" \
  --cache-dir "$EMBEDDING_CACHE" \
  --output-dir "$ZERO_SHOT_DIR" \
  --iterations "$STAT_ITERATIONS" \
  > "$OUTPUT_DIR/logs/evaluate_zero_shot.log" 2>&1

SEEDS=(13 23 42 71 101)
write_status "RUNNING" "train_task_adapters" "training five fixed rank32 adapters without test selection"
CHILD_PIDS=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  if [[ -s "$ADAPTER_ROOT/seed_${seed}/adapter.pt" ]] && \
     [[ -s "$ADAPTER_ROOT/seed_${seed}/train_summary.json" ]]; then
    continue
  fi
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$index]}" \
    "$PYTHON" -m app.audio_cvr_audiovlm2vec train-adapter \
      --records-dir "$RECORDS_DIR" \
      --inventory "$EMBEDDING_INVENTORY" \
      --cache-dir "$EMBEDDING_CACHE" \
      --output-dir "$ADAPTER_ROOT/seed_${seed}" \
      --seed "$seed" \
      --rank 32 \
      --steps 400 \
      --learning-rate 0.001 \
      --batch-size 8 \
      --device cuda \
      > "$OUTPUT_DIR/logs/train_adapter_seed_${seed}.log" 2>&1 &
  pid=$!
  CHILD_PIDS+=("$pid")
  printf '%s\n' "$pid" > "$OUTPUT_DIR/workers/train_adapter_seed_${seed}.pid"
done
failed=0
for pid in "${CHILD_PIDS[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
CHILD_PIDS=()
[[ "$failed" -eq 0 ]] || {
  echo "ERROR: task adapter training failed; completed adapters and losses are retained" >&2
  exit 7
}

write_status "RUNNING" "adapter_evaluation" "evaluating five adapters on both benchmarks"
adapter_args=()
for seed in "${SEEDS[@]}"; do adapter_args+=(--adapter-dir "$ADAPTER_ROOT/seed_${seed}"); done
CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON" -m app.audio_cvr_audiovlm2vec evaluate-adapters \
  --records-dir "$RECORDS_DIR" \
  --inventory "$EMBEDDING_INVENTORY" \
  --cache-dir "$EMBEDDING_CACHE" \
  "${adapter_args[@]}" \
  --output-dir "$FINAL_DIR" \
  --device cuda \
  --iterations "$STAT_ITERATIONS" \
  > "$OUTPUT_DIR/logs/evaluate_adapters.log" 2>&1

write_status "RUNNING" "summarize" "writing paper tables and provenance"
"$PYTHON" -m app.audio_cvr_audiovlm2vec summarize \
  --zero-shot "$ZERO_SHOT_DIR/zero_shot_results.json" \
  --adapter-results "$FINAL_DIR/adapter_results.json" \
  --prepare-summary "$RECORDS_DIR/prepare_summary.json" \
  --output-dir "$FINAL_DIR" \
  > "$OUTPUT_DIR/logs/summarize.log" 2>&1

"$PYTHON" - "$OUTPUT_DIR" "$AUDIO_TEST_SHA256" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected_sha = sys.argv[2]
prepare = json.loads((root / "records/prepare_summary.json").read_text(encoding="utf-8"))
caption = json.loads((root / "caption_audit.json").read_text(encoding="utf-8"))
embedding = json.loads((root / "embedding_audit.json").read_text(encoding="utf-8"))
adapter = json.loads((root / "final_results/adapter_results.json").read_text(encoding="utf-8"))
violations = []
if prepare["audio_test_sha256"] != expected_sha:
    violations.append("test_sha256_mismatch")
if prepare["audio"]["test"]["count"] != 1000:
    violations.append("audiocvr_query_count")
if prepare["audio"]["train"]["count"] != 89:
    violations.append("audiocvr_train_count")
if prepare["audio"]["val"]["count"] != 28:
    violations.append("audiocvr_val_count")
if prepare["omnicvr"]["query_count"] != 1000:
    violations.append("omnicvr_query_count")
if not caption["complete"]:
    violations.append("caption_cache_incomplete")
if not embedding["complete"]:
    violations.append("embedding_cache_incomplete")
if adapter["seeds"] != [13, 23, 42, 71, 101]:
    violations.append("adapter_seeds")
payload = {
    "test_sha256": prepare["audio_test_sha256"],
    "audiocvr_query_count": prepare["audio"]["test"]["count"],
    "audiocvr_train_count": prepare["audio"]["train"]["count"],
    "audiocvr_val_count": prepare["audio"]["val"]["count"],
    "omnicvr_query_count": prepare["omnicvr"]["query_count"],
    "caption_count": caption["complete_count"],
    "embedding_count": embedding["complete_count"],
    "adapter_seeds": adapter["seeds"],
    "violation_count": len(violations),
    "violations": violations,
}
(root / "final_results/final_audit.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
assert not violations, payload
PY

tar -czf "$OUTPUT_DIR/audiovlm2vec_paper_results.tar.gz" \
  -C "$OUTPUT_DIR" records/prepare_summary.json caption_audit.json embedding_audit.json \
  zero_shot final_results task_adapters/*/train_summary.json task_adapters/*/loss_curve.jsonl

write_status "COMPLETE" "complete" "Audio-CVR1000 and OmniCVR1000 independent VLM2Vec diagnostics complete"
trap - EXIT INT TERM
exit 0
