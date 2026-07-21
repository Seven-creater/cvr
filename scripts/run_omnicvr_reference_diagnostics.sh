#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  setsid nohup bash scripts/run_omnicvr_reference_diagnostics.sh \
    --omnicvr-root /path/to/OmniCVR \
    --adapter-root /path/to/final_forward_only \
    --output-dir /path/to/output \
    --expected-head GIT_COMMIT \
    --e5-model /path/to/e5-omni-7B \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    > /path/to/output.launch.log 2>&1 < /dev/null &

The launcher performs a zero-shot cross-benchmark diagnostic on the official
OmniCVR audio-centered split. It never trains or modifies an adapter. Dataset
downloads use hf-mirror only. ModelScope is used only when an explicit,
verified --modelscope-dataset-id is supplied.
EOF
}

OMNICVR_ROOT=""
ADAPTER_ROOT=""
OUTPUT_DIR=""
EXPECTED_HEAD=""
E5_MODEL=""
GPU_IDS="0,1,2,3,4,5,6,7"
SEEDS="13,23,42,71,101"
QUERY_COUNT=1000
GALLERY_SIZE=2000
HF_DATASET_ID="Jun-Yang/OmniCVR"
MODELSCOPE_DATASET_ID=""
VIDEO_FPS=1
VIDEO_MAX_PIXELS=401408
BOOTSTRAP_SAMPLES=20000
PERMUTATION_SAMPLES=20000
MAX_EVAL_JOBS=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --omnicvr-root) OMNICVR_ROOT="$2"; shift 2 ;;
    --adapter-root) ADAPTER_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --expected-head) EXPECTED_HEAD="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --query-count) QUERY_COUNT="$2"; shift 2 ;;
    --gallery-size) GALLERY_SIZE="$2"; shift 2 ;;
    --modelscope-dataset-id) MODELSCOPE_DATASET_ID="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --video-max-pixels) VIDEO_MAX_PIXELS="$2"; shift 2 ;;
    --bootstrap-samples) BOOTSTRAP_SAMPLES="$2"; shift 2 ;;
    --permutation-samples) PERMUTATION_SAMPLES="$2"; shift 2 ;;
    --max-eval-jobs) MAX_EVAL_JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for name in OMNICVR_ROOT ADAPTER_ROOT OUTPUT_DIR EXPECTED_HEAD E5_MODEL; do
  [[ -n "${!name}" ]] || { echo "ERROR: --$(tr '_' '-' <<< "${name,,}") is required" >&2; exit 2; }
done

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
[[ "${#GPU_ARRAY[@]}" -ge 2 ]] || { echo "ERROR: at least two GPU ids are required" >&2; exit 2; }
[[ "${#SEED_ARRAY[@]}" -gt 0 ]] || { echo "ERROR: seeds must not be empty" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR/logs" "$OMNICVR_ROOT"
STATUS_PATH="$OUTPUT_DIR/status.json"
STARTED_AT="$(date -Iseconds)"
CHILD_PIDS=()

write_status() {
  local state="$1" stage="$2" message="$3"
  python3 - "$STATUS_PATH" "$state" "$stage" "$message" "$STARTED_AT" <<'PY'
import json, pathlib, sys
path, state, stage, message, started = sys.argv[1:]
payload = {"state": state, "stage": stage, "message": message, "started_at": started}
pathlib.Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    write_status "FAILED" "launcher" "exit_code=$code"
  fi
}
trap on_exit EXIT INT TERM

write_status "RUNNING" "git_audit" "checking clean GitHub revision"
ACTUAL_HEAD="$(git rev-parse HEAD)"
[[ "$ACTUAL_HEAD" == "$EXPECTED_HEAD" ]] || { echo "ERROR: HEAD=$ACTUAL_HEAD expected=$EXPECTED_HEAD" >&2; exit 3; }
[[ -z "$(git status --short)" ]] || { echo "ERROR: git worktree is not clean" >&2; git status --short; exit 3; }
python3 -m py_compile app/e5_audio_delta_train.py app/audio_cvr_paper_experiment.py
python3 -m unittest \
  tests.test_e5_audio_delta_train.E5AudioDeltaTrainTests.test_prepare_omnicvr_and_per_query_reference_mask \
  tests.test_e5_audio_delta_train.E5AudioDeltaTrainTests.test_prepare_omnicvr_rejects_missing_source_candidate -v

DATASET_REPO="$OMNICVR_ROOT/repository"
ANNOTATION_PATH="$DATASET_REPO/omnicvr.jsonl"
ARCHIVE_DIR="$DATASET_REPO/videos"
EXTRACTED_VIDEOS="$OMNICVR_ROOT/extracted_videos"

download_hf_mirror() {
  mkdir -p "$DATASET_REPO"
  export HF_ENDPOINT="https://hf-mirror.com"
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_DISABLE_XET=1
  if command -v hf >/dev/null 2>&1; then
    hf download "$HF_DATASET_ID" --repo-type dataset --local-dir "$DATASET_REPO"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$HF_DATASET_ID" --repo-type dataset --local-dir "$DATASET_REPO"
  else
    echo "ERROR: neither hf nor huggingface-cli is installed" >&2
    return 1
  fi
}

download_modelscope_explicit() {
  [[ -n "$MODELSCOPE_DATASET_ID" ]] || return 1
  command -v modelscope >/dev/null 2>&1 || { echo "ERROR: modelscope CLI is unavailable" >&2; return 1; }
  echo "Using explicitly supplied ModelScope dataset: $MODELSCOPE_DATASET_ID"
  modelscope download --dataset "$MODELSCOPE_DATASET_ID" --local_dir "$DATASET_REPO"
}

if [[ ! -s "$ANNOTATION_PATH" ]] || ! compgen -G "$ARCHIVE_DIR/omnivideos-*.tar" >/dev/null; then
  write_status "RUNNING" "download" "downloading OmniCVR through hf-mirror"
  if ! download_hf_mirror > "$OUTPUT_DIR/logs/download_hf_mirror.log" 2>&1; then
    echo "hf-mirror download failed; see $OUTPUT_DIR/logs/download_hf_mirror.log" >&2
    if [[ -n "$MODELSCOPE_DATASET_ID" ]]; then
      download_modelscope_explicit > "$OUTPUT_DIR/logs/download_modelscope.log" 2>&1
    else
      echo "No verified ModelScope dataset id was supplied; refusing to guess or use direct Hugging Face." >&2
      exit 4
    fi
  fi
fi
[[ -s "$ANNOTATION_PATH" ]] || { echo "ERROR: annotation file missing after download: $ANNOTATION_PATH" >&2; exit 4; }
compgen -G "$ARCHIVE_DIR/omnivideos-*.tar" >/dev/null || { echo "ERROR: OmniCVR video archives are missing" >&2; exit 4; }

mkdir -p "$EXTRACTED_VIDEOS"
EXTRACT_MARKER="$EXTRACTED_VIDEOS/.omnicvr_extracted"
if [[ ! -f "$EXTRACT_MARKER" ]]; then
  write_status "RUNNING" "extract" "extracting official video archives"
  for archive in "$ARCHIVE_DIR"/omnivideos-*.tar; do
    tar -xf "$archive" -C "$EXTRACTED_VIDEOS"
  done
  VIDEO_COUNT="$(find "$EXTRACTED_VIDEOS" -type f -name 'omnicvr_video*.mp4' | wc -l)"
  [[ "$VIDEO_COUNT" -gt 0 ]] || { echo "ERROR: no videos found after extraction" >&2; exit 4; }
  printf '%s\n' "$VIDEO_COUNT" > "$EXTRACT_MARKER"
fi

RECORDS_DIR="$OUTPUT_DIR/records_audio_center"
write_status "RUNNING" "prepare" "validating official queries and candidate galleries"
python3 -m app.e5_audio_delta_train prepare-omnicvr \
  --annotation-path "$ANNOTATION_PATH" \
  --videos-dir "$EXTRACTED_VIDEOS" \
  --output-dir "$RECORDS_DIR" \
  --start-index 0 \
  --query-count "$QUERY_COUNT" \
  --expected-gallery-size "$GALLERY_SIZE" \
  > "$OUTPUT_DIR/logs/prepare_omnicvr.log" 2>&1

python3 - "$RECORDS_DIR/summary.json" "$QUERY_COUNT" "$GALLERY_SIZE" <<'PY'
import json, pathlib, sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_q, expected_g = map(int, sys.argv[2:])
assert summary["query_count"] == expected_q, summary
assert summary["candidate_count_min"] == expected_g == summary["candidate_count_max"], summary
assert summary["source_in_candidates_count"] == expected_q, summary
assert summary["target_in_candidates_count"] == expected_q, summary
assert summary["missing_media_count"] == 0, summary
PY

CACHE_VAT="$OUTPUT_DIR/cache_V_A_T"
CACHE_VT="$OUTPUT_DIR/cache_V_T"
cache_mode() {
  local mode="$1" gpu="$2" cache="$3" audio_mode="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -m app.e5_audio_delta_train cache-embeddings \
    --records-dir "$RECORDS_DIR" \
    --output-dir "$cache" \
    --e5-model "$E5_MODEL" \
    --device cuda \
    --torch-dtype bfloat16 \
    --attn-implementation sdpa \
    --batch-size 1 \
    --video-max-pixels "$VIDEO_MAX_PIXELS" \
    --video-fps "$VIDEO_FPS" \
    --video-audio-mode "$audio_mode" \
    --query-input-mode composed \
    --document-input-mode video \
    --local-segments 0 \
    --skip-train \
    > "$OUTPUT_DIR/logs/cache_${mode}.log" 2>&1
}

write_status "RUNNING" "cache" "encoding V+A+T and V+T in parallel"
cache_mode V_A_T "${GPU_ARRAY[0]}" "$CACHE_VAT" on & CHILD_PIDS+=("$!")
cache_mode V_T "${GPU_ARRAY[1]}" "$CACHE_VT" off & CHILD_PIDS+=("$!")
for pid in "${CHILD_PIDS[@]}"; do wait "$pid"; done
CHILD_PIDS=()

for seed in "${SEED_ARRAY[@]}"; do
  adapter="$ADAPTER_ROOT/seed_${seed}/adapter"
  [[ -s "$adapter/adapter.pt" ]] || { echo "ERROR: missing adapter for seed $seed: $adapter/adapter.pt" >&2; exit 5; }
  [[ -s "$adapter/adapter_config.json" ]] || { echo "ERROR: missing adapter config for seed $seed" >&2; exit 5; }
done

eval_one() {
  local seed="$1" mode="$2" cache="$3" no_reference="$4" gpu="$5"
  local eval_dir="$OUTPUT_DIR/final_forward_only/seed_${seed}/eval_${mode}"
  local args=(python3 -m app.e5_audio_delta_train eval
    --cache-dir "$cache"
    --adapter-dir "$ADAPTER_ROOT/seed_${seed}/adapter"
    --output-dir "$eval_dir"
    --topk 1,5,10
    --save-topk 20
    --device cuda)
  [[ "$no_reference" == "true" ]] && args+=(--exclude-query-reference)
  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" > "$OUTPUT_DIR/logs/eval_seed${seed}_${mode}.log" 2>&1
}

write_status "RUNNING" "eval" "evaluating five frozen adapters with and without per-query source"
job_index=0
for seed in "${SEED_ARRAY[@]}"; do
  for spec in \
    "V_A_T|$CACHE_VAT|false" \
    "V_T|$CACHE_VT|false" \
    "V_A_T_no_ref|$CACHE_VAT|true" \
    "V_T_no_ref|$CACHE_VT|true"; do
    IFS='|' read -r mode cache no_reference <<< "$spec"
    gpu="${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
    eval_one "$seed" "$mode" "$cache" "$no_reference" "$gpu" &
    CHILD_PIDS+=("$!")
    job_index=$((job_index + 1))
    if [[ "${#CHILD_PIDS[@]}" -ge "$MAX_EVAL_JOBS" ]]; then
      for pid in "${CHILD_PIDS[@]}"; do wait "$pid"; done
      CHILD_PIDS=()
    fi
  done
done
for pid in "${CHILD_PIDS[@]:-}"; do [[ -n "$pid" ]] && wait "$pid"; done
CHILD_PIDS=()

write_status "RUNNING" "aggregate" "computing paired cross-benchmark statistics"
python3 -m app.audio_cvr_paper_experiment aggregate-final \
  --input-root "$OUTPUT_DIR/final_forward_only" \
  --output-dir "$OUTPUT_DIR/statistics" \
  --required-seeds "$SEEDS" \
  --primary-mode V_A_T \
  --reference-mode V_T \
  --comparison V_A_T:V_T \
  --comparison V_A_T_no_ref:V_A_T \
  --comparison V_T_no_ref:V_T \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --permutation-samples "$PERMUTATION_SAMPLES" \
  > "$OUTPUT_DIR/logs/aggregate.log" 2>&1

write_status "COMPLETE" "done" "OmniCVR audio-centered reference diagnostics completed"
trap - EXIT INT TERM
echo "COMPLETE: $OUTPUT_DIR"
