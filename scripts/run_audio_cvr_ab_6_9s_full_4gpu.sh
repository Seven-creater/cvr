#!/usr/bin/env bash
set -euo pipefail

if [ -f /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
  conda activate omni_src
fi

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
SINGLE_SOURCE_ROOT=${SINGLE_SOURCE_ROOT:-$ROOT/clips/audio_cvr_6_9s}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/audio_cvr_ab_6_9s_full_$(date +%Y%m%d_%H%M%S)}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
PORT=${PORT:-8093}
MODEL=${MODEL:-qwen3-omni-30b-a3b-instruct}
MODEL_PATH=${MODEL_PATH:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct}
GPU_IDS=${GPU_IDS:-0,1,2,3}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.86}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
START_OMNI=${START_OMNI:-auto}
BUILD_CLIPS=${BUILD_CLIPS:-1}
VLLM_HEALTH_TIMEOUT_SECONDS=${VLLM_HEALTH_TIMEOUT_SECONDS:-900}

CLIP_SECONDS=${CLIP_SECONDS:-8}
MIN_CLIP_SECONDS=${MIN_CLIP_SECONDS:-6}
MAX_CLIP_SECONDS=${MAX_CLIP_SECONDS:-9}
MAX_SOURCE_VIDEOS=${MAX_SOURCE_VIDEOS:-0}
MAX_SOURCE_FOLDERS=${MAX_SOURCE_FOLDERS:-0}
MAX_CLIPS=${MAX_CLIPS:-0}
MAX_A_CANDIDATES=${MAX_A_CANDIDATES:-0}
MAX_B_CANDIDATES=${MAX_B_CANDIDATES:-0}
TARGET_A_COUNT=${TARGET_A_COUNT:-1000000}
TARGET_B_COUNT=${TARGET_B_COUNT:-1000000}

PROPOSE_SHARDS=${PROPOSE_SHARDS:-64}
PROPOSE_PARALLEL_JOBS=${PROPOSE_PARALLEL_JOBS:-8}
CONCURRENCY=${CONCURRENCY:-4}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-240}
SHARD_TIMEOUT_SECONDS=${SHARD_TIMEOUT_SECONDS:-10800}
ANNOTATION_TIMEOUT_SECONDS=${ANNOTATION_TIMEOUT_SECONDS:-900}

usage() {
  cat <<'EOF'
Usage: run_audio_cvr_ab_6_9s_full_4gpu.sh [options]

Large-scale A+B Audio-CVR construction over clips/audio_cvr_6_9s.
This script runs both:
  A-line: visual_audio_anchor
  B-line: speech_audio_content with b_audio_blind_review_v2

Options:
  --root PATH
  --single-source-root PATH
  --run-root PATH
  --base-url URL
  --port N
  --model NAME
  --model-path PATH
  --gpu-ids IDS                default: 0,1,2,3
  --tensor-parallel-size N     default: 4
  --max-model-len N            default: 16384
  --max-num-seqs N             default: 8
  --start-omni auto|always|never
  --skip-clip-build
  --clip-seconds N             default: 8
  --min-clip-seconds N         default: 6
  --max-clip-seconds N         default: 9
  --no-include-tail-segment    disable tail clip for short videos
  --max-source-videos N        default: 0 (all)
  --max-source-folders N       default: 0 (all)
  --max-clips N                default: 0 (all)
  --max-a-candidates N         default: 0 (all)
  --max-b-candidates N         default: 0 (all)
  --target-a-count N           default: 1000000
  --target-b-count N           default: 1000000
  --propose-shards N           default: 64
  --propose-parallel-jobs N    default: 8
  --concurrency N              default: 4
  --request-timeout-seconds N
  --shard-timeout-seconds N
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --single-source-root) SINGLE_SOURCE_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --start-omni) START_OMNI="$2"; shift 2 ;;
    --skip-clip-build) BUILD_CLIPS=0; shift ;;
    --clip-seconds) CLIP_SECONDS="$2"; shift 2 ;;
    --min-clip-seconds) MIN_CLIP_SECONDS="$2"; shift 2 ;;
    --max-clip-seconds) MAX_CLIP_SECONDS="$2"; shift 2 ;;
    --no-include-tail-segment) INCLUDE_TAIL_FLAG=--no-include-tail-segment; shift ;;
    --max-source-videos) MAX_SOURCE_VIDEOS="$2"; shift 2 ;;
    --max-source-folders) MAX_SOURCE_FOLDERS="$2"; shift 2 ;;
    --max-clips) MAX_CLIPS="$2"; shift 2 ;;
    --max-a-candidates) MAX_A_CANDIDATES="$2"; shift 2 ;;
    --max-b-candidates) MAX_B_CANDIDATES="$2"; shift 2 ;;
    --target-a-count) TARGET_A_COUNT="$2"; shift 2 ;;
    --target-b-count) TARGET_B_COUNT="$2"; shift 2 ;;
    --propose-shards) PROPOSE_SHARDS="$2"; shift 2 ;;
    --propose-parallel-jobs) PROPOSE_PARALLEL_JOBS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --request-timeout-seconds) REQUEST_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --shard-timeout-seconds) SHARD_TIMEOUT_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[audio-cvr-ab-6-9s-full] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$REPO_ROOT/logs"

vllm_pids() {
  ps -ef | awk -v port="$PORT" '
    $0 ~ /vllm.entrypoints.openai.api_server/ && $0 ~ "--port " port {
      print $2
    }
  '
}

models_healthy() {
  curl -fsS -m 15 "$BASE_URL/models" >/dev/null
}

chat_healthy() {
  curl -fsS -m 45 "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with OK only."}],"max_tokens":8}' >/dev/null
}

service_healthy() {
  models_healthy && chat_healthy
}

stop_vllm() {
  local pids
  pids=$(vllm_pids || true)
  if [ -z "$pids" ]; then
    return
  fi
  echo "[audio-cvr-ab-6-9s-full] stopping existing vllm pids=$pids"
  # shellcheck disable=SC2086
  kill -TERM $pids || true
  sleep 15
  # shellcheck disable=SC2086
  kill -KILL $pids 2>/dev/null || true
}

start_vllm() {
  local log_path
  test -f "$MODEL_PATH/config.json" || { echo "[audio-cvr-ab-6-9s-full] missing model config: $MODEL_PATH/config.json" >&2; exit 2; }
  log_path="$REPO_ROOT/logs/qwen3_omni_${PORT}_4gpu_$(date +%Y%m%d_%H%M%S).log"
  echo "[audio-cvr-ab-6-9s-full] starting vllm log=$log_path gpu_ids=$GPU_IDS max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype bfloat16 \
    --enforce-eager \
    > "$log_path" 2>&1 < /dev/null &
  echo $! | tee "$REPO_ROOT/logs/qwen3_omni_${PORT}.pid"
  echo "[audio-cvr-ab-6-9s-full] vllm_log=$log_path"
}

wait_for_vllm() {
  local deadline
  deadline=$((SECONDS + VLLM_HEALTH_TIMEOUT_SECONDS))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if service_healthy; then
      echo "[audio-cvr-ab-6-9s-full] vllm health check passed"
      return
    fi
    sleep 10
  done
  echo "[audio-cvr-ab-6-9s-full] vllm did not become healthy within ${VLLM_HEALTH_TIMEOUT_SECONDS}s" >&2
  exit 1
}

case "$START_OMNI" in
  auto)
    if service_healthy; then
      echo "[audio-cvr-ab-6-9s-full] existing vllm service is healthy"
    else
      stop_vllm
      start_vllm
      wait_for_vllm
    fi
    ;;
  always)
    stop_vllm
    start_vllm
    wait_for_vllm
    ;;
  never)
    service_healthy || { echo "[audio-cvr-ab-6-9s-full] vllm service is not healthy and --start-omni never was set" >&2; exit 1; }
    ;;
  *) echo "[audio-cvr-ab-6-9s-full] invalid --start-omni: $START_OMNI" >&2; exit 2 ;;
esac

echo "[audio-cvr-ab-6-9s-full] run_root=$RUN_ROOT"
echo "[audio-cvr-ab-6-9s-full] single_source_root=$SINGLE_SOURCE_ROOT"
echo "[audio-cvr-ab-6-9s-full] clip_seconds=$CLIP_SECONDS min=$MIN_CLIP_SECONDS max=$MAX_CLIP_SECONDS"
echo "[audio-cvr-ab-6-9s-full] concurrency=$CONCURRENCY propose_parallel_jobs=$PROPOSE_PARALLEL_JOBS propose_shards=$PROPOSE_SHARDS"
echo "[audio-cvr-ab-6-9s-full] target_a_count=$TARGET_A_COUNT target_b_count=$TARGET_B_COUNT"

if [ "$BUILD_CLIPS" = "1" ]; then
  INCLUDE_TAIL_FLAG=${INCLUDE_TAIL_FLAG:-}
  clip_args=(
    scripts/build_audio_cvr_6_9s_clips.sh
    --root "$ROOT"
    --output-root "$SINGLE_SOURCE_ROOT"
    --clip-seconds "$CLIP_SECONDS"
    --min-clip-seconds "$MIN_CLIP_SECONDS"
    --max-clip-seconds "$MAX_CLIP_SECONDS"
    --min-clips-per-source 2
    --max-source-videos "$MAX_SOURCE_VIDEOS"
  )
  if [ -n "$INCLUDE_TAIL_FLAG" ]; then
    clip_args+=("$INCLUDE_TAIL_FLAG")
  fi
  bash "${clip_args[@]}"
fi

AUDIO_DATASET_LINE=both \
AUDIO_LINE_QUALITY_PROFILE=b_audio_blind_review_v2 \
B_ACCEPTANCE_PROFILE=b_audio_blind_review_v2 \
A_CANDIDATE_MODE=omni_first \
B_CANDIDATE_MODE=audio_first \
KEEP_ALL_B=1 \
FORCE_AUDIO_FOCUSED_REFRESH=1 \
FRESH_ANNOTATIONS=1 \
ANNOTATION_TIMEOUT_SECONDS="$ANNOTATION_TIMEOUT_SECONDS" \
REQUEST_TIMEOUT_SECONDS="$REQUEST_TIMEOUT_SECONDS" \
OMNI_TRANSIENT_RETRIES=2 \
FAIL_ON_TRANSIENT_OMNI_ERRORS=1 \
bash scripts/run_audio_lines_single_source_reuse.sh \
  --root "$ROOT" \
  --single-source-root "$SINGLE_SOURCE_ROOT" \
  --run-root "$RUN_ROOT" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --audio-dataset-line both \
  --target-a-count "$TARGET_A_COUNT" \
  --target-b-count "$TARGET_B_COUNT" \
  --max-source-folders "$MAX_SOURCE_FOLDERS" \
  --max-clips "$MAX_CLIPS" \
  --propose-shards "$PROPOSE_SHARDS" \
  --propose-parallel-jobs "$PROPOSE_PARALLEL_JOBS" \
  --concurrency "$CONCURRENCY" \
  --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS" \
  --shard-timeout-seconds "$SHARD_TIMEOUT_SECONDS" \
  --audio-line-quality-profile b_audio_blind_review_v2 \
  --acceptance-profile b_audio_blind_review_v2 \
  --a-candidate-mode omni_first \
  --b-candidate-mode audio_first \
  --min-clips-per-folder 2 \
  --min-group-clips 2 \
  --keep-all-b \
  --fresh-annotations \
  --force-audio-focused-refresh \
  --omni-transient-retries 2

python3 -m app.audio_lines_single_source build-b-splits --run-root "$RUN_ROOT"
