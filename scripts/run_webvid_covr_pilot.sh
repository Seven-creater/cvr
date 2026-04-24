#!/usr/bin/env bash
set -euo pipefail

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/webvid_covr_pilot}
WEBVID_COVR_ROOT=${WEBVID_COVR_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/02_related_cvr/webvid-covr}
SPLIT=${SPLIT:-train}
MAX_SEED_ROWS=${MAX_SEED_ROWS:-80}
SEED_OFFSET=${SEED_OFFSET:-0}
MODEL=${MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
MODEL_STAGE=${MODEL_STAGE:-instruct}
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
MAX_GPUS=${MAX_GPUS:-6}

usage() {
  cat <<'EOF'
Usage: run_webvid_covr_pilot.sh [options]

Options:
  --root PATH
  --run-root PATH
  --webvid-covr-root PATH
  --split VALUE
  --max-seed-rows N
  --seed-offset N
  --model PATH
  --base-url URL
  --model-stage VALUE
  --gpu-ids IDS
  --max-gpus N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --webvid-covr-root)
      WEBVID_COVR_ROOT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --max-seed-rows)
      MAX_SEED_ROWS="$2"
      shift 2
      ;;
    --seed-offset)
      SEED_OFFSET="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --model-stage)
      MODEL_STAGE="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --max-gpus)
      MAX_GPUS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[webvid-covr] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

count_gpu_ids() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo 0
    return
  fi
  python - "$value" <<'PY'
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(items))
PY
}

GPU_COUNT=$(count_gpu_ids "$GPU_IDS")
if (( GPU_COUNT > MAX_GPUS )); then
  echo "[resource-policy] refusing to run with GPU_COUNT=$GPU_COUNT > MAX_GPUS=$MAX_GPUS" >&2
  exit 2
fi

case "$MODEL_STAGE" in
  instruct)
    ;;
  captioner|thinking)
    echo "[resource-policy] this script runs the Instruct seeded-pair stage only; stop the current service and run a dedicated $MODEL_STAGE stage separately" >&2
    exit 2
    ;;
  *)
    echo "[resource-policy] unsupported MODEL_STAGE=$MODEL_STAGE; expected instruct, captioner, or thinking" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"

echo "[webvid-covr] start $(date)"
echo "[webvid-covr] root=$ROOT"
echo "[webvid-covr] run_root=$RUN_ROOT"
echo "[webvid-covr] webvid_covr_root=$WEBVID_COVR_ROOT"
echo "[webvid-covr] split=$SPLIT max_seed_rows=$MAX_SEED_ROWS seed_offset=$SEED_OFFSET"
echo "[webvid-covr] base_url=$BASE_URL"
echo "[webvid-covr] model=$MODEL"
echo "[resource-policy] one Omni model per run; do not keep Captioner/Instruct/Thinking loaded together"
echo "[resource-policy] model_stage=$MODEL_STAGE gpu_ids=${GPU_IDS:-unset} gpu_count=$GPU_COUNT max_gpus=$MAX_GPUS"
curl -fsS "$BASE_URL/models"
echo

python -m app.composed_sources prepare \
  --root "$ROOT" \
  --webvid-covr-root "$WEBVID_COVR_ROOT" \
  --webvid-covr-splits "$SPLIT" \
  --clip-limit 0

echo "[webvid-covr] source prepare done $(date)"

python -m app.composed_data build-seeded-pair-slice \
  --pair-seeds-path "$ROOT/metadata/webvid_covr_pair_seeds.jsonl" \
  --source-clips-path "$ROOT/metadata/source_clips_all.jsonl" \
  --output-seeds-path "$RUN_ROOT/webvid_covr_seed_slice.jsonl" \
  --output-clips-path "$RUN_ROOT/webvid_covr_seed_source_clips.jsonl" \
  --split "$SPLIT" \
  --max-seed-rows "$MAX_SEED_ROWS" \
  --seed-offset "$SEED_OFFSET"

echo "[webvid-covr] seed slice done $(date)"

python -m app.composed_data detective-annotate-clips \
  --root "$ROOT" \
  --clips-manifest-path "$RUN_ROOT/webvid_covr_seed_source_clips.jsonl" \
  --output-path "$RUN_ROOT/detective_annotations.jsonl" \
  --base-url "$BASE_URL" \
  --api-key EMPTY \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --overwrite

echo "[webvid-covr] annotation done $(date)"

python -m app.composed_data propose-seeded-pairs \
  --root "$ROOT" \
  --clip-annotations-path "$RUN_ROOT/detective_annotations.jsonl" \
  --pair-seeds-path "$RUN_ROOT/webvid_covr_seed_slice.jsonl" \
  --output-path "$RUN_ROOT/judged_pair_proposals.jsonl" \
  --accepted-output-path "$RUN_ROOT/accepted_pairs.jsonl" \
  --base-url "$BASE_URL" \
  --api-key EMPTY \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --max-accepted-pairs 10 \
  --overwrite

echo "[webvid-covr] seeded proposal and judge done $(date)"

if [ -s "$RUN_ROOT/accepted_pairs.jsonl" ]; then
  python -m app.composed_data validate-pilot \
    --root "$ROOT" \
    --pilot-jsonl-path "$RUN_ROOT/accepted_pairs.jsonl" \
    --gallery-output-path "$RUN_ROOT/gallery.jsonl" \
    --report-output-path "$RUN_ROOT/pilot_review.md"
else
  echo "[webvid-covr] no accepted pairs; skip validate-pilot"
fi

echo "[verify] outputs"
ls -lh "$RUN_ROOT/webvid_covr_seed_slice.jsonl"
ls -lh "$RUN_ROOT/webvid_covr_seed_source_clips.jsonl"
ls -lh "$RUN_ROOT/detective_annotations.jsonl"
ls -lh "$RUN_ROOT/judged_pair_proposals.jsonl"
ls -lh "$RUN_ROOT/accepted_pairs.jsonl" || true
ls -lh "$RUN_ROOT/gallery.jsonl" || true
cat "$RUN_ROOT/pilot_review.md" || true

echo "[webvid-covr] done $(date)"
