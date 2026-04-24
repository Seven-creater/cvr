#!/usr/bin/env bash
set -euo pipefail

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/synthetic_video_edit_validation}
KNOWN_PAIRS=${KNOWN_PAIRS:-$RUN_ROOT/synthetic_candidate_pairs.jsonl}
CLIP_ANNOTATIONS=${CLIP_ANNOTATIONS:-$RUN_ROOT/detective_annotations.jsonl}
MODEL=${MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
MODEL_STAGE=${MODEL_STAGE:-instruct}
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
MAX_GPUS=${MAX_GPUS:-6}

usage() {
  cat <<'EOF'
Usage: run_synthetic_known_pairs_validation.sh [options]

Options:
  --root PATH
  --run-root PATH
  --known-pairs PATH
  --clip-annotations PATH
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
    --known-pairs)
      KNOWN_PAIRS="$2"
      shift 2
      ;;
    --clip-annotations)
      CLIP_ANNOTATIONS="$2"
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
      echo "[synthetic-validation] unknown argument: $1" >&2
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

if [[ "$MODEL_STAGE" != "instruct" ]]; then
  echo "[resource-policy] synthetic known-pair validation uses the Instruct verifier stage only" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

echo "[synthetic-validation] start $(date)"
echo "[synthetic-validation] root=$ROOT"
echo "[synthetic-validation] run_root=$RUN_ROOT"
echo "[synthetic-validation] known_pairs=$KNOWN_PAIRS"
echo "[synthetic-validation] clip_annotations=$CLIP_ANNOTATIONS"
echo "[synthetic-validation] base_url=$BASE_URL"
echo "[synthetic-validation] model=$MODEL"
echo "[resource-policy] one Omni model per run; do not keep Captioner/Instruct/Thinking loaded together"
echo "[resource-policy] model_stage=$MODEL_STAGE gpu_ids=${GPU_IDS:-unset} gpu_count=$GPU_COUNT max_gpus=$MAX_GPUS"
curl -fsS "$BASE_URL/models"
echo

python -m app.composed_data validate-known-pairs \
  --root "$ROOT" \
  --known-pairs-path "$KNOWN_PAIRS" \
  --clip-annotations-path "$CLIP_ANNOTATIONS" \
  --output-path "$RUN_ROOT/judged_synthetic_pair_proposals.jsonl" \
  --accepted-output-path "$RUN_ROOT/accepted_synthetic_pairs.jsonl" \
  --base-url "$BASE_URL" \
  --api-key EMPTY \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --max-accepted-pairs 10 \
  --overwrite

if [ -s "$RUN_ROOT/accepted_synthetic_pairs.jsonl" ]; then
  python -m app.composed_data validate-pilot \
    --root "$ROOT" \
    --pilot-jsonl-path "$RUN_ROOT/accepted_synthetic_pairs.jsonl" \
    --gallery-output-path "$RUN_ROOT/synthetic_gallery.jsonl" \
    --report-output-path "$RUN_ROOT/synthetic_pilot_review.md"
else
  echo "[synthetic-validation] no accepted synthetic pairs; skip validate-pilot"
fi

echo "[synthetic-validation] outputs"
ls -lh "$RUN_ROOT/judged_synthetic_pair_proposals.jsonl"
ls -lh "$RUN_ROOT/accepted_synthetic_pairs.jsonl" || true
ls -lh "$RUN_ROOT/synthetic_gallery.jsonl" || true
cat "$RUN_ROOT/synthetic_pilot_review.md" || true

echo "[synthetic-validation] done $(date)"
