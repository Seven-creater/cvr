#!/usr/bin/env bash
set -euo pipefail

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/omni_detective_pilot_20260422}
MODEL=${MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
SOURCE_CLIPS=${SOURCE_CLIPS:-$ROOT/metadata/source_clips_all.jsonl}
MAX_SOURCE_VIDEOS=${MAX_SOURCE_VIDEOS:-80}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-8}
MODEL_STAGE=${MODEL_STAGE:-instruct}
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
MAX_GPUS=${MAX_GPUS:-6}

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
    echo "[resource-policy] this script runs the Instruct pair-construction stage only; stop the current service and run a dedicated $MODEL_STAGE stage separately" >&2
    exit 2
    ;;
  *)
    echo "[resource-policy] unsupported MODEL_STAGE=$MODEL_STAGE; expected instruct, captioner, or thinking" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"

echo "[omni-detective] start $(date)"
echo "[omni-detective] root=$ROOT"
echo "[omni-detective] run_root=$RUN_ROOT"
echo "[omni-detective] source_clips=$SOURCE_CLIPS"
echo "[omni-detective] base_url=$BASE_URL"
echo "[omni-detective] model=$MODEL"
echo "[resource-policy] one Omni model per run; do not keep Captioner/Instruct/Thinking loaded together"
echo "[resource-policy] model_stage=$MODEL_STAGE gpu_ids=${GPU_IDS:-unset} gpu_count=$GPU_COUNT max_gpus=$MAX_GPUS"
curl -fsS "$BASE_URL/models"
echo

python -m app.composed_data plan-detective-clips \
  --root "$ROOT" \
  --source-clips-path "$SOURCE_CLIPS" \
  --clip-plan-output-path "$RUN_ROOT/clip_plan_detective.jsonl" \
  --clip-groups-output-path "$RUN_ROOT/clip_groups.jsonl" \
  --max-source-videos "$MAX_SOURCE_VIDEOS" \
  --segment-seconds "$SEGMENT_SECONDS" \
  --min-clip-seconds 3 \
  --max-clip-seconds 15

echo "[omni-detective] planning done $(date)"

python -m app.composed_data extract-clips \
  --root "$ROOT" \
  --plan-path "$RUN_ROOT/clip_plan_detective.jsonl" \
  --output-manifest-path "$RUN_ROOT/extracted_event_clips.jsonl" \
  --overwrite

echo "[omni-detective] extraction done $(date)"

python -m app.composed_data detective-annotate-clips \
  --root "$ROOT" \
  --clips-manifest-path "$RUN_ROOT/extracted_event_clips.jsonl" \
  --output-path "$RUN_ROOT/detective_annotations.jsonl" \
  --base-url "$BASE_URL" \
  --api-key EMPTY \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --overwrite

echo "[omni-detective] annotation done $(date)"

python -m app.composed_data propose-group-pairs \
  --root "$ROOT" \
  --clip-annotations-path "$RUN_ROOT/detective_annotations.jsonl" \
  --clip-groups-path "$RUN_ROOT/clip_groups.jsonl" \
  --output-path "$RUN_ROOT/judged_pair_proposals.jsonl" \
  --accepted-output-path "$RUN_ROOT/accepted_pairs.jsonl" \
  --base-url "$BASE_URL" \
  --api-key EMPTY \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --max-accepted-pairs 10 \
  --overwrite

echo "[omni-detective] group proposal and judge done $(date)"

if [ -s "$RUN_ROOT/accepted_pairs.jsonl" ]; then
  python -m app.composed_data validate-pilot \
    --root "$ROOT" \
    --pilot-jsonl-path "$RUN_ROOT/accepted_pairs.jsonl" \
    --gallery-output-path "$RUN_ROOT/gallery.jsonl" \
    --report-output-path "$RUN_ROOT/pilot_review.md"
else
  echo "[omni-detective] no accepted pairs; skip validate-pilot"
fi

echo "[verify] outputs"
ls -lh "$RUN_ROOT/clip_plan_detective.jsonl"
ls -lh "$RUN_ROOT/clip_groups.jsonl"
ls -lh "$RUN_ROOT/extracted_event_clips.jsonl"
ls -lh "$RUN_ROOT/detective_annotations.jsonl"
ls -lh "$RUN_ROOT/judged_pair_proposals.jsonl"
ls -lh "$RUN_ROOT/accepted_pairs.jsonl" || true
ls -lh "$RUN_ROOT/gallery.jsonl" || true
cat "$RUN_ROOT/pilot_review.md" || true

echo "[omni-detective] done $(date)"
