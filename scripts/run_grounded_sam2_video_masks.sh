#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT=${DATA_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
MASK_ROOT=${MASK_ROOT:-$MODEL_ROOT/MaskEdit}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/mask_guided_vace_plan}
MASK_PLAN=${MASK_PLAN:-$RUN_ROOT/video_mask_plan.jsonl}
MASK_MANIFEST=${MASK_MANIFEST:-$RUN_ROOT/video_mask_manifest.jsonl}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-$RUN_ROOT/video_mask_manifest.generated.jsonl}
REPORT_PATH=${REPORT_PATH:-$RUN_ROOT/grounded_sam2_mask_report.md}
GROUNDED_SAM2_CODE=${GROUNDED_SAM2_CODE:-$MASK_ROOT/Grounded-SAM-2/code}
GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-$MASK_ROOT/Grounded-SAM-2/code/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py}
GROUNDING_DINO_CHECKPOINT=${GROUNDING_DINO_CHECKPOINT:-$MASK_ROOT/GroundingDINO/checkpoints}
SAM2_CONFIG=${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}
SAM2_CHECKPOINT=${SAM2_CHECKPOINT:-$MASK_ROOT/SAM2.1/checkpoints/sam2.1-hiera-large}
CONDA_ENV=${CONDA_ENV:-grounded_sam2}
GPU_IDS=${GPU_IDS:-2}
MAX_MASKS=${MAX_MASKS:-}
BOX_THRESHOLD=${BOX_THRESHOLD:-0.35}
TEXT_THRESHOLD=${TEXT_THRESHOLD:-0.25}

usage() {
  cat <<'EOF'
Usage: run_grounded_sam2_video_masks.sh [options]

Options:
  --data-root PATH
  --model-root PATH
  --mask-root PATH
  --run-root PATH
  --mask-plan PATH
  --mask-manifest PATH
  --output-manifest PATH
  --report-path PATH
  --grounded-sam2-code PATH
  --grounding-dino-config PATH
  --grounding-dino-checkpoint PATH
  --sam2-config VALUE
  --sam2-checkpoint PATH
  --conda-env NAME
  --gpu-ids IDS
  --max-masks N
  --box-threshold FLOAT
  --text-threshold FLOAT
  -h, --help

Generates binary mask videos from video_mask_plan.jsonl using
Grounded-SAM-2 / GroundingDINO / SAM2.1. This script does not start Omni
and does not run VACE.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --mask-root) MASK_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --mask-plan) MASK_PLAN="$2"; shift 2 ;;
    --mask-manifest) MASK_MANIFEST="$2"; shift 2 ;;
    --output-manifest) OUTPUT_MANIFEST="$2"; shift 2 ;;
    --report-path) REPORT_PATH="$2"; shift 2 ;;
    --grounded-sam2-code) GROUNDED_SAM2_CODE="$2"; shift 2 ;;
    --grounding-dino-config) GROUNDING_DINO_CONFIG="$2"; shift 2 ;;
    --grounding-dino-checkpoint) GROUNDING_DINO_CHECKPOINT="$2"; shift 2 ;;
    --sam2-config) SAM2_CONFIG="$2"; shift 2 ;;
    --sam2-checkpoint) SAM2_CHECKPOINT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max-masks) MAX_MASKS="$2"; shift 2 ;;
    --box-threshold) BOX_THRESHOLD="$2"; shift 2 ;;
    --text-threshold) TEXT_THRESHOLD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[grounded-sam2-masks] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$MASK_PLAN" ]]; then
  echo "[grounded-sam2-masks] missing mask plan: $MASK_PLAN" >&2
  exit 1
fi
if [[ ! -s "$MASK_MANIFEST" ]]; then
  echo "[grounded-sam2-masks] missing mask manifest: $MASK_MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$GROUNDED_SAM2_CODE" ]]; then
  echo "[grounded-sam2-masks] missing Grounded-SAM-2 code: $GROUNDED_SAM2_CODE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_MANIFEST")" "$(dirname "$REPORT_PATH")"

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$GROUNDED_SAM2_CODE:$GROUNDED_SAM2_CODE/sam2:$PYTHONPATH"

ARGS=(
  --root "$DATA_ROOT"
  --mask-plan-path "$MASK_PLAN"
  --mask-manifest-path "$MASK_MANIFEST"
  --output-manifest-path "$OUTPUT_MANIFEST"
  --report-path "$REPORT_PATH"
  --grounded-sam2-code "$GROUNDED_SAM2_CODE"
  --grounding-dino-config "$GROUNDING_DINO_CONFIG"
  --grounding-dino-checkpoint "$GROUNDING_DINO_CHECKPOINT"
  --sam2-config "$SAM2_CONFIG"
  --sam2-checkpoint "$SAM2_CHECKPOINT"
  --box-threshold "$BOX_THRESHOLD"
  --text-threshold "$TEXT_THRESHOLD"
)
if [[ -n "$MAX_MASKS" ]]; then
  ARGS+=(--max-masks "$MAX_MASKS")
fi

echo "[grounded-sam2-masks] start $(date)"
echo "[grounded-sam2-masks] mask_plan=$MASK_PLAN"
echo "[grounded-sam2-masks] mask_manifest=$MASK_MANIFEST"
echo "[grounded-sam2-masks] output_manifest=$OUTPUT_MANIFEST"
echo "[grounded-sam2-masks] grounded_sam2_code=$GROUNDED_SAM2_CODE"
echo "[grounded-sam2-masks] gpu_ids=$GPU_IDS conda_env=$CONDA_ENV"
python scripts/generate_grounded_sam2_video_masks.py "${ARGS[@]}"
echo "[grounded-sam2-masks] done $(date)"
