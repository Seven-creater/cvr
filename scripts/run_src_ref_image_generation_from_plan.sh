#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"

SRC_REF_IMAGE_PLAN=${SRC_REF_IMAGE_PLAN:-}
MODEL_DIR=${MODEL_DIR:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/ImageGen/Qwen-Image-2512}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-}
CONDA_ENV=${CONDA_ENV:-omni_src}
GPU_IDS=${GPU_IDS:-6}
MAX_PLANS=${MAX_PLANS:-0}
STEPS=${STEPS:-30}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4.0}

usage() {
  cat <<'EOF'
Usage: run_src_ref_image_generation_from_plan.sh [options]

Options:
  --src-ref-image-plan PATH
  --model-dir PATH
  --output-manifest PATH
  --conda-env NAME
  --gpu-ids IDS
  --max-plans N
  --steps N
  --guidance-scale N

Generates VACE src_ref_images from src_ref_image_plan.jsonl using a local
Diffusers-compatible image generation model, such as Qwen-Image.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-ref-image-plan) SRC_REF_IMAGE_PLAN="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --output-manifest) OUTPUT_MANIFEST="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max-plans) MAX_PLANS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --guidance-scale) GUIDANCE_SCALE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[src-ref-gen] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$SRC_REF_IMAGE_PLAN" ]]; then
  echo "[src-ref-gen] missing src_ref image plan: $SRC_REF_IMAGE_PLAN" >&2
  exit 1
fi
if [[ -z "$OUTPUT_MANIFEST" ]]; then
  OUTPUT_MANIFEST="$(dirname "$SRC_REF_IMAGE_PLAN")/src_ref_image_generation_manifest.jsonl"
fi

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python scripts/generate_src_ref_images_from_plan.py \
  --src-ref-image-plan "$SRC_REF_IMAGE_PLAN" \
  --model-dir "$MODEL_DIR" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --max-plans "$MAX_PLANS" \
  --steps "$STEPS" \
  --guidance-scale "$GUIDANCE_SCALE"
