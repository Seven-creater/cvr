#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/vace_visual_synthetic_smoke}
VIDEO_EDIT_PLAN=${VIDEO_EDIT_PLAN:-$RUN_ROOT/video_edit_plan.jsonl}
PLAN_INDEX=${PLAN_INDEX:-1}
PLAN_ID=${PLAN_ID:-}
MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
WAN_ROOT=${WAN_ROOT:-$MODEL_ROOT/Wan2.1}
WAN_CODE=${WAN_CODE:-$WAN_ROOT/code}
WAN_CKPT=${WAN_CKPT:-$WAN_ROOT/Wan2.1-VACE-1.3B}
VACE_TASK=${VACE_TASK:-}
CONDA_ENV=${CONDA_ENV:-wan_vace}
GPU_IDS=${GPU_IDS:-0,1}
MAX_GPUS=${MAX_GPUS:-2}
USE_TORCHRUN=${USE_TORCHRUN:-0}
ULYSSES_SIZE=${ULYSSES_SIZE:-}
RING_SIZE=${RING_SIZE:-0}
SIZE=${SIZE:-832*480}
FRAME_NUM=${FRAME_NUM:-49}
SAMPLE_STEPS=${SAMPLE_STEPS:-25}
SAMPLE_GUIDE_SCALE=${SAMPLE_GUIDE_SCALE:-5.0}
OFFLOAD_MODEL=${OFFLOAD_MODEL:-False}
T5_CPU=${T5_CPU:-0}
OUT_ROOT=${OUT_ROOT:-$RUN_ROOT/visual_synthetic_smoke}
ALLOW_CPU_OFFLOAD=${ALLOW_CPU_OFFLOAD:-0}

usage() {
  cat <<'EOF'
Usage: run_vace_visual_synthetic_smoke.sh [options]

Options:
  --data-root PATH
  --run-root PATH
  --video-edit-plan PATH
  --plan-index N
  --plan-id ID
  --wan-root PATH
  --wan-code PATH
  --wan-ckpt PATH
  --vace-task vace-1.3B|vace-14B
  --conda-env NAME
  --gpu-ids IDS
  --use-torchrun 0|1
  --ulysses-size N
  --ring-size N
  --out-root PATH
  --allow-cpu-offload 0|1
  -h, --help

Generates one VACE visual synthetic target from video_edit_plan.jsonl.
The target video is remuxed with reference audio and a known-pair JSONL is
written for later validate-known-pairs. This script does not start Omni.
By default this script is GPU-only: CPU offload and T5-on-CPU are refused.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --video-edit-plan) VIDEO_EDIT_PLAN="$2"; shift 2 ;;
    --plan-index) PLAN_INDEX="$2"; shift 2 ;;
    --plan-id) PLAN_ID="$2"; shift 2 ;;
    --wan-root) WAN_ROOT="$2"; WAN_CODE="$2/code"; WAN_CKPT="$2/Wan2.1-VACE-1.3B"; shift 2 ;;
    --wan-code) WAN_CODE="$2"; shift 2 ;;
    --wan-ckpt) WAN_CKPT="$2"; shift 2 ;;
    --vace-task) VACE_TASK="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --use-torchrun) USE_TORCHRUN="$2"; shift 2 ;;
    --ulysses-size) ULYSSES_SIZE="$2"; shift 2 ;;
    --ring-size) RING_SIZE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --allow-cpu-offload) ALLOW_CPU_OFFLOAD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[vace-smoke] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

GPU_COUNT=$(python3 - <<PY
ids = [item.strip() for item in "$GPU_IDS".split(",") if item.strip()]
print(len(ids))
PY
)
if [[ "$GPU_COUNT" -gt "$MAX_GPUS" ]]; then
  echo "[vace-smoke] refusing to run with GPU_COUNT=$GPU_COUNT > MAX_GPUS=$MAX_GPUS" >&2
  exit 1
fi
if [[ "$ALLOW_CPU_OFFLOAD" != "1" ]]; then
  if [[ "$OFFLOAD_MODEL" != "False" && "$OFFLOAD_MODEL" != "false" && "$OFFLOAD_MODEL" != "0" ]]; then
    echo "[vace-smoke] refusing CPU offload: OFFLOAD_MODEL=$OFFLOAD_MODEL. Use GPU-only OFFLOAD_MODEL=False." >&2
    exit 1
  fi
  if [[ "$T5_CPU" != "0" && "$T5_CPU" != "False" && "$T5_CPU" != "false" ]]; then
    echo "[vace-smoke] refusing CPU text encoder: T5_CPU=$T5_CPU. Use GPU-only T5_CPU=0." >&2
    exit 1
  fi
fi

if [[ -z "$VACE_TASK" ]]; then
  case "$(basename "$WAN_CKPT")" in
    *14B*) VACE_TASK="vace-14B" ;;
    *) VACE_TASK="vace-1.3B" ;;
  esac
fi
if [[ -z "$ULYSSES_SIZE" ]]; then
  ULYSSES_SIZE="$GPU_COUNT"
fi

mkdir -p "$OUT_ROOT/videos" "$OUT_ROOT/logs" "$OUT_ROOT/pairs" "$OUT_ROOT/metadata"

SELECTED_PLAN="$OUT_ROOT/metadata/selected_video_edit_plan.json"
ENV_FILE="$OUT_ROOT/metadata/selected_video_edit_plan.env"

python3 - "$DATA_ROOT" "$VIDEO_EDIT_PLAN" "$PLAN_INDEX" "$PLAN_ID" "$SELECTED_PLAN" "$ENV_FILE" "$OUT_ROOT" "$WAN_CKPT" <<'PY'
import json
import os
import re
import shlex
import sys
from pathlib import Path

data_root = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
plan_index = int(sys.argv[3])
plan_id_filter = sys.argv[4].strip()
selected_plan_path = Path(sys.argv[5])
env_path = Path(sys.argv[6])
out_root = Path(sys.argv[7])
wan_ckpt = Path(sys.argv[8])

rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"empty video edit plan: {plan_path}")
if plan_id_filter:
    matches = [row for row in rows if str(row.get("plan_id", "")) == plan_id_filter]
    if not matches:
        raise SystemExit(f"plan_id not found: {plan_id_filter}")
    plan = matches[0]
else:
    if plan_index < 1 or plan_index > len(rows):
        raise SystemExit(f"plan-index {plan_index} out of range 1..{len(rows)}")
    plan = rows[plan_index - 1]

route = str(plan.get("model_route", "")).strip()
if route != "vace_controlled":
    raise SystemExit(f"selected plan route must be vace_controlled for this smoke, got {route!r}")

reference_raw = str(plan.get("reference_video", "")).strip()
if not reference_raw:
    raise SystemExit("selected plan is missing reference_video")
reference_path = Path(reference_raw)
if not reference_path.is_absolute():
    reference_path = data_root / reference_path
if not reference_path.exists():
    raise SystemExit(f"reference video does not exist: {reference_path}")

plan_id = str(plan.get("plan_id", "")).strip() or "visual_plan"
safe_plan_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", plan_id)[:80]
raw_video = out_root / "videos" / f"{safe_plan_id}_raw.mp4"
target_video = out_root / "videos" / f"{safe_plan_id}_with_ref_audio.mp4"

target_prompt = str(plan.get("target_prompt", "")).strip()
negative_prompt = str(plan.get("negative_prompt", "")).strip()
edit_region = str(plan.get("edit_region", "")).strip()
preserve_tokens = [str(item).strip() for item in plan.get("preserve_tokens", []) if str(item).strip()]
if not target_prompt:
    raise SystemExit("selected plan has empty target_prompt")
prompt = target_prompt
if edit_region:
    prompt += f" Edit only the {edit_region}."
if preserve_tokens:
    prompt += " Preserve: " + ", ".join(preserve_tokens[:8]) + "."
if negative_prompt:
    prompt += " Negative constraints: " + negative_prompt

selected_plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

known_pair = {
    "proposal_id": f"synthetic_visual_pair_{safe_plan_id}",
    "source_type": "synthetic_edit",
    "reference_video": str(reference_path),
    "target_clip_id": f"synthetic_visual_target_{safe_plan_id}",
    "target_video": str(target_video),
    "edit_text": str(plan.get("edit_text", "")).strip(),
    "modalities": ["visual"],
    "difference": plan.get("difference", {}),
    "hard_negatives": [],
    "quality": {"visual_near_duplicate_score": 0.90},
    "generation": {
        "model": wan_ckpt.name or str(wan_ckpt),
        "model_route": route,
        "source_video": str(reference_path),
        "prompt": prompt,
        "source_prompt": str(plan.get("source_prompt", "")).strip(),
        "target_prompt": target_prompt,
        "edit_token": str(plan.get("edit_token", "")).strip(),
        "preserve_tokens": preserve_tokens,
        "negative_prompt": negative_prompt,
        "edit_region": edit_region,
        "video_edit_plan_id": plan_id,
        "postprocess": {
            "audio_copied_from_reference": True,
            "raw_generated_video": str(raw_video),
        },
    },
    "source_context": {
        "relation": "synthetic_from_reference",
        "score": 0.95,
        "generation_source_video": str(reference_path),
    },
}
manifest = {
    "clip_id": known_pair["target_clip_id"],
    "output_path": str(target_video),
    "source_path": str(target_video),
    "role": "synthetic_visual_target",
    "notes": f"generated from {plan_id}",
}

(out_root / "pairs" / "synthetic_visual_candidate_pairs.jsonl").write_text(
    json.dumps(known_pair, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
(out_root / "metadata" / "synthetic_visual_target_manifest.jsonl").write_text(
    json.dumps(manifest, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

env_values = {
    "PLAN_ID": plan_id,
    "REFERENCE_VIDEO": str(reference_path),
    "RAW_VIDEO": str(raw_video),
    "TARGET_VIDEO": str(target_video),
    "PROMPT": prompt,
    "KNOWN_PAIRS": str(out_root / "pairs" / "synthetic_visual_candidate_pairs.jsonl"),
    "TARGET_MANIFEST": str(out_root / "metadata" / "synthetic_visual_target_manifest.jsonl"),
}
env_path.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in env_values.items()), encoding="utf-8")
PY

# shellcheck disable=SC1090
source "$ENV_FILE"

echo "[vace-smoke] start $(date)"
echo "[vace-smoke] plan_id=$PLAN_ID"
echo "[vace-smoke] reference=$REFERENCE_VIDEO"
echo "[vace-smoke] raw_video=$RAW_VIDEO"
echo "[vace-smoke] target_video=$TARGET_VIDEO"
echo "[vace-smoke] wan_code=$WAN_CODE"
echo "[vace-smoke] wan_ckpt=$WAN_CKPT"
echo "[vace-smoke] vace_task=$VACE_TASK"
echo "[vace-smoke] conda_env=$CONDA_ENV"
echo "[vace-smoke] gpu_ids=$GPU_IDS offload_model=$OFFLOAD_MODEL t5_cpu=$T5_CPU allow_cpu_offload=$ALLOW_CPU_OFFLOAD"
echo "[vace-smoke] ulysses_size=$ULYSSES_SIZE ring_size=$RING_SIZE"
echo "[vace-smoke] prompt=$PROMPT"

if [[ ! -d "$WAN_CODE" ]]; then
  echo "[vace-smoke] missing WAN_CODE=$WAN_CODE" >&2
  exit 1
fi
if [[ ! -d "$WAN_CKPT" ]]; then
  echo "[vace-smoke] missing WAN_CKPT=$WAN_CKPT" >&2
  exit 1
fi

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

python - <<'PY'
import importlib.util
import sys

import torch

print("[vace-smoke] python", sys.executable)
print("[vace-smoke] torch", torch.__version__, "cuda", torch.version.cuda)
print("[vace-smoke] flash_attn", "ok" if importlib.util.find_spec("flash_attn") else "missing")
if importlib.util.find_spec("flash_attn") is None:
    raise SystemExit(
        "flash_attn is missing in the active conda env. "
        "Use CONDA_ENV=wan_vace or install a CUDA/PyTorch-compatible flash-attn build."
    )
PY

GEN_ARGS=(
  "$WAN_CODE/generate.py"
  --task "$VACE_TASK"
  --size "$SIZE"
  --ckpt_dir "$WAN_CKPT"
  --src_video "$REFERENCE_VIDEO"
  --prompt "$PROMPT"
  --frame_num "$FRAME_NUM"
  --sample_steps "$SAMPLE_STEPS"
  --sample_guide_scale "$SAMPLE_GUIDE_SCALE"
  --offload_model "$OFFLOAD_MODEL"
  --save_file "$RAW_VIDEO"
)
if [[ "$T5_CPU" == "1" ]]; then
  GEN_ARGS+=(--t5_cpu)
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
if [[ "$USE_TORCHRUN" == "1" ]]; then
  echo "[vace-smoke] running with torchrun on $GPU_COUNT process(es)"
  DIST_ARGS=(--dit_fsdp --t5_fsdp)
  if [[ "$ULYSSES_SIZE" != "0" ]]; then
    DIST_ARGS+=(--ulysses_size "$ULYSSES_SIZE")
  fi
  if [[ "$RING_SIZE" != "0" ]]; then
    DIST_ARGS+=(--ring_size "$RING_SIZE")
  fi
  torchrun --nproc_per_node="$GPU_COUNT" "${GEN_ARGS[@]}" \
    "${DIST_ARGS[@]}" \
    > "$OUT_ROOT/logs/vace_generate.log" 2>&1
else
  echo "[vace-smoke] running single process with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  python "${GEN_ARGS[@]}" > "$OUT_ROOT/logs/vace_generate.log" 2>&1
fi

if [[ ! -s "$RAW_VIDEO" ]]; then
  echo "[vace-smoke] raw generation missing: $RAW_VIDEO" >&2
  tail -80 "$OUT_ROOT/logs/vace_generate.log" >&2 || true
  exit 1
fi

ffmpeg -y -i "$RAW_VIDEO" -i "$REFERENCE_VIDEO" \
  -map 0:v:0 -map 1:a? -c:v copy -c:a aac -shortest "$TARGET_VIDEO" \
  > "$OUT_ROOT/logs/remux_audio.log" 2>&1

ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$TARGET_VIDEO" || true
echo "[vace-smoke] known_pairs=$KNOWN_PAIRS"
echo "[vace-smoke] target_manifest=$TARGET_MANIFEST"
echo "[vace-smoke] done $(date)"
