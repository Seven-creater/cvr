#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/vace_visual_synthetic_smoke}
VIDEO_EDIT_PLAN=${VIDEO_EDIT_PLAN:-$RUN_ROOT/video_edit_plan.jsonl}
MASK_MANIFEST=${MASK_MANIFEST:-}
SRC_REF_SELECTION=${SRC_REF_SELECTION:-}
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
FRAME_NUM=${FRAME_NUM:-81}
VACE_CLIP_SECONDS=${VACE_CLIP_SECONDS:-5}
VACE_SOURCE_FPS=${VACE_SOURCE_FPS:-16}
VACE_INPUT_DURATION_DRIFT_MAX=${VACE_INPUT_DURATION_DRIFT_MAX:-0.15}
VACE_DURATION_DRIFT_MAX=${VACE_DURATION_DRIFT_MAX:-0.5}
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
  --mask-manifest PATH
  --src-ref-selection PATH
  --plan-index N
  --plan-id ID
  --wan-root PATH
  --wan-code PATH
  --wan-ckpt PATH
  --vace-task vace-1.3B|vace-14B
  --conda-env NAME
  --gpu-ids IDS
  --max-gpus N
  --use-torchrun 0|1
  --ulysses-size N
  --ring-size N
  --frame-num N
  --vace-clip-seconds N
  --vace-source-fps N
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
    --mask-manifest) MASK_MANIFEST="$2"; shift 2 ;;
    --src-ref-selection) SRC_REF_SELECTION="$2"; shift 2 ;;
    --plan-index) PLAN_INDEX="$2"; shift 2 ;;
    --plan-id) PLAN_ID="$2"; shift 2 ;;
    --wan-root) WAN_ROOT="$2"; WAN_CODE="$2/code"; WAN_CKPT="$2/Wan2.1-VACE-1.3B"; shift 2 ;;
    --wan-code) WAN_CODE="$2"; shift 2 ;;
    --wan-ckpt) WAN_CKPT="$2"; shift 2 ;;
    --vace-task) VACE_TASK="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --use-torchrun) USE_TORCHRUN="$2"; shift 2 ;;
    --ulysses-size) ULYSSES_SIZE="$2"; shift 2 ;;
    --ring-size) RING_SIZE="$2"; shift 2 ;;
    --frame-num) FRAME_NUM="$2"; shift 2 ;;
    --vace-clip-seconds) VACE_CLIP_SECONDS="$2"; shift 2 ;;
    --vace-source-fps) VACE_SOURCE_FPS="$2"; shift 2 ;;
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

python3 - "$DATA_ROOT" "$VIDEO_EDIT_PLAN" "$MASK_MANIFEST" "$SRC_REF_SELECTION" "$PLAN_INDEX" "$PLAN_ID" "$SELECTED_PLAN" "$ENV_FILE" "$OUT_ROOT" "$WAN_CKPT" <<'PY'
import json
import os
import re
import shlex
import sys
from pathlib import Path

data_root = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
mask_manifest_path = Path(sys.argv[3]) if sys.argv[3].strip() else None
src_ref_selection_path = Path(sys.argv[4]) if sys.argv[4].strip() else None
plan_index = int(sys.argv[5])
plan_id_filter = sys.argv[6].strip()
selected_plan_path = Path(sys.argv[7])
env_path = Path(sys.argv[8])
out_root = Path(sys.argv[9])
wan_ckpt = Path(sys.argv[10])

def resolve_existing_path(raw_path, base_dirs):
    raw = str(raw_path).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [path]
    for base_dir in base_dirs:
        candidates.append(Path(base_dir) / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

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
reference_video_original = reference_path
reference_video_for_vace = out_root / "videos" / f"{safe_plan_id}_reference_for_vace.mp4"
raw_video = out_root / "videos" / f"{safe_plan_id}_raw.mp4"
target_video = out_root / "videos" / f"{safe_plan_id}_with_ref_audio.mp4"
src_video_for_vace = out_root / "videos" / f"{safe_plan_id}_src_video_for_vace.mp4"
src_mask = ""
src_mask_original = ""
src_mask_for_vace = out_root / "videos" / f"{safe_plan_id}_mask_for_vace.mp4"
mask_metrics = {}
if mask_manifest_path:
    mask_rows = [json.loads(line) for line in mask_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mask_matches = [row for row in mask_rows if str(row.get("plan_id", "")) == plan_id]
    if not mask_matches:
        raise SystemExit(f"mask manifest has no row for plan_id: {plan_id}")
    mask_status = str(mask_matches[0].get("status", "")).strip()
    if mask_status and mask_status != "generated":
        raise SystemExit(f"mask manifest row is not generated for plan_id {plan_id}: {mask_status}")
    mask_raw = str(mask_matches[0].get("mask_video", "")).strip()
    if not mask_raw:
        raise SystemExit(f"mask manifest row is missing mask_video for plan_id: {plan_id}")
    mask_path = Path(mask_raw)
    if not mask_path.is_absolute():
        mask_path = mask_manifest_path.parent / mask_path
    if not mask_path.exists():
        raise SystemExit(f"mask video does not exist for plan_id {plan_id}: {mask_path}")
    src_mask_original = str(mask_path)
    src_mask = str(src_mask_for_vace)
    mask_metrics = mask_matches[0].get("mask_metrics", {}) if isinstance(mask_matches[0].get("mask_metrics"), dict) else {}

src_ref_images = []
if src_ref_selection_path:
    selection_rows = [json.loads(line) for line in src_ref_selection_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selection_matches = [row for row in selection_rows if str(row.get("plan_id", "")) == plan_id]
    if selection_matches:
        selection_status = str(selection_matches[0].get("status", "")).strip()
        if (plan.get("src_ref_requirements") or {}).get("required") and selection_status != "selected":
            raise SystemExit(f"required src_ref selection is not selected for plan_id {plan_id}: {selection_status}")
        for raw_image in selection_matches[0].get("selected_src_ref_images", []):
            image_path = resolve_existing_path(raw_image, [src_ref_selection_path.parent, data_root])
            if image_path is None or not image_path.exists():
                raise SystemExit(f"selected src_ref_image does not exist for plan_id {plan_id}: {image_path}")
            src_ref_images.append(str(image_path))
    elif (plan.get("src_ref_requirements") or {}).get("required"):
        raise SystemExit(f"src_ref selection has no row for required plan_id: {plan_id}")
else:
    for raw_image in ((plan.get("vace_inputs") or {}).get("src_ref_images") or []):
        image_path = resolve_existing_path(raw_image, [data_root, plan_path.parent])
        if image_path is not None and image_path.exists():
            src_ref_images.append(str(image_path))
if (plan.get("src_ref_requirements") or {}).get("required") and not src_ref_images:
    raise SystemExit(f"plan_id {plan_id} requires src_ref_images but none were selected")

target_prompt = str(plan.get("target_prompt", "")).strip()
negative_prompt = str(plan.get("negative_prompt", "")).strip()
edit_region = str(plan.get("edit_region", "")).strip()
preserve_tokens = [str(item).strip() for item in plan.get("preserve_tokens", []) if str(item).strip()]
if not target_prompt:
    raise SystemExit("selected plan has empty target_prompt")
prompt = target_prompt

selected_plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

known_pair = {
    "proposal_id": f"synthetic_visual_pair_{safe_plan_id}",
    "source_type": "synthetic_edit",
    "reference_video": str(reference_video_for_vace),
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
        "source_video": str(reference_video_for_vace),
        "original_reference_video": str(reference_video_original),
        "src_video_for_vace": str(src_video_for_vace if src_mask else reference_video_for_vace),
        "prompt": prompt,
        "source_prompt": str(plan.get("source_prompt", "")).strip(),
        "target_prompt": target_prompt,
        "edit_token": str(plan.get("edit_token", "")).strip(),
        "preserve_tokens": preserve_tokens,
        "negative_prompt": negative_prompt,
        "edit_region": edit_region,
        "mask_query": str(plan.get("mask_query", "")).strip(),
        "src_mask": src_mask,
        "original_src_mask": src_mask_original,
        "src_ref_images": src_ref_images,
        "src_ref_requirements": plan.get("src_ref_requirements", {}),
        "mask_metrics": mask_metrics,
        "video_edit_plan_id": plan_id,
        "review_inputs_dir": str(out_root / "review_inputs"),
        "postprocess": {
            "audio_copied_from_reference": True,
            "raw_generated_video": str(raw_video),
        },
    },
    "source_context": {
        "relation": "synthetic_from_reference",
        "score": 0.95,
        "generation_source_video": str(reference_video_for_vace),
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
    "REFERENCE_VIDEO_ORIGINAL": str(reference_video_original),
    "REFERENCE_VIDEO": str(reference_video_for_vace),
    "SRC_VIDEO_FOR_VACE": str(src_video_for_vace if src_mask else reference_video_for_vace),
    "RAW_VIDEO": str(raw_video),
    "TARGET_VIDEO": str(target_video),
    "SRC_MASK": src_mask,
    "SRC_MASK_ORIGINAL": src_mask_original,
    "SRC_REF_IMAGES": ",".join(src_ref_images),
    "SRC_REF_REQUIRED": "1" if (plan.get("src_ref_requirements") or {}).get("required") else "0",
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
echo "[vace-smoke] original_reference=$REFERENCE_VIDEO_ORIGINAL"
echo "[vace-smoke] src_video_for_vace=$SRC_VIDEO_FOR_VACE"
echo "[vace-smoke] raw_video=$RAW_VIDEO"
echo "[vace-smoke] target_video=$TARGET_VIDEO"
echo "[vace-smoke] src_mask=${SRC_MASK:-none}"
echo "[vace-smoke] src_ref_images=${SRC_REF_IMAGES:-none}"
echo "[vace-smoke] wan_code=$WAN_CODE"
echo "[vace-smoke] wan_ckpt=$WAN_CKPT"
echo "[vace-smoke] vace_task=$VACE_TASK"
echo "[vace-smoke] conda_env=$CONDA_ENV"
echo "[vace-smoke] gpu_ids=$GPU_IDS offload_model=$OFFLOAD_MODEL t5_cpu=$T5_CPU allow_cpu_offload=$ALLOW_CPU_OFFLOAD"
echo "[vace-smoke] ulysses_size=$ULYSSES_SIZE ring_size=$RING_SIZE"
echo "[vace-smoke] frame_num=$FRAME_NUM vace_clip_seconds=$VACE_CLIP_SECONDS vace_source_fps=$VACE_SOURCE_FPS"
echo "[vace-smoke] prompt=$PROMPT"

mkdir -p "$OUT_ROOT/review_inputs/src_ref_images"

if ! python3 - "$FRAME_NUM" <<'PY'; then
import sys
frame_num = int(sys.argv[1])
if frame_num <= 0 or (frame_num - 1) % 4 != 0:
    raise SystemExit(f"FRAME_NUM must be positive and 4n+1 for Wan/VACE, got {frame_num}")
PY
  exit 1
fi

if ! python3 - "$VACE_SOURCE_FPS" <<'PY'; then
import sys
fps = float(sys.argv[1])
if fps <= 0:
    raise SystemExit(f"VACE_SOURCE_FPS must be positive, got {fps}")
PY
  exit 1
fi

VACE_SOURCE_DURATION=$(python3 - "$FRAME_NUM" "$VACE_SOURCE_FPS" <<'PY'
import sys
frame_num = int(sys.argv[1])
fps = float(sys.argv[2])
print(f"{frame_num / fps:.6f}")
PY
)
echo "[vace-smoke] vace_source_duration=$VACE_SOURCE_DURATION exact_frames=${FRAME_NUM}@${VACE_SOURCE_FPS}fps"

ffmpeg -y -i "$REFERENCE_VIDEO_ORIGINAL" \
  -filter_complex "[0:v]fps=${VACE_SOURCE_FPS},tpad=stop_mode=clone:stop_duration=1,trim=start_frame=0:end_frame=${FRAME_NUM},setpts=N/${VACE_SOURCE_FPS}/TB[v]" \
  -map "[v]" -map 0:a? -t "$VACE_SOURCE_DURATION" \
  -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p -c:a aac -movflags +faststart "$REFERENCE_VIDEO" \
  > "$OUT_ROOT/logs/reference_for_vace.log" 2>&1 || {
    echo "[vace-smoke] failed to create clipped reference=$REFERENCE_VIDEO" >&2
    tail -80 "$OUT_ROOT/logs/reference_for_vace.log" >&2 || true
    exit 1
  }
if [[ -n "${SRC_MASK_ORIGINAL:-}" ]]; then
  WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of default=nw=1:nk=1 "$REFERENCE_VIDEO" | head -1)
  HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 "$REFERENCE_VIDEO" | head -1)
  ffmpeg -y -i "$SRC_MASK_ORIGINAL" \
    -filter_complex "[0:v]fps=${VACE_SOURCE_FPS},scale=${WIDTH}:${HEIGHT},format=gray,tpad=stop_mode=clone:stop_duration=1,trim=start_frame=0:end_frame=${FRAME_NUM},setpts=N/${VACE_SOURCE_FPS}/TB[v]" \
    -map "[v]" -an -c:v libx264 -crf 0 -preset veryfast -pix_fmt yuv420p "$SRC_MASK" \
    > "$OUT_ROOT/logs/src_mask_for_vace.log" 2>&1 || {
      echo "[vace-smoke] failed to create clipped src_mask=$SRC_MASK" >&2
      tail -80 "$OUT_ROOT/logs/src_mask_for_vace.log" >&2 || true
      exit 1
    }
fi

{
  echo "plan_id=$PLAN_ID"
  echo "reference=$REFERENCE_VIDEO"
  echo "original_reference=$REFERENCE_VIDEO_ORIGINAL"
  echo "src_video_for_vace=$SRC_VIDEO_FOR_VACE"
  echo "src_mask=${SRC_MASK:-none}"
  echo "src_ref_images=${SRC_REF_IMAGES:-none}"
  echo
  echo "$PROMPT"
} > "$OUT_ROOT/review_inputs/vace_prompt.txt"
ffmpeg -y -i "$REFERENCE_VIDEO" -vf "fps=1,scale=240:-1,tile=5x1" -frames:v 1 \
  "$OUT_ROOT/review_inputs/reference_contact.jpg" > "$OUT_ROOT/logs/contact_reference.log" 2>&1 || true
if [[ -n "${SRC_MASK:-}" ]]; then
  ffmpeg -y -i "$SRC_MASK" -vf "fps=1,scale=240:-1,tile=5x1" -frames:v 1 \
    "$OUT_ROOT/review_inputs/mask_contact.jpg" > "$OUT_ROOT/logs/contact_mask.log" 2>&1 || true
  WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of default=nw=1:nk=1 "$REFERENCE_VIDEO" | head -1)
  HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 "$REFERENCE_VIDEO" | head -1)
  ffmpeg -y -i "$REFERENCE_VIDEO" -i "$SRC_MASK" \
    -filter_complex "[1:v]scale=${WIDTH}:${HEIGHT},format=gray[m];color=c=gray:s=${WIDTH}x${HEIGHT}:r=${VACE_SOURCE_FPS}:d=${VACE_SOURCE_DURATION},format=yuv420p[gray];[0:v]format=yuv420p[base];[base][gray][m]maskedmerge[out]" \
    -map "[out]" -frames:v "$FRAME_NUM" -an -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p "$SRC_VIDEO_FOR_VACE" \
    > "$OUT_ROOT/logs/src_video_for_vace.log" 2>&1 || {
      echo "[vace-smoke] failed to create src_video_for_vace=$SRC_VIDEO_FOR_VACE" >&2
      tail -80 "$OUT_ROOT/logs/src_video_for_vace.log" >&2 || true
      exit 1
    }
fi
if [[ -n "${SRC_REF_IMAGES:-}" ]]; then
  IFS=',' read -r -a SRC_REF_IMAGE_ARRAY <<< "$SRC_REF_IMAGES"
  idx=0
  for image_path in "${SRC_REF_IMAGE_ARRAY[@]}"; do
    idx=$((idx + 1))
    cp "$image_path" "$OUT_ROOT/review_inputs/src_ref_images/$(printf '%03d' "$idx")_$(basename "$image_path")" || true
  done
fi

python3 - "$OUT_ROOT" "$REFERENCE_VIDEO" "$SRC_VIDEO_FOR_VACE" "${SRC_MASK:-}" "$SRC_REF_IMAGES" "$SRC_REF_REQUIRED" "$VACE_INPUT_DURATION_DRIFT_MAX" "$FRAME_NUM" "$VACE_SOURCE_FPS" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
reference_video = Path(sys.argv[2])
src_video = Path(sys.argv[3])
src_mask = Path(sys.argv[4]) if sys.argv[4].strip() else None
src_ref_images = [Path(item) for item in sys.argv[5].split(",") if item.strip()]
src_ref_required = sys.argv[6] == "1"
max_drift = float(sys.argv[7])
expected_frame_num = int(sys.argv[8])
expected_fps = float(sys.argv[9])

def parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_float = float(denominator)
            return float(numerator) / denominator_float if denominator_float else 0.0
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def probe(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"preflight missing required VACE input: {path}")
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout or "{}")
    video_stream = next((row for row in payload.get("streams", []) if row.get("codec_type") == "video"), {})
    raw_frame_count = video_stream.get("nb_read_frames") or video_stream.get("nb_frames") or 0
    try:
        frame_count = int(raw_frame_count)
    except (TypeError, ValueError):
        frame_count = 0
    return {
        "path": str(path),
        "duration_seconds": float((payload.get("format") or {}).get("duration") or video_stream.get("duration") or 0.0),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": parse_fraction(str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "")),
        "frame_count": frame_count,
        "has_video": bool(video_stream),
    }

reference = probe(reference_video)
src = probe(src_video)
if not reference["has_video"] or not src["has_video"]:
    raise SystemExit("preflight failed: reference/src_video must both contain video streams")
if (reference["width"], reference["height"]) != (src["width"], src["height"]):
    raise SystemExit(f"preflight failed: reference/src_video size mismatch {reference} vs {src}")
if abs(reference["duration_seconds"] - src["duration_seconds"]) > max_drift:
    raise SystemExit(f"preflight failed: reference/src_video duration mismatch {reference} vs {src}")
for label, media in (("reference", reference), ("src_video", src)):
    if media["frame_count"] != expected_frame_num:
        raise SystemExit(f"preflight failed: {label} frame_count {media['frame_count']} != expected {expected_frame_num}: {media}")
    if abs(media["fps"] - expected_fps) > 0.01:
        raise SystemExit(f"preflight failed: {label} fps {media['fps']:.4f} != expected {expected_fps:.4f}: {media}")
mask = None
if src_mask is not None:
    mask = probe(src_mask)
    if (reference["width"], reference["height"]) != (mask["width"], mask["height"]):
        raise SystemExit(f"preflight failed: reference/src_mask size mismatch {reference} vs {mask}")
    if abs(reference["duration_seconds"] - mask["duration_seconds"]) > max_drift:
        raise SystemExit(f"preflight failed: reference/src_mask duration mismatch {reference} vs {mask}")
    if mask["frame_count"] != expected_frame_num:
        raise SystemExit(f"preflight failed: src_mask frame_count {mask['frame_count']} != expected {expected_frame_num}: {mask}")
    if abs(mask["fps"] - expected_fps) > 0.01:
        raise SystemExit(f"preflight failed: src_mask fps {mask['fps']:.4f} != expected {expected_fps:.4f}: {mask}")
if src_ref_required and not src_ref_images:
    raise SystemExit("preflight failed: required src_ref_images are missing")
for image in src_ref_images:
    if not image.exists() or image.stat().st_size <= 0:
        raise SystemExit(f"preflight failed: selected src_ref_image missing: {image}")
copied_refs = sorted((out_root / "review_inputs" / "src_ref_images").glob("*"))
if src_ref_required and not copied_refs:
    raise SystemExit("preflight failed: required src_ref_images were not copied into review_inputs")
report = {
    "reference": reference,
    "src_video_for_vace": src,
    "src_mask": mask,
    "src_ref_image_count": len(src_ref_images),
    "copied_src_ref_image_count": len(copied_refs),
    "max_input_duration_drift_seconds": max_drift,
    "expected_frame_num": expected_frame_num,
    "expected_fps": expected_fps,
    "passed": True,
}
(out_root / "metadata" / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
PY

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
  --src_video "$SRC_VIDEO_FOR_VACE"
  --prompt "$PROMPT"
  --frame_num "$FRAME_NUM"
  --sample_steps "$SAMPLE_STEPS"
  --sample_guide_scale "$SAMPLE_GUIDE_SCALE"
  --offload_model "$OFFLOAD_MODEL"
  --save_file "$RAW_VIDEO"
)
if [[ -n "${SRC_MASK:-}" ]]; then
  GEN_ARGS+=(--src_mask "$SRC_MASK")
fi
if [[ -n "${SRC_REF_IMAGES:-}" ]]; then
  GEN_ARGS+=(--src_ref_images "$SRC_REF_IMAGES")
fi
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
  RUN_COMMAND=(torchrun --nproc_per_node="$GPU_COUNT" "${GEN_ARGS[@]}" "${DIST_ARGS[@]}")
else
  echo "[vace-smoke] running single process with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  RUN_COMMAND=(python "${GEN_ARGS[@]}")
fi

python3 - "$OUT_ROOT/metadata/vace_command.json" "$CUDA_VISIBLE_DEVICES" "${RUN_COMMAND[@]}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
out_path.write_text(
    json.dumps(
        {
            "cuda_visible_devices": sys.argv[2],
            "argv": sys.argv[3:],
            "shell_quoted": " ".join(shlex.quote(item) for item in sys.argv[3:]),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PY

"${RUN_COMMAND[@]}" > "$OUT_ROOT/logs/vace_generate.log" 2>&1

if [[ ! -s "$RAW_VIDEO" ]]; then
  echo "[vace-smoke] raw generation missing: $RAW_VIDEO" >&2
  tail -80 "$OUT_ROOT/logs/vace_generate.log" >&2 || true
  exit 1
fi

python3 - "$REFERENCE_VIDEO" "$RAW_VIDEO" "$VACE_DURATION_DRIFT_MAX" "$FRAME_NUM" "$VACE_SOURCE_FPS" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

reference_video = Path(sys.argv[1])
raw_video = Path(sys.argv[2])
max_drift = float(sys.argv[3])
expected_frame_num = int(sys.argv[4])
expected_fps = float(sys.argv[5])

def parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_float = float(denominator)
            return float(numerator) / denominator_float if denominator_float else 0.0
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def probe(path: Path) -> dict:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout or "{}")
    video_stream = next((row for row in payload.get("streams", []) if row.get("codec_type") == "video"), {})
    raw_frame_count = video_stream.get("nb_read_frames") or video_stream.get("nb_frames") or 0
    try:
        frame_count = int(raw_frame_count)
    except (TypeError, ValueError):
        frame_count = 0
    return {
        "path": str(path),
        "duration_seconds": float((payload.get("format") or {}).get("duration") or video_stream.get("duration") or 0.0),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": parse_fraction(str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "")),
        "frame_count": frame_count,
        "has_video": bool(video_stream),
    }

reference = probe(reference_video)
raw = probe(raw_video)
drift = abs(reference["duration_seconds"] - raw["duration_seconds"])
errors = []
if drift > max_drift:
    errors.append(
        f"raw VACE target duration drift {drift:.3f}s exceeds {max_drift:.3f}s: "
        f"reference={reference['duration_seconds']:.3f}s raw={raw['duration_seconds']:.3f}s"
    )
if raw["frame_count"] != expected_frame_num:
    errors.append(f"raw VACE target frame_count {raw['frame_count']} != expected {expected_frame_num}")
if abs(raw["fps"] - expected_fps) > 0.01:
    errors.append(f"raw VACE target fps {raw['fps']:.4f} != expected {expected_fps:.4f}")
if errors:
    raise SystemExit("; ".join(errors))
PY

ffmpeg -y -i "$RAW_VIDEO" -i "$REFERENCE_VIDEO" \
  -map 0:v:0 -map 1:a? -c:v copy -c:a aac "$TARGET_VIDEO" \
  > "$OUT_ROOT/logs/remux_audio.log" 2>&1

python3 - "$OUT_ROOT" "$REFERENCE_VIDEO" "$RAW_VIDEO" "$TARGET_VIDEO" "$VACE_DURATION_DRIFT_MAX" "$KNOWN_PAIRS" "$FRAME_NUM" "$VACE_SOURCE_FPS" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
reference_video = Path(sys.argv[2])
raw_video = Path(sys.argv[3])
target_video = Path(sys.argv[4])
max_drift = float(sys.argv[5])
known_pairs_path = Path(sys.argv[6])
expected_frame_num = int(sys.argv[7])
expected_fps = float(sys.argv[8])

def parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_float = float(denominator)
            return float(numerator) / denominator_float if denominator_float else 0.0
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video_stream = next((row for row in streams if row.get("codec_type") == "video"), {})
    audio_stream = next((row for row in streams if row.get("codec_type") == "audio"), {})
    raw_frame_count = video_stream.get("nb_read_frames") or video_stream.get("nb_frames") or 0
    try:
        frame_count = int(raw_frame_count)
    except (TypeError, ValueError):
        frame_count = 0
    return {
        "path": str(path),
        "duration_seconds": float((payload.get("format") or {}).get("duration") or video_stream.get("duration") or 0.0),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": parse_fraction(str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "")),
        "frame_count": frame_count,
        "has_video": bool(video_stream),
        "has_audio": bool(audio_stream),
    }

reference = probe(reference_video)
raw = probe(raw_video)
target = probe(target_video)
raw_drift = abs(reference["duration_seconds"] - raw["duration_seconds"])
target_drift = abs(reference["duration_seconds"] - target["duration_seconds"])
passed = (
    raw_drift <= max_drift
    and target_drift <= max_drift
    and raw["frame_count"] == expected_frame_num
    and target["frame_count"] == expected_frame_num
    and abs(raw["fps"] - expected_fps) <= 0.01
    and abs(target["fps"] - expected_fps) <= 0.01
    and target["has_video"]
)
metrics = {
    "reference": reference,
    "raw_generated_video": raw,
    "audio_remux_target": target,
    "raw_duration_drift_seconds": round(raw_drift, 3),
    "target_duration_drift_seconds": round(target_drift, 3),
    "max_duration_drift_seconds": max_drift,
    "expected_frame_num": expected_frame_num,
    "expected_fps": expected_fps,
    "duration_gate": {"passed": passed, "errors": []},
}
if raw_drift > max_drift:
    metrics["duration_gate"]["errors"].append(f"raw_duration_drift_seconds {raw_drift:.3f} > {max_drift:.3f}")
if target_drift > max_drift:
    metrics["duration_gate"]["errors"].append(f"target_duration_drift_seconds {target_drift:.3f} > {max_drift:.3f}")
if not target["has_video"]:
    metrics["duration_gate"]["errors"].append("audio-remux target has no video stream")
if raw["frame_count"] != expected_frame_num:
    metrics["duration_gate"]["errors"].append(f"raw_frame_count {raw['frame_count']} != {expected_frame_num}")
if target["frame_count"] != expected_frame_num:
    metrics["duration_gate"]["errors"].append(f"target_frame_count {target['frame_count']} != {expected_frame_num}")
if abs(raw["fps"] - expected_fps) > 0.01:
    metrics["duration_gate"]["errors"].append(f"raw_fps {raw['fps']:.4f} != {expected_fps:.4f}")
if abs(target["fps"] - expected_fps) > 0.01:
    metrics["duration_gate"]["errors"].append(f"target_fps {target['fps']:.4f} != {expected_fps:.4f}")
if target["has_video"] and (reference["width"], reference["height"]) != (target["width"], target["height"]):
    metrics["target_resize_note"] = (
        "target resolution differs from reference; Wan/VACE may resize internally via --size, "
        "but duration/audio alignment still must pass"
    )
duration_path = out_root / "metadata" / "duration_metrics.json"
duration_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

command_path = out_root / "metadata" / "vace_command.json"
command = json.loads(command_path.read_text(encoding="utf-8")) if command_path.exists() else {}
pairs = [json.loads(line) for line in known_pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
for pair in pairs:
    generation = pair.setdefault("generation", {})
    generation["duration_metrics"] = metrics
    generation["vace_command"] = command
    generation["post_vace_verdict"] = {
        "stage": "post_vace_pre_omni_validation",
        "duration_gate_passed": passed,
        "requires_omni_validation": True,
    }
known_pairs_path.write_text(
    "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs),
    encoding="utf-8",
)
if not passed:
    raise SystemExit("post-VACE duration gate failed: " + "; ".join(metrics["duration_gate"]["errors"]))
PY

cp "$OUT_ROOT/metadata/preflight_report.json" "$OUT_ROOT/review_inputs/preflight_report.json" 2>/dev/null || true
cp "$OUT_ROOT/metadata/duration_metrics.json" "$OUT_ROOT/review_inputs/duration_metrics.json" 2>/dev/null || true
cp "$OUT_ROOT/metadata/vace_command.json" "$OUT_ROOT/review_inputs/vace_command.json" 2>/dev/null || true
tail -120 "$OUT_ROOT/logs/vace_generate.log" > "$OUT_ROOT/review_inputs/vace_generate_tail.log" 2>/dev/null || true
tail -80 "$OUT_ROOT/logs/remux_audio.log" > "$OUT_ROOT/review_inputs/remux_audio_tail.log" 2>/dev/null || true

ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$TARGET_VIDEO" || true
echo "[vace-smoke] known_pairs=$KNOWN_PAIRS"
echo "[vace-smoke] target_manifest=$TARGET_MANIFEST"
echo "[vace-smoke] done $(date)"
