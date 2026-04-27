#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/vace_visual_batch}
VIDEO_EDIT_PLAN=${VIDEO_EDIT_PLAN:-$RUN_ROOT/video_edit_plan.jsonl}
MASK_MANIFEST=${MASK_MANIFEST:-}
PLAN_IDS=${PLAN_IDS:-}
TOP_K=${TOP_K:-3}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MAX_GPUS=${MAX_GPUS:-4}
WAN_CKPT=${WAN_CKPT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/Wan2.1/Wan2.1-VACE-14B}
VACE_TASK=${VACE_TASK:-vace-14B}
CONDA_ENV=${CONDA_ENV:-wan_vace}
USE_TORCHRUN=${USE_TORCHRUN:-1}
ULYSSES_SIZE=${ULYSSES_SIZE:-4}
RING_SIZE=${RING_SIZE:-0}
FAIL_FAST=${FAIL_FAST:-0}

usage() {
  cat <<'EOF'
Usage: run_vace_visual_batch_from_plan.sh [options]

Options:
  --data-root PATH
  --run-root PATH
  --video-edit-plan PATH
  --mask-manifest PATH
  --plan-ids ID1,ID2
  --top-k N
  --gpu-ids IDS
  --max-gpus N
  --wan-ckpt PATH
  --vace-task vace-14B|vace-1.3B
  --conda-env NAME
  --use-torchrun 0|1
  --ulysses-size N
  --ring-size N
  --fail-fast 0|1
  -h, --help

Batch VACE visual generation from video_edit_plan.jsonl.
This script does not start Omni. Run it only during the VACE phase after
the Omni planning stage has finished and Omni has been stopped.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --video-edit-plan) VIDEO_EDIT_PLAN="$2"; shift 2 ;;
    --mask-manifest) MASK_MANIFEST="$2"; shift 2 ;;
    --plan-ids) PLAN_IDS="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --wan-ckpt) WAN_CKPT="$2"; shift 2 ;;
    --vace-task) VACE_TASK="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --use-torchrun) USE_TORCHRUN="$2"; shift 2 ;;
    --ulysses-size) ULYSSES_SIZE="$2"; shift 2 ;;
    --ring-size) RING_SIZE="$2"; shift 2 ;;
    --fail-fast) FAIL_FAST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[vace-batch] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/items" "$RUN_ROOT/pairs" "$RUN_ROOT/metadata"

SELECTED_IDS="$RUN_ROOT/metadata/selected_plan_ids.txt"
BATCH_PAIRS="$RUN_ROOT/synthetic_visual_candidate_pairs.jsonl"
BATCH_MANIFEST="$RUN_ROOT/synthetic_visual_target_manifest.jsonl"
BATCH_REPORT="$RUN_ROOT/batch_generation_report.md"
: > "$BATCH_PAIRS"
: > "$BATCH_MANIFEST"

python3 - "$VIDEO_EDIT_PLAN" "$PLAN_IDS" "$TOP_K" "$SELECTED_IDS" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
plan_ids = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
top_k = int(sys.argv[3])
out_path = Path(sys.argv[4])
rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if plan_ids:
    wanted = set(plan_ids)
    selected = [row for row in rows if str(row.get("plan_id", "")) in wanted]
else:
    priority_rank = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4, "": 9}
    def key(row):
        suitability = row.get("route_suitability") if isinstance(row.get("route_suitability"), dict) else {}
        risk = row.get("visual_edit_risk") if isinstance(row.get("visual_edit_risk"), dict) else {}
        return (
            0 if row.get("model_route") == "vace_controlled" else 1,
            priority_rank.get(str(suitability.get("priority", "")), 9),
            float(risk.get("score", 1.0) or 1.0),
            1 if (row.get("planner") or {}).get("fallback_used") else 0,
        )
    selected = sorted((row for row in rows if row.get("model_route") == "vace_controlled"), key=key)[:top_k]
if not selected:
    raise SystemExit("no vace_controlled plans selected")
out_path.write_text("\n".join(str(row.get("plan_id", "")) for row in selected) + "\n", encoding="utf-8")
PY

{
  echo "# VACE Visual Batch Report"
  echo
  echo "- run_root: \`$RUN_ROOT\`"
  echo "- video_edit_plan: \`$VIDEO_EDIT_PLAN\`"
  echo "- mask_manifest: \`${MASK_MANIFEST:-none}\`"
  echo "- gpu_ids: \`$GPU_IDS\`"
  echo "- wan_ckpt: \`$WAN_CKPT\`"
  echo "- started: \`$(date)\`"
  echo
  echo "## Items"
  echo
} > "$BATCH_REPORT"

echo "[vace-batch] start $(date)"
echo "[vace-batch] selected ids:"
cat "$SELECTED_IDS"

mapfile -t SELECTED_PLAN_IDS < "$SELECTED_IDS"
for PLAN_ID in "${SELECTED_PLAN_IDS[@]}"; do
  [[ -z "$PLAN_ID" ]] && continue
  SAFE_PLAN_ID=$(python3 - "$PLAN_ID" <<'PY'
import re, sys
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", sys.argv[1])[:80])
PY
)
  ITEM_ROOT="$RUN_ROOT/items/$SAFE_PLAN_ID"
  echo "[vace-batch] generate plan_id=$PLAN_ID item_root=$ITEM_ROOT"
  MASK_ARGS=()
  if [[ -n "$MASK_MANIFEST" ]]; then
    MASK_ARGS=(--mask-manifest "$MASK_MANIFEST")
  fi
  if scripts/run_vace_visual_synthetic_smoke.sh \
      --data-root "$DATA_ROOT" \
      --run-root "$RUN_ROOT" \
      --video-edit-plan "$VIDEO_EDIT_PLAN" \
      "${MASK_ARGS[@]}" \
      --plan-id "$PLAN_ID" \
      --out-root "$ITEM_ROOT" \
      --wan-ckpt "$WAN_CKPT" \
      --vace-task "$VACE_TASK" \
      --conda-env "$CONDA_ENV" \
      --gpu-ids "$GPU_IDS" \
      --max-gpus "$MAX_GPUS" \
      --use-torchrun "$USE_TORCHRUN" \
      --ulysses-size "$ULYSSES_SIZE" \
      --ring-size "$RING_SIZE" \
      < /dev/null; then
    cat "$ITEM_ROOT/pairs/synthetic_visual_candidate_pairs.jsonl" >> "$BATCH_PAIRS"
    cat "$ITEM_ROOT/metadata/synthetic_visual_target_manifest.jsonl" >> "$BATCH_MANIFEST"
    echo "- PASS \`$PLAN_ID\`: \`$ITEM_ROOT\`" >> "$BATCH_REPORT"
  else
    echo "- FAIL \`$PLAN_ID\`: \`$ITEM_ROOT\`" >> "$BATCH_REPORT"
    if [[ "$FAIL_FAST" == "1" ]]; then
      exit 1
    fi
  fi
done

{
  echo
  echo "- finished: \`$(date)\`"
  echo "- known_pairs: \`$BATCH_PAIRS\`"
  echo "- target_manifest: \`$BATCH_MANIFEST\`"
} >> "$BATCH_REPORT"

echo "[vace-batch] known_pairs=$BATCH_PAIRS"
echo "[vace-batch] target_manifest=$BATCH_MANIFEST"
echo "[vace-batch] report=$BATCH_REPORT"
echo "[vace-batch] done $(date)"
