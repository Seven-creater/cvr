#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUNS_ROOT=${RUNS_ROOT:-/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs}
RUN_ROOT=${RUN_ROOT:-$RUNS_ROOT/e5_audio_matters_eval_$(date +%Y%m%d_%H%M%S)}
AUDIO_MATTERS_TRIPLETS=${AUDIO_MATTERS_TRIPLETS:-}
FULL_GALLERY_TRIPLETS=${FULL_GALLERY_TRIPLETS:-}
GPU_ID=${GPU_ID:-4}
SMOKE_SIZE=${SMOKE_SIZE:-20}
TOPK=${TOPK:-1,5,10}
TOPK_TRACE=${TOPK_TRACE:-10}
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
VIDEO_FPS=${VIDEO_FPS:-1}
E5_MODEL=${E5_MODEL:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B}

usage() {
  cat <<'EOF'
Usage: run_e5_audio_matters_eval.sh [options]

Runs three e5 comparisons on an audio-anchor triplet subset while keeping the
target gallery fixed to the full CVR manifest.

Options:
  --audio-matters-triplets PATH
  --full-gallery-triplets PATH
  --run-root PATH
  --runs-root PATH
  --gpu-id ID
  --smoke-size N
  --topk 1,5,10
  --topk-trace N
  --video-max-pixels N
  --video-fps N
  --e5-model PATH
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-matters-triplets) AUDIO_MATTERS_TRIPLETS="$2"; shift 2 ;;
    --full-gallery-triplets) FULL_GALLERY_TRIPLETS="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --smoke-size) SMOKE_SIZE="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --topk-trace) TOPK_TRACE="$2"; shift 2 ;;
    --video-max-pixels) VIDEO_MAX_PIXELS="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[e5-audio-matters] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$AUDIO_MATTERS_TRIPLETS" ]; then
  LATEST_AUDIO_MATTERS=$(ls -td "$RUNS_ROOT"/audio_matters_filter_* 2>/dev/null | head -1 || true)
  if [ -n "$LATEST_AUDIO_MATTERS" ]; then
    AUDIO_MATTERS_TRIPLETS="$LATEST_AUDIO_MATTERS/audio_matters_triplets.jsonl"
  fi
fi
if [ -z "$FULL_GALLERY_TRIPLETS" ]; then
  LATEST_FULL=$(ls -td "$RUNS_ROOT"/composed_triplets_full_* 2>/dev/null | head -1 || true)
  if [ -n "$LATEST_FULL" ]; then
    FULL_GALLERY_TRIPLETS="$LATEST_FULL/triplets.jsonl"
  fi
fi

test -f "$AUDIO_MATTERS_TRIPLETS" || { echo "[e5-audio-matters] missing audio matters triplets: $AUDIO_MATTERS_TRIPLETS" >&2; exit 1; }
test -f "$FULL_GALLERY_TRIPLETS" || { echo "[e5-audio-matters] missing full gallery triplets: $FULL_GALLERY_TRIPLETS" >&2; exit 1; }
test -f "$E5_MODEL/config.json" || { echo "[e5-audio-matters] missing e5 config: $E5_MODEL/config.json" >&2; exit 1; }

QUERY_COUNT=$(wc -l < "$AUDIO_MATTERS_TRIPLETS")
GALLERY_COUNT=$(wc -l < "$FULL_GALLERY_TRIPLETS")
if [ "$QUERY_COUNT" -le 0 ]; then
  echo "[e5-audio-matters] audio matters triplets are empty: $AUDIO_MATTERS_TRIPLETS" >&2
  exit 1
fi
if [ "$GALLERY_COUNT" -le 0 ]; then
  echo "[e5-audio-matters] full gallery triplets are empty: $FULL_GALLERY_TRIPLETS" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"/target_index

echo "[e5-audio-matters] repo=$REPO_ROOT"
echo "[e5-audio-matters] run_root=$RUN_ROOT"
echo "[e5-audio-matters] audio_matters_triplets=$AUDIO_MATTERS_TRIPLETS query_count=$QUERY_COUNT"
echo "[e5-audio-matters] full_gallery_triplets=$FULL_GALLERY_TRIPLETS gallery_count=$GALLERY_COUNT"
echo "[e5-audio-matters] gpu_id=$GPU_ID"

run_one() {
  local name="$1"
  local video_audio_mode="$2"
  local reference_audio_mode="$3"
  local target_index_dir="$4"
  local subrun="$RUN_ROOT/$name"
  mkdir -p "$subrun" "$target_index_dir"
  echo "[e5-audio-matters] start $name video_audio_mode=$video_audio_mode reference_audio_mode=$reference_audio_mode"
  bash scripts/run_e5_cvr_eval.sh \
    --triplets-jsonl "$AUDIO_MATTERS_TRIPLETS" \
    --gallery-triplets-jsonl "$FULL_GALLERY_TRIPLETS" \
    --run-root "$subrun" \
    --runs-root "$RUNS_ROOT" \
    --target-index-dir "$target_index_dir" \
    --e5-model "$E5_MODEL" \
    --gpu-id "$GPU_ID" \
    --expected-count "$QUERY_COUNT" \
    --gallery-expected-count "$GALLERY_COUNT" \
    --smoke-size "$SMOKE_SIZE" \
    --query-mode composed \
    --reference-audio-mode "$reference_audio_mode" \
    --video-audio-mode "$video_audio_mode" \
    --topk "$TOPK" \
    --topk-trace "$TOPK_TRACE" \
    --video-max-pixels "$VIDEO_MAX_PIXELS" \
    --video-fps "$VIDEO_FPS"
  echo "[e5-audio-matters] done $name"
}

run_one "audio_on" "on" "original" "$RUN_ROOT/target_index/audio_on"
run_one "audio_off" "off" "original" "$RUN_ROOT/target_index/audio_off"
run_one "ref_muted" "on" "muted" "$RUN_ROOT/target_index/audio_on"

RUN_ROOT="$RUN_ROOT" QUERY_COUNT="$QUERY_COUNT" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
query_count = int(os.environ["QUERY_COUNT"])
rows = []
for name, label in [
    ("audio_on", "E5 composed audio-on"),
    ("audio_off", "E5 composed audio-off"),
    ("ref_muted", "E5 ref-muted composed"),
]:
    summary_path = root / name / f"full{query_count}" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append({"name": name, "method": label, **summary["recall"]})

comparison = {
    "run_root": str(root),
    "query_count": query_count,
    "rows": rows,
}
(root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = [
    "# E5 Audio-Matters Comparison",
    "",
    "| Method | R@1 | R@5 | R@10 |",
    "|---|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['method']} | {row.get('R@1', 0):.4f} | {row.get('R@5', 0):.4f} | {row.get('R@10', 0):.4f} |"
    )
(root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((root / "comparison.md").read_text(encoding="utf-8"))
PY

echo "[e5-audio-matters] comparison=$RUN_ROOT/comparison.md"
