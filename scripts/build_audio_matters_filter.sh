#!/usr/bin/env bash
set -euo pipefail

if [ -f /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-omni_src}"
fi

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUNS_ROOT=${RUNS_ROOT:-/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs}
TRIPLETS_JSONL=${TRIPLETS_JSONL:-}
OUTPUT_DIR=${OUTPUT_DIR:-$RUNS_ROOT/audio_matters_filter_$(date +%Y%m%d_%H%M%S)}
EXPECTED_COUNT=${EXPECTED_COUNT:-943}
MIN_AUDIO_ANCHOR_SCORE=${MIN_AUDIO_ANCHOR_SCORE:-0.85}
MIN_RMS=${MIN_RMS:-0.0001}
SAMPLE_RATE=${SAMPLE_RATE:-16000}
MAX_AUDIO_SECONDS=${MAX_AUDIO_SECONDS:-12}
FFMPEG=${FFMPEG:-ffmpeg}

usage() {
  cat <<'EOF'
Usage: build_audio_matters_filter.sh [options]

Filters existing CVR triplets into a visual-edit subset whose reference and
target audio are highly similar. This script only reads existing videos and
writes a new run directory.

Options:
  --triplets-jsonl PATH
  --runs-root PATH
  --output-dir PATH
  --expected-count N
  --min-audio-anchor-score FLOAT
  --min-rms FLOAT
  --sample-rate N
  --max-audio-seconds N
  --ffmpeg PATH
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --triplets-jsonl) TRIPLETS_JSONL="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --expected-count) EXPECTED_COUNT="$2"; shift 2 ;;
    --min-audio-anchor-score) MIN_AUDIO_ANCHOR_SCORE="$2"; shift 2 ;;
    --min-rms) MIN_RMS="$2"; shift 2 ;;
    --sample-rate) SAMPLE_RATE="$2"; shift 2 ;;
    --max-audio-seconds) MAX_AUDIO_SECONDS="$2"; shift 2 ;;
    --ffmpeg) FFMPEG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[audio-matters] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$TRIPLETS_JSONL" ]; then
  LATEST_TRIPLETS_DIR=$(ls -td "$RUNS_ROOT"/composed_triplets_full_* 2>/dev/null | head -1 || true)
  if [ -n "$LATEST_TRIPLETS_DIR" ]; then
    TRIPLETS_JSONL="$LATEST_TRIPLETS_DIR/triplets.jsonl"
  fi
fi

if [ -z "$TRIPLETS_JSONL" ] || [ ! -f "$TRIPLETS_JSONL" ]; then
  echo "[audio-matters] missing triplets jsonl: ${TRIPLETS_JSONL:-}" >&2
  exit 1
fi
command -v "$FFMPEG" >/dev/null || { echo "[audio-matters] missing ffmpeg: $FFMPEG" >&2; exit 1; }

COUNT=$(wc -l < "$TRIPLETS_JSONL")
if [ "$EXPECTED_COUNT" -gt 0 ] && [ "$COUNT" -ne "$EXPECTED_COUNT" ]; then
  echo "[audio-matters] expected $EXPECTED_COUNT triplets, got $COUNT: $TRIPLETS_JSONL" >&2
  exit 1
fi

echo "[audio-matters] repo=$REPO_ROOT"
echo "[audio-matters] triplets_jsonl=$TRIPLETS_JSONL"
echo "[audio-matters] output_dir=$OUTPUT_DIR"
echo "[audio-matters] min_audio_anchor_score=$MIN_AUDIO_ANCHOR_SCORE"
echo "[audio-matters] min_rms=$MIN_RMS"
echo "[audio-matters] sample_rate=$SAMPLE_RATE max_audio_seconds=$MAX_AUDIO_SECONDS"

python3 -m app.audio_matters_filter \
  --triplets-jsonl "$TRIPLETS_JSONL" \
  --runs-root "$RUNS_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-count "$EXPECTED_COUNT" \
  --min-audio-anchor-score "$MIN_AUDIO_ANCHOR_SCORE" \
  --min-rms "$MIN_RMS" \
  --sample-rate "$SAMPLE_RATE" \
  --max-audio-seconds "$MAX_AUDIO_SECONDS" \
  --ffmpeg "$FFMPEG"

echo "[audio-matters] summary=$OUTPUT_DIR/summary.json"
echo "[audio-matters] triplets=$OUTPUT_DIR/audio_matters_triplets.jsonl"
