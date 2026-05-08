#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATASET_ROOT=${DATASET_ROOT:-/data02/usr/wangqihao/Demo/test/data}
RUNS_ROOT=${RUNS_ROOT:-/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs}
OUTPUT_DIR=${OUTPUT_DIR:-$RUNS_ROOT/composed_triplets_full_$(date +%Y%m%d_%H%M%S)}
EXPECTED_COUNT=${EXPECTED_COUNT:-943}

usage() {
  cat <<'EOF'
Usage: build_composed_triplets.sh [options]

Build only the ref-target-edit triplet manifest. This script does not use GPU,
does not start AVIGATE, and does not contact any Omni service.

Options:
  --dataset-root PATH
  --output-dir PATH
  --expected-count N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --expected-count) EXPECTED_COUNT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[build-composed-triplets] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ ! -d "$DATASET_ROOT" ]; then
  echo "[build-composed-triplets] dataset root not found: $DATASET_ROOT" >&2
  exit 1
fi

echo "[build-composed-triplets] repo=$REPO_ROOT"
echo "[build-composed-triplets] dataset_root=$DATASET_ROOT"
echo "[build-composed-triplets] output_dir=$OUTPUT_DIR"
echo "[build-composed-triplets] expected_count=$EXPECTED_COUNT"

python3 -m app.composed_triplets \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-count "$EXPECTED_COUNT"

echo "[build-composed-triplets] wrote $OUTPUT_DIR/triplets.jsonl"
