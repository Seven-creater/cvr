#!/usr/bin/env bash
set -euo pipefail

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

cd /data02/usr/wangqihao/Demo/test/cvr
export PYTHONPATH=/data02/usr/wangqihao/Demo/test/cvr

ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
DAILY_ROOT="$ROOT/raw_datasets/daily_omni"
WORLDSENSE_ROOT="$ROOT/raw_datasets/worldsense"

python -m pip install -U pyarrow

echo "[prepare] start $(date)"
echo "[prepare] root=$ROOT"
echo "[prepare] daily_omni=$DAILY_ROOT"
echo "[prepare] worldsense=$WORLDSENSE_ROOT"

python -m app.composed_sources prepare \
  --root "$ROOT" \
  --daily-omni-root "$DAILY_ROOT" \
  --worldsense-root "$WORLDSENSE_ROOT" \
  --clip-limit 50

echo "[verify] outputs"
ls -lh "$ROOT/metadata/source_rows.jsonl"
ls -lh "$ROOT/metadata/source_clips_all.jsonl"
ls -lh "$ROOT"/metadata/source_clips_pilot*.jsonl
cat "$ROOT/reports/source_dataset_prepare_summary.md"

echo "[prepare] done $(date)"
