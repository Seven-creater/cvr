#!/usr/bin/env bash
set -euo pipefail

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

cd /data02/usr/wangqihao/Demo/test/cvr
export PYTHONPATH=/data02/usr/wangqihao/Demo/test/cvr

ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
RUN_ROOT=/data02/usr/wangqihao/Demo/test/cvr/runs/composed_pilot50_20260422
MODEL=/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
BASE_URL=http://127.0.0.1:8093/v1

mkdir -p "$RUN_ROOT"

echo "[pilot50] start $(date)"
echo "[pilot50] root=$ROOT"
echo "[pilot50] base_url=$BASE_URL"
echo "[pilot50] model=$MODEL"

curl -fsS "$BASE_URL/models"
echo

python -m app.composed_data detective-annotate-clips \
  --root "$ROOT" \
  --clips-manifest-path "$ROOT/metadata/source_clips_pilot50.jsonl" \
  --output-path "$RUN_ROOT/clip_annotations_pilot50.jsonl" \
  --base-url "$BASE_URL" \
  --api-key EMPTY \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --overwrite

echo "[pilot50] annotation done $(date)"

python -m app.composed_data propose-pairs \
  --root "$ROOT" \
  --clip-annotations-path "$RUN_ROOT/clip_annotations_pilot50.jsonl" \
  --output-path "$RUN_ROOT/pilot_pair_proposals.jsonl" \
  --base-url "$BASE_URL" \
  --api-key EMPTY \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --overwrite

echo "[pilot50] proposal done $(date)"

python - <<'PY'
import json
import hashlib
from pathlib import Path

MIN_SELECTION_CONTEXT_SCORES = [0.12, 0.10, 0.08]
run_root = Path("/data02/usr/wangqihao/Demo/test/cvr/runs/composed_pilot50_20260422")
pairs_path = run_root / "pilot_pair_proposals.jsonl"
pilot_path = run_root / "pilot_10.jsonl"
records = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def context_score(record):
    try:
        return float(record.get("quality", {}).get("same_context_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def proposal_id_for(record):
    reference = str(record.get("reference_video", "")).strip()
    target = str(record.get("target_video", "")).strip()
    digest = hashlib.sha1(f"{reference}::{target}".encode("utf-8")).hexdigest()[:16]
    return f"proposal__{digest}"


non_cross_records = [
    record
    for record in records
    if record.get("source_context", {}).get("relation") != "cross_dataset"
]
selection_threshold = None
high_context_records = []
for threshold in MIN_SELECTION_CONTEXT_SCORES:
    threshold_records = [record for record in non_cross_records if context_score(record) >= threshold]
    if len(threshold_records) >= 5:
        selection_threshold = threshold
        high_context_records = threshold_records
        break
if not high_context_records:
    selection_threshold = MIN_SELECTION_CONTEXT_SCORES[-1]
    high_context_records = [
        record
        for record in non_cross_records
        if context_score(record) >= selection_threshold
    ]
selection_records = high_context_records if len(high_context_records) >= 5 else non_cross_records
selection_records = selection_records if len(selection_records) >= 5 else records
if len(records) < 5:
    raise SystemExit(f"need at least 5 pair proposals, got {len(records)}")
pilot_target_count = min(10, len(selection_records))


def quality_score(record):
    quality = record.get("quality", {})
    return (
        float(quality.get("same_context_score", 0.0)) * 0.45
        + float(quality.get("edit_match_score", 0.0)) * 0.35
        + float(quality.get("target_uniqueness_score", 0.0)) * 0.20
    )


def sort_key(record):
    difference_type = str(record.get("difference", {}).get("type", ""))
    priority_bonus = {
        "action": 0.10,
        "audio_event": 0.09,
        "object_count": 0.08,
        "object_presence": 0.07,
        "speech": 0.03,
        "scene": -0.04,
    }.get(difference_type, 0.0)
    audio_bonus = 0.03 if "audio" in record.get("modalities", []) else 0.0
    source_bonus = float(record.get("source_context", {}).get("score", 0.0)) * 0.05
    fallback_penalty = 0.15 if record.get("fallback_used") else 0.0
    return quality_score(record) + priority_bonus + audio_bonus + source_bonus - fallback_penalty


def is_audio(record):
    return "audio" in record.get("modalities", [])


def is_object_change(record):
    return str(record.get("difference", {}).get("type", "")) in {"object_count", "object_presence"}


def is_action(record):
    return str(record.get("difference", {}).get("type", "")) == "action"


def is_non_scene(record):
    return str(record.get("difference", {}).get("type", "")) != "scene"


ranked = sorted(selection_records, key=sort_key, reverse=True)
selected = []
selected_ids = set()
selected_pair_keys = set()


def take(predicate, target_count):
    for record in ranked:
        if len(selected) >= pilot_target_count:
            return
        if len([item for item in selected if predicate(item)]) >= target_count:
            return
        proposal_id = proposal_id_for(record)
        if proposal_id in selected_ids:
            continue
        pair_key = (record.get("reference_video"), record.get("target_video"))
        if pair_key in selected_pair_keys:
            continue
        if predicate(record):
            selected.append(record)
            selected_ids.add(proposal_id)
            selected_pair_keys.add(pair_key)


take(is_audio, 2)
take(is_object_change, 2)
take(is_action, 1)
take(is_non_scene, 5)

difference_counts = {}
for record in selected:
    difference_type = str(record.get("difference", {}).get("type", "unknown"))
    difference_counts[difference_type] = difference_counts.get(difference_type, 0) + 1

for record in ranked:
    if len(selected) >= pilot_target_count:
        break
    proposal_id = proposal_id_for(record)
    if proposal_id in selected_ids:
        continue
    pair_key = (record.get("reference_video"), record.get("target_video"))
    if pair_key in selected_pair_keys:
        continue
    difference_type = str(record.get("difference", {}).get("type", "unknown"))
    if difference_counts.get(difference_type, 0) >= 4 and len(selected) < 8:
        continue
    selected.append(record)
    selected_ids.add(proposal_id)
    selected_pair_keys.add(pair_key)
    difference_counts[difference_type] = difference_counts.get(difference_type, 0) + 1

for record in ranked:
    if len(selected) >= pilot_target_count:
        break
    proposal_id = proposal_id_for(record)
    pair_key = (record.get("reference_video"), record.get("target_video"))
    if proposal_id not in selected_ids and pair_key not in selected_pair_keys:
        selected.append(record)
        selected_ids.add(proposal_id)
        selected_pair_keys.add(pair_key)

with pilot_path.open("w", encoding="utf-8") as handle:
    for index, record in enumerate(selected, start=1):
        sample = {
            "sample_id": f"covr_pilot_{index:04d}",
            "reference_video": record["reference_video"],
            "target_video": record["target_video"],
            "edit_text": record["edit_text"],
            "modalities": record["modalities"],
            "reference_caption": record["reference_caption"],
            "target_caption": record["target_caption"],
            "difference": record["difference"],
            "hard_negatives": record["hard_negatives"],
            "quality": record["quality"],
            "source": record["source"],
            "source_context": record.get("source_context", {}),
            "proposal_id": proposal_id_for(record),
            "proposal_reason": record.get("proposal_reason", ""),
            "fallback_used": record.get("fallback_used", False),
        }
        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
summary = {
    "proposal_count": len(records),
    "non_cross_pool_count": len(non_cross_records),
    "high_context_pool_count": len(high_context_records),
    "selection_pool_count": len(selection_records),
    "min_selection_context_scores": MIN_SELECTION_CONTEXT_SCORES,
    "selected_context_threshold": selection_threshold,
    "pilot_count": len(selected),
    "audio_count": sum(1 for record in selected if is_audio(record)),
    "object_change_count": sum(1 for record in selected if is_object_change(record)),
    "action_count": sum(1 for record in selected if is_action(record)),
    "difference_type_counts": {},
    "source_context_counts": {},
    "fallback_count": sum(1 for record in selected if record.get("fallback_used")),
    "pilot_path": str(pilot_path),
}
selected_context_scores = [context_score(record) for record in selected]
if selected_context_scores:
    summary["selected_context_score"] = {
        "min": round(min(selected_context_scores), 3),
        "avg": round(sum(selected_context_scores) / len(selected_context_scores), 3),
        "max": round(max(selected_context_scores), 3),
    }
for record in selected:
    difference_type = str(record.get("difference", {}).get("type", "unknown"))
    summary["difference_type_counts"][difference_type] = summary["difference_type_counts"].get(difference_type, 0) + 1
    relation = str(record.get("source_context", {}).get("relation", "unknown"))
    summary["source_context_counts"][relation] = summary["source_context_counts"].get(relation, 0) + 1
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

python -m app.composed_data validate-pilot \
  --root "$ROOT" \
  --pilot-jsonl-path "$RUN_ROOT/pilot_10.jsonl" \
  --gallery-output-path "$RUN_ROOT/gallery.jsonl" \
  --report-output-path "$RUN_ROOT/pilot_review.md"

echo "[verify] outputs"
ls -lh "$RUN_ROOT/clip_annotations_pilot50.jsonl"
ls -lh "$RUN_ROOT/pilot_pair_proposals.jsonl"
ls -lh "$RUN_ROOT/pilot_10.jsonl"
ls -lh "$RUN_ROOT/gallery.jsonl"
cat "$RUN_ROOT/pilot_review.md"

echo "[pilot50] done $(date)"
