#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/deterministic_audio_synthetic_smoke}
REFERENCE_GLOB=${REFERENCE_GLOB:-$DATA_ROOT/clips/detective/daily_omni/*.mp4}
mkdir -p "$RUN_ROOT/videos" "$RUN_ROOT/captions" "$RUN_ROOT/pairs"

echo "[audio-smoke] start $(date)"
echo "[audio-smoke] data_root=$DATA_ROOT"
echo "[audio-smoke] run_root=$RUN_ROOT"
echo "[audio-smoke] deterministic audio only; no large model is loaded"

mapfile -t REFERENCES < <(ls $REFERENCE_GLOB 2>/dev/null | head -3)
if [[ "${#REFERENCES[@]}" -lt 3 ]]; then
  echo "[audio-smoke] need at least 3 reference mp4 files from $REFERENCE_GLOB" >&2
  exit 1
fi

EVENTS=("whoosh" "scratching sound" "low-frequency hum")
FILTERS=(
  "anoisesrc=color=pink:duration=12,highpass=f=500,lowpass=f=3500,afade=t=in:st=0:d=0.15,afade=t=out:st=1.0:d=0.4,volume=0.35"
  "anoisesrc=color=white:duration=12,highpass=f=1200,lowpass=f=6000,volume=0.12"
  "sine=frequency=120:duration=12,volume=0.18"
)

ANNOTATIONS="$RUN_ROOT/synthetic_annotations.jsonl"
PAIRS="$RUN_ROOT/synthetic_candidate_pairs.jsonl"
: > "$ANNOTATIONS"
: > "$PAIRS"

for idx in 0 1 2; do
  ref="${REFERENCES[$idx]}"
  event="${EVENTS[$idx]}"
  filter="${FILTERS[$idx]}"
  sample_num=$(printf "%04d" $((idx + 1)))
  target="$RUN_ROOT/videos/synthetic_audio_${sample_num}.mp4"
  wav="$RUN_ROOT/videos/synthetic_audio_${sample_num}.wav"

  ffmpeg -y -f lavfi -i "$filter" -t 12 -c:a pcm_s16le "$wav" >/dev/null 2>&1
  ffmpeg -y -i "$ref" -i "$wav" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "$target" >/dev/null 2>&1

  ref_rel="$ref"
  target_rel="$target"
  python3 - "$ANNOTATIONS" "$PAIRS" "$idx" "$ref_rel" "$target_rel" "$event" <<'PY'
import json
import sys
from pathlib import Path

annotations_path = Path(sys.argv[1])
pairs_path = Path(sys.argv[2])
idx = int(sys.argv[3])
ref = sys.argv[4]
target = sys.argv[5]
event = sys.argv[6]
sample_num = f"{idx + 1:04d}"
ref_id = f"det_audio_ref_{sample_num}"
target_id = f"det_audio_target_{sample_num}"

def write(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

base_summary = "Reference video preserved exactly for deterministic audio editing."
write(annotations_path, {
    "clip_id": ref_id,
    "output_path": ref,
    "summary": base_summary,
    "subjects": [],
    "object_counts": {},
    "actions": [],
    "scene": "same visual scene",
    "attributes": [],
    "visible_text": [],
    "speech": [],
    "audio_events": [],
    "modalities": ["visual", "audio"],
})
write(annotations_path, {
    "clip_id": target_id,
    "output_path": target,
    "summary": f"{base_summary} The audio contains {event}.",
    "subjects": [],
    "object_counts": {},
    "actions": [],
    "scene": "same visual scene",
    "attributes": [],
    "visible_text": [],
    "speech": [],
    "audio_events": [event],
    "modalities": ["visual", "audio"],
})
write(pairs_path, {
    "proposal_id": f"synthetic_audio_pair_{sample_num}",
    "source_type": "synthetic_edit",
    "reference_clip_id": ref_id,
    "target_clip_id": target_id,
    "reference_video": ref,
    "target_video": target,
    "edit_text": f"add {event} to the audio",
    "modalities": ["audio"],
    "difference": {"type": "audio_event", "from": f"no {event}", "to": event},
    "quality": {
        "same_context_score": 0.98,
        "edit_match_score": 0.9,
        "target_uniqueness_score": 0.9,
        "difference_strength_score": 0.8,
        "visual_near_duplicate_score": 0.99,
    },
    "hard_negatives": [],
    "generation": {
        "model": "ffmpeg-deterministic-audio",
        "model_route": "deterministic_overlay",
        "source_video": ref,
        "prompt": f"Add {event} while preserving the video stream.",
        "audio_edit_plan": {
            "route": "deterministic_overlay",
            "audio_prompt": event,
            "negative_audio_prompt": "speech, narration, talking, voiceover",
            "preserve_video": True,
            "mixing": "replace",
            "expected_event": event,
        },
        "postprocess": {"video_copied_from_reference": True, "audio_copied_from_reference": False},
    },
})
PY
done

echo "[audio-smoke] wrote $ANNOTATIONS"
echo "[audio-smoke] wrote $PAIRS"
echo "[audio-smoke] done $(date)"
