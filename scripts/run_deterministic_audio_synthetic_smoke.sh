#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/deterministic_audio_synthetic_smoke}
REFERENCE_GLOB=${REFERENCE_GLOB:-$DATA_ROOT/clips/detective/daily_omni/*.mp4}
MAX_AUDIO_SAMPLES=${MAX_AUDIO_SAMPLES:-3}
mkdir -p "$RUN_ROOT/videos" "$RUN_ROOT/captions" "$RUN_ROOT/pairs"
mkdir -p "$RUN_ROOT/logs"

echo "[audio-smoke] start $(date)"
echo "[audio-smoke] data_root=$DATA_ROOT"
echo "[audio-smoke] run_root=$RUN_ROOT"
echo "[audio-smoke] max_audio_samples=$MAX_AUDIO_SAMPLES"
echo "[audio-smoke] deterministic audio only; no large model is loaded"

mapfile -t REFERENCES < <(ls $REFERENCE_GLOB 2>/dev/null | head -100)
if [[ "${#REFERENCES[@]}" -lt 1 ]]; then
  echo "[audio-smoke] need at least 1 reference mp4 file from $REFERENCE_GLOB" >&2
  exit 1
fi

EVENTS=(
  "whoosh"
  "scratching sound"
  "low-frequency hum"
  "wind noise"
  "rain-like noise"
  "high-pitched beep"
  "mechanical buzz"
  "static hiss"
  "soft bell chime"
  "water splash noise"
)
FILTERS=(
  "anoisesrc=color=pink:duration=12,highpass=f=500,lowpass=f=3500,afade=t=in:st=0:d=0.15,afade=t=out:st=1.0:d=0.4,volume=0.35"
  "anoisesrc=color=white:duration=12,highpass=f=1200,lowpass=f=6000,volume=0.12"
  "sine=frequency=120:duration=12,volume=0.18"
  "anoisesrc=color=brown:duration=12,lowpass=f=900,volume=0.16"
  "anoisesrc=color=white:duration=12,highpass=f=350,lowpass=f=4500,volume=0.10"
  "sine=frequency=880:duration=12,volume=0.12"
  "sine=frequency=240:duration=12,volume=0.14"
  "anoisesrc=color=white:duration=12,highpass=f=2500,volume=0.08"
  "sine=frequency=660:duration=12,afade=t=in:st=0:d=0.05,afade=t=out:st=1.4:d=0.5,volume=0.13"
  "anoisesrc=color=pink:duration=12,highpass=f=250,lowpass=f=2200,afade=t=in:st=0:d=0.05,afade=t=out:st=0.7:d=0.3,volume=0.20"
)

ANNOTATIONS="$RUN_ROOT/synthetic_annotations.jsonl"
PAIRS="$RUN_ROOT/synthetic_candidate_pairs.jsonl"
if [[ "$MAX_AUDIO_SAMPLES" -gt "${#EVENTS[@]}" ]]; then
  echo "[audio-smoke] refusing MAX_AUDIO_SAMPLES=$MAX_AUDIO_SAMPLES; only ${#EVENTS[@]} deterministic events are defined" >&2
  exit 1
fi

: > "$ANNOTATIONS"
: > "$PAIRS"

for ((idx = 0; idx < MAX_AUDIO_SAMPLES; idx++)); do
  ref="${REFERENCES[$((idx % ${#REFERENCES[@]}))]}"
  event="${EVENTS[$idx]}"
  filter="${FILTERS[$idx]}"
  sample_num=$(printf "%04d" $((idx + 1)))
  target="$RUN_ROOT/videos/synthetic_audio_${sample_num}.mp4"
  wav="$RUN_ROOT/videos/synthetic_audio_${sample_num}.wav"
  wav_log="$RUN_ROOT/logs/synthetic_audio_${sample_num}_wav.log"
  mux_log="$RUN_ROOT/logs/synthetic_audio_${sample_num}_mux.log"

  ffmpeg -y -f lavfi -i "$filter" -t 12 -c:a pcm_s16le "$wav" >"$wav_log" 2>&1
  if ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 "$ref" | grep -q audio; then
    ffmpeg -y -i "$ref" -i "$wav" \
      -filter_complex "[0:a:0][1:a:0]amix=inputs=2:duration=first:dropout_transition=0[a]" \
      -map 0:v:0 -map "[a]" -c:v copy -c:a aac -shortest "$target" >"$mux_log" 2>&1
    audio_strategy="overlay_reference_audio"
  else
    ffmpeg -y -i "$ref" -i "$wav" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "$target" >"$mux_log" 2>&1
    audio_strategy="replace_missing_reference_audio"
  fi

  ref_rel="$ref"
  target_rel="$target"
  python3 - "$ANNOTATIONS" "$PAIRS" "$idx" "$ref_rel" "$target_rel" "$event" "$audio_strategy" <<'PY'
import json
import sys
from pathlib import Path

annotations_path = Path(sys.argv[1])
pairs_path = Path(sys.argv[2])
idx = int(sys.argv[3])
ref = sys.argv[4]
target = sys.argv[5]
event = sys.argv[6]
audio_strategy = sys.argv[7]
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
            "mixing": "overlay",
            "expected_event": event,
            "audio_strategy": audio_strategy,
        },
        "postprocess": {
            "video_copied_from_reference": True,
            "audio_copied_from_reference": audio_strategy == "overlay_reference_audio",
            "audio_strategy": audio_strategy,
        },
    },
})
PY
done

echo "[audio-smoke] wrote $ANNOTATIONS"
echo "[audio-smoke] wrote $PAIRS"
echo "[audio-smoke] done $(date)"
