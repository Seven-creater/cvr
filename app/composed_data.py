from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.composed_omni import ALLOWED_DIFFERENCE_TYPES, OpenAIComposedDataClient


DEFAULT_DATA_ROOT = "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval"
LAYOUT_DIRS = ("raw", "clips", "metadata", "captions", "pairs", "splits", "reports", "caches")
DEFAULT_RAW_INDEX_NAME = "raw_assets.jsonl"
DEFAULT_CLIP_MANIFEST_NAME = "clips.jsonl"
DEFAULT_CLIP_ANNOTATIONS_NAME = "clip_annotations.jsonl"
DEFAULT_PAIR_PROPOSALS_NAME = "pilot_candidates.jsonl"
DEFAULT_CLIP_GROUPS_NAME = "clip_groups.jsonl"
DEFAULT_DETECTIVE_CLIP_PLAN_NAME = "clip_plan_detective.jsonl"
DEFAULT_EVENT_CLIP_MANIFEST_NAME = "extracted_event_clips.jsonl"
DEFAULT_ACCEPTED_PAIRS_NAME = "accepted_pairs.jsonl"
DEFAULT_SYNTHETIC_JUDGED_PAIRS_NAME = "judged_synthetic_pair_proposals.jsonl"
DEFAULT_SYNTHETIC_ACCEPTED_PAIRS_NAME = "accepted_synthetic_pairs.jsonl"
DEFAULT_SYNTHETIC_PILOT_REVIEW_NAME = "synthetic_pilot_review.md"
DEFAULT_VIDEO_EDIT_PLAN_NAME = "video_edit_plan.jsonl"
DEFAULT_VIDEO_EDIT_PLANNER_CACHE_NAME = "video_edit_planner_cache.jsonl"
DEFAULT_VIDEO_MASK_PLAN_NAME = "video_mask_plan.jsonl"
DEFAULT_VIDEO_MASK_MANIFEST_NAME = "video_mask_manifest.jsonl"
DEFAULT_OMNI_STABLE_CLIP_SELECTION_CACHE_NAME = "omni_stable_clip_selection_cache.jsonl"
DEFAULT_REFERENCE_UNDERSTANDING_CACHE_NAME = "reference_understanding_cache.jsonl"
DEFAULT_SRC_REF_IMAGE_PLAN_NAME = "src_ref_image_plan.jsonl"
DEFAULT_SRC_REF_IMAGE_SELECTION_NAME = "src_ref_image_selection.jsonl"
DEFAULT_AUDIO_EDIT_PLAN_NAME = "audio_edit_plan.jsonl"
DEFAULT_LICENSE_NOTE = "internal research pilot only"
VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
ALLOWED_MODALITIES = {"visual", "audio"}
ALLOWED_SOURCE_TYPES = {"natural", "synthetic_edit"}
MAX_PAIR_CANDIDATES = 40
MIN_PAIR_CONTEXT_SCORE = 0.03
MAX_PAIR_CHANGED_TYPES = 5
MIN_PAIR_EDIT_MATCH_SCORE = 0.15
PAIR_PRIORITY = (
    "object_count",
    "object_presence",
    "action",
    "audio_event",
    "attribute",
    "scene",
    "speech",
    "visible_text",
)
HIGH_CONTEXT_PAIR_PRIORITY = (
    "object_count",
    "object_presence",
    "action",
    "audio_event",
    "speech",
    "visible_text",
    "attribute",
    "scene",
)
DIVERSE_PAIR_BUCKET_TARGETS = {
    "object_count": 3,
    "action": 3,
    "audio_event": 4,
    "speech": 3,
    "object_presence": 3,
    "visible_text": 3,
    "attribute": 2,
    "scene": 1,
}
MIN_ACCEPT_SAME_CONTEXT_SCORE = 0.55
MIN_ACCEPT_EDIT_MATCH_SCORE = 0.75
MIN_ACCEPT_TARGET_UNIQUENESS_SCORE = 0.70
MIN_ACCEPT_EDIT_NECESSITY_SCORE = 0.70
MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE = 0.75
MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE = 0.65
MIN_ACCEPT_ACTION_EVIDENCE_SCORE = 0.65
MIN_ACCEPT_SPEECH_EVIDENCE_SCORE = 0.75
MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE = 0.70
MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE = 0.70
MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE = 0.75
MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE = 0.85
MIN_SYNTHETIC_AUDIO_VISUAL_CONTEXT_SCORE = 0.95
MIN_VIDEO_MASK_COVERAGE_RATIO = 0.02
MAX_VIDEO_MASK_COVERAGE_RATIO = 0.65
MIN_VIDEO_MASK_TEMPORAL_STABILITY = 0.75
MIN_VIDEO_MASK_NONEMPTY_FRAME_RATIO = 0.90
MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE = 0.995
VISUAL_DIFFERENCE_TYPES = {"object_count", "object_presence", "attribute", "action", "scene", "visible_text"}
SYNTHETIC_VISUAL_ROUTES = {"vace_controlled", "ltx2_retake", "tokenflow_style"}
SYNTHETIC_AUDIO_ROUTES = {"deterministic_overlay", "foleycrafter_temporal", "frieren_benchmark", "audio_deterministic"}
VACE_ATTRIBUTE_MARKERS = {
    "attribute",
    "color",
    "colour",
    "bright",
    "yellow",
    "red",
    "blue",
    "green",
    "silver",
    "gold",
    "black",
    "white",
    "body",
    "shell",
    "surface",
    "material",
    "metal",
    "metallic",
    "matte",
    "plastic",
    "texture",
    "style",
    "visor",
    "light",
    "clothing",
    "shirt",
    "jacket",
    "dress",
    "vehicle",
    "car",
    "robot",
    "background",
    "backdrop",
    "room",
    "street",
    "office",
    "kitchen",
    "laboratory",
    "lab",
    "cyberpunk",
    "anime",
    "cinematic",
    "neon",
    "weather",
    "rain",
    "night",
    "day",
}
VACE_TINY_OR_INSERTION_MARKERS = {
    "sticker",
    "poster",
    "plant",
    "potted",
    "badge",
    "button",
    "logo",
    "label",
    "sign",
    "text",
    "caption",
    "nose ring",
    "earring",
    "ear ring",
    "necklace",
    "bracelet",
    "watch",
    "flower",
    "cube",
    "eraser",
}
VACE_BACKGROUND_STYLE_MARKERS = {
    "background",
    "backdrop",
    "room",
    "street",
    "office",
    "kitchen",
    "laboratory",
    "lab",
    "cyberpunk",
    "anime",
    "oil painting",
    "cinematic",
    "neon",
    "night",
    "day",
    "rain",
    "sunset",
    "studio",
}
VACE_EXPLORATION_OBJECT_REPLACEMENTS = {
    "cup": "bottle",
    "mug": "bottle",
    "glass": "bottle",
    "phone": "tablet",
    "smartphone": "tablet",
    "mobile phone": "tablet",
    "laptop": "tablet",
    "computer": "tablet",
    "book": "notebook",
    "bag": "backpack",
    "tote bag": "backpack",
    "box": "suitcase",
    "chair": "stool",
    "bottle": "thermos",
    "toy": "wooden toy",
}
VACE_EXPLORATION_REMOVABLE_OBJECTS = {
    "cup",
    "mug",
    "glass",
    "phone",
    "smartphone",
    "mobile phone",
    "bag",
    "tote bag",
    "backpack",
    "glasses",
    "sunglasses",
    "hat",
    "chair",
    "box",
    "bottle",
}
VACE_SCREEN_TEXT_OBJECTS = {
    "computer",
    "desktop",
    "laptop",
    "monitor",
    "screen",
    "tablet",
    "television",
    "tv",
}
VACE_SEATED_SUPPORT_OBJECTS = {"bench", "chair", "seat", "sofa", "stool"}
INTRACLIP_CHANGE_MARKERS = (
    "change from",
    "changes from",
    "changed from",
    "changes to",
    "changed to",
    "replace",
    "replaced by",
    "replaces",
    "transition from",
    "transitions from",
    "turns into",
    "becomes",
    "followed by",
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
GENERIC_SPEECH_TOKENS = {
    "camera",
    "conversation",
    "dialog",
    "dialogue",
    "discuss",
    "discusses",
    "discussing",
    "female",
    "interview",
    "male",
    "man",
    "monologue",
    "narrate",
    "narrates",
    "narrating",
    "narration",
    "narrator",
    "person",
    "says",
    "speak",
    "speaker",
    "speaking",
    "speech",
    "talk",
    "talking",
    "voice",
    "voiceover",
    "woman",
}
GENERIC_SPEECH_PHRASES = {
    "speaks to camera",
    "speaking to camera",
    "talks to camera",
    "talking to camera",
    "speaks directly to the camera",
    "speaking directly to the camera",
    "speech",
    "narration",
    "talking",
    "voiceover",
}
GENERIC_EDIT_TEXT_PHRASES = {
    "change the mood",
    "make it better",
    "make it cinematic",
    "make it more cinematic",
    "make the video better",
    "make the scene better",
    "make the scene more interesting",
    "change the topic",
    "change the vibe",
}
EDIT_ACTION_VERBS = {
    "add",
    "adds",
    "appear",
    "appears",
    "begin",
    "begins",
    "change",
    "changes",
    "convert",
    "converts",
    "delete",
    "deletes",
    "disappear",
    "disappears",
    "increase",
    "increases",
    "insert",
    "inserts",
    "introduce",
    "introduced",
    "introduces",
    "launch",
    "launched",
    "make",
    "remove",
    "removes",
    "replace",
    "replaced",
    "replaces",
    "start",
    "starts",
    "swap",
    "swaps",
    "turn",
    "turns",
    "wave",
    "waving",
}
EDIT_TEXT_AUDIO_TOKENS = {
    "audio",
    "hum",
    "music",
    "noise",
    "scratch",
    "scratching",
    "sound",
    "speech",
    "voice",
    "whoosh",
}
EDIT_TEXT_VISUAL_TOKENS = {
    "background",
    "color",
    "colour",
    "object",
    "scene",
    "shirt",
    "text",
    "video",
    "visible",
}
VISUAL_DESCRIPTION_TOKENS = {
    "background",
    "beard",
    "blue",
    "camera",
    "clothes",
    "forest",
    "glasses",
    "hat",
    "jacket",
    "looking",
    "scene",
    "shirt",
    "standing",
    "wearing",
}
GENERIC_HUMAN_GROUP_TOKENS = {
    "audience",
    "controllers",
    "crew",
    "crowd",
    "employees",
    "group",
    "operators",
    "people",
    "personnel",
    "persons",
    "staff",
    "team",
    "workers",
}
OBJECT_ALIAS_GROUPS = (
    (
        "dollhouse",
        "toy house",
        "toy home",
        "play house",
        "playhouse",
    ),
    (
        "framed picture",
        "framed pictures",
        "picture",
        "pictures",
        "painting",
        "paintings",
        "poster",
        "posters",
        "wall art",
        "artwork",
        "frame",
        "frames",
    ),
    (
        "personnel",
        "people",
        "staff",
        "crowd",
        "workers",
        "persons",
        "team",
        "crew",
        "operators",
        "controllers",
        "employees",
    ),
)
BACKGROUND_DECOR_OBJECTS = {"framed picture"}
OBJECT_LABEL_STOPWORDS = {
    "a",
    "an",
    "the",
    "present",
    "visible",
    "appears",
    "appear",
    "shown",
    "showing",
}
MIN_COMPETING_DIFFERENCE_STRENGTH = 0.72
NON_SPEECH_AUDIO_TOKENS = {
    "ambient",
    "ambience",
    "applause",
    "bark",
    "barking",
    "beep",
    "bell",
    "bird",
    "birds",
    "buzz",
    "buzzing",
    "chain",
    "chainsaw",
    "cheer",
    "cheering",
    "clap",
    "clapping",
    "crash",
    "crowd",
    "drum",
    "electronic",
    "engine",
    "footstep",
    "gunshot",
    "hiss",
    "hum",
    "instrument",
    "machine",
    "mechanical",
    "laugh",
    "laughter",
    "melody",
    "music",
    "noise",
    "orchestra",
    "orchestral",
    "piano",
    "rain",
    "ring",
    "ringing",
    "river",
    "roar",
    "rumble",
    "rustle",
    "rustling",
    "score",
    "siren",
    "song",
    "splash",
    "static",
    "stream",
    "thunder",
    "traffic",
    "water",
    "waves",
    "whir",
    "whirring",
    "whoosh",
    "wind",
}
SPEECH_ONLY_AUDIO_PATTERNS = (
    "only speech",
    "speech only",
    "contains only speech",
    "contains speech only",
    "only narration",
    "narration only",
    "only talking",
    "talking only",
    "only voiceover",
    "voiceover only",
)
SPEECH_CONTENT_EDIT_PATTERNS = (
    "speech",
    "spoken content",
    "transcript",
    "narration",
    "narrator",
    "voiceover",
    "says",
    "say ",
    "topic",
    "talks about",
    "talk about",
    "discussing",
    "discussion",
)
NON_SPEECH_AUDIO_ABSENCE_PATTERNS = (
    "no background music",
    "no background noise",
    "no ambient noise",
    "no ambient sound",
    "no ambient sounds",
    "no distinctive audio",
    "no non speech audio",
    "without background music",
    "without background noise",
    "without ambient noise",
)
EDIT_TEXT_START_VERBS = {
    "add",
    "change",
    "include",
    "increase",
    "introduce",
    "make",
    "reduce",
    "remove",
    "replace",
    "start",
    "starts",
    "stop",
    "stops",
    "switch",
    "turn",
}
EDIT_TEXT_CAPTION_MAX_TOKENS = 24
EDIT_TEXT_VISUAL_LEAK_TOKENS = {
    "background",
    "blonde",
    "camera",
    "desk",
    "dollhouse",
    "hair",
    "man",
    "nose",
    "person",
    "room",
    "shirt",
    "speaking",
    "toy",
    "woman",
}
EDIT_TEXT_AUDIO_TOKENS = NON_SPEECH_AUDIO_TOKENS | {"audio", "sound", "sounds", "effect", "effects"}
EDIT_TEXT_VISIBLE_TEXT_TOKENS = {"caption", "ocr", "on", "screen", "text", "subtitle", "subtitles"}
EDIT_TEXT_SPEECH_TOKENS = GENERIC_SPEECH_TOKENS | {"transcript", "spoken", "says", "say", "topic", "topics"}
FINAL_ACCEPT_BUCKET_TARGETS = {
    "object_count": 2,
    "object_presence": 3,
    "action": 2,
    "audio_event": 2,
    "speech": 2,
    "visible_text": 2,
    "attribute": 2,
    "scene": 1,
}


@dataclass(frozen=True, slots=True)
class RawAsset:
    asset_id: str
    dataset: str
    path: str
    relative_path: str
    file_name: str
    extension: str
    size_bytes: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "dataset": self.dataset,
            "path": self.path,
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True, slots=True)
class ClipManifestRecord:
    clip_id: str
    source_asset_id: str | None
    source_path: str
    output_path: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    role: str | None
    notes: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "clip_id": self.clip_id,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
        }
        if self.source_asset_id:
            payload["source_asset_id"] = self.source_asset_id
        if self.role:
            payload["role"] = self.role
        if self.notes:
            payload["notes"] = self.notes
        return payload


def ensure_layout(root: str | Path) -> dict[str, Path]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    paths = {"root": root_path}
    for name in LAYOUT_DIRS:
        path = root_path / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def discover_raw_sources(root: str | Path) -> list[tuple[str, Path]]:
    raw_datasets_root = Path(root) / "raw_datasets"
    if not raw_datasets_root.exists():
        return []
    sources: list[tuple[str, Path]] = []
    for candidate in sorted(raw_datasets_root.iterdir()):
        if candidate.is_dir():
            sources.append((candidate.name, candidate))
    return sources


def index_raw_sources(
    *,
    root: str | Path,
    sources: list[tuple[str, str | Path]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    output = Path(output_path) if output_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME

    assets: list[RawAsset] = []
    per_dataset_counts: dict[str, int] = {}
    for dataset_name, raw_source in sources:
        source_path = Path(raw_source)
        if not source_path.exists():
            raise FileNotFoundError(f"raw source does not exist: {source_path}")
        if not source_path.is_dir():
            raise NotADirectoryError(f"raw source must be a directory: {source_path}")

        count = 0
        for path in sorted(source_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            stat = path.stat()
            relative_path = path.relative_to(source_path).as_posix()
            assets.append(
                RawAsset(
                    asset_id=_build_asset_id(dataset_name, relative_path),
                    dataset=dataset_name,
                    path=str(path),
                    relative_path=relative_path,
                    file_name=path.name,
                    extension=path.suffix.lower(),
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            count += 1
        per_dataset_counts[dataset_name] = count

    _write_jsonl(output, [asset.to_dict() for asset in assets])
    report_path = layout["reports"] / "raw_assets_summary.md"
    report_path.write_text(_build_raw_summary_report(output, per_dataset_counts), encoding="utf-8")
    return {
        "output_path": str(output),
        "report_path": str(report_path),
        "asset_count": len(assets),
        "dataset_counts": per_dataset_counts,
    }


def extract_clips(
    *,
    root: str | Path,
    plan_path: str | Path,
    raw_index_path: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    plan = list(_load_jsonl(Path(plan_path)))
    if not plan:
        raise ValueError("clip plan is empty")

    raw_index = _load_raw_asset_index(Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME)
    output_manifest = Path(output_manifest_path) if output_manifest_path else layout["metadata"] / DEFAULT_CLIP_MANIFEST_NAME

    commands: list[list[str]] = []
    records: list[ClipManifestRecord] = []
    seen_clip_ids: set[str] = set()
    for line_number, item in enumerate(plan, start=1):
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError(f"clip plan line {line_number}: clip_id is required")
        if clip_id in seen_clip_ids:
            raise ValueError(f"clip plan line {line_number}: duplicate clip_id={clip_id}")
        seen_clip_ids.add(clip_id)

        source_asset_id = str(item.get("source_asset_id", "")).strip() or None
        source_path = str(item.get("source_path", "")).strip()
        if source_asset_id:
            if source_asset_id not in raw_index:
                raise ValueError(f"clip plan line {line_number}: unknown source_asset_id={source_asset_id}")
            source_path = raw_index[source_asset_id]["path"]
        if not source_path:
            raise ValueError(f"clip plan line {line_number}: source_asset_id or source_path is required")

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"clip plan line {line_number}: source video not found: {source}")

        start_seconds = _as_non_negative_float(item.get("start_seconds"), f"clip plan line {line_number}: start_seconds")
        end_seconds = _as_non_negative_float(item.get("end_seconds"), f"clip plan line {line_number}: end_seconds")
        if end_seconds <= start_seconds:
            raise ValueError(f"clip plan line {line_number}: end_seconds must be greater than start_seconds")

        output_value = str(item.get("output_path", "")).strip() or f"clips/{clip_id}.mp4"
        output_path = _resolve_under_root(layout["root"], output_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        role = str(item.get("role", "")).strip() or None
        notes = str(item.get("notes", "")).strip() or None
        command = build_ffmpeg_extract_command(
            source_path=source,
            output_path=output_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            overwrite=overwrite,
        )
        commands.append(command)
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_asset_id=source_asset_id,
                source_path=str(source),
                output_path=_display_path(layout["root"], output_path),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=end_seconds - start_seconds,
                role=role,
                notes=notes,
            )
        )

        if not dry_run:
            subprocess.run(command, check=True)

    if not dry_run:
        _write_jsonl(output_manifest, [record.to_dict() for record in records])

    return {
        "plan_path": str(plan_path),
        "dry_run": dry_run,
        "clip_count": len(records),
        "output_manifest_path": str(output_manifest),
        "commands": [" ".join(command) for command in commands],
    }


def plan_detective_event_clips(
    *,
    root: str | Path,
    source_clips_path: str | Path,
    clip_plan_output_path: str | Path | None = None,
    clip_groups_output_path: str | Path | None = None,
    max_source_videos: int = 100,
    segment_seconds: float = 8.0,
    min_clip_seconds: float = 3.0,
    max_clip_seconds: float = 15.0,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    source_clips = list(_load_jsonl(Path(source_clips_path)))
    if not source_clips:
        raise ValueError("source clip manifest is empty")
    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be positive")
    if max_clip_seconds < min_clip_seconds:
        raise ValueError("max_clip_seconds must be >= min_clip_seconds")
    if segment_seconds < min_clip_seconds or segment_seconds > max_clip_seconds:
        raise ValueError("segment_seconds must stay within min/max clip seconds")

    clip_plan_output = Path(clip_plan_output_path) if clip_plan_output_path else layout["metadata"] / DEFAULT_DETECTIVE_CLIP_PLAN_NAME
    clip_groups_output = Path(clip_groups_output_path) if clip_groups_output_path else layout["metadata"] / DEFAULT_CLIP_GROUPS_NAME

    plan_records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    used_source_keys: set[str] = set()
    used_clip_ids: set[str] = set()
    single_segment_records: list[dict[str, Any]] = []
    skipped_count = 0
    probed_count = 0

    for item in source_clips:
        if len(used_source_keys) >= max_source_videos:
            break
        source_path = _source_clip_video_path(layout["root"], item)
        if not source_path.exists():
            skipped_count += 1
            continue
        source_key = str(source_path.resolve())
        if source_key in used_source_keys:
            continue
        used_source_keys.add(source_key)
        media = probe_media(source_path)
        probed_count += 1
        duration = _source_clip_duration_seconds(item, media)
        if duration < min_clip_seconds:
            skipped_count += 1
            continue

        source_clip_id = str(item.get("clip_id", "")).strip() or _stable_hash(source_key)
        dataset = str(item.get("dataset", "unknown")).strip() or "unknown"
        source_group_id = f"group_{dataset}_{_stable_hash(source_key)}"
        segments = _event_segments(
            duration_seconds=duration,
            segment_seconds=segment_seconds,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
        )
        candidate_clip_ids: list[str] = []
        for segment_index, (start_seconds, end_seconds) in enumerate(segments, start=1):
            clip_id = f"{_safe_id(source_clip_id)}__seg_{segment_index:03d}"
            if clip_id in used_clip_ids:
                continue
            used_clip_ids.add(clip_id)
            output_path = f"clips/detective/{dataset}/{clip_id}.mp4"
            record = {
                "clip_id": clip_id,
                "source_path": str(source_path),
                "output_path": output_path,
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "duration_seconds": round(end_seconds - start_seconds, 3),
                "role": "event_clip",
                "notes": "planned by Omni-Detective event segmentation",
                "dataset": dataset,
                "source_clip_id": source_clip_id,
                "group_id": source_group_id,
                "source_row_ids": list(item.get("source_row_ids", [])),
                "text_fields": item.get("text_fields", {}),
                "media_probe": media,
            }
            source_asset_id = str(item.get("source_asset_id", "")).strip()
            if source_asset_id:
                record["source_asset_id"] = source_asset_id
            plan_records.append(record)
            candidate_clip_ids.append(clip_id)

        if len(candidate_clip_ids) >= 2:
            group_records.append(
                {
                    "group_id": source_group_id,
                    "dataset": dataset,
                    "group_reason": "same_source_video",
                    "source_clip_ids": [source_clip_id],
                    "candidate_clip_ids": candidate_clip_ids,
                    "group_tags": _group_tags_from_clip(item),
                    "source_path": _display_source_path(layout["root"], str(source_path)),
                    "media_probe": media,
                }
            )
        elif candidate_clip_ids:
            single_segment_records.append(
                {
                    "dataset": dataset,
                    "clip_id": candidate_clip_ids[0],
                    "source_clip_id": source_clip_id,
                    "tokens": sorted(_group_tokens_from_clip(item)),
                }
            )

    group_records.extend(_semantic_singleton_groups(single_segment_records))
    _write_jsonl(clip_plan_output, plan_records)
    _write_jsonl(clip_groups_output, group_records)
    return {
        "source_clips_path": str(source_clips_path),
        "clip_plan_output_path": str(clip_plan_output),
        "clip_groups_output_path": str(clip_groups_output),
        "source_video_count": len(used_source_keys),
        "probed_count": probed_count,
        "skipped_count": skipped_count,
        "planned_clip_count": len(plan_records),
        "group_count": len(group_records),
        "segment_seconds": segment_seconds,
        "min_clip_seconds": min_clip_seconds,
        "max_clip_seconds": max_clip_seconds,
    }


def annotate_clips(
    *,
    root: str | Path,
    clips_manifest_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    concurrency: int = 1,
) -> dict[str, Any]:
    return _annotate_clips_impl(
        root=root,
        clips_manifest_path=clips_manifest_path,
        output_path=output_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        overwrite=overwrite,
        detective=False,
        concurrency=concurrency,
    )


def detective_annotate_clips(
    *,
    root: str | Path,
    clips_manifest_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    concurrency: int = 1,
) -> dict[str, Any]:
    return _annotate_clips_impl(
        root=root,
        clips_manifest_path=clips_manifest_path,
        output_path=output_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        overwrite=overwrite,
        detective=True,
        concurrency=concurrency,
    )


def _annotate_clips_impl(
    *,
    root: str | Path,
    clips_manifest_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None,
    overwrite: bool,
    timeout_seconds: float,
    detective: bool,
    concurrency: int,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    manifest_path = Path(clips_manifest_path)
    clips = list(_load_jsonl(manifest_path))
    if not clips:
        raise ValueError("clip manifest is empty")

    output = Path(output_path) if output_path else layout["captions"] / DEFAULT_CLIP_ANNOTATIONS_NAME
    # Long Omni annotation runs must be restartable.  Even when callers pass
    # overwrite=True for a fresh run, keep already written records as a resume
    # cache; delete the output file explicitly to force a full re-annotation.
    existing_records = _load_records_by_key(output, "clip_id")
    if not output.exists():
        _write_jsonl(output, [])
    concurrency = max(1, int(concurrency or 1))

    def annotate_one(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        local_client = OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        clip_id = str(item.get("clip_id", "")).strip()
        clip_path = _resolve_under_root(layout["root"], str(item.get("output_path", "")).strip())
        if not clip_path.exists():
            raise FileNotFoundError(f"clip output does not exist: {clip_path}")

        fallback_reason = ""
        detective_fallback_reason = ""
        detective_fallback_used = False
        detective_to_single_pass = False
        raw_model_output: dict[str, Any] = {}
        if detective:
            tool_observations = _build_toolbox_observations(clip_path)
            try:
                normalized, raw_model_output = local_client.annotate_clip_detective(
                    clip_path=str(clip_path),
                    tool_observations=tool_observations,
                )
                fallback_used = False
            except Exception as detective_exc:
                detective_fallback_used = True
                detective_fallback_reason = "detective_to_single_pass"
                try:
                    normalized, single_pass_output = local_client.annotate_clip(clip_path=str(clip_path))
                    raw_model_output = {
                        "detective_error": f"{type(detective_exc).__name__}: {detective_exc}",
                        "single_pass_fallback": single_pass_output,
                    }
                    normalized["storyline"] = []
                    normalized["visible_text"] = []
                    normalized["speakers_and_transcript"] = []
                    normalized["detective_notes"] = ["detective annotation failed; used single-pass annotation"]
                    normalized["detective_trajectory"] = [
                        *tool_observations,
                        {"stage": "detective_error", "error": raw_model_output["detective_error"]},
                        {"stage": "single_pass_fallback", "payload": single_pass_output},
                    ]
                    normalized["uncertainties"] = ["detective annotation failed; used single-pass annotation"]
                    fallback_used = False
                    detective_to_single_pass = True
                except Exception as single_pass_exc:
                    normalized = _fallback_clip_annotation()
                    raw_model_output = {
                        "detective_error": f"{type(detective_exc).__name__}: {detective_exc}",
                        "single_pass_error": f"{type(single_pass_exc).__name__}: {single_pass_exc}",
                    }
                    fallback_used = True
                    fallback_reason = "annotation_fallback"
                    detective_fallback_reason = "detective_and_single_pass_failed"
        else:
            try:
                normalized, raw_model_output = local_client.annotate_clip(clip_path=str(clip_path))
                fallback_used = False
            except Exception as exc:
                normalized = _fallback_clip_annotation()
                raw_model_output = {"error": f"{type(exc).__name__}: {exc}"}
                fallback_used = True
                fallback_reason = "annotation_fallback"

        record = {
            "clip_id": clip_id,
            "output_path": _display_path(layout["root"], clip_path),
            "summary": normalized["summary"],
            "subjects": list(normalized["subjects"]),
            "object_counts": dict(normalized["object_counts"]),
            "actions": list(normalized["actions"]),
            "scene": normalized["scene"],
            "attributes": list(normalized["attributes"]),
            "on_screen_text": list(normalized["on_screen_text"]),
            "speech": list(normalized["speech"]),
            "audio_events": list(normalized["audio_events"]),
            "modalities": list(normalized["modalities"]),
            "source_asset_id": str(item.get("source_asset_id", "")).strip() or None,
            "fallback_used": fallback_used,
            "raw_model_output": raw_model_output,
        }
        if detective:
            record.update(
                {
                    "storyline": list(normalized.get("storyline", [])),
                    "visible_text": list(normalized.get("visible_text", [])),
                    "speakers_and_transcript": list(normalized.get("speakers_and_transcript", [])),
                    "detective_notes": list(normalized.get("detective_notes", [])),
                    "detective_trajectory": list(normalized.get("detective_trajectory", [])),
                    "uncertainties": list(normalized.get("uncertainties", [])),
                    "detective_fallback_used": detective_fallback_used,
                }
            )
            if detective_fallback_reason:
                record["detective_fallback_reason"] = detective_fallback_reason
        record.update(_clip_manifest_metadata(item=item, root=layout["root"]))
        if fallback_reason:
            record["fallback_reason"] = fallback_reason
        return record, detective_to_single_pass

    records_by_clip_id: dict[str, dict[str, Any]] = {}
    pending_items: list[dict[str, Any]] = []
    annotated_count = 0
    reused_count = 0
    detective_to_single_pass_count = 0
    for item in clips:
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError("clip manifest contains an entry without clip_id")

        if clip_id in existing_records:
            records_by_clip_id[clip_id] = existing_records[clip_id]
            reused_count += 1
        else:
            pending_items.append(item)

    if concurrency <= 1:
        for item in pending_items:
            record, detective_to_single_pass = annotate_one(item)
            records_by_clip_id[str(record["clip_id"])] = record
            annotated_count += 1
            if detective_to_single_pass:
                detective_to_single_pass_count += 1
            _append_jsonl_record(output, record)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(annotate_one, item) for item in pending_items]
            for future in as_completed(futures):
                record, detective_to_single_pass = future.result()
                records_by_clip_id[str(record["clip_id"])] = record
                annotated_count += 1
                if detective_to_single_pass:
                    detective_to_single_pass_count += 1
                _append_jsonl_record(output, record)

    output_records: list[dict[str, Any]] = []
    for item in clips:
        clip_id = str(item.get("clip_id", "")).strip()
        record = records_by_clip_id[clip_id]
        output_records.append(record)

    fallback_count = 0
    for record in output_records:
        if bool(record.get("fallback_used")):
            fallback_count += 1

    _write_jsonl(output, output_records)
    return {
        "clips_manifest_path": str(manifest_path),
        "output_path": str(output),
        "clip_count": len(output_records),
        "annotated_count": annotated_count,
        "reused_count": reused_count,
        "fallback_count": fallback_count,
        "annotation_mode": "detective" if detective else "single_pass",
        "detective_to_single_pass_count": detective_to_single_pass_count if detective else 0,
        "concurrency": concurrency,
    }


def _clip_manifest_metadata(*, item: dict[str, Any], root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    dataset = str(item.get("dataset", "")).strip()
    if dataset:
        metadata["dataset"] = dataset

    source_row_ids = [str(value).strip() for value in item.get("source_row_ids", []) if str(value).strip()]
    if source_row_ids:
        metadata["source_row_ids"] = source_row_ids

    text_fields = item.get("text_fields")
    if isinstance(text_fields, dict) and text_fields:
        metadata["text_fields"] = text_fields

    source_path = str(item.get("source_path", "")).strip()
    if source_path:
        metadata["source_path"] = _display_source_path(root, source_path)

    clip_timing: dict[str, Any] = {}
    for field_name in ("start_seconds", "end_seconds", "duration_seconds"):
        if field_name in item:
            try:
                clip_timing[field_name] = round(float(item[field_name]), 3)
            except (TypeError, ValueError):
                continue
    role = str(item.get("role", "")).strip()
    notes = str(item.get("notes", "")).strip()
    if role:
        clip_timing["role"] = role
    if notes:
        clip_timing["notes"] = notes
    if clip_timing:
        metadata["source_clip"] = clip_timing
    return metadata


def _display_source_path(root: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)
    return raw_path


def propose_pairs(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    raw_index_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    annotations_path = Path(clip_annotations_path)
    annotations = list(_load_jsonl(annotations_path))
    if not annotations:
        raise ValueError("clip annotations are empty")

    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_PAIR_PROPOSALS_NAME
    existing_records = {} if overwrite else _load_records_by_key(output, "proposal_id")
    raw_index = _load_raw_asset_index(Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME)
    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

    candidates = _build_pair_candidates(root=layout["root"], annotations=annotations)
    output_records: list[dict[str, Any]] = []
    proposed_count = 0
    reused_count = 0
    fallback_count = 0
    for candidate in candidates:
        proposal_id = candidate["proposal_id"]
        if proposal_id in existing_records:
            record = existing_records[proposal_id]
            reused_count += 1
        else:
            reference_annotation = candidate["reference_annotation"]
            target_annotation = candidate["target_annotation"]
            raw_model_output: dict[str, Any] = {}
            try:
                model_fields, raw_model_output = client.propose_pair(
                    reference_annotation=_annotation_prompt_view(reference_annotation),
                    target_annotation=_annotation_prompt_view(target_annotation),
                    hard_negative_candidates=[
                        _annotation_prompt_view(annotation) for annotation in candidate["hard_negative_annotations"]
                    ],
                )
                fallback_used = False
            except Exception as exc:
                model_fields = _fallback_pair_model_fields(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    primary_difference=candidate["primary_difference"],
                )
                raw_model_output = {"error": f"{type(exc).__name__}: {exc}"}
                fallback_used = True

            candidate, model_fields, direction_corrected = _maybe_reorient_candidate_for_model_fields(
                root=layout["root"],
                candidate=candidate,
                model_fields=model_fields,
                annotations=annotations,
            )
            if direction_corrected:
                proposal_id = candidate["proposal_id"]
                reference_annotation = candidate["reference_annotation"]
                target_annotation = candidate["target_annotation"]
            source = _build_source_metadata(
                root=layout["root"],
                target_annotation=target_annotation,
                raw_index=raw_index,
            )
            record = {
                "proposal_id": proposal_id,
                "reference_video": reference_annotation["output_path"],
                "target_video": target_annotation["output_path"],
                "edit_text": model_fields["edit_text"],
                "modalities": list(model_fields["modalities"]),
                "reference_caption": model_fields["reference_caption"],
                "target_caption": model_fields["target_caption"],
                "difference": model_fields["difference"],
                "hard_negatives": list(candidate["hard_negative_paths"]),
                "quality": {
                    "same_context_score": candidate["quality"]["same_context_score"],
                    "edit_match_score": candidate["quality"]["edit_match_score"],
                    "target_uniqueness_score": candidate["quality"]["target_uniqueness_score"],
                },
                "source_context": dict(candidate["source_context"]),
                "source": source,
                "proposal_reason": model_fields["proposal_reason"],
                "direction_corrected": direction_corrected,
                "fallback_used": fallback_used,
                "raw_model_output": raw_model_output,
            }
            proposed_count += 1

        if bool(record.get("fallback_used")):
            fallback_count += 1
        output_records.append(record)

    _write_jsonl(output, output_records)
    return {
        "clip_annotations_path": str(annotations_path),
        "output_path": str(output),
        "candidate_count": len(candidates),
        "proposal_count": len(output_records),
        "proposed_count": proposed_count,
        "reused_count": reused_count,
        "fallback_count": fallback_count,
    }


def propose_group_pairs(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    clip_groups_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    accepted_output_path: str | Path | None = None,
    raw_index_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    max_accepted_pairs: int = 10,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    annotations_path = Path(clip_annotations_path)
    groups_path = Path(clip_groups_path)
    annotations = list(_load_jsonl(annotations_path))
    groups = list(_load_jsonl(groups_path))
    if not annotations:
        raise ValueError("clip annotations are empty")
    if not groups:
        raise ValueError("clip groups are empty")

    output = Path(output_path) if output_path else layout["pairs"] / "judged_pair_proposals.jsonl"
    accepted_output = Path(accepted_output_path) if accepted_output_path else layout["pairs"] / DEFAULT_ACCEPTED_PAIRS_NAME
    existing_records = {} if overwrite else _load_records_by_key(output, "proposal_id")
    raw_index = _load_raw_asset_index(Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME)
    annotations_by_id = {str(item.get("clip_id", "")).strip(): item for item in annotations if str(item.get("clip_id", "")).strip()}
    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

    output_records: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    candidate_count = 0
    proposed_count = 0
    reused_count = 0
    fallback_count = 0
    rejected_count = 0
    accepted_total_count = 0
    seen_proposal_ids: set[str] = set()

    for group in groups:
        group_metadata = {
            "group_id": str(group.get("group_id", "")).strip(),
            "group_reason": str(group.get("group_reason", "")).strip(),
        }
        candidate_clip_ids = [str(value).strip() for value in group.get("candidate_clip_ids", []) if str(value).strip()]
        group_annotations = [
            annotations_by_id[clip_id]
            for clip_id in candidate_clip_ids
            if clip_id in annotations_by_id and not bool(annotations_by_id[clip_id].get("fallback_used"))
        ]
        if len(group_annotations) < 4:
            continue
        candidates = _build_pair_candidates(root=layout["root"], annotations=group_annotations)
        candidate_count += len(candidates)
        for candidate in candidates:
            proposal_id = candidate["proposal_id"]
            if proposal_id in seen_proposal_ids:
                continue
            seen_proposal_ids.add(proposal_id)
            reference_annotation = candidate["reference_annotation"]
            target_annotation = candidate["target_annotation"]
            if proposal_id in existing_records:
                record = existing_records[proposal_id]
                reused_count += 1
            else:
                raw_model_output: dict[str, Any] = {}
                judge_raw_output: dict[str, Any] = {}
                verification_raw_output: dict[str, Any] = {}
                try:
                    model_fields, raw_model_output = client.propose_pair(
                        reference_annotation=_annotation_prompt_view(reference_annotation),
                        target_annotation=_annotation_prompt_view(target_annotation),
                        hard_negative_candidates=[
                            _annotation_prompt_view(annotation) for annotation in candidate["hard_negative_annotations"]
                        ],
                        heuristic_pair={
                            "primary_difference": dict(candidate["primary_difference"]),
                            "changed_difference_types": list(candidate["changed_difference_types"]),
                            "heuristic_quality": dict(candidate["quality"]),
                            "source_context": dict(candidate["source_context"]),
                        },
                    )
                    proposal_fallback_used = False
                except Exception as exc:
                    model_fields = _fallback_pair_model_fields(
                        reference_annotation=reference_annotation,
                        target_annotation=target_annotation,
                        primary_difference=candidate["primary_difference"],
                    )
                    raw_model_output = {"error": f"{type(exc).__name__}: {exc}"}
                    proposal_fallback_used = True

                model_fields = _repair_pair_model_fields(
                    model_fields=model_fields,
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                )
                direction_corrected = False
                oriented_candidate, model_fields, direction_corrected = _maybe_reorient_candidate_for_model_fields(
                    root=layout["root"],
                    candidate=candidate,
                    model_fields=model_fields,
                    annotations=group_annotations,
                )
                if direction_corrected:
                    seen_proposal_ids.discard(proposal_id)
                    proposal_id = oriented_candidate["proposal_id"]
                    if proposal_id in seen_proposal_ids:
                        continue
                    seen_proposal_ids.add(proposal_id)
                    candidate = oriented_candidate
                    reference_annotation = candidate["reference_annotation"]
                    target_annotation = candidate["target_annotation"]
                source = _build_source_metadata(
                    root=layout["root"],
                    target_annotation=target_annotation,
                    raw_index=raw_index,
                )
                proposal_quality = _quality_for_model_fields(
                    base_quality=candidate["quality"],
                    model_fields=model_fields,
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                )
                edit_text_quality = _edit_text_quality_payload(
                    edit_text=model_fields["edit_text"],
                    difference=model_fields["difference"],
                    modalities=model_fields["modalities"],
                    reference_caption=model_fields["reference_caption"],
                    target_caption=model_fields["target_caption"],
                )
                observable_difference = _observable_difference_gate(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    difference=model_fields["difference"],
                    visual_near_duplicate_score=proposal_quality.get("visual_near_duplicate_score"),
                )
                _apply_structured_gate_quality(
                    proposal_quality,
                    edit_text_quality=edit_text_quality,
                    observable_difference=observable_difference,
                )
                proposal_difference_evidence = _difference_evidence_from_annotations(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    primary_difference=model_fields["difference"],
                )
                proposal_view = {
                    "proposal_id": proposal_id,
                    "edit_text": model_fields["edit_text"],
                    "modalities": list(model_fields["modalities"]),
                    "reference_caption": model_fields["reference_caption"],
                    "target_caption": model_fields["target_caption"],
                    "difference": model_fields["difference"],
                    "quality": dict(proposal_quality),
                    "heuristic_primary_difference": dict(candidate["primary_difference"]),
                    "changed_difference_types": list(candidate["changed_difference_types"]),
                    "source_context": dict(candidate["source_context"]),
                    "difference_evidence": dict(proposal_difference_evidence),
                    "edit_text_quality": dict(edit_text_quality),
                    "observable_difference": dict(observable_difference),
                    "acceptance_thresholds": {
                        "same_context_score": MIN_ACCEPT_SAME_CONTEXT_SCORE,
                        "edit_match_score": MIN_ACCEPT_EDIT_MATCH_SCORE,
                        "target_uniqueness_score": MIN_ACCEPT_TARGET_UNIQUENESS_SCORE,
                        "difference_strength_score": MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE,
                        "action_evidence_score_for_action_edits": MIN_ACCEPT_ACTION_EVIDENCE_SCORE,
                        "speech_evidence_score_for_speech_edits": MIN_ACCEPT_SPEECH_EVIDENCE_SCORE,
                        "speech_specificity_score_for_speech_edits": MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE,
                        "non_speech_audio_event_score_for_audio_event_edits": MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE,
                        "max_visual_near_duplicate_score_for_visual_edits": MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE,
                    },
                }
                try:
                    judge, judge_raw_output = client.judge_pair(
                        proposal=proposal_view,
                        reference_annotation=_annotation_prompt_view(reference_annotation),
                        target_annotation=_annotation_prompt_view(target_annotation),
                        hard_negative_candidates=[
                            _annotation_prompt_view(annotation) for annotation in candidate["hard_negative_annotations"]
                        ],
                    )
                    judge_fallback_used = False
                except Exception as exc:
                    judge = _fallback_pair_judge(candidate["quality"], reason=f"{type(exc).__name__}: {exc}")
                    judge_raw_output = {"error": f"{type(exc).__name__}: {exc}"}
                    judge_fallback_used = True

                try:
                    (
                        verification,
                        verification_raw_output,
                        verification_context_retry_used,
                    ) = _verify_pair_difference_with_context_retry(
                        client,
                        proposal=proposal_view,
                        reference_annotation=_annotation_prompt_view(reference_annotation),
                        target_annotation=_annotation_prompt_view(target_annotation),
                        reference_clip_path=str(_resolve_under_root(layout["root"], reference_annotation["output_path"])),
                        target_clip_path=str(_resolve_under_root(layout["root"], target_annotation["output_path"])),
                    )
                    verification_fallback_used = False
                except Exception as exc:
                    verification = _fallback_pair_verification(reason=f"{type(exc).__name__}: {exc}")
                    verification_raw_output = {"error": f"{type(exc).__name__}: {exc}"}
                    verification_context_retry_used = False
                    verification_fallback_used = True

                judge = _finalize_pair_judge(judge)
                verification = _finalize_pair_verification(verification)
                fallback_used = proposal_fallback_used or judge_fallback_used or verification_fallback_used
                effective_quality = _effective_pair_quality(judge, verification, proposal_quality)
                accepted = _judge_accepts(judge, verification, effective_quality)
                if not accepted:
                    judge["reject_reason"] = _compose_reject_reason(judge, verification, effective_quality)
                speech_quality = _speech_quality_payload(effective_quality)
                audio_event_quality = _audio_event_quality_payload(effective_quality)
                record = {
                    "proposal_id": proposal_id,
                    "group_id": group_metadata["group_id"],
                    "group_reason": group_metadata["group_reason"],
                    "reference_clip_id": reference_annotation.get("clip_id", ""),
                    "target_clip_id": target_annotation.get("clip_id", ""),
                    "reference_video": reference_annotation["output_path"],
                    "target_video": target_annotation["output_path"],
                    "edit_text": model_fields["edit_text"],
                    "modalities": list(model_fields["modalities"]),
                    "reference_caption": model_fields["reference_caption"],
                    "target_caption": model_fields["target_caption"],
                    "difference": model_fields["difference"],
                    "hard_negatives": list(candidate["hard_negative_paths"]),
                    "judge_quality": {
                        "same_context_score": judge["same_context_score"],
                        "edit_match_score": judge["edit_match_score"],
                        "target_uniqueness_score": judge["target_uniqueness_score"],
                    },
                    "quality": effective_quality,
                    "heuristic_quality": dict(proposal_quality),
                    "source_context": dict(candidate["source_context"]),
                    "source": source,
                    "proposal_reason": model_fields["proposal_reason"],
                    "direction_corrected": direction_corrected,
                    "evidence": _evidence_from_annotations(
                        reference_annotation,
                        target_annotation,
                        difference_evidence=proposal_difference_evidence,
                    ),
                    "judge": judge,
                    "verification": verification,
                    "speech_quality": speech_quality,
                    "audio_event_quality": audio_event_quality,
                    "edit_text_quality": edit_text_quality,
                    "observable_difference": observable_difference,
                    "transcript_backed": speech_quality.get("transcript_backed"),
                    "accepted": accepted,
                    "fallback_used": fallback_used,
                    "raw_model_output": raw_model_output,
                    "raw_judge_output": judge_raw_output,
                    "raw_verification_output": verification_raw_output,
                    "verification_annotation_only_retry_used": verification_context_retry_used,
                }
                proposed_count += 1

            if "verification" not in record:
                record = dict(record)
                record["verification"] = _fallback_pair_verification(reason="existing record has no verification")
                record["accepted"] = False
                record["fallback_used"] = True
                judge = dict(record.get("judge", {}))
                judge["reject_reason"] = _compose_reject_reason(judge, record["verification"], record.get("quality"))
                record["judge"] = judge
            record = _prepare_record_for_acceptance(
                record,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            judge = dict(record.get("judge", {}))
            verification = record.get("verification", {})
            quality = record.get("quality", {})
            record["accepted"] = _judge_accepts(judge, verification, quality)
            if not bool(record.get("accepted")):
                judge["accept"] = False
                judge["reject_reason"] = _compose_reject_reason(judge, verification, quality)
                record["judge"] = judge
            acceptance_issues = _pair_record_acceptance_issues(
                root=layout["root"],
                record=record,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            if acceptance_issues:
                record = _reject_record_with_acceptance_issues(record, acceptance_issues)
                quality = dict(record.get("quality", {}))
                if any("single clip" in issue for issue in acceptance_issues):
                    quality["intraclip_change_conflict"] = 1.0
                record["quality"] = quality
            if bool(record.get("fallback_used")):
                fallback_count += 1
            if bool(record.get("accepted")):
                accepted_total_count += 1
            else:
                rejected_count += 1
            output_records.append(record)

    accepted_records = _select_final_accepted_records(output_records, max_accepted_pairs=max_accepted_pairs)
    _write_jsonl(output, output_records)
    _write_jsonl(accepted_output, accepted_records)
    verification_counts = _pair_verification_counts(output_records)
    return {
        "clip_annotations_path": str(annotations_path),
        "clip_groups_path": str(groups_path),
        "output_path": str(output),
        "accepted_output_path": str(accepted_output),
        "group_count": len(groups),
        "candidate_count": candidate_count,
        "proposal_count": len(output_records),
        "accepted_count": len(accepted_records),
        "accepted_total_count": accepted_total_count,
        "rejected_count": rejected_count,
        "proposed_count": proposed_count,
        "reused_count": reused_count,
        "fallback_count": fallback_count,
        "verification_counts": verification_counts,
        "thresholds": {
            "same_context_score": MIN_ACCEPT_SAME_CONTEXT_SCORE,
            "edit_match_score": MIN_ACCEPT_EDIT_MATCH_SCORE,
            "target_uniqueness_score": MIN_ACCEPT_TARGET_UNIQUENESS_SCORE,
            "edit_necessity_score": MIN_ACCEPT_EDIT_NECESSITY_SCORE,
            "edit_target_alignment_score": MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE,
            "difference_strength_score": MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE,
            "action_evidence_score_for_action_edits": MIN_ACCEPT_ACTION_EVIDENCE_SCORE,
            "speech_evidence_score_for_speech_edits": MIN_ACCEPT_SPEECH_EVIDENCE_SCORE,
            "speech_specificity_score_for_speech_edits": MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE,
            "non_speech_audio_event_score_for_audio_event_edits": MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE,
        },
    }


def plan_video_edits(
    *,
    root: str | Path,
    pair_candidates_path: str | Path,
    clip_annotations_path: str | Path,
    output_path: str | Path | None = None,
    max_plans: int = 10,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    model: str | None = None,
    timeout_seconds: float = 180.0,
    planning_mode: str = "production",
    planner_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    planning_mode = str(planning_mode).strip() or "production"
    if planning_mode not in {"production", "exploration"}:
        raise ValueError("planning_mode must be 'production' or 'exploration'")
    layout = ensure_layout(root)
    candidates = list(_load_jsonl(Path(pair_candidates_path)))
    original_candidate_count = len(candidates)
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    if not candidates:
        raise ValueError("pair candidates file is empty")
    if not annotations:
        raise ValueError("clip annotations are empty")

    annotation_lookup = _annotation_lookup(root=layout["root"], annotations=annotations)
    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_VIDEO_EDIT_PLAN_NAME
    cache_output = Path(planner_cache_path) if planner_cache_path else output.with_name(DEFAULT_VIDEO_EDIT_PLANNER_CACHE_NAME)
    planner_cache = _load_video_edit_planner_cache(cache_output)
    planner_cache_dirty = False
    planner_client = (
        OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        if base_url and model
        else None
    )

    plans: list[dict[str, Any]] = []
    skipped_by_type: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    cache_hits = 0
    cache_misses = 0
    if planning_mode == "exploration":
        exploration_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            reference_video = str(candidate.get("reference_video", "")).strip()
            if not reference_video:
                skipped_by_type["unknown"] += 1
                skipped_reasons["missing_reference_video"] += 1
                continue
            reference_annotation = _annotation_for_video_edit_plan(
                root=layout["root"],
                lookup=annotation_lookup,
                record=candidate,
                video_field="reference_video",
                caption_field="reference_caption",
            )
            if not _annotation_is_usable_for_reference_understanding(reference_annotation):
                skipped_reasons["reference_annotation_unusable"] += 1
                continue
            generated = _video_edit_exploration_candidates(candidate, reference_annotation)
            if generated:
                exploration_candidates.extend(generated)
                skipped_reasons["exploration_ideation_from_reference"] += len(generated)
            else:
                skipped_reasons["exploration_no_suitable_reference_edit"] += 1
        candidates = exploration_candidates

    seen_sources: set[str] = set()
    seen_plan_keys: set[tuple[str, str, str, str, str]] = set()
    for candidate in candidates:
        if len(plans) >= max_plans:
            break
        difference = dict(candidate.get("difference") or {})
        difference_type = str(difference.get("type", "")).strip()
        route = _video_edit_model_route(difference_type)
        safe_visual_ideation_used = False

        reference_video = str(candidate.get("reference_video", "")).strip()
        if not reference_video:
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons["missing_reference_video"] += 1
            continue
        if planning_mode != "exploration" and reference_video in seen_sources:
            skipped_reasons["duplicate_reference_video"] += 1
            continue

        reference_annotation = _annotation_for_video_edit_plan(
            root=layout["root"],
            lookup=annotation_lookup,
            record=candidate,
            video_field="reference_video",
            caption_field="reference_caption",
        )
        if not _annotation_is_usable_for_reference_understanding(reference_annotation):
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons["reference_annotation_unusable"] += 1
            continue
        if route not in {None, "vace_controlled"} and planner_client is not None:
            ideation_candidate = _safe_visual_ideation_candidate(candidate, reference_annotation)
            if ideation_candidate is not None:
                candidate = ideation_candidate
                difference = dict(candidate.get("difference") or {})
                difference_type = str(difference.get("type", "")).strip()
                route = _video_edit_model_route(difference_type)
                safe_visual_ideation_used = True
                skipped_reasons["safe_visual_ideation_from_non_vace_candidate"] += 1
        if route is None:
            ideation_candidate = (
                _safe_visual_ideation_candidate(candidate, reference_annotation)
                if planner_client is not None
                else None
            )
            if ideation_candidate is None:
                skipped_by_type[difference_type or "unknown"] += 1
                skipped_reasons["unsupported_difference_type"] += 1
                continue
            candidate = ideation_candidate
            difference = dict(candidate.get("difference") or {})
            difference_type = str(difference.get("type", "")).strip()
            route = _video_edit_model_route(difference_type)
            safe_visual_ideation_used = True
            skipped_reasons["safe_visual_ideation_from_unsupported_type"] += 1
            if route is None:
                skipped_by_type[difference_type or "unknown"] += 1
                skipped_reasons["unsupported_difference_type"] += 1
                continue
        risk = _video_edit_risk_assessment(reference_annotation, difference_type=difference_type)
        if planning_mode == "exploration":
            risk = _relax_visual_exploration_risk(risk, candidate)
        if safe_visual_ideation_used:
            risk = _relax_safe_visual_ideation_risk(risk, candidate)
        if not risk["allow_generation"]:
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons[f"high_risk_reference_{risk['risk_level']}"] += 1
            for reason in risk["risk_reasons"]:
                skipped_reasons[f"risk_{reason}"] += 1
            continue
        edit_text = str(candidate.get("edit_text", "")).strip() or _build_fallback_edit_text(difference)
        edit_token = str(candidate.get("suggested_edit_token", "")).strip() or _video_edit_token(difference, edit_text)
        edit_region = str(candidate.get("suggested_edit_region", "")).strip() or _video_edit_region(edit_text, difference, reference_annotation, route)
        if difference_type in {"object_presence", "object_count"} and (not edit_token or not edit_region):
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons["missing_object_edit_token_or_region"] += 1
            continue

        source_prompt = _video_edit_source_prompt(reference_annotation, candidate)
        target_prompt = _video_edit_target_prompt(
            source_prompt=source_prompt,
            edit_text=edit_text,
            difference=difference,
        )
        preserve_tokens = _video_edit_preserve_tokens(reference_annotation, difference, edit_token)
        negative_prompt = _video_edit_negative_prompt(preserve_tokens, risk=risk)
        planner_metadata: dict[str, Any] = {
            "stage": "heuristic_prompt_planner",
            "input": "annotation_and_candidate_edit",
            "output": "source_prompt_target_prompt_edit_token_preserve_constraints",
            "fallback_used": True,
        }
        raw_planner_output: dict[str, Any] = {}
        planned_mask_query = str(candidate.get("suggested_mask_query", "")).strip()
        planned_preserve_regions: list[str] = [
            str(item).strip()
            for item in candidate.get("suggested_preserve_regions", [])
            if str(item).strip()
        ] if isinstance(candidate.get("suggested_preserve_regions"), list) else []
        planner_input = {
            "edit_text": edit_text,
            "difference": difference,
            "reference_video": reference_video,
            "reference_caption": str(candidate.get("reference_caption", "")).strip(),
            "model_route_hint": route,
            "planning_mode": planning_mode,
            "exploration_family": str(candidate.get("exploration_family", "")).strip(),
        }
        planner_cache_key = _video_edit_planner_cache_key(
            model=model,
            planning_mode=planning_mode,
            route=route,
            reference_video=reference_video,
            reference_annotation=reference_annotation,
            candidate=planner_input,
        )
        cached_planner_record = planner_cache.get(planner_cache_key)
        if cached_planner_record:
            cache_hits += 1
        elif planner_client is not None:
            cache_misses += 1
        if planner_client is not None or cached_planner_record is not None:
            try:
                if cached_planner_record is not None:
                    planned = dict(cached_planner_record.get("planned", {}))
                    raw_planner_output = dict(cached_planner_record.get("raw_planner_output", {}))
                else:
                    planned, raw_planner_output = planner_client.plan_video_edit(
                        reference_clip_path=str(_resolve_under_root(layout["root"], reference_video)),
                        reference_annotation=_annotation_prompt_view(reference_annotation),
                        candidate=planner_input,
                        route_hint=route,
                    )
                    planner_cache[planner_cache_key] = {
                        "cache_key": planner_cache_key,
                        "model": model,
                        "planning_mode": planning_mode,
                        "route": route,
                        "reference_video": reference_video,
                        "candidate": planner_input,
                        "planned": planned,
                        "raw_planner_output": raw_planner_output,
                    }
                    planner_cache_dirty = True
                if not bool(planned.get("should_generate")):
                    skipped_by_type[difference_type or "unknown"] += 1
                    skipped_reasons["model_planner_rejected"] += 1
                    continue
                planned_edit_text = str(planned.get("edit_text", "")).strip()
                if planned_edit_text:
                    edit_text = planned_edit_text
                planned_difference = planned.get("difference")
                if isinstance(planned_difference, dict) and str(planned_difference.get("type", "")).strip():
                    planned_difference = _normalize_model_planned_visual_difference(
                        dict(planned_difference),
                        edit_text=edit_text,
                    )
                    planned_difference_type = str(planned_difference.get("type", "")).strip()
                    planned_difference_route = _video_edit_model_route(planned_difference_type)
                    if planned_difference_route is None:
                        skipped_by_type[planned_difference_type or "unknown"] += 1
                        skipped_reasons["model_planner_revised_to_unsupported_difference_type"] += 1
                        continue
                    difference = dict(planned_difference)
                    difference_type = planned_difference_type
                    route = planned_difference_route
                    risk = _video_edit_risk_assessment(reference_annotation, difference_type=difference_type)
                    if safe_visual_ideation_used:
                        risk = _relax_safe_visual_ideation_risk(risk, candidate)
                    if not risk["allow_generation"]:
                        skipped_by_type[difference_type or "unknown"] += 1
                        skipped_reasons[f"model_planner_revised_to_high_risk_{risk['risk_level']}"] += 1
                        for reason in risk["risk_reasons"]:
                            skipped_reasons[f"risk_{reason}"] += 1
                        continue
                source_prompt = str(planned["source_prompt"]).strip()
                target_prompt = str(planned["target_prompt"]).strip()
                edit_token = str(planned["edit_token"]).strip()
                preserve_tokens = [str(item).strip() for item in planned["preserve_tokens"] if str(item).strip()]
                negative_prompt = str(planned["negative_prompt"]).strip()
                negative_prompt = _merge_video_edit_locks(negative_prompt, risk)
                edit_region = str(planned["edit_region"]).strip()
                planned_mask_query = str(planned.get("mask_query", "")).strip()
                planned_preserve_regions = [
                    str(item).strip()
                    for item in planned.get("preserve_regions", [])
                    if str(item).strip()
                ]
                planned_route = str(planned.get("model_route", "")).strip()
                if planned_route in SYNTHETIC_VISUAL_ROUTES and _planned_route_matches_difference(
                    planned_route,
                    difference_type,
                ):
                    route = planned_route
                planner_metadata = {
                    "stage": "strongest_omni_prompt_planner",
                    "input": "short_clip_reference_video",
                    "output": "source_prompt_target_prompt_edit_token_preserve_constraints",
                    "fallback_used": False,
                    "cache_hit": cached_planner_record is not None,
                    "model": model,
                    "reason": str(planned.get("reason", "")).strip(),
                    "repaired_fields": list(planned.get("repaired_fields", [])),
                }
            except Exception as exc:
                raw_planner_output = {"error": f"{type(exc).__name__}: {exc}"}
                skipped_reasons["model_planner_fallback"] += 1
                planner_metadata = {
                    "stage": "heuristic_prompt_planner",
                    "input": "annotation_and_candidate_edit",
                    "output": "source_prompt_target_prompt_edit_token_preserve_constraints",
                    "fallback_used": True,
                    "model": model,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                }
        suitability = _video_edit_route_suitability(
            route=route,
            difference=difference,
            edit_text=edit_text,
            edit_token=edit_token,
            edit_region=edit_region,
            reference_annotation=reference_annotation,
        )
        if not suitability["allow_generation"]:
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons[str(suitability["reason"])] += 1
            continue
        default_mask_query = _video_mask_query(
            difference=difference,
            edit_text=edit_text,
            edit_token=edit_token,
            edit_region=edit_region,
            route=route,
            suitability=suitability,
            reference_annotation=reference_annotation,
        )
        mask_query = default_mask_query
        if planned_mask_query:
            if not (
                str(difference.get("type", "")).strip() == "scene"
                and _normalized_phrase(planned_mask_query) == "background"
            ):
                mask_query = planned_mask_query
        target_prompt, preserve_tokens, negative_prompt, prompt_repairs = _repair_video_edit_prompt_contract(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_text=edit_text,
            difference=difference,
            edit_token=edit_token,
            preserve_tokens=preserve_tokens,
            negative_prompt=negative_prompt,
            mask_query=mask_query,
            risk=risk,
        )
        if prompt_repairs:
            planner_metadata = dict(planner_metadata)
            repaired_fields = list(planner_metadata.get("repaired_fields", []))
            repaired_fields.extend(prompt_repairs)
            planner_metadata["repaired_fields"] = sorted(set(repaired_fields))
            planner_metadata["post_lint_repaired"] = True
        plan_lint = _video_edit_plan_lint(
            target_prompt=target_prompt,
            edit_text=edit_text,
            difference=difference,
            preserve_tokens=preserve_tokens,
            negative_prompt=negative_prompt,
            reference_annotation=reference_annotation,
        )
        if not plan_lint["passed"]:
            skipped_by_type[difference_type or "unknown"] += 1
            for reason in plan_lint["errors"]:
                skipped_reasons[f"plan_lint_{reason}"] += 1
            continue
        plan_key = (
            reference_video,
            str(difference.get("type", "")).strip(),
            _normalized_phrase(str(difference.get("from", "")).strip()),
            _normalized_phrase(str(difference.get("to", "")).strip()),
            _normalized_phrase(edit_text),
        )
        if plan_key in seen_plan_keys:
            skipped_reasons["duplicate_exploration_plan"] += 1
            continue
        control_plan = _video_edit_control_plan(route)
        preserve_regions = _video_preserve_regions(
            preserve_tokens=preserve_tokens,
            edit_region=edit_region,
            reference_annotation=reference_annotation,
        )
        if planned_preserve_regions:
            preserve_regions = planned_preserve_regions
        mask_plan_name = "grounded_sam2_video_mask" if route == "vace_controlled" else (
            "none" if route == "audio_deterministic" else "local_roi"
        )
        src_ref_requirements = _src_ref_requirement_for_video_plan(
            {
                "difference": difference,
                "edit_text": edit_text,
                "edit_token": edit_token,
                "edit_region": edit_region,
                "model_route": route,
                "exploration_family": str(candidate.get("exploration_family", "")).strip(),
            }
        )
        plan = {
            "plan_id": str(candidate.get("proposal_id", "")).strip()
            or f"video_edit_plan_{_stable_hash(reference_video + edit_text)}",
            "reference_video": reference_video,
            "source_candidate_edit_text": str(candidate.get("source_candidate_edit_text", edit_text)).strip(),
            "source_candidate_difference": candidate.get("source_candidate_difference", difference),
            "edit_text": edit_text,
            "planner": planner_metadata,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "edit_token": edit_token,
            "preserve_tokens": preserve_tokens,
            "negative_prompt": negative_prompt,
            "edit_region": edit_region,
            "mask_query": mask_query,
            "preserve_regions": preserve_regions,
            "mask_plan": mask_plan_name,
            "control_plan": control_plan,
            "model_route": route,
            "route": "vace14b_masked_v2v" if route == "vace_controlled" else route,
            "vace_inputs": {
                "src_video": reference_video,
                "src_mask": "to_be_generated" if route == "vace_controlled" else "",
                "src_ref_images": [],
            },
            "src_ref_requirements": src_ref_requirements,
            "difference": difference,
            "raw_planner_output": raw_planner_output,
            "reference_understanding": _video_edit_reference_understanding(reference_annotation),
            "route_suitability": suitability,
            "plan_lint": plan_lint,
            "visual_edit_risk": risk,
            "planning_mode": planning_mode,
            "exploration_family": str(candidate.get("exploration_family", "")).strip(),
            "exploration_goal": str(candidate.get("exploration_goal", "")).strip(),
            "generation_defaults": _video_edit_generation_defaults(route),
            "validation_requirements": {
                "visual_near_duplicate_min": MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE,
                "preserve_reference_audio": route != "audio_deterministic",
                "single_edit_token": True,
                "requires_mask": route == "vace_controlled",
                "outside_mask_visual_near_duplicate_min": MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE,
                "mask_gate": _video_mask_gate_defaults() if route == "vace_controlled" else {},
            },
        }
        plans.append(plan)
        seen_plan_keys.add(plan_key)
        if route != "audio_deterministic" and planning_mode != "exploration":
            seen_sources.add(reference_video)

    if planner_cache_dirty:
        _write_jsonl(cache_output, list(planner_cache.values()))
    _write_jsonl(output, plans)
    return {
        "candidate_count": original_candidate_count,
        "expanded_candidate_count": len(candidates),
        "plan_count": len(plans),
        "planning_mode": planning_mode,
        "output_path": str(output),
        "planner_cache_path": str(cache_output),
        "planner_cache_hits": cache_hits,
        "planner_cache_misses": cache_misses,
        "skipped_by_type": dict(skipped_by_type),
        "skipped_reasons": dict(skipped_reasons),
    }


def plan_audio_edits(
    *,
    root: str | Path,
    pair_candidates_path: str | Path,
    clip_annotations_path: str | Path,
    output_path: str | Path | None = None,
    max_plans: int = 10,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    candidates = list(_load_jsonl(Path(pair_candidates_path)))
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    if not candidates:
        raise ValueError("pair candidates file is empty")
    if not annotations:
        raise ValueError("clip annotations are empty")

    annotation_lookup = _annotation_lookup(root=layout["root"], annotations=annotations)
    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_AUDIO_EDIT_PLAN_NAME
    plans: list[dict[str, Any]] = []
    skipped_by_type: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    for candidate in candidates:
        if len(plans) >= max_plans:
            break
        difference = dict(candidate.get("difference") or {})
        difference_type = str(difference.get("type", "")).strip()
        edit_text = str(candidate.get("edit_text", "")).strip() or _build_fallback_edit_text(difference)
        reference_video = str(candidate.get("reference_video", "")).strip()
        if not reference_video:
            skipped_by_type[difference_type] += 1
            skipped_reasons["missing_reference_video"] += 1
            continue
        reference_annotation = _annotation_for_video_edit_plan(
            root=layout["root"],
            lookup=annotation_lookup,
            record=candidate,
            video_field="reference_video",
            caption_field="reference_caption",
        )
        source_candidate_edit_text = str(candidate.get("source_candidate_edit_text", edit_text)).strip()
        source_candidate_difference = candidate.get("source_candidate_difference", difference)
        speech_issues = _speech_content_edit_issues(edit_text=edit_text, difference=difference)
        if speech_issues or difference_type == "speech":
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons["speech_content_or_speech_only_audio"] += 1
            continue
        if difference_type != "audio_event":
            ideation_candidate = _safe_audio_ideation_candidate(candidate, reference_annotation)
            if ideation_candidate is None:
                skipped_by_type[difference_type or "unknown"] += 1
                skipped_reasons["not_audio_event"] += 1
                continue
            candidate = ideation_candidate
            difference = dict(candidate.get("difference") or {})
            difference_type = str(difference.get("type", "")).strip()
            edit_text = str(candidate.get("edit_text", "")).strip()
            source_candidate_edit_text = str(candidate.get("source_candidate_edit_text", source_candidate_edit_text)).strip()
            source_candidate_difference = candidate.get("source_candidate_difference", source_candidate_difference)
            skipped_reasons["safe_audio_ideation_from_non_audio_candidate"] += 1
        speech_issues = _speech_content_edit_issues(edit_text=edit_text, difference=difference)
        if speech_issues:
            skipped_by_type[difference_type] += 1
            skipped_reasons["speech_content_or_speech_only_audio"] += 1
            continue
        expected_event = _audio_expected_event(difference, edit_text)
        if not expected_event:
            skipped_by_type[difference_type] += 1
            skipped_reasons["missing_expected_audio_event"] += 1
            continue
        plan_id = str(candidate.get("proposal_id", "")).strip() or f"audio_edit_plan_{_stable_hash(reference_video + edit_text)}"
        route = _audio_edit_route(expected_event, reference_annotation)
        suitability = _audio_edit_route_suitability(
            expected_event=expected_event,
            difference=difference,
            edit_text=edit_text,
            reference_annotation=reference_annotation,
        )
        if not suitability["allow_generation"]:
            skipped_by_type[difference_type] += 1
            skipped_reasons[str(suitability["reason"])] += 1
            continue
        target_video = str(candidate.get("target_video", "")).strip() or f"clips/synthetic_audio/{plan_id}.mp4"
        audio_plan = {
            "route": route,
            "audio_prompt": _audio_edit_prompt(expected_event, reference_annotation, edit_text),
            "negative_audio_prompt": "speech, narration, talking, voiceover, crowd chatter, unrelated music",
            "timing_strategy": _audio_timing_strategy(expected_event, reference_annotation),
            "preserve_video": True,
            "mixing": "overlay",
            "expected_event": expected_event,
        }
        plans.append(
            {
                "plan_id": plan_id,
                "reference_video": reference_video,
                "target_video": target_video,
                "source_candidate_edit_text": source_candidate_edit_text,
                "source_candidate_difference": source_candidate_difference,
                "edit_text": edit_text,
                "difference": difference,
                "planner": {
                    "stage": "strongest_omni_audio_prompt_planner",
                    "input": "short_clip_reference_video_and_audio_understanding",
                    "output": "non_speech_audio_event_plan",
                },
                "audio_reference_understanding": _audio_edit_reference_understanding(reference_annotation),
                "route_suitability": suitability,
                "audio_edit_plan": audio_plan,
                "generation_defaults": {
                    "preserve_video_stream": True,
                    "generate_video": False,
                    "visual_near_duplicate_min": MIN_SYNTHETIC_AUDIO_VISUAL_CONTEXT_SCORE,
                    "duration_drift_max": 0.10,
                },
            }
        )

    _write_jsonl(output, plans)
    return {
        "candidate_count": len(candidates),
        "plan_count": len(plans),
        "output_path": str(output),
        "skipped_by_type": dict(skipped_by_type),
        "skipped_reasons": dict(skipped_reasons),
    }


def plan_stable_omni_clips(
    *,
    root: str | Path,
    raw_index_path: str | Path | None = None,
    output_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    max_source_videos: int = 50,
    min_clip_seconds: float = 5.0,
    max_clip_seconds: float = 8.0,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    model: str | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    raw_index_file = Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME
    raw_index = _load_raw_asset_index(raw_index_file)
    if not raw_index:
        raise ValueError("raw asset index is empty")
    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be positive")
    if max_clip_seconds < min_clip_seconds:
        raise ValueError("max_clip_seconds must be >= min_clip_seconds")

    output = Path(output_path) if output_path else layout["metadata"] / "omni_stable_clip_plan.jsonl"
    cache_output = Path(cache_path) if cache_path else layout["caches"] / DEFAULT_OMNI_STABLE_CLIP_SELECTION_CACHE_NAME
    cache = _load_records_by_key(cache_output, "cache_key")
    _write_jsonl(output, [])
    client = (
        OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model or "",
            timeout_seconds=timeout_seconds,
        )
        if base_url and model
        else None
    )

    plan_records: list[dict[str, Any]] = []
    cache_records = dict(cache)
    cache_hits = 0
    cache_misses = 0
    skipped_reasons: Counter[str] = Counter()
    assets = sorted(raw_index.values(), key=lambda item: (str(item.get("dataset", "")), str(item.get("relative_path", ""))))
    for asset in assets[: max(0, max_source_videos)]:
        source_path = Path(str(asset.get("path", "")).strip())
        if not source_path.exists():
            skipped_reasons["missing_source_video"] += 1
            continue
        media = probe_media(source_path)
        duration = float(media.get("duration_seconds") or 0.0)
        if duration < min_clip_seconds:
            skipped_reasons["too_short"] += 1
            continue

        cache_key = _stable_json_hash(
            {
                "asset_id": asset.get("asset_id"),
                "path": str(source_path),
                "mtime_ns": asset.get("mtime_ns"),
                "min_clip_seconds": min_clip_seconds,
                "max_clip_seconds": max_clip_seconds,
                "model": model or "",
            }
        )
        cached_record = cache.get(cache_key)
        if cached_record is not None:
            cache_hits += 1
            selection = dict(cached_record.get("selection") or {})
        else:
            cache_misses += 1
            selection = _heuristic_stable_clip_selection(
                media=media,
                min_clip_seconds=min_clip_seconds,
                max_clip_seconds=max_clip_seconds,
            )
            if client is not None:
                try:
                    model_selection, raw_payload = client.select_stable_clip_window(
                        source_video_path=str(source_path),
                        media_info=media,
                        min_clip_seconds=min_clip_seconds,
                        max_clip_seconds=max_clip_seconds,
                    )
                    selection = _coerce_stable_clip_selection(
                        model_selection,
                        fallback=selection,
                        media=media,
                        min_clip_seconds=min_clip_seconds,
                        max_clip_seconds=max_clip_seconds,
                    )
                    selection["planner"] = {
                        "stage": "strongest_omni_stable_clip_selector",
                        "fallback_used": False,
                        "model": model,
                        "raw_payload": raw_payload,
                    }
                except Exception as exc:
                    selection["planner"] = {
                        "stage": "heuristic_stable_clip_selector",
                        "fallback_used": True,
                        "model": model,
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                    }
            else:
                selection["planner"] = {
                    "stage": "heuristic_stable_clip_selector",
                    "fallback_used": True,
                    "reason": "no Omni endpoint supplied",
                }
            cache_record = {
                "cache_key": cache_key,
                "asset_id": asset.get("asset_id"),
                "source_video": str(source_path),
                "selection": selection,
            }
            cache_records[cache_key] = cache_record
            _append_jsonl_record(cache_output, cache_record)

        if not bool(selection.get("recommended_for_vace", True)):
            skipped_reasons["not_recommended_for_vace"] += 1
            continue
        start_seconds = float(selection.get("start_sec", 0.0) or 0.0)
        end_seconds = float(selection.get("end_sec", 0.0) or 0.0)
        if end_seconds - start_seconds < min_clip_seconds or end_seconds - start_seconds > max_clip_seconds + 1e-6:
            skipped_reasons["invalid_selected_window"] += 1
            continue

        dataset = str(asset.get("dataset", "raw")).strip() or "raw"
        clip_id = f"{dataset}_{Path(str(asset.get('relative_path', source_path.name))).stem}__omni_{_stable_hash(cache_key)[:8]}"
        clip_record = {
            "clip_id": clip_id,
            "source_asset_id": str(asset.get("asset_id", "")).strip(),
            "source_path": str(source_path),
            "output_path": f"clips/omni_stable/{clip_id}.mp4",
            "start_seconds": round(start_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "duration_seconds": round(end_seconds - start_seconds, 3),
            "role": "reference",
            "notes": "Omni-selected stable short clip for VACE planning",
            "stable_clip_selection": selection,
        }
        plan_records.append(clip_record)
        _append_jsonl_record(output, clip_record)

    _write_jsonl(output, plan_records)
    _write_jsonl(cache_output, list(cache_records.values()))
    return {
        "raw_index_path": str(raw_index_file),
        "clip_plan_count": len(plan_records),
        "output_path": str(output),
        "cache_path": str(cache_output),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "skipped_reasons": dict(skipped_reasons),
    }


def cache_reference_understandings(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    if not annotations:
        raise ValueError("clip annotations are empty")
    output = Path(output_path) if output_path else layout["caches"] / DEFAULT_REFERENCE_UNDERSTANDING_CACHE_NAME
    records: list[dict[str, Any]] = []
    skipped_unusable_count = 0
    for annotation in annotations:
        if not _annotation_is_usable_for_reference_understanding(annotation):
            skipped_unusable_count += 1
            continue
        reference_video = str(annotation.get("output_path") or annotation.get("source_path") or "").strip()
        clip_id = str(annotation.get("clip_id", "")).strip() or _stable_hash(reference_video)
        visual_understanding = _video_edit_reference_understanding(annotation)
        audio_understanding = _audio_edit_reference_understanding(annotation)
        stable_targets = _stable_edit_targets_from_understanding(visual_understanding, annotation)
        records.append(
            {
                "clip_id": clip_id,
                "reference_video": reference_video,
                "summary": str(annotation.get("summary", "")).strip(),
                "subjects": _dedupe_strings(_normalize_list(annotation.get("subjects", []))),
                "actions": _dedupe_strings(_normalize_list(annotation.get("actions", []))),
                "scene": str(annotation.get("scene", "")).strip(),
                "camera_motion": str(annotation.get("camera_motion", "")).strip() or "unknown",
                "visible_text": _dedupe_strings(
                    _normalize_list(annotation.get("visible_text", []))
                    + _normalize_list(annotation.get("on_screen_text", []))
                ),
                "stable_edit_targets": stable_targets,
                "bad_edits": visual_understanding.get("bad_edits", []),
                "reference_understanding": visual_understanding,
                "audio_reference_understanding": audio_understanding,
            }
        )
    _write_jsonl(output, records)
    return {
        "clip_annotations_path": str(clip_annotations_path),
        "understanding_count": len(records),
        "skipped_unusable_annotation_count": skipped_unusable_count,
        "output_path": str(output),
    }


def plan_video_masks(
    *,
    root: str | Path,
    video_edit_plan_path: str | Path,
    output_path: str | Path | None = None,
    mask_manifest_path: str | Path | None = None,
    max_masks: int | None = None,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    edit_plans = list(_load_jsonl(Path(video_edit_plan_path)))
    if not edit_plans:
        raise ValueError("video edit plan file is empty")

    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_VIDEO_MASK_PLAN_NAME
    manifest_output = Path(mask_manifest_path) if mask_manifest_path else output.with_name(DEFAULT_VIDEO_MASK_MANIFEST_NAME)
    mask_dir = output.parent / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_plans: list[dict[str, Any]] = []
    mask_manifest: list[dict[str, Any]] = []
    skipped_reasons: Counter[str] = Counter()
    for edit_plan in edit_plans:
        if max_masks is not None and max_masks > 0 and len(mask_plans) >= max_masks:
            break
        if str(edit_plan.get("model_route", "")).strip() != "vace_controlled":
            skipped_reasons["non_vace_route"] += 1
            continue
        plan_id = str(edit_plan.get("plan_id", "")).strip()
        if not plan_id:
            skipped_reasons["missing_plan_id"] += 1
            continue
        mask_query = str(edit_plan.get("mask_query", "")).strip() or _video_mask_query(
            difference=dict(edit_plan.get("difference") or {}),
            edit_text=str(edit_plan.get("edit_text", "")).strip(),
            edit_token=str(edit_plan.get("edit_token", "")).strip(),
            edit_region=str(edit_plan.get("edit_region", "")).strip(),
            route=str(edit_plan.get("model_route", "")).strip(),
            suitability=edit_plan.get("route_suitability") if isinstance(edit_plan.get("route_suitability"), dict) else {},
            reference_annotation=edit_plan.get("reference_understanding")
            if isinstance(edit_plan.get("reference_understanding"), dict)
            else {},
        )
        if (
            str((edit_plan.get("difference") or {}).get("type", "")).strip() == "scene"
            and _normalized_phrase(mask_query) == "background"
        ):
            mask_query = _foreground_mask_query_from_annotation(
                edit_plan.get("reference_understanding") if isinstance(edit_plan.get("reference_understanding"), dict) else edit_plan
            )
        if not mask_query:
            skipped_reasons["missing_mask_query"] += 1
            continue
        reference_video = str(edit_plan.get("reference_video", "")).strip()
        reference_path = _resolve_under_root(layout["root"], reference_video)
        if not reference_video or not reference_path.exists():
            skipped_reasons["missing_reference_video"] += 1
            continue
        safe_id = _safe_id(plan_id)
        mask_video = mask_dir / f"{safe_id}_mask.mp4"
        mask_record = {
            "plan_id": plan_id,
            "reference_video": reference_video,
            "reference_video_absolute": str(reference_path),
            "mask_video": str(mask_video),
            "mask_query": mask_query,
            "mask_mode": _video_mask_mode(edit_plan),
            "edit_region": str(edit_plan.get("edit_region", "")).strip(),
            "preserve_regions": _video_preserve_regions(
                preserve_tokens=_normalize_list(edit_plan.get("preserve_tokens", [])),
                edit_region=str(edit_plan.get("edit_region", "")).strip(),
                reference_annotation={},
            ),
            "toolchain": {
                "grounder": "GroundingDINO_or_Florence-2",
                "segmenter": "SAM2.1_video_predictor",
                "wrapper": "Grounded-SAM-2",
            },
            "mask_gate": _video_mask_gate_defaults(mask_mode=_video_mask_mode(edit_plan), mask_query=mask_query),
            "status": "planned",
        }
        mask_plans.append(mask_record)
        mask_manifest.append(
            {
                "plan_id": plan_id,
                "reference_video": reference_video,
                "mask_video": str(mask_video),
                "mask_query": mask_query,
                "mask_mode": mask_record["mask_mode"],
                "status": "planned",
            }
        )

    _write_jsonl(output, mask_plans)
    _write_jsonl(manifest_output, mask_manifest)
    return {
        "video_edit_plan_path": str(video_edit_plan_path),
        "mask_plan_count": len(mask_plans),
        "output_path": str(output),
        "mask_manifest_path": str(manifest_output),
        "skipped_reasons": dict(skipped_reasons),
    }


def plan_src_ref_images(
    *,
    root: str | Path,
    video_edit_plan_path: str | Path,
    output_path: str | Path | None = None,
    image_root: str | Path | None = None,
    num_candidates: int = 4,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    edit_plans = list(_load_jsonl(Path(video_edit_plan_path)))
    if not edit_plans:
        raise ValueError("video edit plan file is empty")
    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_SRC_REF_IMAGE_PLAN_NAME
    image_base = Path(image_root) if image_root else output.parent / "src_ref_images"
    plans: list[dict[str, Any]] = []
    skipped_reasons: Counter[str] = Counter()
    for edit_plan in edit_plans:
        requirement = _src_ref_requirement_for_video_plan(edit_plan)
        if not requirement.get("required") and not requirement.get("recommended"):
            skipped_reasons["src_ref_not_needed"] += 1
            continue
        plan_id = str(edit_plan.get("plan_id", "")).strip()
        if not plan_id:
            skipped_reasons["missing_plan_id"] += 1
            continue
        target = str(requirement.get("target", "")).strip()
        if not target:
            skipped_reasons["missing_src_ref_target"] += 1
            continue
        safe_id = _safe_id(plan_id)
        candidate_dir = image_base / safe_id
        plans.append(
            {
                "plan_id": plan_id,
                "reference_video": str(edit_plan.get("reference_video", "")).strip(),
                "edit_text": str(edit_plan.get("edit_text", "")).strip(),
                "difference": edit_plan.get("difference", {}),
                "target_object": target,
                "src_ref_role": str(requirement.get("role", "")).strip(),
                "required": bool(requirement.get("required")),
                "recommended": bool(requirement.get("recommended")),
                "image_prompts": _src_ref_image_prompts(requirement=requirement, edit_plan=edit_plan),
                "negative_prompt": _src_ref_image_negative_prompt(requirement),
                "num_candidates": max(1, int(num_candidates)),
                "candidate_dir": str(candidate_dir),
                "planner": {
                    "stage": "src_ref_image_requirement_planner",
                    "input": "video_edit_plan_and_omni_reference_understanding",
                    "output": "image_generation_prompts_for_vace_src_ref_images",
                },
            }
        )
    _write_jsonl(output, plans)
    return {
        "video_edit_plan_path": str(video_edit_plan_path),
        "plan_count": len(plans),
        "output_path": str(output),
        "image_root": str(image_base),
        "skipped_reasons": dict(skipped_reasons),
    }


def select_src_ref_images(
    *,
    root: str | Path,
    src_ref_image_plan_path: str | Path,
    output_path: str | Path | None = None,
    max_selected: int = 2,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    model: str | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    ensure_layout(root)
    image_plans = list(_load_jsonl(Path(src_ref_image_plan_path)))
    if not image_plans:
        raise ValueError("src_ref image plan file is empty")
    output = Path(output_path) if output_path else Path(src_ref_image_plan_path).with_name(DEFAULT_SRC_REF_IMAGE_SELECTION_NAME)
    audit_client = (
        OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        if base_url and model
        else None
    )
    records: list[dict[str, Any]] = []
    selected_count = 0
    missing_count = 0
    audit_rejected_count = 0
    audit_failed_count = 0
    for plan in image_plans:
        candidate_dir = Path(str(plan.get("candidate_dir", "")).strip())
        candidates = _find_src_ref_image_candidates(candidate_dir)
        audited = sorted(
            (_audit_src_ref_image_candidate(path, plan) for path in candidates),
            key=lambda item: (-float(item.get("score", 0.0)), str(item.get("path", ""))),
        )
        role = str(plan.get("src_ref_role", "")).strip()
        selection_limit = max(1, int(max_selected))
        if role == "replacement_object":
            selection_limit = min(selection_limit, 2)
        selected_audits = audited[:selection_limit]
        selection_method = "deterministic_src_ref_quality_audit"
        selection_reason = "selected highest-scoring candidate image(s) by deterministic VACE src_ref quality audit"
        omni_audit: dict[str, Any] | None = None
        raw_omni_audit: dict[str, Any] | None = None
        if audit_client and audited:
            try:
                candidate_image_paths = [str(item["path"]) for item in audited]
                omni_audit, raw_omni_audit = audit_client.audit_src_ref_images(
                    src_ref_plan=plan,
                    candidate_image_paths=candidate_image_paths,
                    max_selected=selection_limit,
                )
                selected_indices = [
                    int(index)
                    for index in omni_audit.get("selected_indices", [])
                    if isinstance(index, int) or str(index).isdigit()
                ]
                selected_audits = [
                    audited[index - 1]
                    for index in selected_indices
                    if 1 <= index <= len(audited)
                ]
                selection_method = "omni_src_ref_image_audit"
                selection_reason = str(omni_audit.get("reason", "")).strip() or (
                    "selected candidate image(s) by Omni src_ref visual audit"
                    if selected_audits
                    else "Omni audit rejected all generated candidate images"
                )
            except Exception as exc:
                selected_audits = []
                selection_method = "omni_src_ref_image_audit_failed"
                selection_reason = f"Omni src_ref audit failed: {exc}"
                audit_failed_count += 1
        selected = [str(item["path"]) for item in selected_audits]
        selected_set = set(selected)
        rejected = [
            {
                "path": str(item.get("path", "")),
                "reason": (
                    "not selected by Omni src_ref audit"
                    if audit_client and audited
                    else "lower deterministic src_ref audit score"
                ),
                "audit": item,
            }
            for item in audited
            if str(item.get("path", "")) not in selected_set
        ]
        if selected:
            status = "selected"
            selected_count += 1
        elif audited and audit_client:
            status = "rejected_by_omni_audit" if selection_method == "omni_src_ref_image_audit" else "omni_audit_failed"
            audit_rejected_count += int(selection_method == "omni_src_ref_image_audit")
        else:
            status = "missing_candidates"
            selection_reason = "no generated candidate images found"
            missing_count += 1
        record = {
            "plan_id": str(plan.get("plan_id", "")).strip(),
            "selected_src_ref_images": selected,
            "rejected": rejected,
            "status": status,
            "required": bool(plan.get("required")),
            "src_ref_role": str(plan.get("src_ref_role", "")).strip(),
            "selection_reason": selection_reason,
            "selection_method": selection_method,
            "candidate_dir": str(candidate_dir),
            "candidate_audit": audited,
        }
        if omni_audit is not None:
            record["omni_audit"] = omni_audit
        if raw_omni_audit is not None:
            record["raw_omni_audit"] = raw_omni_audit
        records.append(record)
    _write_jsonl(output, records)
    return {
        "src_ref_image_plan_path": str(src_ref_image_plan_path),
        "selection_count": len(records),
        "selected_plan_count": selected_count,
        "missing_candidate_count": missing_count,
        "audit_rejected_count": audit_rejected_count,
        "audit_failed_count": audit_failed_count,
        "output_path": str(output),
    }


def build_manual_review_bundle(
    *,
    root: str | Path,
    pairs_path: str | Path,
    output_dir: str | Path,
    clip_annotations_path: str | Path | None = None,
    limit: int | None = None,
    copy_videos: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    pairs = list(_load_jsonl(Path(pairs_path)))
    annotations = list(_load_jsonl(Path(clip_annotations_path))) if clip_annotations_path else []
    annotation_lookup = _annotation_lookup(root=root_path, annotations=annotations) if annotations else {}
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    missing_videos: list[str] = []
    selected_pairs = pairs[: limit if limit and limit > 0 else None]
    for index, record in enumerate(selected_pairs, start=1):
        sample_id = str(record.get("sample_id") or record.get("proposal_id") or f"sample_{index:04d}").strip()
        safe_sample_id = _safe_id(sample_id)
        item_dir = output_root / f"{index:04d}_{safe_sample_id}"
        item_dir.mkdir(parents=True, exist_ok=True)

        reference_video_raw = str(record.get("reference_video", "")).strip()
        target_video_raw = str(record.get("target_video", "")).strip()
        reference_path = _resolve_under_root(root_path, reference_video_raw) if reference_video_raw else Path()
        target_path = _resolve_under_root(root_path, target_video_raw) if target_video_raw else Path()
        reference_annotation = _review_annotation_for_record(
            root=root_path,
            lookup=annotation_lookup,
            record=record,
            video_field="reference_video",
            clip_id_field="reference_clip_id",
        )
        target_annotation = _review_annotation_for_record(
            root=root_path,
            lookup=annotation_lookup,
            record=record,
            video_field="target_video",
            clip_id_field="target_clip_id",
        )
        reference_caption = (
            str(record.get("reference_caption", "")).strip()
            or str(reference_annotation.get("summary", "")).strip()
            or str(reference_annotation.get("caption", "")).strip()
        )
        target_caption = (
            str(record.get("target_caption", "")).strip()
            or str(target_annotation.get("summary", "")).strip()
            or str(target_annotation.get("caption", "")).strip()
        )
        reference_copy = item_dir / "reference.mp4"
        target_copy = item_dir / "target.mp4"
        if copy_videos:
            if reference_path.exists():
                shutil.copy2(reference_path, reference_copy)
            else:
                missing_videos.append(str(reference_path or reference_video_raw))
            if target_path.exists():
                shutil.copy2(target_path, target_copy)
            else:
                missing_videos.append(str(target_path or target_video_raw))
        generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
        src_ref_images = _normalize_list(generation.get("src_ref_images", []))
        copied_src_ref_images: list[str] = []
        if copy_videos and src_ref_images:
            src_ref_dir = item_dir / "src_ref_images"
            src_ref_dir.mkdir(parents=True, exist_ok=True)
            for image_index, image_raw in enumerate(src_ref_images, start=1):
                image_path = _resolve_under_root(root_path, image_raw)
                if image_path.exists():
                    image_copy = src_ref_dir / f"{image_index:03d}_{_safe_id(image_path.name)}"
                    shutil.copy2(image_path, image_copy)
                    copied_src_ref_images.append(str(image_copy))
                else:
                    missing_videos.append(str(image_path or image_raw))
        src_mask = str(generation.get("src_mask", "")).strip()
        mask_copy_path = ""
        if copy_videos and src_mask:
            mask_path = _resolve_under_root(root_path, src_mask)
            if mask_path.exists():
                mask_copy = item_dir / "mask.mp4"
                shutil.copy2(mask_path, mask_copy)
                mask_copy_path = str(mask_copy)
            else:
                missing_videos.append(str(mask_path or src_mask))

        metadata = {
            "index": index,
            "sample_id": sample_id,
            "proposal_id": record.get("proposal_id"),
            "difference": record.get("difference", {}),
            "edit_text": str(record.get("edit_text", "")).strip(),
            "reference_video": reference_video_raw,
            "target_video": target_video_raw,
            "reference_video_absolute": str(reference_path) if reference_video_raw else "",
            "target_video_absolute": str(target_path) if target_video_raw else "",
            "reference_caption": reference_caption,
            "target_caption": target_caption,
            "verification": record.get("verification", {}),
            "observable_difference": record.get("observable_difference", {}),
            "competing_difference": record.get("competing_difference", {}),
            "generation": generation,
            "src_ref_images": src_ref_images,
            "copied_src_ref_images": copied_src_ref_images,
            "src_mask": src_mask,
            "copied_src_mask": mask_copy_path,
        }
        (item_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        review_md = _manual_review_item_markdown(
            metadata=metadata,
            reference_filename="reference.mp4" if copy_videos and reference_copy.exists() else "",
            target_filename="target.mp4" if copy_videos and target_copy.exists() else "",
        )
        (item_dir / "review.md").write_text(review_md, encoding="utf-8")
        items.append(
            {
                "index": index,
                "sample_id": sample_id,
                "difference_type": (record.get("difference") or {}).get("type") if isinstance(record.get("difference"), dict) else "",
                "edit_text": str(record.get("edit_text", "")).strip(),
                "item_dir": str(item_dir),
                "review_md": str(item_dir / "review.md"),
                "reference_video": str(reference_copy if copy_videos and reference_copy.exists() else reference_path),
                "target_video": str(target_copy if copy_videos and target_copy.exists() else target_path),
            }
        )

    _write_jsonl(output_root / "review_items.jsonl", items)
    index_md = _manual_review_index_markdown(items=items, source_pairs_path=str(pairs_path), missing_videos=missing_videos)
    (output_root / "index.md").write_text(index_md, encoding="utf-8")
    return {
        "pair_count": len(pairs),
        "bundle_count": len(items),
        "output_dir": str(output_root),
        "index_path": str(output_root / "index.md"),
        "items_path": str(output_root / "review_items.jsonl"),
        "missing_video_count": len(missing_videos),
        "missing_videos": missing_videos,
    }


def validate_known_pairs(
    *,
    root: str | Path,
    known_pairs_path: str | Path,
    clip_annotations_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    accepted_output_path: str | Path | None = None,
    raw_index_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    max_accepted_pairs: int = 10,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    known_pairs = list(_load_jsonl(Path(known_pairs_path)))
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    if not known_pairs:
        raise ValueError("known pairs file is empty")
    if not annotations:
        raise ValueError("clip annotations are empty")

    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_SYNTHETIC_JUDGED_PAIRS_NAME
    accepted_output = Path(accepted_output_path) if accepted_output_path else layout["pairs"] / DEFAULT_SYNTHETIC_ACCEPTED_PAIRS_NAME
    existing_records = {} if overwrite else _load_records_by_key(output, "proposal_id")
    raw_index = _load_raw_asset_index(Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME)
    annotation_lookup = _annotation_lookup(root=layout["root"], annotations=annotations)
    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

    output_records: list[dict[str, Any]] = []
    proposed_count = 0
    reused_count = 0
    fallback_count = 0
    rejected_count = 0
    accepted_total_count = 0
    seen_proposal_ids: set[str] = set()

    for line_number, pair in enumerate(known_pairs, start=1):
        reference_annotation = _annotation_for_known_pair(
            root=layout["root"],
            lookup=annotation_lookup,
            pair=pair,
            clip_id_field="reference_clip_id",
            video_field="reference_video",
            line_number=line_number,
        )
        target_annotation = _annotation_for_known_pair(
            root=layout["root"],
            lookup=annotation_lookup,
            pair=pair,
            clip_id_field="target_clip_id",
            video_field="target_video",
            line_number=line_number,
        )

        reference_video = _known_pair_video_path(layout["root"], pair, reference_annotation, "reference_video")
        target_video = _known_pair_video_path(layout["root"], pair, target_annotation, "target_video")
        proposal_id = str(pair.get("proposal_id", "")).strip() or _build_proposal_id(reference_video, target_video)
        if proposal_id in seen_proposal_ids:
            continue
        seen_proposal_ids.add(proposal_id)

        if proposal_id in existing_records:
            record = existing_records[proposal_id]
            reused_count += 1
        else:
            raw_judge_output: dict[str, Any] = {}
            raw_verification_output: dict[str, Any] = {}
            model_fields = _known_pair_model_fields(
                pair=pair,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            model_fields = _repair_pair_model_fields(
                model_fields=model_fields,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            source = _build_source_metadata(
                root=layout["root"],
                target_annotation=target_annotation,
                raw_index=raw_index,
            )
            source["source_type"] = str(pair.get("source_type", "synthetic_edit")).strip() or "synthetic_edit"
            source_context = _known_pair_source_context(pair)
            hard_negative_annotations = _known_pair_hard_negative_annotations(
                root=layout["root"],
                lookup=annotation_lookup,
                annotations=annotations,
                pair=pair,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                difference=model_fields["difference"],
            )
            hard_negative_paths = _known_pair_hard_negative_paths(
                root=layout["root"],
                pair=pair,
                hard_negative_annotations=hard_negative_annotations,
            )
            base_quality = _known_pair_base_quality(
                root=layout["root"],
                pair=pair,
                annotations=annotations,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                difference=model_fields["difference"],
                source_context=source_context,
            )
            proposal_quality = _quality_for_model_fields(
                base_quality=base_quality,
                model_fields=model_fields,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            edit_text_quality = _edit_text_quality_payload(
                edit_text=model_fields["edit_text"],
                difference=model_fields["difference"],
                modalities=model_fields["modalities"],
                reference_caption=model_fields["reference_caption"],
                target_caption=model_fields["target_caption"],
            )
            observable_difference = _observable_difference_gate(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                difference=model_fields["difference"],
                visual_near_duplicate_score=proposal_quality.get("visual_near_duplicate_score"),
            )
            _apply_structured_gate_quality(
                proposal_quality,
                edit_text_quality=edit_text_quality,
                observable_difference=observable_difference,
            )
            proposal_difference_evidence = _difference_evidence_from_annotations(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                primary_difference=model_fields["difference"],
            )
            proposal_view = {
                "proposal_id": proposal_id,
                "edit_text": model_fields["edit_text"],
                "modalities": list(model_fields["modalities"]),
                "reference_caption": model_fields["reference_caption"],
                "target_caption": model_fields["target_caption"],
                "difference": model_fields["difference"],
                "quality": dict(proposal_quality),
                "source_context": dict(source_context),
                "generation": dict(pair.get("generation", {})),
                "difference_evidence": dict(proposal_difference_evidence),
                "edit_text_quality": dict(edit_text_quality),
                "observable_difference": dict(observable_difference),
                "acceptance_thresholds": {
                    "same_context_score": MIN_ACCEPT_SAME_CONTEXT_SCORE,
                    "edit_match_score": MIN_ACCEPT_EDIT_MATCH_SCORE,
                    "target_uniqueness_score": MIN_ACCEPT_TARGET_UNIQUENESS_SCORE,
                    "difference_strength_score": MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE,
                    "max_visual_near_duplicate_score_for_visual_edits": MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE,
                    "edit_text_quality_score": MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE,
                },
            }
            try:
                judge, raw_judge_output = client.judge_pair(
                    proposal=proposal_view,
                    reference_annotation=_annotation_prompt_view(reference_annotation),
                    target_annotation=_annotation_prompt_view(target_annotation),
                    hard_negative_candidates=[
                        _annotation_prompt_view(annotation) for annotation in hard_negative_annotations
                    ],
                )
                judge_fallback_used = False
            except Exception as exc:
                judge = _fallback_pair_judge(proposal_quality, reason=f"{type(exc).__name__}: {exc}")
                raw_judge_output = {"error": f"{type(exc).__name__}: {exc}"}
                judge_fallback_used = True

            try:
                (
                    verification,
                    raw_verification_output,
                    verification_context_retry_used,
                ) = _verify_pair_difference_with_context_retry(
                    client,
                    proposal=proposal_view,
                    reference_annotation=_annotation_prompt_view(reference_annotation),
                    target_annotation=_annotation_prompt_view(target_annotation),
                    reference_clip_path=str(_resolve_under_root(layout["root"], reference_video)),
                    target_clip_path=str(_resolve_under_root(layout["root"], target_video)),
                )
                verification_fallback_used = False
            except Exception as exc:
                verification = _fallback_pair_verification(reason=f"{type(exc).__name__}: {exc}")
                raw_verification_output = {"error": f"{type(exc).__name__}: {exc}"}
                verification_context_retry_used = False
                verification_fallback_used = True

            judge = _finalize_pair_judge(judge)
            verification = _finalize_pair_verification(verification)
            fallback_used = judge_fallback_used or verification_fallback_used
            effective_quality = _effective_pair_quality(judge, verification, proposal_quality)
            accepted = _judge_accepts(judge, verification, effective_quality)
            if not accepted:
                judge["reject_reason"] = _compose_reject_reason(judge, verification, effective_quality)
            speech_quality = _speech_quality_payload(effective_quality)
            audio_event_quality = _audio_event_quality_payload(effective_quality)
            record = {
                "proposal_id": proposal_id,
                "source_type": str(pair.get("source_type", "synthetic_edit")).strip() or "synthetic_edit",
                "generation": dict(pair.get("generation", {})),
                "group_id": str(pair.get("group_id", "synthetic_edit")).strip() or "synthetic_edit",
                "group_reason": str(pair.get("group_reason", "known_synthetic_pair")).strip() or "known_synthetic_pair",
                "reference_clip_id": reference_annotation.get("clip_id", ""),
                "target_clip_id": target_annotation.get("clip_id", ""),
                "reference_video": reference_video,
                "target_video": target_video,
                "edit_text": model_fields["edit_text"],
                "modalities": list(model_fields["modalities"]),
                "reference_caption": model_fields["reference_caption"],
                "target_caption": model_fields["target_caption"],
                "difference": model_fields["difference"],
                "hard_negatives": hard_negative_paths,
                "judge_quality": {
                    "same_context_score": judge["same_context_score"],
                    "edit_match_score": judge["edit_match_score"],
                    "target_uniqueness_score": judge["target_uniqueness_score"],
                },
                "quality": effective_quality,
                "heuristic_quality": dict(proposal_quality),
                "source_context": dict(source_context),
                "source": source,
                "proposal_reason": str(pair.get("proposal_reason", "known pair validation")).strip(),
                "evidence": _evidence_from_annotations(
                    reference_annotation,
                    target_annotation,
                    difference_evidence=proposal_difference_evidence,
                ),
                "judge": judge,
                "verification": verification,
                "speech_quality": speech_quality,
                "audio_event_quality": audio_event_quality,
                "edit_text_quality": edit_text_quality,
                "observable_difference": observable_difference,
                "transcript_backed": speech_quality.get("transcript_backed"),
                "accepted": accepted,
                "fallback_used": fallback_used,
                "raw_model_output": {"known_pair": True},
                "raw_judge_output": raw_judge_output,
                "raw_verification_output": raw_verification_output,
                "verification_annotation_only_retry_used": verification_context_retry_used,
            }
            proposed_count += 1

        record = _prepare_record_for_acceptance(
            record,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
        )
        judge = dict(record.get("judge", {}))
        verification = record.get("verification", {})
        quality = record.get("quality", {})
        record["accepted"] = _judge_accepts(judge, verification, quality)
        if not bool(record.get("accepted")):
            judge["accept"] = False
            judge["reject_reason"] = _compose_reject_reason(judge, verification, quality)
            record["judge"] = judge
        acceptance_issues = _pair_record_acceptance_issues(
            root=layout["root"],
            record=record,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
        )
        acceptance_issues.extend(_known_pair_generation_issues(record))
        if acceptance_issues:
            record = _reject_record_with_acceptance_issues(record, acceptance_issues)
        if bool(record.get("fallback_used")):
            fallback_count += 1
        if bool(record.get("accepted")):
            accepted_total_count += 1
        else:
            rejected_count += 1
        output_records.append(record)

    accepted_records = _select_final_accepted_records(output_records, max_accepted_pairs=max_accepted_pairs)
    _write_jsonl(output, output_records)
    _write_jsonl(accepted_output, accepted_records)
    verification_counts = _pair_verification_counts(output_records)
    return {
        "known_pairs_path": str(known_pairs_path),
        "clip_annotations_path": str(clip_annotations_path),
        "output_path": str(output),
        "accepted_output_path": str(accepted_output),
        "pair_count": len(known_pairs),
        "proposal_count": len(output_records),
        "accepted_count": len(accepted_records),
        "accepted_total_count": accepted_total_count,
        "rejected_count": rejected_count,
        "proposed_count": proposed_count,
        "reused_count": reused_count,
        "fallback_count": fallback_count,
        "verification_counts": verification_counts,
    }


def validate_pilot_dataset(
    *,
    root: str | Path,
    pilot_jsonl_path: str | Path,
    gallery_output_path: str | Path,
    report_output_path: str | Path,
) -> dict[str, Any]:
    root_path = Path(root)
    pilot_records = list(_load_jsonl(Path(pilot_jsonl_path)))
    if not pilot_records:
        raise ValueError("pilot dataset is empty")

    errors: list[str] = []
    seen_sample_ids: set[str] = set()
    seen_proposal_ids: set[str] = set()
    seen_pair_keys: set[tuple[str, str]] = set()
    difference_counter: Counter[str] = Counter()
    modality_counter: Counter[str] = Counter()
    source_type_counter: Counter[str] = Counter()
    source_type_difference_counter: Counter[str] = Counter()
    speech_count = 0
    high_quality_speech_count = 0
    transcript_backed_speech_count = 0
    non_speech_audio_event_count = 0
    same_context_scores: list[float] = []
    difference_strength_scores: list[float] = []
    source_context_counter: Counter[str] = Counter()
    gallery_accumulator: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(pilot_records, start=1):
        errors.extend(_validate_pilot_record(root_path, record, index))

        sample_id = str(record.get("sample_id", "")).strip()
        if sample_id:
            if sample_id in seen_sample_ids:
                errors.append(f"pilot line {index}: duplicate sample_id={sample_id}")
            seen_sample_ids.add(sample_id)

        proposal_id = str(record.get("proposal_id", "")).strip()
        if not proposal_id:
            errors.append(f"pilot line {index}: proposal_id is required")
        elif proposal_id in seen_proposal_ids:
            errors.append(f"pilot line {index}: duplicate proposal_id={proposal_id}")
        else:
            seen_proposal_ids.add(proposal_id)

        reference_video = str(record.get("reference_video", "")).strip()
        target_video = str(record.get("target_video", "")).strip()
        if reference_video and target_video:
            record_source_type = str(record.get("source_type", "natural")).strip() or "natural"
            expected_proposal_id = _build_proposal_id(reference_video, target_video)
            if record_source_type != "synthetic_edit" and proposal_id and proposal_id != expected_proposal_id:
                errors.append(
                    f"pilot line {index}: proposal_id={proposal_id} does not match expected {expected_proposal_id}"
                )
            pair_key = (reference_video, target_video)
            if pair_key in seen_pair_keys:
                errors.append(f"pilot line {index}: duplicate reference-target pair={pair_key}")
            seen_pair_keys.add(pair_key)

        modalities = [str(item).strip() for item in record.get("modalities", []) if str(item).strip()]
        modality_counter.update(modalities)

        source_type = str(record.get("source_type", "natural")).strip() or "natural"
        source_type_counter[source_type] += 1

        difference = record.get("difference", {})
        difference_type = str(difference.get("type", "")).strip()
        if difference_type:
            difference_counter[difference_type] += 1
            source_type_difference_counter[f"{source_type}:{difference_type}"] += 1

        quality = record.get("quality", {})
        if isinstance(quality, dict):
            if difference_type == "speech":
                speech_count += 1
                if _score_float(quality.get("speech_transcript_backed")) >= 1.0:
                    transcript_backed_speech_count += 1
                if (
                    _score_float(quality.get("speech_evidence_score")) >= MIN_ACCEPT_SPEECH_EVIDENCE_SCORE
                    and _score_float(quality.get("speech_specificity_score")) >= MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE
                ):
                    high_quality_speech_count += 1
            if (
                difference_type == "audio_event"
                and _score_float(quality.get("non_speech_audio_event_score")) >= MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE
            ):
                non_speech_audio_event_count += 1
            try:
                same_context_scores.append(float(quality.get("same_context_score", 0.0)))
            except (TypeError, ValueError):
                pass
            if "difference_strength_score" in quality:
                try:
                    difference_strength_scores.append(float(quality.get("difference_strength_score", 0.0)))
                except (TypeError, ValueError):
                    pass

        source_context = record.get("source_context", {})
        if isinstance(source_context, dict):
            source_context_counter[str(source_context.get("relation", "unknown"))] += 1

        target_video = str(record.get("target_video", "")).strip()
        if target_video:
            _merge_gallery_entry(
                accumulator=gallery_accumulator,
                video_path=target_video,
                sample_id=sample_id,
                role="target",
            )
        for negative in record.get("hard_negatives", []):
            negative_path = str(negative).strip()
            if negative_path:
                _merge_gallery_entry(
                    accumulator=gallery_accumulator,
                    video_path=negative_path,
                    sample_id=sample_id,
                    role="hard_negative",
                )

    if errors:
        raise ValueError("\n".join(errors[:20]))

    gallery_records = [
        {
            "gallery_id": _build_gallery_id(video_path),
            "video_path": video_path,
            "sample_ids": sorted(entry["sample_ids"]),
            "roles": sorted(entry["roles"]),
        }
        for video_path, entry in sorted(gallery_accumulator.items())
    ]
    _write_jsonl(Path(gallery_output_path), gallery_records)

    verification_counts = _load_pair_verification_counts(Path(pilot_jsonl_path))
    summary = {
        "sample_count": len(pilot_records),
        "gallery_count": len(gallery_records),
        "modality_counts": dict(sorted(modality_counter.items())),
        "difference_type_counts": dict(sorted(difference_counter.items())),
        "source_type_counts": dict(sorted(source_type_counter.items())),
        "source_type_difference_counts": dict(sorted(source_type_difference_counter.items())),
        "source_context_counts": dict(sorted(source_context_counter.items())),
        "quality_summary": _quality_summary(same_context_scores),
        "difference_strength_summary": _score_summary(difference_strength_scores, "difference_strength"),
        "verification_counts": verification_counts,
        "speech_audio_quality_counts": {
            "speech_count": speech_count,
            "high_quality_speech_count": high_quality_speech_count,
            "transcript_backed_speech_count": transcript_backed_speech_count,
            "non_speech_audio_event_count": non_speech_audio_event_count,
            "speech_rejected_as_too_generic_count": verification_counts.get("speech_rejected_as_too_generic_count", 0),
            "audio_event_rejected_as_speech_only_count": verification_counts.get(
                "audio_event_rejected_as_speech_only_count",
                0,
            ),
        },
        "automated_acceptance": {
            "sample_count_between_5_and_10": 5 <= len(pilot_records) <= 10,
            "audio_samples_at_least_2": modality_counter.get("audio", 0) >= 2,
            "non_speech_audio_samples_at_least_1": non_speech_audio_event_count >= 1,
            "speech_samples_all_have_evidence": speech_count == high_quality_speech_count,
            "speech_samples_all_transcript_backed": speech_count == transcript_backed_speech_count,
            "object_change_samples_at_least_2": difference_counter.get("object_count", 0)
            + difference_counter.get("object_presence", 0)
            >= 2,
            "action_samples_at_least_1": difference_counter.get("action", 0) >= 1,
        },
    }
    Path(report_output_path).write_text(_build_pilot_report(summary), encoding="utf-8")
    summary["gallery_output_path"] = str(gallery_output_path)
    summary["report_output_path"] = str(report_output_path)
    return summary


def _quality_summary(same_context_scores: list[float]) -> dict[str, float]:
    if not same_context_scores:
        return {"same_context_min": 0.0, "same_context_avg": 0.0, "same_context_max": 0.0}
    return {
        "same_context_min": round(min(same_context_scores), 3),
        "same_context_avg": round(sum(same_context_scores) / len(same_context_scores), 3),
        "same_context_max": round(max(same_context_scores), 3),
    }


def _score_summary(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_min": 0.0, f"{prefix}_avg": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_min": round(min(values), 3),
        f"{prefix}_avg": round(sum(values) / len(values), 3),
        f"{prefix}_max": round(max(values), 3),
    }


def _load_pair_verification_counts(pilot_jsonl_path: Path) -> dict[str, int]:
    candidate_names = ["judged_pair_proposals.jsonl"]
    if "synthetic" in pilot_jsonl_path.name:
        candidate_names.insert(0, DEFAULT_SYNTHETIC_JUDGED_PAIRS_NAME)
    for candidate_name in candidate_names:
        judged_path = pilot_jsonl_path.with_name(candidate_name)
        if judged_path.exists():
            return _pair_verification_counts(list(_load_jsonl(judged_path)))
    return _empty_pair_verification_counts()


def _empty_pair_verification_counts() -> dict[str, int]:
    return {
        "verification_passed_count": 0,
        "verification_passed_rejected_count": 0,
        "verification_override_accept_count": 0,
        "caption_equivalent_reject_count": 0,
        "missing_delta_reject_count": 0,
        "difference_mismatch_reject_count": 0,
        "edit_projection_reject_count": 0,
        "edit_not_needed_reject_count": 0,
        "speech_rejected_as_too_generic_count": 0,
        "speech_rejected_for_missing_transcript_count": 0,
        "audio_event_rejected_as_speech_only_count": 0,
        "good_edit_text_count": 0,
        "bad_edit_text_rejected_count": 0,
        "caption_like_edit_rejected_count": 0,
        "modality_leakage_rejected_count": 0,
        "near_duplicate_without_delta_rejected_count": 0,
        "visual_presence_contradiction_reject_count": 0,
        "visible_text_without_ocr_reject_count": 0,
        "audio_event_without_independent_audio_evidence_reject_count": 0,
        "competing_difference_reject_count": 0,
        "duplicate_target_reject_count": 0,
        "synthetic_context_override_count": 0,
        "synthetic_visual_count": 0,
        "synthetic_audio_count": 0,
        "deterministic_audio_count": 0,
        "foleycrafter_audio_count": 0,
        "frieren_audio_count": 0,
        "speech_content_reject_count": 0,
        "audio_stream_missing_reject_count": 0,
        "visual_changed_in_audio_sample_reject_count": 0,
        "audio_event_not_detected_reject_count": 0,
        "audio_remux_count": 0,
        "missing_target_audio_reject_count": 0,
        "accepted_after_verification_count": 0,
    }


def _pair_verification_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = _empty_pair_verification_counts()
    accepted_target_counts = Counter(
        str(record.get("target_video", "")).strip()
        for record in records
        if bool(record.get("accepted")) and str(record.get("target_video", "")).strip()
    )
    counts["duplicate_target_reject_count"] = sum(max(0, count - 1) for count in accepted_target_counts.values())
    for record in records:
        verification = record.get("verification")
        if not isinstance(verification, dict):
            continue
        quality = record.get("quality", {})
        if not isinstance(quality, dict):
            quality = {}
        if bool(record.get("accepted")) and not _structured_edit_text_failures(quality):
            counts["good_edit_text_count"] += 1
        if _score_float(quality.get("synthetic_context_override")) >= 1.0:
            counts["synthetic_context_override_count"] += 1
        generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
        route = _synthetic_generation_route(generation)
        if str(record.get("source_type", "")).strip() == "synthetic_edit" and bool(record.get("accepted")):
            if route in SYNTHETIC_AUDIO_ROUTES:
                counts["synthetic_audio_count"] += 1
                if route in {"deterministic_overlay", "audio_deterministic"}:
                    counts["deterministic_audio_count"] += 1
                elif route == "foleycrafter_temporal":
                    counts["foleycrafter_audio_count"] += 1
                elif route == "frieren_benchmark":
                    counts["frieren_audio_count"] += 1
            else:
                counts["synthetic_visual_count"] += 1
        postprocess = generation.get("postprocess", {}) if isinstance(generation.get("postprocess"), dict) else {}
        if postprocess.get("audio_copied_from_reference"):
            counts["audio_remux_count"] += 1
        reject_reason_text = str(record.get("judge", {}).get("reject_reason", "")).lower() if isinstance(record.get("judge"), dict) else ""
        if "missing audio copied from the reference" in reject_reason_text:
            counts["missing_target_audio_reject_count"] += 1
        if "missing audio" in reject_reason_text:
            counts["audio_stream_missing_reject_count"] += 1
        if "speech content edits are disabled" in reject_reason_text or "speech difference type is disabled" in reject_reason_text:
            counts["speech_content_reject_count"] += 1
        if "audio synthetic target changed visual stream" in reject_reason_text:
            counts["visual_changed_in_audio_sample_reject_count"] += 1
        if "audio_event target sound was not detected" in reject_reason_text:
            counts["audio_event_not_detected_reject_count"] += 1
        verification_passed = _verification_accepts(verification)
        if verification_passed:
            counts["verification_passed_count"] += 1
        if bool(record.get("accepted")) and verification_passed:
            counts["accepted_after_verification_count"] += 1
            judge = record.get("judge", {})
            if isinstance(judge, dict) and not _boolish(judge.get("accept")):
                counts["verification_override_accept_count"] += 1
            continue
        if verification_passed and not bool(record.get("accepted")):
            counts["verification_passed_rejected_count"] += 1
        if not bool(record.get("accepted")):
            difference_type = str(record.get("difference", {}).get("type", "")).strip()
            if difference_type == "speech" and (
                _score_float(quality.get("speech_evidence_score")) < MIN_ACCEPT_SPEECH_EVIDENCE_SCORE
                or _score_float(quality.get("speech_specificity_score")) < MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE
            ):
                counts["speech_rejected_as_too_generic_count"] += 1
            if difference_type == "speech" and _score_float(quality.get("speech_transcript_backed")) < 1.0:
                counts["speech_rejected_for_missing_transcript_count"] += 1
            if (
                difference_type == "audio_event"
                and _score_float(quality.get("non_speech_audio_event_score")) < MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE
            ):
                counts["audio_event_rejected_as_speech_only_count"] += 1
            edit_text_failures = _structured_edit_text_failures(quality)
            if edit_text_failures:
                counts["bad_edit_text_rejected_count"] += 1
            if any("caption-like" in failure for failure in edit_text_failures):
                counts["caption_like_edit_rejected_count"] += 1
            if any("leaks another modality" in failure for failure in edit_text_failures):
                counts["modality_leakage_rejected_count"] += 1
            if _observable_difference_rejects(quality):
                counts["near_duplicate_without_delta_rejected_count"] += 1
            observable = record.get("observable_difference", {})
            if isinstance(observable, dict):
                observable_reason = str(observable.get("failure_reason", "")).strip().lower()
                if "already appears to contain equivalent object" in observable_reason:
                    counts["visual_presence_contradiction_reject_count"] += 1
                if "visible_text lacks" in observable_reason:
                    counts["visible_text_without_ocr_reject_count"] += 1
            if _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) < 1.0:
                counts["audio_event_without_independent_audio_evidence_reject_count"] += 1
            if _score_float(quality.get("competing_difference_passed", 1.0)) < 1.0:
                counts["competing_difference_reject_count"] += 1
        caption_delta = verification.get("caption_delta", {})
        edit_projection = verification.get("edit_projection", {})
        edit_necessity = verification.get("edit_necessity", {})
        if _boolish(caption_delta.get("caption_equivalent")):
            counts["caption_equivalent_reject_count"] += 1
        if not _boolish(caption_delta.get("has_concrete_difference")):
            counts["missing_delta_reject_count"] += 1
        if not _boolish(caption_delta.get("difference_matches_edit")):
            counts["difference_mismatch_reject_count"] += 1
        if (
            not _boolish(edit_projection.get("target_matches_projection"))
            or _score_float(edit_projection.get("score")) < MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE
        ):
            counts["edit_projection_reject_count"] += 1
        if (
            not _boolish(edit_necessity.get("edit_needed"))
            or _boolish(edit_necessity.get("reference_satisfies_edit"))
            or not _boolish(edit_necessity.get("target_satisfies_edit"))
            or _score_float(edit_necessity.get("score")) < MIN_ACCEPT_EDIT_NECESSITY_SCORE
        ):
            counts["edit_not_needed_reject_count"] += 1
    return counts


def probe_media(source_path: str | Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout or "{}")
    except Exception as exc:
        return {
            "duration_seconds": 0.0,
            "has_audio": False,
            "has_video": False,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    duration = _media_duration(payload, video_stream)
    return {
        "duration_seconds": round(duration, 3),
        "has_audio": bool(audio_stream),
        "has_video": bool(video_stream),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": round(_parse_fraction(str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "")), 3),
    }


def _build_toolbox_observations(clip_path: Path) -> list[dict[str, Any]]:
    media = probe_media(clip_path)
    duration = float(media.get("duration_seconds") or 0.0)
    frame_times = _sample_frame_times(duration)
    audio_note = (
        "audio track present; inspect speech, music, acoustic events, and audio-visual synchronization"
        if media.get("has_audio")
        else "no audio stream detected by ffprobe"
    )
    return [
        {
            "tool": "media_probe",
            "observation": media,
        },
        {
            "tool": "frame_sampler",
            "observation": {
                "sample_times": frame_times,
                "instruction": "use these timestamps as key visual moments for subjects, actions, scene, and visible text",
            },
        },
        {
            "tool": "audio_observer",
            "observation": {
                "note": audio_note,
                "max_audio_window_seconds": 30.0,
            },
        },
        {
            "tool": "ocr_asr_observer",
            "observation": {
                "instruction": "extract visible text and spoken content when present; leave uncertainty if unreadable or inaudible",
            },
        },
    ]


def _sample_frame_times(duration_seconds: float) -> list[float]:
    if duration_seconds <= 0:
        return []
    count = 3 if duration_seconds <= 6 else 6
    if count == 1:
        return [round(duration_seconds / 2, 3)]
    step = duration_seconds / (count + 1)
    return [round(step * index, 3) for index in range(1, count + 1)]


def _media_duration(payload: dict[str, Any], video_stream: dict[str, Any]) -> float:
    for raw_value in (
        payload.get("format", {}).get("duration") if isinstance(payload.get("format"), dict) else None,
        video_stream.get("duration"),
    ):
        try:
            duration = float(raw_value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    return 0.0


def _parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _source_clip_video_path(root: Path, item: dict[str, Any]) -> Path:
    source_path = str(item.get("source_path", "")).strip()
    if source_path:
        path = Path(source_path)
        return path if path.is_absolute() else root / path
    output_path = str(item.get("output_path", "")).strip()
    if output_path:
        return _resolve_under_root(root, output_path)
    return root / "__missing_source_clip__"


def _source_clip_duration_seconds(item: dict[str, Any], media: dict[str, Any]) -> float:
    for value in (
        media.get("duration_seconds"),
        item.get("duration_seconds"),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    start = _optional_float(item.get("start_seconds"))
    end = _optional_float(item.get("end_seconds"))
    if start is not None and end is not None and end > start:
        return end - start
    return 0.0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_segments(
    *,
    duration_seconds: float,
    segment_seconds: float,
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[tuple[float, float]]:
    if duration_seconds < min_clip_seconds:
        return []
    if duration_seconds <= max_clip_seconds:
        return [(0.0, duration_seconds)]
    segment_length = min(max(segment_seconds, min_clip_seconds), max_clip_seconds)
    segments: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + segment_length, duration_seconds)
        if end - start >= min_clip_seconds:
            segments.append((start, end))
        elif segments:
            previous_start, _previous_end = segments[-1]
            segments[-1] = (previous_start, duration_seconds)
        start += segment_length
    return segments


def _group_tags_from_clip(item: dict[str, Any]) -> list[str]:
    tokens = _group_tokens_from_clip(item)
    return sorted(tokens)[:8]


def _group_tokens_from_clip(item: dict[str, Any]) -> set[str]:
    tokens = set()
    tokens.update(_text_field_tokens(item.get("text_fields", {})))
    tokens.update(_tokenize_text(str(item.get("dataset", ""))))
    tokens.update(_tokenize_text(str(item.get("clip_id", ""))))
    return tokens


def _semantic_singleton_groups(items: list[dict[str, Any]], *, group_size: int = 8) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_dataset.setdefault(str(item.get("dataset", "unknown")), []).append(item)

    groups: list[dict[str, Any]] = []
    for dataset, dataset_items in sorted(by_dataset.items()):
        dataset_items.sort(key=lambda item: (item.get("tokens", []), item["clip_id"]))
        for group_index, start in enumerate(range(0, len(dataset_items), group_size), start=1):
            chunk = dataset_items[start : start + group_size]
            if len(chunk) < 2:
                continue
            clip_ids = [str(item["clip_id"]) for item in chunk]
            token_counter: Counter[str] = Counter()
            for item in chunk:
                token_counter.update(item.get("tokens", []))
            group_tags = [token for token, _count in token_counter.most_common(8)]
            groups.append(
                {
                    "group_id": f"group_{dataset}_semantic_{group_index:03d}",
                    "dataset": dataset,
                    "group_reason": "semantic_cluster",
                    "source_clip_ids": [str(item.get("source_clip_id", "")) for item in chunk],
                    "candidate_clip_ids": clip_ids,
                    "group_tags": group_tags,
                }
            )
    return groups


def build_ffmpeg_extract_command(
    *,
    source_path: str | Path,
    output_path: str | Path,
    start_seconds: float,
    end_seconds: float,
    overwrite: bool,
) -> list[str]:
    return [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-ss",
        _format_seconds(start_seconds),
        "-to",
        _format_seconds(end_seconds),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _build_asset_id(dataset_name: str, relative_path: str) -> str:
    stem = Path(relative_path).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "video"
    slug = slug[:32]
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}__{slug}__{digest}"


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _stable_json_hash(value: Any) -> str:
    return _stable_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _load_video_edit_planner_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        key = str(payload.get("cache_key", "")).strip()
        if key:
            cache[key] = payload
    return cache


def _video_edit_planner_cache_key(
    *,
    model: str | None,
    planning_mode: str,
    route: str,
    reference_video: str,
    reference_annotation: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    payload = {
        "model": model or "",
        "planning_mode": planning_mode,
        "route": route,
        "reference_video": reference_video,
        "reference_annotation": _annotation_prompt_view(reference_annotation),
        "candidate": candidate,
    }
    return _stable_json_hash(payload)


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")[:80] or "clip"


def _build_gallery_id(video_path: str) -> str:
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]
    return f"gallery__{digest}"


def _build_proposal_id(reference_path: str, target_path: str) -> str:
    digest = hashlib.sha1(f"{reference_path}::{target_path}".encode("utf-8")).hexdigest()[:16]
    return f"proposal__{digest}"


def _format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _merge_gallery_entry(*, accumulator: dict[str, dict[str, Any]], video_path: str, sample_id: str, role: str) -> None:
    entry = accumulator.setdefault(video_path, {"sample_ids": set(), "roles": set()})
    if sample_id:
        entry["sample_ids"].add(sample_id)
    entry["roles"].add(role)


def _build_raw_summary_report(output_path: Path, dataset_counts: dict[str, int]) -> str:
    lines = [
        "# Raw Asset Index Summary",
        "",
        f"- Index: `{output_path}`",
        f"- Total assets: `{sum(dataset_counts.values())}`",
        "",
        "| Dataset | Video Count |",
        "|---|---:|",
    ]
    for dataset, count in sorted(dataset_counts.items()):
        lines.append(f"| `{dataset}` | `{count}` |")
    return "\n".join(lines) + "\n"


def _build_pilot_report(summary: dict[str, Any]) -> str:
    acceptance = summary["automated_acceptance"]
    lines = [
        "# Pilot Review Summary",
        "",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Gallery count: `{summary['gallery_count']}`",
        "",
        "## Modality Counts",
    ]
    for key, value in summary["modality_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary["modality_counts"]:
        lines.append("- none")

    lines.extend(["", "## Difference Type Counts"])
    for key, value in summary["difference_type_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary["difference_type_counts"]:
        lines.append("- none")

    lines.extend(["", "## Source Type Counts"])
    for key, value in summary.get("source_type_counts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary.get("source_type_counts"):
        lines.append("- none")

    lines.extend(["", "## Source Type Difference Counts"])
    for key, value in summary.get("source_type_difference_counts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary.get("source_type_difference_counts"):
        lines.append("- none")

    lines.extend(["", "## Source Context Counts"])
    for key, value in summary["source_context_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary["source_context_counts"]:
        lines.append("- none")

    quality = summary["quality_summary"]
    lines.extend(
        [
            "",
            "## Quality Summary",
            f"- `same_context_min`: `{quality['same_context_min']}`",
            f"- `same_context_avg`: `{quality['same_context_avg']}`",
            f"- `same_context_max`: `{quality['same_context_max']}`",
        ]
    )
    strength = summary.get("difference_strength_summary", {})
    if strength:
        lines.extend(
            [
                "",
                "## Difference Strength Summary",
                f"- `difference_strength_min`: `{strength.get('difference_strength_min', 0.0)}`",
                f"- `difference_strength_avg`: `{strength.get('difference_strength_avg', 0.0)}`",
                f"- `difference_strength_max`: `{strength.get('difference_strength_max', 0.0)}`",
            ]
        )

    speech_audio_counts = summary.get("speech_audio_quality_counts", {})
    if speech_audio_counts:
        lines.extend(["", "## Speech / Audio Quality Counts"])
        for key in (
            "speech_count",
            "high_quality_speech_count",
            "transcript_backed_speech_count",
            "non_speech_audio_event_count",
            "speech_rejected_as_too_generic_count",
            "audio_event_rejected_as_speech_only_count",
        ):
            lines.append(f"- `{key}`: `{speech_audio_counts.get(key, 0)}`")

    lines.extend(["", "## Automated Acceptance Checks"])
    for key, value in acceptance.items():
        lines.append(f"- `{key}`: `{'PASS' if value else 'FAIL'}`")
    verification_counts = summary.get("verification_counts", {})
    if verification_counts:
        lines.extend(["", "## Synthetic Route Counts"])
        for key in (
            "synthetic_visual_count",
            "synthetic_audio_count",
            "deterministic_audio_count",
            "foleycrafter_audio_count",
            "frieren_audio_count",
            "audio_remux_count",
            "speech_content_reject_count",
            "audio_stream_missing_reject_count",
            "visual_changed_in_audio_sample_reject_count",
            "audio_event_not_detected_reject_count",
        ):
            lines.append(f"- `{key}`: `{verification_counts.get(key, 0)}`")
        lines.extend(["", "## Edit Text / Difference Gate Counts"])
        for key in (
            "good_edit_text_count",
            "bad_edit_text_rejected_count",
            "caption_like_edit_rejected_count",
            "modality_leakage_rejected_count",
            "near_duplicate_without_delta_rejected_count",
            "visual_presence_contradiction_reject_count",
            "visible_text_without_ocr_reject_count",
            "audio_event_without_independent_audio_evidence_reject_count",
            "competing_difference_reject_count",
            "duplicate_target_reject_count",
        ):
            lines.append(f"- `{key}`: `{verification_counts.get(key, 0)}`")
        lines.extend(["", "## Verification Reject Counts"])
        for key in (
            "verification_passed_count",
            "verification_passed_rejected_count",
            "verification_override_accept_count",
            "caption_equivalent_reject_count",
            "missing_delta_reject_count",
            "difference_mismatch_reject_count",
            "edit_projection_reject_count",
            "edit_not_needed_reject_count",
            "speech_rejected_as_too_generic_count",
            "speech_rejected_for_missing_transcript_count",
            "audio_event_rejected_as_speech_only_count",
            "accepted_after_verification_count",
        ):
            lines.append(f"- `{key}`: `{verification_counts.get(key, 0)}`")
    lines.append("")
    lines.append("Manual review is still required for semantic correctness and target uniqueness.")
    return "\n".join(lines) + "\n"


def _validate_pilot_record(root: Path, record: dict[str, Any], line_number: int) -> list[str]:
    errors: list[str] = []
    sample_id = str(record.get("sample_id", "")).strip()
    if not sample_id:
        errors.append(f"pilot line {line_number}: sample_id is required")

    reference_video = str(record.get("reference_video", "")).strip()
    target_video = str(record.get("target_video", "")).strip()
    edit_text = str(record.get("edit_text", "")).strip()
    reference_caption = str(record.get("reference_caption", "")).strip()
    target_caption = str(record.get("target_caption", "")).strip()
    source_type = str(record.get("source_type", "natural")).strip() or "natural"
    if source_type not in ALLOWED_SOURCE_TYPES:
        errors.append(f"pilot line {line_number}: unsupported source_type={source_type!r}")
    if source_type == "synthetic_edit":
        errors.extend(f"pilot line {line_number}: {issue}" for issue in _known_pair_generation_issues(record))

    for field_name, value in (
        ("reference_video", reference_video),
        ("target_video", target_video),
        ("edit_text", edit_text),
        ("reference_caption", reference_caption),
        ("target_caption", target_caption),
    ):
        if not value:
            errors.append(f"pilot line {line_number}: {field_name} is required")

    if reference_video and target_video and reference_video == target_video:
        errors.append(f"pilot line {line_number}: reference_video and target_video must differ")

    for field_name, raw_value in (("reference_video", reference_video), ("target_video", target_video)):
        if raw_value:
            resolved = _resolve_under_root(root, raw_value)
            if not resolved.exists():
                errors.append(f"pilot line {line_number}: {field_name} does not exist: {raw_value}")

    modalities = record.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        errors.append(f"pilot line {line_number}: modalities must be a non-empty list")
    else:
        invalid_modalities = sorted({str(item).strip() for item in modalities} - ALLOWED_MODALITIES)
        if invalid_modalities:
            errors.append(f"pilot line {line_number}: invalid modalities={invalid_modalities}")

    difference = record.get("difference")
    if not isinstance(difference, dict):
        errors.append(f"pilot line {line_number}: difference must be an object")
    else:
        difference_type = str(difference.get("type", "")).strip()
        if difference_type not in ALLOWED_DIFFERENCE_TYPES:
            errors.append(f"pilot line {line_number}: unsupported difference.type={difference_type!r}")
        if difference_type == "speech":
            errors.append(f"pilot line {line_number}: speech difference type is disabled for final Omni-CVR samples")
        if not any(str(difference.get(key, "")).strip() for key in ("from", "to", "description")):
            errors.append(f"pilot line {line_number}: difference must include from/to/description")

    hard_negatives = record.get("hard_negatives")
    if not isinstance(hard_negatives, list) or not hard_negatives:
        errors.append(f"pilot line {line_number}: hard_negatives must be a non-empty list")
    else:
        normalized_negatives = [str(item).strip() for item in hard_negatives if str(item).strip()]
        if len(normalized_negatives) != len(hard_negatives):
            errors.append(f"pilot line {line_number}: hard_negatives must only contain non-empty strings")
        if reference_video and reference_video in normalized_negatives:
            errors.append(f"pilot line {line_number}: reference_video cannot appear in hard_negatives")
        if target_video and target_video in normalized_negatives:
            errors.append(f"pilot line {line_number}: target_video cannot appear in hard_negatives")
        for negative_path in normalized_negatives:
            resolved = _resolve_under_root(root, negative_path)
            if not resolved.exists():
                errors.append(f"pilot line {line_number}: hard_negative does not exist: {negative_path}")

    quality = record.get("quality")
    if not isinstance(quality, dict):
        errors.append(f"pilot line {line_number}: quality must be an object")
    else:
        for field_name in ("same_context_score", "edit_match_score", "target_uniqueness_score"):
            if field_name not in quality:
                errors.append(f"pilot line {line_number}: quality.{field_name} is required")
                continue
            try:
                float(quality[field_name])
            except (TypeError, ValueError):
                errors.append(f"pilot line {line_number}: quality.{field_name} must be numeric")

    source = record.get("source")
    if not isinstance(source, dict):
        errors.append(f"pilot line {line_number}: source must be an object")
    else:
        for field_name in ("platform", "url", "license_note"):
            if not str(source.get(field_name, "")).strip():
                errors.append(f"pilot line {line_number}: source.{field_name} is required")

    return errors


def _build_pair_candidates(*, root: Path, annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [annotation for annotation in annotations if _annotation_has_signal(annotation)]
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            forward = _score_ordered_pair(root=root, reference_annotation=left, target_annotation=right, annotations=eligible)
            backward = _score_ordered_pair(root=root, reference_annotation=right, target_annotation=left, annotations=eligible)
            chosen = _select_better_pair(forward, backward)
            if chosen is not None:
                candidates.append(chosen)
    candidates.sort(key=lambda item: (-item["composite_score"], item["proposal_id"]))
    return _select_diverse_pair_candidates(candidates, max_candidates=MAX_PAIR_CANDIDATES)


def _select_diverse_pair_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for difference_type, target_count in DIVERSE_PAIR_BUCKET_TARGETS.items():
        bucket_count = 0
        for candidate in candidates:
            if len(selected) >= max_candidates or bucket_count >= target_count:
                break
            if candidate["proposal_id"] in selected_ids:
                continue
            if candidate["primary_difference"]["type"] != difference_type:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["proposal_id"])
            bucket_count += 1

    for difference_type, target_count in DIVERSE_PAIR_BUCKET_TARGETS.items():
        bucket_count = sum(
            1 for candidate in selected if candidate["primary_difference"]["type"] == difference_type
        )
        for candidate in candidates:
            if len(selected) >= max_candidates or bucket_count >= target_count:
                break
            if candidate["proposal_id"] in selected_ids:
                continue
            if difference_type not in candidate.get("changed_difference_types", []):
                continue
            retargeted = _retarget_pair_candidate(candidate, difference_type)
            if retargeted is None:
                continue
            selected.append(retargeted)
            selected_ids.add(retargeted["proposal_id"])
            bucket_count += 1

    for candidate in candidates:
        if len(selected) >= max_candidates:
            break
        if candidate["proposal_id"] in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate["proposal_id"])

    return selected


def _retarget_pair_candidate(candidate: dict[str, Any], difference_type: str) -> dict[str, Any] | None:
    if candidate["primary_difference"]["type"] == difference_type:
        return candidate

    priority_order = (difference_type,) + tuple(item for item in PAIR_PRIORITY if item != difference_type)
    reference_annotation = candidate["reference_annotation"]
    target_annotation = candidate["target_annotation"]
    primary_difference = _detect_primary_difference(
        reference_annotation,
        target_annotation,
        priority_order=priority_order,
    )
    if primary_difference is None or primary_difference["type"] != difference_type:
        return None

    changed_types = primary_difference.pop("changed_types")
    same_context_score = _score_float(candidate["quality"].get("same_context_score"))
    edit_match_score = _edit_match_score(
        same_context_score=same_context_score,
        primary_difference_type=difference_type,
        changed_types=changed_types,
    )
    if edit_match_score < MIN_PAIR_EDIT_MATCH_SCORE:
        return None

    retargeted = dict(candidate)
    retargeted["primary_difference"] = primary_difference
    retargeted["changed_difference_types"] = list(changed_types)
    quality = dict(candidate["quality"])
    quality["edit_match_score"] = round(edit_match_score, 3)
    quality["difference_strength_score"] = round(
        _difference_strength_score(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            primary_difference=primary_difference,
            changed_types=changed_types,
        ),
        3,
    )
    quality["difference_type"] = primary_difference["type"]
    if primary_difference["type"] == "action":
        quality["action_evidence_score"] = _action_evidence_score(reference_annotation, target_annotation)
    if primary_difference["type"] == "speech":
        quality["speech_evidence_score"] = _speech_evidence_score(reference_annotation, target_annotation)
        quality["speech_specificity_score"] = _speech_specificity_score(reference_annotation, target_annotation)
        quality["speech_transcript_backed"] = 1.0 if _speech_is_transcript_backed(reference_annotation, target_annotation) else 0.0
        quality["has_audio_modality"] = 1.0
    if primary_difference["type"] == "audio_event":
        quality["non_speech_audio_event_score"] = _non_speech_audio_event_score(
            reference_annotation,
            target_annotation,
        )
        quality["has_audio_modality"] = 1.0
    retargeted["quality"] = quality
    retargeted["composite_score"] = _candidate_composite_score(quality, candidate["source_context"])
    retargeted["difference_evidence"] = _difference_evidence_from_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference=primary_difference,
    )
    return retargeted


def _maybe_reorient_candidate_for_model_fields(
    *,
    root: Path,
    candidate: dict[str, Any],
    model_fields: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    difference = model_fields.get("difference", {})
    if not isinstance(difference, dict):
        return candidate, model_fields, False
    if not _model_difference_prefers_reverse_direction(
        difference=difference,
        reference_annotation=candidate["reference_annotation"],
        target_annotation=candidate["target_annotation"],
    ):
        return candidate, model_fields, False

    swapped = _score_ordered_pair(
        root=root,
        reference_annotation=candidate["target_annotation"],
        target_annotation=candidate["reference_annotation"],
        annotations=annotations,
    )
    if swapped is None:
        return candidate, model_fields, False
    difference_type = str(difference.get("type", "")).strip()
    if swapped["primary_difference"]["type"] != difference_type and difference_type in swapped.get("changed_difference_types", []):
        retargeted = _retarget_pair_candidate(swapped, difference_type)
        if retargeted is not None:
            swapped = retargeted
    if swapped["primary_difference"]["type"] != difference_type:
        return candidate, model_fields, False

    oriented_fields = dict(model_fields)
    oriented_fields["reference_caption"] = str(swapped["reference_annotation"].get("summary", "")).strip() or str(
        model_fields.get("target_caption", "")
    ).strip()
    oriented_fields["target_caption"] = str(swapped["target_annotation"].get("summary", "")).strip() or str(
        model_fields.get("reference_caption", "")
    ).strip()
    reason = str(oriented_fields.get("proposal_reason", "")).strip()
    correction_reason = "direction corrected because difference.from/to matched target-to-reference evidence"
    oriented_fields["proposal_reason"] = f"{reason} {correction_reason}".strip()
    return swapped, oriented_fields, True


def _model_difference_prefers_reverse_direction(
    *,
    difference: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> bool:
    forward_score = _difference_direction_alignment_score(difference, reference_annotation, target_annotation)
    reverse_score = _difference_direction_alignment_score(difference, target_annotation, reference_annotation)
    return reverse_score >= 0.72 and reverse_score >= forward_score + 0.20


def _difference_direction_alignment_score(
    difference: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> float:
    difference_type = str(difference.get("type", "")).strip()
    if not difference_type:
        return 0.0
    priority_order = (difference_type,) + tuple(item for item in PAIR_PRIORITY if item != difference_type)
    detected = _detect_primary_difference(
        reference_annotation,
        target_annotation,
        priority_order=priority_order,
    )
    if not detected or detected.get("type") != difference_type:
        return 0.0
    from_score = _difference_value_similarity(
        str(difference.get("from", "")),
        str(detected.get("from", "")),
    )
    to_score = _difference_value_similarity(
        str(difference.get("to", "")),
        str(detected.get("to", "")),
    )
    return round((from_score + to_score) / 2.0, 3)


def _difference_value_similarity(left: str, right: str) -> float:
    left_norm = _normalized_phrase(left)
    right_norm = _normalized_phrase(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.95
    left_absent = left_norm.startswith("no ") or left_norm in {"none", "no distinctive audio event"}
    right_absent = right_norm.startswith("no ") or right_norm in {"none", "no distinctive audio event"}
    if left_absent != right_absent:
        return 0.0
    if left_absent and right_absent:
        return 1.0
    left_tokens = _tokenize_text(_strip_presence_prefix(left_norm))
    right_tokens = _tokenize_text(_strip_presence_prefix(right_norm))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / max(1, min(len(left_tokens), len(right_tokens)))


def _repair_pair_model_fields(
    *,
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    repaired = dict(model_fields)
    if str(repaired.get("difference", {}).get("type", "")).strip() == "audio_event":
        repaired = _normalize_audio_event_model_fields(repaired)
    current_quality = _edit_text_quality_payload(
        edit_text=str(repaired.get("edit_text", "")),
        difference=repaired.get("difference", {}),
        modalities=repaired.get("modalities", []),
        reference_caption=str(repaired.get("reference_caption", "")),
        target_caption=str(repaired.get("target_caption", "")),
    )
    if _edit_text_quality_passes(current_quality):
        return repaired

    template_edit = _build_fallback_edit_text(repaired.get("difference", {}))
    template_quality = _edit_text_quality_payload(
        edit_text=template_edit,
        difference=repaired.get("difference", {}),
        modalities=repaired.get("modalities", []),
        reference_caption=str(reference_annotation.get("summary", "")),
        target_caption=str(target_annotation.get("summary", "")),
    )
    if _edit_text_quality_passes(template_quality):
        repaired["edit_text"] = template_edit
        reason = str(repaired.get("proposal_reason", "")).strip()
        repaired["proposal_reason"] = f"{reason} edit_text normalized from evidence template".strip()
    return repaired


def _edit_text_quality_payload(
    *,
    edit_text: str,
    difference: dict[str, Any],
    modalities: list[str] | tuple[str, ...] | Any,
    reference_caption: str,
    target_caption: str,
) -> dict[str, Any]:
    text = str(edit_text).strip()
    tokens = _tokenize_text(text)
    difference_type = str(difference.get("type", "")).strip()
    modality_set = {str(item).strip() for item in modalities if str(item).strip()} if isinstance(modalities, (list, tuple, set)) else set()
    bad_patterns: list[str] = []

    if not text:
        bad_patterns.append("edit_text is empty")
    if any(phrase in _normalized_phrase(text) for phrase in GENERIC_EDIT_TEXT_PHRASES):
        bad_patterns.append("edit_text is too broad")

    first_token = _normalized_phrase(text).split()[0] if _normalized_phrase(text).split() else ""
    is_imperative_edit = first_token in EDIT_TEXT_START_VERBS or first_token in EDIT_ACTION_VERBS
    if not is_imperative_edit:
        bad_patterns.append("edit_text is not an imperative edit")

    matches_difference_type = _edit_text_matches_difference_type(
        edit_text=text,
        difference=difference,
        modalities=modality_set,
    )
    if not matches_difference_type:
        bad_patterns.append(f"edit_text does not match difference type {difference_type or 'unknown'}")

    single_change = _edit_text_single_change(text, difference_type)
    if not single_change:
        bad_patterns.append("edit_text appears to contain multiple unrelated changes")

    not_caption_like = _edit_text_not_caption_like(
        edit_text=text,
        reference_caption=reference_caption,
        target_caption=target_caption,
    )
    if not not_caption_like:
        bad_patterns.append("edit_text reads like a caption instead of an edit instruction")

    no_modality_leakage = _edit_text_no_modality_leakage(text, modalities, difference_type)
    if not no_modality_leakage:
        bad_patterns.append("edit_text mentions a modality outside the declared difference")

    malformed_presence = _edit_text_has_malformed_presence(text)
    if malformed_presence:
        bad_patterns.append("edit_text uses malformed object-presence wording")

    score = 1.0
    for failed, penalty in (
        (not bool(text), 0.50),
        (not is_imperative_edit, 0.30),
        (not matches_difference_type, 0.35),
        (not single_change, 0.25),
        (not not_caption_like, 0.35),
        (not no_modality_leakage, 0.35),
        (malformed_presence, 0.35),
        (any(phrase in _normalized_phrase(text) for phrase in GENERIC_EDIT_TEXT_PHRASES), 0.30),
    ):
        if failed:
            score -= penalty
    score = round(max(0.0, min(1.0, score)), 3)
    return {
        "score": score,
        "is_imperative_edit": is_imperative_edit,
        "matches_difference_type": matches_difference_type,
        "single_change": single_change,
        "not_caption_like": not_caption_like,
        "no_modality_leakage": no_modality_leakage,
        "bad_patterns": bad_patterns,
    }


def _edit_text_quality_passes(payload: dict[str, Any]) -> bool:
    return bool(
        _score_float(payload.get("score")) >= MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE
        and bool(payload.get("is_imperative_edit"))
        and bool(payload.get("matches_difference_type"))
        and bool(payload.get("single_change"))
        and bool(payload.get("not_caption_like"))
        and bool(payload.get("no_modality_leakage"))
        and not payload.get("bad_patterns")
    )


def _edit_text_matches_difference_type(
    *,
    edit_text: str,
    difference: dict[str, Any],
    modalities: set[str],
) -> bool:
    tokens = _tokenize_text(edit_text)
    difference_type = str(difference.get("type", "")).strip()
    from_tokens = _tokenize_text(str(difference.get("from", "")))
    to_tokens = _tokenize_text(str(difference.get("to", "")))
    delta_tokens = from_tokens | to_tokens
    if not difference_type:
        return bool(tokens & EDIT_ACTION_VERBS)
    if difference_type in {"object_count", "object_presence"}:
        leaked_modality_tokens = {"audio", "sound", "sounds", "speech", "transcript", "spoken", "voiceover", "narration"}
        return bool(tokens & delta_tokens) and not bool(tokens & leaked_modality_tokens)
    if difference_type == "action":
        return bool({"action", "gesture", "doing"} & tokens or tokens & delta_tokens or tokens & EDIT_ACTION_VERBS)
    if difference_type == "audio_event":
        return bool("audio" in modalities and tokens & EDIT_TEXT_AUDIO_TOKENS) and not (
            _is_speech_only_or_absence_audio_phrase(edit_text) or bool(tokens & EDIT_TEXT_VISUAL_LEAK_TOKENS)
        )
    if difference_type == "speech":
        return bool("audio" in modalities and tokens & EDIT_TEXT_SPEECH_TOKENS)
    if difference_type == "visible_text":
        return bool(tokens & EDIT_TEXT_VISIBLE_TEXT_TOKENS)
    if difference_type in {"attribute", "scene"}:
        return bool(tokens & EDIT_ACTION_VERBS or tokens & delta_tokens)
    return bool(tokens & EDIT_ACTION_VERBS)


def _edit_text_single_change(edit_text: str, difference_type: str) -> bool:
    normalized = _normalized_phrase(edit_text)
    if not normalized:
        return False
    if len(normalized.split()) > 32:
        return False
    multi_markers = ("and also", "as well as", " plus ")
    if any(marker in normalized for marker in multi_markers):
        return False
    tokens = _tokenize_text(edit_text)
    modality_hits = 0
    if tokens & EDIT_TEXT_AUDIO_TOKENS:
        modality_hits += 1
    if tokens & EDIT_TEXT_SPEECH_TOKENS:
        modality_hits += 1
    if tokens & EDIT_TEXT_VISIBLE_TEXT_TOKENS:
        modality_hits += 1
    return not (difference_type not in {"integrated", "speech"} and modality_hits > 1)


def _edit_text_not_caption_like(*, edit_text: str, reference_caption: str, target_caption: str) -> bool:
    text = edit_text.strip()
    if not text:
        return False
    text_tokens = _tokenize_text(text)
    if len(text_tokens) > EDIT_TEXT_CAPTION_MAX_TOKENS:
        return False
    normalized_text = _normalized_phrase(text)
    if len(text_tokens) <= 8:
        return True
    for caption in (reference_caption, target_caption):
        normalized_caption = _normalized_phrase(caption)
        caption_tokens = _tokenize_text(caption)
        if not caption_tokens:
            continue
        if normalized_text and normalized_text in normalized_caption:
            return False
        if _jaccard(text_tokens, caption_tokens) >= 0.72:
            return False
    return True


def _edit_text_no_modality_leakage(
    edit_text: str,
    modalities: list[str] | tuple[str, ...] | Any,
    difference_type: str,
) -> bool:
    modality_set = {str(item).strip() for item in modalities if str(item).strip()} if isinstance(modalities, (list, tuple, set)) else set()
    tokens = _tokenize_text(edit_text)
    if difference_type == "audio_event":
        if "audio" not in modality_set:
            return False
        if tokens & EDIT_TEXT_VISUAL_LEAK_TOKENS:
            return False
        if _is_speech_only_or_absence_audio_phrase(edit_text):
            return False
        return True
    if difference_type == "speech":
        return "audio" in modality_set and not bool(tokens & (NON_SPEECH_AUDIO_TOKENS - {"voice"}))
    if difference_type == "visible_text":
        return not bool((tokens & EDIT_TEXT_AUDIO_TOKENS) or (tokens & GENERIC_SPEECH_TOKENS))
    if difference_type in VISUAL_DIFFERENCE_TYPES and tokens & {"audio", "sound", "sounds", "speech", "transcript", "spoken"}:
        return False
    return True


def _edit_text_has_malformed_presence(edit_text: str) -> bool:
    normalized = _normalized_phrase(edit_text)
    return bool(normalized.startswith("change no ") and " into " in normalized and re.search(r"\b\d+\b", normalized))


def _observable_difference_gate(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
    visual_near_duplicate_score: Any,
) -> dict[str, Any]:
    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    near_duplicate_score = _score_float(visual_near_duplicate_score)
    if near_duplicate_score >= MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE:
        near_duplicate_risk = "high"
    elif near_duplicate_score >= 0.97:
        near_duplicate_risk = "medium"
    else:
        near_duplicate_risk = "low"

    if difference_type not in VISUAL_DIFFERENCE_TYPES:
        return {
            "passed": True,
            "type": difference_type,
            "frame_backed": False,
            "reference_missing": [],
            "target_present": [],
            "reference_value": from_value,
            "target_value": to_value,
            "supporting_fields": ["non_visual_difference_type"],
            "near_duplicate_risk": near_duplicate_risk,
            "visual_near_duplicate_score": near_duplicate_score,
            "failure_reason": "",
        }

    supporting_fields: list[str] = []
    reference_missing: list[str] = []
    target_present: list[str] = []
    reference_evidence: list[str] = []
    target_evidence: list[str] = []
    reference_counts = _normalize_object_counts(reference_annotation.get("object_counts", {}))
    target_counts = _normalize_object_counts(target_annotation.get("object_counts", {}))
    reference_actions = _normalize_list(reference_annotation.get("actions", []))
    target_actions = _normalize_list(target_annotation.get("actions", []))
    reference_text = _annotation_observable_text(reference_annotation)
    target_text = _annotation_observable_text(target_annotation)
    conflict_reasons: list[str] = []

    if difference_type in {"object_count", "object_presence"}:
        label = _strip_presence_prefix(to_value) or _strip_presence_prefix(from_value)
        canonical_label = _canonical_object_label(label)
        reference_mentions_label = _annotation_mentions_presence_label(reference_annotation, label)
        target_mentions_label = _annotation_mentions_presence_label(target_annotation, label)
        reference_label_count = _object_count_for_label(reference_counts, label)
        target_label_count = _object_count_for_label(target_counts, label)
        if label and reference_label_count != target_label_count:
            supporting_fields.append("object_counts")
            reference_evidence.append(f"object_counts:{reference_label_count}")
            target_evidence.append(f"object_counts:{target_label_count}")
        if label and not reference_mentions_label and target_mentions_label:
            reference_missing.append(label)
            target_present.append(label)
            supporting_fields.append("summary")
        if label and _presence_value_claims_absent(from_value) and reference_mentions_label:
            conflict_reasons.append(f"reference already appears to contain equivalent object: {label}")
        if label and _presence_value_claims_absent(to_value) and target_mentions_label:
            conflict_reasons.append(f"target still appears to contain {label}")
        if (
            canonical_label in BACKGROUND_DECOR_OBJECTS
            and target_mentions_label
            and not _annotation_has_label_frame_evidence(target_annotation, label)
        ):
            conflict_reasons.append(f"background decor object lacks frame-level evidence: {label}")
    elif difference_type == "action":
        if _first_unique(reference_actions, target_actions) or _first_unique(target_actions, reference_actions):
            supporting_fields.append("actions")
            reference_evidence.append(_first_unique(reference_actions, target_actions))
            target_evidence.append(_first_unique(target_actions, reference_actions))
        if from_value and _text_mentions_phrase(reference_text, from_value):
            supporting_fields.append("storyline")
        if to_value and _text_mentions_phrase(target_text, to_value):
            supporting_fields.append("events")
    elif difference_type == "visible_text":
        reference_visible_text = _visible_text_values(reference_annotation)
        target_visible_text = _visible_text_values(target_annotation)
        if reference_visible_text != target_visible_text:
            supporting_fields.append("visible_text")
        if from_value and not _text_collection_mentions_phrase(reference_visible_text, from_value):
            conflict_reasons.append(f"visible_text lacks reference OCR/frame evidence for {from_value}")
        if to_value and not _text_collection_mentions_phrase(target_visible_text, to_value):
            conflict_reasons.append(f"visible_text lacks target OCR/frame evidence for {to_value}")
        reference_evidence.extend(reference_visible_text)
        target_evidence.extend(target_visible_text)
    elif difference_type == "attribute":
        if _normalize_list(reference_annotation.get("attributes", [])) != _normalize_list(target_annotation.get("attributes", [])):
            supporting_fields.append("attributes")
        if to_value and _text_mentions_phrase(target_text, to_value):
            supporting_fields.append("summary")
    elif difference_type == "scene":
        if str(reference_annotation.get("scene", "")).strip() != str(target_annotation.get("scene", "")).strip():
            supporting_fields.append("scene")

    supporting_fields = _dedupe_strings(supporting_fields)
    competing_reasons = _competing_difference_reasons(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference_type=difference_type,
    )
    conflict_reasons.extend(competing_reasons)
    hard_fields = {"object_counts", "actions", "events", "visible_text", "attributes", "scene"}
    frame_backed = bool(set(supporting_fields) & hard_fields)
    passed = bool(supporting_fields)
    if conflict_reasons:
        passed = False
    if near_duplicate_risk == "high" and not bool(set(supporting_fields) & hard_fields):
        passed = False
    if passed:
        failure_reason = ""
    elif conflict_reasons:
        failure_reason = "; ".join(_dedupe_strings(conflict_reasons))
    else:
        failure_reason = "no observable annotation delta supports this visual edit"
    return {
        "passed": passed,
        "type": difference_type,
        "frame_backed": frame_backed,
        "reference_missing": _dedupe_strings(reference_missing),
        "target_present": _dedupe_strings(target_present),
        "reference_evidence": _dedupe_strings(reference_evidence),
        "target_evidence": _dedupe_strings(target_evidence),
        "reference_value": from_value,
        "target_value": to_value,
        "supporting_fields": supporting_fields,
        "near_duplicate_risk": near_duplicate_risk,
        "visual_near_duplicate_score": near_duplicate_score,
        "failure_reason": failure_reason,
    }


def _annotation_observable_text(annotation: dict[str, Any]) -> str:
    texts: list[str] = []
    for field in ("summary", "scene"):
        value = str(annotation.get(field, "")).strip()
        if value:
            texts.append(value)
    texts.extend(_normalize_list(annotation.get("storyline", [])))
    texts.extend(_normalize_list(annotation.get("visible_text", [])))
    for event in annotation.get("events", []):
        if isinstance(event, dict):
            texts.extend(_normalize_list([event.get("visual", "")]))
            texts.extend(_normalize_list(event.get("actions", [])))
            texts.extend(_normalize_list(event.get("objects", [])))
        else:
            texts.extend(_normalize_list([event]))
    return " ".join(texts)


def _visible_text_values(annotation: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("visible_text", "on_screen_text", "ocr_text"):
        values.extend(_normalize_list(annotation.get(field, [])))
    for event in annotation.get("events", []):
        if not isinstance(event, dict):
            continue
        values.extend(_normalize_list(event.get("visible_text", [])))
        values.extend(_normalize_list(event.get("on_screen_text", [])))
        values.extend(_normalize_list(event.get("ocr_text", [])))
    return _dedupe_strings(values)


def _text_collection_mentions_phrase(values: list[str], phrase: str) -> bool:
    if not phrase:
        return True
    return any(_text_mentions_phrase(value, phrase) for value in values)


def _competing_difference_reasons(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference_type: str,
) -> list[str]:
    reasons: list[str] = []
    if primary_difference_type != "action" and _strong_action_delta(reference_annotation, target_annotation):
        reasons.append("single_main_difference failed: competing stronger action difference")
    if primary_difference_type not in {"visible_text", "speech"} and _strong_visible_text_delta(reference_annotation, target_annotation):
        reasons.append("single_main_difference failed: competing stronger visible_text difference")
    if primary_difference_type not in {"speech", "visible_text"} and _strong_speech_delta(reference_annotation, target_annotation):
        reasons.append("single_main_difference failed: competing stronger speech difference")
    return reasons


def _strong_action_delta(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> bool:
    reference_actions = _action_terms_from_annotation(reference_annotation)
    target_actions = _action_terms_from_annotation(target_annotation)
    if not reference_actions or not target_actions:
        return False
    if not _first_unique(reference_actions, target_actions) or not _first_unique(target_actions, reference_actions):
        return False
    return _list_delta_strength(reference_actions, target_actions) >= MIN_COMPETING_DIFFERENCE_STRENGTH


def _strong_visible_text_delta(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> bool:
    reference_text = _visible_text_values(reference_annotation)
    target_text = _visible_text_values(target_annotation)
    if not reference_text or not target_text:
        return False
    if not _first_unique(reference_text, target_text) or not _first_unique(target_text, reference_text):
        return False
    return _list_delta_strength(reference_text, target_text) >= MIN_COMPETING_DIFFERENCE_STRENGTH


def _strong_speech_delta(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> bool:
    reference_speech = _speech_texts_from_annotation(reference_annotation)
    target_speech = _speech_texts_from_annotation(target_annotation)
    if not reference_speech or not target_speech:
        return False
    if not _first_unique(reference_speech, target_speech) or not _first_unique(target_speech, reference_speech):
        return False
    return _list_delta_strength(reference_speech, target_speech) >= MIN_COMPETING_DIFFERENCE_STRENGTH


def _annotation_mentions_value(annotation: dict[str, Any], value: str) -> bool:
    if not value:
        return False
    texts = [
        _annotation_observable_text(annotation),
        " ".join(_normalize_object_counts(annotation.get("object_counts", {})).keys()),
        " ".join(_normalize_list(annotation.get("subjects", []))),
    ]
    return any(_text_mentions_phrase(text, value) for text in texts)


def _annotation_mentions_presence_label(annotation: dict[str, Any], label: str) -> bool:
    if not label:
        return False
    if _annotation_mentions_value(annotation, label):
        return True
    for alias in _object_label_aliases(label):
        if alias != _normalized_object_label(label) and _annotation_mentions_value(annotation, alias):
            return True
    label_tokens = _tokenize_text(label)
    if not label_tokens or not (label_tokens & GENERIC_HUMAN_GROUP_TOKENS):
        return False
    texts = [
        _annotation_observable_text(annotation),
        " ".join(_normalize_object_counts(annotation.get("object_counts", {})).keys()),
        " ".join(_normalize_list(annotation.get("subjects", []))),
    ]
    annotation_tokens = _tokenize_text(" ".join(texts))
    if not (annotation_tokens & GENERIC_HUMAN_GROUP_TOKENS):
        return False
    context_tokens = {
        token
        for token in label_tokens
        if token not in GENERIC_HUMAN_GROUP_TOKENS and not token.isdigit()
    }
    return not context_tokens or bool(context_tokens & annotation_tokens)


def _normalized_object_label(value: str) -> str:
    label = _strip_presence_prefix(value)
    normalized_tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(label.lower()):
        if token.isdigit() or token in OBJECT_LABEL_STOPWORDS:
            continue
        normalized_tokens.append(_singular_object_token(token))
    return " ".join(normalized_tokens)


def _singular_object_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _canonical_object_label(value: str) -> str:
    normalized = _normalized_object_label(value)
    if not normalized:
        return ""
    for alias_group in OBJECT_ALIAS_GROUPS:
        normalized_group = [_normalized_object_label(alias) for alias in alias_group]
        if normalized in normalized_group:
            return normalized_group[0]
    return normalized


def _object_label_aliases(label: str) -> list[str]:
    normalized = _normalized_object_label(label)
    canonical = _canonical_object_label(label)
    aliases = [normalized, canonical]
    for alias_group in OBJECT_ALIAS_GROUPS:
        normalized_group = [_normalized_object_label(alias) for alias in alias_group]
        if canonical in normalized_group or normalized in normalized_group:
            aliases.extend(normalized_group)
    return _dedupe_strings([alias for alias in aliases if alias])


def _annotation_has_label_frame_evidence(annotation: dict[str, Any], label: str) -> bool:
    aliases = _object_label_aliases(label)
    if not aliases:
        return False
    for container_name in ("events", "storyline"):
        container = annotation.get(container_name, [])
        if not isinstance(container, list):
            continue
        for item in container:
            if isinstance(item, dict):
                values = [item.get("visual", ""), item.get("description", "")]
                values.extend(_normalize_list(item.get("objects", [])))
                values.extend(_normalize_list(item.get("actions", [])))
            else:
                values = [item]
            text = " ".join(str(value) for value in values)
            if any(_text_mentions_phrase(text, alias) for alias in aliases):
                return True
    return False


def _presence_value_claims_absent(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if normalized.startswith("no "):
        return True
    count = _first_integer(normalized)
    return count == 0


def _object_count_for_label(counts: dict[str, int], label: str) -> int:
    label_tokens = _tokenize_text(label)
    canonical_label = _canonical_object_label(label)
    for key, count in counts.items():
        if canonical_label and _canonical_object_label(key) == canonical_label:
            return count
        key_tokens = _tokenize_text(key)
        if label_tokens and key_tokens and (label_tokens <= key_tokens or key_tokens <= label_tokens):
            return count
    return 0


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in result:
            result.append(value)
    return result


def _apply_structured_gate_quality(
    quality: dict[str, Any],
    *,
    edit_text_quality: dict[str, Any],
    observable_difference: dict[str, Any],
) -> None:
    quality["edit_text_quality_score"] = _score_float(edit_text_quality.get("score"))
    quality["edit_text_is_imperative"] = 1.0 if edit_text_quality.get("is_imperative_edit") else 0.0
    quality["edit_text_matches_difference_type"] = 1.0 if edit_text_quality.get("matches_difference_type") else 0.0
    quality["edit_text_single_change"] = 1.0 if edit_text_quality.get("single_change") else 0.0
    quality["edit_text_not_caption_like"] = 1.0 if edit_text_quality.get("not_caption_like") else 0.0
    quality["edit_text_no_modality_leakage"] = 1.0 if edit_text_quality.get("no_modality_leakage") else 0.0
    quality["observable_difference_passed"] = 1.0 if observable_difference.get("passed") else 0.0
    quality["observable_difference_frame_backed"] = 1.0 if observable_difference.get("frame_backed") else 0.0
    quality["near_duplicate_without_delta"] = 1.0 if observable_difference.get("near_duplicate_risk") == "high" and not observable_difference.get("passed") else 0.0


def _competing_difference_gate(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
) -> dict[str, Any]:
    reasons = _competing_difference_reasons(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference_type=str(difference.get("type", "")).strip(),
    )
    return {
        "passed": not reasons,
        "failure_reason": "; ".join(_dedupe_strings(reasons)),
    }


def _audio_event_independent_evidence_gate(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
) -> dict[str, Any]:
    if str(difference.get("type", "")).strip() != "audio_event":
        return {
            "passed": True,
            "reference_evidence": [],
            "target_evidence": [],
            "supporting_fields": [],
            "failure_reason": "",
        }
    reference_terms = _non_speech_audio_terms(reference_annotation)
    target_terms = _non_speech_audio_terms(target_annotation)
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    from_absent = _is_audio_absence_edit_phrase(from_value)
    to_absent = _is_audio_absence_edit_phrase(to_value)
    reference_supported = (
        (from_absent and not _audio_terms_match(reference_terms, to_value))
        or _audio_terms_match(reference_terms, from_value)
    )
    target_supported = (
        (to_absent and not _audio_terms_match(target_terms, from_value))
        or _audio_terms_match(target_terms, to_value)
    )
    terms_differ = bool(_first_unique(reference_terms, target_terms) or _first_unique(target_terms, reference_terms))
    passed = bool(reference_terms or target_terms) and reference_supported and target_supported and terms_differ
    failure_reason = ""
    if not passed:
        failure_reason = "audio_event lacks independent non-speech audio evidence"
    return {
        "passed": passed,
        "reference_evidence": reference_terms,
        "target_evidence": target_terms,
        "supporting_fields": ["audio_events"] if reference_terms or target_terms else [],
        "failure_reason": failure_reason,
    }


def _is_audio_absence_edit_phrase(value: str) -> bool:
    return _is_non_speech_absence_audio_phrase(value) or _absence_like_phrase(value)


def _audio_terms_match(terms: list[str], phrase: str) -> bool:
    if not phrase:
        return True
    phrase_tokens = _tokenize_text(phrase)
    if not phrase_tokens:
        return False
    for term in terms:
        term_tokens = _tokenize_text(term)
        if not term_tokens:
            continue
        if _text_mentions_phrase(term, phrase) or _text_mentions_phrase(phrase, term):
            return True
        if _jaccard(phrase_tokens, term_tokens) >= 0.5:
            return True
    return False


def _ensure_structured_gate_fields(
    record: dict[str, Any],
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    record = dict(record)
    quality = dict(record.get("quality", {}))
    edit_text_quality = dict(record.get("edit_text_quality") or {})
    if not edit_text_quality:
        edit_text_quality = _edit_text_quality_payload(
            edit_text=str(record.get("edit_text", "")),
            difference=record.get("difference", {}),
            modalities=record.get("modalities", []),
            reference_caption=str(record.get("reference_caption", "")),
            target_caption=str(record.get("target_caption", "")),
        )
    observable_difference = dict(record.get("observable_difference") or {})
    if not observable_difference:
        observable_difference = _observable_difference_gate(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            difference=record.get("difference", {}),
            visual_near_duplicate_score=quality.get("visual_near_duplicate_score"),
        )
    _apply_structured_gate_quality(
        quality,
        edit_text_quality=edit_text_quality,
        observable_difference=observable_difference,
    )
    competing_difference = dict(record.get("competing_difference") or {})
    if not competing_difference:
        competing_difference = _competing_difference_gate(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            difference=record.get("difference", {}),
        )
    audio_event_evidence = dict(record.get("audio_event_evidence") or {})
    if not audio_event_evidence:
        audio_event_evidence = _audio_event_independent_evidence_gate(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            difference=record.get("difference", {}),
        )
    quality["competing_difference_passed"] = 1.0 if competing_difference.get("passed") else 0.0
    quality["audio_event_independent_evidence_passed"] = 1.0 if audio_event_evidence.get("passed") else 0.0
    verification = dict(record.get("verification", {}))
    if verification:
        existing_check = dict(verification.get("edit_text_quality_check", {}))
        edit_necessity = dict(verification.get("edit_necessity", {}))
        judge = dict(record.get("judge", {}))
        reference_does_not_satisfy = not _boolish(
            edit_necessity.get(
                "reference_satisfies_edit",
                judge.get("reference_satisfies_edit", False),
            )
        )
        target_satisfies = _boolish(
            edit_necessity.get(
                "target_satisfies_edit",
                judge.get("target_satisfies_edit", False),
            )
        )
        local_check = {
            "not_caption_like": bool(edit_text_quality.get("not_caption_like")),
            "matches_modality": bool(edit_text_quality.get("no_modality_leakage")),
            "single_primary_difference": bool(edit_text_quality.get("single_change")),
            "reference_does_not_satisfy": reference_does_not_satisfy,
            "target_satisfies": target_satisfies,
            "score": _score_float(edit_text_quality.get("score")),
            "failure_reason": "; ".join(edit_text_quality.get("bad_patterns", [])),
        }
        local_reason = str(local_check.get("failure_reason", "")).strip()
        model_reason = str(existing_check.get("failure_reason", "")).strip()
        failure_reason = local_reason
        if local_reason and model_reason:
            failure_reason = f"{local_reason}; model verifier note: {model_reason}"
        verification["edit_text_quality_check"] = {
            "not_caption_like": local_check["not_caption_like"],
            "matches_modality": local_check["matches_modality"],
            "single_primary_difference": local_check["single_primary_difference"],
            "reference_does_not_satisfy": local_check["reference_does_not_satisfy"],
            "target_satisfies": local_check["target_satisfies"],
            "score": local_check["score"],
            "failure_reason": failure_reason,
        }
        _sync_observable_difference_failure(
            verification,
            observable_difference=observable_difference,
        )
        _sync_synthetic_audio_verification_from_evidence(
            record,
            verification=verification,
            audio_event_evidence=audio_event_evidence,
        )
        _sync_local_gate_failure(
            verification,
            passed=bool(competing_difference.get("passed", True)),
            reason=str(competing_difference.get("failure_reason", "")).strip(),
        )
        _sync_local_gate_failure(
            verification,
            passed=bool(audio_event_evidence.get("passed", True)),
            reason=str(audio_event_evidence.get("failure_reason", "")).strip(),
        )
        verification["passed"] = _verification_accepts(verification)
        verification["failures"] = _verification_failures(verification)
        record["verification"] = verification
    record["quality"] = quality
    record["edit_text_quality"] = edit_text_quality
    record["observable_difference"] = observable_difference
    record["competing_difference"] = competing_difference
    record["audio_event_evidence"] = audio_event_evidence
    return record


def _sync_synthetic_audio_verification_from_evidence(
    record: dict[str, Any],
    *,
    verification: dict[str, Any],
    audio_event_evidence: dict[str, Any],
) -> None:
    if str(record.get("source_type", "")).strip() != "synthetic_edit":
        return
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    if not _is_audio_synthetic_route(_synthetic_generation_route(generation)):
        return
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    if str(difference.get("type", "")).strip() != "audio_event":
        return
    if not _boolish(audio_event_evidence.get("passed")):
        return

    expected_event = _synthetic_audio_expected_event(record)
    reason = (
        "synthetic audio plan and independent audio evidence confirm "
        f"target contains the requested non-speech audio event: {expected_event}"
    )
    caption_delta = verification.setdefault("caption_delta", {})
    caption_delta["caption_equivalent"] = False
    caption_delta["has_concrete_difference"] = True
    caption_delta["difference_matches_edit"] = True
    differences = _normalize_list(caption_delta.get("concrete_differences", []))
    if expected_event and not any(_text_mentions_phrase(item, expected_event) for item in differences):
        differences.append(f"target contains {expected_event}; reference does not")
    caption_delta["concrete_differences"] = differences
    caption_delta["reason"] = _append_reason(caption_delta.get("reason"), reason)

    edit_projection = verification.setdefault("edit_projection", {})
    edit_projection["target_matches_projection"] = True
    edit_projection["score"] = max(_score_float(edit_projection.get("score")), 0.9)
    edit_projection["reason"] = _append_reason(edit_projection.get("reason"), reason)

    edit_necessity = verification.setdefault("edit_necessity", {})
    edit_necessity["edit_needed"] = True
    edit_necessity["reference_satisfies_edit"] = False
    edit_necessity["target_satisfies_edit"] = True
    edit_necessity["score"] = max(_score_float(edit_necessity.get("score")), 0.9)
    edit_necessity["reason"] = _append_reason(edit_necessity.get("reason"), reason)


def _sync_observable_difference_failure(
    verification: dict[str, Any],
    *,
    observable_difference: dict[str, Any],
) -> None:
    if _boolish(observable_difference.get("passed", True)):
        return
    reason = str(observable_difference.get("failure_reason", "")).strip()
    if not reason:
        reason = "observable_difference gate found no concrete visual delta evidence"
    reason = f"observable_difference gate failed: {reason}"
    _sync_local_gate_failure(verification, passed=False, reason=reason)


def _sync_local_gate_failure(
    verification: dict[str, Any],
    *,
    passed: bool,
    reason: str,
) -> None:
    if passed:
        return
    reason = reason.strip() or "local quality gate failed"
    existing_reason = str(verification.get("observable_difference_failure", "")).strip()
    if existing_reason:
        verification["observable_difference_failure"] = _append_reason(existing_reason, reason)
    else:
        verification["observable_difference_failure"] = reason

    caption_delta = verification.setdefault("caption_delta", {})
    caption_delta["has_concrete_difference"] = False
    caption_delta["difference_matches_edit"] = False
    caption_delta["reason"] = _append_reason(caption_delta.get("reason"), reason)

    edit_projection = verification.setdefault("edit_projection", {})
    edit_projection["target_matches_projection"] = False
    edit_projection["score"] = min(_score_float(edit_projection.get("score")), 0.0)
    edit_projection["reason"] = _append_reason(edit_projection.get("reason"), reason)

    edit_necessity = verification.setdefault("edit_necessity", {})
    edit_necessity["edit_needed"] = False
    if "reference already appears to contain" in reason:
        edit_necessity["reference_satisfies_edit"] = True
    if "target still appears to contain" in reason:
        edit_necessity["target_satisfies_edit"] = False
    edit_necessity["score"] = min(_score_float(edit_necessity.get("score")), 0.0)
    edit_necessity["reason"] = _append_reason(edit_necessity.get("reason"), reason)


def _prepare_record_for_acceptance(
    record: dict[str, Any],
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    record = _ensure_structured_gate_fields(
        record,
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
    )
    judge = dict(record.get("judge", {}))
    verification = record.get("verification", {})
    heuristic_quality = record.get("heuristic_quality")
    if not isinstance(heuristic_quality, dict) or not heuristic_quality:
        heuristic_quality = record.get("quality", {})
    local_gate_quality = dict(record.get("quality", {}))
    record["quality"] = _effective_pair_quality(judge, verification, heuristic_quality)
    _carry_local_gate_quality(record["quality"], local_gate_quality)
    return record


def _carry_local_gate_quality(target_quality: dict[str, Any], source_quality: dict[str, Any]) -> None:
    for key in (
        "edit_text_quality_score",
        "edit_text_is_imperative",
        "edit_text_matches_difference_type",
        "edit_text_single_change",
        "edit_text_not_caption_like",
        "edit_text_no_modality_leakage",
        "observable_difference_passed",
        "observable_difference_frame_backed",
        "near_duplicate_without_delta",
        "competing_difference_passed",
        "audio_event_independent_evidence_passed",
        "synthetic_context_override",
    ):
        if key in source_quality:
            target_quality[key] = source_quality[key]


def _quality_for_model_fields(
    *,
    base_quality: dict[str, Any],
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    if str(model_fields.get("difference", {}).get("type", "")).strip() == "audio_event":
        model_fields = _normalize_audio_event_model_fields(model_fields)
    quality = dict(base_quality)
    difference = model_fields.get("difference", {})
    difference_type = str(difference.get("type", "")).strip()
    quality["difference_type"] = difference_type
    quality["has_audio_modality"] = 1.0 if "audio" in set(model_fields.get("modalities", [])) else 0.0
    if difference_type == "speech":
        quality["speech_evidence_score"] = _speech_evidence_score(reference_annotation, target_annotation)
        quality["speech_specificity_score"] = _speech_specificity_score(reference_annotation, target_annotation)
        quality["speech_transcript_backed"] = 1.0 if _speech_is_transcript_backed(reference_annotation, target_annotation) else 0.0
    if difference_type == "audio_event":
        quality["non_speech_audio_event_score"] = _non_speech_audio_event_score(
            reference_annotation,
            target_annotation,
        )
    if _has_intraclip_difference_conflict(
        difference=difference,
        reference_caption=str(model_fields.get("reference_caption", "")),
        target_caption=str(model_fields.get("target_caption", "")),
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
    ):
        quality["intraclip_change_conflict"] = 1.0
    return quality


def _score_ordered_pair(
    *,
    root: Path,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if reference_annotation["clip_id"] == target_annotation["clip_id"]:
        return None

    semantic_context_score = _same_context_score(reference_annotation, target_annotation)
    source_context = _source_context(reference_annotation, target_annotation)
    if source_context["relation"] == "cross_dataset":
        return None
    same_context_score = _pair_context_score(
        semantic_context_score=semantic_context_score,
        source_context=source_context,
    )
    priority_order = _difference_priority_order(same_context_score=same_context_score)
    primary_difference = _detect_primary_difference(
        reference_annotation,
        target_annotation,
        priority_order=priority_order,
    )
    if primary_difference is None:
        return None
    changed_types = primary_difference.pop("changed_types")
    if same_context_score < MIN_PAIR_CONTEXT_SCORE:
        return None
    if len(changed_types) > MAX_PAIR_CHANGED_TYPES:
        return None

    edit_match_score = _edit_match_score(
        same_context_score=same_context_score,
        primary_difference_type=primary_difference["type"],
        changed_types=changed_types,
    )
    if edit_match_score < MIN_PAIR_EDIT_MATCH_SCORE:
        return None

    hard_negative_annotations = _select_hard_negative_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        primary_difference=primary_difference,
    )
    if len(hard_negative_annotations) < 2:
        return None

    target_uniqueness_score = _target_uniqueness_score(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        primary_difference=primary_difference,
    )
    visual_near_duplicate_score = _visual_near_duplicate_score(
        _resolve_under_root(root, reference_annotation["output_path"]),
        _resolve_under_root(root, target_annotation["output_path"]),
    )
    hard_negative_paths = [
        _display_path(root, _resolve_under_root(root, annotation["output_path"])) for annotation in hard_negative_annotations[:3]
    ]
    if len(hard_negative_paths) < 2:
        return None

    reference_path = _display_path(root, _resolve_under_root(root, reference_annotation["output_path"]))
    target_path = _display_path(root, _resolve_under_root(root, target_annotation["output_path"]))
    if reference_path in hard_negative_paths:
        return None
    if target_path in hard_negative_paths:
        return None

    quality = {
        "same_context_score": round(same_context_score, 3),
        "semantic_context_score": round(semantic_context_score, 3),
        "edit_match_score": round(edit_match_score, 3),
        "target_uniqueness_score": round(target_uniqueness_score, 3),
        "difference_strength_score": round(
            _difference_strength_score(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                primary_difference=primary_difference,
                changed_types=changed_types,
            ),
            3,
        ),
        "difference_type": primary_difference["type"],
    }
    if primary_difference["type"] == "action":
        quality["action_evidence_score"] = _action_evidence_score(reference_annotation, target_annotation)
    if primary_difference["type"] == "speech":
        quality["speech_evidence_score"] = _speech_evidence_score(reference_annotation, target_annotation)
        quality["speech_specificity_score"] = _speech_specificity_score(reference_annotation, target_annotation)
        quality["speech_transcript_backed"] = 1.0 if _speech_is_transcript_backed(reference_annotation, target_annotation) else 0.0
        quality["has_audio_modality"] = 1.0
    if primary_difference["type"] == "audio_event":
        quality["non_speech_audio_event_score"] = _non_speech_audio_event_score(
            reference_annotation,
            target_annotation,
        )
        quality["has_audio_modality"] = 1.0
    if visual_near_duplicate_score is not None:
        quality["visual_near_duplicate_score"] = round(visual_near_duplicate_score, 3)
    composite_score = _candidate_composite_score(quality, source_context)
    return {
        "proposal_id": _build_proposal_id(reference_path, target_path),
        "reference_annotation": _sanitize_annotation_for_output(reference_annotation, root),
        "target_annotation": _sanitize_annotation_for_output(target_annotation, root),
        "primary_difference": primary_difference,
        "changed_difference_types": list(changed_types),
        "quality": quality,
        "composite_score": composite_score,
        "source_context": source_context,
        "difference_evidence": _difference_evidence_from_annotations(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            primary_difference=primary_difference,
        ),
        "hard_negative_annotations": [_sanitize_annotation_for_output(annotation, root) for annotation in hard_negative_annotations[:3]],
        "hard_negative_paths": hard_negative_paths,
    }


def _candidate_composite_score(quality: dict[str, Any], source_context: dict[str, Any]) -> float:
    composite_score = round(
        _score_float(quality.get("same_context_score")) * 0.45
        + _score_float(quality.get("edit_match_score")) * 0.35
        + _score_float(quality.get("target_uniqueness_score")) * 0.15
        + _score_float(quality.get("difference_strength_score")) * 0.05,
        4,
    )
    return round(composite_score + _score_float(source_context.get("score")) * 0.08, 4)


def _visual_near_duplicate_score(left_path: Path, right_path: Path) -> float | None:
    if not left_path.exists() or not right_path.exists():
        return None
    left_frames = _sample_video_rgb_frames(left_path)
    right_frames = _sample_video_rgb_frames(right_path)
    if not left_frames or not right_frames:
        return None

    best_scores: list[float] = []
    for left_frame in left_frames:
        left_hash = _average_frame_hash(left_frame)
        frame_scores: list[float] = []
        for right_frame in right_frames:
            pixel_score = 1.0 - _frame_mae(left_frame, right_frame)
            hash_score = 1.0 - _hash_hamming(left_hash, _average_frame_hash(right_frame))
            frame_scores.append(max(0.0, min(1.0, min(pixel_score, hash_score))))
        best_scores.append(max(frame_scores))
    return sum(best_scores) / len(best_scores)


def _sample_video_rgb_frames(path: Path, *, size: int = 32, max_frames: int = 6) -> list[bytes]:
    if shutil.which("ffmpeg") is None:
        return []
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps=1,scale={size}:{size},format=rgb24",
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    frame_size = size * size * 3
    data = completed.stdout
    return [data[index : index + frame_size] for index in range(0, len(data), frame_size) if len(data[index : index + frame_size]) == frame_size]


def _frame_mae(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        return 1.0
    return sum(abs(a - b) for a, b in zip(left, right)) / (255.0 * len(left))


def _average_frame_hash(frame: bytes) -> tuple[bool, ...]:
    if not frame:
        return tuple()
    luminance = [
        (int(frame[index]) * 299 + int(frame[index + 1]) * 587 + int(frame[index + 2]) * 114) // 1000
        for index in range(0, len(frame) - 2, 3)
    ]
    if not luminance:
        return tuple()
    mean_value = sum(luminance) / len(luminance)
    return tuple(value >= mean_value for value in luminance)


def _hash_hamming(left: tuple[bool, ...], right: tuple[bool, ...]) -> float:
    if not left or len(left) != len(right):
        return 1.0
    return sum(1 for left_bit, right_bit in zip(left, right) if left_bit != right_bit) / len(left)


def _sanitize_annotation_for_output(annotation: dict[str, Any], root: Path) -> dict[str, Any]:
    sanitized = dict(annotation)
    sanitized["output_path"] = _display_path(root, _resolve_under_root(root, annotation["output_path"]))
    return sanitized


def _select_better_pair(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if left is None:
        return right
    if right is None:
        return left
    left_tuple = (
        left["composite_score"],
        left["quality"]["edit_match_score"],
        left["quality"]["same_context_score"],
        left["proposal_id"],
    )
    right_tuple = (
        right["composite_score"],
        right["quality"]["edit_match_score"],
        right["quality"]["same_context_score"],
        right["proposal_id"],
    )
    return left if left_tuple >= right_tuple else right


def _annotation_has_signal(annotation: dict[str, Any]) -> bool:
    return bool(
        str(annotation.get("summary", "")).strip()
        or annotation.get("subjects")
        or annotation.get("actions")
        or annotation.get("audio_events")
        or _timeline_audio_terms(annotation)
        or annotation.get("speech")
    )


def _source_context(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {str(value).strip() for value in left.get("source_row_ids", []) if str(value).strip()}
    right_rows = {str(value).strip() for value in right.get("source_row_ids", []) if str(value).strip()}
    shared_rows = sorted(left_rows & right_rows)
    if shared_rows:
        return {
            "relation": "shared_source_row",
            **_source_temporal_context(left, right, default_score=0.9),
            "shared_source_row_ids": shared_rows,
        }

    left_source_path = str(left.get("source_path", "")).strip()
    right_source_path = str(right.get("source_path", "")).strip()
    if left_source_path and left_source_path == right_source_path:
        return {
            "relation": "same_source_video",
            **_source_temporal_context(left, right, default_score=0.65),
        }

    left_dataset = str(left.get("dataset", "")).strip()
    right_dataset = str(right.get("dataset", "")).strip()
    if left_dataset and right_dataset:
        if left_dataset == right_dataset:
            text_score = _source_text_similarity(left, right)
            return {
                "relation": "same_dataset",
                "score": round(0.25 + text_score * 0.35, 3),
                "dataset": left_dataset,
                "text_similarity": round(text_score, 3),
            }
        return {"relation": "cross_dataset", "score": 0.0, "datasets": [left_dataset, right_dataset]}

    text_score = _source_text_similarity(left, right)
    return {"relation": "unknown", "score": round(text_score * 0.2, 3), "text_similarity": round(text_score, 3)}


def _pair_context_score(*, semantic_context_score: float, source_context: dict[str, Any]) -> float:
    source_score = _score_float(source_context.get("score"))
    relation = str(source_context.get("relation", "")).strip()
    if relation in {"shared_source_row", "same_source_video", "synthetic_from_reference"}:
        return max(semantic_context_score, source_score)
    return semantic_context_score


def _source_temporal_context(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    default_score: float,
) -> dict[str, Any]:
    left_bounds = _clip_time_bounds(left)
    right_bounds = _clip_time_bounds(right)
    if left_bounds is None or right_bounds is None:
        return {"score": round(default_score, 3), "temporal_relation": "unknown"}

    left_start, left_end = left_bounds
    right_start, right_end = right_bounds
    gap_seconds = max(0.0, max(left_start, right_start) - min(left_end, right_end))
    if gap_seconds <= 0.5:
        score = 0.9
        temporal_relation = "adjacent_or_overlapping"
    elif gap_seconds <= 8.0:
        score = 0.78
        temporal_relation = "nearby"
    elif gap_seconds <= 16.0:
        score = 0.65
        temporal_relation = "loose"
    else:
        score = 0.45
        temporal_relation = "distant"
    return {
        "score": round(score, 3),
        "temporal_relation": temporal_relation,
        "temporal_gap_seconds": round(gap_seconds, 3),
    }


def _clip_time_bounds(annotation: dict[str, Any]) -> tuple[float, float] | None:
    source_clip = annotation.get("source_clip")
    if not isinstance(source_clip, dict):
        return None
    try:
        start_seconds = float(source_clip["start_seconds"])
        end_seconds = float(source_clip["end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    if end_seconds <= start_seconds:
        return None
    return start_seconds, end_seconds


def _source_text_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    return _jaccard(_text_field_tokens(left.get("text_fields", {})), _text_field_tokens(right.get("text_fields", {})))


def _text_field_tokens(text_fields: Any) -> set[str]:
    if not isinstance(text_fields, dict):
        return set()
    tokens: set[str] = set()
    for value in text_fields.values():
        if isinstance(value, list):
            for item in value:
                tokens.update(_tokenize_text(str(item)))
        else:
            tokens.update(_tokenize_text(str(value)))
    return tokens


def _same_context_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    subject_score = _jaccard(_tokenize_values(left.get("subjects", [])), _tokenize_values(right.get("subjects", [])))
    scene_score = _scene_similarity(str(left.get("scene", "")), str(right.get("scene", "")))
    summary_score = _jaccard(_tokenize_text(str(left.get("summary", ""))), _tokenize_text(str(right.get("summary", ""))))
    text_score = _jaccard(
        _tokenize_values(left.get("on_screen_text", [])),
        _tokenize_values(right.get("on_screen_text", [])),
    )
    attribute_score = _jaccard(_tokenize_values(left.get("attributes", [])), _tokenize_values(right.get("attributes", [])))
    score = (
        subject_score * 0.35
        + scene_score * 0.30
        + summary_score * 0.20
        + text_score * 0.10
        + attribute_score * 0.05
    )
    return max(0.0, min(1.0, score))


def _difference_priority_order(*, same_context_score: float) -> tuple[str, ...]:
    if same_context_score >= 0.70:
        return HIGH_CONTEXT_PAIR_PRIORITY
    return PAIR_PRIORITY


def _scene_similarity(left: str, right: str) -> float:
    left_value = left.strip().lower()
    right_value = right.strip().lower()
    if not left_value or not right_value:
        return 0.0
    if left_value == right_value:
        return 1.0
    return _jaccard(_tokenize_text(left_value), _tokenize_text(right_value))


def _detect_primary_difference(
    reference: dict[str, Any],
    target: dict[str, Any],
    *,
    priority_order: tuple[str, ...] = PAIR_PRIORITY,
) -> dict[str, Any] | None:
    differences: dict[str, dict[str, Any]] = {}

    reference_counts = _normalize_object_counts(reference.get("object_counts", {}))
    target_counts = _normalize_object_counts(target.get("object_counts", {}))
    shared_count_labels = sorted(set(reference_counts) & set(target_counts))
    for label in shared_count_labels:
        if reference_counts[label] != target_counts[label]:
            differences["object_count"] = {
                "type": "object_count",
                "from": f"{reference_counts[label]} {label}",
                "to": f"{target_counts[label]} {label}",
                "description": f"the count of {label} changes from {reference_counts[label]} to {target_counts[label]}",
            }
            break

    reference_only = sorted(set(reference_counts) - set(target_counts))
    target_only = sorted(set(target_counts) - set(reference_counts))
    if "object_presence" not in differences:
        if target_only:
            label = target_only[0]
            differences["object_presence"] = {
                "type": "object_presence",
                "from": f"no {label}",
                "to": f"{target_counts[label]} {label}",
                "description": f"{label} appears in the target clip",
            }
        elif reference_only:
            label = reference_only[0]
            differences["object_presence"] = {
                "type": "object_presence",
                "from": f"{reference_counts[label]} {label}",
                "to": f"no {label}",
                "description": f"{label} disappears in the target clip",
            }

    reference_actions = _action_terms_from_annotation(reference)
    target_actions = _action_terms_from_annotation(target)
    added_action = _first_unique(target_actions, reference_actions)
    removed_action = _first_unique(reference_actions, target_actions)
    if added_action or removed_action:
        differences["action"] = {
            "type": "action",
            "from": removed_action or _first_item(reference_actions) or "current action",
            "to": added_action or _first_item(target_actions) or "new action",
            "description": "the main action changes between the clips and is supported by action/timeline evidence",
        }

    reference_audio = _non_speech_audio_terms(reference)
    target_audio = _non_speech_audio_terms(target)
    added_audio = _first_unique(target_audio, reference_audio)
    removed_audio = _first_unique(reference_audio, target_audio)
    if added_audio or removed_audio:
        differences["audio_event"] = {
            "type": "audio_event",
            "from": removed_audio or _first_item(reference_audio) or "no distinctive audio event",
            "to": added_audio or _first_item(target_audio) or "no distinctive audio event",
            "description": "the audible event changes between the clips",
        }

    reference_attributes = _normalize_list(reference.get("attributes", []))
    target_attributes = _normalize_list(target.get("attributes", []))
    added_attribute = _first_unique(target_attributes, reference_attributes)
    removed_attribute = _first_unique(reference_attributes, target_attributes)
    if added_attribute or removed_attribute:
        differences["attribute"] = {
            "type": "attribute",
            "from": removed_attribute or _first_item(reference_attributes) or "current attribute",
            "to": added_attribute or _first_item(target_attributes) or "new attribute",
            "description": "an attribute of the scene or subject changes",
        }

    reference_scene = str(reference.get("scene", "")).strip()
    target_scene = str(target.get("scene", "")).strip()
    if reference_scene and target_scene and reference_scene.lower() != target_scene.lower():
        differences["scene"] = {
            "type": "scene",
            "from": reference_scene,
            "to": target_scene,
            "description": "the scene changes between the clips",
        }

    reference_speech = _speech_texts_from_annotation(reference)
    target_speech = _speech_texts_from_annotation(target)
    added_speech = _first_unique(target_speech, reference_speech)
    removed_speech = _first_unique(reference_speech, target_speech)
    if (added_speech or removed_speech) and _speech_evidence_score(reference, target) >= MIN_ACCEPT_SPEECH_EVIDENCE_SCORE:
        differences["speech"] = {
            "type": "speech",
            "from": removed_speech or _first_item(reference_speech) or "no speech",
            "to": added_speech or _first_item(target_speech) or "new speech",
            "description": "the spoken content changes between the clips",
        }

    reference_text = _normalize_list(reference.get("visible_text") or reference.get("on_screen_text", []))
    target_text = _normalize_list(target.get("visible_text") or target.get("on_screen_text", []))
    added_text = _first_unique(target_text, reference_text)
    removed_text = _first_unique(reference_text, target_text)
    if added_text or removed_text:
        differences["visible_text"] = {
            "type": "visible_text",
            "from": removed_text or _first_item(reference_text) or "no visible text",
            "to": added_text or _first_item(target_text) or "new visible text",
            "description": "the visible on-screen text changes between the clips",
        }

    changed_types = [difference_type for difference_type in priority_order if difference_type in differences]
    if not changed_types:
        return None
    primary = dict(differences[changed_types[0]])
    primary["changed_types"] = changed_types
    return primary


def _edit_match_score(
    *,
    same_context_score: float,
    primary_difference_type: str,
    changed_types: list[str],
) -> float:
    if primary_difference_type not in PAIR_PRIORITY:
        return 0.0
    base_score = 0.5 + same_context_score * 0.35
    if primary_difference_type in {"object_count", "object_presence", "action", "audio_event", "speech", "visible_text"}:
        base_score += 0.1
    if primary_difference_type in {"audio_event", "speech", "visible_text"} and same_context_score >= 0.70:
        base_score += 0.08
    penalty = max(0, len(changed_types) - 1) * 0.10
    return max(0.0, min(1.0, base_score - penalty))


def _difference_strength_score(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference: dict[str, Any],
    changed_types: list[str],
) -> float:
    difference_type = str(primary_difference.get("type", "")).strip()
    from_value = str(primary_difference.get("from", "")).strip().lower()
    to_value = str(primary_difference.get("to", "")).strip().lower()
    if not difference_type or not from_value or not to_value or from_value == to_value:
        return 0.0

    if difference_type == "object_count":
        score = _object_count_delta_score(from_value, to_value)
    elif difference_type == "object_presence":
        score = 0.82 if from_value.startswith("no ") or to_value.startswith("no ") else 0.70
    elif difference_type == "action":
        score = _action_evidence_score(reference_annotation, target_annotation)
    elif difference_type == "audio_event":
        score = _non_speech_audio_event_score(reference_annotation, target_annotation)
    elif difference_type == "speech":
        score = _speech_evidence_score(reference_annotation, target_annotation)
    elif difference_type == "visible_text":
        score = _list_delta_strength(
            reference_annotation.get("visible_text") or reference_annotation.get("on_screen_text", []),
            target_annotation.get("visible_text") or target_annotation.get("on_screen_text", []),
        )
    elif difference_type == "attribute":
        score = _list_delta_strength(reference_annotation.get("attributes", []), target_annotation.get("attributes", []))
    elif difference_type == "scene":
        score = 0.65 + _scene_similarity(
            str(reference_annotation.get("scene", "")),
            str(target_annotation.get("scene", "")),
        ) * 0.10
    else:
        score = 0.0

    evidence = _difference_evidence_from_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference=primary_difference,
    )
    if evidence["supporting_evidence"] and difference_type != "action":
        score = max(score, 0.70)
    if len(changed_types) == 1:
        score += 0.08
    else:
        score -= min(0.15, (len(changed_types) - 1) * 0.04)
    return round(max(0.0, min(1.0, score)), 3)


def _object_count_delta_score(from_value: str, to_value: str) -> float:
    from_count = _first_integer(from_value)
    to_count = _first_integer(to_value)
    if from_count is None or to_count is None or from_count == to_count:
        return 0.65
    delta = abs(to_count - from_count)
    return min(1.0, 0.74 + min(delta, 4) * 0.05)


def _first_integer(value: str) -> int | None:
    match = re.search(r"\d+", value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _list_delta_strength(left: Any, right: Any) -> float:
    left_values = _normalize_list(left)
    right_values = _normalize_list(right)
    if left_values == right_values:
        return 0.0
    token_overlap = _jaccard(_tokenize_values(left_values), _tokenize_values(right_values))
    return max(0.62, min(0.92, 0.92 - token_overlap * 0.25))


def _action_terms_from_annotation(annotation: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add_values(value: Any) -> None:
        for item in _normalize_list(value):
            if item not in terms:
                terms.append(item)

    add_values(annotation.get("actions", []))
    for container_name in ("events", "storyline"):
        container = annotation.get(container_name, [])
        if not isinstance(container, list):
            continue
        for item in container:
            if isinstance(item, dict):
                add_values(item.get("actions", []))
                action_value = item.get("action")
                if action_value:
                    add_values([action_value])
    return terms


def _has_timeline_action_evidence(annotation: dict[str, Any]) -> bool:
    return bool(_timeline_evidence(annotation))


def _action_evidence_score(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> float:
    reference_terms = _action_terms_from_annotation(reference_annotation)
    target_terms = _action_terms_from_annotation(target_annotation)
    if not reference_terms or not target_terms:
        return 0.0
    if not _first_unique(reference_terms, target_terms) and not _first_unique(target_terms, reference_terms):
        return 0.0

    score = _list_delta_strength(reference_terms, target_terms)
    reference_has_timeline = _has_timeline_action_evidence(reference_annotation)
    target_has_timeline = _has_timeline_action_evidence(target_annotation)
    if reference_has_timeline and target_has_timeline:
        return round(max(score, 0.74), 3)
    if reference_annotation.get("actions") and target_annotation.get("actions"):
        return round(min(score, 0.62), 3)
    return round(min(score, 0.55), 3)


def _speech_texts_from_annotation(annotation: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    def add_text(value: Any) -> None:
        for item in _normalize_list(value):
            if item not in texts:
                texts.append(item)

    add_text(annotation.get("speech", []))
    transcript = annotation.get("speakers_and_transcript", [])
    if isinstance(transcript, list):
        for item in transcript:
            if isinstance(item, dict):
                add_text(
                    [
                        item.get("content")
                        or item.get("transcript")
                        or item.get("text")
                        or item.get("utterance")
                        or ""
                    ]
                )
            else:
                add_text([item])
    return texts


def _speech_specificity_score(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> float:
    reference_score = _speech_specificity_score_for_texts(_speech_texts_from_annotation(reference_annotation))
    target_score = _speech_specificity_score_for_texts(_speech_texts_from_annotation(target_annotation))
    if reference_score == 0.0 or target_score == 0.0:
        return 0.0
    return round(min(reference_score, target_score), 3)


def _speech_specificity_score_for_texts(texts: list[str]) -> float:
    if not texts:
        return 0.0
    best_score = 0.0
    for text in texts:
        normalized = text.strip().lower()
        if not normalized:
            continue
        if normalized in GENERIC_SPEECH_PHRASES:
            best_score = max(best_score, 0.2)
            continue
        tokens = _tokenize_text(normalized)
        content_tokens = {
            token
            for token in tokens
            if token not in GENERIC_SPEECH_TOKENS and token not in VISUAL_DESCRIPTION_TOKENS
        }
        generic_overlap = len(tokens & GENERIC_SPEECH_TOKENS)
        score = 0.0
        if len(content_tokens) >= 6:
            score = 0.9
        elif len(content_tokens) >= 4:
            score = 0.78
        elif len(content_tokens) >= 3 and len(normalized) >= 35:
            score = 0.72
        elif len(content_tokens) >= 2 and generic_overlap:
            score = 0.55
        else:
            score = 0.3
        if any(phrase in normalized for phrase in GENERIC_SPEECH_PHRASES) and len(content_tokens) < 4:
            score = min(score, 0.55)
        best_score = max(best_score, score)
    return round(best_score, 3)


def _speech_evidence_score(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> float:
    reference_texts = _speech_texts_from_annotation(reference_annotation)
    target_texts = _speech_texts_from_annotation(target_annotation)
    if not reference_texts or not target_texts:
        return 0.0
    if not _first_unique(reference_texts, target_texts) and not _first_unique(target_texts, reference_texts):
        return 0.0

    specificity = _speech_specificity_score(reference_annotation, target_annotation)
    if specificity < MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE:
        return round(min(specificity, 0.69), 3)

    has_reference_transcript = _has_transcript_evidence(reference_annotation)
    has_target_transcript = _has_transcript_evidence(target_annotation)
    if not (has_reference_transcript and has_target_transcript):
        return round(min(specificity, 0.69), 3)

    score = _list_delta_strength(reference_texts, target_texts)
    score = max(score, 0.88)
    return round(min(score, specificity), 3)


def _has_transcript_evidence(annotation: dict[str, Any]) -> bool:
    transcript = annotation.get("speakers_and_transcript", [])
    if not isinstance(transcript, list):
        return False
    for item in transcript:
        if isinstance(item, dict):
            if str(item.get("content") or item.get("transcript") or item.get("text") or "").strip():
                return True
        elif str(item).strip():
            return True
    return False


def _speech_is_transcript_backed(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> bool:
    return _has_transcript_evidence(reference_annotation) and _has_transcript_evidence(target_annotation)


def _non_speech_audio_terms(annotation: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for item in (
        _normalize_list(annotation.get("audio_events", []))
        + _timeline_audio_terms(annotation)
        + _annotation_audio_text_terms(annotation)
    ):
        if not _is_speech_like_audio_event(item) and item not in terms:
            terms.append(item)
    return terms


def _annotation_audio_text_terms(annotation: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add(value: Any) -> None:
        for item in _normalize_list(value):
            if _is_non_speech_audio_phrase(item) and item not in terms:
                terms.append(item)

    add(annotation.get("summary", ""))
    add(annotation.get("detective_notes", []))
    add(annotation.get("audio_observations", []))
    return terms


def _timeline_audio_terms(annotation: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add_if_relevant(value: Any) -> None:
        for item in _normalize_list(value):
            if _is_non_speech_audio_phrase(item) and item not in terms:
                terms.append(item)

    container = annotation.get("events", [])
    if not isinstance(container, list):
        return terms
    for item in container:
        if isinstance(item, dict):
            add_if_relevant([item.get("audio", "")])
            add_if_relevant(item.get("audio_events", []))
    return terms


def _is_non_speech_audio_phrase(value: str) -> bool:
    tokens = _tokenize_text(value)
    if _is_speech_only_or_absence_audio_phrase(value):
        return False
    return bool(tokens & NON_SPEECH_AUDIO_TOKENS) and not _is_speech_like_audio_event(value)


def _is_speech_like_audio_event(value: str) -> bool:
    tokens = _tokenize_text(value)
    if not tokens:
        return False
    if tokens & NON_SPEECH_AUDIO_TOKENS:
        return False
    return bool(tokens & GENERIC_SPEECH_TOKENS)


def _is_speech_only_or_absence_audio_phrase(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if not normalized:
        return False
    return any(pattern in normalized for pattern in SPEECH_ONLY_AUDIO_PATTERNS + NON_SPEECH_AUDIO_ABSENCE_PATTERNS)


def _is_non_speech_absence_audio_phrase(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if not normalized:
        return False
    return any(pattern in normalized for pattern in NON_SPEECH_AUDIO_ABSENCE_PATTERNS)


def _is_speech_only_audio_phrase(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if not normalized:
        return False
    return any(pattern in normalized for pattern in SPEECH_ONLY_AUDIO_PATTERNS)


def _speech_content_edit_issues(*, edit_text: str, difference: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    difference_type = str(difference.get("type", "")).strip()
    if difference_type == "speech":
        issues.append("speech difference type is disabled for final Omni-CVR samples")

    text_parts = [
        edit_text,
        str(difference.get("from", "")),
        str(difference.get("to", "")),
        str(difference.get("description", "")),
    ]
    normalized = _normalized_phrase(" ".join(text_parts))
    if normalized and any(pattern in normalized for pattern in SPEECH_CONTENT_EDIT_PATTERNS):
        issues.append("speech content edits are disabled for final Omni-CVR samples")

    if difference_type == "audio_event":
        from_value = str(difference.get("from", "")).strip()
        to_value = str(difference.get("to", "")).strip()
        if _is_speech_only_audio_phrase(from_value) or _is_speech_only_audio_phrase(to_value):
            issues.append("audio_event must not use speech-only or narration-only text as the main difference")

    deduped: list[str] = []
    for issue in issues:
        if issue not in deduped:
            deduped.append(issue)
    return deduped


def _normalize_audio_event_model_fields(model_fields: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(model_fields)
    difference = dict(normalized.get("difference", {}))
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if _is_non_speech_absence_audio_phrase(from_value) and to_value:
        normalized["edit_text"] = f"add {to_value} to the audio"
    elif _is_non_speech_absence_audio_phrase(to_value) and from_value:
        normalized["edit_text"] = f"remove {from_value} from the audio"
    normalized["difference"] = difference
    return normalized


def _non_speech_audio_event_score(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> float:
    reference_terms = _non_speech_audio_terms(reference_annotation)
    target_terms = _non_speech_audio_terms(target_annotation)
    if not reference_terms and not target_terms:
        return 0.0
    if not _first_unique(reference_terms, target_terms) and not _first_unique(target_terms, reference_terms):
        return 0.0
    return round(max(_list_delta_strength(reference_terms, target_terms), 0.70), 3)


def _difference_evidence_from_annotations(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference: dict[str, Any],
) -> dict[str, Any]:
    difference_type = str(primary_difference.get("type", "")).strip()
    evidence: list[str] = []
    if difference_type in {"object_count", "object_presence"}:
        evidence.append(
            "object_counts: "
            f"{reference_annotation.get('object_counts', {})} -> {target_annotation.get('object_counts', {})}"
        )
    if difference_type == "action":
        evidence.append(_change_text(_action_terms_from_annotation(reference_annotation), _action_terms_from_annotation(target_annotation)))
        evidence.append(
            "action_evidence_score: "
            f"{_action_evidence_score(reference_annotation, target_annotation):.3f}"
        )
    if difference_type == "audio_event":
        evidence.append(_change_text(_non_speech_audio_terms(reference_annotation), _non_speech_audio_terms(target_annotation)))
        evidence.append(
            "non_speech_audio_event_score: "
            f"{_non_speech_audio_event_score(reference_annotation, target_annotation):.3f}"
        )
    if difference_type == "speech":
        evidence.append(_change_text(_speech_texts_from_annotation(reference_annotation), _speech_texts_from_annotation(target_annotation)))
        evidence.append(
            "speech_evidence_score: "
            f"{_speech_evidence_score(reference_annotation, target_annotation):.3f}"
        )
        evidence.append(
            "speech_specificity_score: "
            f"{_speech_specificity_score(reference_annotation, target_annotation):.3f}"
        )
    if difference_type == "visible_text":
        evidence.append(
            _change_text(
                reference_annotation.get("visible_text") or reference_annotation.get("on_screen_text", []),
                target_annotation.get("visible_text") or target_annotation.get("on_screen_text", []),
            )
        )
    if difference_type == "attribute":
        evidence.append(_change_text(reference_annotation.get("attributes", []), target_annotation.get("attributes", [])))
    if difference_type == "scene":
        evidence.append(f"scene: {reference_annotation.get('scene', '')} -> {target_annotation.get('scene', '')}")

    reference_events = _timeline_evidence(reference_annotation)
    target_events = _timeline_evidence(target_annotation)
    if reference_events or target_events:
        evidence.append(f"events: {' | '.join(reference_events[:2]) or 'none'} -> {' | '.join(target_events[:2]) or 'none'}")

    return {
        "difference_type": difference_type,
        "from": str(primary_difference.get("from", "")).strip(),
        "to": str(primary_difference.get("to", "")).strip(),
        "supporting_evidence": [item for item in evidence if item.strip() and not item.strip().endswith("-> none")],
        "reference_events": reference_events,
        "target_events": target_events,
    }


def _normalize_events_for_evidence(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    events: list[str] = []
    for item in value:
        if isinstance(item, dict):
            pieces = [
                str(item.get("visual", "")).strip(),
                str(item.get("audio", "")).strip(),
                "; ".join(_normalize_list(item.get("objects", []))),
                "; ".join(_normalize_list(item.get("actions", []))),
            ]
            text = " / ".join(piece for piece in pieces if piece)
        else:
            text = str(item).strip()
        if text:
            events.append(text)
    return events


def _timeline_evidence(annotation: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    for field_name in ("events", "storyline"):
        for item in _normalize_events_for_evidence(annotation.get(field_name, [])):
            if item not in evidence:
                evidence.append(item)
    return evidence


def _select_hard_negative_annotations(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
    primary_difference: dict[str, Any],
) -> list[dict[str, Any]]:
    scored_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for other in annotations:
        if other["clip_id"] in {reference_annotation["clip_id"], target_annotation["clip_id"]}:
            continue

        context_score = max(
            _same_context_score(reference_annotation, other),
            _same_context_score(target_annotation, other),
        )
        score = context_score
        other_difference = _detect_primary_difference(reference_annotation, other)
        if other_difference is not None and other_difference["type"] == primary_difference["type"]:
            score -= 0.2
        if other["output_path"] == target_annotation["output_path"]:
            continue
        scored_candidates.append((score, other["clip_id"], other))

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored_candidates[:3]]


def _target_uniqueness_score(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
    primary_difference: dict[str, Any],
) -> float:
    competitor_scores = []
    priority_order = (primary_difference["type"],) + tuple(
        item for item in PAIR_PRIORITY if item != primary_difference["type"]
    )
    for other in annotations:
        if other["clip_id"] in {reference_annotation["clip_id"], target_annotation["clip_id"]}:
            continue
        context_score = _same_context_score(target_annotation, other)
        other_difference = _detect_primary_difference(
            reference_annotation,
            other,
            priority_order=priority_order,
        )
        competitor_scores.append(
            _target_competitor_score(
                context_score=context_score,
                primary_difference=primary_difference,
                competitor_difference=other_difference,
            )
        )
    if not competitor_scores:
        return 1.0
    highest_competitor = max(competitor_scores)
    return max(0.0, min(1.0, 1.0 - highest_competitor * 0.75))


def _target_competitor_score(
    *,
    context_score: float,
    primary_difference: dict[str, Any],
    competitor_difference: dict[str, Any] | None,
) -> float:
    if competitor_difference is None:
        return context_score * 0.35
    if competitor_difference["type"] != primary_difference["type"]:
        return context_score * 0.35
    if _difference_targets_overlap(primary_difference, competitor_difference):
        return context_score
    return context_score * 0.35


def _difference_targets_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_target = str(left.get("to", "")).strip().lower()
    right_target = str(right.get("to", "")).strip().lower()
    if left_target and right_target and left_target == right_target:
        return True
    if left_target and right_target:
        left_tokens = _tokenize_text(left_target)
        right_tokens = _tokenize_text(right_target)
        return _jaccard(left_tokens, right_tokens) >= 0.67
    left_tokens = _tokenize_text(str(left.get("description", "")))
    right_tokens = _tokenize_text(str(right.get("description", "")))
    return _jaccard(left_tokens, right_tokens) >= 0.50


def _annotation_prompt_view(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": annotation["clip_id"],
        "output_path": annotation["output_path"],
        "summary": _truncate_text(annotation.get("summary", ""), 700),
        "subjects": _prompt_list(annotation.get("subjects", []), limit=8, text_limit=80),
        "object_counts": dict(annotation.get("object_counts", {})),
        "actions": _prompt_list(annotation.get("actions", []), limit=8, text_limit=80),
        "scene": _truncate_text(annotation.get("scene", ""), 300),
        "attributes": _prompt_list(annotation.get("attributes", []), limit=8, text_limit=120),
        "on_screen_text": _prompt_list(annotation.get("on_screen_text", []), limit=8, text_limit=120),
        "speech": _prompt_list(annotation.get("speech", []), limit=6, text_limit=180),
        "audio_events": _prompt_list(annotation.get("audio_events", []), limit=8, text_limit=120),
        "modalities": _prompt_list(annotation.get("modalities", []), limit=4, text_limit=40),
        "storyline": _prompt_list(annotation.get("storyline", []), limit=6, text_limit=220),
        "events": _prompt_list(annotation.get("events", []), limit=8, text_limit=220),
        "visible_text": _prompt_list(annotation.get("visible_text", []), limit=8, text_limit=120),
        "speakers_and_transcript": _prompt_list(annotation.get("speakers_and_transcript", []), limit=6, text_limit=220),
        "uncertainties": _prompt_list(annotation.get("uncertainties", []), limit=6, text_limit=160),
    }


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _prompt_list(value: Any, *, limit: int, text_limit: int) -> list[Any]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    compact: list[Any] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            compact.append(
                {
                    str(key): _truncate_text(raw_value, text_limit)
                    for key, raw_value in item.items()
                    if key in {"time", "timestamp", "description", "visual", "audio", "text", "action", "event", "objects"}
                }
            )
        else:
            text = _truncate_text(item, text_limit)
            if text:
                compact.append(text)
    return compact


def _fallback_clip_annotation() -> dict[str, Any]:
    return {
        "summary": "",
        "subjects": [],
        "object_counts": {},
        "actions": [],
        "scene": "",
        "attributes": [],
        "on_screen_text": [],
        "speech": [],
        "audio_events": [],
        "modalities": ["visual"],
    }


def _fallback_pair_model_fields(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "edit_text": _build_fallback_edit_text(primary_difference),
        "modalities": _infer_pair_modalities(reference_annotation, target_annotation, primary_difference["type"]),
        "reference_caption": str(reference_annotation.get("summary", "")).strip(),
        "target_caption": str(target_annotation.get("summary", "")).strip(),
        "difference": dict(primary_difference),
        "proposal_reason": f"heuristic fallback based on {primary_difference['type']}",
    }


def _build_fallback_edit_text(primary_difference: dict[str, Any]) -> str:
    difference_type = primary_difference["type"]
    from_value = str(primary_difference.get("from", "")).strip()
    to_value = str(primary_difference.get("to", "")).strip()
    if difference_type == "object_count":
        from_count, from_label = _count_and_label(from_value)
        to_count, to_label = _count_and_label(to_value)
        label = to_label or from_label or "object"
        if from_count is not None and to_count is not None:
            return f"change the number of {label} from {from_count} to {to_count}"
        return f"change the number of {label} from {from_value} to {to_value}"
    if difference_type == "object_presence":
        if from_value.lower().startswith("no ") and to_value:
            return f"add {_object_phrase_for_edit(to_value)}"
        if to_value.lower().startswith("no ") and from_value:
            return f"remove {_object_phrase_for_edit(from_value)}"
        return f"replace {_object_phrase_for_edit(from_value)} with {_object_phrase_for_edit(to_value)}"
    if difference_type == "action":
        return f"change the action from {from_value} to {to_value}"
    if difference_type == "audio_event":
        if _is_non_speech_absence_audio_phrase(from_value) and to_value:
            return f"add {to_value} to the audio"
        if _is_non_speech_absence_audio_phrase(to_value) and from_value:
            return f"remove {from_value} from the audio"
        return f"replace {from_value} with {to_value} in the audio"
    if difference_type == "attribute":
        return f"change the attribute from {from_value} to {to_value}"
    if difference_type == "scene":
        return f"change the scene from {from_value} to {to_value}"
    if difference_type == "speech":
        return f"change the speech from {_short_edit_phrase(from_value)} to {_short_edit_phrase(to_value)}"
    if difference_type == "visible_text":
        return f"change on-screen text from {_short_edit_phrase(from_value)} to {_short_edit_phrase(to_value)}"
    return str(primary_difference.get("description", "")).strip() or f"change {from_value} to {to_value}"


def _count_and_label(value: str) -> tuple[int | None, str]:
    match = re.match(r"\s*(\d+)\s+(.+?)\s*$", value)
    if not match:
        return None, _strip_presence_prefix(value)
    return int(match.group(1)), _strip_presence_prefix(match.group(2))


def _strip_presence_prefix(value: str) -> str:
    normalized = str(value).strip()
    normalized = re.sub(r"^\s*no\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^\s*\d+\s+", "", normalized)
    return normalized.strip()


def _object_phrase_for_edit(value: str) -> str:
    label = _strip_presence_prefix(value)
    if not label:
        return "the object"
    first_token = label.split()[0].lower()
    if first_token in {"a", "an", "the"}:
        return label
    article = "an" if first_token[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}"


def _short_edit_phrase(value: str, *, max_words: int = 12) -> str:
    words = str(value).strip().split()
    if len(words) <= max_words:
        return str(value).strip()
    return " ".join(words[:max_words])


def _infer_pair_modalities(
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference_type: str,
) -> list[str]:
    modalities: list[str] = []
    for item in list(reference_annotation.get("modalities", [])) + list(target_annotation.get("modalities", [])):
        value = str(item).strip().lower()
        if value in ALLOWED_MODALITIES and value not in modalities:
            modalities.append(value)
    if primary_difference_type in {"audio_event", "speech"} and "audio" not in modalities:
        modalities.append("audio")
    if "visual" not in modalities:
        modalities.insert(0, "visual")
    return modalities


def _fallback_pair_judge(quality: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "reference_satisfies_edit": False,
        "target_satisfies_edit": False,
        "single_main_difference": False,
        "same_context_score": _score_float(quality.get("same_context_score")),
        "edit_match_score": _score_float(quality.get("edit_match_score")),
        "target_uniqueness_score": _score_float(quality.get("target_uniqueness_score")),
        "audio_required": False,
        "hard_negative_quality": "weak",
        "accept": False,
        "reject_reason": f"pair judge fallback: {reason}",
    }


def _is_verification_context_limit_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in message
        for marker in (
            "context length",
            "context window",
            "input length",
            "max_model_len",
            "maximum context",
            "too many tokens",
            "token limit",
        )
    )


def _verify_pair_difference_with_context_retry(
    client: OpenAIComposedDataClient,
    *,
    proposal: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    reference_clip_path: str,
    target_clip_path: str,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    try:
        verification, raw_output = client.verify_pair_difference(
            proposal=proposal,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            reference_clip_path=reference_clip_path,
            target_clip_path=target_clip_path,
        )
        return verification, raw_output, False
    except Exception as exc:
        if not _is_verification_context_limit_error(exc):
            raise
        first_error = f"{type(exc).__name__}: {exc}"
        try:
            verification, retry_raw_output = client.verify_pair_difference(
                proposal=proposal,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                reference_clip_path=None,
                target_clip_path=None,
            )
        except Exception as retry_exc:
            raise RuntimeError(
                "annotation-only verification retry failed after video verification "
                f"context error: {first_error}; retry error: {type(retry_exc).__name__}: {retry_exc}"
            ) from retry_exc
        return (
            verification,
            {
                "video_verification_error": first_error,
                "annotation_only_retry_used": True,
                "annotation_only_retry": retry_raw_output,
            },
            True,
        )


def _fallback_pair_verification(*, reason: str) -> dict[str, Any]:
    return {
        "caption_delta": {
            "caption_equivalent": True,
            "has_concrete_difference": False,
            "difference_matches_edit": False,
            "concrete_differences": [],
            "reason": f"pair verification fallback: {reason}",
        },
        "edit_projection": {
            "projected_target_caption": "",
            "target_matches_projection": False,
            "score": 0.0,
            "missing_requirements": ["verification unavailable"],
            "reason": f"pair verification fallback: {reason}",
        },
        "edit_necessity": {
            "edit_needed": False,
            "reference_satisfies_edit": False,
            "target_satisfies_edit": False,
            "score": 0.0,
            "reason": f"pair verification fallback: {reason}",
        },
    }


def _finalize_pair_verification(verification: dict[str, Any]) -> dict[str, Any]:
    caption_delta = dict(verification.get("caption_delta", {}))
    edit_projection = dict(verification.get("edit_projection", {}))
    edit_necessity = dict(verification.get("edit_necessity", {}))
    edit_text_quality_check = dict(verification.get("edit_text_quality_check", {}))
    normalized = {
        "caption_delta": {
            "caption_equivalent": _boolish(caption_delta.get("caption_equivalent")),
            "has_concrete_difference": _boolish(caption_delta.get("has_concrete_difference")),
            "difference_matches_edit": _boolish(caption_delta.get("difference_matches_edit")),
            "concrete_differences": _normalize_list(caption_delta.get("concrete_differences", [])),
            "reason": str(caption_delta.get("reason", "")).strip(),
        },
        "edit_projection": {
            "projected_target_caption": str(edit_projection.get("projected_target_caption", "")).strip(),
            "target_matches_projection": _boolish(edit_projection.get("target_matches_projection")),
            "score": _score_float(edit_projection.get("score")),
            "missing_requirements": _normalize_list(edit_projection.get("missing_requirements", [])),
            "reason": str(edit_projection.get("reason", "")).strip(),
        },
        "edit_necessity": {
            "edit_needed": _boolish(edit_necessity.get("edit_needed")),
            "reference_satisfies_edit": _boolish(edit_necessity.get("reference_satisfies_edit")),
            "target_satisfies_edit": _boolish(edit_necessity.get("target_satisfies_edit")),
            "score": _score_float(edit_necessity.get("score")),
            "reason": str(edit_necessity.get("reason", "")).strip(),
        },
        "edit_text_quality_check": {
            "not_caption_like": _boolish(edit_text_quality_check.get("not_caption_like", True)),
            "matches_modality": _boolish(edit_text_quality_check.get("matches_modality", True)),
            "single_primary_difference": _boolish(edit_text_quality_check.get("single_primary_difference", True)),
            "reference_does_not_satisfy": _boolish(edit_text_quality_check.get("reference_does_not_satisfy", True)),
            "target_satisfies": _boolish(edit_text_quality_check.get("target_satisfies", True)),
            "score": _score_float(edit_text_quality_check.get("score", 1.0)),
            "failure_reason": str(edit_text_quality_check.get("failure_reason", "")).strip(),
        },
    }
    _apply_verification_semantic_rejections(normalized)
    normalized["passed"] = _verification_accepts(normalized)
    normalized["failures"] = _verification_failures(normalized)
    return normalized


def _apply_verification_semantic_rejections(verification: dict[str, Any]) -> None:
    if not _verification_describes_order_only_difference(verification):
        return
    reason = "same content appears in a different shot/order sequence, not an edit-required target difference"
    caption_delta = verification.setdefault("caption_delta", {})
    caption_delta["caption_equivalent"] = True
    caption_delta["has_concrete_difference"] = False
    caption_delta["difference_matches_edit"] = False
    caption_delta["reason"] = _append_reason(caption_delta.get("reason"), reason)

    edit_projection = verification.setdefault("edit_projection", {})
    edit_projection["target_matches_projection"] = False
    edit_projection["score"] = min(_score_float(edit_projection.get("score")), 0.0)
    edit_projection["reason"] = _append_reason(edit_projection.get("reason"), reason)

    edit_necessity = verification.setdefault("edit_necessity", {})
    edit_necessity["edit_needed"] = False
    edit_necessity["reference_satisfies_edit"] = True
    edit_necessity["target_satisfies_edit"] = True
    edit_necessity["score"] = min(_score_float(edit_necessity.get("score")), 0.0)
    edit_necessity["reason"] = _append_reason(edit_necessity.get("reason"), reason)


def _verification_describes_order_only_difference(verification: dict[str, Any]) -> bool:
    text_parts: list[str] = []
    for section_name in ("caption_delta", "edit_projection", "edit_necessity"):
        section = verification.get(section_name, {})
        if not isinstance(section, dict):
            continue
        text_parts.append(str(section.get("reason", "")))
        text_parts.append(str(section.get("projected_target_caption", "")))
        text_parts.extend(_normalize_list(section.get("concrete_differences", [])))
        text_parts.extend(_normalize_list(section.get("missing_requirements", [])))
    text = _normalized_phrase(" ".join(text_parts))
    if not text:
        return False
    order_markers = (
        "different order",
        "different sequence",
        "order differs",
        "sequence differs",
        "reordered",
        "reverse order",
        "reversed order",
        "shot order",
        "sequence order",
        "temporal order",
        "just the order",
        "only the order",
        "只是顺序",
        "顺序不同",
        "镜头顺序",
    )
    has_order_marker = any(marker in text for marker in order_markers)
    if not has_order_marker:
        return False
    same_content_markers = (
        "same shots",
        "same elements",
        "same scenes",
        "same content",
        "both videos",
        "both clips",
        "both contain",
        "both show",
        "only",
        "just",
        "merely",
        "相同",
        "只是",
    )
    return any(marker in text for marker in same_content_markers)


def _append_reason(existing: Any, reason: str) -> str:
    existing_text = str(existing or "").strip()
    if not existing_text:
        return reason
    if reason in existing_text:
        return existing_text
    return f"{existing_text} {reason}"


def _finalize_pair_judge(judge: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(judge)
    accepted = _judge_accepts(normalized)
    if accepted:
        normalized["reject_reason"] = ""
        return normalized
    normalized["reject_reason"] = _compose_reject_reason(normalized)
    return normalized


def _effective_pair_quality(
    judge: dict[str, Any],
    verification: dict[str, Any] | None,
    heuristic_quality: dict[str, Any] | None,
) -> dict[str, float]:
    heuristic_quality = heuristic_quality or {}
    verification_edit_score = 0.0
    verification_accepted = verification is not None and _verification_accepts(verification)
    if verification_accepted:
        edit_projection = verification.get("edit_projection", {})
        edit_necessity = verification.get("edit_necessity", {})
        verification_edit_score = min(
            _score_float(edit_projection.get("score")),
            _score_float(edit_necessity.get("score")),
        )
    if "difference_strength_score" in heuristic_quality:
        difference_strength_score = _score_float(heuristic_quality.get("difference_strength_score"))
    else:
        difference_strength_score = MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE if verification_accepted else 0.0
    result: dict[str, Any] = {
        "same_context_score": max(
            _score_float(judge.get("same_context_score")),
            _score_float(heuristic_quality.get("same_context_score")),
        ),
        "edit_match_score": max(
            _score_float(judge.get("edit_match_score")),
            verification_edit_score,
        ),
        "target_uniqueness_score": max(
            _score_float(judge.get("target_uniqueness_score")),
            _score_float(heuristic_quality.get("target_uniqueness_score")),
        ),
        "difference_strength_score": difference_strength_score,
    }
    if "visual_near_duplicate_score" in heuristic_quality:
        result["visual_near_duplicate_score"] = _score_float(heuristic_quality.get("visual_near_duplicate_score"))
    if "difference_type" in heuristic_quality:
        result["difference_type"] = str(heuristic_quality.get("difference_type", "")).strip()
    if "action_evidence_score" in heuristic_quality:
        result["action_evidence_score"] = _score_float(heuristic_quality.get("action_evidence_score"))
    if "speech_evidence_score" in heuristic_quality:
        result["speech_evidence_score"] = _score_float(heuristic_quality.get("speech_evidence_score"))
    if "speech_specificity_score" in heuristic_quality:
        result["speech_specificity_score"] = _score_float(heuristic_quality.get("speech_specificity_score"))
    if "speech_transcript_backed" in heuristic_quality:
        result["speech_transcript_backed"] = _score_float(heuristic_quality.get("speech_transcript_backed"))
    if "non_speech_audio_event_score" in heuristic_quality:
        result["non_speech_audio_event_score"] = _score_float(
            heuristic_quality.get("non_speech_audio_event_score")
        )
    if "has_audio_modality" in heuristic_quality:
        result["has_audio_modality"] = _score_float(heuristic_quality.get("has_audio_modality"))
    for key in (
        "edit_text_quality_score",
        "edit_text_is_imperative",
        "edit_text_matches_difference_type",
        "edit_text_single_change",
        "edit_text_not_caption_like",
        "edit_text_no_modality_leakage",
        "observable_difference_passed",
        "observable_difference_frame_backed",
        "near_duplicate_without_delta",
        "synthetic_context_override",
    ):
        if key in heuristic_quality:
            result[key] = _score_float(heuristic_quality.get(key))
    return result


def _speech_quality_payload(quality: dict[str, Any]) -> dict[str, Any]:
    if str(quality.get("difference_type", "")).strip() != "speech":
        return {}
    return {
        "transcript_backed": _score_float(quality.get("speech_transcript_backed")) >= 1.0,
        "evidence_score": _score_float(quality.get("speech_evidence_score")),
        "specificity_score": _score_float(quality.get("speech_specificity_score")),
        "audio_required": _score_float(quality.get("has_audio_modality")) >= 1.0,
    }


def _audio_event_quality_payload(quality: dict[str, Any]) -> dict[str, Any]:
    if str(quality.get("difference_type", "")).strip() != "audio_event":
        return {}
    return {
        "non_speech_score": _score_float(quality.get("non_speech_audio_event_score")),
        "audio_required": _score_float(quality.get("has_audio_modality")) >= 1.0,
    }


def _compose_reject_reason(
    judge: dict[str, Any],
    verification: dict[str, Any] | None = None,
    effective_quality: dict[str, Any] | None = None,
) -> str:
    original_reason = str(judge.get("reject_reason", "")).strip()
    failures: list[str] = []
    if judge.get("reference_satisfies_edit"):
        failures.append("reference already satisfies the edit")
    if not judge.get("target_satisfies_edit"):
        failures.append("target does not satisfy the edit")
    if not judge.get("single_main_difference"):
        failures.append("the pair does not contain a single main difference")
    hard_negative_quality = str(judge.get("hard_negative_quality", "")).strip().lower()
    if hard_negative_quality not in {"good", "weak"}:
        failures.append(f"hard_negative_quality is {hard_negative_quality or 'bad'}")

    quality = effective_quality or judge
    same_context_score = _score_float(quality.get("same_context_score"))
    if same_context_score < MIN_ACCEPT_SAME_CONTEXT_SCORE:
        failures.append(
            f"same_context_score {same_context_score:.3f} is below {MIN_ACCEPT_SAME_CONTEXT_SCORE:.2f}"
        )
    edit_match_score = _score_float(quality.get("edit_match_score"))
    if edit_match_score < MIN_ACCEPT_EDIT_MATCH_SCORE:
        failures.append(
            f"edit_match_score {edit_match_score:.3f} is below {MIN_ACCEPT_EDIT_MATCH_SCORE:.2f}"
        )
    target_uniqueness_score = _score_float(quality.get("target_uniqueness_score"))
    if target_uniqueness_score < MIN_ACCEPT_TARGET_UNIQUENESS_SCORE:
        failures.append(
            f"target_uniqueness_score {target_uniqueness_score:.3f} is below {MIN_ACCEPT_TARGET_UNIQUENESS_SCORE:.2f}"
        )
    if "difference_strength_score" in quality:
        difference_strength_score = _score_float(quality.get("difference_strength_score"))
    else:
        difference_strength_score = MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE
    if difference_strength_score < MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE:
        failures.append(
            f"difference_strength_score {difference_strength_score:.3f} is below {MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE:.2f}"
        )
    visual_near_duplicate_score = _score_float(quality.get("visual_near_duplicate_score"))
    difference_type = str(quality.get("difference_type", "")).strip()
    if _visual_near_duplicate_rejects(visual_near_duplicate_score, difference_type):
        failures.append(
            f"visual_near_duplicate_score {visual_near_duplicate_score:.3f} is too high for visual difference type {difference_type}"
        )
    if difference_type == "action":
        action_evidence_score = _score_float(quality.get("action_evidence_score"))
        if action_evidence_score < MIN_ACCEPT_ACTION_EVIDENCE_SCORE:
            failures.append(
                f"action_evidence_score {action_evidence_score:.3f} is below {MIN_ACCEPT_ACTION_EVIDENCE_SCORE:.2f}"
            )
    if difference_type == "speech":
        if _score_float(quality.get("has_audio_modality")) < 1.0:
            failures.append("speech edit is missing audio modality")
        if not _boolish(judge.get("audio_required")):
            failures.append("speech edit must be marked audio_required")
        if _score_float(quality.get("speech_transcript_backed")) < 1.0:
            failures.append("speech edit is not backed by transcript evidence on both clips")
        speech_evidence_score = _score_float(quality.get("speech_evidence_score"))
        if speech_evidence_score < MIN_ACCEPT_SPEECH_EVIDENCE_SCORE:
            failures.append(
                f"speech_evidence_score {speech_evidence_score:.3f} is below {MIN_ACCEPT_SPEECH_EVIDENCE_SCORE:.2f}"
            )
        speech_specificity_score = _score_float(quality.get("speech_specificity_score"))
        if speech_specificity_score < MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE:
            failures.append(
                f"speech_specificity_score {speech_specificity_score:.3f} is below {MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE:.2f}"
            )
    if difference_type == "audio_event":
        non_speech_audio_event_score = _score_float(quality.get("non_speech_audio_event_score"))
        if non_speech_audio_event_score < MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE:
            failures.append(
                f"non_speech_audio_event_score {non_speech_audio_event_score:.3f} is below {MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE:.2f}"
            )
    if _score_float(quality.get("intraclip_change_conflict")) >= 1.0:
        failures.append("the proposed edit appears to describe an intra-clip transition instead of a cross-clip difference")
    failures.extend(_structured_edit_text_failures(quality))
    if _observable_difference_rejects(quality):
        failures.append("observable_difference gate found no concrete visual delta evidence")
    if _score_float(quality.get("competing_difference_passed", 1.0)) < 1.0:
        failures.append("single_main_difference failed: competing stronger difference")
    if _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) < 1.0:
        failures.append("audio_event lacks independent non-speech audio evidence")
    if verification is not None:
        failures.extend(_verification_failures(verification))
    if not judge.get("accept"):
        failures.append("the model judge did not accept the pair")

    unique_failures: list[str] = []
    for failure in failures:
        if failure not in unique_failures:
            unique_failures.append(failure)
    if original_reason and unique_failures:
        return f"{original_reason} Final gate check: {'; '.join(unique_failures)}."
    if original_reason:
        return original_reason
    if unique_failures:
        return "; ".join(unique_failures)
    return "the pair was rejected without a structured reason from the judge"


def _judge_accepts(
    judge: dict[str, Any],
    verification: dict[str, Any] | None = None,
    effective_quality: dict[str, Any] | None = None,
) -> bool:
    quality = effective_quality or judge
    judge_accepted = bool(
        not judge.get("reference_satisfies_edit")
        and judge.get("target_satisfies_edit")
        and judge.get("single_main_difference")
        and judge.get("hard_negative_quality") in {"good", "weak"}
        and _score_float(quality.get("same_context_score")) >= MIN_ACCEPT_SAME_CONTEXT_SCORE
        and _score_float(quality.get("edit_match_score")) >= MIN_ACCEPT_EDIT_MATCH_SCORE
        and _score_float(quality.get("target_uniqueness_score")) >= MIN_ACCEPT_TARGET_UNIQUENESS_SCORE
        and (
            "difference_strength_score" not in quality
            or _score_float(quality.get("difference_strength_score")) >= MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE
        )
        and not _visual_near_duplicate_rejects(
            _score_float(quality.get("visual_near_duplicate_score")),
            str(quality.get("difference_type", "")).strip(),
        )
        and (
            str(quality.get("difference_type", "")).strip() != "action"
            or _score_float(quality.get("action_evidence_score")) >= MIN_ACCEPT_ACTION_EVIDENCE_SCORE
        )
        and (
            str(quality.get("difference_type", "")).strip() != "speech"
            or (
                _score_float(quality.get("has_audio_modality")) >= 1.0
                and _boolish(judge.get("audio_required"))
                and _score_float(quality.get("speech_transcript_backed")) >= 1.0
                and _score_float(quality.get("speech_evidence_score")) >= MIN_ACCEPT_SPEECH_EVIDENCE_SCORE
                and _score_float(quality.get("speech_specificity_score")) >= MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE
            )
        )
        and (
            str(quality.get("difference_type", "")).strip() != "audio_event"
            or _score_float(quality.get("non_speech_audio_event_score")) >= MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE
        )
        and _score_float(quality.get("intraclip_change_conflict")) < 1.0
        and not _structured_edit_text_failures(quality)
        and not _observable_difference_rejects(quality)
        and _score_float(quality.get("competing_difference_passed", 1.0)) >= 1.0
        and _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) >= 1.0
    )
    if verification is None:
        return bool(judge.get("accept")) and judge_accepted
    return judge_accepted and _verification_accepts(verification)


def _visual_near_duplicate_rejects(score: float, difference_type: str) -> bool:
    return bool(
        difference_type in VISUAL_DIFFERENCE_TYPES
        and score >= MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE
    )


def _verification_accepts(verification: dict[str, Any]) -> bool:
    caption_delta = verification.get("caption_delta", {})
    edit_projection = verification.get("edit_projection", {})
    edit_necessity = verification.get("edit_necessity", {})
    edit_text_quality_check = verification.get("edit_text_quality_check", {})
    return bool(
        not _boolish(caption_delta.get("caption_equivalent"))
        and _boolish(caption_delta.get("has_concrete_difference"))
        and _boolish(caption_delta.get("difference_matches_edit"))
        and _boolish(edit_projection.get("target_matches_projection"))
        and _score_float(edit_projection.get("score")) >= MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE
        and _boolish(edit_necessity.get("edit_needed"))
        and not _boolish(edit_necessity.get("reference_satisfies_edit"))
        and _boolish(edit_necessity.get("target_satisfies_edit"))
        and _score_float(edit_necessity.get("score")) >= MIN_ACCEPT_EDIT_NECESSITY_SCORE
        and _verification_edit_text_quality_accepts(edit_text_quality_check)
    )


def _verification_failures(verification: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    caption_delta = verification.get("caption_delta", {})
    edit_projection = verification.get("edit_projection", {})
    edit_necessity = verification.get("edit_necessity", {})
    edit_text_quality_check = verification.get("edit_text_quality_check", {})
    observable_difference_failure = str(verification.get("observable_difference_failure", "")).strip()
    if observable_difference_failure:
        failures.append(observable_difference_failure)
    if _boolish(caption_delta.get("caption_equivalent")):
        failures.append("caption_delta says reference and target are equivalent")
    if not _boolish(caption_delta.get("has_concrete_difference")):
        failures.append("caption_delta found no concrete difference")
    if not _boolish(caption_delta.get("difference_matches_edit")):
        failures.append("caption_delta difference does not match the edit")
    projection_score = _score_float(edit_projection.get("score"))
    if not _boolish(edit_projection.get("target_matches_projection")):
        failures.append("edit_projection does not match the target")
    if projection_score < MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE:
        failures.append(
            f"edit_projection score {projection_score:.3f} is below {MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE:.2f}"
        )
    necessity_score = _score_float(edit_necessity.get("score"))
    if not _boolish(edit_necessity.get("edit_needed")):
        failures.append("edit_necessity says the edit is not needed")
    if _boolish(edit_necessity.get("reference_satisfies_edit")):
        failures.append("edit_necessity says the reference already satisfies the edit")
    if not _boolish(edit_necessity.get("target_satisfies_edit")):
        failures.append("edit_necessity says the target does not satisfy the edit")
    if necessity_score < MIN_ACCEPT_EDIT_NECESSITY_SCORE:
        failures.append(
            f"edit_necessity score {necessity_score:.3f} is below {MIN_ACCEPT_EDIT_NECESSITY_SCORE:.2f}"
        )
    if not _verification_edit_text_quality_accepts(edit_text_quality_check):
        reason = str(edit_text_quality_check.get("failure_reason", "")).strip() if isinstance(edit_text_quality_check, dict) else ""
        failures.append(f"edit_text_quality_check failed{': ' + reason if reason else ''}")
    return failures


def _structured_edit_text_failures(quality: dict[str, Any]) -> list[str]:
    if "edit_text_quality_score" not in quality:
        return []
    failures: list[str] = []
    edit_text_quality_score = _score_float(quality.get("edit_text_quality_score"))
    if edit_text_quality_score < MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE:
        failures.append(
            f"edit_text_quality_score {edit_text_quality_score:.3f} is below {MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE:.2f}"
        )
    for key, label in (
        ("edit_text_is_imperative", "edit_text is not an imperative edit"),
        ("edit_text_matches_difference_type", "edit_text does not match the difference type"),
        ("edit_text_single_change", "edit_text does not describe a single primary change"),
        ("edit_text_not_caption_like", "edit_text is caption-like"),
        ("edit_text_no_modality_leakage", "edit_text leaks another modality"),
    ):
        if key in quality and _score_float(quality.get(key)) < 1.0:
            failures.append(label)
    return failures


def _observable_difference_rejects(quality: dict[str, Any]) -> bool:
    difference_type = str(quality.get("difference_type", "")).strip()
    if difference_type not in VISUAL_DIFFERENCE_TYPES:
        return False
    if "observable_difference_passed" in quality and _score_float(quality.get("observable_difference_passed")) < 1.0:
        return True
    if "near_duplicate_without_delta" in quality and _score_float(quality.get("near_duplicate_without_delta")) >= 1.0:
        return True
    return False


def _verification_edit_text_quality_accepts(check: Any) -> bool:
    if not isinstance(check, dict) or not check:
        return True
    return bool(
        _boolish(check.get("not_caption_like"))
        and _boolish(check.get("matches_modality"))
        and _boolish(check.get("single_primary_difference"))
        and _boolish(check.get("reference_does_not_satisfy"))
        and _boolish(check.get("target_satisfies"))
        and _score_float(check.get("score")) >= MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE
    )


def _score_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "pass", "accept", "accepted"}


def _evidence_from_annotations(
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    *,
    difference_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_audio_terms = _non_speech_audio_terms(reference_annotation) or _normalize_list(reference_annotation.get("audio_events", []))
    target_audio_terms = _non_speech_audio_terms(target_annotation) or _normalize_list(target_annotation.get("audio_events", []))
    reference_actions = _action_terms_from_annotation(reference_annotation)
    target_actions = _action_terms_from_annotation(target_annotation)
    return {
        "reference_summary": str(reference_annotation.get("summary", "")).strip(),
        "target_summary": str(target_annotation.get("summary", "")).strip(),
        "reference_storyline": list(reference_annotation.get("storyline", [])),
        "target_storyline": list(target_annotation.get("storyline", [])),
        "reference_events": list(reference_annotation.get("events", [])),
        "target_events": list(target_annotation.get("events", [])),
        "reference_timeline_evidence": _timeline_evidence(reference_annotation),
        "target_timeline_evidence": _timeline_evidence(target_annotation),
        "reference_actions": reference_actions,
        "target_actions": target_actions,
        "action_change": _change_text(reference_actions, target_actions),
        "audio_change": _change_text(reference_audio_terms, target_audio_terms),
        "visible_text_change": _change_text(
            reference_annotation.get("visible_text") or reference_annotation.get("on_screen_text", []),
            target_annotation.get("visible_text") or target_annotation.get("on_screen_text", []),
        ),
        "difference_evidence": dict(difference_evidence or {}),
    }


def _change_text(left: Any, right: Any) -> str:
    left_values = _normalize_list(left)
    right_values = _normalize_list(right)
    if left_values == right_values:
        return ""
    return f"{'; '.join(left_values) or 'none'} -> { '; '.join(right_values) or 'none'}"


def _normalized_phrase(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(value).lower()))


def _text_mentions_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = _normalized_phrase(phrase)
    if not normalized_phrase:
        return False
    return normalized_phrase in _normalized_phrase(text)


def _has_intraclip_change_description(text: str, from_value: str, to_value: str) -> bool:
    normalized_text = _normalized_phrase(text)
    if not normalized_text:
        return False
    if not _text_mentions_phrase(text, from_value) or not _text_mentions_phrase(text, to_value):
        return False
    return any(marker in normalized_text for marker in INTRACLIP_CHANGE_MARKERS)


def _annotation_difference_texts(annotation: dict[str, Any], difference_type: str) -> list[str]:
    if difference_type == "speech":
        return _speech_texts_from_annotation(annotation)
    if difference_type == "audio_event":
        values = _non_speech_audio_terms(annotation)
        values.extend(_normalize_list(annotation.get("detective_notes", [])))
        values.extend(_normalize_list(annotation.get("summary", "")))
        return values
    return []


def _has_intraclip_difference_conflict(
    *,
    difference: dict[str, Any],
    reference_caption: str,
    target_caption: str,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> bool:
    difference_type = str(difference.get("type", "")).strip()
    if difference_type not in {"speech", "audio_event"}:
        return False
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if not from_value or not to_value:
        return False

    texts_to_check = [
        reference_caption,
        target_caption,
        *_annotation_difference_texts(reference_annotation, difference_type),
        *_annotation_difference_texts(target_annotation, difference_type),
    ]
    return any(_has_intraclip_change_description(text, from_value, to_value) for text in texts_to_check)


def _reject_record_with_acceptance_issues(record: dict[str, Any], acceptance_issues: list[str]) -> dict[str, Any]:
    updated = dict(record)
    judge = dict(updated.get("judge", {}))
    judge["accept"] = False
    judge["reject_reason"] = "; ".join(acceptance_issues)
    updated["judge"] = judge
    verification = dict(updated.get("verification", {})) if isinstance(updated.get("verification"), dict) else {}
    failures = [str(item) for item in verification.get("failures", []) if str(item).strip()]
    for issue in acceptance_issues:
        failure = f"acceptance gate failed: {issue}"
        if failure not in failures:
            failures.append(failure)
    verification["failures"] = failures
    verification["passed"] = False
    updated["verification"] = verification
    updated["accepted"] = False
    return updated


def _pair_record_acceptance_issues(
    *,
    root: Path,
    record: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    for field_name in ("reference_video", "target_video"):
        raw_path = str(record.get(field_name, "")).strip()
        if raw_path and not _resolve_under_root(root, raw_path).exists():
            issues.append(f"{field_name} does not exist: {raw_path}")

    for negative_path in [str(item).strip() for item in record.get("hard_negatives", []) if str(item).strip()]:
        if not _resolve_under_root(root, negative_path).exists():
            issues.append(f"hard_negative does not exist: {negative_path}")

    if _has_intraclip_difference_conflict(
        difference=record.get("difference", {}),
        reference_caption=str(record.get("reference_caption", "")),
        target_caption=str(record.get("target_caption", "")),
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
    ):
        issues.append("the proposed difference appears inside a single clip instead of between reference and target")
    difference = record.get("difference", {})
    issues.extend(_speech_content_edit_issues(edit_text=str(record.get("edit_text", "")), difference=difference))
    if str(difference.get("type", "")).strip() == "audio_event":
        from_value = str(difference.get("from", "")).strip()
        to_value = str(difference.get("to", "")).strip()
        if _is_speech_only_audio_phrase(from_value) or _is_speech_only_audio_phrase(to_value):
            issues.append("audio_event must not use speech-only or narration-only text as the main difference")
    quality = record.get("quality", {})
    if isinstance(quality, dict):
        issues.extend(_structured_edit_text_failures(quality))
        if _observable_difference_rejects(quality):
            issues.append("observable_difference gate found no concrete visual delta evidence")
        if _score_float(quality.get("competing_difference_passed", 1.0)) < 1.0:
            issues.append("single_main_difference failed: competing stronger difference")
        if _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) < 1.0:
            issues.append("audio_event lacks independent non-speech audio evidence")
    issues.extend(
        _synthetic_edit_record_issues(
            root=root,
            record=record,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
        )
    )
    return issues


def _synthetic_edit_record_issues(
    *,
    root: Path,
    record: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> list[str]:
    if str(record.get("source_type", "natural")).strip() != "synthetic_edit":
        return []
    issues: list[str] = []
    quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
    difference_type = str(record.get("difference", {}).get("type", "")).strip()
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    route = _synthetic_generation_route(generation)
    is_audio_route = _is_audio_synthetic_route(route)
    source_context = record.get("source_context", {}) if isinstance(record.get("source_context"), dict) else {}
    relation = str(source_context.get("relation", "")).strip()
    visual_score = _score_float(quality.get("visual_near_duplicate_score"))
    if (
        relation == "synthetic_from_reference"
        and difference_type in VISUAL_DIFFERENCE_TYPES
        and not is_audio_route
        and visual_score < MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE
    ):
        issues.append(
            f"synthetic target does not preserve reference visual context: visual_near_duplicate_score {visual_score:.3f} is below {MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE:.2f}"
        )
    if is_audio_route and visual_score < MIN_SYNTHETIC_AUDIO_VISUAL_CONTEXT_SCORE:
        issues.append(
            f"audio synthetic target changed visual stream: visual_near_duplicate_score {visual_score:.3f} is below {MIN_SYNTHETIC_AUDIO_VISUAL_CONTEXT_SCORE:.2f}"
        )

    reference_path = _resolve_under_root(root, str(record.get("reference_video", "")).strip())
    target_path = _resolve_under_root(root, str(record.get("target_video", "")).strip())
    reference_media = probe_media(reference_path)
    target_media = probe_media(target_path)
    if "error" not in target_media and not target_media.get("has_video"):
        issues.append("synthetic target is missing a video stream")
    if "error" not in reference_media and "error" not in target_media:
        reference_duration = float(reference_media.get("duration_seconds") or 0.0)
        target_duration = float(target_media.get("duration_seconds") or 0.0)
        if reference_media.get("has_audio") and not target_media.get("has_audio"):
            issues.append("synthetic target is missing audio copied from the reference")
        if reference_duration > 0 and target_duration > 0:
            ratio = abs(reference_duration - target_duration) / reference_duration
            if ratio > 0.10:
                issues.append(
                    f"synthetic target duration drift {ratio:.3f} exceeds 0.10 from the reference"
                )
    postprocess = generation.get("postprocess", {}) if isinstance(generation.get("postprocess"), dict) else {}
    if (
        difference_type in VISUAL_DIFFERENCE_TYPES
        and not is_audio_route
        and "error" not in reference_media
        and reference_media.get("has_audio")
        and not postprocess.get("audio_copied_from_reference")
    ):
        issues.append("visual synthetic edits must record generation.postprocess.audio_copied_from_reference=true")
    if is_audio_route:
        expected_event = _synthetic_audio_expected_event(record)
        if not expected_event:
            issues.append("audio_event target sound was not detected by audio observer: expected_event is missing")
        elif not _audio_terms_mention_event(_non_speech_audio_terms(target_annotation), expected_event):
            issues.append(f"audio_event target sound was not detected by audio observer: {expected_event}")
        elif _audio_terms_mention_event(_non_speech_audio_terms(reference_annotation), expected_event):
            issues.append(f"reference audio already contains requested audio event: {expected_event}")
    return issues


def _accepted_record_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float, str]:
    quality = record.get("quality", {})
    return (
        -_score_float(quality.get("difference_strength_score")),
        -_score_float(quality.get("same_context_score")),
        -_score_float(quality.get("target_uniqueness_score")),
        -_score_float(quality.get("edit_match_score")),
        str(record.get("proposal_id", "")).strip(),
    )


def _accepted_record_signature(record: dict[str, Any]) -> tuple[str, ...]:
    difference = record.get("difference", {})
    from_value = _normalized_phrase(str(difference.get("from", "")).strip())
    to_value = _normalized_phrase(str(difference.get("to", "")).strip())
    if not from_value and not to_value:
        from_value = _normalized_phrase(str(record.get("edit_text", "")).strip())
    if str(record.get("source_type", "natural")).strip() == "synthetic_edit":
        return (
            "synthetic_edit",
            str(record.get("proposal_id", "")).strip(),
            str(record.get("reference_video", "")).strip(),
            str(record.get("target_video", "")).strip(),
            str(difference.get("type", "")).strip(),
            from_value,
            to_value,
        )
    return (
        str(record.get("group_id", "")).strip(),
        str(difference.get("type", "")).strip(),
        from_value,
        to_value,
        str(record.get("source_context", {}).get("relation", "")).strip(),
    )


def _select_final_accepted_records(
    records: list[dict[str, Any]],
    *,
    max_accepted_pairs: int,
) -> list[dict[str, Any]]:
    accepted_candidates = sorted(
        [record for record in records if bool(record.get("accepted"))],
        key=_accepted_record_sort_key,
    )
    if not accepted_candidates or max_accepted_pairs <= 0:
        return []

    selected: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    selected_ids: set[str] = set()
    selected_target_videos: set[str] = set()

    def try_select(record: dict[str, Any]) -> bool:
        signature = _accepted_record_signature(record)
        proposal_id = str(record.get("proposal_id", "")).strip()
        target_video = str(record.get("target_video", "")).strip()
        if signature in seen_signatures or proposal_id in selected_ids:
            return False
        if target_video and target_video in selected_target_videos:
            return False
        selected.append(record)
        seen_signatures.add(signature)
        selected_ids.add(proposal_id)
        if target_video:
            selected_target_videos.add(target_video)
        return True

    for difference_type, target_count in FINAL_ACCEPT_BUCKET_TARGETS.items():
        bucket_count = 0
        for record in accepted_candidates:
            if len(selected) >= max_accepted_pairs or bucket_count >= target_count:
                break
            if str(record.get("difference", {}).get("type", "")).strip() != difference_type:
                continue
            if try_select(record):
                bucket_count += 1

    for record in accepted_candidates:
        if len(selected) >= max_accepted_pairs:
            break
        try_select(record)

    return [_accepted_sample_from_record(record, index + 1) for index, record in enumerate(selected)]


def _accepted_sample_from_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    source_type = str(record.get("source_type", "natural")).strip() or "natural"
    if source_type == "synthetic_edit":
        identity = str(record.get("proposal_id") or record.get("target_video") or record.get("edit_text") or index)
        sample_id = f"covr_omni_synth_{_stable_hash(identity)[:8]}"
    else:
        sample_id = f"covr_omni_pilot_{index:04d}"
    return {
        "sample_id": sample_id,
        "proposal_id": record["proposal_id"],
        "reference_clip_id": record.get("reference_clip_id", ""),
        "target_clip_id": record.get("target_clip_id", ""),
        "reference_video": record["reference_video"],
        "target_video": record["target_video"],
        "source_type": source_type,
        "edit_text": record["edit_text"],
        "modalities": list(record["modalities"]),
        "reference_caption": record["reference_caption"],
        "target_caption": record["target_caption"],
        "difference": dict(record["difference"]),
        "hard_negatives": list(record["hard_negatives"]),
        "quality": dict(record["quality"]),
        "source": dict(record["source"]),
        "generation": dict(record.get("generation", {})),
        "source_context": dict(record.get("source_context", {})),
        "direction_corrected": bool(record.get("direction_corrected")),
        "evidence": dict(record.get("evidence", {})),
        "judge": dict(record.get("judge", {})),
        "verification": dict(record.get("verification", {})),
        "edit_text_quality": dict(record.get("edit_text_quality", {})),
        "observable_difference": dict(record.get("observable_difference", {})),
        "competing_difference": dict(record.get("competing_difference", {})),
        "audio_event_evidence": dict(record.get("audio_event_evidence", {})),
        "speech_quality": dict(record.get("speech_quality", {})),
        "audio_event_quality": dict(record.get("audio_event_quality", {})),
        "transcript_backed": record.get("transcript_backed"),
        "group_id": record.get("group_id", ""),
        "group_reason": record.get("group_reason", ""),
    }


def _build_source_metadata(
    *,
    root: Path,
    target_annotation: dict[str, Any],
    raw_index: dict[str, dict[str, Any]],
) -> dict[str, str]:
    asset_id = str(target_annotation.get("source_asset_id", "")).strip()
    raw_asset = raw_index.get(asset_id, {})
    platform = str(raw_asset.get("dataset") or "unknown").strip()
    raw_path = str(raw_asset.get("path", "")).strip()
    if raw_path:
        resolved_path = Path(raw_path)
    else:
        resolved_path = _resolve_under_root(root, target_annotation["output_path"])
    url = resolved_path.resolve().as_uri() if resolved_path.is_absolute() or resolved_path.exists() else ""
    return {
        "platform": platform or "unknown",
        "url": url,
        "license_note": DEFAULT_LICENSE_NOTE,
    }


def _load_raw_asset_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for item in _load_jsonl(path):
        asset_id = str(item.get("asset_id", "")).strip()
        if asset_id:
            mapping[asset_id] = item
    return mapping


def _load_records_by_key(path: Path, key_name: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for item in _load_jsonl(path):
        key = str(item.get(key_name, "")).strip()
        if key:
            mapping[key] = item
    return mapping


def _annotation_lookup(*, root: Path, annotations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        clip_id = str(annotation.get("clip_id", "")).strip()
        if clip_id:
            mapping[clip_id] = annotation
        output_path = str(annotation.get("output_path", "")).strip()
        if output_path:
            resolved = _resolve_under_root(root, output_path)
            for key in _path_lookup_keys(root, resolved, output_path):
                mapping.setdefault(key, annotation)
    return mapping


def _path_lookup_keys(root: Path, resolved_path: Path, raw_path: str | Path) -> list[str]:
    keys: list[str] = []

    def add(value: str) -> None:
        normalized = value.replace("\\", "/").strip()
        if normalized and normalized not in keys:
            keys.append(normalized)

    add(str(raw_path))
    add(str(resolved_path))
    try:
        add(str(resolved_path.resolve()))
    except OSError:
        pass
    try:
        add(resolved_path.resolve().relative_to(root.resolve()).as_posix())
    except (OSError, ValueError):
        pass
    return keys


def _annotation_for_video_edit_plan(
    *,
    root: Path,
    lookup: dict[str, dict[str, Any]],
    record: dict[str, Any],
    video_field: str,
    caption_field: str,
) -> dict[str, Any]:
    raw_path = str(record.get(video_field, "")).strip()
    if raw_path:
        resolved = _resolve_under_root(root, raw_path)
        for key in _path_lookup_keys(root, resolved, raw_path):
            if key in lookup:
                return lookup[key]
    caption = str(record.get(caption_field, "")).strip()
    return {
        "clip_id": _safe_id(raw_path or caption or "unknown_clip"),
        "output_path": raw_path,
        "summary": caption,
        "subjects": [],
        "object_counts": {},
        "actions": [],
        "scene": "",
        "attributes": [],
        "visible_text": [],
        "speech": [],
        "audio_events": [],
        "modalities": ["visual"],
    }


def _review_annotation_for_record(
    *,
    root: Path,
    lookup: dict[str, dict[str, Any]],
    record: dict[str, Any],
    video_field: str,
    clip_id_field: str,
) -> dict[str, Any]:
    clip_id = str(record.get(clip_id_field, "")).strip()
    if clip_id and clip_id in lookup:
        return lookup[clip_id]
    raw_path = str(record.get(video_field, "")).strip()
    if raw_path:
        resolved = _resolve_under_root(root, raw_path)
        for key in _path_lookup_keys(root, resolved, raw_path):
            if key in lookup:
                return lookup[key]
    return {}


def _manual_review_item_markdown(
    *,
    metadata: dict[str, Any],
    reference_filename: str,
    target_filename: str,
) -> str:
    difference = metadata.get("difference") if isinstance(metadata.get("difference"), dict) else {}
    verification = metadata.get("verification") if isinstance(metadata.get("verification"), dict) else {}
    observable = metadata.get("observable_difference") if isinstance(metadata.get("observable_difference"), dict) else {}
    competing = metadata.get("competing_difference") if isinstance(metadata.get("competing_difference"), dict) else {}
    lines = [
        f"# {metadata.get('index')}. {metadata.get('sample_id')} | {difference.get('type', '')}",
        "",
        f"- 修改文本: {metadata.get('edit_text', '')}",
        f"- 参考视频描述: {metadata.get('reference_caption', '')}",
        f"- 目标视频描述: {metadata.get('target_caption', '')}",
        f"- difference: `{json.dumps(difference, ensure_ascii=False)}`",
        f"- verification.passed: `{verification.get('passed')}`",
        f"- observable_difference.passed: `{observable.get('passed')}`",
        f"- competing_difference.passed: `{competing.get('passed')}`",
        f"- src_ref_images: `{json.dumps(metadata.get('src_ref_images', []), ensure_ascii=False)}`",
        f"- src_mask: `{metadata.get('src_mask', '')}`",
        "",
        "## 视频文件",
        "",
    ]
    if reference_filename:
        lines.append(f"- 参考视频本地副本: `{reference_filename}`")
    lines.append(f"- 参考视频原路径: `{metadata.get('reference_video_absolute', '')}`")
    if target_filename:
        lines.append(f"- 目标视频本地副本: `{target_filename}`")
    lines.append(f"- 目标视频原路径: `{metadata.get('target_video_absolute', '')}`")
    copied_refs = _normalize_list(metadata.get("copied_src_ref_images", []))
    if copied_refs:
        lines.extend(["", "## src_ref_images", ""])
        for copied in copied_refs:
            lines.append(f"- `{copied}`")
    if metadata.get("copied_src_mask"):
        lines.extend(["", "## mask", "", f"- `{metadata.get('copied_src_mask')}`"])
    lines.extend(
        [
            "",
            "## 人工核验问题",
            "",
            "- reference 和 target 是否还是同一视频上下文？",
            "- target 是否只体现 edit_text 的一个主差异？",
            "- edit_text 方向是否正确？",
            "- 是否有额外换场景、换动作、换文字、换主体？",
            "- 如果是视觉 synthetic，target 是否保留 reference audio？",
            "- 如果是音频 synthetic，画面是否完全一致，差异是否只来自音频？",
            "",
        ]
    )
    return "\n".join(lines)


def _manual_review_index_markdown(
    *,
    items: list[dict[str, Any]],
    source_pairs_path: str,
    missing_videos: list[str],
) -> str:
    lines = [
        "# Manual Review Bundle",
        "",
        f"- Source pairs: `{source_pairs_path}`",
        f"- Sample count: `{len(items)}`",
        f"- Missing video count: `{len(missing_videos)}`",
        "",
        "## Samples",
        "",
        "| # | sample_id | type | edit_text | folder |",
        "|---|-----------|------|-----------|--------|",
    ]
    for item in items:
        lines.append(
            f"| {item['index']} | `{item['sample_id']}` | `{item.get('difference_type', '')}` | "
            f"{item.get('edit_text', '')} | `{Path(item['item_dir']).name}` |"
        )
    if missing_videos:
        lines.extend(["", "## Missing Videos", ""])
        lines.extend(f"- `{path}`" for path in missing_videos)
    lines.append("")
    return "\n".join(lines)


def _video_edit_model_route(difference_type: str) -> str | None:
    difference_type = str(difference_type).strip()
    if difference_type == "object_presence":
        return "vace_controlled"
    if difference_type == "attribute":
        return "vace_controlled"
    if difference_type == "scene":
        return "vace_controlled"
    if difference_type == "action":
        return "ltx2_retake"
    return None


def _safe_visual_ideation_candidate(candidate: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any] | None:
    anchor = _safe_visual_edit_anchor(annotation)
    if anchor is None:
        return None
    edit_text, difference, reason = anchor
    source_edit_text = str(candidate.get("edit_text", "")).strip()
    proposal_seed = str(candidate.get("proposal_id", "")) or str(candidate.get("reference_video", "")) + edit_text
    revised = dict(candidate)
    revised["proposal_id"] = f"{str(candidate.get('proposal_id', '')).strip() or 'candidate'}__visual_ideation_{_stable_hash(proposal_seed)[:8]}"
    revised["edit_text"] = edit_text
    revised["difference"] = difference
    revised["source_candidate_edit_text"] = source_edit_text
    revised["source_candidate_difference"] = candidate.get("difference", {})
    revised["candidate_source"] = "safe_visual_ideation_from_reference"
    revised["ideation_reason"] = reason
    return revised


def _video_edit_exploration_candidates(candidate: dict[str, Any], annotation: dict[str, Any]) -> list[dict[str, Any]]:
    reference_video = str(candidate.get("reference_video", "")).strip()
    if not reference_video:
        return []

    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))
    object_names = list(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    scene = str(annotation.get("scene", "")).strip()
    summary = str(annotation.get("summary", "")).strip()
    text = _normalized_phrase(" ".join([summary, scene, " ".join(subjects), " ".join(object_names)]))
    source_edit_text = str(candidate.get("edit_text", "")).strip()
    source_difference = candidate.get("difference", {})
    base_proposal = str(candidate.get("proposal_id", "")).strip() or _stable_hash(reference_video)[:8]

    def build(
        *,
        family: str,
        edit_text: str,
        difference: dict[str, Any],
        edit_token: str,
        edit_region: str,
        mask_query: str,
        goal: str,
    ) -> dict[str, Any]:
        seed = "|".join([base_proposal, reference_video, family, edit_text])
        revised = dict(candidate)
        revised["proposal_id"] = f"{base_proposal}__vace_explore_{_safe_id(family)}_{_stable_hash(seed)[:8]}"
        revised["edit_text"] = edit_text
        revised["difference"] = difference
        revised["source_candidate_edit_text"] = source_edit_text
        revised["source_candidate_difference"] = source_difference
        revised["candidate_source"] = "vace_exploration_from_reference"
        revised["exploration_family"] = family
        revised["exploration_goal"] = goal
        revised["suggested_edit_token"] = edit_token
        revised["suggested_edit_region"] = edit_region
        revised["suggested_mask_query"] = mask_query
        revised["suggested_preserve_regions"] = _exploration_preserve_regions(annotation, edit_region)
        return revised

    candidates: list[dict[str, Any]] = []
    if any(marker in text for marker in ("robot", "robotic", "action figure")):
        candidates.append(
            build(
                family="attribute_color",
                edit_text="change the robot body color from black and gold to bright yellow",
                difference={
                    "type": "attribute",
                    "from": "black and gold robot body",
                    "to": "bright yellow robot body",
                    "description": "The existing robot body changes from black and gold to bright yellow.",
                },
                edit_token="bright yellow robot body",
                edit_region="robot body",
                mask_query="robot body",
                goal="test existing-subject color editing",
            )
        )
        candidates.append(
            build(
                family="attribute_material",
                edit_text="change the robot body material from black and gold plastic to metallic silver",
                difference={
                    "type": "attribute",
                    "from": "black and gold plastic robot body",
                    "to": "metallic silver robot body",
                    "description": "The existing robot body material changes to metallic silver.",
                },
                edit_token="metallic silver robot body",
                edit_region="robot body",
                mask_query="robot body",
                goal="test material and surface editing",
            )
        )

    if any(marker in text for marker in ("shirt", "jacket", "coat", "dress", "clothing", "outfit")):
        candidates.append(
            build(
                family="clothing_color",
                edit_text="change the clothing color to deep navy blue",
                difference={
                    "type": "attribute",
                    "from": "original clothing color",
                    "to": "deep navy blue clothing",
                    "description": "The existing clothing changes to deep navy blue.",
                },
                edit_token="deep navy blue clothing",
                edit_region="clothing",
                mask_query="clothing",
                goal="test clothing recoloring",
            )
        )
        candidates.append(
            build(
                family="clothing_type",
                edit_text="change the outfit into a black jacket",
                difference={
                    "type": "attribute",
                    "from": "original outfit",
                    "to": "black jacket",
                    "description": "The existing outfit changes into a black jacket.",
                },
                edit_token="black jacket",
                edit_region="clothing",
                mask_query="clothing",
                goal="test masked clothing type change",
            )
        )

    if any(marker in text for marker in ("car", "vehicle", "truck", "bus")):
        candidates.append(
            build(
                family="vehicle_color",
                edit_text="change the vehicle body color to bright orange",
                difference={
                    "type": "attribute",
                    "from": "original vehicle body color",
                    "to": "bright orange vehicle body",
                    "description": "The existing vehicle body changes to bright orange.",
                },
                edit_token="bright orange vehicle body",
                edit_region="vehicle body",
                mask_query="vehicle body",
                goal="test large vehicle color editing",
            )
        )

    if any(marker in text for marker in ("room", "office", "kitchen", "street", "studio", "wall", "background")):
        candidates.append(
            build(
                family="background_change",
                edit_text="change the background to a futuristic laboratory",
                difference={
                    "type": "scene",
                    "from": "original background",
                    "to": "futuristic laboratory background",
                    "description": "The background changes to a futuristic laboratory while the main subject remains.",
                },
                edit_token="futuristic laboratory background",
                edit_region="background",
                mask_query=_foreground_mask_query_from_annotation(annotation),
                goal="test masked background replacement",
            )
        )
        candidates.append(
            build(
                family="style_lighting",
                edit_text="change the scene style to cinematic neon lighting",
                difference={
                    "type": "scene",
                    "from": "original scene style",
                    "to": "cinematic neon lighting style",
                    "description": "The scene style changes to cinematic neon lighting.",
                },
                edit_token="cinematic neon lighting style",
                edit_region="background",
                mask_query=_foreground_mask_query_from_annotation(annotation),
                goal="test style and lighting editing",
            )
        )

    replacement_count = 0
    removal_count = 0
    for object_name in object_names + subjects:
        normalized_object = _normalized_phrase(object_name)
        if not normalized_object or normalized_object in {"person", "man", "woman", "people", "hand", "hands"}:
            continue
        replacement = VACE_EXPLORATION_OBJECT_REPLACEMENTS.get(normalized_object)
        if replacement and replacement_count < 2:
            replacement_count += 1
            candidates.append(
                build(
                    family="object_replacement",
                    edit_text=f"replace the {object_name} with a {replacement}",
                    difference={
                        "type": "object_presence",
                        "from": object_name,
                        "to": replacement,
                        "description": f"The existing {object_name} is replaced by a {replacement}.",
                    },
                    edit_token=replacement,
                    edit_region=object_name,
                    mask_query=object_name,
                    goal="test masked object replacement",
                )
            )
        if normalized_object in VACE_EXPLORATION_REMOVABLE_OBJECTS and removal_count < 2:
            removal_count += 1
            candidates.append(
                build(
                    family="object_removal",
                    edit_text=f"remove the {object_name} from the scene",
                    difference={
                        "type": "object_presence",
                        "from": object_name,
                        "to": f"no {object_name}",
                        "description": f"The existing {object_name} is removed from the scene.",
                    },
                    edit_token=object_name,
                    edit_region=object_name,
                    mask_query=object_name,
                    goal="test masked object removal and inpainting",
                )
            )

    return candidates


def _exploration_preserve_regions(annotation: dict[str, Any], edit_region: str) -> list[str]:
    values: list[str] = []
    values.extend(_normalize_list(annotation.get("subjects", [])))
    values.extend(list(_normalize_object_counts(annotation.get("object_counts", {})).keys()))
    values.extend(_normalize_list(annotation.get("actions", [])))
    scene = str(annotation.get("scene", "")).strip()
    if scene:
        values.append(scene)
    values.extend(["camera motion", "lighting", "timing", "visible text"])
    edit_key = _normalized_phrase(edit_region)
    preserved: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalized_phrase(value)
        if not key or key == edit_key or key in seen:
            continue
        seen.add(key)
        preserved.append(str(value).strip())
        if len(preserved) >= 8:
            break
    return preserved


def _safe_visual_edit_anchor(annotation: dict[str, Any]) -> tuple[str, dict[str, Any], str] | None:
    values: list[str] = [
        str(annotation.get("summary", "")),
        str(annotation.get("scene", "")),
    ]
    values.extend(_normalize_list(annotation.get("subjects", [])))
    values.extend(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    text = _normalized_phrase(" ".join(values))
    anchors = (
        (
            ("robot", "robotic", "action figure"),
            "change the robot body color from black and gold to bright yellow",
            "black and gold robot body",
            "bright yellow robot body",
            "robot body",
        ),
        (
            ("car", "vehicle", "truck", "bus"),
            "change the vehicle color to bright red",
            "original vehicle color",
            "bright red vehicle body",
            "vehicle body",
        ),
        (
            ("shirt", "jacket", "coat", "dress", "clothing"),
            "change the clothing color to bright blue",
            "original clothing color",
            "bright blue clothing",
            "clothing",
        ),
        (
            ("room", "office", "kitchen", "street"),
            "change the background to a futuristic laboratory",
            "original background",
            "futuristic laboratory background",
            "background",
        ),
    )
    for markers, edit_text, from_value, to_value, region in anchors:
        if any(marker in text for marker in markers):
            difference = {
                "type": "attribute",
                "from": from_value,
                "to": to_value,
                "description": f"The existing {region} changes from {from_value} to {to_value}.",
            }
            return edit_text, difference, f"reference has a stable existing {region} for attribute-based VACE editing"
    return None


def _relax_safe_visual_ideation_risk(risk: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if str(candidate.get("candidate_source", "")) != "safe_visual_ideation_from_reference":
        return risk
    difference = candidate.get("difference") if isinstance(candidate.get("difference"), dict) else {}
    if str(difference.get("type", "")).strip() != "attribute":
        return risk
    edit_text = str(candidate.get("edit_text", "")).lower()
    stable_surface_markers = {
        "robot",
        "vehicle",
        "clothing",
        "shirt",
        "jacket",
        "body",
        "color",
        "colour",
        "background",
        "laboratory",
        "lab",
        "style",
        "cyberpunk",
    }
    if not any(marker in edit_text for marker in stable_surface_markers):
        return risk
    risk_reasons = [str(reason) for reason in risk.get("risk_reasons", [])]
    hard_reasons = {"visible_text_present", "scene_or_shot_change", "ui_or_text_heavy_scene", "many_subjects"}
    if any(reason in hard_reasons for reason in risk_reasons):
        return risk
    relaxed = dict(risk)
    relaxed["allow_generation"] = True
    relaxed["risk_level"] = "medium" if risk_reasons else str(risk.get("risk_level", "low"))
    relaxed["safe_visual_ideation_relaxed"] = True
    relaxed["relaxed_risk_reasons"] = [
        reason
        for reason in risk_reasons
        if reason in {"multiple_actions", "multi_event_timeline", "speaking_person", "long_storyline"}
    ]
    locks = [
        str(item).strip()
        for item in risk.get("locks", [])
        if str(item).strip()
    ]
    extra_locks = [
        "limit the edit to the named existing subject attribute only",
        "preserve all text, hands, people, actions, motion order, and background content exactly",
    ]
    for lock in extra_locks:
        if lock not in locks:
            locks.append(lock)
    relaxed["locks"] = locks
    return relaxed


def _relax_visual_exploration_risk(risk: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if str(candidate.get("candidate_source", "")) != "vace_exploration_from_reference":
        return risk
    risk_reasons = [str(reason) for reason in risk.get("risk_reasons", [])]
    relaxed = dict(risk)
    relaxed["allow_generation"] = True
    relaxed["risk_level"] = "exploration_high" if risk_reasons else "exploration_low"
    relaxed["vace_exploration_relaxed"] = True
    relaxed["relaxed_risk_reasons"] = risk_reasons
    locks = [
        str(item).strip()
        for item in risk.get("locks", [])
        if str(item).strip()
    ]
    for lock in (
        "this is an exploration run; generate the requested single masked edit even if the reference is risky",
        "preserve all non-masked regions, visible text, people, camera motion, action timing, and scene layout exactly",
    ):
        if lock not in locks:
            locks.append(lock)
    relaxed["locks"] = locks
    return relaxed


def _normalize_model_planned_visual_difference(difference: dict[str, Any], *, edit_text: str) -> dict[str, Any]:
    return dict(difference)


def _video_edit_route_suitability(
    *,
    route: str,
    difference: dict[str, Any],
    edit_text: str,
    edit_token: str,
    edit_region: str,
    reference_annotation: dict[str, Any],
) -> dict[str, Any]:
    if route != "vace_controlled":
        return {"allow_generation": True, "reason": "route_supported"}

    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    combined_text = _normalized_phrase(" ".join([edit_text, edit_token, from_value, to_value, edit_region]))
    combined_tokens = set(TOKEN_PATTERN.findall(combined_text))
    tiny_markers = {_normalized_phrase(marker) for marker in VACE_TINY_OR_INSERTION_MARKERS}
    if any(marker and marker in combined_text for marker in tiny_markers):
        return {
            "allow_generation": False,
            "reason": "vace_rejects_tiny_or_naked_object_edit",
            "priority": "D",
        }

    if difference_type == "object_presence":
        if _absence_like_phrase(from_value) and not _absence_like_phrase(to_value):
            return {
                "allow_generation": False,
                "reason": "vace_rejects_naked_object_insertion",
                "priority": "D",
            }
        if _absence_like_phrase(to_value) and not _absence_like_phrase(from_value):
            return {
                "allow_generation": True,
                "reason": "object_removal_or_inpainting",
                "priority": "S",
            }
        return {
            "allow_generation": True,
            "reason": "existing_object_replacement",
            "priority": "S",
        }

    if difference_type == "attribute":
        if _absence_like_phrase(from_value) or _absence_like_phrase(to_value):
            return {
                "allow_generation": False,
                "reason": "vace_rejects_absence_based_attribute",
                "priority": "D",
            }
        if not (combined_tokens & VACE_ATTRIBUTE_MARKERS):
            return {
                "allow_generation": False,
                "reason": "vace_attribute_lacks_large_visible_property",
                "priority": "C",
            }
        priority = "S" if any(marker in combined_text for marker in ("clothing", "shirt", "jacket", "dress", "background", "style", "robot body", "vehicle")) else "A"
        return {
            "allow_generation": True,
            "reason": "existing_subject_attribute_edit",
            "priority": priority,
        }

    if difference_type == "scene":
        if not any(marker in combined_text for marker in VACE_BACKGROUND_STYLE_MARKERS):
            return {
                "allow_generation": False,
                "reason": "vace_scene_edit_lacks_background_or_style_target",
                "priority": "C",
            }
        return {
            "allow_generation": True,
            "reason": "background_or_style_edit",
            "priority": "S",
        }

    return {
        "allow_generation": False,
        "reason": f"vace_rejects_{difference_type or 'unknown'}_edit",
        "priority": "D",
    }


def _video_mask_query(
    *,
    difference: dict[str, Any],
    edit_text: str,
    edit_token: str,
    edit_region: str,
    route: str,
    suitability: dict[str, Any],
    reference_annotation: dict[str, Any] | None = None,
) -> str:
    if route != "vace_controlled":
        return ""
    difference_type = str(difference.get("type", "")).strip()
    reason = str(suitability.get("reason", "")).strip()
    combined = _normalized_phrase(" ".join([edit_text, edit_token, edit_region, str(difference.get("description", ""))]))
    if difference_type == "scene" or "background" in reason or "background" in combined:
        return _foreground_mask_query_from_annotation(reference_annotation or {})
    if any(marker in combined for marker in ("shirt", "jacket", "coat", "dress", "clothing", "outfit")):
        return "clothing"
    if any(marker in combined for marker in ("robot body", "robotic body", "robot shell")):
        return "robot body"
    if any(marker in combined for marker in ("vehicle body", "car body", "truck body", "bus body")):
        return "vehicle body"
    if difference_type == "object_presence" and _absence_like_phrase(str(difference.get("to", ""))):
        from_value = str(difference.get("from", "")).strip()
        if from_value and not _absence_like_phrase(from_value):
            return from_value[:120]
    if "replacement" in reason or difference_type in {"object_presence", "object_count"}:
        from_value = str(difference.get("from", "")).strip()
        if from_value and not _absence_like_phrase(from_value):
            return from_value[:120]
    if edit_region and not edit_region.startswith("localized region around"):
        return edit_region[:120]
    return (edit_token or str(difference.get("target", "")).strip() or edit_region)[:120]


def _foreground_mask_query_from_annotation(annotation: dict[str, Any]) -> str:
    candidates: list[str] = []
    candidates.extend(_normalize_list(annotation.get("main_subjects", [])))
    candidates.extend(_normalize_list(annotation.get("subjects", [])))
    reference_understanding = annotation.get("reference_understanding")
    if isinstance(reference_understanding, dict):
        candidates.extend(_normalize_list(reference_understanding.get("main_subjects", [])))
    candidates.extend(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    generic = {"background", "scene", "room", "wall", "floor", "table", "desk", "lighting", "camera motion"}
    for candidate in candidates:
        item = str(candidate).strip()
        key = _normalized_phrase(item)
        if not item or key in generic:
            continue
        if any(token in key.split() for token in ("man", "woman", "person", "girl", "boy", "robot", "vehicle", "car", "dog", "cat")):
            return item[:120]
    for candidate in candidates:
        item = str(candidate).strip()
        key = _normalized_phrase(item)
        if item and key not in generic:
            return item[:120]
    return "main subject"


def _video_preserve_regions(
    *,
    preserve_tokens: list[str],
    edit_region: str,
    reference_annotation: dict[str, Any],
) -> list[str]:
    values = [str(item).strip() for item in preserve_tokens if str(item).strip()]
    values.extend(_normalize_list(reference_annotation.get("subjects", [])))
    scene = str(reference_annotation.get("scene", "")).strip()
    if scene:
        values.append(scene)
    values.extend(["camera motion", "lighting", "timing"])
    edit_key = _normalized_phrase(edit_region)
    regions: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalized_phrase(value)
        if not key or key == edit_key or key in seen:
            continue
        seen.add(key)
        regions.append(value)
        if len(regions) >= 8:
            break
    return regions


def _video_mask_mode(plan: dict[str, Any]) -> str:
    difference = plan.get("difference") if isinstance(plan.get("difference"), dict) else {}
    edit_region = _normalized_phrase(str(plan.get("edit_region", "")))
    mask_query = _normalized_phrase(str(plan.get("mask_query", "")))
    difference_type = str(difference.get("type", "")).strip()
    if difference_type == "scene" or "background" in edit_region or mask_query == "background":
        return "edit_background_inverse_subject"
    if difference_type == "object_presence" and _absence_like_phrase(str(difference.get("to", ""))):
        return "remove_or_inpaint_masked_object"
    if difference_type in {"object_presence", "object_count"}:
        return "replace_masked_object"
    return "edit_masked_region"


def _video_mask_gate_defaults(*, mask_mode: str = "", mask_query: str = "") -> dict[str, Any]:
    normalized_query = _normalized_phrase(mask_query)
    if mask_mode in {"replace_masked_object", "remove_or_inpaint_masked_object"}:
        min_coverage = 0.005
        max_coverage = 0.20
    elif mask_mode == "edit_background_inverse_subject":
        min_coverage = 0.20
        max_coverage = 0.90
    elif any(marker in normalized_query for marker in ("clothing", "shirt", "jacket", "outfit")):
        min_coverage = 0.03
        max_coverage = 0.30
    else:
        min_coverage = MIN_VIDEO_MASK_COVERAGE_RATIO
        max_coverage = MAX_VIDEO_MASK_COVERAGE_RATIO
    return {
        "min_coverage_ratio": min_coverage,
        "max_coverage_ratio": max_coverage,
        "min_temporal_stability": MIN_VIDEO_MASK_TEMPORAL_STABILITY,
        "min_nonempty_frame_ratio": MIN_VIDEO_MASK_NONEMPTY_FRAME_RATIO,
        "mask_not_empty_all_frames": True,
        "mask_target_matches_query": True,
    }


def _heuristic_stable_clip_selection(
    *,
    media: dict[str, Any],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> dict[str, Any]:
    duration = float(media.get("duration_seconds") or 0.0)
    clip_seconds = min(max_clip_seconds, duration)
    if clip_seconds < min_clip_seconds:
        clip_seconds = duration
    start = 0.0
    end = min(duration, start + clip_seconds)
    return {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "stability_score": 0.5,
        "camera_motion": "unknown",
        "main_subjects": [],
        "visible_text_risk": False,
        "recommended_for_vace": True,
        "reason": "heuristic first stable-length window; Omni selection was not available",
    }


def _coerce_stable_clip_selection(
    selection: dict[str, Any],
    *,
    fallback: dict[str, Any],
    media: dict[str, Any],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> dict[str, Any]:
    duration = float(media.get("duration_seconds") or 0.0)
    try:
        start = max(0.0, float(selection.get("start_sec", fallback.get("start_sec", 0.0))))
        end = max(start, float(selection.get("end_sec", fallback.get("end_sec", start))))
    except (TypeError, ValueError):
        start = float(fallback.get("start_sec", 0.0) or 0.0)
        end = float(fallback.get("end_sec", min(duration, start + max_clip_seconds)) or 0.0)
    if end > duration:
        end = duration
    window = end - start
    if window < min_clip_seconds or window > max_clip_seconds:
        fallback_start = float(fallback.get("start_sec", 0.0) or 0.0)
        fallback_end = float(fallback.get("end_sec", min(duration, fallback_start + max_clip_seconds)) or 0.0)
        start, end = fallback_start, fallback_end
    coerced = {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "stability_score": _score_float(selection.get("stability_score", fallback.get("stability_score", 0.5))),
        "camera_motion": str(selection.get("camera_motion", fallback.get("camera_motion", "unknown"))).strip() or "unknown",
        "main_subjects": _dedupe_strings(_normalize_list(selection.get("main_subjects", fallback.get("main_subjects", []))))[:6],
        "visible_text_risk": _boolish(selection.get("visible_text_risk", fallback.get("visible_text_risk", False))),
        "recommended_for_vace": _boolish(selection.get("recommended_for_vace", fallback.get("recommended_for_vace", True))),
        "reason": str(selection.get("reason", fallback.get("reason", ""))).strip(),
    }
    return coerced


def _stable_edit_targets_from_understanding(
    visual_understanding: dict[str, Any],
    annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for attribute in visual_understanding.get("editable_attributes", []):
        if not isinstance(attribute, dict):
            continue
        target = str(attribute.get("target", "")).strip()
        current = str(attribute.get("current", "")).strip()
        safe_targets = _normalize_list(attribute.get("safe_targets", []))
        if target and safe_targets:
            targets.append(
                {
                    "target": target,
                    "edit_family": "attribute_color",
                    "suggested_edit": f"change {target} from {current or 'its current appearance'} to {safe_targets[0]}",
                    "mask_query": target,
                    "needs_src_ref_images": False,
                    "src_ref_request": "",
                    "safe_targets": safe_targets,
                }
            )
    object_names = list(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    for source_name, replacement in VACE_EXPLORATION_OBJECT_REPLACEMENTS.items():
        if any(_text_mentions_phrase(name, source_name) for name in object_names):
            targets.append(
                {
                    "target": source_name,
                    "edit_family": "object_replacement",
                    "suggested_edit": f"replace the {source_name} with a {replacement}",
                    "mask_query": source_name,
                    "needs_src_ref_images": True,
                    "src_ref_request": f"a realistic {replacement}, isolated, plain background, no hands, no text",
                }
            )
            break
    return targets[:8]


def _annotation_is_usable_for_reference_understanding(annotation: dict[str, Any]) -> bool:
    if bool(annotation.get("fallback_used")):
        return False
    if str(annotation.get("detective_fallback_reason", "")).strip() == "detective_and_single_pass_failed":
        return False
    if str(annotation.get("fallback_reason", "")).strip() == "annotation_fallback":
        return False
    text_fields = [
        str(annotation.get("summary", "")).strip(),
        str(annotation.get("scene", "")).strip(),
    ]
    list_fields = (
        _normalize_list(annotation.get("subjects", []))
        + _normalize_list(annotation.get("actions", []))
        + _normalize_list(annotation.get("visible_text", []))
        + _normalize_list(annotation.get("on_screen_text", []))
        + _normalize_list(annotation.get("audio_events", []))
        + _normalize_list(annotation.get("speech", []))
        + _normalize_list(annotation.get("storyline", []))
        + _normalize_list(annotation.get("events", []))
    )
    if any(text_fields) or any(str(item).strip() for item in list_fields):
        return True
    object_counts = annotation.get("object_counts")
    return isinstance(object_counts, dict) and bool(object_counts)


def _src_ref_requirement_for_video_plan(plan: dict[str, Any]) -> dict[str, Any]:
    difference = plan.get("difference") if isinstance(plan.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    edit_text = _normalized_phrase(str(plan.get("edit_text", "")))
    edit_token = str(plan.get("edit_token", "")).strip()
    family = str(plan.get("exploration_family", "")).strip()
    target = str(difference.get("to", "")).strip() or edit_token
    from_value = str(difference.get("from", "")).strip()

    if family == "object_replacement" or ("replace" in edit_text and difference_type in {"object_presence", "object_count"}):
        return {
            "required": True,
            "recommended": True,
            "role": "replacement_object",
            "target": target,
            "source_object": from_value,
            "reason": "object replacement needs a visual reference image for the replacement object",
        }
    if family == "background_change" or difference_type == "scene" or "background" in edit_text:
        return {
            "required": True,
            "recommended": True,
            "role": "background_reference",
            "target": target or "target background",
            "source_object": from_value,
            "reason": "background replacement benefits from a clean background reference image",
        }
    if family == "clothing_type" or any(token in edit_text for token in ("shirt", "jacket", "dress", "outfit", "clothing")) and "replace" in edit_text:
        return {
            "required": True,
            "recommended": True,
            "role": "clothing_reference",
            "target": target or edit_token or "target clothing",
            "source_object": from_value,
            "reason": "clothing type replacement needs a reference image for the target clothing",
        }
    if family == "object_removal" or edit_text.startswith("remove "):
        return {
            "required": False,
            "recommended": False,
            "role": "none",
            "target": "",
            "source_object": from_value or edit_token,
            "reason": "object removal uses mask inpainting and does not need src_ref_images",
        }
    return {
        "required": False,
        "recommended": False,
        "role": "none",
        "target": "",
        "source_object": from_value,
        "reason": "attribute/color/material edits can run with video + mask + prompt",
    }


def _src_ref_image_prompts(*, requirement: dict[str, Any], edit_plan: dict[str, Any]) -> list[str]:
    target = str(requirement.get("target", "")).strip() or "target object"
    role = str(requirement.get("role", "")).strip()
    if role == "background_reference":
        return [
            f"a clean 16:9 wide reference image of {target}, empty scene plate, natural camera perspective, no people, no text, no watermark",
            f"{target}, wide empty background plate matching a talking-head video perspective, cinematic but realistic lighting, no foreground subject, no readable text",
        ]
    if role == "clothing_reference":
        return [
            f"a clean upper-body clothing reference photo of {target} on a neutral mannequin torso, front three-quarter view, no face, no text, no logo",
            f"{target}, wearable jacket or shirt reference for a standing performer, clear sleeves and torso shape, neutral background, no watermark",
        ]
    return [
        f"a realistic {target}, isolated product reference, three-quarter view, plain white background, no hands, no people, no text, no logo",
        f"{target}, clean object reference image with visible side and top shape, centered, neutral lighting, transparent or plain background, no watermark",
    ]


def _src_ref_image_negative_prompt(requirement: dict[str, Any]) -> str:
    role = str(requirement.get("role", "")).strip()
    base = "text, watermark, logo, blur, clutter, extra objects, distorted shape"
    if role == "replacement_object":
        return f"hands, people, scene background, {base}"
    if role == "background_reference":
        return f"people, foreground subject, readable signs, {base}"
    if role == "clothing_reference":
        return f"face, full person identity, body pose, readable brand logo, {base}"
    return base


def _find_src_ref_image_candidates(candidate_dir: Path) -> list[Path]:
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(path for path in candidate_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _audit_src_ref_image_candidate(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    role = str(plan.get("src_ref_role", "")).strip()
    score = 0.50
    reasons: list[str] = ["candidate file exists"]
    warnings: list[str] = []
    width = 0
    height = 0
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            width, height = image.size
        score += 0.15
        reasons.append(f"readable image {width}x{height}")
    except Exception:
        warnings.append("image dimensions unavailable for deterministic audit")

    if width > 0 and height > 0:
        aspect = width / max(height, 1)
        if role == "background_reference":
            if 1.45 <= aspect <= 1.95:
                score += 0.20
                reasons.append("background candidate is close to 16:9")
            else:
                warnings.append("background candidate is not close to 16:9")
        elif role == "clothing_reference":
            if 0.55 <= aspect <= 1.35:
                score += 0.15
                reasons.append("clothing candidate has plausible torso/reference aspect")
            else:
                warnings.append("clothing candidate aspect may be hard to fit to a person")
        elif role == "replacement_object":
            if 0.45 <= aspect <= 2.20:
                score += 0.10
                reasons.append("replacement object candidate has usable aspect")

    name_key = _normalized_phrase(path.name)
    if any(token in name_key for token in ("text", "logo", "watermark", "person", "hand", "face")):
        score -= 0.25
        warnings.append("filename suggests a forbidden visual artifact")
    return {
        "path": str(path),
        "score": round(max(0.0, min(1.0, score)), 3),
        "width": width,
        "height": height,
        "reasons": reasons,
        "warnings": warnings,
    }


def _video_edit_reference_understanding(annotation: dict[str, Any]) -> dict[str, Any]:
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))[:6]
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))[:6]
    visible_text = _dedupe_strings(
        _normalize_list(annotation.get("visible_text", []))
        + _normalize_list(annotation.get("on_screen_text", []))
    )[:6]
    object_names = list(_normalize_object_counts(annotation.get("object_counts", {})).keys())[:6]
    summary = str(annotation.get("summary", "")).strip()
    scene = str(annotation.get("scene", "")).strip()
    editable_attributes: list[dict[str, Any]] = []
    text = _normalized_phrase(" ".join([summary, scene, " ".join(subjects), " ".join(object_names)]))
    if any(marker in text for marker in ("robot", "robotic", "action figure")):
        editable_attributes.append(
            {
                "type": "color",
                "target": "robot body",
                "current": "black and gold",
                "safe_targets": ["bright yellow", "silver", "red"],
            }
        )
    if any(marker in text for marker in ("car", "vehicle", "truck", "bus")):
        editable_attributes.append(
            {
                "type": "color",
                "target": "vehicle body",
                "current": "original vehicle color",
                "safe_targets": ["bright red", "blue", "silver"],
            }
        )
    if any(marker in text for marker in ("shirt", "jacket", "coat", "dress", "clothing")):
        editable_attributes.append(
            {
                "type": "color",
                "target": "clothing",
                "current": "original clothing color",
                "safe_targets": ["bright blue", "red", "green"],
            }
        )
    return {
        "main_subjects": subjects or object_names,
        "stable_scene": scene or summary,
        "visible_text": visible_text,
        "actions": actions,
        "editable_attributes": editable_attributes,
        "bad_edits": [
            "add small background object",
            "add text",
            "add tiny accessory",
            "change exact object count",
        ],
    }


def _planned_route_matches_difference(route: str, difference_type: str) -> bool:
    expected_route = _video_edit_model_route(difference_type)
    return bool(expected_route and route == expected_route)


def _video_edit_token(difference: dict[str, Any], edit_text: str) -> str:
    for field_name in ("to", "description", "from"):
        value = str(difference.get(field_name, "")).strip()
        if value and not _absence_like_phrase(value):
            return value[:120]
    tokens = TOKEN_PATTERN.findall(edit_text.lower())
    if not tokens:
        return ""
    return " ".join(tokens[-5:])[:120]


def _absence_like_phrase(value: str) -> bool:
    normalized = _normalized_phrase(value)
    return bool(
        not normalized
        or normalized.startswith("no ")
        or normalized.startswith("none")
        or normalized in {"absent", "missing", "nothing", "no distinctive audio event"}
    )


def _video_edit_region(
    edit_text: str,
    difference: dict[str, Any],
    annotation: dict[str, Any],
    route: str,
) -> str:
    if route == "audio_deterministic":
        return "audio track"
    text = " ".join(
        str(value).strip()
        for value in (
            edit_text,
            difference.get("description", ""),
            difference.get("to", ""),
            annotation.get("summary", ""),
        )
        if str(value).strip()
    ).lower()
    region_patterns = (
        ("top-right", "top-right region"),
        ("top right", "top-right region"),
        ("top-left", "top-left region"),
        ("top left", "top-left region"),
        ("bottom-right", "bottom-right region"),
        ("bottom right", "bottom-right region"),
        ("bottom-left", "bottom-left region"),
        ("bottom left", "bottom-left region"),
        ("background", "background"),
        ("foreground", "foreground"),
        ("wall", "wall area"),
        ("paper", "paper surface"),
        ("desk", "desk surface"),
        ("table", "table surface"),
        ("robot body", "robot body"),
        ("robot", "robot body"),
        ("vehicle", "vehicle body"),
        ("car", "vehicle body"),
        ("clothing", "clothing"),
        ("shirt", "clothing"),
        ("jacket", "clothing"),
        ("visor", "visor"),
        ("floor", "floor area"),
        ("center", "center region"),
        ("left", "left side"),
        ("right", "right side"),
    )
    for marker, region in region_patterns:
        if marker in text:
            return region
    edit_token = _video_edit_token(difference, edit_text)
    if edit_token:
        return f"localized region around {edit_token}"
    return ""


def _video_edit_source_prompt(annotation: dict[str, Any], record: dict[str, Any]) -> str:
    summary = str(annotation.get("summary") or record.get("reference_caption", "")).strip()
    scene = str(annotation.get("scene", "")).strip()
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))[:4]
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))[:3]
    clauses = [summary or "the reference video"]
    if scene:
        clauses.append(f"scene: {scene}")
    if subjects:
        clauses.append("main subjects: " + ", ".join(subjects))
    if actions:
        clauses.append("actions: " + ", ".join(actions))
    return ". ".join(clauses).strip().rstrip(".") + "."


def _is_existing_object_replacement(difference: dict[str, Any], edit_text: str = "") -> bool:
    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if difference_type != "object_presence":
        return False
    if _absence_like_phrase(from_value) or _absence_like_phrase(to_value):
        return False
    return "replace" in _normalized_phrase(edit_text) or bool(from_value and to_value)


def _is_object_removal(difference: dict[str, Any], edit_text: str = "") -> bool:
    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    return bool(
        difference_type == "object_presence"
        and not _absence_like_phrase(from_value)
        and (_absence_like_phrase(to_value) or _normalized_phrase(edit_text).startswith("remove "))
    )


def _video_edit_source_object(difference: dict[str, Any], edit_text: str = "") -> str:
    from_value = str(difference.get("from", "")).strip()
    if from_value and not _absence_like_phrase(from_value):
        return from_value
    match = re.search(r"\breplace\s+(?:the\s+)?(.+?)\s+with\b", str(edit_text), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"\bremove\s+(?:the\s+)?(.+?)(?:\s+from|\s+in|\s*$)", str(edit_text), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _video_edit_target_object(difference: dict[str, Any], edit_text: str = "", edit_token: str = "") -> str:
    to_value = str(difference.get("to", "")).strip()
    if to_value and not _absence_like_phrase(to_value):
        return to_value
    match = re.search(r"\bwith\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|$)", str(edit_text), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return str(edit_token).strip()


def _video_edit_exclusion_keys(difference: dict[str, Any], *, edit_token: str = "", mask_query: str = "") -> set[str]:
    exclusions: set[str] = set()
    for value in (
        _video_edit_source_object(difference),
        _video_edit_target_object(difference, edit_token=edit_token),
        edit_token,
    ):
        key = _normalized_phrase(value)
        if key:
            exclusions.add(key)
    difference_type = str(difference.get("type", "")).strip()
    if difference_type != "scene":
        mask_key = _normalized_phrase(mask_query)
        if mask_key:
            exclusions.add(mask_key)
    if _is_existing_object_replacement(difference) or _is_object_removal(difference):
        from_value = _video_edit_source_object(difference)
        for token in TOKEN_PATTERN.findall(from_value.lower()):
            if token:
                exclusions.add(token)
    return exclusions


def _filter_video_edit_preserve_tokens(
    preserve_tokens: list[str],
    *,
    difference: dict[str, Any],
    edit_token: str,
    mask_query: str = "",
) -> list[str]:
    exclusions = _video_edit_exclusion_keys(difference, edit_token=edit_token, mask_query=mask_query)
    filtered: list[str] = []
    seen: set[str] = set()
    for raw_item in preserve_tokens:
        item = str(raw_item).strip()
        key = _normalized_phrase(item)
        if not item or not key or key in seen:
            continue
        if key in exclusions or any(excluded and excluded in key.split() for excluded in exclusions):
            continue
        filtered.append(item)
        seen.add(key)
    return filtered


def _video_edit_target_prompt(*, source_prompt: str, edit_text: str, difference: dict[str, Any]) -> str:
    difference_type = str(difference.get("type", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    from_value = _video_edit_source_object(difference, edit_text)
    if _is_existing_object_replacement(difference, edit_text):
        target = _video_edit_target_object(difference, edit_text)
        edit_clause = (
            f"Replace only the {from_value} with {target}. "
            f"The same shot shows {target} in the original {from_value} location; no {from_value} is visible."
        )
    elif _is_object_removal(difference, edit_text):
        edit_clause = (
            f"Remove only the {from_value}. "
            f"The {from_value} area is clean and naturally filled; no {from_value} is visible."
        )
    elif difference_type == "object_presence" and to_value:
        edit_clause = f"Add only {to_value}."
    elif difference_type == "object_count" and to_value:
        edit_clause = f"Change only the count to {to_value}."
    elif difference_type == "attribute":
        edit_clause = f"Change only the specified attribute: {edit_text}."
    elif difference_type == "scene":
        target = to_value or str(difference.get("description", "")).strip() or edit_text
        edit_clause = (
            f"The same subject, camera, action timing, and layout are preserved while the background becomes {target}."
        )
    elif difference_type == "action":
        edit_clause = f"Change only the action: {edit_text}."
    elif difference_type == "audio_event":
        edit_clause = f"Change only the audio event: {edit_text}."
    else:
        edit_clause = f"Apply only this edit: {edit_text}."
    return f"{source_prompt} {edit_clause}".strip()


def _video_edit_preserve_tokens(
    annotation: dict[str, Any],
    difference: dict[str, Any],
    edit_token: str,
) -> list[str]:
    values: list[str] = []
    values.extend(_normalize_list(annotation.get("subjects", [])))
    values.extend(list(_normalize_object_counts(annotation.get("object_counts", {})).keys()))
    values.extend(_normalize_list(annotation.get("actions", [])))
    scene = str(annotation.get("scene", "")).strip()
    if scene:
        values.append(scene)
    values.extend(["camera motion", "lighting", "timing"])
    preserved = _filter_video_edit_preserve_tokens(
        _dedupe_strings([str(value).strip() for value in values if str(value).strip()]),
        difference=difference,
        edit_token=edit_token,
    )
    return preserved[:8]


def _video_edit_risk_assessment(annotation: dict[str, Any], *, difference_type: str) -> dict[str, Any]:
    visible_text = _dedupe_strings(
        _normalize_list(annotation.get("visible_text", []))
        + _normalize_list(annotation.get("on_screen_text", []))
    )
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))
    storyline = _dedupe_strings(_normalize_list(annotation.get("storyline", [])))
    events = annotation.get("events", [])
    event_count = len(events) if isinstance(events, list) else 0
    summary_tokens = _tokenize_text(str(annotation.get("summary", "")))
    scene_text = _normalized_phrase(str(annotation.get("scene", "")))
    risk_reasons: list[str] = []
    if visible_text:
        risk_reasons.append("visible_text_present")
    if difference_type != "action" and len(actions) >= 2:
        risk_reasons.append("multiple_actions")
    if difference_type != "action" and event_count >= 2:
        risk_reasons.append("multi_event_timeline")
    if len(subjects) >= 4:
        risk_reasons.append("many_subjects")
    if any(token in summary_tokens for token in {"speaks", "speaking", "talks", "talking", "vlogging", "interview"}):
        risk_reasons.append("speaking_person")
    if any(token in summary_tokens for token in {"transition", "transitions", "followed", "split", "screen", "cut"}):
        risk_reasons.append("scene_or_shot_change")
    if any(token in scene_text for token in ("ui", "screen", "interface", "control room")):
        risk_reasons.append("ui_or_text_heavy_scene")
    if storyline and len(storyline) >= 3 and difference_type != "action":
        risk_reasons.append("long_storyline")

    score = min(1.0, 0.18 * len(risk_reasons))
    allow_generation = not any(
        reason in set(risk_reasons)
        for reason in {
            "visible_text_present",
            "multiple_actions",
            "multi_event_timeline",
            "scene_or_shot_change",
            "ui_or_text_heavy_scene",
        }
    )
    risk_level = "low"
    if score >= 0.55 or not allow_generation:
        risk_level = "high"
    elif score >= 0.25:
        risk_level = "medium"
    locks = ["preserve camera motion, lighting, timing, and layout exactly"]
    if visible_text:
        locks.append("preserve all visible text exactly; do not alter letters, captions, labels, signs, subtitles, or UI text")
    if actions and difference_type != "action":
        locks.append("preserve the exact action and motion timing; do not change gestures, pose, order, or movement")
    if subjects:
        locks.append("preserve all existing people, subjects, and object identities")
    return {
        "score": round(score, 3),
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "allow_generation": allow_generation,
        "locks": locks,
    }


def _merge_video_edit_locks(negative_prompt: str, risk: dict[str, Any] | None = None) -> str:
    prompt = str(negative_prompt).strip()
    locks = [
        str(item).strip()
        for item in (risk or {}).get("locks", [])
        if str(item).strip()
    ]
    for lock in locks:
        if lock.lower() not in prompt.lower():
            prompt = f"{prompt} {lock}." if prompt else f"{lock}."
    return prompt.strip()


def _video_edit_negative_prompt(preserve_tokens: list[str], *, risk: dict[str, Any] | None = None) -> str:
    protected = ", ".join(preserve_tokens[:6]) if preserve_tokens else "the original subject, scene, camera, timing"
    prompt = (
        f"Do not change {protected}. Do not add extra people, change the scene, alter visible text, "
        "reorder shots, or introduce additional edits."
    )
    return _merge_video_edit_locks(prompt, risk)


def _target_prompt_contract_mentions_absence(prompt: str, source_object: str) -> bool:
    source = _normalized_phrase(source_object)
    text = _normalized_phrase(prompt)
    return bool(source and (f"no {source}" in text or f"without {source}" in text))


def _repair_video_edit_prompt_contract(
    *,
    source_prompt: str,
    target_prompt: str,
    edit_text: str,
    difference: dict[str, Any],
    edit_token: str,
    preserve_tokens: list[str],
    negative_prompt: str,
    mask_query: str,
    risk: dict[str, Any] | None,
) -> tuple[str, list[str], str, list[str]]:
    repairs: list[str] = []
    source_object = _video_edit_source_object(difference, edit_text)
    normalized_target = _normalized_phrase(target_prompt)
    if _is_existing_object_replacement(difference, edit_text):
        if "add only" in normalized_target or "replace" not in normalized_target:
            target_prompt = _video_edit_target_prompt(source_prompt=source_prompt, edit_text=edit_text, difference=difference)
            repairs.append("target_prompt_rewritten_for_object_replacement")
        elif not _target_prompt_contract_mentions_absence(target_prompt, source_object):
            target_prompt = f"{target_prompt.rstrip('.')} No {source_object} is visible."
            repairs.append("target_prompt_added_source_absence")
    elif _is_object_removal(difference, edit_text):
        if "add only no" in normalized_target or (
            "remove" not in normalized_target and not _target_prompt_contract_mentions_absence(target_prompt, source_object)
        ):
            target_prompt = _video_edit_target_prompt(source_prompt=source_prompt, edit_text=edit_text, difference=difference)
            repairs.append("target_prompt_rewritten_for_object_removal")
        elif not _target_prompt_contract_mentions_absence(target_prompt, source_object):
            target_prompt = f"{target_prompt.rstrip('.')} No {source_object} is visible."
            repairs.append("target_prompt_added_source_absence")

    filtered_preserve = _filter_video_edit_preserve_tokens(
        preserve_tokens,
        difference=difference,
        edit_token=edit_token,
        mask_query=mask_query,
    )
    if filtered_preserve != preserve_tokens:
        preserve_tokens = filtered_preserve
        repairs.append("preserve_tokens_removed_edit_source")

    source_key = _normalized_phrase(source_object)
    negative_key = _normalized_phrase(negative_prompt)
    if not negative_prompt or (source_key and source_key in negative_key):
        negative_prompt = _video_edit_negative_prompt(preserve_tokens, risk=risk)
        repairs.append("negative_prompt_regenerated_without_edit_source")
    return target_prompt, preserve_tokens, negative_prompt, repairs


def _annotation_mentions_object(annotation: dict[str, Any], object_name: str) -> bool:
    object_key = _normalized_phrase(object_name)
    if not object_key:
        return False
    counts = _normalize_object_counts(annotation.get("object_counts", {}))
    if any(object_key == _normalized_phrase(name) for name in counts):
        return True
    text = _normalized_phrase(
        " ".join(
            [
                str(annotation.get("summary", "")),
                str(annotation.get("scene", "")),
                " ".join(_normalize_list(annotation.get("subjects", []))),
                " ".join(_normalize_list(annotation.get("actions", []))),
            ]
        )
    )
    return object_key in text


def _reference_has_screen_text_risk(annotation: dict[str, Any], source_object: str) -> bool:
    source_tokens = set(TOKEN_PATTERN.findall(_normalized_phrase(source_object)))
    if not source_tokens & VACE_SCREEN_TEXT_OBJECTS:
        return False
    visible_text = _normalize_list(annotation.get("visible_text", [])) + _normalize_list(annotation.get("on_screen_text", []))
    text = _normalized_phrase(
        " ".join(
            [str(annotation.get("summary", "")), str(annotation.get("scene", "")), " ".join(visible_text)]
        )
    )
    risky_markers = {"webpage", "website", "screen", "browser", "ui", "interface", "text", "logo", "caption"}
    return bool(visible_text or source_tokens & {"laptop", "computer", "screen", "monitor"} and any(marker in text for marker in risky_markers))


def _reference_has_seated_support_conflict(annotation: dict[str, Any], source_object: str) -> bool:
    source_tokens = set(TOKEN_PATTERN.findall(_normalized_phrase(source_object)))
    if not source_tokens & VACE_SEATED_SUPPORT_OBJECTS:
        return False
    text = _normalized_phrase(
        " ".join(
            [
                str(annotation.get("summary", "")),
                " ".join(_normalize_list(annotation.get("actions", []))),
                " ".join(_normalize_list(annotation.get("subjects", []))),
            ]
        )
    )
    return any(marker in text for marker in ("sit", "sits", "sitting", "seated", "seat", "sits in", "sits on"))


def _video_edit_plan_lint(
    *,
    target_prompt: str,
    edit_text: str,
    difference: dict[str, Any],
    preserve_tokens: list[str],
    negative_prompt: str,
    reference_annotation: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    target_key = _normalized_phrase(target_prompt)
    source_object = _video_edit_source_object(difference, edit_text)
    source_key = _normalized_phrase(source_object)
    preserve_keys = {_normalized_phrase(item) for item in preserve_tokens}
    negative_key = _normalized_phrase(negative_prompt)

    if "add only no" in target_key:
        errors.append("target_prompt_contains_add_only_no")
    if _is_existing_object_replacement(difference, edit_text) and "add only" in target_key:
        errors.append("replacement_target_prompt_uses_add_instead_of_replace")
    if source_key and (source_key in preserve_keys or any(source_key == key for key in preserve_keys)):
        errors.append("preserve_tokens_lock_edit_source")
    if source_key and source_key in negative_key:
        errors.append("negative_prompt_locks_edit_source")

    if (_is_existing_object_replacement(difference, edit_text) or _is_object_removal(difference, edit_text)) and source_object:
        if not _annotation_mentions_object(reference_annotation, source_object):
            warnings.append("edit_source_not_clearly_present_in_annotation")
    if _is_existing_object_replacement(difference, edit_text) and _reference_has_screen_text_risk(reference_annotation, source_object):
        errors.append("object_replacement_screen_or_visible_text_risk")
    if _is_object_removal(difference, edit_text) and _reference_has_seated_support_conflict(reference_annotation, source_object):
        errors.append("object_removal_breaks_seated_support")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _video_edit_control_plan(route: str) -> list[str]:
    if route == "vace_controlled":
        return ["first_frame_reference", "local_roi_mask", "depth_or_lineart_control"]
    if route == "tokenflow_style":
        return ["first_frame_reference", "tokenflow_consistency", "local_roi_mask"]
    if route == "ltx2_retake":
        return ["first_frame_reference", "retake_reference", "motion_consistency_check"]
    return []


def _video_edit_generation_defaults(route: str) -> dict[str, Any]:
    return {
        "gpu_ids": "0,1",
        "offload_model": False,
        "frame_count": 49,
        "steps": 25,
        "resolution": "832x480",
        "postprocess": {"audio_copied_from_reference": True},
    }


def _audio_expected_event(difference: dict[str, Any], edit_text: str) -> str:
    for field_name in ("to", "description", "from"):
        value = str(difference.get(field_name, "")).strip()
        if value and not _absence_like_phrase(value) and not _is_speech_only_audio_phrase(value):
            return value[:120]
    tokens = [token for token in TOKEN_PATTERN.findall(edit_text.lower()) if token in NON_SPEECH_AUDIO_TOKENS]
    return " ".join(tokens[:4])[:120]


def _synthetic_audio_expected_event(record: dict[str, Any]) -> str:
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    audio_plan = generation.get("audio_edit_plan", {}) if isinstance(generation.get("audio_edit_plan"), dict) else {}
    expected_event = str(audio_plan.get("expected_event", "")).strip()
    if expected_event:
        return expected_event
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    return _audio_expected_event(difference, str(record.get("edit_text", "")))


def _audio_terms_mention_event(terms: list[str], expected_event: str) -> bool:
    expected_tokens = _tokenize_text(expected_event) - {"audio", "event", "sound", "sounds", "noise", "no"}
    if not expected_tokens:
        return False
    for term in terms:
        if _text_mentions_phrase(term, expected_event):
            return True
        term_tokens = _tokenize_text(term)
        if expected_tokens.issubset(term_tokens):
            return True
        if _jaccard(expected_tokens, term_tokens) >= 0.5:
            return True
    return False


def _audio_edit_route(expected_event: str, annotation: dict[str, Any]) -> str:
    event = _normalized_phrase(expected_event)
    if any(token in event for token in ("footstep", "walking", "scratch", "writing", "whoosh", "splash")):
        return "foleycrafter_temporal"
    return "deterministic_overlay"


def _safe_audio_ideation_candidate(candidate: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any] | None:
    suggestion = _audio_edit_suggestion(annotation)
    if suggestion is None:
        return None
    source_edit_text = str(candidate.get("edit_text", "")).strip()
    event = str(suggestion["expected_event"]).strip()
    edit_text = str(suggestion["edit_text"]).strip()
    proposal_seed = str(candidate.get("proposal_id", "")) or str(candidate.get("reference_video", "")) + edit_text
    revised = dict(candidate)
    revised["proposal_id"] = f"{str(candidate.get('proposal_id', '')).strip() or 'candidate'}__audio_ideation_{_stable_hash(proposal_seed)[:8]}"
    revised["edit_text"] = edit_text
    revised["difference"] = {
        "type": "audio_event",
        "from": f"no {event}",
        "to": event,
        "description": str(suggestion["description"]).strip(),
    }
    revised["source_candidate_edit_text"] = source_edit_text
    revised["source_candidate_difference"] = candidate.get("difference", {})
    revised["candidate_source"] = "safe_audio_ideation_from_reference"
    revised["ideation_reason"] = str(suggestion["reason"]).strip()
    return revised


def _audio_edit_suggestion(annotation: dict[str, Any]) -> dict[str, str] | None:
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))
    scene = str(annotation.get("scene", "")).strip()
    summary = str(annotation.get("summary", "")).strip()
    text = _normalized_phrase(" ".join([summary, scene, " ".join(actions), " ".join(subjects)]))
    suggestions = (
        (
            ("writing", "write", "pen", "pencil"),
            "scratching sound",
            "add a scratching sound synchronized with the writing",
            "A pen scratching sound is synchronized with the visible writing motion.",
            "visible writing motion can support a synchronized Foley sound",
        ),
        (
            ("jumping", "jump", "launched", "launch", "flying", "gliding"),
            "whoosh",
            "add a whoosh sound synchronized with the jump or launch",
            "A short whoosh sound is synchronized with the visible jump or launch.",
            "visible jump or launch motion can support a synchronized Foley sound",
        ),
        (
            ("walking", "walk", "running", "run", "foot", "steps"),
            "footsteps",
            "add footsteps synchronized with the walking or running",
            "Footsteps are synchronized with the visible walking or running.",
            "visible walking or running can support synchronized footsteps",
        ),
        (
            ("clapping", "clap", "applaud", "applause"),
            "applause",
            "add applause to the audio",
            "Applause is added to match the visible clapping or audience context.",
            "visible clapping or audience context can support applause",
        ),
        (
            ("water", "river", "ocean", "waves", "splash"),
            "water splash",
            "add a water splash sound",
            "A water splash sound is added to match the visible water context.",
            "visible water context can support a water Foley sound",
        ),
        (
            ("forest", "trees", "outdoor", "wind"),
            "wind ambience",
            "add soft wind ambience to the audio",
            "Wind ambience is added while preserving the video stream.",
            "outdoor or forest context can support ambient wind",
        ),
    )
    for markers, expected_event, edit_text, description, reason in suggestions:
        if any(marker in text for marker in markers):
            return {
                "expected_event": expected_event,
                "edit_text": edit_text,
                "description": description,
                "reason": reason,
            }
    return None


def _audio_edit_reference_understanding(annotation: dict[str, Any]) -> dict[str, Any]:
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))[:6]
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))[:6]
    audio_events = _dedupe_strings(_non_speech_audio_terms(annotation))[:6]
    visible_text = _dedupe_strings(
        _normalize_list(annotation.get("visible_text", []))
        + _normalize_list(annotation.get("on_screen_text", []))
    )[:6]
    suggestion = _audio_edit_suggestion(annotation)
    suggested_events: list[dict[str, Any]] = []
    if suggestion is not None:
        suggested_events.append(
            {
                "expected_event": suggestion["expected_event"],
                "edit_text": suggestion["edit_text"],
                "reason": suggestion["reason"],
                "route": _audio_edit_route(suggestion["expected_event"], annotation),
                "timing_strategy": _audio_timing_strategy(suggestion["expected_event"], annotation),
            }
        )
    return {
        "main_subjects": subjects,
        "visible_actions": actions,
        "existing_non_speech_audio_events": audio_events,
        "visible_text": visible_text,
        "scene": str(annotation.get("scene", "")).strip(),
        "suggested_non_speech_audio_events": suggested_events,
        "bad_audio_edits": [
            "speech topic change",
            "transcript change",
            "narration-only change",
            "voiceover-only change",
            "unrelated music that conflicts with visible context",
        ],
    }


def _audio_edit_route_suitability(
    *,
    expected_event: str,
    difference: dict[str, Any],
    edit_text: str,
    reference_annotation: dict[str, Any],
) -> dict[str, Any]:
    issues = _speech_content_edit_issues(edit_text=edit_text, difference=difference)
    if issues:
        return {
            "allow_generation": False,
            "reason": "speech_content_or_speech_only_audio",
            "issues": issues,
        }
    if _audio_terms_mention_event(_non_speech_audio_terms(reference_annotation), expected_event):
        return {
            "allow_generation": False,
            "reason": "reference_already_has_expected_audio_event",
        }
    route = _audio_edit_route(expected_event, reference_annotation)
    timing = _audio_timing_strategy(expected_event, reference_annotation)
    priority = "S" if route == "foleycrafter_temporal" and timing == "visual_sync" else "A"
    return {
        "allow_generation": True,
        "reason": "contextual_non_speech_audio_edit",
        "route": route,
        "timing_strategy": timing,
        "priority": priority,
    }


def _audio_edit_prompt(expected_event: str, annotation: dict[str, Any], edit_text: str) -> str:
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))[:3]
    if actions:
        return f"{expected_event}, synchronized with {', '.join(actions)}"
    return f"{expected_event}. {edit_text}".strip()


def _audio_timing_strategy(expected_event: str, annotation: dict[str, Any]) -> str:
    event = _normalized_phrase(expected_event)
    if any(token in event for token in ("ambient", "wind", "rain", "waves", "hum", "music")):
        return "whole_clip_ambience"
    if annotation.get("events") or annotation.get("actions"):
        return "visual_sync"
    return "fixed_timestamp"


def _annotation_for_known_pair(
    *,
    root: Path,
    lookup: dict[str, dict[str, Any]],
    pair: dict[str, Any],
    clip_id_field: str,
    video_field: str,
    line_number: int,
) -> dict[str, Any]:
    clip_id = str(pair.get(clip_id_field, "")).strip()
    if clip_id and clip_id in lookup:
        return lookup[clip_id]

    video_path = str(pair.get(video_field, "")).strip()
    if video_path:
        resolved = _resolve_under_root(root, video_path)
        for key in _path_lookup_keys(root, resolved, video_path):
            if key in lookup:
                return lookup[key]
    raise ValueError(f"known pair line {line_number}: cannot resolve {clip_id_field} or {video_field}")


def _known_pair_video_path(
    root: Path,
    pair: dict[str, Any],
    annotation: dict[str, Any],
    field_name: str,
) -> str:
    raw_value = str(pair.get(field_name, "")).strip()
    if raw_value:
        return _display_path(root, _resolve_under_root(root, raw_value))
    return _display_path(root, _resolve_under_root(root, str(annotation.get("output_path", ""))))


def _known_pair_model_fields(
    *,
    pair: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    difference = dict(pair.get("difference") or {})
    if not difference:
        difference = _detect_primary_difference(reference_annotation, target_annotation) or {}
        difference.pop("changed_types", None)
    if not difference:
        difference = {
            "type": "attribute",
            "from": "",
            "to": "",
            "description": str(pair.get("edit_text", "")).strip(),
        }
    difference_type = str(difference.get("type", "")).strip()
    edit_text = str(pair.get("edit_text", "")).strip() or _build_fallback_edit_text(difference)
    modalities = pair.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        modalities = _infer_pair_modalities(reference_annotation, target_annotation, difference_type)
    return {
        "edit_text": edit_text,
        "modalities": [str(item).strip() for item in modalities if str(item).strip()],
        "reference_caption": str(pair.get("reference_caption") or reference_annotation.get("summary", "")).strip(),
        "target_caption": str(pair.get("target_caption") or target_annotation.get("summary", "")).strip(),
        "difference": difference,
        "proposal_reason": str(pair.get("proposal_reason", "known pair validation")).strip(),
    }


def _synthetic_pair_source_matches_reference(pair: dict[str, Any]) -> bool:
    if str(pair.get("source_type", "synthetic_edit")).strip() != "synthetic_edit":
        return False
    generation = pair.get("generation")
    if not isinstance(generation, dict):
        return False
    source_video = _normalized_path_text(generation.get("source_video", ""))
    reference_video = _normalized_path_text(pair.get("reference_video", ""))
    if not source_video or not reference_video:
        return False
    return bool(
        source_video == reference_video
        or source_video.endswith("/" + reference_video)
        or reference_video.endswith("/" + source_video)
    )


def _normalized_path_text(value: Any) -> str:
    return str(value).replace("\\", "/").strip().lstrip("./")


def _known_pair_source_context(pair: dict[str, Any]) -> dict[str, Any]:
    source_context = pair.get("source_context")
    if isinstance(source_context, dict) and source_context:
        normalized = dict(source_context)
        normalized.setdefault("relation", "known_pair")
        normalized.setdefault("score", 0.9)
        return normalized
    if _synthetic_pair_source_matches_reference(pair):
        return {
            "relation": "synthetic_from_reference",
            "score": 0.95,
            "generation_source_video": str(pair.get("generation", {}).get("source_video", "")).strip(),
        }
    return {"relation": "synthetic_edit", "score": 0.9}


def _known_pair_hard_negative_annotations(
    *,
    root: Path,
    lookup: dict[str, dict[str, Any]],
    annotations: list[dict[str, Any]],
    pair: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw_value in pair.get("hard_negatives", []) if isinstance(pair.get("hard_negatives"), list) else []:
        raw_path = str(raw_value).strip()
        if not raw_path:
            continue
        resolved = _resolve_under_root(root, raw_path)
        for key in _path_lookup_keys(root, resolved, raw_path):
            if key in lookup and lookup[key].get("clip_id") not in {
                reference_annotation.get("clip_id"),
                target_annotation.get("clip_id"),
            }:
                selected.append(lookup[key])
                break
    if selected:
        unique: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for annotation in selected:
            clip_id = str(annotation.get("clip_id", "")).strip()
            if clip_id and clip_id not in seen_ids:
                unique.append(annotation)
                seen_ids.add(clip_id)
        return unique[:3]
    return _select_hard_negative_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        primary_difference=difference,
    )


def _known_pair_hard_negative_paths(
    *,
    root: Path,
    pair: dict[str, Any],
    hard_negative_annotations: list[dict[str, Any]],
) -> list[str]:
    raw_values = pair.get("hard_negatives", [])
    if isinstance(raw_values, list) and raw_values:
        return [_display_path(root, _resolve_under_root(root, str(item).strip())) for item in raw_values if str(item).strip()]
    return [
        _display_path(root, _resolve_under_root(root, str(annotation.get("output_path", ""))))
        for annotation in hard_negative_annotations
        if str(annotation.get("output_path", "")).strip()
    ][:3]


def _known_pair_base_quality(
    *,
    root: Path,
    pair: dict[str, Any],
    annotations: list[dict[str, Any]],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
    source_context: dict[str, Any],
) -> dict[str, Any]:
    provided = pair.get("quality") if isinstance(pair.get("quality"), dict) else {}
    semantic_context_score = _same_context_score(reference_annotation, target_annotation)
    same_context_score = _pair_context_score(
        semantic_context_score=semantic_context_score,
        source_context=source_context,
    )
    synthetic_context_override = str(source_context.get("relation", "")).strip() == "synthetic_from_reference"
    if synthetic_context_override:
        same_context_score = max(same_context_score, _score_float(source_context.get("score")))
    detected_difference = _detect_primary_difference(reference_annotation, target_annotation)
    changed_types = list(detected_difference.get("changed_types", [])) if detected_difference else [str(difference.get("type", "")).strip()]
    quality: dict[str, Any] = {
        "same_context_score": _score_float(provided.get("same_context_score", same_context_score)),
        "edit_match_score": _score_float(provided.get("edit_match_score", 0.75)),
        "target_uniqueness_score": _score_float(
            provided.get(
                "target_uniqueness_score",
                _target_uniqueness_score(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    annotations=annotations,
                    primary_difference=difference,
                ),
            )
        ),
        "difference_strength_score": _score_float(
            provided.get(
                "difference_strength_score",
                _difference_strength_score(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    primary_difference=difference,
                    changed_types=changed_types,
                ),
            )
        ),
    }
    visual_score = provided.get("visual_near_duplicate_score")
    if visual_score is None:
        visual_score = _visual_near_duplicate_score(
            _resolve_under_root(root, str(reference_annotation.get("output_path", ""))),
            _resolve_under_root(root, str(target_annotation.get("output_path", ""))),
        )
    if visual_score is not None:
        quality["visual_near_duplicate_score"] = _score_float(visual_score)
    if synthetic_context_override:
        quality["synthetic_context_override"] = 1.0
    return quality


def _synthetic_generation_route(generation: dict[str, Any]) -> str:
    route = str(generation.get("model_route", "")).strip()
    if route:
        return route
    audio_plan = generation.get("audio_edit_plan", {})
    if isinstance(audio_plan, dict):
        return str(audio_plan.get("route", "")).strip()
    return ""


def _is_audio_synthetic_route(route: str) -> bool:
    return route in SYNTHETIC_AUDIO_ROUTES


def _known_pair_generation_issues(record: dict[str, Any]) -> list[str]:
    source_type = str(record.get("source_type", "")).strip() or "natural"
    if source_type not in ALLOWED_SOURCE_TYPES:
        return [f"unsupported source_type: {source_type}"]
    if source_type != "synthetic_edit":
        return []
    generation = record.get("generation")
    if not isinstance(generation, dict) or not generation:
        return ["synthetic_edit pair is missing generation metadata"]
    issues: list[str] = []
    route = _synthetic_generation_route(generation)
    for field_name in ("model", "source_video", "model_route"):
        if not str(generation.get(field_name, "")).strip():
            issues.append(f"generation.{field_name} is required for synthetic_edit pairs")
    if _is_audio_synthetic_route(route):
        audio_plan = generation.get("audio_edit_plan")
        if not isinstance(audio_plan, dict) or not audio_plan:
            issues.append("generation.audio_edit_plan is required for synthetic audio pairs")
        else:
            for field_name in ("audio_prompt", "expected_event"):
                if not str(audio_plan.get(field_name, "")).strip():
                    issues.append(f"generation.audio_edit_plan.{field_name} is required for synthetic audio pairs")
            if not _boolish(audio_plan.get("preserve_video")):
                issues.append("generation.audio_edit_plan.preserve_video=true is required for synthetic audio pairs")
        return issues

    for field_name in ("prompt", "source_prompt", "target_prompt"):
        if not str(generation.get(field_name, "")).strip():
            issues.append(f"generation.{field_name} is required for synthetic visual pairs")
    preserve_tokens = generation.get("preserve_tokens")
    if not isinstance(preserve_tokens, list) or not [item for item in preserve_tokens if str(item).strip()]:
        issues.append("generation.preserve_tokens is required for synthetic visual pairs")
    postprocess = generation.get("postprocess")
    if not isinstance(postprocess, dict) or "audio_copied_from_reference" not in postprocess:
        issues.append("generation.postprocess.audio_copied_from_reference is required for synthetic visual pairs")
    return issues


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"jsonl file not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_number}: expected a JSON object")
        records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_under_root(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _as_non_negative_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _parse_sources(raw_sources: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for raw in raw_sources:
        if "=" not in raw:
            raise ValueError(f"source must use dataset=/path form: {raw}")
        dataset_name, raw_path = raw.split("=", 1)
        dataset_name = dataset_name.strip()
        raw_path = raw_path.strip()
        if not dataset_name or not raw_path:
            raise ValueError(f"source must use dataset=/path form: {raw}")
        parsed.append((dataset_name, Path(raw_path)))
    return parsed


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        candidate = str(item).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _normalize_object_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = str(raw_key).strip()
        if not key:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        normalized[key] = count
    return normalized


def _first_unique(candidates: list[str], excluded: list[str]) -> str:
    excluded_lower = {item.lower() for item in excluded}
    for candidate in candidates:
        if candidate.lower() not in excluded_lower:
            return candidate
    return ""


def _first_item(values: list[str]) -> str:
    return values[0] if values else ""


def _tokenize_values(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(_tokenize_text(value))
    return tokens


def _tokenize_text(value: str) -> set[str]:
    tokens = set()
    for match in TOKEN_PATTERN.finditer(value.lower()):
        token = match.group(0)
        if token in STOPWORDS or len(token) <= 1:
            continue
        tokens.add(token)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Composed Omni Retrieval data helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_layout = subparsers.add_parser("init-layout")
    init_layout.add_argument("--root", default=DEFAULT_DATA_ROOT)

    index_raw = subparsers.add_parser("index-raw")
    index_raw.add_argument("--root", default=DEFAULT_DATA_ROOT)
    index_raw.add_argument(
        "--source",
        action="append",
        default=[],
        help="dataset=/absolute/path. If omitted, discover immediate children under <root>/raw_datasets.",
    )
    index_raw.add_argument("--output-path")

    extract_clips_parser = subparsers.add_parser("extract-clips")
    extract_clips_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    extract_clips_parser.add_argument("--plan-path", required=True)
    extract_clips_parser.add_argument("--raw-index-path")
    extract_clips_parser.add_argument("--output-manifest-path")
    extract_clips_parser.add_argument("--dry-run", action="store_true")
    extract_clips_parser.add_argument("--overwrite", action="store_true")

    plan_detective_parser = subparsers.add_parser("plan-detective-clips")
    plan_detective_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_detective_parser.add_argument("--source-clips-path", required=True)
    plan_detective_parser.add_argument("--clip-plan-output-path")
    plan_detective_parser.add_argument("--clip-groups-output-path")
    plan_detective_parser.add_argument("--max-source-videos", type=int, default=100)
    plan_detective_parser.add_argument("--segment-seconds", type=float, default=8.0)
    plan_detective_parser.add_argument("--min-clip-seconds", type=float, default=3.0)
    plan_detective_parser.add_argument("--max-clip-seconds", type=float, default=15.0)

    stable_clips_parser = subparsers.add_parser("plan-stable-omni-clips")
    stable_clips_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    stable_clips_parser.add_argument("--raw-index-path")
    stable_clips_parser.add_argument("--output-path")
    stable_clips_parser.add_argument("--cache-path")
    stable_clips_parser.add_argument("--max-source-videos", type=int, default=50)
    stable_clips_parser.add_argument("--min-clip-seconds", type=float, default=5.0)
    stable_clips_parser.add_argument("--max-clip-seconds", type=float, default=8.0)
    stable_clips_parser.add_argument("--base-url")
    stable_clips_parser.add_argument("--api-key", default="EMPTY")
    stable_clips_parser.add_argument("--model")
    stable_clips_parser.add_argument("--timeout-seconds", type=float, default=180.0)

    reference_understanding_parser = subparsers.add_parser("cache-reference-understandings")
    reference_understanding_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    reference_understanding_parser.add_argument("--clip-annotations-path", required=True)
    reference_understanding_parser.add_argument("--output-path")

    annotate_clips_parser = subparsers.add_parser("annotate-clips")
    annotate_clips_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    annotate_clips_parser.add_argument("--clips-manifest-path", required=True)
    annotate_clips_parser.add_argument("--output-path")
    annotate_clips_parser.add_argument("--base-url", required=True)
    annotate_clips_parser.add_argument("--api-key", required=True)
    annotate_clips_parser.add_argument("--model", required=True)
    annotate_clips_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    annotate_clips_parser.add_argument("--concurrency", type=int, default=1)
    annotate_clips_parser.add_argument("--overwrite", action="store_true")

    detective_annotate_parser = subparsers.add_parser("detective-annotate-clips")
    detective_annotate_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    detective_annotate_parser.add_argument("--clips-manifest-path", required=True)
    detective_annotate_parser.add_argument("--output-path")
    detective_annotate_parser.add_argument("--base-url", required=True)
    detective_annotate_parser.add_argument("--api-key", required=True)
    detective_annotate_parser.add_argument("--model", required=True)
    detective_annotate_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    detective_annotate_parser.add_argument("--concurrency", type=int, default=1)
    detective_annotate_parser.add_argument("--overwrite", action="store_true")

    propose_pairs_parser = subparsers.add_parser("propose-pairs")
    propose_pairs_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    propose_pairs_parser.add_argument("--clip-annotations-path", required=True)
    propose_pairs_parser.add_argument("--output-path")
    propose_pairs_parser.add_argument("--raw-index-path")
    propose_pairs_parser.add_argument("--base-url", required=True)
    propose_pairs_parser.add_argument("--api-key", required=True)
    propose_pairs_parser.add_argument("--model", required=True)
    propose_pairs_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    propose_pairs_parser.add_argument("--overwrite", action="store_true")

    propose_group_pairs_parser = subparsers.add_parser("propose-group-pairs")
    propose_group_pairs_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    propose_group_pairs_parser.add_argument("--clip-annotations-path", required=True)
    propose_group_pairs_parser.add_argument("--clip-groups-path", required=True)
    propose_group_pairs_parser.add_argument("--output-path")
    propose_group_pairs_parser.add_argument("--accepted-output-path")
    propose_group_pairs_parser.add_argument("--raw-index-path")
    propose_group_pairs_parser.add_argument("--base-url", required=True)
    propose_group_pairs_parser.add_argument("--api-key", required=True)
    propose_group_pairs_parser.add_argument("--model", required=True)
    propose_group_pairs_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    propose_group_pairs_parser.add_argument("--max-accepted-pairs", type=int, default=10)
    propose_group_pairs_parser.add_argument("--overwrite", action="store_true")

    plan_video_edits_parser = subparsers.add_parser("plan-video-edits")
    plan_video_edits_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_video_edits_parser.add_argument("--pair-candidates-path", required=True)
    plan_video_edits_parser.add_argument("--clip-annotations-path", required=True)
    plan_video_edits_parser.add_argument("--output-path")
    plan_video_edits_parser.add_argument("--max-plans", type=int, default=10)
    plan_video_edits_parser.add_argument("--base-url")
    plan_video_edits_parser.add_argument("--api-key", default="EMPTY")
    plan_video_edits_parser.add_argument("--model")
    plan_video_edits_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    plan_video_edits_parser.add_argument("--planning-mode", choices=("production", "exploration"), default="production")
    plan_video_edits_parser.add_argument("--planner-cache-path")

    plan_audio_edits_parser = subparsers.add_parser("plan-audio-edits")
    plan_audio_edits_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_audio_edits_parser.add_argument("--pair-candidates-path", required=True)
    plan_audio_edits_parser.add_argument("--clip-annotations-path", required=True)
    plan_audio_edits_parser.add_argument("--output-path")
    plan_audio_edits_parser.add_argument("--max-plans", type=int, default=10)

    plan_video_masks_parser = subparsers.add_parser("plan-video-masks")
    plan_video_masks_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_video_masks_parser.add_argument("--video-edit-plan-path", required=True)
    plan_video_masks_parser.add_argument("--output-path")
    plan_video_masks_parser.add_argument("--mask-manifest-path")
    plan_video_masks_parser.add_argument("--max-masks", type=int)

    src_ref_plan_parser = subparsers.add_parser("plan-src-ref-images")
    src_ref_plan_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    src_ref_plan_parser.add_argument("--video-edit-plan-path", required=True)
    src_ref_plan_parser.add_argument("--output-path")
    src_ref_plan_parser.add_argument("--image-root")
    src_ref_plan_parser.add_argument("--num-candidates", type=int, default=4)

    src_ref_select_parser = subparsers.add_parser("select-src-ref-images")
    src_ref_select_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    src_ref_select_parser.add_argument("--src-ref-image-plan-path", required=True)
    src_ref_select_parser.add_argument("--output-path")
    src_ref_select_parser.add_argument("--max-selected", type=int, default=2)
    src_ref_select_parser.add_argument("--base-url")
    src_ref_select_parser.add_argument("--api-key", default="EMPTY")
    src_ref_select_parser.add_argument("--model")
    src_ref_select_parser.add_argument("--timeout-seconds", type=float, default=180.0)

    validate_known_pairs_parser = subparsers.add_parser("validate-known-pairs")
    validate_known_pairs_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    validate_known_pairs_parser.add_argument("--known-pairs-path", required=True)
    validate_known_pairs_parser.add_argument("--clip-annotations-path", required=True)
    validate_known_pairs_parser.add_argument("--output-path")
    validate_known_pairs_parser.add_argument("--accepted-output-path")
    validate_known_pairs_parser.add_argument("--raw-index-path")
    validate_known_pairs_parser.add_argument("--base-url", required=True)
    validate_known_pairs_parser.add_argument("--api-key", required=True)
    validate_known_pairs_parser.add_argument("--model", required=True)
    validate_known_pairs_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    validate_known_pairs_parser.add_argument("--max-accepted-pairs", type=int, default=10)
    validate_known_pairs_parser.add_argument("--overwrite", action="store_true")

    validate_pilot_parser = subparsers.add_parser("validate-pilot")
    validate_pilot_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    validate_pilot_parser.add_argument("--pilot-jsonl-path", required=True)
    validate_pilot_parser.add_argument("--gallery-output-path", required=True)
    validate_pilot_parser.add_argument("--report-output-path", required=True)

    review_bundle_parser = subparsers.add_parser("build-review-bundle")
    review_bundle_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    review_bundle_parser.add_argument("--pairs-path", required=True)
    review_bundle_parser.add_argument("--output-dir", required=True)
    review_bundle_parser.add_argument("--clip-annotations-path")
    review_bundle_parser.add_argument("--limit", type=int)
    review_bundle_parser.add_argument("--no-copy-videos", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "init-layout":
        result = {name: str(path) for name, path in ensure_layout(args.root).items()}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "index-raw":
        sources = _parse_sources(args.source) if args.source else discover_raw_sources(args.root)
        if not sources:
            raise ValueError("no raw sources found; pass --source or create <root>/raw_datasets/<dataset>")
        result = index_raw_sources(root=args.root, sources=sources, output_path=args.output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "extract-clips":
        result = extract_clips(
            root=args.root,
            plan_path=args.plan_path,
            raw_index_path=args.raw_index_path,
            output_manifest_path=args.output_manifest_path,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "annotate-clips":
        result = annotate_clips(
            root=args.root,
            clips_manifest_path=args.clips_manifest_path,
            output_path=args.output_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            concurrency=args.concurrency,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-detective-clips":
        result = plan_detective_event_clips(
            root=args.root,
            source_clips_path=args.source_clips_path,
            clip_plan_output_path=args.clip_plan_output_path,
            clip_groups_output_path=args.clip_groups_output_path,
            max_source_videos=args.max_source_videos,
            segment_seconds=args.segment_seconds,
            min_clip_seconds=args.min_clip_seconds,
            max_clip_seconds=args.max_clip_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-stable-omni-clips":
        result = plan_stable_omni_clips(
            root=args.root,
            raw_index_path=args.raw_index_path,
            output_path=args.output_path,
            cache_path=args.cache_path,
            max_source_videos=args.max_source_videos,
            min_clip_seconds=args.min_clip_seconds,
            max_clip_seconds=args.max_clip_seconds,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "cache-reference-understandings":
        result = cache_reference_understandings(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "detective-annotate-clips":
        result = detective_annotate_clips(
            root=args.root,
            clips_manifest_path=args.clips_manifest_path,
            output_path=args.output_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            concurrency=args.concurrency,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "propose-pairs":
        result = propose_pairs(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
            raw_index_path=args.raw_index_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "propose-group-pairs":
        result = propose_group_pairs(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            clip_groups_path=args.clip_groups_path,
            output_path=args.output_path,
            accepted_output_path=args.accepted_output_path,
            raw_index_path=args.raw_index_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_accepted_pairs=args.max_accepted_pairs,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-video-edits":
        result = plan_video_edits(
            root=args.root,
            pair_candidates_path=args.pair_candidates_path,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
            max_plans=args.max_plans,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            planning_mode=args.planning_mode,
            planner_cache_path=args.planner_cache_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-audio-edits":
        result = plan_audio_edits(
            root=args.root,
            pair_candidates_path=args.pair_candidates_path,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
            max_plans=args.max_plans,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-video-masks":
        result = plan_video_masks(
            root=args.root,
            video_edit_plan_path=args.video_edit_plan_path,
            output_path=args.output_path,
            mask_manifest_path=args.mask_manifest_path,
            max_masks=args.max_masks,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-src-ref-images":
        result = plan_src_ref_images(
            root=args.root,
            video_edit_plan_path=args.video_edit_plan_path,
            output_path=args.output_path,
            image_root=args.image_root,
            num_candidates=args.num_candidates,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "select-src-ref-images":
        result = select_src_ref_images(
            root=args.root,
            src_ref_image_plan_path=args.src_ref_image_plan_path,
            output_path=args.output_path,
            max_selected=args.max_selected,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "validate-known-pairs":
        result = validate_known_pairs(
            root=args.root,
            known_pairs_path=args.known_pairs_path,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
            accepted_output_path=args.accepted_output_path,
            raw_index_path=args.raw_index_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_accepted_pairs=args.max_accepted_pairs,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "build-review-bundle":
        result = build_manual_review_bundle(
            root=args.root,
            pairs_path=args.pairs_path,
            output_dir=args.output_dir,
            clip_annotations_path=args.clip_annotations_path,
            limit=args.limit,
            copy_videos=not args.no_copy_videos,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    result = validate_pilot_dataset(
        root=args.root,
        pilot_jsonl_path=args.pilot_jsonl_path,
        gallery_output_path=args.gallery_output_path,
        report_output_path=args.report_output_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
