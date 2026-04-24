from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
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
DEFAULT_LICENSE_NOTE = "internal research pilot only"
VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
ALLOWED_MODALITIES = {"visual", "audio"}
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
MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE = 0.995
VISUAL_DIFFERENCE_TYPES = {"object_count", "object_presence", "attribute", "action", "scene", "visible_text"}
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
) -> dict[str, Any]:
    layout = ensure_layout(root)
    manifest_path = Path(clips_manifest_path)
    clips = list(_load_jsonl(manifest_path))
    if not clips:
        raise ValueError("clip manifest is empty")

    output = Path(output_path) if output_path else layout["captions"] / DEFAULT_CLIP_ANNOTATIONS_NAME
    existing_records = {} if overwrite else _load_records_by_key(output, "clip_id")
    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

    output_records: list[dict[str, Any]] = []
    annotated_count = 0
    reused_count = 0
    fallback_count = 0
    detective_to_single_pass_count = 0
    for item in clips:
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError("clip manifest contains an entry without clip_id")

        if clip_id in existing_records:
            record = existing_records[clip_id]
            reused_count += 1
        else:
            clip_path = _resolve_under_root(layout["root"], str(item.get("output_path", "")).strip())
            if not clip_path.exists():
                raise FileNotFoundError(f"clip output does not exist: {clip_path}")

            fallback_reason = ""
            detective_fallback_reason = ""
            detective_fallback_used = False
            raw_model_output: dict[str, Any] = {}
            if detective:
                tool_observations = _build_toolbox_observations(clip_path)
                try:
                    normalized, raw_model_output = client.annotate_clip_detective(
                        clip_path=str(clip_path),
                        tool_observations=tool_observations,
                    )
                    fallback_used = False
                except Exception as detective_exc:
                    detective_fallback_used = True
                    detective_fallback_reason = "detective_to_single_pass"
                    try:
                        normalized, single_pass_output = client.annotate_clip(clip_path=str(clip_path))
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
                        detective_to_single_pass_count += 1
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
                    normalized, raw_model_output = client.annotate_clip(clip_path=str(clip_path))
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
            annotated_count += 1

        if bool(record.get("fallback_used")):
            fallback_count += 1
        output_records.append(record)

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
            if proposal_id in existing_records:
                record = existing_records[proposal_id]
                reused_count += 1
            else:
                reference_annotation = candidate["reference_annotation"]
                target_annotation = candidate["target_annotation"]
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
                    verification, verification_raw_output = client.verify_pair_difference(
                        proposal=proposal_view,
                        reference_annotation=_annotation_prompt_view(reference_annotation),
                        target_annotation=_annotation_prompt_view(target_annotation),
                    )
                    verification_fallback_used = False
                except Exception as exc:
                    verification = _fallback_pair_verification(reason=f"{type(exc).__name__}: {exc}")
                    verification_raw_output = {"error": f"{type(exc).__name__}: {exc}"}
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
                    "evidence": _evidence_from_annotations(
                        reference_annotation,
                        target_annotation,
                        difference_evidence=proposal_difference_evidence,
                    ),
                    "judge": judge,
                    "verification": verification,
                    "speech_quality": speech_quality,
                    "audio_event_quality": audio_event_quality,
                    "transcript_backed": speech_quality.get("transcript_backed"),
                    "accepted": accepted,
                    "fallback_used": fallback_used,
                    "raw_model_output": raw_model_output,
                    "raw_judge_output": judge_raw_output,
                    "raw_verification_output": verification_raw_output,
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
            acceptance_issues = _pair_record_acceptance_issues(
                root=layout["root"],
                record=record,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            if acceptance_issues:
                record = dict(record)
                judge = dict(record.get("judge", {}))
                judge["accept"] = False
                judge["reject_reason"] = "; ".join(acceptance_issues)
                record["judge"] = judge
                record["accepted"] = False
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
            expected_proposal_id = _build_proposal_id(reference_video, target_video)
            if proposal_id and proposal_id != expected_proposal_id:
                errors.append(
                    f"pilot line {index}: proposal_id={proposal_id} does not match expected {expected_proposal_id}"
                )
            pair_key = (reference_video, target_video)
            if pair_key in seen_pair_keys:
                errors.append(f"pilot line {index}: duplicate reference-target pair={pair_key}")
            seen_pair_keys.add(pair_key)

        modalities = [str(item).strip() for item in record.get("modalities", []) if str(item).strip()]
        modality_counter.update(modalities)

        difference = record.get("difference", {})
        difference_type = str(difference.get("type", "")).strip()
        if difference_type:
            difference_counter[difference_type] += 1

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
    judged_path = pilot_jsonl_path.with_name("judged_pair_proposals.jsonl")
    if not judged_path.exists():
        return _empty_pair_verification_counts()
    return _pair_verification_counts(list(_load_jsonl(judged_path)))


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
        "accepted_after_verification_count": 0,
    }


def _pair_verification_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = _empty_pair_verification_counts()
    for record in records:
        verification = record.get("verification")
        if not isinstance(verification, dict):
            continue
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
            quality = record.get("quality", {})
            if not isinstance(quality, dict):
                quality = {}
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
    if relation in {"shared_source_row", "same_source_video"}:
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
        "summary": annotation.get("summary", ""),
        "subjects": list(annotation.get("subjects", [])),
        "object_counts": dict(annotation.get("object_counts", {})),
        "actions": list(annotation.get("actions", [])),
        "scene": annotation.get("scene", ""),
        "attributes": list(annotation.get("attributes", [])),
        "on_screen_text": list(annotation.get("on_screen_text", [])),
        "speech": list(annotation.get("speech", [])),
        "audio_events": list(annotation.get("audio_events", [])),
        "modalities": list(annotation.get("modalities", [])),
        "storyline": list(annotation.get("storyline", [])),
        "events": list(annotation.get("events", [])),
        "visible_text": list(annotation.get("visible_text", [])),
        "speakers_and_transcript": list(annotation.get("speakers_and_transcript", [])),
        "uncertainties": list(annotation.get("uncertainties", [])),
    }


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
        return f"change {from_value} into {to_value}"
    if difference_type == "object_presence":
        return f"change {from_value} into {to_value}"
    if difference_type == "action":
        return f"change the action from {from_value} to {to_value}"
    if difference_type == "audio_event":
        if _is_non_speech_absence_audio_phrase(from_value) and to_value:
            return f"add {to_value} to the audio"
        if _is_non_speech_absence_audio_phrase(to_value) and from_value:
            return f"remove {from_value} from the audio"
        return f"change the audio from {from_value} to {to_value}"
    if difference_type == "attribute":
        return f"change the attribute from {from_value} to {to_value}"
    if difference_type == "scene":
        return f"change the scene from {from_value} to {to_value}"
    if difference_type == "speech":
        return f"change the speech from {from_value} to {to_value}"
    if difference_type == "visible_text":
        return f"change the visible text from {from_value} to {to_value}"
    return str(primary_difference.get("description", "")).strip() or f"change {from_value} to {to_value}"


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
    }
    normalized["passed"] = _verification_accepts(normalized)
    normalized["failures"] = _verification_failures(normalized)
    return normalized


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
    )


def _verification_failures(verification: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    caption_delta = verification.get("caption_delta", {})
    edit_projection = verification.get("edit_projection", {})
    edit_necessity = verification.get("edit_necessity", {})
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
    return failures


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
    if str(difference.get("type", "")).strip() == "audio_event":
        from_value = str(difference.get("from", "")).strip()
        to_value = str(difference.get("to", "")).strip()
        if _is_speech_only_audio_phrase(from_value) or _is_speech_only_audio_phrase(to_value):
            issues.append("audio_event must not use speech-only or narration-only text as the main difference")
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


def _accepted_record_signature(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    difference = record.get("difference", {})
    from_value = _normalized_phrase(str(difference.get("from", "")).strip())
    to_value = _normalized_phrase(str(difference.get("to", "")).strip())
    if not from_value and not to_value:
        from_value = _normalized_phrase(str(record.get("edit_text", "")).strip())
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
    seen_signatures: set[tuple[str, str, str, str, str]] = set()
    selected_ids: set[str] = set()

    def try_select(record: dict[str, Any]) -> bool:
        signature = _accepted_record_signature(record)
        proposal_id = str(record.get("proposal_id", "")).strip()
        if signature in seen_signatures or proposal_id in selected_ids:
            return False
        selected.append(record)
        seen_signatures.add(signature)
        selected_ids.add(proposal_id)
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
    return {
        "sample_id": f"covr_omni_pilot_{index:04d}",
        "proposal_id": record["proposal_id"],
        "reference_clip_id": record.get("reference_clip_id", ""),
        "target_clip_id": record.get("target_clip_id", ""),
        "reference_video": record["reference_video"],
        "target_video": record["target_video"],
        "edit_text": record["edit_text"],
        "modalities": list(record["modalities"]),
        "reference_caption": record["reference_caption"],
        "target_caption": record["target_caption"],
        "difference": dict(record["difference"]),
        "hard_negatives": list(record["hard_negatives"]),
        "quality": dict(record["quality"]),
        "source": dict(record["source"]),
        "source_context": dict(record.get("source_context", {})),
        "evidence": dict(record.get("evidence", {})),
        "judge": dict(record.get("judge", {})),
        "verification": dict(record.get("verification", {})),
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

    annotate_clips_parser = subparsers.add_parser("annotate-clips")
    annotate_clips_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    annotate_clips_parser.add_argument("--clips-manifest-path", required=True)
    annotate_clips_parser.add_argument("--output-path")
    annotate_clips_parser.add_argument("--base-url", required=True)
    annotate_clips_parser.add_argument("--api-key", required=True)
    annotate_clips_parser.add_argument("--model", required=True)
    annotate_clips_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    annotate_clips_parser.add_argument("--overwrite", action="store_true")

    detective_annotate_parser = subparsers.add_parser("detective-annotate-clips")
    detective_annotate_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    detective_annotate_parser.add_argument("--clips-manifest-path", required=True)
    detective_annotate_parser.add_argument("--output-path")
    detective_annotate_parser.add_argument("--base-url", required=True)
    detective_annotate_parser.add_argument("--api-key", required=True)
    detective_annotate_parser.add_argument("--model", required=True)
    detective_annotate_parser.add_argument("--timeout-seconds", type=float, default=180.0)
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

    validate_pilot_parser = subparsers.add_parser("validate-pilot")
    validate_pilot_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    validate_pilot_parser.add_argument("--pilot-jsonl-path", required=True)
    validate_pilot_parser.add_argument("--gallery-output-path", required=True)
    validate_pilot_parser.add_argument("--report-output-path", required=True)
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

    if args.command == "detective-annotate-clips":
        result = detective_annotate_clips(
            root=args.root,
            clips_manifest_path=args.clips_manifest_path,
            output_path=args.output_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
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

    result = validate_pilot_dataset(
        root=args.root,
        pilot_jsonl_path=args.pilot_jsonl_path,
        gallery_output_path=args.gallery_output_path,
        report_output_path=args.report_output_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
