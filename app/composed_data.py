from __future__ import annotations

import argparse
import hashlib
import json
import re
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
            raw_model_output: dict[str, Any] = {}
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
    same_context_scores: list[float] = []
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
        if proposal_id:
            if proposal_id in seen_proposal_ids:
                errors.append(f"pilot line {index}: duplicate proposal_id={proposal_id}")
            seen_proposal_ids.add(proposal_id)

        reference_video = str(record.get("reference_video", "")).strip()
        target_video = str(record.get("target_video", "")).strip()
        if reference_video and target_video:
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
            try:
                same_context_scores.append(float(quality.get("same_context_score", 0.0)))
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

    summary = {
        "sample_count": len(pilot_records),
        "gallery_count": len(gallery_records),
        "modality_counts": dict(sorted(modality_counter.items())),
        "difference_type_counts": dict(sorted(difference_counter.items())),
        "source_context_counts": dict(sorted(source_context_counter.items())),
        "quality_summary": _quality_summary(same_context_scores),
        "automated_acceptance": {
            "sample_count_between_5_and_10": 5 <= len(pilot_records) <= 10,
            "audio_samples_at_least_2": modality_counter.get("audio", 0) >= 2,
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

    lines.extend(["", "## Automated Acceptance Checks"])
    for key, value in acceptance.items():
        lines.append(f"- `{key}`: `{'PASS' if value else 'FAIL'}`")
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
    return candidates[:MAX_PAIR_CANDIDATES]


def _score_ordered_pair(
    *,
    root: Path,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if reference_annotation["clip_id"] == target_annotation["clip_id"]:
        return None

    same_context_score = _same_context_score(reference_annotation, target_annotation)
    source_context = _source_context(reference_annotation, target_annotation)
    if source_context["relation"] == "cross_dataset":
        return None
    primary_difference = _detect_primary_difference(reference_annotation, target_annotation)
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
    )
    hard_negative_paths = [
        _display_path(root, _resolve_under_root(root, annotation["output_path"])) for annotation in hard_negative_annotations[:3]
    ]
    if len(hard_negative_paths) < 2:
        return None

    reference_path = _display_path(root, _resolve_under_root(root, reference_annotation["output_path"]))
    target_path = _display_path(root, _resolve_under_root(root, target_annotation["output_path"]))
    if target_path in hard_negative_paths:
        return None

    quality = {
        "same_context_score": round(same_context_score, 3),
        "edit_match_score": round(edit_match_score, 3),
        "target_uniqueness_score": round(target_uniqueness_score, 3),
    }
    composite_score = round(
        quality["same_context_score"] * 0.45
        + quality["edit_match_score"] * 0.35
        + quality["target_uniqueness_score"] * 0.20,
        4,
    )
    composite_score = round(composite_score + source_context["score"] * 0.08, 4)
    return {
        "proposal_id": _build_proposal_id(reference_path, target_path),
        "reference_annotation": _sanitize_annotation_for_output(reference_annotation, root),
        "target_annotation": _sanitize_annotation_for_output(target_annotation, root),
        "primary_difference": primary_difference,
        "quality": quality,
        "composite_score": composite_score,
        "source_context": source_context,
        "hard_negative_annotations": [_sanitize_annotation_for_output(annotation, root) for annotation in hard_negative_annotations[:3]],
        "hard_negative_paths": hard_negative_paths,
    }


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
        or annotation.get("speech")
    )


def _source_context(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {str(value).strip() for value in left.get("source_row_ids", []) if str(value).strip()}
    right_rows = {str(value).strip() for value in right.get("source_row_ids", []) if str(value).strip()}
    shared_rows = sorted(left_rows & right_rows)
    if shared_rows:
        return {"relation": "shared_source_row", "score": 1.0, "shared_source_row_ids": shared_rows}

    left_source_path = str(left.get("source_path", "")).strip()
    right_source_path = str(right.get("source_path", "")).strip()
    if left_source_path and left_source_path == right_source_path:
        return {"relation": "same_source_video", "score": 0.9}

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


def _scene_similarity(left: str, right: str) -> float:
    left_value = left.strip().lower()
    right_value = right.strip().lower()
    if not left_value or not right_value:
        return 0.0
    if left_value == right_value:
        return 1.0
    return _jaccard(_tokenize_text(left_value), _tokenize_text(right_value))


def _detect_primary_difference(reference: dict[str, Any], target: dict[str, Any]) -> dict[str, Any] | None:
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

    reference_actions = _normalize_list(reference.get("actions", []))
    target_actions = _normalize_list(target.get("actions", []))
    added_action = _first_unique(target_actions, reference_actions)
    removed_action = _first_unique(reference_actions, target_actions)
    if added_action or removed_action:
        differences["action"] = {
            "type": "action",
            "from": removed_action or _first_item(reference_actions) or "current action",
            "to": added_action or _first_item(target_actions) or "new action",
            "description": "the main action changes between the clips",
        }

    reference_audio = _normalize_list(reference.get("audio_events", []))
    target_audio = _normalize_list(target.get("audio_events", []))
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

    reference_speech = _normalize_list(reference.get("speech", []))
    target_speech = _normalize_list(target.get("speech", []))
    added_speech = _first_unique(target_speech, reference_speech)
    removed_speech = _first_unique(reference_speech, target_speech)
    if added_speech or removed_speech:
        differences["speech"] = {
            "type": "speech",
            "from": removed_speech or _first_item(reference_speech) or "no speech",
            "to": added_speech or _first_item(target_speech) or "new speech",
            "description": "the spoken content changes between the clips",
        }

    changed_types = [difference_type for difference_type in PAIR_PRIORITY if difference_type in differences]
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
    if primary_difference_type in {"object_count", "object_presence", "action", "audio_event"}:
        base_score += 0.1
    penalty = max(0, len(changed_types) - 1) * 0.10
    return max(0.0, min(1.0, base_score - penalty))


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
) -> float:
    competitor_scores = []
    for other in annotations:
        if other["clip_id"] in {reference_annotation["clip_id"], target_annotation["clip_id"]}:
            continue
        competitor_scores.append(_same_context_score(target_annotation, other))
    if not competitor_scores:
        return 1.0
    highest_competitor = max(competitor_scores)
    return max(0.0, min(1.0, 1.0 - highest_competitor * 0.75))


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
        return f"change the audio from {from_value} to {to_value}"
    if difference_type == "attribute":
        return f"change the attribute from {from_value} to {to_value}"
    if difference_type == "scene":
        return f"change the scene from {from_value} to {to_value}"
    if difference_type == "speech":
        return f"change the speech from {from_value} to {to_value}"
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

    annotate_clips_parser = subparsers.add_parser("annotate-clips")
    annotate_clips_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    annotate_clips_parser.add_argument("--clips-manifest-path", required=True)
    annotate_clips_parser.add_argument("--output-path")
    annotate_clips_parser.add_argument("--base-url", required=True)
    annotate_clips_parser.add_argument("--api-key", required=True)
    annotate_clips_parser.add_argument("--model", required=True)
    annotate_clips_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    annotate_clips_parser.add_argument("--overwrite", action="store_true")

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

    result = validate_pilot_dataset(
        root=args.root,
        pilot_jsonl_path=args.pilot_jsonl_path,
        gallery_output_path=args.gallery_output_path,
        report_output_path=args.report_output_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
