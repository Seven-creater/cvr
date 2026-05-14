from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from app.composed_data import (
    DEFAULT_DATA_ROOT,
    build_ffmpeg_extract_command,
    ensure_layout,
    probe_media,
    _display_path,
    _safe_id,
    _stable_hash,
    _write_jsonl,
)


VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
DEFAULT_CLIP_SECONDS = 10.0
DEFAULT_MIN_CLIP_SECONDS = 8.0
DEFAULT_MAX_CLIP_SECONDS = 12.0

# Keep this aligned with doc/linux_data_structure.md. The builder should not
# silently depend on a single "video/" convention because the server datasets
# use several layouts.
SERVER_RAW_DATASET_VIDEO_ROOTS: dict[str, tuple[str, ...]] = {
    "daily_omni": ("video",),
    "hdtf": ("videos", "clips"),
    "avatar": (".", "video"),
    "vggsound": ("scratch",),
    "vgg_monoaudio": ("inter_class/mixed",),
    "worldsense": ("videos",),
    "voxceleb": ("vox2_mp4/dev",),
}


def build_audio_cvr_clips(
    *,
    root: str | Path,
    output_root: str | Path | None = None,
    datasets: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
    dataset_video_roots: dict[str, tuple[str, ...]] | None = None,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    min_clip_seconds: float = DEFAULT_MIN_CLIP_SECONDS,
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
    stride_seconds: float | None = None,
    min_clips_per_source: int = 2,
    max_clips_per_source: int = 0,
    max_source_videos: int = 0,
    max_source_videos_per_dataset: int = 0,
    include_tail_segment: bool = False,
    short_clip_group_datasets: set[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    root_path = layout["root"]
    raw_root = root_path / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"raw root does not exist: {raw_root}")
    output_dir = Path(output_root) if output_root else root_path / "clips" / "audio_cvr_8_12s"
    if not output_dir.is_absolute():
        output_dir = root_path / output_dir
    manifest_dir = output_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    clip_set_name = _safe_id(output_dir.name or "audio_cvr_clips")

    clip_seconds = float(clip_seconds)
    min_clip_seconds = float(min_clip_seconds)
    max_clip_seconds = float(max_clip_seconds)
    stride = float(stride_seconds) if stride_seconds is not None else clip_seconds
    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be positive")
    if max_clip_seconds < min_clip_seconds:
        raise ValueError("max_clip_seconds must be >= min_clip_seconds")
    if clip_seconds < min_clip_seconds or clip_seconds > max_clip_seconds:
        raise ValueError("clip_seconds must stay within min/max clip seconds")
    if stride <= 0:
        raise ValueError("stride_seconds must be positive")
    min_clips_per_source = max(1, int(min_clips_per_source or 1))
    max_clips_per_source = max(0, int(max_clips_per_source or 0))

    dataset_names = _dataset_names(raw_root, datasets, exclude_datasets)
    scan_root_overrides = {key.lower(): value for key, value in (dataset_video_roots or {}).items()}
    short_clip_group_dataset_names = {name.lower() for name in (short_clip_group_datasets or set())}
    segments: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    short_clip_groups: dict[str, dict[str, Any]] = {}
    short_clip_group_segments: dict[str, list[dict[str, Any]]] = {}
    skipped: Counter[str] = Counter()
    skipped_by_dataset: dict[str, Counter[str]] = {}
    source_counts: Counter[str] = Counter()
    discovered_video_counts: Counter[str] = Counter()
    dataset_scan_roots: dict[str, list[str]] = {}
    extracted_count = 0
    planned_sources = 0

    for dataset in dataset_names:
        skipped_by_dataset.setdefault(dataset, Counter())
        dataset_root = raw_root / dataset
        dataset_videos, scan_roots = _iter_dataset_videos(
            dataset_root,
            dataset_name=dataset,
            configured_roots=scan_root_overrides.get(dataset.lower()),
        )
        discovered_video_counts[dataset] = len(dataset_videos)
        dataset_scan_roots[dataset] = [_display_path(root_path, path) for path in scan_roots]
        dataset_seen = 0
        for source_path in dataset_videos:
            media = probe_media(source_path)
            if "error" in media:
                skipped["probe_error"] += 1
                skipped_by_dataset[dataset]["probe_error"] += 1
                continue
            if not media.get("has_video"):
                skipped["missing_video_stream"] += 1
                skipped_by_dataset[dataset]["missing_video_stream"] += 1
                continue
            if not media.get("has_audio"):
                skipped["missing_audio_stream"] += 1
                skipped_by_dataset[dataset]["missing_audio_stream"] += 1
                continue
            duration = float(media.get("duration_seconds") or 0.0)
            segment_spans = _fixed_segments(
                duration_seconds=duration,
                clip_seconds=clip_seconds,
                min_clip_seconds=min_clip_seconds,
                stride_seconds=stride,
                max_clips=max_clips_per_source,
                include_tail_segment=include_tail_segment,
            )
            use_short_clip_group = (
                dataset.lower() in short_clip_group_dataset_names
                and len(segment_spans) == 1
                and min_clip_seconds <= duration <= max_clip_seconds
            )
            if len(segment_spans) < min_clips_per_source and not use_short_clip_group:
                reason = f"too_few_segments:{len(segment_spans)}"
                skipped[reason] += 1
                skipped_by_dataset[dataset][reason] += 1
                continue

            if use_short_clip_group:
                source_id = _short_clip_group_source_id(dataset, source_path, raw_root)
                if source_id not in short_clip_groups:
                    if max_source_videos > 0 and planned_sources >= max_source_videos:
                        break
                    if max_source_videos_per_dataset > 0 and dataset_seen >= max_source_videos_per_dataset:
                        break
                    dataset_seen += 1
                    planned_sources += 1
                    source_counts[dataset] += 1
            else:
                if max_source_videos > 0 and planned_sources >= max_source_videos:
                    break
                if max_source_videos_per_dataset > 0 and dataset_seen >= max_source_videos_per_dataset:
                    break
                dataset_seen += 1
                planned_sources += 1
                source_counts[dataset] += 1
                source_id = _source_id(dataset, source_path, raw_root)
            source_folder = output_dir / source_id
            source_folder.mkdir(parents=True, exist_ok=True)
            candidate_clip_ids: list[str] = [] if not use_short_clip_group else short_clip_groups.get(source_id, {}).get("candidate_clip_ids", [])
            for segment_index, (start_seconds, end_seconds) in enumerate(segment_spans, start=1):
                if use_short_clip_group:
                    clip_id = _short_clip_id(source_id, source_path, raw_root)
                else:
                    clip_id = f"{source_id}__single_{segment_index:03d}"
                output_path = source_folder / f"{clip_id}.mp4"
                record = {
                    "clip_id": clip_id,
                    "source_path": str(source_path),
                    "output_path": _display_path(root_path, output_path),
                    "start_seconds": round(start_seconds, 3),
                    "end_seconds": round(end_seconds, 3),
                    "duration_seconds": round(end_seconds - start_seconds, 3),
                    "role": f"{clip_set_name}_segment",
                    "notes": f"audio-cvr fixed {clip_seconds:g}s segment, min {min_clip_seconds:g}s",
                    "dataset": dataset,
                    "source_clip_id": source_id,
                    "source_window_start_seconds": 0.0,
                    "source_window_duration_seconds": round(duration, 3),
                    "relative_start_seconds": round(start_seconds, 3),
                    "relative_end_seconds": round(end_seconds, 3),
                    "group_id": f"single_source_{source_id}",
                    "media_probe": media,
                    "audio_cvr_clip_policy": {
                        "clip_seconds": clip_seconds,
                        "min_clip_seconds": min_clip_seconds,
                        "max_clip_seconds": max_clip_seconds,
                        "stride_seconds": stride,
                        "include_tail_segment": include_tail_segment,
                    },
                }
                if use_short_clip_group:
                    short_clip_group_segments.setdefault(source_id, []).append(record)
                else:
                    segments.append(record)
                candidate_clip_ids.append(clip_id)
                if not dry_run and (overwrite or not output_path.exists()):
                    _extract_clip_atomic(
                        source_path=source_path,
                        output_path=output_path,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        source_duration_seconds=duration,
                        overwrite=overwrite,
                    )
                    extracted_count += 1

            if use_short_clip_group:
                group_record = short_clip_groups.setdefault(
                    source_id,
                    {
                        "group_id": f"single_source_{source_id}",
                        "dataset": dataset,
                        "group_reason": f"{clip_set_name}_short_clip_parent_group",
                        "source_clip_ids": [source_id],
                        "candidate_clip_ids": [],
                        "group_tags": ["single_source", dataset, clip_set_name, "b_line_first", "short_clip_group"],
                        "source_paths": [],
                        "clip_seconds": clip_seconds,
                        "min_clip_seconds": min_clip_seconds,
                        "max_clip_seconds": max_clip_seconds,
                        "stride_seconds": stride,
                        "include_tail_segment": include_tail_segment,
                    },
                )
                group_record["candidate_clip_ids"].extend([clip_id for clip_id in candidate_clip_ids if clip_id not in group_record["candidate_clip_ids"]])
                group_record["source_paths"].append(str(source_path))
            else:
                groups.append(
                    {
                        "group_id": f"single_source_{source_id}",
                        "dataset": dataset,
                        "group_reason": f"{clip_set_name}_source_video",
                        "source_clip_ids": [source_id],
                        "candidate_clip_ids": candidate_clip_ids,
                        "group_tags": ["single_source", dataset, clip_set_name, "b_line_first"],
                        "source_path": str(source_path),
                        "media_probe": media,
                        "clip_seconds": clip_seconds,
                        "min_clip_seconds": min_clip_seconds,
                        "max_clip_seconds": max_clip_seconds,
                        "stride_seconds": stride,
                        "include_tail_segment": include_tail_segment,
                    }
                )
            print(
                f"[audio-cvr-clips] dataset={dataset} source={planned_sources} clips={len(candidate_clip_ids)} path={source_path}",
                file=sys.stderr,
                flush=True,
            )

    for group_record in short_clip_groups.values():
        if len(group_record.get("candidate_clip_ids", [])) >= min_clips_per_source:
            groups.append(group_record)
            source_ids = group_record.get("source_clip_ids") or []
            if source_ids:
                segments.extend(short_clip_group_segments.get(str(source_ids[0]), []))
        else:
            reason = f"too_few_grouped_short_clips:{len(group_record.get('candidate_clip_ids', []))}"
            skipped[reason] += 1
            skipped_by_dataset.setdefault(str(group_record.get("dataset") or "unknown"), Counter())[reason] += 1

    final_source_counts = Counter(str(group.get("dataset") or "unknown") for group in groups)

    manifest_path = manifest_dir / f"{clip_set_name}_clips.jsonl"
    groups_path = manifest_dir / f"{clip_set_name}_groups.jsonl"
    summary_path = manifest_dir / f"{clip_set_name}_summary.json"
    _write_jsonl(manifest_path, segments)
    _write_jsonl(groups_path, groups)
    summary = {
        "root": str(root_path),
        "raw_root": str(raw_root),
        "output_root": str(output_dir),
        "manifest_path": str(manifest_path),
        "groups_path": str(groups_path),
        "dataset_names": dataset_names,
        "source_video_count": sum(final_source_counts.values()),
        "segment_count": len(segments),
        "extracted_count": extracted_count,
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "clip_seconds": clip_seconds,
        "min_clip_seconds": min_clip_seconds,
        "max_clip_seconds": max_clip_seconds,
        "stride_seconds": stride,
        "include_tail_segment": include_tail_segment,
        "short_clip_group_datasets": sorted(short_clip_group_dataset_names),
        "min_clips_per_source": min_clips_per_source,
        "max_clips_per_source": max_clips_per_source,
        "source_counts": dict(final_source_counts),
        "discovered_video_counts": dict(discovered_video_counts),
        "dataset_scan_roots": dataset_scan_roots,
        "skipped_counts": dict(skipped),
        "skipped_counts_by_dataset": {dataset: dict(counter) for dataset, counter in skipped_by_dataset.items()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _dataset_names(raw_root: Path, datasets: list[str] | None, exclude_datasets: list[str] | None = None) -> list[str]:
    excluded = {
        part.strip().lower()
        for raw in (exclude_datasets or [])
        for part in str(raw).split(",")
        if part.strip()
    }
    if datasets:
        selected: list[str] = []
        for raw in datasets:
            selected.extend(part.strip() for part in str(raw).split(",") if part.strip())
        return [name for name in selected if name.lower() not in excluded]
    return [
        path.name
        for path in sorted(raw_root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and path.name.lower() not in excluded
    ]


def _iter_dataset_videos(
    dataset_root: Path,
    *,
    dataset_name: str,
    configured_roots: tuple[str, ...] | None = None,
) -> tuple[list[Path], list[Path]]:
    if not dataset_root.exists():
        return [], []
    if configured_roots is None:
        configured_roots = SERVER_RAW_DATASET_VIDEO_ROOTS.get(dataset_name.lower())
    scan_roots: list[Path] = []
    if configured_roots:
        for relative_root in configured_roots:
            root = dataset_root if relative_root == "." else dataset_root / relative_root
            if root.exists():
                scan_roots.append(root)
    if not scan_roots:
        scan_roots = [dataset_root]

    videos: dict[str, Path] = {}
    for scan_root in scan_roots:
        for path in sorted(scan_root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                videos[path.resolve().as_posix()] = path
    return [videos[key] for key in sorted(videos)], scan_roots


def _fixed_segments(
    *,
    duration_seconds: float,
    clip_seconds: float,
    min_clip_seconds: float,
    stride_seconds: float,
    max_clips: int,
    include_tail_segment: bool = False,
) -> list[tuple[float, float]]:
    if duration_seconds <= 0:
        return []
    segments: list[tuple[float, float]] = []
    start = 0.0
    if include_tail_segment:
        while start + clip_seconds <= duration_seconds + 1e-6:
            segments.append((round(start, 3), round(start + clip_seconds, 3)))
            if max_clips > 0 and len(segments) >= max_clips:
                return segments
            start += stride_seconds
        if duration_seconds >= min_clip_seconds:
            tail_start = max(0.0, duration_seconds - clip_seconds)
            tail_end = duration_seconds
            if tail_end - tail_start >= min_clip_seconds and (
                not segments or tail_start > segments[-1][0] + 1e-3
            ):
                segments.append((round(tail_start, 3), round(tail_end, 3)))
        return segments

    while start < duration_seconds:
        end = min(start + clip_seconds, duration_seconds)
        if end - start >= min_clip_seconds:
            segments.append((round(start, 3), round(end, 3)))
            if max_clips > 0 and len(segments) >= max_clips:
                break
        start += stride_seconds
    return segments


def _source_id(dataset: str, source_path: Path, raw_root: Path) -> str:
    try:
        relative = source_path.relative_to(raw_root).as_posix()
    except ValueError:
        relative = source_path.as_posix()
    return _safe_id(f"{dataset}_{source_path.stem}_{_stable_hash(relative)[:8]}")


def _short_clip_group_source_id(dataset: str, source_path: Path, raw_root: Path) -> str:
    try:
        relative_parent = source_path.parent.relative_to(raw_root).as_posix()
    except ValueError:
        relative_parent = source_path.parent.as_posix()
    return _safe_id(f"{dataset}_{relative_parent}_{_stable_hash(relative_parent)[:8]}")


def _short_clip_id(source_id: str, source_path: Path, raw_root: Path) -> str:
    try:
        relative = source_path.relative_to(raw_root).as_posix()
    except ValueError:
        relative = source_path.as_posix()
    return _safe_id(f"{source_id}_{source_path.stem}_{_stable_hash(relative)[:8]}")


def _extract_clip_atomic(
    *,
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
    source_duration_seconds: float | None = None,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_full_mp4_span(source_path, start_seconds, end_seconds, source_duration_seconds):
        _materialize_full_mp4_atomic(source_path=source_path, output_path=output_path, overwrite=overwrite)
        return
    temp_path = output_path.with_name(f".{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    command = build_ffmpeg_extract_command(
        source_path=source_path,
        output_path=temp_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        overwrite=True,
    )
    try:
        subprocess.run(command, check=True)
        if overwrite or not output_path.exists():
            temp_path.replace(output_path)
        else:
            temp_path.unlink(missing_ok=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _is_full_mp4_span(
    source_path: Path,
    start_seconds: float,
    end_seconds: float,
    source_duration_seconds: float | None,
) -> bool:
    if source_path.suffix.lower() != ".mp4":
        return False
    if source_duration_seconds is None or source_duration_seconds <= 0:
        return False
    return start_seconds <= 0.001 and abs(float(end_seconds) - float(source_duration_seconds)) <= 0.05


def _materialize_full_mp4_atomic(*, source_path: Path, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    try:
        try:
            os.link(source_path, temp_path)
        except OSError:
            shutil.copy2(source_path, temp_path)
        if overwrite or not output_path.exists():
            temp_path.replace(output_path)
        else:
            temp_path.unlink(missing_ok=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 8-12s Audio-CVR single-source clips from raw datasets.")
    parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument(
        "--dataset-video-root",
        action="append",
        default=[],
        help="Override scan roots for one dataset, e.g. hdtf=videos or avatar=.,video.",
    )
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--min-clip-seconds", type=float, default=DEFAULT_MIN_CLIP_SECONDS)
    parser.add_argument("--max-clip-seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS)
    parser.add_argument("--stride-seconds", type=float)
    parser.add_argument("--min-clips-per-source", type=int, default=2)
    parser.add_argument("--max-clips-per-source", type=int, default=0)
    parser.add_argument("--max-source-videos", type=int, default=0)
    parser.add_argument("--max-source-videos-per-dataset", type=int, default=0)
    parser.add_argument("--include-tail-segment", action="store_true")
    parser.add_argument(
        "--short-clip-group-dataset",
        action="append",
        default=[],
        help="For datasets made of short mp4 clips, group full short clips by their parent folder.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_audio_cvr_clips(
        root=args.root,
        output_root=args.output_root,
        datasets=args.dataset,
        exclude_datasets=args.exclude_dataset,
        dataset_video_roots=_parse_dataset_video_roots(args.dataset_video_root),
        clip_seconds=args.clip_seconds,
        min_clip_seconds=args.min_clip_seconds,
        max_clip_seconds=args.max_clip_seconds,
        stride_seconds=args.stride_seconds,
        min_clips_per_source=args.min_clips_per_source,
        max_clips_per_source=args.max_clips_per_source,
        max_source_videos=args.max_source_videos,
        max_source_videos_per_dataset=args.max_source_videos_per_dataset,
        include_tail_segment=args.include_tail_segment,
        short_clip_group_datasets={item.strip() for raw in args.short_clip_group_dataset for item in raw.split(",") if item.strip()},
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _parse_dataset_video_roots(items: list[str]) -> dict[str, tuple[str, ...]]:
    overrides: dict[str, tuple[str, ...]] = {}
    for raw_item in items or []:
        if "=" not in raw_item:
            raise ValueError(f"--dataset-video-root must be DATASET=ROOT[,ROOT...], got: {raw_item}")
        dataset, raw_roots = raw_item.split("=", 1)
        dataset = dataset.strip()
        roots = tuple(root.strip() for root in raw_roots.split(",") if root.strip())
        if not dataset or not roots:
            raise ValueError(f"--dataset-video-root must include dataset and at least one root, got: {raw_item}")
        overrides[dataset] = roots
    return overrides


if __name__ == "__main__":
    main()
