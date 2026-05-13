from __future__ import annotations

import argparse
import json
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
}


def build_audio_cvr_clips(
    *,
    root: str | Path,
    output_root: str | Path | None = None,
    datasets: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    min_clip_seconds: float = DEFAULT_MIN_CLIP_SECONDS,
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
    stride_seconds: float | None = None,
    min_clips_per_source: int = 2,
    max_clips_per_source: int = 0,
    max_source_videos: int = 0,
    max_source_videos_per_dataset: int = 0,
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
    segments: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    discovered_video_counts: Counter[str] = Counter()
    dataset_scan_roots: dict[str, list[str]] = {}
    extracted_count = 0
    planned_sources = 0

    for dataset in dataset_names:
        dataset_root = raw_root / dataset
        dataset_videos, scan_roots = _iter_dataset_videos(dataset_root, dataset_name=dataset)
        discovered_video_counts[dataset] = len(dataset_videos)
        dataset_scan_roots[dataset] = [_display_path(root_path, path) for path in scan_roots]
        dataset_seen = 0
        for source_path in dataset_videos:
            if max_source_videos > 0 and planned_sources >= max_source_videos:
                break
            if max_source_videos_per_dataset > 0 and dataset_seen >= max_source_videos_per_dataset:
                break
            media = probe_media(source_path)
            if "error" in media:
                skipped["probe_error"] += 1
                continue
            if not media.get("has_video"):
                skipped["missing_video_stream"] += 1
                continue
            if not media.get("has_audio"):
                skipped["missing_audio_stream"] += 1
                continue
            duration = float(media.get("duration_seconds") or 0.0)
            segment_spans = _fixed_segments(
                duration_seconds=duration,
                clip_seconds=clip_seconds,
                min_clip_seconds=min_clip_seconds,
                stride_seconds=stride,
                max_clips=max_clips_per_source,
            )
            if len(segment_spans) < min_clips_per_source:
                skipped[f"too_few_segments:{len(segment_spans)}"] += 1
                continue

            dataset_seen += 1
            planned_sources += 1
            source_counts[dataset] += 1
            source_id = _source_id(dataset, source_path, raw_root)
            source_folder = output_dir / source_id
            source_folder.mkdir(parents=True, exist_ok=True)
            candidate_clip_ids: list[str] = []
            for segment_index, (start_seconds, end_seconds) in enumerate(segment_spans, start=1):
                clip_id = f"{source_id}__single_{segment_index:03d}"
                output_path = source_folder / f"{clip_id}.mp4"
                record = {
                    "clip_id": clip_id,
                    "source_path": str(source_path),
                    "output_path": _display_path(root_path, output_path),
                    "start_seconds": round(start_seconds, 3),
                    "end_seconds": round(end_seconds, 3),
                    "duration_seconds": round(end_seconds - start_seconds, 3),
                    "role": "audio_cvr_8_12s_segment",
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
                    },
                }
                segments.append(record)
                candidate_clip_ids.append(clip_id)
                if not dry_run and (overwrite or not output_path.exists()):
                    command = build_ffmpeg_extract_command(
                        source_path=source_path,
                        output_path=output_path,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        overwrite=overwrite,
                    )
                    subprocess.run(command, check=True)
                    extracted_count += 1

            groups.append(
                {
                    "group_id": f"single_source_{source_id}",
                    "dataset": dataset,
                    "group_reason": "audio_cvr_8_12s_source_video",
                    "source_clip_ids": [source_id],
                    "candidate_clip_ids": candidate_clip_ids,
                    "group_tags": ["single_source", dataset, "audio_cvr_8_12s", "b_line_first"],
                    "source_path": str(source_path),
                    "media_probe": media,
                    "clip_seconds": clip_seconds,
                    "min_clip_seconds": min_clip_seconds,
                    "max_clip_seconds": max_clip_seconds,
                    "stride_seconds": stride,
                }
            )
            print(
                f"[audio-cvr-clips] dataset={dataset} source={planned_sources} clips={len(candidate_clip_ids)} path={source_path}",
                file=sys.stderr,
                flush=True,
            )

    manifest_path = manifest_dir / "audio_cvr_8_12s_clips.jsonl"
    groups_path = manifest_dir / "audio_cvr_8_12s_groups.jsonl"
    summary_path = manifest_dir / "audio_cvr_8_12s_summary.json"
    _write_jsonl(manifest_path, segments)
    _write_jsonl(groups_path, groups)
    summary = {
        "root": str(root_path),
        "raw_root": str(raw_root),
        "output_root": str(output_dir),
        "manifest_path": str(manifest_path),
        "groups_path": str(groups_path),
        "dataset_names": dataset_names,
        "source_video_count": planned_sources,
        "segment_count": len(segments),
        "extracted_count": extracted_count,
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "clip_seconds": clip_seconds,
        "min_clip_seconds": min_clip_seconds,
        "max_clip_seconds": max_clip_seconds,
        "stride_seconds": stride,
        "min_clips_per_source": min_clips_per_source,
        "max_clips_per_source": max_clips_per_source,
        "source_counts": dict(source_counts),
        "discovered_video_counts": dict(discovered_video_counts),
        "dataset_scan_roots": dataset_scan_roots,
        "skipped_counts": dict(skipped),
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


def _iter_dataset_videos(dataset_root: Path, *, dataset_name: str) -> tuple[list[Path], list[Path]]:
    if not dataset_root.exists():
        return [], []
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
) -> list[tuple[float, float]]:
    if duration_seconds <= 0:
        return []
    segments: list[tuple[float, float]] = []
    start = 0.0
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 8-12s Audio-CVR single-source clips from raw datasets.")
    parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--min-clip-seconds", type=float, default=DEFAULT_MIN_CLIP_SECONDS)
    parser.add_argument("--max-clip-seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS)
    parser.add_argument("--stride-seconds", type=float)
    parser.add_argument("--min-clips-per-source", type=int, default=2)
    parser.add_argument("--max-clips-per-source", type=int, default=0)
    parser.add_argument("--max-source-videos", type=int, default=0)
    parser.add_argument("--max-source-videos-per-dataset", type=int, default=0)
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
        clip_seconds=args.clip_seconds,
        min_clip_seconds=args.min_clip_seconds,
        max_clip_seconds=args.max_clip_seconds,
        stride_seconds=args.stride_seconds,
        min_clips_per_source=args.min_clips_per_source,
        max_clips_per_source=args.max_clips_per_source,
        max_source_videos=args.max_source_videos,
        max_source_videos_per_dataset=args.max_source_videos_per_dataset,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
