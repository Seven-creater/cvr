from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from app.composed_data import DEFAULT_DATA_ROOT, ensure_layout


VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
DAILY_OMNI_NAME = "daily_omni"
WORLDSENSE_NAME = "worldsense"


def prepare_source_datasets(
    *,
    root: str | Path,
    daily_omni_root: str | Path | None = None,
    worldsense_root: str | Path | None = None,
    clip_limit: int = 50,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    root_path = layout["root"]
    raw_datasets_root = root_path / "raw_datasets"
    source_specs = [
        (DAILY_OMNI_NAME, Path(daily_omni_root) if daily_omni_root else raw_datasets_root / DAILY_OMNI_NAME),
        (WORLDSENSE_NAME, Path(worldsense_root) if worldsense_root else raw_datasets_root / WORLDSENSE_NAME),
    ]

    all_rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, dict[str, int]] = {}
    for dataset_name, source_root in source_specs:
        if not source_root.exists():
            dataset_counts[dataset_name] = {"rows": 0, "clips": 0, "missing_root": 1}
            continue
        rows = list(_load_dataset_rows(dataset_name=dataset_name, source_root=source_root, output_root=root_path))
        if not rows:
            rows = list(_rows_from_media_files(dataset_name=dataset_name, source_root=source_root))
        all_rows.extend(rows)
        dataset_counts[dataset_name] = {
            "rows": len(rows),
            "clips": len({row["video_path"] for row in rows if row.get("video_path")}),
            "missing_root": 0,
        }

    clips_all = _build_clip_records(all_rows, root_path)
    clips_pilot = _select_balanced_pilot_clips(clips_all, max(0, int(clip_limit)))

    rows_path = layout["metadata"] / "source_rows.jsonl"
    clips_all_path = layout["metadata"] / "source_clips_all.jsonl"
    clips_pilot_path = layout["metadata"] / f"source_clips_pilot{len(clips_pilot)}.jsonl"
    report_path = layout["reports"] / "source_dataset_prepare_summary.md"

    _write_jsonl(rows_path, all_rows)
    _write_jsonl(clips_all_path, clips_all)
    _write_jsonl(clips_pilot_path, clips_pilot)
    report_path.write_text(
        _build_prepare_report(
            rows_path=rows_path,
            clips_all_path=clips_all_path,
            clips_pilot_path=clips_pilot_path,
            dataset_counts=dataset_counts,
            row_count=len(all_rows),
            clip_count=len(clips_all),
            pilot_clip_count=len(clips_pilot),
        ),
        encoding="utf-8",
    )

    return {
        "root": str(root_path),
        "source_rows_path": str(rows_path),
        "source_clips_all_path": str(clips_all_path),
        "source_clips_pilot_path": str(clips_pilot_path),
        "report_path": str(report_path),
        "row_count": len(all_rows),
        "clip_count": len(clips_all),
        "pilot_clip_count": len(clips_pilot),
        "dataset_counts": dataset_counts,
    }


def _load_dataset_rows(*, dataset_name: str, source_root: Path, output_root: Path) -> Iterable[dict[str, Any]]:
    parquet_files = sorted(source_root.rglob("*.parquet"))
    if not parquet_files:
        return []

    rows: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        for row_index, raw_row in enumerate(_read_parquet_rows(parquet_path), start=1):
            normalized = _normalize_source_row(
                dataset_name=dataset_name,
                source_root=source_root,
                parquet_path=parquet_path,
                row_index=row_index,
                raw_row=raw_row,
                output_root=output_root,
            )
            if normalized:
                rows.append(normalized)
    return rows


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read downloaded dataset parquet files") from exc
    table = pq.read_table(path)
    return table.to_pylist()


def _normalize_source_row(
    *,
    dataset_name: str,
    source_root: Path,
    parquet_path: Path,
    row_index: int,
    raw_row: dict[str, Any],
    output_root: Path,
) -> dict[str, Any] | None:
    video_path = _extract_media_value(
        raw_row=raw_row,
        source_root=source_root,
        output_dir=output_root / "raw" / dataset_name / "video",
        row_key=f"{parquet_path.stem}_{row_index}",
        media_kind="video",
    )
    audio_path = _extract_media_value(
        raw_row=raw_row,
        source_root=source_root,
        output_dir=output_root / "raw" / dataset_name / "audio",
        row_key=f"{parquet_path.stem}_{row_index}",
        media_kind="audio",
    )
    video_id = _first_string(raw_row, ["video_id", "videoid", "video", "id", "uid", "sample_id"])
    if not video_id:
        video_id = Path(video_path).stem if video_path else f"{parquet_path.stem}_{row_index}"

    text_fields = _extract_text_fields(raw_row)
    if not video_path and not audio_path and not text_fields:
        return None

    return {
        "source_row_id": _stable_id(dataset_name, parquet_path.relative_to(source_root).as_posix(), str(row_index)),
        "dataset": dataset_name,
        "split": _infer_split(parquet_path),
        "row_index": row_index,
        "source_file": str(parquet_path),
        "video_id": str(video_id),
        "video_path": video_path,
        "audio_path": audio_path,
        "text_fields": text_fields,
        "raw_columns": sorted(raw_row.keys()),
    }


def _rows_from_media_files(*, dataset_name: str, source_root: Path) -> Iterable[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(source_root.rglob("*")), start=1):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        rows.append(
            {
                "source_row_id": _stable_id(dataset_name, path.relative_to(source_root).as_posix()),
                "dataset": dataset_name,
                "split": "unknown",
                "row_index": index,
                "source_file": str(path),
                "video_id": path.stem,
                "video_path": str(path),
                "audio_path": _find_sibling_audio(path),
                "text_fields": {},
                "raw_columns": [],
            }
        )
    return rows


def _extract_media_value(
    *,
    raw_row: dict[str, Any],
    source_root: Path,
    output_dir: Path,
    row_key: str,
    media_kind: str,
) -> str:
    candidates = _media_candidate_values(raw_row, media_kind)
    for column_name, value in candidates:
        path = _materialize_media_value(
            value=value,
            source_root=source_root,
            output_dir=output_dir,
            row_key=f"{row_key}_{_safe_name(column_name)}",
            media_kind=media_kind,
        )
        if path:
            return path
    return ""


def _media_candidate_values(raw_row: dict[str, Any], media_kind: str) -> list[tuple[str, Any]]:
    suffixes = VIDEO_SUFFIXES if media_kind == "video" else AUDIO_SUFFIXES
    keyword = media_kind
    candidates: list[tuple[str, Any]] = []
    for column_name, value in raw_row.items():
        name = str(column_name).lower()
        if keyword in name or "path" in name or "file" in name:
            candidates.append((str(column_name), value))
            continue
        if isinstance(value, str) and Path(value).suffix.lower() in suffixes:
            candidates.append((str(column_name), value))
        if isinstance(value, dict):
            raw_path = value.get("path") or value.get("file") or value.get("filename")
            if isinstance(raw_path, str) and Path(raw_path).suffix.lower() in suffixes:
                candidates.append((str(column_name), value))
    return candidates


def _materialize_media_value(
    *,
    value: Any,
    source_root: Path,
    output_dir: Path,
    row_key: str,
    media_kind: str,
) -> str:
    suffixes = VIDEO_SUFFIXES if media_kind == "video" else AUDIO_SUFFIXES
    if isinstance(value, dict):
        raw_path = value.get("path") or value.get("file") or value.get("filename")
        raw_bytes = value.get("bytes")
        if raw_bytes:
            suffix = Path(str(raw_path or "")).suffix.lower()
            if suffix not in suffixes:
                suffix = ".mp4" if media_kind == "video" else ".wav"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{row_key}{suffix}"
            if not output_path.exists():
                output_path.write_bytes(bytes(raw_bytes))
            return str(output_path)
        if raw_path:
            return _resolve_source_path(str(raw_path), source_root, suffixes)
        return ""
    if isinstance(value, bytes):
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".mp4" if media_kind == "video" else ".wav"
        output_path = output_dir / f"{row_key}{suffix}"
        if not output_path.exists():
            output_path.write_bytes(value)
        return str(output_path)
    if isinstance(value, str):
        return _resolve_source_path(value, source_root, suffixes)
    return ""


def _resolve_source_path(value: str, source_root: Path, suffixes: set[str]) -> str:
    value = value.strip()
    if not value:
        return ""
    if Path(value).suffix.lower() not in suffixes:
        return ""
    path = Path(value)
    candidates = [path] if path.is_absolute() else [source_root / path, source_root / value.lstrip("./")]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _extract_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    preferred_names = {
        "question",
        "query",
        "caption",
        "video_caption",
        "answer",
        "label",
        "choices",
        "candidates",
        "subtitle",
        "transcript",
        "text",
    }
    fields: dict[str, Any] = {}
    for key, value in row.items():
        lowered = str(key).lower()
        if lowered not in preferred_names and not any(name in lowered for name in preferred_names):
            continue
        normalized = _json_safe_text_value(value)
        if normalized not in ("", [], {}):
            fields[str(key)] = normalized
    return fields


def _json_safe_text_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe_text_value(item) for item in value if _json_safe_text_value(item) != ""]
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"bytes"}:
                continue
            normalized = _json_safe_text_value(item)
            if normalized != "":
                safe[str(key)] = normalized
        return safe
    return str(value)


def _build_clip_records(rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    by_video_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_path = str(row.get("video_path", "")).strip()
        if not video_path:
            continue
        if video_path not in by_video_path:
            by_video_path[video_path] = {
                "clip_id": f"{row['dataset']}_{_safe_name(str(row.get('video_id') or Path(video_path).stem))}",
                "source_path": video_path,
                "output_path": _display_path(root, Path(video_path)),
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "duration_seconds": 0.0,
                "role": "source_clip",
                "notes": "whole source video; run manual clipping before final pilot if this video is too long",
                "dataset": row["dataset"],
                "source_row_ids": [],
                "text_fields": {},
            }
        by_video_path[video_path]["source_row_ids"].append(row["source_row_id"])
        for key, value in row.get("text_fields", {}).items():
            by_video_path[video_path]["text_fields"].setdefault(key, value)
    return list(by_video_path.values())


def _select_balanced_pilot_clips(clips: list[dict[str, Any]], clip_limit: int) -> list[dict[str, Any]]:
    if clip_limit <= 0 or not clips:
        return []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for clip in clips:
        dataset = str(clip.get("dataset") or "unknown")
        by_dataset.setdefault(dataset, []).append(clip)

    for dataset_clips in by_dataset.values():
        dataset_clips.sort(key=_pilot_priority_key)

    selected: list[dict[str, Any]] = []
    seen_clip_ids: set[str] = set()
    dataset_names = sorted(by_dataset)
    while len(selected) < clip_limit:
        added_this_round = False
        for dataset_name in dataset_names:
            dataset_clips = by_dataset[dataset_name]
            while dataset_clips:
                candidate = dataset_clips.pop(0)
                clip_id = str(candidate.get("clip_id", "")).strip()
                if clip_id in seen_clip_ids:
                    continue
                selected.append(candidate)
                seen_clip_ids.add(clip_id)
                added_this_round = True
                break
            if len(selected) >= clip_limit:
                break
        if not added_this_round:
            break
    return selected


def _pilot_priority_key(clip: dict[str, Any]) -> tuple[int, str]:
    text_fields = clip.get("text_fields", {})
    if not isinstance(text_fields, dict):
        text_fields = {}
    has_audio_or_sync_question = any(
        keyword in json.dumps(text_fields, ensure_ascii=False).lower()
        for keyword in ("audio", "sound", "speech", "music", "voice", "synchronized", "simultaneously")
    )
    has_text = bool(text_fields)
    return (0 if has_audio_or_sync_question else 1, 0 if has_text else 1, str(clip.get("clip_id", "")))


def _first_string(row: dict[str, Any], names: list[str]) -> str:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def _infer_split(path: Path) -> str:
    lowered = path.as_posix().lower()
    for split in ("train", "validation", "valid", "val", "test", "dev"):
        if split in lowered:
            return "validation" if split in {"valid", "val"} else split
    return "unknown"


def _find_sibling_audio(video_path: Path) -> str:
    for suffix in AUDIO_SUFFIXES:
        candidate = video_path.with_suffix(suffix)
        if candidate.exists():
            return str(candidate)
    return ""


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value[:120] or "item"


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
    prefix = _safe_name(parts[0]) if parts else "row"
    return f"{prefix}_{digest}"


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_prepare_report(
    *,
    rows_path: Path,
    clips_all_path: Path,
    clips_pilot_path: Path,
    dataset_counts: dict[str, dict[str, int]],
    row_count: int,
    clip_count: int,
    pilot_clip_count: int,
) -> str:
    lines = [
        "# Source Dataset Prepare Summary",
        "",
        f"- source rows: {row_count}",
        f"- unique clips: {clip_count}",
        f"- pilot clips: {pilot_clip_count}",
        f"- rows path: `{rows_path}`",
        f"- all clips path: `{clips_all_path}`",
        f"- pilot clips path: `{clips_pilot_path}`",
        "",
        "## Dataset Counts",
        "",
        "| dataset | rows | clips | missing_root |",
        "|---|---:|---:|---:|",
    ]
    for dataset_name, counts in sorted(dataset_counts.items()):
        lines.append(
            f"| {dataset_name} | {counts.get('rows', 0)} | {counts.get('clips', 0)} | {counts.get('missing_root', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Use the pilot clip manifest with `python -m app.composed_data annotate-clips` once the Qwen3-Omni service is ready.",
            "For long videos, manually create a clip plan and run `extract-clips` before annotation.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare downloaded source datasets for composed Omni retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", default=DEFAULT_DATA_ROOT)
    prepare.add_argument("--daily-omni-root")
    prepare.add_argument("--worldsense-root")
    prepare.add_argument("--clip-limit", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_source_datasets(
            root=args.root,
            daily_omni_root=args.daily_omni_root,
            worldsense_root=args.worldsense_root,
            clip_limit=args.clip_limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
