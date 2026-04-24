from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from app.composed_data import DEFAULT_DATA_ROOT, ensure_layout


VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
DAILY_OMNI_NAME = "daily_omni"
WORLDSENSE_NAME = "worldsense"
WEBVID_COVR_NAME = "webvid_covr"


def prepare_source_datasets(
    *,
    root: str | Path,
    daily_omni_root: str | Path | None = None,
    worldsense_root: str | Path | None = None,
    webvid_covr_root: str | Path | None = None,
    webvid_covr_splits: Iterable[str] | None = None,
    clip_limit: int = 50,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    root_path = layout["root"]
    raw_datasets_root = root_path / "raw_datasets"
    source_specs = [
        (DAILY_OMNI_NAME, Path(daily_omni_root) if daily_omni_root else raw_datasets_root / DAILY_OMNI_NAME),
        (WORLDSENSE_NAME, Path(worldsense_root) if worldsense_root else raw_datasets_root / WORLDSENSE_NAME),
    ]
    webvid_root = Path(webvid_covr_root) if webvid_covr_root else raw_datasets_root / WEBVID_COVR_NAME
    webvid_splits = _normalize_requested_splits(webvid_covr_splits or ("train",))

    all_rows: list[dict[str, Any]] = []
    pair_seeds: list[dict[str, Any]] = []
    dataset_counts: dict[str, dict[str, int]] = {}
    for dataset_name, source_root in source_specs:
        if not source_root.exists():
            dataset_counts[dataset_name] = {"rows": 0, "clips": 0, "missing_root": 1}
            continue
        extraction_summary = extract_archives(source_root)
        rows = list(_load_dataset_rows(dataset_name=dataset_name, source_root=source_root, output_root=root_path))
        if not rows:
            rows = list(_rows_from_media_files(dataset_name=dataset_name, source_root=source_root))
        all_rows.extend(rows)
        dataset_counts[dataset_name] = {
            "rows": len(rows),
            "clips": len({row["video_path"] for row in rows if row.get("video_path")}),
            "missing_root": 0,
            "archives": extraction_summary["archive_count"],
            "extracted_archives": extraction_summary["extracted_count"],
        }

    if not webvid_root.exists():
        dataset_counts[WEBVID_COVR_NAME] = {"rows": 0, "clips": 0, "pair_seeds": 0, "missing_root": 1, "missing_video_seeds": 0}
    else:
        webvid_rows, webvid_pair_seeds, webvid_summary = _load_webvid_covr_rows(
            source_root=webvid_root,
            splits=webvid_splits,
        )
        all_rows.extend(webvid_rows)
        pair_seeds.extend(webvid_pair_seeds)
        dataset_counts[WEBVID_COVR_NAME] = {
            "rows": len(webvid_rows),
            "clips": len({row["video_path"] for row in webvid_rows if row.get("video_path")}),
            "pair_seeds": len(webvid_pair_seeds),
            "missing_root": 0,
            "missing_video_seeds": webvid_summary["missing_video_seeds"],
            "csv_files": webvid_summary["csv_file_count"],
        }

    clips_all = _build_clip_records(all_rows, root_path)
    clips_pilot = _select_balanced_pilot_clips(clips_all, max(0, int(clip_limit)))

    rows_path = layout["metadata"] / "source_rows.jsonl"
    clips_all_path = layout["metadata"] / "source_clips_all.jsonl"
    clips_pilot_path = layout["metadata"] / f"source_clips_pilot{len(clips_pilot)}.jsonl"
    pair_seeds_path = layout["metadata"] / "webvid_covr_pair_seeds.jsonl"
    report_path = layout["reports"] / "source_dataset_prepare_summary.md"

    _write_jsonl(rows_path, all_rows)
    _write_jsonl(clips_all_path, clips_all)
    _write_jsonl(clips_pilot_path, clips_pilot)
    _write_jsonl(pair_seeds_path, pair_seeds)
    report_path.write_text(
        _build_prepare_report(
            rows_path=rows_path,
            clips_all_path=clips_all_path,
            clips_pilot_path=clips_pilot_path,
            pair_seeds_path=pair_seeds_path,
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
        "webvid_covr_pair_seeds_path": str(pair_seeds_path),
        "report_path": str(report_path),
        "row_count": len(all_rows),
        "clip_count": len(clips_all),
        "pilot_clip_count": len(clips_pilot),
        "pair_seed_count": len(pair_seeds),
        "dataset_counts": dataset_counts,
    }


def _load_webvid_covr_rows(*, source_root: Path, splits: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    pair_seeds: list[dict[str, Any]] = []
    missing_video_seeds = 0
    csv_paths = _find_webvid_covr_csv_paths(source_root, splits)
    for csv_path in csv_paths:
        split = _normalize_split_name(csv_path.stem)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, raw_row in enumerate(reader, start=1):
                normalized = _normalize_webvid_covr_row(
                    source_root=source_root,
                    csv_path=csv_path,
                    split=split,
                    row_index=row_index,
                    raw_row=raw_row,
                )
                if normalized is None:
                    missing_video_seeds += 1
                    continue
                reference_row, target_row, pair_seed = normalized
                rows.extend([reference_row, target_row])
                pair_seeds.append(pair_seed)
    return rows, pair_seeds, {"missing_video_seeds": missing_video_seeds, "csv_file_count": len(csv_paths)}


def _find_webvid_covr_csv_paths(source_root: Path, splits: list[str]) -> list[Path]:
    csv_paths: list[Path] = []
    indexed: dict[str, Path] = {}
    for csv_path in sorted(source_root.rglob("*.csv")):
        indexed[_normalize_split_name(csv_path.stem)] = csv_path
    for split in splits:
        if split not in indexed:
            raise FileNotFoundError(f"WebVid-CoVR split CSV not found for split={split} under {source_root}")
        csv_paths.append(indexed[split])
    return csv_paths


def _normalize_webvid_covr_row(
    *,
    source_root: Path,
    csv_path: Path,
    split: str,
    row_index: int,
    raw_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    reference_video_path = _resolve_source_path(str(raw_row.get("pth1", "")).strip(), source_root, VIDEO_SUFFIXES)
    target_video_path = _resolve_source_path(str(raw_row.get("pth2", "")).strip(), source_root, VIDEO_SUFFIXES)
    if not reference_video_path or not target_video_path:
        return None
    if not Path(reference_video_path).exists() or not Path(target_video_path).exists():
        return None

    txt1 = str(raw_row.get("txt1", "")).strip()
    txt2 = str(raw_row.get("txt2", "")).strip()
    edit = str(raw_row.get("edit", "")).strip()
    pair_seed_id = _stable_id(
        WEBVID_COVR_NAME,
        split,
        str(csv_path.relative_to(source_root).as_posix()),
        str(row_index),
        reference_video_path,
        target_video_path,
    )
    sim_txt = _score_or_text_value(raw_row.get("sim_txt"))
    sim_vid = _score_or_text_value(raw_row.get("sim_vid"))
    scores = _jsonish_value(raw_row.get("scores"))
    person_prob = _score_or_text_value(raw_row.get("person_prob", raw_row.get("person-prob")))

    pair_seed = {
        "pair_seed_id": pair_seed_id,
        "dataset": WEBVID_COVR_NAME,
        "split": split,
        "reference_video_path": reference_video_path,
        "target_video_path": target_video_path,
        "txt1": txt1,
        "txt2": txt2,
        "edit": edit,
        "sim_txt": sim_txt,
        "sim_vid": sim_vid,
        "scores": scores,
        "person_prob": person_prob,
        "source_file": str(csv_path),
        "row_index": row_index,
    }

    reference_row = _webvid_source_row(
        csv_path=csv_path,
        split=split,
        row_index=row_index,
        pair_seed_id=pair_seed_id,
        video_role="reference",
        video_path=reference_video_path,
        original_caption=txt1,
        original_edit=edit,
    )
    target_row = _webvid_source_row(
        csv_path=csv_path,
        split=split,
        row_index=row_index,
        pair_seed_id=pair_seed_id,
        video_role="target",
        video_path=target_video_path,
        original_caption=txt2,
        original_edit=edit,
    )
    return reference_row, target_row, pair_seed


def _webvid_source_row(
    *,
    csv_path: Path,
    split: str,
    row_index: int,
    pair_seed_id: str,
    video_role: str,
    video_path: str,
    original_caption: str,
    original_edit: str,
) -> dict[str, Any]:
    video_id = Path(video_path).stem
    return {
        "source_row_id": _stable_id(WEBVID_COVR_NAME, split, pair_seed_id, video_role, video_path),
        "dataset": WEBVID_COVR_NAME,
        "split": split,
        "row_index": row_index,
        "source_file": str(csv_path),
        "video_id": video_id,
        "video_path": video_path,
        "audio_path": _find_sibling_audio(Path(video_path)),
        "text_fields": {
            "original_caption": original_caption,
            "original_edit": original_edit,
            "video_role": video_role,
            "pair_seed_id": pair_seed_id,
        },
        "raw_columns": ["txt1", "txt2", "edit", "pth1", "pth2", "sim_txt", "sim_vid", "scores", "person_prob"],
    }


def extract_archives(source_root: Path) -> dict[str, int]:
    extracted_root = source_root / "_extracted"
    archive_count = 0
    extracted_count = 0
    for archive_path in sorted(source_root.rglob("*.zip")):
        if "_extracted" in archive_path.parts:
            continue
        archive_count += 1
        destination = extracted_root / archive_path.relative_to(source_root).with_suffix("")
        marker = destination / ".extract_complete"
        if marker.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        marker.write_text(str(archive_path), encoding="utf-8")
        extracted_count += 1
    return {"archive_count": archive_count, "extracted_count": extracted_count}


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
    candidates = [path] if path.is_absolute() else _relative_media_candidates(value, source_root)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _relative_media_candidates(value: str, source_root: Path) -> list[Path]:
    normalized = value.lstrip("./")
    candidates = [source_root / value, source_root / normalized]
    extracted_root = source_root / "_extracted"
    candidates.extend([extracted_root / value, extracted_root / normalized])
    file_name = Path(normalized).name
    if file_name:
        candidates.extend(sorted(source_root.rglob(file_name)))
        if extracted_root.exists():
            candidates.extend(sorted(extracted_root.rglob(file_name)))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


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


def _jsonish_value(value: Any) -> Any:
    normalized = _json_safe_text_value(value)
    if not isinstance(normalized, str):
        return normalized
    candidate = normalized.strip()
    if not candidate:
        return ""
    if candidate[:1] in {"{", "["}:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return candidate
    return candidate


def _score_or_text_value(value: Any) -> Any:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return round(float(text), 6)
    except ValueError:
        return text


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
                "splits": [],
            }
        by_video_path[video_path]["source_row_ids"].append(row["source_row_id"])
        split = str(row.get("split", "")).strip()
        if split and split not in by_video_path[video_path]["splits"]:
            by_video_path[video_path]["splits"].append(split)
        for key, value in row.get("text_fields", {}).items():
            by_video_path[video_path]["text_fields"].setdefault(key, value)
    for record in by_video_path.values():
        if len(record["splits"]) == 1:
            record["split"] = record["splits"][0]
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
            return _normalize_split_name(split)
    return "unknown"


def _normalize_split_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"valid", "val"}:
        return "validation"
    if normalized == "dev":
        return "test"
    return normalized or "unknown"


def _normalize_requested_splits(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        candidate = _normalize_split_name(value)
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized or ["train"]


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
    pair_seeds_path: Path,
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
        f"- webvid pair seeds path: `{pair_seeds_path}`",
        "",
        "## Dataset Counts",
        "",
        "| dataset | rows | clips | pair_seeds | missing_root | missing_video_seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name, counts in sorted(dataset_counts.items()):
        lines.append(
            "| "
            + f"{dataset_name} | {counts.get('rows', 0)} | {counts.get('clips', 0)} | "
            + f"{counts.get('pair_seeds', 0)} | {counts.get('missing_root', 0)} | {counts.get('missing_video_seeds', 0)} |"
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
    prepare.add_argument("--webvid-covr-root")
    prepare.add_argument("--webvid-covr-splits", nargs="+", default=["train"])
    prepare.add_argument("--clip-limit", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_source_datasets(
            root=args.root,
            daily_omni_root=args.daily_omni_root,
            worldsense_root=args.worldsense_root,
            webvid_covr_root=args.webvid_covr_root,
            webvid_covr_splits=args.webvid_covr_splits,
            clip_limit=args.clip_limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
