from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from app.audio_matters_natural import audio_anchor_score, extract_audio_feature
from app.composed_data import (
    AUDIO_MATTERS_ACCEPTANCE_PROFILE,
    SPEECH_AUDIO_CONTENT_LINE,
    VISUAL_AUDIO_ANCHOR_LINE,
    _build_proposal_id,
    _dedupe_strings,
    _display_path,
    _load_jsonl,
    _non_speech_audio_event_score,
    _normalize_list,
    _resolve_under_root,
    _score_float,
    _speech_evidence_score,
    _speech_is_transcript_backed,
    _speech_specificity_score,
    _speech_texts_from_annotation,
    _stable_hash,
    _write_jsonl,
    probe_media,
)


VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
VISUAL_LINE_TYPES = {"attribute", "object_presence", "object_count", "action", "scene"}


def prepare_existing_single_source_clips(
    *,
    root: str | Path,
    single_source_root: str | Path,
    run_root: str | Path,
    max_source_folders: int | None = None,
    annotation_search_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    source_root = Path(single_source_root)
    output_root = Path(run_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not source_root.exists():
        raise FileNotFoundError(f"single_source_root does not exist: {source_root}")

    folders = [path for path in sorted(source_root.iterdir(), key=lambda item: item.name) if path.is_dir()]
    if max_source_folders and max_source_folders > 0:
        folders = folders[:max_source_folders]

    annotation_index, annotation_sources = _build_annotation_reuse_index(
        root=root_path,
        search_roots=annotation_search_roots or [],
    )
    segments: list[dict[str, Any]] = []
    whole_records: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    reused_annotations: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    missing_annotation_manifest: list[dict[str, Any]] = []
    audio_refresh_manifest: list[dict[str, Any]] = []
    skipped_folders: list[dict[str, Any]] = []

    for folder_index, folder in enumerate(folders, start=1):
        media_files = [path for path in sorted(folder.iterdir(), key=lambda item: _clip_sort_key(item.name)) if path.suffix.lower() in VIDEO_SUFFIXES]
        whole = next((path for path in media_files if "whole" in path.stem.lower()), None)
        single_files = [path for path in media_files if path != whole and "single" in path.stem.lower()]
        if not single_files:
            single_files = [path for path in media_files if path != whole]
        if len(single_files) < 4:
            skipped_folders.append({"folder": str(folder), "reason": f"too_few_segments:{len(single_files)}"})
            continue

        dataset = _infer_dataset(folder)
        source_clip_id = _safe_source_id(folder.name)
        group_id = f"single_source_{source_clip_id}"
        candidate_clip_ids: list[str] = []
        for segment_index, video_path in enumerate(single_files, start=1):
            clip_id = video_path.stem
            start_seconds = _infer_segment_start(video_path.name, segment_index)
            media = probe_media(video_path)
            duration = _score_float(media.get("duration_seconds")) or 6.0
            record = {
                "clip_id": clip_id,
                "source_path": str(whole or folder),
                "output_path": _display_path(root_path, video_path),
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(start_seconds + duration, 3),
                "duration_seconds": round(duration, 3),
                "role": "single_source_segment_reused",
                "dataset": dataset,
                "source_clip_id": source_clip_id,
                "source_window_start_seconds": 0.0,
                "source_window_duration_seconds": round(max(30.0, len(single_files) * duration), 3),
                "relative_start_seconds": round(start_seconds, 3),
                "relative_end_seconds": round(start_seconds + duration, 3),
                "group_id": group_id,
                "media_probe": media,
                "reuse_source_folder": str(folder),
            }
            segments.append(record)
            candidate_clip_ids.append(clip_id)
            reused = _match_reused_annotation(annotation_index, root_path=root_path, clip_record=record)
            if reused is None:
                missing_annotation_manifest.append(record)
                reuse_rows.append({"clip_id": clip_id, "output_path": record["output_path"], "reused": False, "audio_fields_present": False})
            else:
                normalized = dict(reused["annotation"])
                normalized["clip_id"] = clip_id
                normalized["output_path"] = record["output_path"]
                normalized.setdefault("dataset", dataset)
                normalized["annotation_reused_from"] = reused["source"]
                normalized["annotation_reuse_key"] = reused["key"]
                reused_annotations.append(normalized)
                audio_present = _annotation_has_audio_fields(normalized)
                if not audio_present:
                    audio_refresh_manifest.append(record)
                reuse_rows.append(
                    {
                        "clip_id": clip_id,
                        "output_path": record["output_path"],
                        "reused": True,
                        "annotation_reused_from": reused["source"],
                        "annotation_reuse_key": reused["key"],
                        "audio_fields_present": audio_present,
                    }
                )
        groups.append(
            {
                "group_id": group_id,
                "dataset": dataset,
                "group_reason": "single_source_existing_folder",
                "source_clip_ids": [source_clip_id],
                "candidate_clip_ids": candidate_clip_ids,
                "group_tags": ["single_source", dataset, "existing_6s_segments", "audio_lines"],
                "source_path": str(whole or folder),
                "source_folder": str(folder),
            }
        )
        if whole is not None:
            whole_records.append(
                {
                    "clip_id": whole.stem,
                    "source_path": str(whole),
                    "output_path": _display_path(root_path, whole),
                    "start_seconds": 0.0,
                    "end_seconds": _score_float(probe_media(whole).get("duration_seconds")) or 30.0,
                    "duration_seconds": _score_float(probe_media(whole).get("duration_seconds")) or 30.0,
                    "role": "single_source_whole_video_reused",
                    "dataset": dataset,
                    "source_clip_id": source_clip_id,
                    "group_id": group_id,
                    "reuse_source_folder": str(folder),
                }
            )
        print(
            f"[audio-lines-prepare] folder {folder_index}/{len(folders)} {folder.name} segments={len(single_files)}",
            file=sys.stderr,
            flush=True,
        )

    paths = _run_paths(output_root)
    _write_jsonl(paths["segments"], segments)
    _write_jsonl(paths["whole"], whole_records)
    _write_jsonl(paths["groups"], groups)
    _write_jsonl(paths["annotations"], reused_annotations)
    _write_jsonl(paths["clips_to_annotate"], segments)
    _write_jsonl(paths["missing_annotation_manifest"], missing_annotation_manifest)
    _write_jsonl(paths["audio_refresh_manifest"], audio_refresh_manifest)
    _write_jsonl(paths["reuse_report_jsonl"], reuse_rows)

    summary = {
        "root": str(root_path),
        "single_source_root": str(source_root),
        "run_root": str(output_root),
        "source_folder_count": len(folders),
        "usable_group_count": len(groups),
        "skipped_folder_count": len(skipped_folders),
        "segment_count": len(segments),
        "whole_count": len(whole_records),
        "reused_annotation_count": len(reused_annotations),
        "missing_annotation_count": len(missing_annotation_manifest),
        "audio_refresh_needed_count": len(audio_refresh_manifest),
        "annotation_sources": annotation_sources,
        "outputs": {key: str(value) for key, value in paths.items()},
        "skipped_folders": skipped_folders[:50],
    }
    paths["reuse_report_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def merge_annotations(
    *,
    base_annotations_path: str | Path,
    refresh_annotations_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    base = list(_load_jsonl(Path(base_annotations_path)))
    refresh = list(_load_jsonl(Path(refresh_annotations_path))) if Path(refresh_annotations_path).exists() else []
    merged_by_id = {str(item.get("clip_id", "")).strip(): item for item in base if str(item.get("clip_id", "")).strip()}
    refreshed_count = 0
    for item in refresh:
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            continue
        item = dict(item)
        item["audio_refresh_annotation"] = True
        merged_by_id[clip_id] = item
        refreshed_count += 1
    ordered = [merged_by_id[str(item.get("clip_id", "")).strip()] for item in base if str(item.get("clip_id", "")).strip() in merged_by_id]
    seen = {str(item.get("clip_id", "")).strip() for item in ordered}
    ordered.extend(item for key, item in sorted(merged_by_id.items()) if key not in seen)
    _write_jsonl(Path(output_path), ordered)
    return {"base_count": len(base), "refresh_count": len(refresh), "refreshed_count": refreshed_count, "output_count": len(ordered), "output_path": str(output_path)}


def split_audio_line_candidates(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    pair_candidates_path: str | Path,
    a_output_path: str | Path,
    b_output_path: str | Path,
    summary_path: str | Path,
    min_audio_anchor_score: float = 0.86,
    max_a_candidates: int | None = None,
    max_b_candidates: int | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    candidates = list(_load_jsonl(Path(pair_candidates_path)))
    annotations_by_id = {str(item.get("clip_id", "")).strip(): item for item in annotations if str(item.get("clip_id", "")).strip()}
    audio_features: dict[str, Any] = {}
    a_records: list[dict[str, Any]] = []
    b_records: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()

    for index, candidate in enumerate(candidates, start=1):
        reference = annotations_by_id.get(str(candidate.get("reference_clip_id", "")).strip())
        target = annotations_by_id.get(str(candidate.get("target_clip_id", "")).strip())
        if reference is None or target is None:
            reject_counts["missing_annotation"] += 1
            continue
        difference = candidate.get("difference") if isinstance(candidate.get("difference"), dict) else {}
        difference_type = str(difference.get("type", "")).strip()
        if difference_type in VISUAL_LINE_TYPES:
            score, min_rms = _pair_audio_anchor_score(root_path, reference, target, audio_features)
            if score >= min_audio_anchor_score:
                a_records.append(_line_candidate(candidate, VISUAL_AUDIO_ANCHOR_LINE, score=score, min_rms=min_rms))
            else:
                reject_counts["a_audio_anchor_below_threshold"] += 1
        if _speech_is_transcript_backed(reference, target):
            b_records.append(_speech_line_candidate(candidate, reference, target))
        else:
            non_speech_score = _non_speech_audio_event_score(reference, target)
            if non_speech_score >= 0.45:
                b_records.append(_audio_event_line_candidate(candidate, reference, target, non_speech_score))
            else:
                reject_counts["b_missing_audio_evidence"] += 1
        if index % 50 == 0:
            print(f"[audio-lines-split] processed {index}/{len(candidates)}", file=sys.stderr, flush=True)

    a_records = sorted(a_records, key=_line_candidate_sort_key, reverse=True)
    b_records = sorted(b_records, key=_line_candidate_sort_key, reverse=True)
    if max_a_candidates and max_a_candidates > 0:
        a_records = a_records[:max_a_candidates]
    if max_b_candidates and max_b_candidates > 0:
        b_records = b_records[:max_b_candidates]
    for idx, record in enumerate(a_records, start=1):
        record["candidate_index"] = idx
    for idx, record in enumerate(b_records, start=1):
        record["candidate_index"] = idx
    _write_jsonl(Path(a_output_path), a_records)
    _write_jsonl(Path(b_output_path), b_records)
    summary = {
        "candidate_count": len(candidates),
        "a_candidate_count": len(a_records),
        "b_candidate_count": len(b_records),
        "min_audio_anchor_score": min_audio_anchor_score,
        "reject_counts": dict(reject_counts),
        "a_output_path": str(a_output_path),
        "b_output_path": str(b_output_path),
    }
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def shard_jsonl(*, input_path: str | Path, output_dir: str | Path, shards: int, prefix: str) -> dict[str, Any]:
    records = list(_load_jsonl(Path(input_path)))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shards = max(1, int(shards or 1))
    shard_paths: list[str] = []
    buckets = [[] for _ in range(shards)]
    for index, record in enumerate(records):
        buckets[index % shards].append(record)
    for shard_index, bucket in enumerate(buckets, start=1):
        path = output_root / f"{prefix}_shard_{shard_index:02d}.jsonl"
        _write_jsonl(path, bucket)
        shard_paths.append(str(path))
    manifest = {"input_path": str(input_path), "record_count": len(records), "shards": shards, "shard_paths": shard_paths}
    (output_root / f"{prefix}_shards.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def merge_line_results(
    *,
    run_root: str | Path,
    target_a_count: int = 8,
    target_b_count: int = 8,
) -> dict[str, Any]:
    root = Path(run_root)
    a_ranked = _load_many_jsonl(root / "a_shards", "ranked_*.jsonl")
    b_ranked = _load_many_jsonl(root / "b_shards", "ranked_*.jsonl")
    a_accepted_all = [record for record in a_ranked if bool(record.get("accepted"))]
    b_accepted_all = [record for record in b_ranked if bool(record.get("accepted"))]
    a_selected = a_accepted_all[: max(0, target_a_count)]
    b_selected = b_accepted_all[: max(0, target_b_count)]
    _write_jsonl(root / "a_ranked_single_source_pairs.jsonl", a_ranked)
    _write_jsonl(root / "b_ranked_single_source_pairs.jsonl", b_ranked)
    _write_jsonl(root / "a_visual_audio_anchor_triplets.jsonl", a_selected)
    _write_jsonl(root / "b_speech_audio_content_triplets.jsonl", b_selected)
    _write_jsonl(root / "a_accepted.progress.jsonl", _load_many_jsonl(root / "a_shards", "accepted_progress_*.jsonl"))
    _write_jsonl(root / "a_rejected.progress.jsonl", _load_many_jsonl(root / "a_shards", "rejected_progress_*.jsonl"))
    _write_jsonl(root / "b_accepted.progress.jsonl", _load_many_jsonl(root / "b_shards", "accepted_progress_*.jsonl"))
    _write_jsonl(root / "b_rejected.progress.jsonl", _load_many_jsonl(root / "b_shards", "rejected_progress_*.jsonl"))
    summary = {
        "run_root": str(root),
        "a_ranked_count": len(a_ranked),
        "b_ranked_count": len(b_ranked),
        "a_accepted_count": len(a_accepted_all),
        "b_accepted_count": len(b_accepted_all),
        "a_exported_count": len(a_selected),
        "b_exported_count": len(b_selected),
        "target_a_count": target_a_count,
        "target_b_count": target_b_count,
        "a_reject_reason_counts": _reject_reason_counts(a_ranked),
        "b_reject_reason_counts": _reject_reason_counts(b_ranked),
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_paths(root: Path) -> dict[str, Path]:
    return {
        "segments": root / "extracted_single_source_clips.jsonl",
        "whole": root / "extracted_single_source_whole.jsonl",
        "groups": root / "single_source_clip_groups.jsonl",
        "annotations": root / "single_source_annotations.jsonl",
        "clips_to_annotate": root / "clips_to_annotate.jsonl",
        "missing_annotation_manifest": root / "missing_annotation_clips.jsonl",
        "audio_refresh_manifest": root / "audio_refresh_clips.jsonl",
        "reuse_report_jsonl": root / "annotation_reuse_report.jsonl",
        "reuse_report_json": root / "annotation_reuse_report.json",
    }


def _build_annotation_reuse_index(*, root: Path, search_roots: list[str | Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    index: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    filenames = {"single_source_annotations.jsonl", "single_source_whole_annotation.jsonl", "detective_annotations.jsonl", "clip_annotations.jsonl"}
    for search_root in search_roots:
        base = Path(search_root)
        if not base.exists():
            continue
        for path in base.rglob("*.jsonl"):
            if path.name not in filenames:
                continue
            sources.append(str(path))
            for annotation in _load_jsonl(path):
                wrapped = {"annotation": annotation, "source": str(path)}
                for key in _annotation_keys(root=root, annotation=annotation):
                    index.setdefault(key, {**wrapped, "key": key})
    return index, sources


def _annotation_keys(*, root: Path, annotation: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    clip_id = str(annotation.get("clip_id", "")).strip()
    output_path = str(annotation.get("output_path", "")).strip()
    if clip_id:
        keys.append(f"clip_id:{clip_id}")
    if output_path:
        path = _resolve_under_root(root, output_path)
        keys.extend([f"path:{str(path).lower()}", f"name:{path.name.lower()}", f"stem:{path.stem.lower()}"])
    return _dedupe_strings(keys)


def _match_reused_annotation(index: dict[str, dict[str, Any]], *, root_path: Path, clip_record: dict[str, Any]) -> dict[str, Any] | None:
    fake = {"clip_id": clip_record.get("clip_id", ""), "output_path": clip_record.get("output_path", "")}
    for key in _annotation_keys(root=root_path, annotation=fake):
        if key in index:
            return index[key]
    return None


def _annotation_has_audio_fields(annotation: dict[str, Any]) -> bool:
    return bool(
        _normalize_list(annotation.get("speech", []))
        or _normalize_list(annotation.get("speakers_and_transcript", []))
        or _normalize_list(annotation.get("audio_events", []))
        or "audio" in {str(item).lower() for item in _normalize_list(annotation.get("modalities", []))}
    )


def _clip_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(?:single|segment|clip)[_-]?(\d+)", name, re.IGNORECASE)
    if match:
        return (int(match.group(1)), name)
    numbers = re.findall(r"\d+", name)
    return (int(numbers[-1]) if numbers else 9999, name)


def _infer_segment_start(name: str, segment_index: int) -> float:
    return float(max(0, segment_index - 1) * 6)


def _infer_dataset(path: Path) -> str:
    text = str(path).lower()
    if "worldsense" in text:
        return "worldsense"
    if "daily" in text:
        return "daily_omni"
    return "unknown"


def _safe_source_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or _stable_hash(value)[:12]


def _pair_audio_anchor_score(root: Path, reference: dict[str, Any], target: dict[str, Any], cache: dict[str, Any]) -> tuple[float, float]:
    features = []
    for annotation in (reference, target):
        clip_id = str(annotation.get("clip_id", "")).strip()
        if clip_id not in cache:
            cache[clip_id] = extract_audio_feature(_resolve_under_root(root, str(annotation.get("output_path", ""))))
        features.append(cache[clip_id])
    if not features[0] or not features[1]:
        return 0.0, 0.0
    return audio_anchor_score(features[0], features[1]), min(float(features[0].rms), float(features[1].rms))


def _line_candidate(candidate: dict[str, Any], line: str, *, score: float = 0.0, min_rms: float = 0.0) -> dict[str, Any]:
    record = dict(candidate)
    quality = dict(record.get("quality", {})) if isinstance(record.get("quality"), dict) else {}
    quality.update(
        {
            "audio_dataset_line": line,
            "audio_anchor_required": 1.0 if line == VISUAL_AUDIO_ANCHOR_LINE else quality.get("audio_anchor_required", 0.0),
            "audio_anchor_score": round(score, 4) if line == VISUAL_AUDIO_ANCHOR_LINE else quality.get("audio_anchor_score", 0.0),
            "audio_anchor_min_rms": round(min_rms, 6) if line == VISUAL_AUDIO_ANCHOR_LINE else quality.get("audio_anchor_min_rms", 0.0),
            "audio_anchor_type": "same_source_similar_audio" if line == VISUAL_AUDIO_ANCHOR_LINE else quality.get("audio_anchor_type", ""),
            "edit_primary_modality": "visual" if line == VISUAL_AUDIO_ANCHOR_LINE else "audio",
        }
    )
    record["quality"] = quality
    record["audio_dataset_line"] = line
    record["risk_flags"] = _dedupe_strings(_normalize_list(record.get("risk_flags", [])) + [line])
    record["proposal_id"] = f"{line}_{record.get('proposal_id') or _build_proposal_id(str(record.get('reference_video', '')), str(record.get('target_video', '')))}"
    return record


def _speech_line_candidate(candidate: dict[str, Any], reference: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    record = _line_candidate(candidate, SPEECH_AUDIO_CONTENT_LINE)
    reference_speech = "; ".join(_speech_texts_from_annotation(reference)[:2]) or "reference speech"
    target_speech = "; ".join(_speech_texts_from_annotation(target)[:2]) or "target speech"
    record["difference"] = {
        "type": "speech",
        "from": reference_speech[:180],
        "to": target_speech[:180],
        "description": "the spoken-language content differs between the reference and target clips",
    }
    quality = dict(record.get("quality", {}))
    quality.update(
        {
            "difference_type": "speech",
            "has_audio_modality": 1.0,
            "speech_evidence_score": _speech_evidence_score(reference, target),
            "speech_specificity_score": _speech_specificity_score(reference, target),
            "speech_transcript_backed": 1.0,
        }
    )
    record["quality"] = quality
    return record


def _audio_event_line_candidate(candidate: dict[str, Any], reference: dict[str, Any], target: dict[str, Any], score: float) -> dict[str, Any]:
    record = _line_candidate(candidate, SPEECH_AUDIO_CONTENT_LINE)
    record["difference"] = {
        "type": "audio_event",
        "from": "; ".join(_normalize_list(reference.get("audio_events", []))[:2]) or "reference audio",
        "to": "; ".join(_normalize_list(target.get("audio_events", []))[:2]) or "target audio",
        "description": "the non-speech audio event differs between the reference and target clips",
    }
    quality = dict(record.get("quality", {}))
    quality.update({"difference_type": "audio_event", "has_audio_modality": 1.0, "non_speech_audio_event_score": round(score, 3)})
    record["quality"] = quality
    return record


def _line_candidate_sort_key(record: dict[str, Any]) -> tuple[float, float, str]:
    quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
    return (
        _score_float(quality.get("audio_anchor_score", quality.get("speech_evidence_score", quality.get("non_speech_audio_event_score", 0.0)))),
        _score_float(record.get("composite_score")),
        str(record.get("proposal_id", "")),
    )


def _load_many_jsonl(root: Path, pattern: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(root.glob(pattern)):
        records.extend(_load_jsonl(path))
    return records


def _reject_reason_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        if bool(record.get("accepted")):
            continue
        judge = record.get("judge", {}) if isinstance(record.get("judge"), dict) else {}
        reason = str(judge.get("reject_reason", "")).strip() or "unknown"
        counter[reason[:180]] += 1
    return dict(counter.most_common(20))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build A/B audio dataset pilots from existing single-source clips.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-existing")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--single-source-root", required=True)
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--max-source-folders", type=int)
    prepare.add_argument("--annotation-search-root", action="append", default=[])

    merge_ann = subparsers.add_parser("merge-annotations")
    merge_ann.add_argument("--base-annotations-path", required=True)
    merge_ann.add_argument("--refresh-annotations-path", required=True)
    merge_ann.add_argument("--output-path", required=True)

    split = subparsers.add_parser("split-candidates")
    split.add_argument("--root", required=True)
    split.add_argument("--clip-annotations-path", required=True)
    split.add_argument("--pair-candidates-path", required=True)
    split.add_argument("--a-output-path", required=True)
    split.add_argument("--b-output-path", required=True)
    split.add_argument("--summary-path", required=True)
    split.add_argument("--min-audio-anchor-score", type=float, default=0.86)
    split.add_argument("--max-a-candidates", type=int)
    split.add_argument("--max-b-candidates", type=int)

    shard = subparsers.add_parser("shard-jsonl")
    shard.add_argument("--input-path", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--shards", type=int, default=1)
    shard.add_argument("--prefix", required=True)

    merge = subparsers.add_parser("merge-line-results")
    merge.add_argument("--run-root", required=True)
    merge.add_argument("--target-a-count", type=int, default=8)
    merge.add_argument("--target-b-count", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-existing":
        result = prepare_existing_single_source_clips(
            root=args.root,
            single_source_root=args.single_source_root,
            run_root=args.run_root,
            max_source_folders=args.max_source_folders,
            annotation_search_roots=args.annotation_search_root,
        )
    elif args.command == "merge-annotations":
        result = merge_annotations(
            base_annotations_path=args.base_annotations_path,
            refresh_annotations_path=args.refresh_annotations_path,
            output_path=args.output_path,
        )
    elif args.command == "split-candidates":
        result = split_audio_line_candidates(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            pair_candidates_path=args.pair_candidates_path,
            a_output_path=args.a_output_path,
            b_output_path=args.b_output_path,
            summary_path=args.summary_path,
            min_audio_anchor_score=args.min_audio_anchor_score,
            max_a_candidates=args.max_a_candidates,
            max_b_candidates=args.max_b_candidates,
        )
    elif args.command == "shard-jsonl":
        result = shard_jsonl(input_path=args.input_path, output_dir=args.output_dir, shards=args.shards, prefix=args.prefix)
    elif args.command == "merge-line-results":
        result = merge_line_results(run_root=args.run_root, target_a_count=args.target_a_count, target_b_count=args.target_b_count)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
