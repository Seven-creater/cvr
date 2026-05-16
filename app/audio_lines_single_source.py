from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.audio_matters_natural import audio_anchor_score, extract_audio_feature
from app.composed_data import (
    AUDIO_MATTERS_ACCEPTANCE_PROFILE,
    SPEECH_AUDIO_CONTENT_LINE,
    VISUAL_AUDIO_ANCHOR_LINE,
    _append_jsonl_record,
    _build_proposal_id,
    _boolish,
    _call_omni_with_retries,
    _dedupe_strings,
    _display_path,
    _extract_audio_only_cache,
    _extract_video_only_cache,
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
from app.composed_omni import OpenAIComposedDataClient


VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
VISUAL_LINE_TYPES = {"attribute", "object_presence", "object_count", "action", "scene"}
AUDIO_LINE_PROFILE_DEFAULT = "default"
AUDIO_LINE_PROFILE_V4_STRICT = "v4_strict"
AUDIO_LINE_PROFILE_V5_AUDIO_PRIMARY = "v5_audio_primary"
AUDIO_LINE_PROFILE_B_CONTEXT_CVR = "b_audio_context_cvr"
AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW = "b_audio_blind_review"
AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2 = "b_audio_blind_review_v2"
AUDIO_LINE_PROFILE_ALIASES = {
    "b_context_cvr": AUDIO_LINE_PROFILE_B_CONTEXT_CVR,
    "b_audio_blind": AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW,
    "b_blind_review": AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW,
    "b_audio_blind_v2": AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2,
    "b_blind_review_v2": AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2,
}
AUDIO_LINE_PROFILES = {
    AUDIO_LINE_PROFILE_DEFAULT,
    AUDIO_LINE_PROFILE_V4_STRICT,
    AUDIO_LINE_PROFILE_V5_AUDIO_PRIMARY,
    AUDIO_LINE_PROFILE_B_CONTEXT_CVR,
    AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW,
    AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2,
}
B_CANDIDATE_MODE_HYBRID = "hybrid"
B_CANDIDATE_MODE_AUDIO_FIRST = "audio_first"
B_CANDIDATE_MODES = {B_CANDIDATE_MODE_HYBRID, B_CANDIDATE_MODE_AUDIO_FIRST}
A_CANDIDATE_MODE_HYBRID = "hybrid"
A_CANDIDATE_MODE_OMNI_FIRST = "omni_first"
A_CANDIDATE_MODES = {A_CANDIDATE_MODE_HYBRID, A_CANDIDATE_MODE_OMNI_FIRST}
V4_A_STRONG_VISUAL_TYPES = {"scene", "action", "object_presence"}
V4_VAGUE_AUDIO_TERMS = {
    "buzz",
    "buzzing",
    "click",
    "clicking",
    "electronic tone",
    "electronic hum",
    "hum",
    "humming",
    "low frequency",
    "low-frequency",
    "tone",
}
V4_CONCRETE_AUDIO_TERMS = {
    "applause",
    "cheer",
    "cheering",
    "chant",
    "crowd",
    "music",
    "song",
    "whistle",
    "siren",
    "bell",
    "rain",
    "water",
    "wind",
    "engine",
    "machinery",
    "footstep",
    "footsteps",
}
V4_B_MIN_VISUAL_CONTEXT_SIMILARITY = 0.30
V4_B_MAX_VISUAL_DELTA_STRENGTH = 0.55


def prepare_existing_single_source_clips(
    *,
    root: str | Path,
    single_source_root: str | Path,
    run_root: str | Path,
    max_source_folders: int | None = None,
    max_clips: int | None = None,
    annotation_search_roots: list[str | Path] | None = None,
    force_audio_focused_refresh: bool = False,
    reuse_annotations: bool = True,
    min_clips_per_folder: int = 4,
) -> dict[str, Any]:
    root_path = Path(root)
    source_root = Path(single_source_root)
    output_root = Path(run_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not source_root.exists():
        raise FileNotFoundError(f"single_source_root does not exist: {source_root}")
    min_clips_per_folder = max(1, int(min_clips_per_folder or 1))

    folders = [path for path in sorted(source_root.iterdir(), key=lambda item: item.name) if path.is_dir()]
    if max_source_folders and max_source_folders > 0:
        folders = folders[:max_source_folders]

    annotation_index, annotation_sources = ({}, [])
    if reuse_annotations:
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
        if max_clips and max_clips > 0 and len(segments) >= max_clips:
            break
        media_files = [path for path in sorted(folder.iterdir(), key=lambda item: _clip_sort_key(item.name)) if path.suffix.lower() in VIDEO_SUFFIXES]
        whole = next((path for path in media_files if "whole" in path.stem.lower()), None)
        single_files = [path for path in media_files if path != whole and "single" in path.stem.lower()]
        if not single_files:
            single_files = [path for path in media_files if path != whole]
        if len(single_files) < min_clips_per_folder:
            skipped_folders.append({"folder": str(folder), "reason": f"too_few_segments:{len(single_files)}"})
            continue
        if max_clips and max_clips > 0:
            remaining = max_clips - len(segments)
            if remaining < min_clips_per_folder:
                break
            single_files = single_files[:remaining]

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
                if force_audio_focused_refresh or not audio_present:
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
        "force_audio_focused_refresh": bool(force_audio_focused_refresh),
        "reuse_annotations": bool(reuse_annotations),
        "min_clips_per_folder": min_clips_per_folder,
        "max_clips": int(max_clips or 0),
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
    audio_line_quality_profile: str = AUDIO_LINE_PROFILE_DEFAULT,
    a_candidate_mode: str = A_CANDIDATE_MODE_HYBRID,
    b_candidate_mode: str = B_CANDIDATE_MODE_HYBRID,
) -> dict[str, Any]:
    audio_line_quality_profile = _normalize_audio_line_quality_profile(audio_line_quality_profile)
    a_candidate_mode = _normalize_a_candidate_mode(a_candidate_mode)
    b_candidate_mode = _normalize_b_candidate_mode(b_candidate_mode)
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
        visual_delta_strength = _visual_delta_strength(candidate, reference, target)
        visual_context_similarity = _visual_context_similarity(reference, target)
        video_context_type = _b_context_type(reference, target)
        video_context_strength = _b_context_strength(reference, target, visual_context_similarity)
        asr_degeneracy_risk = _b_asr_degeneracy_risk(reference, target)
        should_score_a = difference_type in VISUAL_LINE_TYPES or a_candidate_mode == A_CANDIDATE_MODE_OMNI_FIRST
        if should_score_a:
            score, min_rms = _pair_audio_anchor_score(root_path, reference, target, audio_features)
            if a_candidate_mode == A_CANDIDATE_MODE_OMNI_FIRST:
                a_gate_passed = score >= min_audio_anchor_score
            else:
                a_gate_passed = _v4_a_candidate_allowed(
                    audio_line_quality_profile=audio_line_quality_profile,
                    difference_type=difference_type,
                    visual_delta_strength=visual_delta_strength,
                )
            if score >= min_audio_anchor_score and a_gate_passed:
                a_records.append(
                    _line_candidate(
                        candidate,
                        VISUAL_AUDIO_ANCHOR_LINE,
                        score=score,
                        min_rms=min_rms,
                        visual_delta_strength=visual_delta_strength,
                        visual_context_similarity=visual_context_similarity,
                        audio_line_quality_profile=audio_line_quality_profile,
                    )
                )
            else:
                if score < min_audio_anchor_score:
                    reject_counts["a_audio_anchor_below_threshold"] += 1
                elif difference_type not in VISUAL_LINE_TYPES:
                    reject_counts["a_non_visual_hint_rejected"] += 1
                else:
                    reject_counts["a_v4_visual_delta_too_weak"] += 1
        if b_candidate_mode == B_CANDIDATE_MODE_HYBRID:
            speech_score = _speech_evidence_score(reference, target)
            if _uses_audio_primary_mining_profile(audio_line_quality_profile):
                speech_score = max(speech_score, _speech_content_delta_score(reference, target))
                speech_threshold = 0.45
            else:
                speech_threshold = 0.70
            b_context_ok = (
                audio_line_quality_profile
                not in {AUDIO_LINE_PROFILE_B_CONTEXT_CVR, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2}
                or _b_context_candidate_allowed(
                    difference_type="speech",
                    video_context_strength=video_context_strength,
                    asr_degeneracy_risk=asr_degeneracy_risk,
                )
            )
            if _speech_is_transcript_backed(reference, target) and speech_score >= speech_threshold and b_context_ok and _v4_b_candidate_allowed(
                audio_line_quality_profile=audio_line_quality_profile,
                visual_delta_strength=visual_delta_strength,
                visual_context_similarity=visual_context_similarity,
                audio_text=" ".join(_speech_texts_from_annotation(reference)[:2] + _speech_texts_from_annotation(target)[:2]),
                difference_type="speech",
            ):
                b_records.append(
                    _speech_line_candidate(
                        candidate,
                        reference,
                        target,
                        visual_delta_strength=visual_delta_strength,
                        visual_context_similarity=visual_context_similarity,
                        audio_line_quality_profile=audio_line_quality_profile,
                        video_context_type=video_context_type,
                        video_context_strength=video_context_strength,
                        asr_degeneracy_risk=asr_degeneracy_risk,
                    )
                )
            else:
                non_speech_score = _non_speech_audio_event_score(reference, target)
                non_speech_threshold = 0.55 if _uses_audio_primary_mining_profile(audio_line_quality_profile) else 0.70
                audio_text = " ".join(_normalize_list(reference.get("audio_events", [])) + _normalize_list(target.get("audio_events", [])))
                event_context_ok = (
                    audio_line_quality_profile
                    not in {AUDIO_LINE_PROFILE_B_CONTEXT_CVR, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2}
                    or _b_context_candidate_allowed(
                        difference_type="audio_event",
                        video_context_strength=video_context_strength,
                        asr_degeneracy_risk=asr_degeneracy_risk,
                    )
                )
                if non_speech_score >= non_speech_threshold and event_context_ok and _v4_b_candidate_allowed(
                    audio_line_quality_profile=audio_line_quality_profile,
                    visual_delta_strength=visual_delta_strength,
                    visual_context_similarity=visual_context_similarity,
                    audio_text=audio_text,
                    difference_type="audio_event",
                ):
                    b_records.append(
                        _audio_event_line_candidate(
                            candidate,
                            reference,
                            target,
                            non_speech_score,
                            visual_delta_strength=visual_delta_strength,
                            visual_context_similarity=visual_context_similarity,
                            audio_line_quality_profile=audio_line_quality_profile,
                            video_context_type=video_context_type,
                            video_context_strength=video_context_strength,
                            asr_degeneracy_risk=asr_degeneracy_risk,
                        )
                    )
                else:
                    if non_speech_score < non_speech_threshold:
                        reject_counts["b_missing_audio_evidence"] += 1
                    elif (
                        audio_line_quality_profile
                        in {AUDIO_LINE_PROFILE_B_CONTEXT_CVR, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2}
                        and not event_context_ok
                    ):
                        reject_counts["b_context_cvr_gate_failed"] += 1
                    else:
                        reject_counts["b_v4_visual_or_audio_gate_failed"] += 1
        if index % 50 == 0:
            print(f"[audio-lines-split] processed {index}/{len(candidates)}", file=sys.stderr, flush=True)

    existing_b_keys = {_b_pair_key(record) for record in b_records}
    direct_b_records, direct_b_rejects = _mine_audio_first_b_candidates(
        annotations_by_id=annotations_by_id,
        existing_keys=existing_b_keys,
        audio_line_quality_profile=audio_line_quality_profile,
    )
    b_records.extend(direct_b_records)
    reject_counts.update(direct_b_rejects)

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
        "b_audio_first_candidate_count": len(direct_b_records),
        "min_audio_anchor_score": min_audio_anchor_score,
        "audio_line_quality_profile": audio_line_quality_profile,
        "a_candidate_mode": a_candidate_mode,
        "b_candidate_mode": b_candidate_mode,
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
    keep_all_b: bool = False,
) -> dict[str, Any]:
    root = Path(run_root)
    a_accepted_progress = _load_many_jsonl(root / "a_shards", "accepted_progress_*.jsonl")
    a_rejected_progress = _load_many_jsonl(root / "a_shards", "rejected_progress_*.jsonl")
    b_accepted_progress = _load_many_jsonl(root / "b_shards", "accepted_progress_*.jsonl")
    b_rejected_progress = _load_many_jsonl(root / "b_shards", "rejected_progress_*.jsonl")
    a_ranked = _dedupe_line_records(_load_many_jsonl(root / "a_shards", "ranked_*.jsonl") + a_accepted_progress + a_rejected_progress)
    b_ranked = _dedupe_line_records(_load_many_jsonl(root / "b_shards", "ranked_*.jsonl") + b_accepted_progress + b_rejected_progress)
    a_accepted_all = [record for record in a_ranked if bool(record.get("accepted"))]
    b_accepted_all = _assign_b_line_tiers([record for record in b_ranked if bool(record.get("accepted"))])
    a_selected = a_accepted_all[: max(0, target_a_count)]
    b_selected = b_accepted_all if keep_all_b else _select_b_line_records(b_accepted_all, target_b_count=target_b_count)
    b_main = [record for record in b_accepted_all if record.get("split_tier") == "main"]
    b_extended = [record for record in b_accepted_all if record.get("split_tier") == "extended"]
    b_diagnostic = [record for record in b_accepted_all if record.get("split_tier") == "diagnostic"]
    b_speech = [record for record in b_accepted_all if _b_line_record_subtype(record) == "speech_topic_in_video_context"]
    b_music = [record for record in b_accepted_all if _b_line_record_subtype(record) == "music"]
    b_sound = [record for record in b_accepted_all if _b_line_record_subtype(record) == "sound_event"]
    _write_jsonl(root / "a_ranked_single_source_pairs.jsonl", a_ranked)
    _write_jsonl(root / "b_ranked_single_source_pairs.jsonl", b_ranked)
    _write_jsonl(root / "a_visual_audio_anchor_triplets.jsonl", a_selected)
    _write_jsonl(root / "b_speech_audio_content_triplets.jsonl", b_accepted_all)
    _write_jsonl(root / "b_all_audio_cvr_triplets.jsonl", b_accepted_all)
    _write_jsonl(root / "b_main_audio_cvr_triplets.jsonl", b_main)
    _write_jsonl(root / "b_extended_audio_cvr_triplets.jsonl", b_extended)
    _write_jsonl(root / "b_diagnostic_asr_risk_triplets.jsonl", b_diagnostic)
    _write_jsonl(root / "b_speech_context_triplets.jsonl", b_speech)
    _write_jsonl(root / "b_music_triplets.jsonl", b_music)
    _write_jsonl(root / "b_sound_event_triplets.jsonl", b_sound)
    _write_jsonl(root / "a_accepted.progress.jsonl", a_accepted_progress)
    _write_jsonl(root / "a_rejected.progress.jsonl", a_rejected_progress)
    _write_jsonl(root / "b_accepted.progress.jsonl", b_accepted_progress)
    _write_jsonl(root / "b_rejected.progress.jsonl", b_rejected_progress)
    summary = {
        "run_root": str(root),
        "a_ranked_count": len(a_ranked),
        "b_ranked_count": len(b_ranked),
        "a_accepted_count": len(a_accepted_all),
        "b_accepted_count": len(b_accepted_all),
        "a_exported_count": len(a_selected),
        "b_exported_count": len(b_accepted_all),
        "b_selected_count": len(b_selected),
        "b_main_count": len(b_main),
        "b_extended_count": len(b_extended),
        "b_diagnostic_count": len(b_diagnostic),
        "b_split_tier_counts": dict(Counter(str(record.get("split_tier") or "unknown") for record in b_accepted_all)),
        "b_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_accepted_all)),
        "b_exported_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_accepted_all)),
        "b_main_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_main)),
        "b_extended_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_extended)),
        "b_diagnostic_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_diagnostic)),
        "b_main_speech_ratio": _ratio(
            sum(1 for record in b_main if _b_line_record_subtype(record) == "speech_topic_in_video_context"),
            len(b_main),
        ),
        "b_context_cvr_summary_path": str(root / "b_context_cvr_summary.json"),
        "target_a_count": target_a_count,
        "target_b_count": target_b_count,
        "keep_all_b": bool(keep_all_b),
        "a_reject_reason_counts": _reject_reason_counts(a_ranked),
        "b_reject_reason_counts": _reject_reason_counts(b_ranked),
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "b_context_cvr_summary.json").write_text(
        json.dumps(_b_context_cvr_summary(root=root, b_ranked=b_ranked, b_selected=b_accepted_all), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return summary


def augment_b_inverse(
    *,
    run_root: str | Path,
    input_path: str | Path | None = None,
    root: str | Path | None = None,
    max_records: int | None = None,
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float = 180.0,
    omni_retries: int = 2,
    fail_on_transient_omni_errors: bool = False,
) -> dict[str, Any]:
    run_root_path = Path(run_root)
    root_path = _infer_cvr_root(run_root_path, root)
    input_file = Path(input_path) if input_path else run_root_path / "b_main_audio_cvr_triplets.jsonl"
    records = [record for record in _load_jsonl(input_file) if bool(record.get("accepted", True))]
    if max_records and max_records > 0:
        records = records[:max_records]

    candidates_path = run_root_path / "b_inverse_candidates.jsonl"
    accepted_path = run_root_path / "b_inverse_accepted.jsonl"
    rejected_path = run_root_path / "b_inverse_rejected.jsonl"
    train_path = run_root_path / "b_train_bidirectional_triplets.jsonl"
    for path in (candidates_path, accepted_path, rejected_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        inverse_edit = _inverse_b_line_edit_text(str(record.get("edit_text") or record.get("audio_only_edit_text") or ""))
        candidate = _build_inverse_candidate_record(record, inverse_edit=inverse_edit, index=index)
        candidates.append(candidate)
        _append_jsonl_record(candidates_path, candidate)
        if not inverse_edit.get("ok"):
            candidate["inverse_accept"] = False
            candidate["accepted"] = False
            candidate["inverse_reject_reason"] = inverse_edit.get("reject_reason", "inverse_edit_not_parseable")
            rejected.append(candidate)
            _append_jsonl_record(rejected_path, candidate)
            print(
                f"[augment-b-inverse] {index}/{len(records)} rejected parse proposal_id={record.get('proposal_id', '')} "
                f"reason={candidate['inverse_reject_reason']}",
                file=sys.stderr,
                flush=True,
            )
            continue

        try:
            checked = _verify_inverse_candidate(
                candidate,
                root=root_path,
                run_root=run_root_path,
                client=client,
                omni_retries=omni_retries,
                fail_on_transient_omni_errors=fail_on_transient_omni_errors,
            )
        except Exception as exc:
            if fail_on_transient_omni_errors:
                raise
            checked = dict(candidate)
            checked["inverse_accept"] = False
            checked["accepted"] = False
            checked["inverse_reject_reason"] = f"inverse_verification_error: {type(exc).__name__}: {exc}"

        if bool(checked.get("inverse_accept")):
            accepted.append(checked)
            _append_jsonl_record(accepted_path, checked)
        else:
            rejected.append(checked)
            _append_jsonl_record(rejected_path, checked)
        print(
            f"[augment-b-inverse] {index}/{len(records)} accepted={bool(checked.get('inverse_accept'))} "
            f"proposal_id={checked.get('proposal_id', '')} reason={checked.get('inverse_reject_reason', '')}",
            file=sys.stderr,
            flush=True,
        )

    forward_train = [_forward_train_record(record) for record in records]
    _write_jsonl(train_path, forward_train + accepted)
    summary = {
        "run_root": str(run_root_path),
        "input_path": str(input_file),
        "root": str(root_path),
        "input_count": len(records),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "outputs": {
            "candidates": str(candidates_path),
            "accepted": str(accepted_path),
            "rejected": str(rejected_path),
            "train_bidirectional": str(train_path),
        },
        "reject_reason_counts": dict(Counter(str(record.get("inverse_reject_reason") or "unknown")[:180] for record in rejected)),
    }
    (run_root_path / "b_inverse_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _infer_cvr_root(run_root: Path, explicit_root: str | Path | None) -> Path:
    if explicit_root:
        return Path(explicit_root)
    for path in (run_root / "annotation_reuse_report.json", run_root / "summary.json"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for key in ("root",):
            value = str(payload.get(key) or "").strip()
            if value:
                return Path(value)
    return Path("/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval")


def _build_inverse_candidate_record(record: dict[str, Any], *, inverse_edit: dict[str, Any], index: int) -> dict[str, Any]:
    forward_pair_id = str(record.get("proposal_id") or record.get("candidate_id") or record.get("pair_group_id") or f"forward_{index:06d}")
    ref_video = str(record.get("reference_video", "")).strip()
    tgt_video = str(record.get("target_video", "")).strip()
    ref_clip_id = str(record.get("reference_clip_id", "")).strip()
    tgt_clip_id = str(record.get("target_clip_id", "")).strip()
    inverse_pair_group_id = _inverse_pair_group_id(record)
    pair_group_id = str(record.get("pair_group_id") or inverse_pair_group_id).strip()
    edit_metadata = _b_line_edit_metadata(str(inverse_edit.get("edit_text", "")).strip(), direction="inverse")
    candidate = dict(record)
    candidate.update(
        {
            "proposal_id": f"inverse_{forward_pair_id}",
            "is_inverse": True,
            "derived_from_inverse": True,
            "audio_delta_training_record": True,
            "forward_pair_id": forward_pair_id,
            "inverse_pair_group_id": inverse_pair_group_id,
            "pair_group_id": pair_group_id,
            "split_group_id": _source_split_group_id(record),
            "forward_edit_text": str(record.get("edit_text", "")).strip(),
            "inverse_edit_text": str(inverse_edit.get("edit_text", "")).strip(),
            "inverse_generation_rule": str(inverse_edit.get("rule", "")).strip(),
            "direction": "inverse",
            "edit_type": edit_metadata["edit_type"],
            "audio_delta_type": edit_metadata["audio_delta_type"],
            "old_audio": edit_metadata["old_audio"],
            "new_audio": edit_metadata["new_audio"],
            "edit_metadata": edit_metadata,
            "reference_video": tgt_video,
            "target_video": ref_video,
            "reference_clip_id": tgt_clip_id,
            "target_clip_id": ref_clip_id,
            "reference_caption": str(record.get("target_caption", "")).strip(),
            "target_caption": str(record.get("reference_caption", "")).strip(),
            "audio_only_reference_content": str(record.get("audio_only_target_content", "")).strip(),
            "audio_only_target_content": str(record.get("audio_only_reference_content", "")).strip(),
            "edit_text": str(inverse_edit.get("edit_text", "")).strip(),
            "benchmark_eligible": False,
            "training_eligible": bool(inverse_edit.get("ok")),
            "split_tier": "extended" if inverse_edit.get("ok") else "diagnostic",
            "audio_delta_hard_negatives": _audio_delta_hard_negatives(record, reference_video=tgt_video),
            "visual_constraint": _visual_constraint_payload(record),
            "shortcut_label": _shortcut_label(record),
            "source_disjoint_group_id": _source_split_group_id(record),
        }
    )
    quality = dict(candidate.get("quality") if isinstance(candidate.get("quality"), dict) else {})
    quality.update({"split_tier": candidate["split_tier"], "benchmark_eligible": False, "training_eligible": candidate["training_eligible"]})
    candidate["quality"] = quality
    return candidate


def _verify_inverse_candidate(
    candidate: dict[str, Any],
    *,
    root: Path,
    run_root: Path,
    client: OpenAIComposedDataClient,
    omni_retries: int,
    fail_on_transient_omni_errors: bool,
) -> dict[str, Any]:
    checked = dict(candidate)
    reference_path = _resolve_under_root(root, str(checked.get("reference_video", "")))
    target_path = _resolve_under_root(root, str(checked.get("target_video", "")))
    reference_clip_id = str(checked.get("reference_clip_id") or Path(str(checked.get("reference_video", ""))).stem)
    target_clip_id = str(checked.get("target_clip_id") or Path(str(checked.get("target_video", ""))).stem)
    reference_audio_path = _extract_audio_only_cache(video_path=reference_path, cache_dir=run_root / "inverse_audio_only_cache", clip_id=reference_clip_id)
    target_audio_path = _extract_audio_only_cache(video_path=target_path, cache_dir=run_root / "inverse_audio_only_cache", clip_id=target_clip_id)
    reference_video_only_path = _extract_video_only_cache(video_path=reference_path, cache_dir=run_root / "inverse_video_only_cache", clip_id=reference_clip_id)
    target_video_only_path = _extract_video_only_cache(video_path=target_path, cache_dir=run_root / "inverse_video_only_cache", clip_id=target_clip_id)
    inverse_edit_text = str(checked.get("inverse_edit_text") or checked.get("edit_text") or "").strip()
    inverse_proposal = _inverse_audio_only_proposal(checked)
    local_gate_report = _inverse_local_gate_report(checked)

    audio_verify, raw_audio_verify = _call_omni_with_retries(
        label=f"inverse_audio_only_verify:{checked.get('proposal_id', '')}",
        retries=omni_retries,
        fail_on_transient=fail_on_transient_omni_errors,
        func=lambda: client.verify_b_line_audio_only_edit(
            reference_audio_path=str(reference_audio_path),
            target_audio_path=str(target_audio_path),
            edit_text=inverse_edit_text,
            audio_only_proposal=inverse_proposal,
        ),
    )
    video_shortcut, raw_video_shortcut = _call_omni_with_retries(
        label=f"inverse_video_only_shortcut:{checked.get('proposal_id', '')}",
        retries=omni_retries,
        fail_on_transient=fail_on_transient_omni_errors,
        func=lambda: client.verify_b_line_video_only_shortcut(
            reference_clip_path=str(reference_video_only_path),
            target_clip_path=str(target_video_only_path),
            edit_text=inverse_edit_text,
            audio_only_evidence={"inverse_proposal": inverse_proposal, "inverse_audio_only_verification": audio_verify},
            local_gate_report=local_gate_report,
        ),
    )
    full_av, raw_full_av = _call_omni_with_retries(
        label=f"inverse_full_av_consistency:{checked.get('proposal_id', '')}",
        retries=omni_retries,
        fail_on_transient=fail_on_transient_omni_errors,
        func=lambda: client.verify_b_line_full_av_consistency(
            reference_clip_path=str(reference_path),
            target_clip_path=str(target_path),
            edit_text=inverse_edit_text,
            audio_only_evidence={"inverse_proposal": inverse_proposal, "inverse_audio_only_verification": audio_verify},
            local_gate_report=local_gate_report,
        ),
    )
    issues = _inverse_verification_issues(audio_verify, video_shortcut, full_av)
    accepted = not issues
    checked.update(
        {
            "accepted": accepted,
            "inverse_accept": accepted,
            "final_omni_accept": accepted,
            "model_accepted": accepted,
            "inverse_reject_reason": "" if accepted else "; ".join(issues),
            "inverse_audio_only_proposal": inverse_proposal,
            "inverse_audio_only_verification": audio_verify,
            "raw_inverse_audio_only_verification": raw_audio_verify,
            "inverse_video_only_shortcut": video_shortcut,
            "raw_inverse_video_only_shortcut": raw_video_shortcut,
            "inverse_full_av_consistency": full_av,
            "raw_inverse_full_av_consistency": raw_full_av,
            "audio_only_verification": audio_verify,
            "video_only_shortcut": video_shortcut,
            "full_av_consistency": full_av,
            "single_source_pair_acceptance_issues": issues,
            "split_tier": "extended" if accepted else "diagnostic",
            "benchmark_eligible": False,
            "training_eligible": accepted,
        }
    )
    quality = dict(checked.get("quality") if isinstance(checked.get("quality"), dict) else {})
    quality.update({"split_tier": checked["split_tier"], "benchmark_eligible": False, "training_eligible": accepted})
    checked["quality"] = quality
    return checked


def _inverse_audio_only_proposal(record: dict[str, Any]) -> dict[str, Any]:
    difference_type = str((record.get("difference") or {}).get("type", "") if isinstance(record.get("difference"), dict) else "").strip()
    if difference_type not in {"speech", "audio_event"}:
        delta_type = str(record.get("audio_delta_type", "")).strip()
        difference_type = "speech" if delta_type in {"speech", "speech_topic"} else "audio_event"
    return {
        "accept": True,
        "difference_type": difference_type,
        "b_subtype": record.get("b_subtype", ""),
        "reference_audio_content": str(record.get("audio_only_reference_content", "")).strip(),
        "target_audio_content": str(record.get("audio_only_target_content", "")).strip(),
        "edit_text": str(record.get("inverse_edit_text") or record.get("edit_text") or "").strip(),
        "audio_difference_specific": True,
        "edit_text_audio_only": True,
        "confidence": max(0.7, _score_float(record.get("confidence"))),
        "evidence": _dedupe_strings(["inverse edit generated from accepted forward B-line sample"] + _normalize_list(record.get("audio_evidence", []))),
        "reject_reason": "",
    }


def _inverse_local_gate_report(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": True,
        "hard_reject": [],
        "review_required": ["inverse_reverification"],
        "difference_type": str((record.get("difference") or {}).get("type", "") if isinstance(record.get("difference"), dict) else ""),
        "confidence": _score_float(record.get("confidence")),
        "acceptance_profile": "b_audio_blind_review_v2_inverse",
        "audio_dataset_line": SPEECH_AUDIO_CONTENT_LINE,
        "visual_context_type": str(record.get("video_context_type", "")),
        "video_context_strength": _score_float(record.get("video_context_strength")),
        "asr_degeneracy_risk": _score_float(record.get("asr_degeneracy_risk")),
        "is_inverse": True,
        "forward_pair_id": record.get("forward_pair_id", ""),
    }


def _inverse_verification_issues(audio_verify: dict[str, Any], video_shortcut: dict[str, Any], full_av: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not _boolish(audio_verify.get("accept")):
        reason = str(audio_verify.get("reject_reason", "")).strip()
        issues.append("inverse_audio_only_verification_reject" + (f": {reason}" if reason else ""))
    if _boolish(audio_verify.get("reference_satisfies_edit")):
        issues.append("inverse_audio_only_reference_satisfies_edit")
    if not _boolish(audio_verify.get("target_satisfies_edit")):
        issues.append("inverse_audio_only_target_missing_edit")
    if not _boolish(audio_verify.get("audio_difference_specific")):
        issues.append("inverse_audio_only_difference_not_specific")
    if not _boolish(audio_verify.get("edit_text_audio_only")):
        issues.append("inverse_audio_only_edit_text_not_audio_only")
    if _score_float(audio_verify.get("confidence")) < 0.70:
        issues.append(f"inverse_audio_only_confidence_below_threshold: {_score_float(audio_verify.get('confidence')):.2f} < 0.70")
    if not _boolish(video_shortcut.get("accept")):
        reason = str(video_shortcut.get("reject_reason", "")).strip()
        issues.append("inverse_video_only_shortcut_reject" + (f": {reason}" if reason else ""))
    if _boolish(video_shortcut.get("visual_shortcut_risk")):
        issues.append("inverse_video_only_shortcut_risk")
    if _boolish(video_shortcut.get("can_identify_target_without_audio")):
        issues.append("inverse_video_only_can_identify_target_without_audio")
    if _score_float(video_shortcut.get("confidence")) < 0.60:
        issues.append(f"inverse_video_only_confidence_below_threshold: {_score_float(video_shortcut.get('confidence')):.2f} < 0.60")
    if not _boolish(full_av.get("accept")):
        reason = str(full_av.get("reject_reason", "")).strip()
        issues.append("inverse_full_av_consistency_reject" + (f": {reason}" if reason else ""))
    if not _boolish(full_av.get("visual_context_preserved")):
        issues.append("inverse_full_av_visual_context_not_preserved")
    if _boolish(full_av.get("visual_shortcut_risk")):
        issues.append("inverse_full_av_visual_shortcut_risk")
    if not _boolish(full_av.get("audio_edit_still_valid")):
        issues.append("inverse_full_av_audio_edit_not_valid")
    if _score_float(full_av.get("confidence")) < 0.60:
        issues.append(f"inverse_full_av_confidence_below_threshold: {_score_float(full_av.get('confidence')):.2f} < 0.60")
    return _dedupe_strings(issues)


def _forward_train_record(record: dict[str, Any]) -> dict[str, Any]:
    forward = dict(record)
    edit_metadata = _b_line_edit_metadata(str(forward.get("edit_text") or forward.get("audio_only_edit_text") or ""), direction="forward")
    forward.setdefault("is_inverse", False)
    forward.setdefault("derived_from_inverse", False)
    forward["audio_delta_training_record"] = True
    forward["direction"] = "forward"
    forward["edit_type"] = edit_metadata["edit_type"]
    forward["audio_delta_type"] = edit_metadata["audio_delta_type"]
    forward["old_audio"] = edit_metadata["old_audio"]
    forward["new_audio"] = edit_metadata["new_audio"]
    forward["edit_metadata"] = edit_metadata
    forward["pair_group_id"] = str(forward.get("pair_group_id") or _inverse_pair_group_id(forward)).strip()
    forward["inverse_pair_group_id"] = str(forward.get("inverse_pair_group_id") or _inverse_pair_group_id(forward)).strip()
    forward["split_group_id"] = _source_split_group_id(forward)
    forward["source_disjoint_group_id"] = _source_split_group_id(forward)
    forward["audio_delta_hard_negatives"] = _audio_delta_hard_negatives(forward, reference_video=str(forward.get("reference_video", "")).strip())
    forward["visual_constraint"] = _visual_constraint_payload(forward)
    forward["shortcut_label"] = _shortcut_label(forward)
    forward.setdefault("training_eligible", True)
    return forward


def _inverse_pair_group_id(record: dict[str, Any]) -> str:
    ref = str(record.get("reference_clip_id") or record.get("reference_video") or "").strip()
    tgt = str(record.get("target_clip_id") or record.get("target_video") or "").strip()
    group = str(record.get("group_id") or record.get("source_id") or "").strip()
    pair_key = "::".join(sorted([ref, tgt]))
    return f"inverse_pair_{_stable_hash(group + '::' + pair_key)[:16]}"


def _source_split_group_id(record: dict[str, Any]) -> str:
    for key in ("source_clip_id", "group_id", "reuse_source_folder", "source_path"):
        value = str(record.get(key) or "").strip()
        if value:
            return f"source_{_stable_hash(value)[:16]}"
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    value = str(source.get("url") or source.get("path") or "").strip()
    if value:
        return f"source_{_stable_hash(value)[:16]}"
    return _inverse_pair_group_id(record)


def _b_line_edit_metadata(edit_text: str, *, direction: str) -> dict[str, Any]:
    text = " ".join(str(edit_text or "").strip().split())
    parsed = _parse_b_line_edit_components(text)
    return {
        "edit_type": parsed["edit_type"],
        "audio_delta_type": parsed["audio_delta_type"],
        "old_audio": parsed["old_audio"],
        "new_audio": parsed["new_audio"],
        "direction": direction,
        "edit_text": text,
        "parse_ok": bool(parsed["parse_ok"]),
    }


def _parse_b_line_edit_components(edit_text: str) -> dict[str, Any]:
    text = " ".join(str(edit_text or "").strip().split())
    defaults = {"edit_type": "unknown", "audio_delta_type": "unknown", "old_audio": "", "new_audio": "", "parse_ok": False}
    patterns = [
        (r"^change the speech from discussing (?P<a>.+?) to discussing (?P<b>.+)$", "replace", "speech_topic"),
        (r"^change the voice from saying [\"'](?P<a>.+?)[\"'] to saying [\"'](?P<b>.+?)[\"']$", "replace", "speech_phrase"),
        (r"^change the singing from [\"']?(?P<a>.+?)[\"']? to [\"']?(?P<b>.+?)[\"']?$", "replace", "music"),
        (r"^replace (?P<a>.+?) with (?P<b>.+?)(?: in the audio)?$", "replace", "sound_event"),
    ]
    for pattern, edit_type, audio_delta_type in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        old_audio = _clean_inverse_endpoint(match.group("a"))
        new_audio = _clean_inverse_endpoint(match.group("b"))
        return {
            "edit_type": edit_type,
            "audio_delta_type": audio_delta_type,
            "old_audio": old_audio,
            "new_audio": new_audio,
            "parse_ok": not bool(_inverse_endpoint_issue(old_audio, new_audio)),
        }
    add_match = re.match(r"^add (?P<x>.+?)(?: to the audio)?$", text, flags=re.IGNORECASE)
    if add_match:
        new_audio = _clean_inverse_endpoint(add_match.group("x"))
        return {"edit_type": "add", "audio_delta_type": "sound_event", "old_audio": "none_or_weaker", "new_audio": new_audio, "parse_ok": not bool(_inverse_endpoint_issue(new_audio, "valid_target"))}
    remove_match = re.match(r"^remove (?P<x>.+?)(?: from the audio)?$", text, flags=re.IGNORECASE)
    if remove_match:
        old_audio = _clean_inverse_endpoint(remove_match.group("x"))
        return {"edit_type": "remove", "audio_delta_type": "sound_event", "old_audio": old_audio, "new_audio": "none_or_weaker", "parse_ok": not bool(_inverse_endpoint_issue(old_audio, "valid_target"))}
    for word in ("increase", "decrease"):
        match = re.match(rf"^{word} (?P<x>.+?)(?: in the audio)?$", text, flags=re.IGNORECASE)
        if match:
            audio = _clean_inverse_endpoint(match.group("x"))
            return {"edit_type": word, "audio_delta_type": "sound_event", "old_audio": f"{word}_source_state", "new_audio": audio, "parse_ok": not bool(_inverse_endpoint_issue(audio, "valid_target"))}
    return defaults


def _audio_delta_hard_negatives(record: dict[str, Any], *, reference_video: str) -> list[dict[str, str]]:
    negatives: list[dict[str, str]] = []
    if reference_video:
        negatives.append({"type": "reference", "video": reference_video})
    labels = ["visual_hard", "audio_hard", "asr_hard"]
    raw_negatives = [str(item).strip() for item in record.get("hard_negatives", []) if str(item).strip()] if isinstance(record.get("hard_negatives"), list) else []
    for label, video in zip(labels, raw_negatives):
        if video and video != reference_video:
            negatives.append({"type": label, "video": video})
    return negatives


def _visual_constraint_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "visual_context_similarity": _b_line_metric(record, "visual_context_similarity"),
        "video_context_strength": _b_line_metric(record, "video_context_strength"),
        "visual_shortcut_risk": _b_line_visual_shortcut_risk(record),
        "video_only_can_identify_target": _b_line_bool(record, "video_only_can_identify_target_without_audio")
        or _b_line_bool(record, "can_identify_target_without_audio"),
    }


def _shortcut_label(record: dict[str, Any]) -> str:
    tier = str(record.get("split_tier") or "").strip()
    if tier == "diagnostic":
        reasons = " ".join(_normalize_list(record.get("diagnostic_reason", []))).lower()
        if "asr" in reasons:
            return "ASR-like"
        if "visual" in reasons:
            return "visual-shortcut"
        return "ambiguous"
    if _b_line_audio_only_solvability(
        record,
        asr_degeneracy_risk=_b_line_metric(record, "asr_degeneracy_risk"),
        audio_delta_strength=_b_line_metric(record, "audio_delta_strength"),
    ) >= 0.85:
        return "audio-only-shortcut"
    return "clean_audio_delta"


def _inverse_b_line_edit_text(edit_text: str) -> dict[str, Any]:
    text = " ".join(str(edit_text or "").strip().split())
    if not text:
        return {"ok": False, "edit_text": "", "rule": "", "reject_reason": "inverse_edit_not_parseable: empty edit_text"}
    patterns = [
        (r"^change the speech from discussing (?P<a>.+?) to discussing (?P<b>.+)$", "speech_discussing", "change the speech from discussing {b} to discussing {a}"),
        (r"^change the voice from saying [\"'](?P<a>.+?)[\"'] to saying [\"'](?P<b>.+?)[\"']$", "voice_saying", 'change the voice from saying "{b}" to saying "{a}"'),
        (r"^change the singing from [\"']?(?P<a>.+?)[\"']? to [\"']?(?P<b>.+?)[\"']?$", "singing", 'change the singing from "{b}" to "{a}"'),
        (r"^replace (?P<a>.+?) with (?P<b>.+?)(?: in the audio)?$", "replace_audio", "replace {b} with {a}"),
    ]
    for pattern, rule, template in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        left = _clean_inverse_endpoint(match.group("a"))
        right = _clean_inverse_endpoint(match.group("b"))
        issue = _inverse_endpoint_issue(left, right)
        if issue:
            return {"ok": False, "edit_text": "", "rule": rule, "reject_reason": issue}
        return {"ok": True, "edit_text": template.format(a=left, b=right), "rule": rule, "reject_reason": ""}
    add_match = re.match(r"^add (?P<x>.+?)(?: to the audio)?$", text, flags=re.IGNORECASE)
    if add_match:
        endpoint = _clean_inverse_endpoint(add_match.group("x"))
        issue = _inverse_endpoint_issue(endpoint, "valid_target")
        if issue:
            return {"ok": False, "edit_text": "", "rule": "add_to_remove", "reject_reason": issue}
        return {"ok": True, "edit_text": f"remove {endpoint} from the audio", "rule": "add_to_remove", "reject_reason": ""}
    remove_match = re.match(r"^remove (?P<x>.+?)(?: from the audio)?$", text, flags=re.IGNORECASE)
    if remove_match:
        endpoint = _clean_inverse_endpoint(remove_match.group("x"))
        issue = _inverse_endpoint_issue(endpoint, "valid_target")
        if issue:
            return {"ok": False, "edit_text": "", "rule": "remove_to_add", "reject_reason": issue}
        return {"ok": True, "edit_text": f"add {endpoint} to the audio", "rule": "remove_to_add", "reject_reason": ""}
    for left_word, right_word in (("increase", "decrease"), ("decrease", "increase")):
        match = re.match(rf"^{left_word} (?P<x>.+?)(?: in the audio)?$", text, flags=re.IGNORECASE)
        if match:
            endpoint = _clean_inverse_endpoint(match.group("x"))
            issue = _inverse_endpoint_issue(endpoint, "valid_target")
            if issue:
                return {"ok": False, "edit_text": "", "rule": f"{left_word}_to_{right_word}", "reject_reason": issue}
            return {"ok": True, "edit_text": f"{right_word} {endpoint} in the audio", "rule": f"{left_word}_to_{right_word}", "reject_reason": ""}
    return {"ok": False, "edit_text": "", "rule": "", "reject_reason": "inverse_edit_not_parseable"}


def _clean_inverse_endpoint(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().strip("\"'., ")).strip()


def _inverse_endpoint_issue(left: str, right: str) -> str:
    if not left or not right:
        return "inverse_edit_not_parseable: empty endpoint"
    if left.lower() == right.lower():
        return "inverse_edit_not_parseable: identical endpoints"
    hollow = {"speech", "audio", "sound", "noise", "unknown", "unintelligible", "not transcribed", "different sentence", "speaking"}
    if left.lower() in hollow or right.lower() in hollow:
        return "inverse_edit_not_parseable: hollow endpoint"
    return ""


def _assign_b_line_tiers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tiered = [_b_line_record_with_tier(record) for record in records]
    main_speech = [record for record in tiered if record.get("split_tier") == "main" and _b_line_record_subtype(record) == "speech_topic_in_video_context"]
    main_non_speech = [record for record in tiered if record.get("split_tier") == "main" and _b_line_record_subtype(record) in {"music", "sound_event"}]
    speech_cap_35 = int(len(main_non_speech) * 0.35 / 0.65) if main_non_speech else 0
    speech_cap_40 = int(len(main_non_speech) * 0.40 / 0.60) if main_non_speech else 0
    speech_cap = speech_cap_35 if len(main_speech) <= speech_cap_35 else speech_cap_40
    for record in main_speech[speech_cap:]:
        reasons = _dedupe_strings(_normalize_list(record.get("diagnostic_reason", [])) + ["main_speech_cap_exceeded"])
        record["split_tier"] = "extended"
        record["benchmark_eligible"] = False
        record["training_eligible"] = True
        record["diagnostic_reason"] = reasons
    return tiered


def _b_line_record_with_tier(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    tier_metadata = _b_line_record_tier(record)
    enriched.update(tier_metadata)
    quality = dict(enriched.get("quality") if isinstance(enriched.get("quality"), dict) else {})
    for key in (
        "split_tier",
        "benchmark_eligible",
        "training_eligible",
        "diagnostic_reason",
        "b_subtype",
        "video_context_strength",
        "asr_degeneracy_risk",
        "audio_delta_strength",
        "visual_shortcut_risk",
        "audio_only_solvability",
        "full_av_required",
    ):
        quality[key] = enriched.get(key)
    enriched["quality"] = quality
    return enriched


def _b_line_record_tier(record: dict[str, Any]) -> dict[str, Any]:
    subtype = _b_line_record_subtype(record)
    video_context_strength = _b_line_metric(record, "video_context_strength")
    asr_degeneracy_risk = _b_line_metric(record, "asr_degeneracy_risk")
    audio_delta_strength = _b_line_metric(record, "audio_delta_strength")
    visual_shortcut_risk = _b_line_visual_shortcut_risk(record)
    audio_only_solvability = _b_line_audio_only_solvability(record, asr_degeneracy_risk=asr_degeneracy_risk, audio_delta_strength=audio_delta_strength)
    full_av_required = _b_line_full_av_required(record, video_context_strength=video_context_strength, asr_degeneracy_risk=asr_degeneracy_risk, visual_shortcut_risk=visual_shortcut_risk)
    speech_role = _b_line_text_value(record, "speech_role")
    reasons: list[str] = []
    if bool(record.get("fallback")):
        reasons.append("fallback_pair_proposal")
    if subtype not in {"speech_topic_in_video_context", "music", "sound_event"}:
        reasons.append(f"unsupported_b_subtype:{subtype or 'unknown'}")
    if asr_degeneracy_risk > 0.70:
        reasons.append("asr_degeneracy_risk_high")
    if speech_role in {"asr_only", "generic_talking_head"}:
        reasons.append(f"speech_role_{speech_role}")
    if audio_only_solvability >= 0.85:
        reasons.append("audio_only_solvability_high")
    if _b_line_transcript_like_edit(record):
        reasons.append("transcript_like_edit_text")
    if _b_line_hollow_audio_edit(record):
        reasons.append("hollow_audio_edit_text")
    if visual_shortcut_risk > 0.35 or _b_line_bool(record, "can_identify_target_without_audio") or _b_line_bool(record, "video_only_can_identify_target_without_audio"):
        reasons.append("visual_shortcut_risk")

    main_ready = (
        bool(record.get("accepted"))
        and not bool(record.get("fallback"))
        and subtype in {"speech_topic_in_video_context", "music", "sound_event"}
        and audio_delta_strength >= 0.70
        and video_context_strength >= 0.60
        and asr_degeneracy_risk <= 0.35
        and visual_shortcut_risk <= 0.35
        and not _b_line_bool(record, "can_identify_target_without_audio")
        and not _b_line_bool(record, "video_only_can_identify_target_without_audio")
        and (audio_only_solvability < 0.85 or full_av_required)
        and not _b_line_transcript_like_edit(record)
        and not _b_line_hollow_audio_edit(record)
    )
    extended_ready = (
        bool(record.get("accepted"))
        and not bool(record.get("fallback"))
        and subtype in {"speech_topic_in_video_context", "music", "sound_event"}
        and audio_delta_strength >= 0.60
        and video_context_strength >= 0.35
        and asr_degeneracy_risk <= 0.70
        and visual_shortcut_risk <= 0.35
        and not _b_line_bool(record, "can_identify_target_without_audio")
        and not _b_line_bool(record, "video_only_can_identify_target_without_audio")
        and not _b_line_hollow_audio_edit(record)
    )
    if main_ready and not reasons:
        split_tier = "main"
    elif extended_ready:
        split_tier = "extended"
    else:
        split_tier = "diagnostic"
        if not reasons:
            reasons.append("below_extended_threshold")
    return {
        "split_tier": split_tier,
        "benchmark_eligible": split_tier == "main",
        "training_eligible": split_tier in {"main", "extended"},
        "diagnostic_reason": _dedupe_strings(reasons),
        "b_subtype": subtype,
        "video_context_strength": round(video_context_strength, 3),
        "asr_degeneracy_risk": round(asr_degeneracy_risk, 3),
        "audio_delta_strength": round(audio_delta_strength, 3),
        "visual_shortcut_risk": round(visual_shortcut_risk, 3),
        "audio_only_solvability": round(audio_only_solvability, 3),
        "full_av_required": bool(full_av_required),
    }


def _select_b_line_records(records: list[dict[str, Any]], *, target_b_count: int) -> list[dict[str, Any]]:
    target = max(0, int(target_b_count or 0))
    if target <= 0:
        return []
    speech_cap = max(1, int(target * 0.40)) if target else 0
    non_speech = [record for record in records if _b_line_record_subtype(record) in {"music", "sound_event"}]
    speech = [record for record in records if _b_line_record_subtype(record) == "speech_topic_in_video_context"]
    other = [record for record in records if _b_line_record_subtype(record) not in {"music", "sound_event", "speech_topic_in_video_context"}]
    selected = non_speech[:target]
    remaining = target - len(selected)
    if remaining > 0:
        selected.extend(speech[: min(remaining, speech_cap)])
    remaining = target - len(selected)
    if remaining > 0:
        selected.extend(other[:remaining])
    return selected[:target]


def _b_line_record_subtype(record: dict[str, Any]) -> str:
    subtype = str(record.get("b_subtype") or "").strip()
    if subtype in {"speech_topic_in_video_context", "music", "sound_event"}:
        return subtype
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    subtype = str(quality.get("b_subtype") or "").strip()
    if subtype in {"speech_topic_in_video_context", "music", "sound_event"}:
        return subtype
    difference = record.get("difference") if isinstance(record.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    text = _annotation_text(record, ("edit_text", "audio_evidence"))
    if any(term in text.lower() for term in ("music", "song", "sing", "guitar", "piano", "melody")):
        return "music"
    if difference_type == "speech":
        return "speech_topic_in_video_context"
    return "sound_event" if difference_type == "audio_event" else "unknown"


def _b_line_metric(record: dict[str, Any], key: str) -> float:
    if key in record:
        return _score_float(record.get(key))
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    if key in quality:
        return _score_float(quality.get(key))
    final = record.get("final_omni_verification") if isinstance(record.get("final_omni_verification"), dict) else {}
    if key in final:
        return _score_float(final.get(key))
    audio_delta = record.get("audio_delta_analysis") if isinstance(record.get("audio_delta_analysis"), dict) else {}
    if key in audio_delta:
        return _score_float(audio_delta.get(key))
    audio_verify = record.get("audio_only_verification") if isinstance(record.get("audio_only_verification"), dict) else {}
    if key in audio_verify:
        return _score_float(audio_verify.get(key))
    full_av = record.get("full_av_consistency") if isinstance(record.get("full_av_consistency"), dict) else {}
    if key in full_av:
        return _score_float(full_av.get(key))
    video_only = record.get("video_only_shortcut") if isinstance(record.get("video_only_shortcut"), dict) else {}
    if key in video_only:
        return _score_float(video_only.get(key))
    return 0.0


def _b_line_text_value(record: dict[str, Any], key: str) -> str:
    for source in (
        record,
        record.get("quality") if isinstance(record.get("quality"), dict) else {},
        record.get("final_omni_verification") if isinstance(record.get("final_omni_verification"), dict) else {},
        record.get("audio_delta_analysis") if isinstance(record.get("audio_delta_analysis"), dict) else {},
        record.get("audio_only_proposal") if isinstance(record.get("audio_only_proposal"), dict) else {},
    ):
        value = source.get(key) if isinstance(source, dict) else None
        if str(value or "").strip():
            return str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return ""


def _b_line_bool(record: dict[str, Any], key: str) -> bool:
    for source in (
        record,
        record.get("quality") if isinstance(record.get("quality"), dict) else {},
        record.get("final_omni_verification") if isinstance(record.get("final_omni_verification"), dict) else {},
        record.get("video_only_shortcut") if isinstance(record.get("video_only_shortcut"), dict) else {},
        record.get("full_av_consistency") if isinstance(record.get("full_av_consistency"), dict) else {},
    ):
        if isinstance(source, dict) and key in source:
            value = source.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value) >= 0.5
            return str(value or "").strip().lower() in {"1", "true", "yes", "y"}
    return False


def _b_line_visual_shortcut_risk(record: dict[str, Any]) -> float:
    if _b_line_bool(record, "visual_shortcut_risk") or _b_line_bool(record, "video_only_shortcut_risk"):
        return 1.0
    return max(
        _b_line_metric(record, "visual_shortcut_risk"),
        _b_line_metric(record, "video_only_shortcut_risk"),
    )


def _b_line_audio_only_solvability(record: dict[str, Any], *, asr_degeneracy_risk: float, audio_delta_strength: float) -> float:
    explicit = _b_line_metric(record, "audio_only_solvability")
    if explicit > 0:
        return explicit
    audio_verify = record.get("audio_only_verification") if isinstance(record.get("audio_only_verification"), dict) else {}
    verification_confidence = _score_float(audio_verify.get("confidence") or audio_verify.get("quality_score"))
    if asr_degeneracy_risk > 0.55 or _b_line_text_value(record, "speech_role") in {"asr_only", "generic_talking_head"}:
        return max(audio_delta_strength, verification_confidence)
    return min(0.75, max(audio_delta_strength * 0.75, verification_confidence * 0.75))


def _b_line_full_av_required(
    record: dict[str, Any],
    *,
    video_context_strength: float,
    asr_degeneracy_risk: float,
    visual_shortcut_risk: float,
) -> bool:
    if "full_av_required" in record:
        return _b_line_bool(record, "full_av_required")
    return video_context_strength >= 0.45 and asr_degeneracy_risk <= 0.55 and visual_shortcut_risk <= 0.35


def _b_line_transcript_like_edit(record: dict[str, Any]) -> bool:
    text = str(record.get("edit_text") or record.get("audio_only_edit_text") or "").strip().lower()
    if not text:
        return True
    if re.search(r'\bfrom saying\s+"[^"]+"\s+to saying\s+"[^"]+"', text):
        return True
    transcript_terms = ("sentence", "transcript", "word for word", "verbatim", "saying a different", "says something different")
    return any(term in text for term in transcript_terms)


def _b_line_hollow_audio_edit(record: dict[str, Any]) -> bool:
    text = str(record.get("edit_text") or record.get("audio_only_edit_text") or "").strip().lower()
    hollow_terms = (
        "a to b",
        "discussing a to discussing b",
        "speech changed",
        "speech content changed",
        "different tone",
        "different sentence",
        "unintelligible",
        "not transcribed",
        "not discernible",
        "unknown",
        "unspecified",
    )
    return not text or any(term in text for term in hollow_terms)


def _ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def _b_context_cvr_summary(*, root: Path, b_ranked: list[dict[str, Any]], b_selected: list[dict[str, Any]]) -> dict[str, Any]:
    risk_bins: Counter[str] = Counter()
    for record in b_ranked:
        risk = _score_float(record.get("asr_degeneracy_risk", (record.get("quality") or {}).get("asr_degeneracy_risk") if isinstance(record.get("quality"), dict) else 0.0))
        if risk > 0.55:
            risk_bins[">0.55"] += 1
        elif risk > 0.40:
            risk_bins["0.40-0.55"] += 1
        else:
            risk_bins["<=0.40"] += 1
    return {
        "run_root": str(root),
        "b_ranked_count": len(b_ranked),
        "b_accepted_count": sum(1 for record in b_ranked if bool(record.get("accepted"))),
        "b_exported_count": len(b_selected),
        "split_tier_counts": dict(Counter(str(record.get("split_tier") or "unknown") for record in b_selected)),
        "main_count": sum(1 for record in b_selected if record.get("split_tier") == "main"),
        "extended_count": sum(1 for record in b_selected if record.get("split_tier") == "extended"),
        "diagnostic_count": sum(1 for record in b_selected if record.get("split_tier") == "diagnostic"),
        "accepted_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_ranked if bool(record.get("accepted")))),
        "exported_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_selected)),
        "main_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_selected if record.get("split_tier") == "main")),
        "extended_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_selected if record.get("split_tier") == "extended")),
        "diagnostic_subtype_counts": dict(Counter(_b_line_record_subtype(record) for record in b_selected if record.get("split_tier") == "diagnostic")),
        "diagnostic_reason_counts": dict(Counter(reason for record in b_selected for reason in _normalize_list(record.get("diagnostic_reason", [])))),
        "asr_degeneracy_risk_bins": dict(risk_bins),
        "reject_reason_counts": _reject_reason_counts(b_ranked),
    }


def _dedupe_line_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        key = str(
            record.get("proposal_id")
            or record.get("candidate_id")
            or record.get("sample_id")
            or f"{record.get('reference_clip_id', '')}->{record.get('target_clip_id', '')}:{index}"
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
    return selected


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


def _normalize_audio_line_quality_profile(value: str | None) -> str:
    profile = str(value or AUDIO_LINE_PROFILE_DEFAULT).strip().lower().replace("-", "_")
    profile = AUDIO_LINE_PROFILE_ALIASES.get(profile, profile)
    if profile in {"", "none"}:
        return AUDIO_LINE_PROFILE_DEFAULT
    if profile not in AUDIO_LINE_PROFILES:
        raise ValueError(f"unsupported audio_line_quality_profile={value!r}; expected one of {sorted(AUDIO_LINE_PROFILES)}")
    return profile


def _normalize_a_candidate_mode(value: str | None) -> str:
    mode = str(value or A_CANDIDATE_MODE_HYBRID).strip().lower().replace("-", "_")
    if mode in {"", "default"}:
        return A_CANDIDATE_MODE_HYBRID
    if mode not in A_CANDIDATE_MODES:
        raise ValueError(f"unsupported a_candidate_mode={value!r}; expected one of {sorted(A_CANDIDATE_MODES)}")
    return mode


def _normalize_b_candidate_mode(value: str | None) -> str:
    mode = str(value or B_CANDIDATE_MODE_HYBRID).strip().lower().replace("-", "_")
    if mode in {"", "default"}:
        return B_CANDIDATE_MODE_HYBRID
    if mode not in B_CANDIDATE_MODES:
        raise ValueError(f"unsupported b_candidate_mode={value!r}; expected one of {sorted(B_CANDIDATE_MODES)}")
    return mode


def _metadata_value(annotation: dict[str, Any], key: str) -> Any:
    if key in annotation:
        return annotation.get(key)
    metadata = annotation.get("metadata") if isinstance(annotation.get("metadata"), dict) else {}
    return metadata.get(key)


def _annotation_text(annotation: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field in fields:
        value = _metadata_value(annotation, field)
        if isinstance(value, dict):
            parts.extend(str(key) for key in value.keys())
            parts.extend(str(item) for item in value.values())
        else:
            parts.extend(_normalize_list(value))
            if isinstance(value, str):
                parts.append(value)
    return " ".join(part for part in parts if str(part).strip())


def _token_set(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) >= 3}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _visual_context_similarity(reference: dict[str, Any], target: dict[str, Any]) -> float:
    fields = ("summary", "subjects", "actions", "scene", "attributes")
    ref_tokens = _token_set(_annotation_text(reference, fields))
    tgt_tokens = _token_set(_annotation_text(target, fields))
    base = _jaccard(ref_tokens, tgt_tokens)
    if str(reference.get("dataset", "")) == str(target.get("dataset", "")):
        base += 0.05
    return round(min(1.0, base), 3)


def _visual_delta_strength(candidate: dict[str, Any], reference: dict[str, Any], target: dict[str, Any]) -> float:
    difference = candidate.get("difference") if isinstance(candidate.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    ref_visual = _annotation_text(reference, ("summary", "subjects", "actions", "scene", "attributes", "object_counts"))
    tgt_visual = _annotation_text(target, ("summary", "subjects", "actions", "scene", "attributes", "object_counts"))
    distance = 1.0 - _jaccard(_token_set(ref_visual), _token_set(tgt_visual))
    type_bonus = {
        "scene": 0.28,
        "action": 0.22,
        "object_presence": 0.18,
        "object_count": 0.12,
        "attribute": 0.04,
        "visible_text": -0.25,
    }.get(difference_type, 0.0)
    return round(max(0.0, min(1.0, distance + type_bonus)), 3)


def _v4_a_candidate_allowed(*, audio_line_quality_profile: str, difference_type: str, visual_delta_strength: float) -> bool:
    if audio_line_quality_profile != AUDIO_LINE_PROFILE_V4_STRICT:
        return True
    if difference_type == "visible_text":
        return False
    return (difference_type in V4_A_STRONG_VISUAL_TYPES and visual_delta_strength >= 0.45) or visual_delta_strength >= 0.72


def _uses_audio_primary_mining_profile(audio_line_quality_profile: str) -> bool:
    return _normalize_audio_line_quality_profile(audio_line_quality_profile) in {
        AUDIO_LINE_PROFILE_V5_AUDIO_PRIMARY,
        AUDIO_LINE_PROFILE_B_CONTEXT_CVR,
        AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW,
        AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2,
    }


def _v4_b_candidate_allowed(
    *,
    audio_line_quality_profile: str,
    visual_delta_strength: float,
    visual_context_similarity: float,
    audio_text: str,
    difference_type: str,
) -> bool:
    if _uses_audio_primary_mining_profile(audio_line_quality_profile):
        normalized_audio = audio_text.lower()
        if difference_type == "audio_event":
            return any(term in normalized_audio for term in V4_CONCRETE_AUDIO_TERMS)
        return bool(normalized_audio.strip())
    if audio_line_quality_profile != AUDIO_LINE_PROFILE_V4_STRICT:
        return True
    if visual_delta_strength > V4_B_MAX_VISUAL_DELTA_STRENGTH or visual_context_similarity < V4_B_MIN_VISUAL_CONTEXT_SIMILARITY:
        return False
    normalized_audio = audio_text.lower()
    if difference_type == "audio_event":
        if not any(term in normalized_audio for term in V4_CONCRETE_AUDIO_TERMS):
            return False
        if any(term in normalized_audio for term in V4_VAGUE_AUDIO_TERMS) and not any(term in normalized_audio for term in V4_CONCRETE_AUDIO_TERMS):
            return False
    return True


def _b_context_type(reference: dict[str, Any], target: dict[str, Any]) -> str:
    text = _annotation_text(
        reference,
        ("summary", "subjects", "actions", "scene", "storyline", "events", "video_context_type", "speech_role"),
    ).lower()
    text += " " + _annotation_text(
        target,
        ("summary", "subjects", "actions", "scene", "storyline", "events", "video_context_type", "speech_role"),
    ).lower()
    if any(term in text for term in ("news", "report", "anchor", "broadcast")):
        return "news/reporting"
    if any(term in text for term in ("sport", "match", "game", "cricket", "football", "player", "commentary")):
        return "sports_commentary"
    if any(term in text for term in ("tutorial", "instruction", "cook", "recipe", "repair", "demo", "how to")):
        return "tutorial_instruction"
    if any(term in text for term in ("interview", "podium", "press", "stage", "panel")):
        return "interview_context"
    if any(term in text for term in ("livestream", "live stream", "streamer", "vlog", "studio", "desk")):
        return "livestream_context"
    if any(term in text for term in ("singing", "song", "music", "guitar", "piano", "performance", "concert")):
        return "performance_or_singing"
    if any(term in text for term in ("meeting", "conference call", "webinar", "zoom", "podcast", "black screen", "static image")):
        return "asr_only"
    if any(term in text for term in ("talking head", "speaking to camera", "speaker")):
        return "generic_talking_head"
    return "unknown"


def _b_context_strength(reference: dict[str, Any], target: dict[str, Any], visual_context_similarity: float) -> float:
    provided = max(_score_float(_metadata_value(reference, "video_context_strength")), _score_float(_metadata_value(target, "video_context_strength")))
    context_type = _b_context_type(reference, target)
    score = provided
    if context_type in {
        "news/reporting",
        "sports_commentary",
        "tutorial_instruction",
        "interview_context",
        "livestream_context",
        "performance_or_singing",
    }:
        score = max(score, 0.65)
    elif context_type == "generic_talking_head":
        score = max(score, 0.35)
    text_tokens = _token_set(_annotation_text(reference, ("summary", "subjects", "actions", "scene")) + " " + _annotation_text(target, ("summary", "subjects", "actions", "scene")))
    if len(text_tokens) >= 8:
        score = max(score, 0.45)
    if visual_context_similarity > 0:
        score = max(score, min(0.75, 0.25 + visual_context_similarity * 0.5))
    return round(min(1.0, max(0.0, score)), 3)


def _b_asr_degeneracy_risk(reference: dict[str, Any], target: dict[str, Any]) -> float:
    provided = max(_score_float(_metadata_value(reference, "asr_degeneracy_risk")), _score_float(_metadata_value(target, "asr_degeneracy_risk")))
    context_type = _b_context_type(reference, target)
    text = (
        _annotation_text(reference, ("summary", "scene", "subjects", "actions", "video_context_type", "speech_role"))
        + " "
        + _annotation_text(target, ("summary", "scene", "subjects", "actions", "video_context_type", "speech_role"))
    ).lower()
    risk = provided
    if context_type == "asr_only":
        risk = max(risk, 0.80)
    elif context_type == "generic_talking_head":
        risk = max(risk, 0.62)
    elif context_type == "unknown":
        risk = max(risk, 0.56)
    if any(term in text for term in ("black screen", "static image", "podcast", "meeting", "webinar", "zoom")):
        risk = max(risk, 0.80)
    if context_type in {"news/reporting", "sports_commentary", "tutorial_instruction", "interview_context", "livestream_context", "performance_or_singing"}:
        risk = min(risk or 0.45, 0.45)
    return round(min(1.0, max(0.0, risk)), 3)


def _b_context_candidate_allowed(
    *,
    difference_type: str,
    video_context_strength: float,
    asr_degeneracy_risk: float,
) -> bool:
    if difference_type == "speech":
        return video_context_strength >= 0.45 and asr_degeneracy_risk <= 0.55
    return video_context_strength >= 0.35 and asr_degeneracy_risk <= 0.70


def _speech_content_delta_score(reference: dict[str, Any], target: dict[str, Any]) -> float:
    reference_texts = _speech_texts_from_annotation(reference)
    target_texts = _speech_texts_from_annotation(target)
    if not reference_texts or not target_texts:
        return 0.0
    reference_tokens = _token_set(" ".join(reference_texts))
    target_tokens = _token_set(" ".join(target_texts))
    if not reference_tokens or not target_tokens:
        return 0.0
    if reference_tokens == target_tokens:
        return 0.0
    lexical_delta = 1.0 - _jaccard(reference_tokens, target_tokens)
    specificity = _speech_specificity_score(reference, target)
    if specificity <= 0.0:
        return 0.0
    score = 0.35 + lexical_delta * 0.45 + specificity * 0.20
    return round(max(0.0, min(1.0, score)), 3)


def _mine_audio_first_b_candidates(
    *,
    annotations_by_id: dict[str, dict[str, Any]],
    existing_keys: set[tuple[str, str, str]],
    audio_line_quality_profile: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    audio_line_quality_profile = _normalize_audio_line_quality_profile(audio_line_quality_profile)
    records: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations_by_id.values():
        group_key = _annotation_group_key(annotation)
        if group_key:
            groups[group_key].append(annotation)

    for group_key, group_annotations in sorted(groups.items()):
        ordered = sorted(group_annotations, key=_annotation_segment_sort_key)
        for left_index, reference in enumerate(ordered):
            for target in ordered[left_index + 1 :]:
                visual_context_similarity = _visual_context_similarity(reference, target)
                video_context_type = _b_context_type(reference, target)
                video_context_strength = _b_context_strength(reference, target, visual_context_similarity)
                asr_degeneracy_risk = _b_asr_degeneracy_risk(reference, target)
                speech_score = _speech_evidence_score(reference, target)
                if _uses_audio_primary_mining_profile(audio_line_quality_profile):
                    speech_score = max(speech_score, _speech_content_delta_score(reference, target))
                    speech_threshold = 0.45
                else:
                    speech_threshold = 0.70
                if speech_score >= speech_threshold:
                    candidate = _audio_first_base_candidate(reference, target, difference_type="speech", group_key=group_key)
                    visual_delta_strength = _visual_delta_strength(candidate, reference, target)
                    audio_text = " ".join(_speech_texts_from_annotation(reference)[:2] + _speech_texts_from_annotation(target)[:2])
                    context_ok = (
                        audio_line_quality_profile
                        not in {AUDIO_LINE_PROFILE_B_CONTEXT_CVR, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2}
                        or _b_context_candidate_allowed(
                            difference_type="speech",
                            video_context_strength=video_context_strength,
                            asr_degeneracy_risk=asr_degeneracy_risk,
                        )
                    )
                    if context_ok and _v4_b_candidate_allowed(
                        audio_line_quality_profile=audio_line_quality_profile,
                        visual_delta_strength=visual_delta_strength,
                        visual_context_similarity=visual_context_similarity,
                        audio_text=audio_text,
                        difference_type="speech",
                    ):
                        key = _b_pair_key(candidate)
                        if key not in existing_keys:
                            records.append(
                                _speech_line_candidate(
                                    candidate,
                                    reference,
                                    target,
                                    visual_delta_strength=visual_delta_strength,
                                    visual_context_similarity=visual_context_similarity,
                                    audio_line_quality_profile=audio_line_quality_profile,
                                    video_context_type=video_context_type,
                                    video_context_strength=video_context_strength,
                                    asr_degeneracy_risk=asr_degeneracy_risk,
                                )
                            )
                            existing_keys.add(key)
                    else:
                        reject_counts["b_audio_first_speech_visual_gate_failed"] += 1
                non_speech_score = _non_speech_audio_event_score(reference, target)
                non_speech_threshold = 0.55 if _uses_audio_primary_mining_profile(audio_line_quality_profile) else 0.70
                if non_speech_score >= non_speech_threshold:
                    candidate = _audio_first_base_candidate(reference, target, difference_type="audio_event", group_key=group_key)
                    visual_delta_strength = _visual_delta_strength(candidate, reference, target)
                    audio_text = " ".join(_normalize_list(reference.get("audio_events", [])) + _normalize_list(target.get("audio_events", [])))
                    context_ok = (
                        audio_line_quality_profile
                        not in {AUDIO_LINE_PROFILE_B_CONTEXT_CVR, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW, AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2}
                        or _b_context_candidate_allowed(
                            difference_type="audio_event",
                            video_context_strength=video_context_strength,
                            asr_degeneracy_risk=asr_degeneracy_risk,
                        )
                    )
                    if context_ok and _v4_b_candidate_allowed(
                        audio_line_quality_profile=audio_line_quality_profile,
                        visual_delta_strength=visual_delta_strength,
                        visual_context_similarity=visual_context_similarity,
                        audio_text=audio_text,
                        difference_type="audio_event",
                    ):
                        key = _b_pair_key(candidate)
                        if key not in existing_keys:
                            records.append(
                                _audio_event_line_candidate(
                                    candidate,
                                    reference,
                                    target,
                                    non_speech_score,
                                    visual_delta_strength=visual_delta_strength,
                                    visual_context_similarity=visual_context_similarity,
                                    audio_line_quality_profile=audio_line_quality_profile,
                                    video_context_type=video_context_type,
                                    video_context_strength=video_context_strength,
                                    asr_degeneracy_risk=asr_degeneracy_risk,
                                )
                            )
                            existing_keys.add(key)
                    else:
                        reject_counts["b_audio_first_event_visual_gate_failed"] += 1
    return records, reject_counts


def _annotation_group_key(annotation: dict[str, Any]) -> str:
    for key in ("group_id", "source_clip_id", "reuse_source_folder", "source_path"):
        value = str(annotation.get(key, "")).strip()
        if value:
            return value.replace("\\", "/")
    output_path = str(annotation.get("output_path", "")).strip().replace("\\", "/")
    if not output_path:
        return ""
    return output_path.rsplit("/", 1)[0] if "/" in output_path else "unknown"


def _annotation_segment_sort_key(annotation: dict[str, Any]) -> tuple[float, tuple[int, str]]:
    start = _score_float(annotation.get("start_seconds", annotation.get("relative_start_seconds")))
    return (start, _clip_sort_key(str(annotation.get("clip_id") or annotation.get("output_path") or "")))


def _audio_first_base_candidate(reference: dict[str, Any], target: dict[str, Any], *, difference_type: str, group_key: str) -> dict[str, Any]:
    reference_id = str(reference.get("clip_id", "")).strip()
    target_id = str(target.get("clip_id", "")).strip()
    reference_video = str(reference.get("output_path", "")).strip()
    target_video = str(target.get("output_path", "")).strip()
    candidate_id = f"audio_first_{difference_type}_{_stable_hash(reference_id + '->' + target_id)[:12]}"
    return {
        "candidate_id": candidate_id,
        "proposal_id": candidate_id,
        "single_source_pair": True,
        "reference_clip_id": reference_id,
        "target_clip_id": target_id,
        "reference_video": reference_video,
        "target_video": target_video,
        "reference_start_seconds": reference.get("start_seconds", reference.get("relative_start_seconds")),
        "target_start_seconds": target.get("start_seconds", target.get("relative_start_seconds")),
        "difference": {"type": difference_type, "from": "reference audio", "to": "target audio", "description": "audio-first B-line candidate"},
        "source_context": {"relation": "same_source_video", "single_source_pair": True, "audio_first_b_candidate": True, "group_key": group_key},
        "quality": {"candidate_source": "audio_first_annotation_pair", "same_context_score": 0.9},
        "risk_flags": ["audio_first_b_candidate"],
    }


def _b_pair_key(record: dict[str, Any]) -> tuple[str, str, str]:
    difference = record.get("difference") if isinstance(record.get("difference"), dict) else {}
    return (
        str(record.get("reference_clip_id", "")).strip(),
        str(record.get("target_clip_id", "")).strip(),
        str(difference.get("type", "")).strip(),
    )


def _line_candidate(
    candidate: dict[str, Any],
    line: str,
    *,
    score: float = 0.0,
    min_rms: float = 0.0,
    visual_delta_strength: float = 0.0,
    visual_context_similarity: float = 0.0,
    audio_line_quality_profile: str = AUDIO_LINE_PROFILE_DEFAULT,
) -> dict[str, Any]:
    record = dict(candidate)
    audio_line_quality_profile = _normalize_audio_line_quality_profile(audio_line_quality_profile)
    quality = dict(record.get("quality", {})) if isinstance(record.get("quality"), dict) else {}
    visual_hint_difference = record.get("difference") if isinstance(record.get("difference"), dict) else {}
    quality.update(
        {
            "audio_dataset_line": line,
            "audio_line_quality_profile": audio_line_quality_profile,
            "audio_anchor_required": 1.0 if line == VISUAL_AUDIO_ANCHOR_LINE else quality.get("audio_anchor_required", 0.0),
            "audio_anchor_score": round(score, 4) if line == VISUAL_AUDIO_ANCHOR_LINE else quality.get("audio_anchor_score", 0.0),
            "audio_anchor_min_rms": round(min_rms, 6) if line == VISUAL_AUDIO_ANCHOR_LINE else quality.get("audio_anchor_min_rms", 0.0),
            "audio_anchor_type": "same_source_similar_audio" if line == VISUAL_AUDIO_ANCHOR_LINE else quality.get("audio_anchor_type", ""),
            "edit_primary_modality": "visual" if line == VISUAL_AUDIO_ANCHOR_LINE else "audio",
            "visual_delta_strength": round(float(visual_delta_strength), 3),
            "visual_context_similarity": round(float(visual_context_similarity), 3),
            "visual_hint_difference_type": str(visual_hint_difference.get("type", "")).strip(),
        }
    )
    record["quality"] = quality
    record["audio_dataset_line"] = line
    record["audio_line_quality_profile"] = audio_line_quality_profile
    record["risk_flags"] = _dedupe_strings(_normalize_list(record.get("risk_flags", [])) + [line])
    record["proposal_id"] = f"{line}_{record.get('proposal_id') or _build_proposal_id(str(record.get('reference_video', '')), str(record.get('target_video', '')))}"
    return record


def _speech_line_candidate(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    target: dict[str, Any],
    *,
    visual_delta_strength: float = 0.0,
    visual_context_similarity: float = 0.0,
    audio_line_quality_profile: str = AUDIO_LINE_PROFILE_DEFAULT,
    video_context_type: str = "",
    video_context_strength: float = 0.0,
    asr_degeneracy_risk: float = 0.0,
) -> dict[str, Any]:
    record = _line_candidate(
        candidate,
        SPEECH_AUDIO_CONTENT_LINE,
        visual_delta_strength=visual_delta_strength,
        visual_context_similarity=visual_context_similarity,
        audio_line_quality_profile=audio_line_quality_profile,
    )
    reference_speech = "; ".join(_speech_texts_from_annotation(reference)[:2]) or "reference speech"
    target_speech = "; ".join(_speech_texts_from_annotation(target)[:2]) or "target speech"
    record["difference"] = {
        "type": "speech",
        "from": reference_speech[:180],
        "to": target_speech[:180],
        "description": "the spoken-language content differs between the reference and target clips",
    }
    quality = dict(record.get("quality", {}))
    speech_score = max(_speech_evidence_score(reference, target), _speech_content_delta_score(reference, target))
    quality.update(
        {
            "difference_type": "speech",
            "has_audio_modality": 1.0,
            "speech_evidence_score": speech_score,
            "speech_specificity_score": _speech_specificity_score(reference, target),
            "speech_transcript_backed": 1.0 if _speech_is_transcript_backed(reference, target) else 0.0,
            "audio_content_delta_strength": max(speech_score, _speech_specificity_score(reference, target)),
            "b_subtype": "speech_topic_in_video_context",
            "video_context_type": video_context_type or _b_context_type(reference, target),
            "video_context_strength": video_context_strength or _b_context_strength(reference, target, visual_context_similarity),
            "asr_degeneracy_risk": asr_degeneracy_risk or _b_asr_degeneracy_risk(reference, target),
        }
    )
    record["quality"] = quality
    return record


def _audio_event_line_candidate(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    target: dict[str, Any],
    score: float,
    *,
    visual_delta_strength: float = 0.0,
    visual_context_similarity: float = 0.0,
    audio_line_quality_profile: str = AUDIO_LINE_PROFILE_DEFAULT,
    video_context_type: str = "",
    video_context_strength: float = 0.0,
    asr_degeneracy_risk: float = 0.0,
) -> dict[str, Any]:
    record = _line_candidate(
        candidate,
        SPEECH_AUDIO_CONTENT_LINE,
        visual_delta_strength=visual_delta_strength,
        visual_context_similarity=visual_context_similarity,
        audio_line_quality_profile=audio_line_quality_profile,
    )
    record["difference"] = {
        "type": "audio_event",
        "from": "; ".join(_normalize_list(reference.get("audio_events", []))[:2]) or "reference audio",
        "to": "; ".join(_normalize_list(target.get("audio_events", []))[:2]) or "target audio",
        "description": "the non-speech audio event differs between the reference and target clips",
    }
    quality = dict(record.get("quality", {}))
    audio_text = _annotation_text(reference, ("audio_events", "music_description")) + " " + _annotation_text(target, ("audio_events", "music_description"))
    subtype = "music" if any(term in audio_text.lower() for term in ("music", "song", "sing", "guitar", "piano", "melody")) else "sound_event"
    quality.update(
        {
            "difference_type": "audio_event",
            "has_audio_modality": 1.0,
            "non_speech_audio_event_score": round(score, 3),
            "audio_content_delta_strength": round(score, 3),
            "b_subtype": subtype,
            "video_context_type": video_context_type or _b_context_type(reference, target),
            "video_context_strength": video_context_strength or _b_context_strength(reference, target, visual_context_similarity),
            "asr_degeneracy_risk": asr_degeneracy_risk or _b_asr_degeneracy_risk(reference, target),
        }
    )
    record["quality"] = quality
    return record


def _line_candidate_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
    profile = str(quality.get("audio_line_quality_profile", ""))
    if profile in {
        AUDIO_LINE_PROFILE_V4_STRICT,
        AUDIO_LINE_PROFILE_V5_AUDIO_PRIMARY,
        AUDIO_LINE_PROFILE_B_CONTEXT_CVR,
        AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW,
        AUDIO_LINE_PROFILE_B_AUDIO_BLIND_REVIEW_V2,
    }:
        if str(record.get("audio_dataset_line", "")) == VISUAL_AUDIO_ANCHOR_LINE:
            return (
                _score_float(quality.get("visual_delta_strength")),
                _score_float(quality.get("audio_anchor_score")),
                _score_float(record.get("composite_score")),
                str(record.get("proposal_id", "")),
            )
        if str(record.get("audio_dataset_line", "")) == SPEECH_AUDIO_CONTENT_LINE:
            return (
                -_score_float(quality.get("asr_degeneracy_risk")),
                _score_float(quality.get("video_context_strength")),
                _score_float(quality.get("visual_context_similarity")),
                _score_float(quality.get("audio_content_delta_strength")),
                _score_float(record.get("composite_score")),
                str(record.get("proposal_id", "")),
            )
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
    prepare.add_argument("--max-clips", type=int)
    prepare.add_argument("--annotation-search-root", action="append", default=[])
    prepare.add_argument("--force-audio-focused-refresh", action="store_true")
    prepare.add_argument("--no-annotation-reuse", action="store_true")
    prepare.add_argument("--min-clips-per-folder", type=int, default=4)

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
    split.add_argument("--audio-line-quality-profile", default=AUDIO_LINE_PROFILE_DEFAULT)
    split.add_argument("--a-candidate-mode", choices=sorted(A_CANDIDATE_MODES), default=A_CANDIDATE_MODE_HYBRID)
    split.add_argument("--b-candidate-mode", choices=sorted(B_CANDIDATE_MODES), default=B_CANDIDATE_MODE_HYBRID)

    shard = subparsers.add_parser("shard-jsonl")
    shard.add_argument("--input-path", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--shards", type=int, default=1)
    shard.add_argument("--prefix", required=True)

    merge = subparsers.add_parser("merge-line-results")
    merge.add_argument("--run-root", required=True)
    merge.add_argument("--target-a-count", type=int, default=8)
    merge.add_argument("--target-b-count", type=int, default=8)
    merge.add_argument("--keep-all-b", action="store_true")

    inverse = subparsers.add_parser("augment-b-inverse")
    inverse.add_argument("--run-root", required=True)
    inverse.add_argument("--input-path")
    inverse.add_argument("--root")
    inverse.add_argument("--max-records", type=int)
    inverse.add_argument("--base-url", required=True)
    inverse.add_argument("--api-key", default="EMPTY")
    inverse.add_argument("--model", required=True)
    inverse.add_argument("--timeout-seconds", type=float, default=180.0)
    inverse.add_argument("--omni-retries", type=int, default=2)
    inverse.add_argument("--fail-on-transient-omni-errors", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-existing":
        result = prepare_existing_single_source_clips(
            root=args.root,
            single_source_root=args.single_source_root,
            run_root=args.run_root,
            max_source_folders=args.max_source_folders,
            max_clips=args.max_clips,
            annotation_search_roots=args.annotation_search_root,
            force_audio_focused_refresh=args.force_audio_focused_refresh,
            reuse_annotations=not args.no_annotation_reuse,
            min_clips_per_folder=args.min_clips_per_folder,
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
            audio_line_quality_profile=args.audio_line_quality_profile,
            a_candidate_mode=args.a_candidate_mode,
            b_candidate_mode=args.b_candidate_mode,
        )
    elif args.command == "shard-jsonl":
        result = shard_jsonl(input_path=args.input_path, output_dir=args.output_dir, shards=args.shards, prefix=args.prefix)
    elif args.command == "merge-line-results":
        result = merge_line_results(
            run_root=args.run_root,
            target_a_count=args.target_a_count,
            target_b_count=args.target_b_count,
            keep_all_b=args.keep_all_b,
        )
    elif args.command == "augment-b-inverse":
        result = augment_b_inverse(
            run_root=args.run_root,
            input_path=args.input_path,
            root=args.root,
            max_records=args.max_records,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            omni_retries=args.omni_retries,
            fail_on_transient_omni_errors=args.fail_on_transient_omni_errors,
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
