from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.composed_data import (
    AUDIO_MATTERS_ACCEPTANCE_PROFILE,
    _action_evidence_score,
    _build_proposal_id,
    _candidate_composite_score,
    _detect_primary_difference,
    _difference_evidence_from_annotations,
    _difference_strength_score,
    _display_path,
    _edit_match_score,
    _load_jsonl,
    _resolve_under_root,
    _same_context_score,
    _score_float,
    _select_hard_negative_annotations,
    _select_final_accepted_records,
    _source_temporal_context,
    _target_uniqueness_score,
    _write_jsonl,
)


VISUAL_AUDIO_MATTERS_TYPES = ("attribute", "object_presence", "object_count", "action", "scene")
NON_VISUAL_PRIMARY_TYPES = {"audio_event", "speech", "visible_text"}
DEFAULT_MIN_AUDIO_ANCHOR_SCORE = 0.86
DEFAULT_MIN_AUDIO_RMS = 0.001
DEFAULT_MIN_DIFFERENCE_STRENGTH = 0.60
DEFAULT_MAX_LOCAL_COMPARISONS = 20000


@dataclass(frozen=True)
class AudioFeature:
    vector: np.ndarray
    rms: float
    duration_seconds: float
    sample_count: int


def extract_audio_feature(
    video_path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = 16000,
    max_seconds: float = 8.0,
    timeout_seconds: float = 30.0,
) -> AudioFeature | None:
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).exists():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-t",
        str(max_seconds),
        "-f",
        "f32le",
        "-",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    if samples.size < max(1, sample_rate // 2):
        return None
    samples = np.nan_to_num(samples)
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if not math.isfinite(rms):
        return None
    vector = _audio_signature_vector(samples)
    if vector is None:
        return None
    return AudioFeature(
        vector=vector,
        rms=rms,
        duration_seconds=float(samples.size) / float(sample_rate),
        sample_count=int(samples.size),
    )


def _audio_signature_vector(samples: np.ndarray) -> np.ndarray | None:
    if samples.size == 0:
        return None
    centered = samples.astype(np.float32, copy=False)
    centered = centered - float(np.mean(centered))
    if float(np.max(np.abs(centered))) <= 1e-8:
        return None

    envelope_bins = 32
    envelope: list[float] = []
    for chunk in np.array_split(centered, envelope_bins):
        if chunk.size == 0:
            envelope.append(0.0)
        else:
            envelope.append(float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64))))

    max_fft_samples = min(centered.size, 131072)
    fft_input = centered[:max_fft_samples]
    window = np.hanning(fft_input.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(fft_input * window))
    spectrum = np.log1p(spectrum)
    spectral_bins = 64
    bands: list[float] = []
    for band in np.array_split(spectrum, spectral_bins):
        bands.append(float(np.mean(band)) if band.size else 0.0)

    vector = np.asarray(envelope + bands, dtype=np.float32)
    vector = vector - float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def audio_anchor_score(left: AudioFeature, right: AudioFeature) -> float:
    left_norm = float(np.linalg.norm(left.vector))
    right_norm = float(np.linalg.norm(right.vector))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    score = float(np.dot(left.vector, right.vector) / (left_norm * right_norm))
    return max(0.0, min(1.0, score))


def mine_audio_matters_candidates(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    clip_groups_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    max_candidates: int = 240,
    min_audio_anchor_score: float = DEFAULT_MIN_AUDIO_ANCHOR_SCORE,
    min_audio_rms: float = DEFAULT_MIN_AUDIO_RMS,
    min_difference_strength: float = DEFAULT_MIN_DIFFERENCE_STRENGTH,
    max_local_comparisons: int = DEFAULT_MAX_LOCAL_COMPARISONS,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = 16000,
    max_audio_seconds: float = 8.0,
    acceptance_profile: str = AUDIO_MATTERS_ACCEPTANCE_PROFILE,
    audio_workers: int = 1,
    audio_feature_loader: Callable[[Path], AudioFeature | None] | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    groups = list(_load_jsonl(Path(clip_groups_path)))
    if not annotations:
        raise ValueError("clip annotations are empty")
    if not groups:
        raise ValueError("clip groups are empty")

    annotations_by_id = {
        str(annotation.get("clip_id", "")).strip(): annotation
        for annotation in annotations
        if str(annotation.get("clip_id", "")).strip()
    }
    feature_cache: dict[str, AudioFeature | None] = {}
    loader = audio_feature_loader
    if loader is None:
        loader = lambda path: extract_audio_feature(
            path,
            ffmpeg=ffmpeg,
            sample_rate=sample_rate,
            max_seconds=max_audio_seconds,
        )
    if audio_workers > 1:
        _preload_audio_features(
            root=root_path,
            annotations_by_id=annotations_by_id,
            feature_cache=feature_cache,
            audio_feature_loader=loader,
            audio_workers=audio_workers,
        )

    candidates: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    group_candidate_counts: Counter[str] = Counter()
    comparison_count = 0
    seen_pair_keys: set[tuple[str, str]] = set()

    print(
        "[audio-matters-natural] start mining "
        f"annotations={len(annotations)} groups={len(groups)} min_audio_anchor_score={min_audio_anchor_score}",
        file=sys.stderr,
        flush=True,
    )

    for group_index, group in enumerate(groups, start=1):
        group_id = str(group.get("group_id", f"group_{group_index}")).strip()
        clip_ids = [
            str(item).strip()
            for item in group.get("candidate_clip_ids", [])
            if str(item).strip() in annotations_by_id
        ]
        group_candidates_before = len(candidates)
        print(
            f"[audio-matters-natural] group {group_index}/{len(groups)} group_id={group_id} clips={len(clip_ids)}",
            file=sys.stderr,
            flush=True,
        )
        for left_index, left_clip_id in enumerate(clip_ids):
            for right_clip_id in clip_ids[left_index + 1 :]:
                if comparison_count >= max_local_comparisons:
                    break
                pair_key = tuple(sorted((left_clip_id, right_clip_id)))
                if pair_key in seen_pair_keys:
                    continue
                seen_pair_keys.add(pair_key)
                comparison_count += 1
                left = annotations_by_id[left_clip_id]
                right = annotations_by_id[right_clip_id]
                forward = _build_audio_matters_mined_record(
                    root=root_path,
                    reference_annotation=left,
                    target_annotation=right,
                    annotations=annotations,
                    feature_cache=feature_cache,
                    audio_feature_loader=loader,
                    min_audio_anchor_score=min_audio_anchor_score,
                    min_audio_rms=min_audio_rms,
                    min_difference_strength=min_difference_strength,
                    group=group,
                    acceptance_profile=acceptance_profile,
                    rejection_counts=rejection_counts,
                )
                backward = _build_audio_matters_mined_record(
                    root=root_path,
                    reference_annotation=right,
                    target_annotation=left,
                    annotations=annotations,
                    feature_cache=feature_cache,
                    audio_feature_loader=loader,
                    min_audio_anchor_score=min_audio_anchor_score,
                    min_audio_rms=min_audio_rms,
                    min_difference_strength=min_difference_strength,
                    group=group,
                    acceptance_profile=acceptance_profile,
                    rejection_counts=rejection_counts,
                )
                chosen = _select_better_mined_record(forward, backward)
                if chosen is not None:
                    candidates.append(chosen)
                    group_candidate_counts[group_id] += 1
                    print(
                        "[audio-matters-natural] accepted candidate "
                        f"candidate_id={chosen['candidate_id']} audio_anchor_score={chosen['quality']['audio_anchor_score']} "
                        f"difference_type={chosen['difference']['type']}",
                        file=sys.stderr,
                        flush=True,
                    )
            if comparison_count >= max_local_comparisons:
                break
        print(
            "[audio-matters-natural] group done "
            f"group_id={group_id} new_candidates={len(candidates) - group_candidates_before}",
            file=sys.stderr,
            flush=True,
        )
        if comparison_count >= max_local_comparisons:
            print(
                f"[audio-matters-natural] reached max_local_comparisons={max_local_comparisons}",
                file=sys.stderr,
                flush=True,
            )
            break

    candidates.sort(
        key=lambda item: (
            -_score_float(item.get("scores", {}).get("local_candidate_score")),
            item["candidate_id"],
        )
    )
    selected = candidates[:max_candidates]
    _write_jsonl(Path(output_path), selected)
    report = _build_audio_matters_report(
        annotations_count=len(annotations),
        group_count=len(groups),
        comparison_count=comparison_count,
        candidates=candidates,
        selected=selected,
        rejection_counts=rejection_counts,
        group_candidate_counts=group_candidate_counts,
        min_audio_anchor_score=min_audio_anchor_score,
        min_audio_rms=min_audio_rms,
        min_difference_strength=min_difference_strength,
    )
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(report, encoding="utf-8")

    summary = {
        "clip_annotations_path": str(clip_annotations_path),
        "clip_groups_path": str(clip_groups_path),
        "output_path": str(output_path),
        "report_path": str(report_path or ""),
        "annotation_count": len(annotations),
        "group_count": len(groups),
        "comparison_count": comparison_count,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "rejection_counts": dict(rejection_counts),
        "difference_type_counts": dict(Counter(item["difference"]["type"] for item in selected)),
        "min_audio_anchor_score": min_audio_anchor_score,
        "min_audio_rms": min_audio_rms,
        "min_difference_strength": min_difference_strength,
        "audio_workers": audio_workers,
        "audio_source": "actual_extracted_clip_audio_via_ffmpeg",
        "actual_audio_feature_summary": _audio_feature_summary(
            feature_cache=feature_cache,
            min_audio_rms=min_audio_rms,
        ),
        "annotation_audio_signal_summary": _annotation_audio_signal_summary(annotations),
    }
    print(
        "[audio-matters-natural] wrote "
        f"selected={len(selected)} output_path={output_path}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def _audio_feature_summary(
    *,
    feature_cache: dict[str, AudioFeature | None],
    min_audio_rms: float,
) -> dict[str, Any]:
    features = [feature for feature in feature_cache.values() if feature is not None]
    rms_values = [feature.rms for feature in features]
    low_rms_count = sum(1 for value in rms_values if value < min_audio_rms)
    return {
        "checked_clip_count": len(feature_cache),
        "feature_ok_count": len(features),
        "missing_or_unreadable_audio_count": len(feature_cache) - len(features),
        "low_rms_count": low_rms_count,
        "usable_audio_count": len(features) - low_rms_count,
        "rms_min": round(min(rms_values), 6) if rms_values else 0.0,
        "rms_avg": round(float(sum(rms_values) / len(rms_values)), 6) if rms_values else 0.0,
        "rms_max": round(max(rms_values), 6) if rms_values else 0.0,
    }


def _annotation_audio_signal_summary(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    audio_events_count = 0
    speech_count = 0
    audio_modality_count = 0
    for annotation in annotations:
        audio_events = annotation.get("audio_events")
        if isinstance(audio_events, list) and any(str(item).strip() for item in audio_events):
            audio_events_count += 1
        speech = annotation.get("speech")
        if isinstance(speech, list) and any(str(item).strip() for item in speech):
            speech_count += 1
        modalities = annotation.get("modalities")
        if isinstance(modalities, list) and any(str(item).strip().lower() == "audio" for item in modalities):
            audio_modality_count += 1
    return {
        "annotation_count": len(annotations),
        "audio_events_nonempty_count": audio_events_count,
        "speech_nonempty_count": speech_count,
        "audio_modality_count": audio_modality_count,
        "note": "These fields are reused only as annotation context; audio_anchor_score is recomputed from actual clip audio.",
    }


def _preload_audio_features(
    *,
    root: Path,
    annotations_by_id: dict[str, dict[str, Any]],
    feature_cache: dict[str, AudioFeature | None],
    audio_feature_loader: Callable[[Path], AudioFeature | None],
    audio_workers: int,
) -> None:
    jobs: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, audio_workers)) as executor:
        for clip_id, annotation in sorted(annotations_by_id.items()):
            path = _resolve_under_root(root, str(annotation.get("output_path", "")).strip())
            if not path.exists():
                feature_cache[clip_id] = None
                continue
            jobs[executor.submit(audio_feature_loader, path)] = clip_id
        completed_count = 0
        total_count = len(jobs)
        for future in as_completed(jobs):
            clip_id = jobs[future]
            try:
                feature_cache[clip_id] = future.result()
            except Exception:
                feature_cache[clip_id] = None
            completed_count += 1
            if completed_count == total_count or completed_count % 50 == 0:
                print(
                    "[audio-matters-natural] audio feature preload "
                    f"{completed_count}/{total_count}",
                    file=sys.stderr,
                    flush=True,
                )


def _build_audio_matters_mined_record(
    *,
    root: Path,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
    feature_cache: dict[str, AudioFeature | None],
    audio_feature_loader: Callable[[Path], AudioFeature | None],
    min_audio_anchor_score: float,
    min_audio_rms: float,
    min_difference_strength: float,
    group: dict[str, Any],
    acceptance_profile: str,
    rejection_counts: Counter[str],
) -> dict[str, Any] | None:
    reference_clip_id = str(reference_annotation.get("clip_id", "")).strip()
    target_clip_id = str(target_annotation.get("clip_id", "")).strip()
    if not reference_clip_id or not target_clip_id or reference_clip_id == target_clip_id:
        rejection_counts["invalid_clip_ids"] += 1
        return None
    reference_path = _resolve_under_root(root, str(reference_annotation.get("output_path", "")).strip())
    target_path = _resolve_under_root(root, str(target_annotation.get("output_path", "")).strip())
    if not reference_path.exists() or not target_path.exists():
        rejection_counts["missing_video"] += 1
        return None

    reference_feature = _feature_for_clip(reference_clip_id, reference_path, feature_cache, audio_feature_loader)
    target_feature = _feature_for_clip(target_clip_id, target_path, feature_cache, audio_feature_loader)
    if reference_feature is None or target_feature is None:
        rejection_counts["missing_audio_feature"] += 1
        return None
    if reference_feature.rms < min_audio_rms or target_feature.rms < min_audio_rms:
        rejection_counts["low_audio_rms"] += 1
        return None
    anchor_score = audio_anchor_score(reference_feature, target_feature)
    if anchor_score < min_audio_anchor_score:
        rejection_counts["low_audio_anchor_score"] += 1
        return None

    primary = _detect_primary_difference(
        reference_annotation,
        target_annotation,
        priority_order=VISUAL_AUDIO_MATTERS_TYPES,
    )
    if primary is None:
        rejection_counts["no_visual_difference"] += 1
        return None
    changed_types = list(primary.pop("changed_types"))
    difference_type = str(primary.get("type", "")).strip()
    if difference_type not in VISUAL_AUDIO_MATTERS_TYPES:
        rejection_counts["non_visual_primary_difference"] += 1
        return None
    if set(changed_types) & NON_VISUAL_PRIMARY_TYPES:
        rejection_counts["audio_text_secondary_delta"] += 1
        return None

    semantic_context_score = _same_context_score(reference_annotation, target_annotation)
    temporal_context = _source_temporal_context(reference_annotation, target_annotation, default_score=0.35)
    temporal_score = _score_float(temporal_context.get("score"))
    audio_context_score = max(0.0, min(0.96, 0.35 + anchor_score * 0.60))
    same_context_score = max(semantic_context_score, temporal_score, audio_context_score)
    edit_match_score = _edit_match_score(
        same_context_score=same_context_score,
        primary_difference_type=difference_type,
        changed_types=changed_types,
    )
    hard_negative_annotations = _select_hard_negative_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        primary_difference=primary,
    )
    if len(hard_negative_annotations) < 2:
        rejection_counts["insufficient_hard_negatives"] += 1
        return None
    target_uniqueness_score = _target_uniqueness_score(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        primary_difference=primary,
    )
    difference_strength_score = _difference_strength_score(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference=primary,
        changed_types=changed_types,
    )
    if difference_strength_score < min_difference_strength:
        rejection_counts["weak_visual_difference"] += 1
        return None

    source_context = {
        "relation": "natural_audio_anchor",
        "score": round(audio_context_score, 3),
        "audio_anchor_required": True,
        "audio_anchor_type": "similar_or_same_natural_audio",
        "audio_anchor_score": round(anchor_score, 6),
        "target_audio_mode": "original",
        "edit_primary_modality": "visual",
        "base_temporal_context": temporal_context,
        "group_id": str(group.get("group_id", "")).strip(),
        "group_reason": str(group.get("group_reason", "")).strip(),
        "dataset": str(reference_annotation.get("dataset") or target_annotation.get("dataset") or "").strip(),
    }
    quality = {
        "same_context_score": round(same_context_score, 3),
        "semantic_context_score": round(semantic_context_score, 3),
        "audio_anchor_context_score": round(audio_context_score, 3),
        "edit_match_score": round(edit_match_score, 3),
        "target_uniqueness_score": round(target_uniqueness_score, 3),
        "difference_strength_score": round(difference_strength_score, 3),
        "difference_type": difference_type,
        "audio_anchor_required": 1.0,
        "audio_anchor_score": round(anchor_score, 6),
        "audio_anchor_min_rms": round(min(reference_feature.rms, target_feature.rms), 6),
        "audio_primary_allowed": 0.0,
        "edit_primary_modality": "visual",
        "acceptance_profile": acceptance_profile,
    }
    if difference_type == "action":
        quality["action_evidence_score"] = _action_evidence_score(reference_annotation, target_annotation)
    scores = {
        "same_context_score": quality["same_context_score"],
        "semantic_context_score": quality["semantic_context_score"],
        "audio_anchor_score": quality["audio_anchor_score"],
        "edit_match_score": quality["edit_match_score"],
        "target_uniqueness_score": quality["target_uniqueness_score"],
        "difference_strength_score": quality["difference_strength_score"],
    }
    local_score = _candidate_composite_score(quality, source_context)
    local_score += anchor_score * 0.08
    scores["local_candidate_score"] = round(local_score, 4)

    reference_display_path = _display_path(root, reference_path)
    target_display_path = _display_path(root, target_path)
    candidate_id = _build_proposal_id(reference_display_path, target_display_path)
    return {
        "candidate_id": candidate_id,
        "proposal_id": candidate_id,
        "candidate_kind": "audio_matters_natural",
        "reference_clip_id": reference_clip_id,
        "target_clip_id": target_clip_id,
        "difference": primary,
        "changed_difference_types": changed_types,
        "hard_negative_clip_ids": [str(item.get("clip_id", "")).strip() for item in hard_negative_annotations[:3]],
        "quality": quality,
        "scores": scores,
        "source_context": source_context,
        "evidence": _difference_evidence_from_annotations(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            primary_difference=primary,
        ),
        "risk_flags": [],
    }


def _feature_for_clip(
    clip_id: str,
    path: Path,
    feature_cache: dict[str, AudioFeature | None],
    audio_feature_loader: Callable[[Path], AudioFeature | None],
) -> AudioFeature | None:
    if clip_id not in feature_cache:
        feature_cache[clip_id] = audio_feature_loader(path)
    return feature_cache[clip_id]


def _select_better_mined_record(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if left is None:
        return right
    if right is None:
        return left
    left_score = _score_float(left.get("scores", {}).get("local_candidate_score"))
    right_score = _score_float(right.get("scores", {}).get("local_candidate_score"))
    if right_score > left_score:
        return right
    return left


def _build_audio_matters_report(
    *,
    annotations_count: int,
    group_count: int,
    comparison_count: int,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    rejection_counts: Counter[str],
    group_candidate_counts: Counter[str],
    min_audio_anchor_score: float,
    min_audio_rms: float,
    min_difference_strength: float,
) -> str:
    lines = [
        "# Audio-Matters Natural Candidate Mining Report",
        "",
        "## Summary",
        "",
        f"- annotations: `{annotations_count}`",
        f"- groups: `{group_count}`",
        f"- comparisons: `{comparison_count}`",
        f"- candidate_count: `{len(candidates)}`",
        f"- selected_count: `{len(selected)}`",
        f"- min_audio_anchor_score: `{min_audio_anchor_score}`",
        f"- min_audio_rms: `{min_audio_rms}`",
        f"- min_difference_strength: `{min_difference_strength}`",
        "",
        "## Selected Difference Types",
        "",
    ]
    for key, value in Counter(item["difference"]["type"] for item in selected).most_common():
        lines.append(f"- `{key}`: `{value}`")
    if not selected:
        lines.append("- none")
    lines.extend(["", "## Rejections", ""])
    for key, value in rejection_counts.most_common():
        lines.append(f"- `{key}`: `{value}`")
    if not rejection_counts:
        lines.append("- none")
    lines.extend(["", "## Top Groups", ""])
    for key, value in group_candidate_counts.most_common(20):
        lines.append(f"- `{key}`: `{value}`")
    if not group_candidate_counts:
        lines.append("- none")
    lines.extend(["", "## Top Candidates", ""])
    for candidate in selected[:30]:
        quality = candidate.get("quality", {})
        difference = candidate.get("difference", {})
        lines.append(
            "- "
            f"`{candidate.get('candidate_id', '')}` "
            f"type=`{difference.get('type', '')}` "
            f"audio=`{quality.get('audio_anchor_score', 0.0)}` "
            f"score=`{candidate.get('scores', {}).get('local_candidate_score', 0.0)}` "
            f"`{difference.get('from', '')}` -> `{difference.get('to', '')}`"
        )
    if not selected:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def export_audio_matters_triplets(
    *,
    root: str | Path,
    accepted_pairs_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    records = list(_load_jsonl(Path(accepted_pairs_path)))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    skipped_counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            if not bool(record.get("accepted", True)):
                skipped_counts["not_accepted"] += 1
                continue
            reference_video_raw = str(record.get("reference_video", "")).strip()
            target_video_raw = str(record.get("target_video", "")).strip()
            edit_text = str(record.get("edit_text", "")).strip()
            if not reference_video_raw or not target_video_raw or not edit_text:
                skipped_counts["missing_core_field"] += 1
                continue
            reference_path = _resolve_under_root(root_path, reference_video_raw)
            target_path = _resolve_under_root(root_path, target_video_raw)
            if not reference_path.exists() or not target_path.exists():
                skipped_counts["missing_video"] += 1
                continue
            difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
            visual_delta_type = str(difference.get("type", "")).strip()
            quality = _audio_quality_payload(record)
            hard_negatives = _resolved_hard_negative_paths(root_path, record)
            sample_id = str(record.get("proposal_id", "")).strip() or f"audio_matters_{index:04d}"
            output_record = {
                "sample_id": sample_id,
                "reference_video": str(reference_path),
                "target_video": str(target_path),
                "edit_text": edit_text,
                "reference_caption": str(record.get("reference_caption", "")).strip(),
                "source": record.get("source", {}),
                "difference_type": visual_delta_type,
                "visual_delta_type": visual_delta_type,
                "accepted": bool(record.get("accepted", True)),
                "final_omni_accept": bool(record.get("accepted", True)),
                "final_omni_quality_score": _score_float(record.get("quality", {}).get("final_omni_quality_score")),
                "reference_clip_id": str(record.get("reference_clip_id", "")).strip(),
                "target_clip_id": str(record.get("target_clip_id", "")).strip(),
                "hard_negatives": hard_negatives,
                "audio_anchor_required": True,
                "audio_anchor_score": quality["audio_anchor_score"],
                "audio_anchor_type": quality["audio_anchor_type"],
                "source_pair_proposal_id": str(record.get("proposal_id", "")).strip(),
            }
            output_records.append(output_record)
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()
            edit_preview = edit_text.replace("\n", " ")
            if len(edit_preview) > 160:
                edit_preview = edit_preview[:157].rstrip() + "..."
            print(
                "[audio-matters-natural] GENERATED_TRIPLET "
                f"index={len(output_records)} sample_id={sample_id} "
                f"difference_type={visual_delta_type} "
                f"audio_anchor_score={quality['audio_anchor_score']} "
                f"reference_video={reference_path} target_video={target_path} "
                f"edit_text={edit_preview}",
                file=sys.stderr,
                flush=True,
            )
    summary = {
        "accepted_pairs_path": str(accepted_pairs_path),
        "output_path": str(output_path),
        "input_count": len(records),
        "output_count": len(output_records),
        "skipped_counts": dict(skipped_counts),
        "difference_type_counts": dict(Counter(item["difference_type"] for item in output_records)),
        "contains_target_caption": False,
        "contains_visual_delta_type": True,
        "contains_hard_negatives": True,
    }
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[audio-matters-natural] exported triplets={len(output_records)} output_path={output_path}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def split_mined_candidates(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    shard_count: int,
    summary_path: str | Path | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    records = list(_load_jsonl(Path(input_path)))
    if max_records is not None and max_records > 0:
        records = records[:max_records]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    if not records:
        for shard_index in range(shard_count):
            shard_path = output_root / f"audio_matters_mined_candidates_shard_{shard_index:03d}.jsonl"
            _write_jsonl(shard_path, [])
            shard_paths.append(str(shard_path))
    else:
        chunk_size = max(1, math.ceil(len(records) / shard_count))
        for shard_index in range(shard_count):
            start = shard_index * chunk_size
            end = start + chunk_size
            shard_records = records[start:end]
            shard_path = output_root / f"audio_matters_mined_candidates_shard_{shard_index:03d}.jsonl"
            _write_jsonl(shard_path, shard_records)
            shard_paths.append(str(shard_path))
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_root),
        "input_count": len(records),
        "shard_count": shard_count,
        "shard_paths": shard_paths,
        "shard_counts": [len(_load_jsonl(Path(path))) for path in shard_paths],
        "max_records": max_records,
    }
    if summary_path:
        _write_summary(summary_path, summary)
    print(
        "[audio-matters-natural] split candidates "
        f"records={len(records)} shards={shard_count} output_dir={output_root}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def merge_pair_proposals(
    *,
    input_paths: list[str | Path],
    output_path: str | Path,
    accepted_output_path: str | Path,
    summary_path: str | Path | None = None,
    max_accepted_pairs: int = 80,
    acceptance_profile: str = AUDIO_MATTERS_ACCEPTANCE_PROFILE,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_count = 0
    input_counts: dict[str, int] = {}
    for raw_path in input_paths:
        path = Path(raw_path)
        shard_records = list(_load_jsonl(path))
        input_counts[str(path)] = len(shard_records)
        for record in shard_records:
            proposal_id = str(record.get("proposal_id", "")).strip()
            if proposal_id and proposal_id in seen_ids:
                duplicate_count += 1
                continue
            if proposal_id:
                seen_ids.add(proposal_id)
            records.append(record)
    accepted_records = _select_final_accepted_records(
        records,
        max_accepted_pairs=max_accepted_pairs,
        acceptance_profile=acceptance_profile,
    )
    _write_jsonl(Path(output_path), records)
    _write_jsonl(Path(accepted_output_path), accepted_records)
    summary = {
        "input_paths": [str(path) for path in input_paths],
        "input_counts": input_counts,
        "output_path": str(output_path),
        "accepted_output_path": str(accepted_output_path),
        "proposal_count": len(records),
        "accepted_count": len(accepted_records),
        "duplicate_count": duplicate_count,
        "max_accepted_pairs": max_accepted_pairs,
        "acceptance_profile": acceptance_profile,
    }
    if summary_path:
        _write_summary(summary_path, summary)
    print(
        "[audio-matters-natural] merged proposal shards "
        f"proposal_count={len(records)} accepted_count={len(accepted_records)}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def _resolved_hard_negative_paths(root: Path, record: dict[str, Any]) -> list[str]:
    raw_values = record.get("hard_negatives")
    if not isinstance(raw_values, list):
        raw_values = record.get("hard_negative_paths")
    if not isinstance(raw_values, list):
        return []
    paths: list[str] = []
    for raw_value in raw_values:
        raw_path = str(raw_value).strip()
        if not raw_path:
            continue
        paths.append(str(_resolve_under_root(root, raw_path)))
    return paths


def _audio_quality_payload(record: dict[str, Any]) -> dict[str, Any]:
    for container_name in ("heuristic_quality", "quality", "source_context"):
        container = record.get(container_name)
        if not isinstance(container, dict):
            continue
        score = container.get("audio_anchor_score")
        if score is not None:
            return {
                "audio_anchor_score": _score_float(score),
                "audio_anchor_type": str(container.get("audio_anchor_type", "similar_or_same_natural_audio")).strip()
                or "similar_or_same_natural_audio",
            }
    return {"audio_anchor_score": 0.0, "audio_anchor_type": "similar_or_same_natural_audio"}


def _write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine natural audio-matters composed retrieval samples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine_parser = subparsers.add_parser("mine-candidates")
    mine_parser.add_argument("--root", required=True)
    mine_parser.add_argument("--clip-annotations-path", required=True)
    mine_parser.add_argument("--clip-groups-path", required=True)
    mine_parser.add_argument("--output-path", required=True)
    mine_parser.add_argument("--report-path", required=True)
    mine_parser.add_argument("--summary-path")
    mine_parser.add_argument("--max-candidates", type=int, default=240)
    mine_parser.add_argument("--min-audio-anchor-score", type=float, default=DEFAULT_MIN_AUDIO_ANCHOR_SCORE)
    mine_parser.add_argument("--min-audio-rms", type=float, default=DEFAULT_MIN_AUDIO_RMS)
    mine_parser.add_argument("--min-difference-strength", type=float, default=DEFAULT_MIN_DIFFERENCE_STRENGTH)
    mine_parser.add_argument("--max-local-comparisons", type=int, default=DEFAULT_MAX_LOCAL_COMPARISONS)
    mine_parser.add_argument("--ffmpeg", default="ffmpeg")
    mine_parser.add_argument("--sample-rate", type=int, default=16000)
    mine_parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    mine_parser.add_argument("--acceptance-profile", default=AUDIO_MATTERS_ACCEPTANCE_PROFILE)
    mine_parser.add_argument("--audio-workers", type=int, default=1)

    export_parser = subparsers.add_parser("export-triplets")
    export_parser.add_argument("--root", required=True)
    export_parser.add_argument("--accepted-pairs-path", required=True)
    export_parser.add_argument("--output-path", required=True)
    export_parser.add_argument("--summary-path", required=True)

    split_parser = subparsers.add_parser("split-candidates")
    split_parser.add_argument("--input-path", required=True)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--shard-count", type=int, required=True)
    split_parser.add_argument("--summary-path", required=True)
    split_parser.add_argument("--max-records", type=int)

    merge_parser = subparsers.add_parser("merge-proposals")
    merge_parser.add_argument("--input-path", action="append", default=[])
    merge_parser.add_argument("--input-list-path")
    merge_parser.add_argument("--output-path", required=True)
    merge_parser.add_argument("--accepted-output-path", required=True)
    merge_parser.add_argument("--summary-path", required=True)
    merge_parser.add_argument("--max-accepted-pairs", type=int, default=80)
    merge_parser.add_argument("--acceptance-profile", default=AUDIO_MATTERS_ACCEPTANCE_PROFILE)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "mine-candidates":
        summary = mine_audio_matters_candidates(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            clip_groups_path=args.clip_groups_path,
            output_path=args.output_path,
            report_path=args.report_path,
            max_candidates=args.max_candidates,
            min_audio_anchor_score=args.min_audio_anchor_score,
            min_audio_rms=args.min_audio_rms,
            min_difference_strength=args.min_difference_strength,
            max_local_comparisons=args.max_local_comparisons,
            ffmpeg=args.ffmpeg,
            sample_rate=args.sample_rate,
            max_audio_seconds=args.max_audio_seconds,
            acceptance_profile=args.acceptance_profile,
            audio_workers=args.audio_workers,
        )
        if args.summary_path:
            _write_summary(args.summary_path, summary)
        else:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "export-triplets":
        export_audio_matters_triplets(
            root=args.root,
            accepted_pairs_path=args.accepted_pairs_path,
            output_path=args.output_path,
            summary_path=args.summary_path,
        )
        return
    if args.command == "split-candidates":
        split_mined_candidates(
            input_path=args.input_path,
            output_dir=args.output_dir,
            shard_count=args.shard_count,
            summary_path=args.summary_path,
            max_records=args.max_records,
        )
        return
    if args.command == "merge-proposals":
        input_paths = [str(path).strip() for path in args.input_path if str(path).strip()]
        if args.input_list_path:
            input_paths.extend(
                line.strip()
                for line in Path(args.input_list_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if not input_paths:
            raise ValueError("merge-proposals requires at least one --input-path or --input-list-path")
        merge_pair_proposals(
            input_paths=input_paths,
            output_path=args.output_path,
            accepted_output_path=args.accepted_output_path,
            summary_path=args.summary_path,
            max_accepted_pairs=args.max_accepted_pairs,
            acceptance_profile=args.acceptance_profile,
        )
        return
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
