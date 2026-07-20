from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Iterable

import numpy as np


PRIMARY_MODE = "V_A_T"
REFERENCE_MODE = "V_T"
DEFAULT_FINAL_SEEDS = (13, 23, 42, 71, 101)
BASE_BENCHMARK_REVIEW_CHECKS = (
    "edit_audio_only",
    "reference_does_not_satisfy_edit",
    "target_satisfies_edit",
    "video_only_cannot_identify_target",
    "hard_negatives_do_not_satisfy_edit",
)
EXTENDED_PROMOTION_REVIEW_CHECKS = (
    "audio_change_clearly_audible",
    "video_context_preserved",
    "not_asr_or_transcript_only",
)
DEFAULT_BENCHMARK_SUBTYPE_TARGETS = {
    "sound_event": 90,
    "music": 30,
    "speech_topic_in_video_context": 30,
}
DEFAULT_REVIEW_POOL_TARGETS = {
    "sound_event": 180,
    "music": 70,
    "speech_topic_in_video_context": 180,
}
AUTOMATIC_REVIEW_PROFILE = "audiocvr_benchmark_review_v1"
AUTOMATIC_REVIEW_CRITICAL_FIELDS = (
    "audio_only_pass",
    "video_only_pass",
    "full_av_pass",
    "transcript_like",
    "full_av_required",
)


def prepare_benchmark_review(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    exclude_paths: Iterable[str | Path] = (),
    local_candidate_paths: Iterable[str | Path] = (),
    review_count: int = 225,
    repeat_review_fraction: float = 0.20,
    random_seed: int = 20260719,
    eligible_tiers: Iterable[str] = ("main",),
) -> dict[str, Any]:
    """Prepare a model-blind human review pool for a future test set."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    exclude_paths = tuple(exclude_paths)
    local_candidate_paths = tuple(local_candidate_paths)
    candidates = _read_jsonl(Path(input_path))
    if not candidates:
        raise ValueError(f"no benchmark candidates found in {input_path}")
    local_candidates = _local_candidates_by_sample(_read_many_jsonl(local_candidate_paths))
    if local_candidates:
        candidates = [
            {
                **row,
                "local_same_source_candidates": list(local_candidates.get(_sample_id(row), [])),
            }
            for row in candidates
        ]

    normalized_tiers = _normalize_eligible_tiers(eligible_tiers)
    excluded = _identity_sets(_read_many_jsonl(exclude_paths))
    reasons: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for row in candidates:
        reason = _benchmark_candidate_reject_reason(row, excluded=excluded, eligible_tiers=normalized_tiers)
        sample_id = _sample_id(row)
        if not reason and sample_id in seen_samples:
            reason = "duplicate_sample_id"
        if reason:
            reasons[reason] += 1
            continue
        seen_samples.add(sample_id)
        eligible.append(row)
    if not eligible:
        raise ValueError(f"all benchmark candidates were filtered: {dict(reasons)}")

    eligible.sort(
        key=lambda row: (
            0 if _automatic_split_tier(row) == "main" else 1,
            _stable_row_key(row, random_seed),
        )
    )
    selected = eligible[: min(len(eligible), max(1, int(review_count)))]
    review_rows = [_formal_human_review_row(row, review_round=1) for row in selected]
    repeat_count = min(len(review_rows), max(0, round(len(review_rows) * float(repeat_review_fraction))))
    repeat_rows = [
        {**_formal_human_review_row(row, review_round=2), "repeat_review": True}
        for row in sorted(selected, key=lambda row: _stable_row_key(row, random_seed + 1))[:repeat_count]
    ]

    outputs = {
        "candidate_pool": output_root / "benchmark_review_candidates.jsonl",
        "review_round1": output_root / "human_review_round1.jsonl",
        "review_round2": output_root / "human_review_round2_repeat.jsonl",
        "summary": output_root / "review_preparation_summary.json",
    }
    _write_jsonl(outputs["candidate_pool"], selected)
    _write_jsonl(outputs["review_round1"], review_rows)
    _write_jsonl(outputs["review_round2"], repeat_rows)
    summary = {
        "protocol": "model_blind_test_review_preparation",
        "input_path": str(input_path),
        "exclude_paths": [str(path) for path in exclude_paths],
        "local_candidate_paths": [str(path) for path in local_candidate_paths],
        "local_candidate_query_count": len(local_candidates),
        "local_candidate_count": sum(len(rows) for rows in local_candidates.values()),
        "input_count": len(candidates),
        "eligible_count": len(eligible),
        "review_count": len(selected),
        "repeat_review_count": len(repeat_rows),
        "repeat_review_fraction": float(repeat_review_fraction),
        "eligible_tiers": sorted(normalized_tiers),
        "rejected_counts": dict(sorted(reasons.items())),
        "automatic_tier_distribution": dict(
            sorted(Counter(_automatic_split_tier(row) for row in selected).items())
        ),
        "dataset_distribution": dict(sorted(Counter(_dataset(row) for row in selected).items())),
        "subtype_distribution": dict(sorted(Counter(_subtype(row) for row in selected).items())),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    _write_json(outputs["summary"], summary)
    return summary


def finalize_benchmark(
    *,
    candidate_path: str | Path,
    review_paths: Iterable[str | Path],
    output_dir: str | Path,
    exclude_paths: Iterable[str | Path] = (),
    target_count: int = 150,
    minimum_count: int = 100,
    max_speech_ratio: float = 0.35,
    max_dataset_ratio: float = 0.60,
    max_per_source: int = 1,
    min_strict_local_coverage: float = 0.0,
    random_seed: int = 20260719,
    eligible_tiers: Iterable[str] = ("main",),
) -> dict[str, Any]:
    """Freeze a human-passed, source-disjoint benchmark without model-score selection."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    exclude_paths = tuple(exclude_paths)
    review_paths = tuple(review_paths)
    candidates = _read_jsonl(Path(candidate_path))
    if not candidates:
        raise ValueError(f"no benchmark candidates found in {candidate_path}")
    reviews = _read_many_jsonl(review_paths)
    if not reviews:
        raise ValueError("finalization requires completed human review JSONL")

    normalized_tiers = _normalize_eligible_tiers(eligible_tiers)
    review_by_sample, review_audit = _collate_reviews(reviews)
    excluded = _identity_sets(_read_many_jsonl(exclude_paths))
    rejection_counts: Counter[str] = Counter()
    passed: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    for row in candidates:
        reason = _benchmark_candidate_reject_reason(row, excluded=excluded, eligible_tiers=normalized_tiers)
        sample_id = _sample_id(row)
        review = review_by_sample.get(sample_id)
        if not reason and review is None:
            reason = "not_human_reviewed"
        automatic_tier = _automatic_split_tier(row)
        review_status_key = "promotion_passed" if automatic_tier == "extended" else "passed"
        review_reason_key = "promotion_reason" if automatic_tier == "extended" else "reason"
        if not reason and not review.get(review_status_key, False):
            reason = str(review.get(review_reason_key) or "human_review_not_passed")
        pair_id = _pair_id(row)
        if not reason and pair_id and pair_id in seen_pairs:
            reason = "duplicate_pair_group"
        if reason:
            rejection_counts[reason] += 1
            continue
        if pair_id:
            seen_pairs.add(pair_id)
        passed.append(_benchmark_output_row(row, human_verified_negatives=True))

    selected = _select_balanced_benchmark(
        passed,
        target_count=max(1, int(target_count)),
        max_speech_ratio=float(max_speech_ratio),
        max_dataset_ratio=float(max_dataset_ratio),
        max_per_source=max(1, int(max_per_source)),
        random_seed=int(random_seed),
    )
    if len(selected) < max(1, int(minimum_count)):
        raise ValueError(
            f"only {len(selected)} reviewed candidates satisfy benchmark constraints; minimum is {minimum_count}"
        )

    strict_local_count = sum(_has_strict_local_negative(row) for row in selected)
    strict_local_coverage = strict_local_count / len(selected)
    leakage = _split_leakage_summary(
        train=_read_many_jsonl(exclude_paths),
        val=[],
        test=selected,
    )
    violations: list[str] = []
    if leakage["violation_count"]:
        violations.append("selected benchmark overlaps excluded train/validation identities")
    if strict_local_coverage < float(min_strict_local_coverage):
        violations.append(
            f"strict local coverage {strict_local_coverage:.4f} is below required {float(min_strict_local_coverage):.4f}"
        )
    if violations:
        raise ValueError(f"benchmark audit failed: {violations}")

    test_path = output_root / "test_main.jsonl"
    _write_jsonl(test_path, selected)
    test_hash = _sha256_file(test_path)
    holdout = _identity_sets(selected)
    holdout_path = output_root / "test_holdout_identities.json"
    _write_json(
        holdout_path,
        {
            "source_ids": sorted(holdout["source"]),
            "pair_group_ids": sorted(holdout["pair"]),
            "sample_ids": sorted(holdout["sample"]),
        },
    )
    manifest = {
        "protocol": "audiocvr_frozen_human_verified_test",
        "selection_uses_model_scores": False,
        "target_count": int(target_count),
        "minimum_count": int(minimum_count),
        "final_count": len(selected),
        "target_count_met": len(selected) >= int(target_count),
        "max_speech_ratio": float(max_speech_ratio),
        "max_dataset_ratio": float(max_dataset_ratio),
        "max_per_source": max(1, int(max_per_source)),
        "eligible_tiers": sorted(normalized_tiers),
        "strict_local_count": strict_local_count,
        "strict_local_coverage": strict_local_coverage,
        "dataset_distribution": dict(sorted(Counter(_dataset(row) for row in selected).items())),
        "subtype_distribution": dict(sorted(Counter(_subtype(row) for row in selected).items())),
        "automatic_tier_distribution": dict(
            sorted(Counter(_automatic_split_tier(row) for row in selected).items())
        ),
        "human_promoted_extended_count": sum(
            _truthy(row.get("human_verified_benchmark_eligible")) for row in selected
        ),
        "human_review": review_audit,
        "rejected_counts": dict(sorted(rejection_counts.items())),
        "leakage": leakage,
        "test_main_sha256": test_hash,
        "outputs": {
            "test_main": str(test_path),
            "manifest": str(output_root / "frozen_benchmark_manifest.json"),
            "sha256": str(output_root / "frozen_benchmark.sha256"),
            "holdout_identities": str(holdout_path),
        },
    }
    _write_json(output_root / "frozen_benchmark_manifest.json", manifest)
    (output_root / "frozen_benchmark.sha256").write_text(f"{test_hash}  test_main.jsonl\n", encoding="ascii")
    return manifest


def prepare_automatic_benchmark_review(
    *,
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    review_pool_targets: dict[str, int] | None = None,
    random_seed: int = 20260720,
    max_per_source: int = 2,
) -> dict[str, Any]:
    """Merge accepted runs, deduplicate them, and freeze a model-blind Omni review pool."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = tuple(Path(path) for path in input_paths)
    if not paths:
        raise ValueError("at least one --input-path is required")
    raw_rows: list[dict[str, Any]] = []
    for path in paths:
        rows = _read_jsonl(path)
        for row in rows:
            normalized = _normalize_automatic_pool_row(row, input_path=path)
            raw_rows.append(normalized)
    if not raw_rows:
        raise ValueError("automatic benchmark inputs contain no records")

    filtered: list[dict[str, Any]] = []
    rejected_counts: Counter[str] = Counter()
    for row in raw_rows:
        reason = _automatic_pool_filter_reason(row)
        if reason:
            rejected_counts[reason] += 1
        else:
            filtered.append(row)
    deduplicated, duplicate_counts = _deduplicate_automatic_pool(filtered, random_seed=random_seed)
    targets = dict(review_pool_targets or DEFAULT_REVIEW_POOL_TARGETS)
    candidates = _select_automatic_review_pool(
        deduplicated,
        targets=targets,
        max_per_source=max(1, int(max_per_source)),
        random_seed=int(random_seed),
    )
    if not candidates:
        raise ValueError("automatic benchmark review pool became empty")

    combined_path = output_root / "combined_accepted_pool.jsonl"
    dedup_path = output_root / "combined_pool_deduplicated.jsonl"
    candidate_path = output_root / "automatic_review_candidates.jsonl"
    summary_path = output_root / "combined_pool_summary.json"
    _write_jsonl(combined_path, raw_rows)
    _write_jsonl(dedup_path, deduplicated)
    _write_jsonl(candidate_path, candidates)
    summary = {
        "protocol": "audiocvr_automatic_benchmark_pool_v1",
        "selection_uses_model_scores": False,
        "input_paths": [str(path) for path in paths],
        "input_count": len(raw_rows),
        "post_filter_count": len(filtered),
        "deduplicated_count": len(deduplicated),
        "review_candidate_count": len(candidates),
        "filter_rejection_counts": dict(sorted(rejected_counts.items())),
        "duplicate_drop_counts": dict(sorted(duplicate_counts.items())),
        "review_pool_targets": targets,
        "review_pool_subtypes": dict(sorted(Counter(_canonical_subtype(row) for row in candidates).items())),
        "review_pool_datasets": dict(sorted(Counter(_dataset(row) for row in candidates).items())),
        "legacy_asr_risk_is_advisory_only": True,
        "random_seed": int(random_seed),
        "max_per_source_in_review_pool": max(1, int(max_per_source)),
        "outputs": {
            "combined_pool": str(combined_path),
            "deduplicated_pool": str(dedup_path),
            "review_candidates": str(candidate_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def _normalize_automatic_pool_row(row: dict[str, Any], *, input_path: Path) -> dict[str, Any]:
    normalized = dict(row)
    subtype = _canonical_subtype(normalized)
    normalized["b_subtype"] = subtype
    normalized["automatic_split_tier"] = _automatic_split_tier(normalized)
    normalized["dataset"] = _normalized_dataset(normalized)
    normalized["raw_source_id"] = _stable_source_id(normalized)
    normalized["source_disjoint_group_id"] = normalized["raw_source_id"]
    normalized["sample_id"] = _stable_sample_id(normalized)
    normalized["pair_group_id"] = _stable_pair_group_id(normalized)
    normalized["legacy_asr_risk"] = _numeric_field(normalized, "asr_degeneracy_risk")
    normalized["legacy_asr_risk_advisory_only"] = True
    normalized["automatic_review_input_path"] = str(input_path)
    normalized["existing_verifier_complete"] = _existing_verifier_complete(normalized)
    normalized["existing_min_stage_confidence"] = _existing_min_stage_confidence(normalized)
    return normalized


def _automatic_pool_filter_reason(row: dict[str, Any]) -> str:
    if "accepted" in row and not _truthy(row.get("accepted")):
        return "not_accepted"
    if _truthy(row.get("fallback")):
        return "fallback_record"
    if _truthy(row.get("is_inverse")) or str(row.get("direction") or "forward").strip().lower() == "inverse":
        return "inverse_record"
    if _canonical_subtype(row) not in DEFAULT_BENCHMARK_SUBTYPE_TARGETS:
        return "unsupported_subtype"
    if not _first_text(row, ("reference_video", "reference_clip_path")):
        return "missing_reference_video"
    if not _first_text(row, ("target_video", "target_clip_path")):
        return "missing_target_video"
    if not str(row.get("edit_text") or row.get("audio_only_edit_text") or "").strip():
        return "missing_edit_text"
    explicit_failure = _existing_verifier_explicit_failure(row)
    if explicit_failure:
        return explicit_failure
    return ""


def _deduplicate_automatic_pool(
    rows: list[dict[str, Any]], *, random_seed: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    current = list(rows)
    drops: Counter[str] = Counter()
    for reason, key_fn in (
        ("duplicate_sample_id", lambda row: _sample_id(row)),
        ("duplicate_reference_target", _normalized_reference_target_key),
        ("duplicate_pair_group", _pair_id),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        passthrough: list[dict[str, Any]] = []
        for row in current:
            key = str(key_fn(row) or "").strip()
            if key:
                grouped[key].append(row)
            else:
                passthrough.append(row)
        selected = passthrough
        for values in grouped.values():
            values.sort(key=lambda row: _automatic_pool_priority(row, random_seed))
            selected.append(values[0])
            drops[reason] += max(0, len(values) - 1)
        current = selected
    current.sort(key=lambda row: _stable_row_key(row, random_seed))
    return current, drops


def _automatic_pool_priority(row: dict[str, Any], seed: int) -> tuple[Any, ...]:
    tier_rank = {"main": 0, "extended": 1, "diagnostic": 2}.get(_automatic_split_tier(row), 3)
    speech_cap_only = _diagnostic_reasons(row) == {"main_speech_cap_exceeded"}
    return (
        tier_rank,
        -int(speech_cap_only),
        -int(_truthy(row.get("existing_verifier_complete"))),
        -_numeric_field(row, "audio_delta_strength"),
        -_numeric_field(row, "video_context_strength"),
        -float(row.get("existing_min_stage_confidence") or 0.0),
        _stable_row_key(row, seed),
    )


def _select_automatic_review_pool(
    rows: list[dict[str, Any]],
    *,
    targets: dict[str, int],
    max_per_source: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    for subtype in ("music", "sound_event", "speech_topic_in_video_context"):
        target = max(0, int(targets.get(subtype, 0)))
        candidates = [row for row in rows if _canonical_subtype(row) == subtype]
        while candidates and sum(_canonical_subtype(row) == subtype for row in selected) < target:
            eligible = [row for row in candidates if source_counts[_primary_source_id(row)] < max_per_source]
            if not eligible:
                break
            row = min(
                eligible,
                key=lambda item: (
                    dataset_counts[_dataset(item)],
                    *_automatic_pool_priority(item, random_seed),
                ),
            )
            selected.append(row)
            source_counts[_primary_source_id(row)] += 1
            dataset_counts[_dataset(row)] += 1
            candidates.remove(row)
    selected.sort(key=lambda row: (_canonical_subtype(row), _stable_row_key(row, random_seed)))
    return selected


def review_benchmark_omni(
    *,
    candidate_path: str | Path,
    output_path: str | Path,
    media_root: str | Path,
    cache_dir: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    review_pass_id: int = 1,
    pass1_review_paths: Iterable[str | Path] = (),
    repeat_review_fraction: float = 0.20,
    random_seed: int = 20260720,
    shard_index: int = 0,
    shard_count: int = 1,
    timeout_seconds: float = 180.0,
    omni_retries: int = 2,
    resume: bool = False,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    """Run the independent three-stage Omni audit for one deterministic shard."""
    from app.audio_lines_single_source import (
        _call_omni_with_retries,
        _extract_audio_only_cache,
        _extract_video_only_cache,
    )
    from app.composed_omni import OpenAIComposedDataClient

    candidate_file = Path(candidate_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    candidates = _read_jsonl(candidate_file)
    if not candidates:
        raise ValueError(f"no automatic review candidates found in {candidate_file}")
    if int(review_pass_id) not in {1, 2}:
        raise ValueError("review_pass_id must be 1 or 2")
    if int(shard_count) <= 0 or not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")

    if int(review_pass_id) == 2:
        pass1 = _automatic_reviews_by_sample(_read_many_jsonl(pass1_review_paths))
        if not pass1:
            raise ValueError("review pass 2 requires at least one non-empty --pass1-review-path")
        passing = [row for row in candidates if pass1.get(_sample_id(row), {}).get("decision") == "pass"]
        repeat_ids = _deterministic_repeat_ids(
            passing,
            fraction=float(repeat_review_fraction),
            random_seed=int(random_seed),
        )
        candidates = [row for row in candidates if _sample_id(row) in repeat_ids]
    candidates = sorted(candidates, key=lambda row: _sample_id(row))
    candidates = [row for index, row in enumerate(candidates) if index % int(shard_count) == int(shard_index)]

    completed_ids: set[str] = set()
    if resume and output_file.exists():
        existing_rows = _read_jsonl(output_file)
        retained_rows = [row for row in existing_rows if str(row.get("decision") or "") != "error"]
        completed_ids = {_sample_id(row) for row in retained_rows}
        if len(retained_rows) != len(existing_rows):
            _write_jsonl(output_file, retained_rows)
    elif output_file.exists():
        output_file.write_text("", encoding="utf-8")
    else:
        output_file.touch()

    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=float(timeout_seconds),
    )
    cache_root = Path(cache_dir)
    media_root_path = Path(media_root)
    decision_counts: Counter[str] = Counter()
    processed = 0
    for index, row in enumerate(candidates, start=1):
        sample_id = _sample_id(row)
        if sample_id in completed_ids:
            continue
        try:
            reference_path = _resolve_media_path(media_root_path, _first_text(row, ("reference_video", "reference_clip_path")))
            target_path = _resolve_media_path(media_root_path, _first_text(row, ("target_video", "target_clip_path")))
            reference_audio = _extract_audio_only_cache(
                video_path=reference_path,
                cache_dir=cache_root / "audio_only",
                clip_id=f"{sample_id}_reference",
            )
            target_audio = _extract_audio_only_cache(
                video_path=target_path,
                cache_dir=cache_root / "audio_only",
                clip_id=f"{sample_id}_target",
            )
            reference_silent = _extract_video_only_cache(
                video_path=reference_path,
                cache_dir=cache_root / "video_only",
                clip_id=f"{sample_id}_reference",
            )
            target_silent = _extract_video_only_cache(
                video_path=target_path,
                cache_dir=cache_root / "video_only",
                clip_id=f"{sample_id}_target",
            )
            edit_text = str(row.get("edit_text") or row.get("audio_only_edit_text") or "").strip()
            proposal = _automatic_audio_proposal(row)
            local_gate = row.get("local_gate_report") if isinstance(row.get("local_gate_report"), dict) else {}
            audio_verify, raw_audio = _call_omni_with_retries(
                label=f"benchmark_audio_only:{sample_id}:pass{review_pass_id}",
                retries=int(omni_retries),
                fail_on_transient=bool(fail_on_error),
                func=lambda: client.verify_b_line_audio_only_edit(
                    reference_audio_path=str(reference_audio),
                    target_audio_path=str(target_audio),
                    edit_text=edit_text,
                    audio_only_proposal=proposal,
                ),
            )
            video_verify, raw_video = _call_omni_with_retries(
                label=f"benchmark_video_only:{sample_id}:pass{review_pass_id}",
                retries=int(omni_retries),
                fail_on_transient=bool(fail_on_error),
                func=lambda: client.verify_b_line_video_only_shortcut(
                    reference_clip_path=str(reference_silent),
                    target_clip_path=str(target_silent),
                    edit_text=edit_text,
                    audio_only_evidence={"proposal": proposal, "verification": audio_verify},
                    local_gate_report=local_gate,
                ),
            )
            full_av, raw_full_av = _call_omni_with_retries(
                label=f"benchmark_full_av:{sample_id}:pass{review_pass_id}",
                retries=int(omni_retries),
                fail_on_transient=bool(fail_on_error),
                func=lambda: client.verify_b_line_full_av_consistency(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    edit_text=edit_text,
                    audio_only_evidence={"proposal": proposal, "verification": audio_verify},
                    local_gate_report=local_gate,
                ),
            )
            context, raw_context = _call_omni_with_retries(
                label=f"benchmark_context:{sample_id}:pass{review_pass_id}",
                retries=int(omni_retries),
                fail_on_transient=bool(fail_on_error),
                func=lambda: client.audit_audiocvr_benchmark_context(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    edit_text=edit_text,
                    audio_only_evidence={"proposal": proposal, "verification": audio_verify},
                    review_pass_id=int(review_pass_id),
                ),
            )
            review = _automatic_review_decision(
                row,
                audio_verify=audio_verify,
                video_verify=video_verify,
                full_av=full_av,
                context=context,
                review_pass_id=int(review_pass_id),
                model=model,
            )
            review.update(
                {
                    "raw_audio_only_verification": raw_audio,
                    "raw_video_only_verification": raw_video,
                    "raw_full_av_verification": raw_full_av,
                    "raw_context_verification": raw_context,
                }
            )
        except Exception as exc:
            if fail_on_error:
                raise
            review = {
                "sample_id": sample_id,
                "reviewer_type": "omni",
                "review_profile": AUTOMATIC_REVIEW_PROFILE,
                "review_pass_id": int(review_pass_id),
                "model": model,
                "decision": "error",
                "reject_reasons": [f"review_error:{type(exc).__name__}:{exc}"],
            }
        _append_jsonl(output_file, review)
        decision_counts[str(review.get("decision") or "unknown")] += 1
        processed += 1
        print(
            f"[review-benchmark-omni] pass={review_pass_id} shard={shard_index}/{shard_count} "
            f"{index}/{len(candidates)} sample={sample_id} decision={review.get('decision')}",
            file=sys.stderr,
            flush=True,
        )
    summary = {
        "protocol": AUTOMATIC_REVIEW_PROFILE,
        "candidate_path": str(candidate_file),
        "output_path": str(output_file),
        "review_pass_id": int(review_pass_id),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "candidate_count_for_shard": len(candidates),
        "preexisting_completed_count": len(completed_ids),
        "processed_count": processed,
        "decision_counts": dict(sorted(decision_counts.items())),
    }
    _write_json(output_file.with_suffix(".summary.json"), summary)
    return summary


def _automatic_review_decision(
    row: dict[str, Any],
    *,
    audio_verify: dict[str, Any],
    video_verify: dict[str, Any],
    full_av: dict[str, Any],
    context: dict[str, Any],
    review_pass_id: int,
    model: str,
) -> dict[str, Any]:
    subtype = _canonical_subtype(row)
    transcript_like = bool(context.get("transcript_like")) or _transcript_like_text(str(row.get("edit_text") or ""))
    speech_role = str(context.get("speech_role") or ("not_speech" if subtype != "speech_topic_in_video_context" else "asr_only"))
    min_confidence = min(
        float(audio_verify.get("confidence") or 0.0),
        float(video_verify.get("confidence") or 0.0),
        float(full_av.get("confidence") or 0.0),
        float(context.get("confidence") or 0.0),
    )
    audio_pass = (
        bool(audio_verify.get("accept"))
        and not bool(audio_verify.get("reference_satisfies_edit"))
        and bool(audio_verify.get("target_satisfies_edit"))
        and bool(audio_verify.get("audio_difference_specific"))
        and bool(audio_verify.get("edit_text_audio_only"))
    )
    video_pass = (
        bool(video_verify.get("accept"))
        and bool(video_verify.get("visual_context_preserved"))
        and not bool(video_verify.get("visual_shortcut_risk"))
        and not bool(video_verify.get("can_identify_target_without_audio"))
    )
    full_av_pass = (
        bool(full_av.get("accept"))
        and bool(full_av.get("visual_context_preserved"))
        and not bool(full_av.get("visual_shortcut_risk"))
        and bool(full_av.get("audio_edit_still_valid"))
        and bool(context.get("accept"))
        and bool(context.get("visual_context_preserved"))
        and bool(context.get("audio_edit_still_valid"))
    )
    recomputed_asr_risk = float(context.get("recomputed_asr_risk") or 0.0)
    video_context_strength = float(context.get("video_context_strength") or 0.0)
    audio_only_solvability = float(context.get("audio_only_solvability") or 0.0)
    full_av_required = bool(context.get("full_av_required"))
    reject_reasons: list[str] = []
    if not audio_pass:
        reject_reasons.append("audio_only_gate_failed")
    if not video_pass:
        reject_reasons.append("video_only_gate_failed")
    if not full_av_pass:
        reject_reasons.append("full_av_gate_failed")
    if subtype == "speech_topic_in_video_context":
        if speech_role not in {"contextual_speech", "speech_with_event"}:
            reject_reasons.append(f"speech_role:{speech_role}")
        if transcript_like:
            reject_reasons.append("transcript_like")
        if not full_av_required:
            reject_reasons.append("full_av_not_required")
        if recomputed_asr_risk > 0.35:
            reject_reasons.append("asr_risk_above_0.35")
        if video_context_strength < 0.60:
            reject_reasons.append("video_context_below_0.60")
        if _numeric_field(row, "audio_delta_strength") < 0.70:
            reject_reasons.append("audio_delta_below_0.70")
        if audio_only_solvability >= 0.85:
            reject_reasons.append("audio_only_solvability_high")
    if reject_reasons:
        decision = "reject"
    elif min_confidence < 0.65:
        decision = "uncertain"
        reject_reasons.append("minimum_stage_confidence_below_0.65")
    else:
        decision = "pass"
    return {
        "sample_id": _sample_id(row),
        "pair_group_id": _pair_id(row),
        "raw_source_id": _primary_source_id(row),
        "dataset": _dataset(row),
        "b_subtype": subtype,
        "reviewer_type": "omni",
        "review_profile": AUTOMATIC_REVIEW_PROFILE,
        "review_pass_id": int(review_pass_id),
        "model": model,
        "audio_only_pass": audio_pass,
        "video_only_pass": video_pass,
        "full_av_pass": full_av_pass,
        "speech_role": speech_role,
        "transcript_like": transcript_like,
        "full_av_required": full_av_required,
        "recomputed_asr_risk": recomputed_asr_risk,
        "video_context_strength": video_context_strength,
        "audio_only_solvability": audio_only_solvability,
        "min_stage_confidence": min_confidence,
        "decision": decision,
        "reject_reasons": reject_reasons,
        "audio_only_verification": audio_verify,
        "video_only_verification": video_verify,
        "full_av_verification": full_av,
        "context_verification": context,
    }


def finalize_automatic_benchmark(
    *,
    combined_pool_path: str | Path,
    candidate_path: str | Path,
    pass1_review_paths: Iterable[str | Path],
    pass2_review_paths: Iterable[str | Path],
    output_dir: str | Path,
    subtype_targets: dict[str, int] | None = None,
    validation_targets: dict[str, int] | None = None,
    repeat_review_fraction: float = 0.20,
    max_dataset_ratio: float = 0.50,
    relaxed_dataset_ratio: float = 0.55,
    max_hdtf_ratio: float = 0.15,
    max_voxceleb_ratio: float = 0.05,
    max_per_source: int = 1,
    random_seed: int = 20260720,
) -> dict[str, Any]:
    """Freeze an Omni-consensus benchmark, then rebuild source-disjoint train and validation splits."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    combined_pool = _read_jsonl(Path(combined_pool_path))
    candidates = _read_jsonl(Path(candidate_path))
    if not combined_pool or not candidates:
        raise ValueError("automatic finalization requires non-empty combined pool and candidate JSONL")
    combined_pool = [row for row in combined_pool if not _automatic_pool_filter_reason(row)]
    combined_pool, final_pool_duplicate_drops = _deduplicate_automatic_pool(
        combined_pool, random_seed=int(random_seed)
    )
    if not combined_pool:
        raise ValueError("combined pool became empty after final safety filtering and deduplication")
    candidate_ids = [_sample_id(row) for row in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("automatic review candidate pool contains duplicate sample_id values")
    pass1 = _automatic_reviews_by_sample(_read_many_jsonl(pass1_review_paths))
    pass2 = _automatic_reviews_by_sample(_read_many_jsonl(pass2_review_paths))
    if not pass1:
        raise ValueError("automatic finalization requires pass-1 Omni reviews")
    pool_summary_path = Path(combined_pool_path).parent / "combined_pool_summary.json"
    pool_summary = _read_optional_json(pool_summary_path)

    candidate_by_id = {_sample_id(row): row for row in candidates}
    repeat_ids = _deterministic_repeat_ids(
        [candidate_by_id[sample_id] for sample_id, review in pass1.items() if review.get("decision") == "pass" and sample_id in candidate_by_id],
        fraction=float(repeat_review_fraction),
        random_seed=int(random_seed),
    )
    consensus_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    agreement = _automatic_review_agreement(pass1, pass2, repeat_ids)
    for sample_id, row in candidate_by_id.items():
        first = pass1.get(sample_id)
        if first is None:
            rejection_counts["missing_pass1_review"] += 1
            continue
        consensus_reason = _automatic_consensus_reject_reason(
            first,
            pass2.get(sample_id),
            repeated=sample_id in repeat_ids,
        )
        enriched = _row_with_automatic_review(row, first, pass2.get(sample_id), consensus_reason=consensus_reason)
        if consensus_reason:
            rejection_counts[consensus_reason] += 1
            if _automatic_review_is_diagnostic(first, pass2.get(sample_id)):
                diagnostic_rows.append(enriched)
            continue
        consensus_rows.append(enriched)

    pass1_decisions = Counter(str(row.get("decision") or "missing") for row in pass1.values())
    pass2_decisions = Counter(str(row.get("decision") or "missing") for row in pass2.values())
    pass1_reject_reasons = Counter(
        str(reason)
        for row in pass1.values()
        if row.get("decision") != "pass"
        for reason in row.get("reject_reasons", [])
    )
    pass2_reject_reasons = Counter(
        str(reason)
        for row in pass2.values()
        if row.get("decision") != "pass"
        for reason in row.get("reject_reasons", [])
    )
    automatic_review_summary = {
        "protocol": AUTOMATIC_REVIEW_PROFILE,
        "combined_input_count": int(pool_summary.get("input_count", len(combined_pool))),
        "post_filter_count": int(pool_summary.get("post_filter_count", len(combined_pool))),
        "deduplicated_pool_count": len(combined_pool),
        "final_pool_duplicate_drop_counts": dict(sorted(final_pool_duplicate_drops.items())),
        "review_candidate_count": len(candidates),
        "pass1_review_count": len(pass1),
        "pass1_decisions": dict(sorted(pass1_decisions.items())),
        "pass2_review_count": len(pass2),
        "pass2_decisions": dict(sorted(pass2_decisions.items())),
        "repeat_review": agreement,
        "consensus_eligible_count": len(consensus_rows),
        "diagnostic_count_before_freeze": len(diagnostic_rows),
        "consensus_rejection_counts": dict(sorted(rejection_counts.items())),
        "pass1_reject_reasons": dict(sorted(pass1_reject_reasons.items())),
        "pass2_reject_reasons": dict(sorted(pass2_reject_reasons.items())),
        "legacy_asr_risk_distribution": _numeric_distribution(
            [_numeric_field(row, "legacy_asr_risk") for row in candidates]
        ),
        "recomputed_asr_risk_distribution": _numeric_distribution(
            [float(row.get("recomputed_asr_risk") or 0.0) for row in pass1.values()]
        ),
        "selection_uses_retrieval_model_scores": False,
    }

    test_targets = dict(subtype_targets or DEFAULT_BENCHMARK_SUBTYPE_TARGETS)
    val_targets = dict(
        validation_targets
        or {"sound_event": 45, "music": 15, "speech_topic_in_video_context": 15}
    )
    total_target = sum(max(0, int(value)) for value in test_targets.values())
    total_validation_target = sum(max(0, int(value)) for value in val_targets.values())
    selected, selection, validation, validation_selection, split_allocation_order = (
        _select_disjoint_test_validation(
            consensus_rows,
            test_targets=test_targets,
            validation_targets=val_targets,
            max_dataset_ratio=float(max_dataset_ratio),
            relaxed_dataset_ratio=float(relaxed_dataset_ratio),
            max_hdtf_ratio=float(max_hdtf_ratio),
            max_voxceleb_ratio=float(max_voxceleb_ratio),
            max_per_source=max(1, int(max_per_source)),
            random_seed=int(random_seed),
        )
    )
    if len(selected) < total_target:
        raise ValueError(
            f"only {len(selected)} Omni-consensus candidates satisfy benchmark constraints; target is {total_target}; "
            f"selection={selection}"
        )
    if len(validation) < total_validation_target:
        raise ValueError(
            f"only {len(validation)} validation candidates satisfy source-disjoint quota; "
            f"target is {total_validation_target}; allocation={split_allocation_order}; "
            f"selection={validation_selection}"
        )

    holdout_ids = _identity_sets(selected + validation)
    diagnostic_sample_ids = {_sample_id(row) for row in diagnostic_rows}
    train: list[dict[str, Any]] = []
    for row in combined_pool:
        if _row_overlaps_identities(row, holdout_ids) or _sample_id(row) in diagnostic_sample_ids:
            continue
        item = dict(row)
        item["dataset_split"] = "train"
        item["recommended_sampling_weight"] = _recommended_training_sampling_weight(item)
        train.append(item)
    test = [{**row, "dataset_split": "test_main"} for row in selected]
    validation = [{**row, "dataset_split": "val"} for row in validation]
    diagnostic = [{**row, "dataset_split": "diagnostic_asr"} for row in diagnostic_rows]

    leakage = _split_leakage_summary(train=train, val=validation, test=test)
    if leakage["violation_count"]:
        raise ValueError(f"automatic benchmark split leakage detected: {leakage}")
    if len({_pair_id(row) for row in test}) != len(test):
        raise ValueError("duplicate pair_group_id remains in automatic test benchmark")

    test_path = output_root / "test_main_150.jsonl"
    val_path = output_root / "val.jsonl"
    train_path = output_root / "train.jsonl"
    diagnostic_path = output_root / "test_asr_diagnostic.jsonl"
    _write_jsonl(test_path, test)
    _write_jsonl(val_path, validation)
    _write_jsonl(train_path, train)
    _write_jsonl(diagnostic_path, diagnostic)
    test_hash = _sha256_file(test_path)
    holdout = _identity_sets(test)
    _write_json(
        output_root / "test_holdout_identities.json",
        {
            "source_ids": sorted(holdout["source"]),
            "pair_group_ids": sorted(holdout["pair"]),
            "sample_ids": sorted(holdout["sample"]),
        },
    )
    agreement_path = output_root / "review_agreement_summary.json"
    _write_json(agreement_path, agreement)
    _write_json(output_root / "automatic_review_summary.json", automatic_review_summary)
    asr_summary = {
        "count": len(diagnostic),
        "speech_roles": dict(sorted(Counter(_review_field(row, "speech_role") or "unknown" for row in diagnostic).items())),
        "rejection_reasons": dict(sorted(Counter(reason for row in diagnostic for reason in row.get("automatic_consensus_reasons", [])).items())),
    }
    _write_json(output_root / "asr_diagnostic_summary.json", asr_summary)
    crosstab = _subtype_dataset_crosstab(test)
    _write_json(output_root / "subtype_dataset_crosstab.json", crosstab)
    audit = {
        "selection_uses_model_scores": False,
        "leakage": leakage,
        "duplicate_pair_count": len(test) - len({_pair_id(row) for row in test}),
        "missing_media_count": sum(
            not _first_text(row, ("reference_video",)) or not _first_text(row, ("target_video",)) for row in test
        ),
        "test_count": len(test),
        "validation_count": len(validation),
        "train_count": len(train),
    }
    _write_json(output_root / "leakage_audit.json", audit)
    manifest = {
        "protocol": "audiocvr_automatic_model_verified_benchmark_v1",
        "human_validated": False,
        "automatically_curated": True,
        "model_verified": True,
        "selection_uses_model_scores": False,
        "combined_pool_path": str(combined_pool_path),
        "candidate_path": str(candidate_path),
        "test_target_count": total_target,
        "test_final_count": len(test),
        "test_subtype_targets": test_targets,
        "test_subtype_distribution": dict(sorted(Counter(_canonical_subtype(row) for row in test).items())),
        "test_dataset_distribution": dict(sorted(Counter(_dataset(row) for row in test).items())),
        "validation_targets": val_targets,
        "validation_distribution": dict(sorted(Counter(_canonical_subtype(row) for row in validation).items())),
        "train_distribution": dict(sorted(Counter(_canonical_subtype(row) for row in train).items())),
        "automatic_review": agreement,
        "review_rejection_counts": dict(sorted(rejection_counts.items())),
        "test_selection": selection,
        "validation_selection": validation_selection,
        "split_allocation_order": split_allocation_order,
        "strict_local_count": sum(_has_strict_local_negative(row) for row in test),
        "strict_local_coverage": sum(_has_strict_local_negative(row) for row in test) / max(1, len(test)),
        "leakage": leakage,
        "test_main_sha256": test_hash,
        "random_seed": int(random_seed),
        "outputs": {
            "test_main": str(test_path),
            "test_asr_diagnostic": str(diagnostic_path),
            "train": str(train_path),
            "val": str(val_path),
            "agreement": str(agreement_path),
            "manifest": str(output_root / "frozen_benchmark_manifest.json"),
        },
        "limitation": "Automatically curated and model-verified; not human-validated.",
    }
    _write_json(output_root / "frozen_benchmark_manifest.json", manifest)
    (output_root / "frozen_benchmark.sha256").write_text(
        f"{test_hash}  test_main_150.jsonl\n", encoding="ascii"
    )
    _write_json(
        output_root / "rejection_breakdown.json",
        {
            "consensus_rejections": dict(sorted(rejection_counts.items())),
            "pass1_reject_reasons": dict(sorted(pass1_reject_reasons.items())),
            "pass2_reject_reasons": dict(sorted(pass2_reject_reasons.items())),
        },
    )
    _write_json(
        output_root / "split_summary.json",
        {
            "train_count": len(train),
            "val_count": len(validation),
            "test_main_count": len(test),
            "diagnostic_count": len(diagnostic),
            "train_subtypes": dict(sorted(Counter(_canonical_subtype(row) for row in train).items())),
            "val_subtypes": dict(sorted(Counter(_canonical_subtype(row) for row in validation).items())),
            "test_subtypes": dict(sorted(Counter(_canonical_subtype(row) for row in test).items())),
            "source_group_counts": {
                "train": len({_primary_source_id(row) for row in train}),
                "val": len({_primary_source_id(row) for row in validation}),
                "test": len({_primary_source_id(row) for row in test}),
            },
            "leakage": leakage,
        },
    )
    (output_root / "benchmark_quality_report.md").write_text(
        _automatic_benchmark_markdown(manifest, asr_summary, automatic_review_summary), encoding="utf-8"
    )
    return manifest


def audit_training_splits(
    *,
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Audit the frozen benchmark splits before inverse augmentation or E5 training."""
    paths = {
        "train": Path(train_path),
        "val": Path(val_path),
        "test": Path(test_path),
    }
    rows = {name: _read_jsonl(path) for name, path in paths.items()}
    empty = [name for name, values in rows.items() if not values]
    if empty:
        raise ValueError(f"training split audit requires non-empty files: {empty}")

    leakage = _split_leakage_summary(
        train=rows["train"],
        val=rows["val"],
        test=rows["test"],
    )
    violations: list[dict[str, Any]] = list(leakage["violations"])
    for split_name, values in rows.items():
        sample_counts = Counter(_sample_id(row) for row in values if _sample_id(row))
        duplicate_samples = sorted(key for key, count in sample_counts.items() if count > 1)
        if duplicate_samples:
            violations.append(
                {
                    "type": "duplicate_sample",
                    "split": split_name,
                    "count": len(duplicate_samples),
                    "examples": duplicate_samples[:5],
                }
            )
    test_inverse = [
        row
        for row in rows["test"]
        if _truthy(row.get("is_inverse"))
        or str(row.get("direction") or "forward").strip().lower() == "inverse"
    ]
    if test_inverse:
        violations.append(
            {
                "type": "inverse_in_test_main",
                "split": "test",
                "count": len(test_inverse),
                "examples": [_sample_id(row) for row in test_inverse[:5]],
            }
        )
    test_pairs = [_pair_id(row) for row in rows["test"] if _pair_id(row)]
    duplicate_test_pairs = sorted(
        pair for pair, count in Counter(test_pairs).items() if count > 1
    )
    if duplicate_test_pairs:
        violations.append(
            {
                "type": "duplicate_pair_in_test_main",
                "split": "test",
                "count": len(duplicate_test_pairs),
                "examples": duplicate_test_pairs[:5],
            }
        )

    split_summaries: dict[str, Any] = {}
    for split_name, values in rows.items():
        directions = Counter(
            "inverse"
            if _truthy(row.get("is_inverse"))
            or str(row.get("direction") or "forward").strip().lower() == "inverse"
            else "forward"
            for row in values
        )
        split_summaries[split_name] = {
            "count": len(values),
            "unique_source_count": len({_primary_source_id(row) for row in values}),
            "unique_pair_count": len({_pair_id(row) for row in values}),
            "subtype_distribution": dict(
                sorted(Counter(_canonical_subtype(row) for row in values).items())
            ),
            "dataset_distribution": dict(sorted(Counter(_dataset(row) for row in values).items())),
            "direction_distribution": dict(sorted(directions.items())),
        }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "audiocvr_training_split_audit_v1",
        "input_paths": {name: str(path) for name, path in paths.items()},
        "splits": split_summaries,
        "train_forward_count": int(split_summaries["train"]["direction_distribution"].get("forward", 0)),
        "train_inverse_count": int(split_summaries["train"]["direction_distribution"].get("inverse", 0)),
        "leakage": leakage,
        "violation_count": sum(int(item.get("count") or 0) for item in violations),
        "violations": violations,
        "ready_for_inverse_augmentation": not violations,
        "ready_for_training": not violations,
    }
    _write_json(output_root / "training_split_audit.json", summary)
    (output_root / "training_split_audit.md").write_text(
        _training_split_audit_markdown(summary), encoding="utf-8"
    )
    if violations:
        raise ValueError(f"training split audit failed: {violations}")
    return summary


def _training_split_audit_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Audio-CVR Training Split Audit",
        "",
        f"- Ready for training: `{str(summary['ready_for_training']).lower()}`",
        f"- Leakage violations: {summary['leakage']['violation_count']}",
        f"- Total violations: {summary['violation_count']}",
        "",
        "| Split | Records | Sources | Pairs | Forward | Inverse |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("train", "val", "test"):
        split = summary["splits"][split_name]
        directions = split["direction_distribution"]
        lines.append(
            f"| {split_name} | {split['count']} | {split['unique_source_count']} | "
            f"{split['unique_pair_count']} | {directions.get('forward', 0)} | "
            f"{directions.get('inverse', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _automatic_consensus_reject_reason(
    pass1: dict[str, Any], pass2: dict[str, Any] | None, *, repeated: bool
) -> str:
    if pass1.get("decision") != "pass":
        return f"pass1_{pass1.get('decision') or 'missing'}"
    if not repeated:
        return ""
    if pass2 is None:
        return "missing_repeat_review"
    if pass2.get("decision") != "pass":
        return f"pass2_{pass2.get('decision') or 'missing'}"
    if any(pass1.get(field) != pass2.get(field) for field in AUTOMATIC_REVIEW_CRITICAL_FIELDS):
        return "rejected_review_disagreement"
    if str(pass1.get("speech_role") or "") != str(pass2.get("speech_role") or ""):
        return "speech_role_disagreement"
    return ""


def _automatic_review_agreement(
    pass1: dict[str, dict[str, Any]],
    pass2: dict[str, dict[str, Any]],
    repeat_ids: set[str],
) -> dict[str, Any]:
    compared = [sample_id for sample_id in sorted(repeat_ids) if sample_id in pass1 and sample_id in pass2]
    exact = sum(pass1[sample_id].get("decision") == pass2[sample_id].get("decision") for sample_id in compared)
    field_matches = 0
    field_total = 0
    speech_matches = 0
    disagreements = 0
    for sample_id in compared:
        first, second = pass1[sample_id], pass2[sample_id]
        for field in AUTOMATIC_REVIEW_CRITICAL_FIELDS:
            field_matches += int(first.get(field) == second.get(field))
            field_total += 1
        speech_matches += int(first.get("speech_role") == second.get("speech_role"))
        disagreements += int(bool(_automatic_consensus_reject_reason(first, second, repeated=True)))
    return {
        "repeat_review_count": len(repeat_ids),
        "repeat_review_requested_count": len(repeat_ids),
        "repeat_review_completed_count": len(compared),
        "exact_decision_agreement": exact / len(compared) if compared else None,
        "field_level_agreement": field_matches / field_total if field_total else None,
        "speech_role_agreement": speech_matches / len(compared) if compared else None,
        "disagreement_count": disagreements,
        "missing_repeat_count": len(repeat_ids) - len(compared),
    }


def _select_exact_benchmark_quota(
    rows: list[dict[str, Any]],
    *,
    targets: dict[str, int],
    total_target: int,
    max_dataset_ratio: float,
    max_hdtf_ratio: float,
    max_voxceleb_ratio: float,
    max_per_source: int,
    random_seed: int,
    strict_dataset_ratio: float | None = None,
    relaxed_dataset_non_speech_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_targets = {
        _canonical_subtype_name(key): max(0, int(value)) for key, value in targets.items()
    }
    selected: list[dict[str, Any]] = []
    dataset_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    dataset_limit = max(1, math.floor(total_target * max_dataset_ratio))
    strict_dataset_limit = (
        max(1, math.floor(total_target * float(strict_dataset_ratio)))
        if strict_dataset_ratio is not None
        else dataset_limit
    )
    hdtf_limit = max(0, math.floor(total_target * max_hdtf_ratio))
    voxceleb_limit = max(0, math.floor(total_target * max_voxceleb_ratio))
    remaining = list(rows)
    subtype_order = sorted(
        normalized_targets,
        key=lambda subtype: (
            len([row for row in rows if _canonical_subtype(row) == subtype]) / max(1, normalized_targets[subtype]),
            subtype,
        ),
    )
    for subtype in subtype_order:
        while subtype_counts[subtype] < normalized_targets[subtype]:
            eligible = [
                row
                for row in remaining
                if _canonical_subtype(row) == subtype
                and source_counts[_primary_source_id(row)] < max_per_source
                and dataset_counts[_dataset(row)] < dataset_limit
                and not (
                    relaxed_dataset_non_speech_only
                    and subtype == "speech_topic_in_video_context"
                    and dataset_counts[_dataset(row)] >= strict_dataset_limit
                )
                and (_dataset(row) != "hdtf" or dataset_counts["hdtf"] < hdtf_limit)
                and (_dataset(row) != "voxceleb" or dataset_counts["voxceleb"] < voxceleb_limit)
                and _voxceleb_automatic_review_allowed(row)
            ]
            if not eligible:
                break
            row = min(
                eligible,
                key=lambda item: (
                    -int(_has_strict_local_negative(item)),
                    dataset_counts[_dataset(item)],
                    -float(_review_field(item, "min_stage_confidence") or 0.0),
                    -float(_review_field(item, "video_context_strength") or 0.0),
                    -_numeric_field(item, "audio_delta_strength"),
                    float(_review_field(item, "recomputed_asr_risk") or 0.0),
                    _stable_row_key(item, random_seed),
                ),
            )
            selected.append(row)
            remaining.remove(row)
            subtype_counts[subtype] += 1
            dataset_counts[_dataset(row)] += 1
            source_counts[_primary_source_id(row)] += 1
    selection = {
        "requested_targets": normalized_targets,
        "selected_count": len(selected),
        "selected_subtypes": dict(sorted(subtype_counts.items())),
        "selected_datasets": dict(sorted(dataset_counts.items())),
        "max_dataset_ratio": max_dataset_ratio,
        "dataset_limit": dataset_limit,
        "strict_dataset_limit": strict_dataset_limit,
        "hdtf_limit": hdtf_limit,
        "voxceleb_limit": voxceleb_limit,
        "dataset_ratio_relaxed": dataset_limit > strict_dataset_limit,
        "relaxed_dataset_non_speech_only": bool(relaxed_dataset_non_speech_only),
        "reallocated_to_sound_event": 0,
    }
    return selected, selection


def _select_with_non_speech_reallocation(
    rows: list[dict[str, Any]],
    *,
    original_targets: dict[str, int],
    total_target: int,
    max_dataset_ratio: float,
    max_hdtf_ratio: float,
    max_voxceleb_ratio: float,
    max_per_source: int,
    random_seed: int,
    strict_dataset_ratio: float | None = None,
    relaxed_dataset_non_speech_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    availability = Counter(_canonical_subtype(row) for row in rows)
    targets = {_canonical_subtype_name(key): int(value) for key, value in original_targets.items()}
    music = min(targets.get("music", 0), availability["music"])
    speech = min(targets.get("speech_topic_in_video_context", 0), availability["speech_topic_in_video_context"])
    sound = total_target - music - speech
    effective = {
        "sound_event": sound,
        "music": music,
        "speech_topic_in_video_context": speech,
    }
    selected, summary = _select_exact_benchmark_quota(
        rows,
        targets=effective,
        total_target=total_target,
        max_dataset_ratio=max_dataset_ratio,
        max_hdtf_ratio=max_hdtf_ratio,
        max_voxceleb_ratio=max_voxceleb_ratio,
        max_per_source=max_per_source,
        random_seed=random_seed,
        strict_dataset_ratio=strict_dataset_ratio,
        relaxed_dataset_non_speech_only=relaxed_dataset_non_speech_only,
    )
    summary["original_targets"] = {
        _canonical_subtype_name(key): int(value) for key, value in original_targets.items()
    }
    summary["reallocated_to_sound_event"] = max(0, sound - int(original_targets.get("sound_event", 0)))
    summary["reallocation_policy"] = "music/speech shortages move only to sound_event; speech never increases"
    return selected, summary


def _select_quota_with_fallback(
    rows: list[dict[str, Any]],
    *,
    targets: dict[str, int],
    total_target: int,
    max_dataset_ratio: float,
    relaxed_dataset_ratio: float,
    max_hdtf_ratio: float,
    max_voxceleb_ratio: float,
    max_per_source: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, summary = _select_exact_benchmark_quota(
        rows,
        targets=targets,
        total_target=total_target,
        max_dataset_ratio=max_dataset_ratio,
        max_hdtf_ratio=max_hdtf_ratio,
        max_voxceleb_ratio=max_voxceleb_ratio,
        max_per_source=max_per_source,
        random_seed=random_seed,
    )
    if len(selected) < total_target:
        selected, summary = _select_exact_benchmark_quota(
            rows,
            targets=targets,
            total_target=total_target,
            max_dataset_ratio=relaxed_dataset_ratio,
            max_hdtf_ratio=max_hdtf_ratio,
            max_voxceleb_ratio=max_voxceleb_ratio,
            max_per_source=max_per_source,
            random_seed=random_seed,
            strict_dataset_ratio=max_dataset_ratio,
            relaxed_dataset_non_speech_only=True,
        )
    if len(selected) < total_target:
        selected, summary = _select_with_non_speech_reallocation(
            rows,
            original_targets=targets,
            total_target=total_target,
            max_dataset_ratio=relaxed_dataset_ratio,
            max_hdtf_ratio=max_hdtf_ratio,
            max_voxceleb_ratio=max_voxceleb_ratio,
            max_per_source=max_per_source,
            random_seed=random_seed,
            strict_dataset_ratio=max_dataset_ratio,
            relaxed_dataset_non_speech_only=True,
        )
    return selected, summary


def _select_disjoint_test_validation(
    rows: list[dict[str, Any]],
    *,
    test_targets: dict[str, int],
    validation_targets: dict[str, int],
    max_dataset_ratio: float,
    relaxed_dataset_ratio: float,
    max_hdtf_ratio: float,
    max_voxceleb_ratio: float,
    max_per_source: int,
    random_seed: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    str,
]:
    """Select exact source-disjoint test/validation quotas without greedy starvation."""
    test_total = sum(max(0, int(value)) for value in test_targets.values())
    validation_total = sum(max(0, int(value)) for value in validation_targets.values())

    def select_test(pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _select_quota_with_fallback(
            pool,
            targets=test_targets,
            total_target=test_total,
            max_dataset_ratio=max_dataset_ratio,
            relaxed_dataset_ratio=relaxed_dataset_ratio,
            max_hdtf_ratio=max_hdtf_ratio,
            max_voxceleb_ratio=max_voxceleb_ratio,
            max_per_source=max_per_source,
            random_seed=random_seed,
        )

    def select_validation(pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _select_exact_benchmark_quota(
            pool,
            targets=validation_targets,
            total_target=validation_total,
            max_dataset_ratio=relaxed_dataset_ratio,
            max_hdtf_ratio=max(max_hdtf_ratio, 0.20),
            max_voxceleb_ratio=max(max_voxceleb_ratio, 0.10),
            max_per_source=max_per_source,
            random_seed=random_seed + 1,
            strict_dataset_ratio=max_dataset_ratio,
            relaxed_dataset_non_speech_only=True,
        )

    test, test_summary = select_test(rows)
    test_ids = _identity_sets(test)
    validation_pool = [row for row in rows if not _row_overlaps_identities(row, test_ids)]
    validation, validation_summary = select_validation(validation_pool)
    if len(test) == test_total and len(validation) == validation_total:
        order = "test_first"
        test_summary["split_allocation_order"] = order
        validation_summary["split_allocation_order"] = order
        return test, test_summary, validation, validation_summary, order

    reserved_validation, reserved_validation_summary = select_validation(rows)
    validation_ids = _identity_sets(reserved_validation)
    reserved_test_pool = [
        row for row in rows if not _row_overlaps_identities(row, validation_ids)
    ]
    reserved_test, reserved_test_summary = select_test(reserved_test_pool)
    if len(reserved_test) == test_total and len(reserved_validation) == validation_total:
        order = "validation_reserved_before_test"
        reserved_test_summary["split_allocation_order"] = order
        reserved_validation_summary["split_allocation_order"] = order
        return (
            reserved_test,
            reserved_test_summary,
            reserved_validation,
            reserved_validation_summary,
            order,
        )

    order = "test_first_incomplete"
    test_summary["split_allocation_order"] = order
    validation_summary["split_allocation_order"] = order
    validation_summary["validation_reserved_attempt"] = {
        "test_count": len(reserved_test),
        "validation_count": len(reserved_validation),
        "test_selection": reserved_test_summary,
        "validation_selection": reserved_validation_summary,
    }
    return test, test_summary, validation, validation_summary, order


def _row_with_automatic_review(
    row: dict[str, Any],
    pass1: dict[str, Any],
    pass2: dict[str, Any] | None,
    *,
    consensus_reason: str,
) -> dict[str, Any]:
    output = dict(row)
    output["automatic_review_pass1"] = pass1
    if pass2 is not None:
        output["automatic_review_pass2"] = pass2
    output["automatic_review_consensus"] = not bool(consensus_reason)
    output["automatic_consensus_reasons"] = [consensus_reason] if consensus_reason else []
    output["reviewer_type"] = "omni"
    output["review_profile"] = AUTOMATIC_REVIEW_PROFILE
    output["human_validated"] = False
    output["model_verified"] = not bool(consensus_reason)
    output["recomputed_asr_risk"] = float(pass1.get("recomputed_asr_risk") or 0.0)
    output["speech_role"] = str(pass1.get("speech_role") or "")
    output["transcript_like"] = bool(pass1.get("transcript_like"))
    output["full_av_required"] = bool(pass1.get("full_av_required"))
    output["video_context_strength"] = float(pass1.get("video_context_strength") or 0.0)
    output["audio_only_solvability"] = float(pass1.get("audio_only_solvability") or 0.0)
    output["min_stage_confidence"] = min(
        [float(pass1.get("min_stage_confidence") or 0.0)]
        + ([float(pass2.get("min_stage_confidence") or 0.0)] if pass2 is not None else [])
    )
    if not consensus_reason:
        output["split_tier"] = "main"
        output["benchmark_eligible"] = True
    else:
        output["split_tier"] = "diagnostic"
        output["benchmark_eligible"] = False
    return output


def _automatic_review_is_diagnostic(pass1: dict[str, Any], pass2: dict[str, Any] | None) -> bool:
    values = [pass1] + ([pass2] if pass2 else [])
    return any(
        str(value.get("speech_role") or "") in {"asr_only", "generic_talking_head"}
        or bool(value.get("transcript_like"))
        or float(value.get("recomputed_asr_risk") or 0.0) > 0.35
        or float(value.get("audio_only_solvability") or 0.0) >= 0.85
        for value in values
    )


def _automatic_reviews_by_sample(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _sample_id(row)
        if not sample_id:
            continue
        if sample_id in output:
            raise ValueError(f"duplicate automatic review for sample_id={sample_id}")
        output[sample_id] = row
    return output


def _deterministic_repeat_ids(
    rows: list[dict[str, Any]], *, fraction: float, random_seed: int
) -> set[str]:
    if not rows or fraction <= 0:
        return set()
    count = min(len(rows), max(1, round(len(rows) * min(1.0, fraction))))
    ranked = sorted(rows, key=lambda row: _stable_row_key(row, random_seed + 7919))
    return {_sample_id(row) for row in ranked[:count]}


def _row_overlaps_identities(row: dict[str, Any], identities: dict[str, set[str]]) -> bool:
    source = _row_identity_values(row, ("source_disjoint_group_id", "raw_source_id", "source_id"))
    pair = _row_identity_values(row, ("inverse_pair_group_id", "pair_group_id"))
    sample = {_sample_id(row)} - {""}
    return bool(source & identities["source"] or pair & identities["pair"] or sample & identities["sample"])


def _voxceleb_automatic_review_allowed(row: dict[str, Any]) -> bool:
    if _dataset(row) != "voxceleb":
        return True
    return (
        float(_review_field(row, "recomputed_asr_risk") or 0.0) <= 0.30
        and float(_review_field(row, "video_context_strength") or 0.0) >= 0.70
    )


def _review_field(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    review = row.get("automatic_review_pass1") if isinstance(row.get("automatic_review_pass1"), dict) else {}
    return review.get(key)


def _recommended_training_sampling_weight(row: dict[str, Any]) -> float:
    subtype = _canonical_subtype(row)
    if subtype == "sound_event":
        return 0.50
    if subtype == "music":
        return 0.20
    speech_role = str(row.get("speech_role") or "")
    if speech_role in {"asr_only", "generic_talking_head"} or _truthy(row.get("transcript_like")):
        return 0.05
    if subtype == "speech_topic_in_video_context":
        return 0.25 if speech_role in {"contextual_speech", "speech_with_event"} else 0.05
    return 0.0


def _subtype_dataset_crosstab(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[_canonical_subtype(row)][_dataset(row)] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def _automatic_benchmark_markdown(
    manifest: dict[str, Any],
    asr_summary: dict[str, Any],
    automatic_review_summary: dict[str, Any],
) -> str:
    lines = [
        "# Audio-CVR Automatic Benchmark Quality Report",
        "",
        "> Automatically curated and model-verified; not human-validated.",
        "",
        f"- Test count: {manifest['test_final_count']}",
        f"- Test SHA256: `{manifest['test_main_sha256']}`",
        f"- Strict local coverage: {manifest['strict_local_coverage']:.2%}",
        f"- ASR diagnostic count: {asr_summary['count']}",
        f"- Review candidates: {automatic_review_summary['review_candidate_count']}",
        f"- Omni consensus eligible: {automatic_review_summary['consensus_eligible_count']}",
        f"- Repeat decision agreement: {automatic_review_summary['repeat_review']['exact_decision_agreement']}",
        f"- Leakage violations: {manifest['leakage']['violation_count']}",
        "",
        "## Test Subtypes",
        "",
        "| Subtype | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in manifest["test_subtype_distribution"].items())
    lines.extend(["", "## Dataset Distribution", "", "| Dataset | Count |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in manifest["test_dataset_distribution"].items())
    lines.append("")
    return "\n".join(lines)


def _numeric_distribution(values: Iterable[float]) -> dict[str, Any]:
    finite: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "unique_values": []}
    unique = sorted(set(finite))
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "unique_values": unique[:50],
        "unique_value_count": len(unique),
    }


def prepare_paper_splits(*, split_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_root = Path(split_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train = _read_jsonl(source_root / "train.jsonl")
    val = _read_jsonl(source_root / "val.jsonl")
    test_source = source_root / "test_main.jsonl"
    if not test_source.exists():
        test_source = source_root / "test_main_150.jsonl"
    test_all = _read_jsonl(test_source)
    if not train or not val or not test_all:
        raise ValueError("paper split preparation requires non-empty train, val, and test_main JSONL files")

    test_main = [
        row
        for row in test_all
        if str(row.get("split_tier") or "main").strip().lower() == "main"
        and not _truthy(row.get("is_inverse"))
        and str(row.get("direction") or "forward").strip().lower() != "inverse"
    ]
    if not test_main:
        raise ValueError("test_main became empty after B-main and forward-direction filtering")

    leakage = _split_leakage_summary(train=train, val=val, test=test_all)
    if leakage["violation_count"]:
        raise ValueError(f"source/pair leakage detected: {leakage}")

    outputs = {
        "train": output_root / "train.jsonl",
        "val": output_root / "val.jsonl",
        "test_main": output_root / "test_main.jsonl",
        "test_all": output_root / "test_all.jsonl",
        "human_review_manifest": output_root / "test_main_human_review.jsonl",
    }
    _write_jsonl(outputs["train"], train)
    _write_jsonl(outputs["val"], val)
    _write_jsonl(outputs["test_main"], test_main)
    _write_jsonl(outputs["test_all"], test_all)
    _write_jsonl(outputs["human_review_manifest"], [_human_review_row(row) for row in test_main])

    summary = {
        "split_root": str(source_root),
        "output_dir": str(output_root),
        "test_source_path": str(test_source),
        "counts": {
            "train": len(train),
            "val": len(val),
            "test_all": len(test_all),
            "test_main": len(test_main),
            "test_excluded_non_main": len(test_all) - len(test_main),
        },
        "test_main_subtypes": dict(sorted(Counter(_subtype(row) for row in test_main).items())),
        "leakage": leakage,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    _write_json(output_root / "split_verification.json", summary)
    return summary


def summarize_validation(
    *,
    input_roots: Iterable[str | Path],
    output_dir: str | Path,
    required_seeds: Iterable[int],
    top_n: int = 6,
    selection_rule: str = "lexicographic",
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    required = tuple(sorted(set(int(seed) for seed in required_seeds)))
    if not required:
        raise ValueError("required_seeds must not be empty")

    observations: list[dict[str, Any]] = []
    for root_value in input_roots:
        root = Path(root_value)
        for summary_path in root.glob("**/eval/summary.json"):
            adapter_dir = summary_path.parent.parent / "adapter"
            train_path = adapter_dir / "train_summary.json"
            if not train_path.exists():
                continue
            train_summary = _read_json(train_path)
            eval_summary = _read_json(summary_path)
            observations.append(_validation_observation(train_summary, eval_summary, adapter_dir, summary_path.parent))
    if not observations:
        raise ValueError("no validation eval/summary.json files with matching adapter/train_summary.json were found")

    groups: dict[tuple[int, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        groups[(row["steps"], row["learning_rate"], row["batch_size"])].append(row)

    rows: list[dict[str, Any]] = []
    for (steps, learning_rate, batch_size), values in groups.items():
        by_seed = {int(value["seed"]): value for value in values}
        if not all(seed in by_seed for seed in required):
            continue
        selected = [by_seed[seed] for seed in required]
        r1 = [float(value["R@1"]) for value in selected]
        target_beats = [float(value["target_beats_reference"]) for value in selected]
        rows.append(
            {
                "steps": steps,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "seeds": list(required),
                "seed_count": len(required),
                "R@1_mean": _mean(r1),
                "R@1_std": _std(r1),
                "R@1_se": _sample_std(r1) / math.sqrt(len(r1)),
                "R@5_mean": _mean([float(value["R@5"]) for value in selected]),
                "R@10_mean": _mean([float(value["R@10"]) for value in selected]),
                "target_beats_reference_mean": _mean(target_beats),
                "target_beats_reference_std": _std(target_beats),
                "target_ref_gap_mean": _mean([float(value["target_ref_gap"]) for value in selected]),
                "runs": selected,
            }
        )
    if not rows:
        raise ValueError(f"no validation configuration has all required seeds: {required}")

    # Preregistered lexicographic rule: retrieval first, directionality second,
    # then stability and lower optimization cost. Test results are never read.
    rows.sort(
        key=lambda row: (
            -row["R@1_mean"],
            -row["target_beats_reference_mean"],
            row["R@1_std"],
            row["steps"],
            row["learning_rate"],
            row["batch_size"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["validation_rank"] = rank
    best = rows[0]
    if selection_rule == "lexicographic":
        selected = best
        one_se_threshold = None
        one_se_candidates: list[dict[str, Any]] = []
        selection_description = (
            "max mean R@1; then max target_beats_reference; then min R@1 std; then fewer steps"
        )
    elif selection_rule == "one_se_earliest":
        one_se_threshold = float(best["R@1_mean"] - best["R@1_se"])
        one_se_candidates = [row for row in rows if float(row["R@1_mean"]) >= one_se_threshold]
        selected = min(
            one_se_candidates,
            key=lambda row: (
                row["steps"],
                -row["target_beats_reference_mean"],
                -row["R@1_mean"],
                row["R@1_std"],
                row["learning_rate"],
                row["batch_size"],
            ),
        )
        selection_description = (
            "one-standard-error rule on mean validation R@1; among eligible configurations choose the fewest steps, "
            "then higher target_beats_reference"
        )
    else:
        raise ValueError(f"unsupported validation selection rule: {selection_rule}")
    summary = {
        "selection_split": "validation_only",
        "selection_rule": selection_description,
        "selection_rule_name": selection_rule,
        "one_se_threshold": one_se_threshold,
        "one_se_candidate_count": len(one_se_candidates),
        "required_seeds": list(required),
        "configuration_count": len(rows),
        "selected_config": {
            key: selected[key]
            for key in ("steps", "learning_rate", "batch_size", "R@1_mean", "R@1_std", "target_beats_reference_mean")
        },
        "rows": rows,
    }
    _write_json(output_root / "validation_model_selection.json", summary)
    (output_root / "validation_model_selection.md").write_text(_validation_markdown(summary), encoding="utf-8")
    _write_config_tsv(output_root / "top_configs.tsv", rows[: max(1, int(top_n))])
    _write_config_tsv(output_root / "selected_config.tsv", [selected])
    return summary


def aggregate_final(
    *,
    input_root: str | Path,
    output_dir: str | Path,
    required_seeds: Iterable[int] = DEFAULT_FINAL_SEEDS,
    primary_mode: str = PRIMARY_MODE,
    reference_mode: str = REFERENCE_MODE,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 20_000,
    random_seed: int = 20260718,
    comparisons: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    root = Path(input_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(sorted(set(int(seed) for seed in required_seeds)))
    if not seeds:
        raise ValueError("required_seeds must not be empty")

    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        for eval_dir in seed_root.glob("eval_*"):
            summary_path = eval_dir / "summary.json"
            scores_path = eval_dir / "per_query_scores.jsonl"
            if not summary_path.exists() or not scores_path.exists():
                continue
            mode = eval_dir.name.removeprefix("eval_")
            runs[(seed, mode)] = {
                "seed": seed,
                "mode": mode,
                "eval_dir": str(eval_dir),
                "summary": _read_json(summary_path),
                "scores": _read_jsonl(scores_path),
            }
    modes = sorted({mode for _, mode in runs})
    missing = [f"seed={seed},mode={mode}" for seed in seeds for mode in modes if (seed, mode) not in runs]
    if primary_mode not in modes or reference_mode not in modes:
        raise ValueError(f"required modes are missing: primary={primary_mode}, reference={reference_mode}, found={modes}")
    if missing:
        raise ValueError(f"incomplete final experiment matrix: {missing}")

    audit = _audit_final_runs(runs=runs, seeds=seeds, modes=modes)
    if audit["violation_count"]:
        raise ValueError(f"final experiment audit failed: {audit}")

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        for mode in modes:
            run = runs[(seed, mode)]
            metrics = _eval_metrics(run["summary"])
            per_seed.append({"seed": seed, "mode": mode, "eval_dir": run["eval_dir"], **metrics})

    mode_summary: dict[str, Any] = {}
    for mode in modes:
        values = [row for row in per_seed if row["mode"] == mode]
        mode_summary[mode] = {
            metric: {"mean": _mean([float(row[metric]) for row in values]), "std": _std([float(row[metric]) for row in values])}
            for metric in (
                "R@1",
                "R@5",
                "R@10",
                "target_beats_reference",
                "target_ref_gap",
                "base_R@1",
                "base_R@5",
                "base_R@10",
                "base_target_beats_reference",
                "base_target_ref_gap",
                "reference_rank_median",
                "reference_rank_le_1",
            )
        }

    paired = _paired_mode_statistics(
        runs=runs,
        seeds=seeds,
        mode_a=reference_mode,
        mode_b=primary_mode,
        bootstrap_samples=max(100, int(bootstrap_samples)),
        permutation_samples=max(100, int(permutation_samples)),
        random_seed=int(random_seed),
    )
    requested_comparisons = list(comparisons or [(primary_mode, reference_mode)])
    primary_pair = (primary_mode, reference_mode)
    if primary_pair not in requested_comparisons:
        requested_comparisons.insert(0, primary_pair)
    paired_comparisons: dict[str, Any] = {}
    for index, (mode_b, mode_a) in enumerate(requested_comparisons):
        if mode_a not in modes or mode_b not in modes:
            raise ValueError(f"paired comparison requires missing mode: {mode_b}:{mode_a}; found={modes}")
        key = f"{mode_b}_minus_{mode_a}"
        paired_comparisons[key] = _paired_mode_statistics(
            runs=runs,
            seeds=seeds,
            mode_a=mode_a,
            mode_b=mode_b,
            bootstrap_samples=max(100, int(bootstrap_samples)),
            permutation_samples=max(100, int(permutation_samples)),
            random_seed=int(random_seed) + index,
        )
    _add_comparison_holm_corrections(paired_comparisons)
    error_breakdown = _error_breakdown([runs[(seed, primary_mode)] for seed in seeds])
    hard_negative = _hard_negative_summary([runs[(seed, primary_mode)]["summary"] for seed in seeds])

    summary = {
        "input_root": str(root),
        "output_dir": str(output_root),
        "required_seeds": list(seeds),
        "modes": modes,
        "primary_comparison": f"{primary_mode} - {reference_mode}",
        "mode_summary": mode_summary,
        "paired_statistics": paired,
        "paired_comparisons": paired_comparisons,
        "primary_mode_subtypes": _subtype_result_summary(
            [runs[(seed, primary_mode)]["summary"] for seed in seeds]
        ),
        "hard_negative_summary": hard_negative,
        "audit_path": str(output_root / "audit.json"),
    }
    _write_json(output_root / "audit.json", audit)
    _write_json(output_root / "per_seed_results.json", {"rows": per_seed})
    _write_json(output_root / "test_main_mean_std.json", summary)
    _write_json(output_root / "paired_comparisons.json", paired_comparisons)
    _write_json(output_root / "error_breakdown.json", error_breakdown)
    (output_root / "test_main_comparison.md").write_text(_final_comparison_markdown(summary), encoding="utf-8")
    (output_root / "audio_gain_summary.md").write_text(_audio_gain_markdown(summary), encoding="utf-8")
    (output_root / "paired_comparisons.md").write_text(_paired_comparisons_markdown(summary), encoding="utf-8")
    return summary


def score_fusion(
    *,
    cache_a: str | Path,
    cache_b: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
    alpha: float | None = None,
    alpha_grid: Iterable[float] = tuple(index / 10.0 for index in range(11)),
    device: str = "cuda",
    save_topk: int = 20,
) -> dict[str, Any]:
    from app.e5_audio_delta_train import (
        _AudioDeltaAdapter,
        _eval_gallery_items_for_output,
        _gallery_negative_recall_by_type,
        _grouped_recall_summary,
        _import_torch,
        _load_embedding_npz,
        _recall_from_scores,
        _reference_rank_summary,
        _target_beats_reference_summary,
        _torch_device,
        _write_eval_topk_outputs,
        load_audio_delta_records,
    )

    cache_a_root = Path(cache_a)
    cache_b_root = Path(cache_b)
    adapter_root = Path(adapter_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data_a = _load_embedding_npz(cache_a_root / "eval_embeddings.npz")
    data_b = _load_embedding_npz(cache_b_root / "eval_embeddings.npz")
    records_a = load_audio_delta_records(cache_a_root / "eval_records.jsonl")
    records_b = load_audio_delta_records(cache_b_root / "eval_records.jsonl")
    if [record.sample_id for record in records_a] != [record.sample_id for record in records_b]:
        raise ValueError("fusion caches do not contain identical eval records")
    positive_a = np.asarray(data_a.get("positive_gallery_index"), dtype=np.int64)
    positive_b = np.asarray(data_b.get("positive_gallery_index"), dtype=np.int64)
    reference_a = np.asarray(data_a.get("reference_gallery_index"), dtype=np.int64)
    reference_b = np.asarray(data_b.get("reference_gallery_index"), dtype=np.int64)
    if not np.array_equal(positive_a, positive_b) or not np.array_equal(reference_a, reference_b):
        raise ValueError("fusion caches do not share positive/reference gallery indices")
    if int(data_a["gallery"].shape[0]) != int(data_b["gallery"].shape[0]):
        raise ValueError("fusion caches do not share gallery size")

    torch = _import_torch()
    device_obj = _torch_device(torch, device)
    dim = int(data_a["query"].shape[1])
    model = _AudioDeltaAdapter(torch, dim).to(device_obj)
    state = torch.load(adapter_root / "adapter.pt", map_location=device_obj)
    model.load_state_dict(state, strict=False)
    model.eval()

    base_a = np.asarray(data_a["query"] @ data_a["gallery"].T, dtype=np.float32)
    base_b = np.asarray(data_b["query"] @ data_b["gallery"].T, dtype=np.float32)
    with torch.no_grad():
        query_a = model.query(torch.as_tensor(data_a["query"], dtype=torch.float32, device=device_obj))
        query_b = model.query(torch.as_tensor(data_b["query"], dtype=torch.float32, device=device_obj))
        gallery_a = model.doc(torch.as_tensor(data_a["gallery"], dtype=torch.float32, device=device_obj))
        gallery_b = model.doc(torch.as_tensor(data_b["gallery"], dtype=torch.float32, device=device_obj))
        adapted_a = (query_a @ gallery_a.T).detach().cpu().numpy()
        adapted_b = (query_b @ gallery_b.T).detach().cpu().numpy()

    candidates = [float(alpha)] if alpha is not None else sorted(set(float(value) for value in alpha_grid))
    if not candidates or any(value < 0.0 or value > 1.0 for value in candidates):
        raise ValueError("fusion alpha values must be within [0, 1]")
    selection_rows: list[dict[str, Any]] = []
    for value in candidates:
        scores = value * adapted_a + (1.0 - value) * adapted_b
        recall = _recall_from_scores(scores, topk=(1, 5, 10), positive_index=positive_a)
        reference_scores = np.asarray([scores[row, int(reference_a[row])] for row in range(scores.shape[0])])
        beats = _target_beats_reference_summary(scores, reference_scores, positive_index=positive_a)
        selection_rows.append(
            {
                "alpha_cache_a": value,
                **recall,
                "target_beats_reference": float(beats["target_beats_reference_rate"]),
            }
        )
    selection_rows.sort(key=lambda row: (-row["R@1"], -row["target_beats_reference"], abs(row["alpha_cache_a"] - 0.5)))
    selected_alpha = float(selection_rows[0]["alpha_cache_a"])
    base_scores = selected_alpha * base_a + (1.0 - selected_alpha) * base_b
    adapted_scores = selected_alpha * adapted_a + (1.0 - selected_alpha) * adapted_b
    base_reference_scores = np.asarray([base_scores[row, int(reference_a[row])] for row in range(base_scores.shape[0])])
    adapted_reference_scores = np.asarray([adapted_scores[row, int(reference_a[row])] for row in range(adapted_scores.shape[0])])
    base_recall = _recall_from_scores(base_scores, topk=(1, 5, 10), positive_index=positive_a)
    adapted_recall = _recall_from_scores(adapted_scores, topk=(1, 5, 10), positive_index=positive_a)
    gallery_items = _eval_gallery_items_for_output(cache_a_root, records_a, adapted_scores.shape[1])
    summary = {
        "cache_a": str(cache_a_root),
        "cache_b": str(cache_b_root),
        "adapter_dir": str(adapter_root),
        "output_dir": str(output_root),
        "eval_count": len(records_a),
        "gallery_count": int(adapted_scores.shape[1]),
        "fusion": {
            "definition": "alpha * cache_a_scores + (1-alpha) * cache_b_scores",
            "selected_alpha_cache_a": selected_alpha,
            "selection_split": "validation" if alpha is None else "fixed_from_validation",
            "selection_rows": selection_rows,
        },
        "rows": [
            {"method": "base_e5_global", **base_recall},
            {"method": "audio_delta_adapter_global", **adapted_recall},
        ],
        "target_beats_reference": {
            "base_e5": _target_beats_reference_summary(base_scores, base_reference_scores, positive_index=positive_a),
            "audio_delta_adapter": _target_beats_reference_summary(
                adapted_scores, adapted_reference_scores, positive_index=positive_a
            ),
        },
        "reference_rank_summary": _reference_rank_summary(adapted_scores, adapted_reference_scores),
        "base_reference_rank_summary": _reference_rank_summary(base_scores, base_reference_scores),
        "by_audio_delta_type": _grouped_recall_summary(
            base_scores, adapted_scores, records_a, "audio_delta_type", (1, 5, 10), positive_index=positive_a
        ),
        "base_gallery_negative_recall_by_type": _gallery_negative_recall_by_type(
            base_scores, gallery_items, records_a, positive_index=positive_a
        ),
        "gallery_negative_recall_by_type": _gallery_negative_recall_by_type(
            adapted_scores, gallery_items, records_a, positive_index=positive_a
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "selected_alpha.json", {"alpha_cache_a": selected_alpha})
    _write_eval_topk_outputs(
        output_root=output_root,
        records=records_a,
        gallery_items=gallery_items,
        base_scores=base_scores,
        adapted_scores=adapted_scores,
        positive_index=positive_a,
        reference_index=reference_a,
        save_topk=max(1, int(save_topk)),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-grade Audio-CVR split, selection, and statistics utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("prepare-splits")
    split.add_argument("--split-root", required=True)
    split.add_argument("--output-dir", required=True)

    review = subparsers.add_parser("prepare-benchmark-review")
    review.add_argument("--input-path", required=True)
    review.add_argument("--output-dir", required=True)
    review.add_argument("--exclude-path", action="append", default=[])
    review.add_argument(
        "--local-candidate-path",
        action="append",
        default=[],
        help="Mined local_same_source candidate JSONL to attach to the model-blind review pool.",
    )
    review.add_argument("--review-count", type=int, default=225)
    review.add_argument("--repeat-review-fraction", type=float, default=0.20)
    review.add_argument("--random-seed", type=int, default=20260719)
    review.add_argument(
        "--eligible-tiers",
        default="main",
        help="Comma-separated automatic tiers eligible for review. Extended records require promotion checks.",
    )

    automatic_prepare = subparsers.add_parser("prepare-automatic-benchmark-review")
    automatic_prepare.add_argument("--input-path", action="append", required=True)
    automatic_prepare.add_argument("--output-dir", required=True)
    automatic_prepare.add_argument(
        "--review-pool-targets",
        default="sound_event=180,music=70,speech_topic_in_video_context=180",
    )
    automatic_prepare.add_argument("--max-per-source", type=int, default=2)
    automatic_prepare.add_argument("--random-seed", type=int, default=20260720)

    automatic_review = subparsers.add_parser("review-benchmark-omni")
    automatic_review.add_argument("--candidate-path", required=True)
    automatic_review.add_argument("--output-path", required=True)
    automatic_review.add_argument("--media-root", required=True)
    automatic_review.add_argument("--cache-dir", required=True)
    automatic_review.add_argument("--base-url", required=True)
    automatic_review.add_argument("--api-key", default="EMPTY")
    automatic_review.add_argument("--model", required=True)
    automatic_review.add_argument("--review-pass-id", type=int, choices=(1, 2), default=1)
    automatic_review.add_argument("--pass1-review-path", action="append", default=[])
    automatic_review.add_argument("--repeat-review-fraction", type=float, default=0.20)
    automatic_review.add_argument("--random-seed", type=int, default=20260720)
    automatic_review.add_argument("--shard-index", type=int, default=0)
    automatic_review.add_argument("--shard-count", type=int, default=1)
    automatic_review.add_argument("--timeout-seconds", type=float, default=180.0)
    automatic_review.add_argument("--omni-retries", type=int, default=2)
    automatic_review.add_argument("--resume", action="store_true")
    automatic_review.add_argument("--fail-on-error", action="store_true")

    automatic_finalize = subparsers.add_parser("finalize-automatic-benchmark")
    automatic_finalize.add_argument("--combined-pool-path", required=True)
    automatic_finalize.add_argument("--candidate-path", required=True)
    automatic_finalize.add_argument("--pass1-review-path", action="append", required=True)
    automatic_finalize.add_argument("--pass2-review-path", action="append", required=True)
    automatic_finalize.add_argument("--output-dir", required=True)
    automatic_finalize.add_argument(
        "--subtype-targets",
        default="sound_event=90,music=30,speech_topic_in_video_context=30",
    )
    automatic_finalize.add_argument(
        "--validation-targets",
        default="sound_event=45,music=15,speech_topic_in_video_context=15",
    )
    automatic_finalize.add_argument("--repeat-review-fraction", type=float, default=0.20)
    automatic_finalize.add_argument("--max-dataset-ratio", type=float, default=0.50)
    automatic_finalize.add_argument("--relaxed-dataset-ratio", type=float, default=0.55)
    automatic_finalize.add_argument("--max-hdtf-ratio", type=float, default=0.15)
    automatic_finalize.add_argument("--max-voxceleb-ratio", type=float, default=0.05)
    automatic_finalize.add_argument("--max-per-source", type=int, default=1)
    automatic_finalize.add_argument("--random-seed", type=int, default=20260720)

    split_audit = subparsers.add_parser("audit-training-splits")
    split_audit.add_argument("--train-path", required=True)
    split_audit.add_argument("--val-path", required=True)
    split_audit.add_argument("--test-path", required=True)
    split_audit.add_argument("--output-dir", required=True)

    freeze = subparsers.add_parser("finalize-benchmark")
    freeze.add_argument("--candidate-path", required=True)
    freeze.add_argument("--review-path", action="append", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.add_argument("--exclude-path", action="append", default=[])
    freeze.add_argument("--target-count", type=int, default=150)
    freeze.add_argument("--minimum-count", type=int, default=100)
    freeze.add_argument("--max-speech-ratio", type=float, default=0.35)
    freeze.add_argument(
        "--max-dataset-ratio",
        type=float,
        default=None,
        help="Defaults to 0.60 for human review and 0.50 for omni_consensus.",
    )
    freeze.add_argument("--max-per-source", type=int, default=1)
    freeze.add_argument("--min-strict-local-coverage", type=float, default=0.0)
    freeze.add_argument("--random-seed", type=int, default=20260719)
    freeze.add_argument("--review-policy", choices=("human", "omni_consensus"), default="human")
    freeze.add_argument("--combined-pool-path")
    freeze.add_argument("--pass2-review-path", action="append", default=[])
    freeze.add_argument(
        "--subtype-targets",
        default="sound_event=90,music=30,speech_topic_in_video_context=30",
    )
    freeze.add_argument(
        "--validation-targets",
        default="sound_event=45,music=15,speech_topic_in_video_context=15",
    )
    freeze.add_argument("--repeat-review-fraction", type=float, default=0.20)
    freeze.add_argument("--relaxed-dataset-ratio", type=float, default=0.55)
    freeze.add_argument("--max-hdtf-ratio", type=float, default=0.15)
    freeze.add_argument("--max-voxceleb-ratio", type=float, default=0.05)
    freeze.add_argument(
        "--eligible-tiers",
        default="main",
        help="Comma-separated automatic tiers eligible for freezing. Extended records require promotion checks.",
    )

    validation = subparsers.add_parser("summarize-validation")
    validation.add_argument("--input-root", action="append", required=True)
    validation.add_argument("--output-dir", required=True)
    validation.add_argument("--required-seeds", default="13")
    validation.add_argument("--top-n", type=int, default=6)
    validation.add_argument("--selection-rule", choices=("lexicographic", "one_se_earliest"), default="lexicographic")

    final = subparsers.add_parser("aggregate-final")
    final.add_argument("--input-root", required=True)
    final.add_argument("--output-dir", required=True)
    final.add_argument("--required-seeds", default="13,23,42,71,101")
    final.add_argument("--primary-mode", default=PRIMARY_MODE)
    final.add_argument("--reference-mode", default=REFERENCE_MODE)
    final.add_argument("--bootstrap-samples", type=int, default=20_000)
    final.add_argument("--permutation-samples", type=int, default=20_000)
    final.add_argument("--random-seed", type=int, default=20260718)
    final.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="MODE_B:MODE_A",
        help="Additional paired comparison reported as MODE_B - MODE_A. Repeat as needed.",
    )

    fusion = subparsers.add_parser("score-fusion")
    fusion.add_argument("--cache-a", required=True, help="First modality cache, conventionally V+T.")
    fusion.add_argument("--cache-b", required=True, help="Second modality cache, conventionally A+T.")
    fusion.add_argument("--adapter-dir", required=True)
    fusion.add_argument("--output-dir", required=True)
    fusion.add_argument("--alpha", type=float)
    fusion.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    fusion.add_argument("--device", default="cuda")
    fusion.add_argument("--save-topk", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-splits":
        result = prepare_paper_splits(split_root=args.split_root, output_dir=args.output_dir)
    elif args.command == "prepare-benchmark-review":
        result = prepare_benchmark_review(
            input_path=args.input_path,
            output_dir=args.output_dir,
            exclude_paths=args.exclude_path,
            local_candidate_paths=args.local_candidate_path,
            review_count=args.review_count,
            repeat_review_fraction=args.repeat_review_fraction,
            random_seed=args.random_seed,
            eligible_tiers=_parse_strings(args.eligible_tiers),
        )
    elif args.command == "prepare-automatic-benchmark-review":
        result = prepare_automatic_benchmark_review(
            input_paths=args.input_path,
            output_dir=args.output_dir,
            review_pool_targets=_parse_named_ints(args.review_pool_targets),
            max_per_source=args.max_per_source,
            random_seed=args.random_seed,
        )
    elif args.command == "review-benchmark-omni":
        result = review_benchmark_omni(
            candidate_path=args.candidate_path,
            output_path=args.output_path,
            media_root=args.media_root,
            cache_dir=args.cache_dir,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            review_pass_id=args.review_pass_id,
            pass1_review_paths=args.pass1_review_path,
            repeat_review_fraction=args.repeat_review_fraction,
            random_seed=args.random_seed,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            timeout_seconds=args.timeout_seconds,
            omni_retries=args.omni_retries,
            resume=args.resume,
            fail_on_error=args.fail_on_error,
        )
    elif args.command == "finalize-automatic-benchmark":
        result = finalize_automatic_benchmark(
            combined_pool_path=args.combined_pool_path,
            candidate_path=args.candidate_path,
            pass1_review_paths=args.pass1_review_path,
            pass2_review_paths=args.pass2_review_path,
            output_dir=args.output_dir,
            subtype_targets=_parse_named_ints(args.subtype_targets),
            validation_targets=_parse_named_ints(args.validation_targets),
            repeat_review_fraction=args.repeat_review_fraction,
            max_dataset_ratio=args.max_dataset_ratio,
            relaxed_dataset_ratio=args.relaxed_dataset_ratio,
            max_hdtf_ratio=args.max_hdtf_ratio,
            max_voxceleb_ratio=args.max_voxceleb_ratio,
            max_per_source=args.max_per_source,
            random_seed=args.random_seed,
        )
    elif args.command == "audit-training-splits":
        result = audit_training_splits(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            output_dir=args.output_dir,
        )
    elif args.command == "finalize-benchmark":
        if args.review_policy == "omni_consensus":
            if not args.combined_pool_path or not args.pass2_review_path:
                raise ValueError(
                    "omni_consensus requires --combined-pool-path and at least one --pass2-review-path"
                )
            subtype_targets = _parse_named_ints(args.subtype_targets)
            if sum(subtype_targets.values()) != int(args.target_count):
                raise ValueError("--target-count must equal the sum of --subtype-targets")
            result = finalize_automatic_benchmark(
                combined_pool_path=args.combined_pool_path,
                candidate_path=args.candidate_path,
                pass1_review_paths=args.review_path,
                pass2_review_paths=args.pass2_review_path,
                output_dir=args.output_dir,
                subtype_targets=subtype_targets,
                validation_targets=_parse_named_ints(args.validation_targets),
                repeat_review_fraction=args.repeat_review_fraction,
                max_dataset_ratio=0.50 if args.max_dataset_ratio is None else args.max_dataset_ratio,
                relaxed_dataset_ratio=args.relaxed_dataset_ratio,
                max_hdtf_ratio=args.max_hdtf_ratio,
                max_voxceleb_ratio=args.max_voxceleb_ratio,
                max_per_source=args.max_per_source,
                random_seed=args.random_seed,
            )
        else:
            result = finalize_benchmark(
                candidate_path=args.candidate_path,
                review_paths=args.review_path,
                output_dir=args.output_dir,
                exclude_paths=args.exclude_path,
                target_count=args.target_count,
                minimum_count=args.minimum_count,
                max_speech_ratio=args.max_speech_ratio,
                max_dataset_ratio=0.60 if args.max_dataset_ratio is None else args.max_dataset_ratio,
                max_per_source=args.max_per_source,
                min_strict_local_coverage=args.min_strict_local_coverage,
                random_seed=args.random_seed,
                eligible_tiers=_parse_strings(args.eligible_tiers),
            )
    elif args.command == "summarize-validation":
        result = summarize_validation(
            input_roots=args.input_root,
            output_dir=args.output_dir,
            required_seeds=_parse_ints(args.required_seeds),
            top_n=args.top_n,
            selection_rule=args.selection_rule,
        )
    elif args.command == "aggregate-final":
        result = aggregate_final(
            input_root=args.input_root,
            output_dir=args.output_dir,
            required_seeds=_parse_ints(args.required_seeds),
            primary_mode=args.primary_mode,
            reference_mode=args.reference_mode,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            random_seed=args.random_seed,
            comparisons=[_parse_mode_comparison(value) for value in args.comparison] or None,
        )
    elif args.command == "score-fusion":
        result = score_fusion(
            cache_a=args.cache_a,
            cache_b=args.cache_b,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            alpha=args.alpha,
            alpha_grid=(float(item.strip()) for item in args.alpha_grid.split(",") if item.strip()),
            device=args.device,
            save_topk=args.save_topk,
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _validation_observation(
    train_summary: dict[str, Any], eval_summary: dict[str, Any], adapter_dir: Path, eval_dir: Path
) -> dict[str, Any]:
    metrics = _eval_metrics(eval_summary)
    return {
        "steps": int(train_summary["steps"]),
        "learning_rate": float(train_summary["learning_rate"]),
        "batch_size": int(train_summary["batch_size"]),
        "seed": int(train_summary["seed"]),
        "adapter_dir": str(adapter_dir),
        "eval_dir": str(eval_dir),
        **metrics,
    }


def _eval_metrics(summary: dict[str, Any]) -> dict[str, float]:
    adapter = next((row for row in summary.get("rows", []) if row.get("method") == "audio_delta_adapter_global"), None)
    base = next((row for row in summary.get("rows", []) if row.get("method") == "base_e5_global"), None)
    if not adapter or not base:
        raise ValueError("evaluation summary lacks base_e5_global or audio_delta_adapter_global")
    beats = summary.get("target_beats_reference", {}).get("audio_delta_adapter", {})
    base_beats = summary.get("target_beats_reference", {}).get("base_e5", {})
    reference_rank = summary.get("reference_rank_summary", {})
    base_reference_rank = summary.get("base_reference_rank_summary", {})
    return {
        "R@1": float(adapter.get("R@1", 0.0)),
        "R@5": float(adapter.get("R@5", 0.0)),
        "R@10": float(adapter.get("R@10", 0.0)),
        "base_R@1": float(base.get("R@1", 0.0)),
        "base_R@5": float(base.get("R@5", 0.0)),
        "base_R@10": float(base.get("R@10", 0.0)),
        "target_beats_reference": float(beats.get("target_beats_reference_rate", 0.0)),
        "target_ref_gap": float(beats.get("target_minus_reference_mean", 0.0)),
        "base_target_beats_reference": float(base_beats.get("target_beats_reference_rate", 0.0)),
        "base_target_ref_gap": float(base_beats.get("target_minus_reference_mean", 0.0)),
        "reference_rank_median": float(reference_rank.get("median_rank", 0.0)),
        "reference_rank_le_1": float(reference_rank.get("rank_le_1_rate", 0.0)),
        "base_reference_rank_median": float(base_reference_rank.get("median_rank", 0.0)),
    }


def _paired_mode_statistics(
    *,
    runs: dict[tuple[int, str], dict[str, Any]],
    seeds: tuple[int, ...],
    mode_a: str,
    mode_b: str,
    bootstrap_samples: int,
    permutation_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    metrics = ("R@1", "R@5", "R@10", "target_beats_reference", "target_ref_gap", "reciprocal_rank")
    by_metric: dict[str, list[list[float]]] = {metric: [] for metric in metrics}
    mcnemar_rows: list[dict[str, Any]] = []
    sample_ids: list[str] | None = None
    for seed in seeds:
        rows_a = {str(row["sample_id"]): row for row in runs[(seed, mode_a)]["scores"]}
        rows_b = {str(row["sample_id"]): row for row in runs[(seed, mode_b)]["scores"]}
        ids = sorted(set(rows_a) & set(rows_b))
        if not ids or len(ids) != len(rows_a) or len(ids) != len(rows_b):
            raise ValueError(f"paired sample mismatch for seed {seed}: {len(rows_a)} vs {len(rows_b)}")
        if sample_ids is None:
            sample_ids = ids
        elif ids != sample_ids:
            raise ValueError(f"sample order/set differs across seeds at seed {seed}")
        seed_values: dict[str, list[float]] = {metric: [] for metric in metrics}
        hits_a: list[int] = []
        hits_b: list[int] = []
        for sample_id in ids:
            a = rows_a[sample_id]
            b = rows_b[sample_id]
            rank_a = int(a["adapter_target_rank"])
            rank_b = int(b["adapter_target_rank"])
            hits_a.append(int(rank_a <= 1))
            hits_b.append(int(rank_b <= 1))
            seed_values["R@1"].append(float(rank_b <= 1) - float(rank_a <= 1))
            seed_values["R@5"].append(float(rank_b <= 5) - float(rank_a <= 5))
            seed_values["R@10"].append(float(rank_b <= 10) - float(rank_a <= 10))
            seed_values["target_beats_reference"].append(
                float(float(b.get("adapter_target_minus_reference") or 0.0) > 0)
                - float(float(a.get("adapter_target_minus_reference") or 0.0) > 0)
            )
            seed_values["target_ref_gap"].append(
                float(b.get("adapter_target_minus_reference") or 0.0)
                - float(a.get("adapter_target_minus_reference") or 0.0)
            )
            seed_values["reciprocal_rank"].append((1.0 / rank_b) - (1.0 / rank_a))
        for metric in metrics:
            by_metric[metric].append(seed_values[metric])
        mcnemar_rows.append({"seed": seed, **_mcnemar_exact(hits_a, hits_b)})

    corrected = _holm_bonferroni([float(row["p_value"]) for row in mcnemar_rows])
    for row, adjusted in zip(mcnemar_rows, corrected):
        row["p_value_holm"] = adjusted

    rng = random.Random(random_seed)
    results: dict[str, Any] = {}
    for metric, seed_rows in by_metric.items():
        query_differences = [statistics.fmean(values) for values in zip(*seed_rows)]
        observed = statistics.fmean(query_differences)
        low, high = _paired_bootstrap_ci(query_differences, samples=bootstrap_samples, rng=rng)
        p_value = _sign_flip_p_value(query_differences, samples=permutation_samples, rng=rng)
        results[metric] = {
            "difference_mean": observed,
            "bootstrap_95_ci": [low, high],
            "paired_randomization_p": p_value,
            "query_count": len(query_differences),
            "seed_count": len(seed_rows),
        }
    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "difference_definition": "mode_b - mode_a",
        "query_level_statistics": results,
        "mcnemar_R@1_by_seed": mcnemar_rows,
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
    }


def _mcnemar_exact(a_hits: list[int], b_hits: list[int]) -> dict[str, Any]:
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if not a and b)
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_value": p_value}


def _paired_bootstrap_ci(values: list[float], *, samples: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    estimates = []
    for _ in range(samples):
        estimates.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _sign_flip_p_value(values: list[float], *, samples: int, rng: random.Random) -> float:
    if not values:
        return 1.0
    observed = abs(statistics.fmean(values))
    exceed = 0
    for _ in range(samples):
        permuted = abs(statistics.fmean(value if rng.random() < 0.5 else -value for value in values))
        exceed += int(permuted >= observed - 1e-15)
    return (exceed + 1.0) / (samples + 1.0)


def _holm_bonferroni(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        current = min(1.0, values[index] * (count - rank))
        running = max(running, current)
        adjusted[index] = running
    return adjusted


def _add_comparison_holm_corrections(comparisons: dict[str, Any]) -> None:
    metric_names = sorted(
        {
            metric
            for comparison in comparisons.values()
            for metric in comparison.get("query_level_statistics", {})
        }
    )
    for metric in metric_names:
        keys = [key for key, value in comparisons.items() if metric in value.get("query_level_statistics", {})]
        adjusted = _holm_bonferroni(
            [
                float(comparisons[key]["query_level_statistics"][metric]["paired_randomization_p"])
                for key in keys
            ]
        )
        for key, value in zip(keys, adjusted):
            comparisons[key]["query_level_statistics"][metric]["paired_randomization_p_holm"] = value


def _audit_final_runs(
    *, runs: dict[tuple[int, str], dict[str, Any]], seeds: tuple[int, ...], modes: list[str]
) -> dict[str, Any]:
    violations: list[str] = []
    expected_ids: list[str] | None = None
    expected_count: int | None = None
    expected_gallery: int | None = None
    for seed in seeds:
        for mode in modes:
            run = runs[(seed, mode)]
            ids = [str(row.get("sample_id")) for row in run["scores"]]
            summary = run["summary"]
            if expected_ids is None:
                expected_ids = ids
                expected_count = int(summary.get("eval_count", len(ids)))
                expected_gallery = int(summary.get("gallery_count", 0))
            if ids != expected_ids:
                violations.append(f"sample_ids differ for seed={seed},mode={mode}")
            if int(summary.get("eval_count", len(ids))) != expected_count:
                violations.append(f"eval_count differs for seed={seed},mode={mode}")
            if int(summary.get("gallery_count", 0)) != expected_gallery:
                violations.append(f"gallery_count differs for seed={seed},mode={mode}")
            if _contains_non_finite(summary):
                violations.append(f"non-finite summary value for seed={seed},mode={mode}")
    return {
        "seed_count": len(seeds),
        "mode_count": len(modes),
        "eval_count": expected_count,
        "gallery_count": expected_gallery,
        "sample_ids_identical": not any("sample_ids" in item for item in violations),
        "violation_count": len(violations),
        "violations": violations,
    }


def _error_breakdown(runs: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    for run in runs:
        topk_path = Path(run["eval_dir"]) / "per_query_topk.jsonl"
        for row in _read_jsonl(topk_path):
            if int(row.get("adapter_target_rank", 10**9)) <= 1:
                continue
            total += 1
            top1 = (row.get("adapter_topk") or [{}])[0]
            kind = str(top1.get("negative_type") or top1.get("kind") or "unknown")
            if top1.get("is_reference"):
                kind = "reference_negative"
            counts[kind] += 1
    return {
        "error_count_across_seeds": total,
        "counts": dict(sorted(counts.items())),
        "rates": {key: value / max(1, total) for key, value in sorted(counts.items())},
    }


def _hard_negative_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for summary in summaries:
        for key, value in summary.get("gallery_negative_recall_by_type", {}).items():
            if isinstance(value, dict) and "positive_beats_negative_rate" in value:
                buckets[key].append(float(value["positive_beats_negative_rate"]))
    return {
        key: {"mean": _mean(values), "std": _std(values), "seed_count": len(values)}
        for key, values in sorted(buckets.items())
    }


def _subtype_result_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    subtype_names = sorted({name for summary in summaries for name in summary.get("by_audio_delta_type", {})})
    for name in subtype_names:
        values = [summary.get("by_audio_delta_type", {}).get(name) for summary in summaries]
        values = [value for value in values if isinstance(value, dict)]
        if not values:
            continue
        adapter_r1 = [float(value.get("audio_delta_adapter", {}).get("R@1", 0.0)) for value in values]
        base_r1 = [float(value.get("base_e5", {}).get("R@1", 0.0)) for value in values]
        buckets[name] = {
            "count": int(values[0].get("count", 0)),
            "adapter_R@1_mean": _mean(adapter_r1),
            "adapter_R@1_std": _std(adapter_r1),
            "base_R@1_mean": _mean(base_r1),
        }
    return buckets


def _split_leakage_summary(*, train: list[dict[str, Any]], val: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    splits = {"train": train, "val": val, "test": test}
    violations: list[dict[str, Any]] = []
    for key_name, fields in {
        "source": ("source_disjoint_group_id", "raw_source_id", "source_id"),
        "pair": ("inverse_pair_group_id", "pair_group_id"),
    }.items():
        for field in fields:
            sets = {
                name: {str(row.get(field) or "").strip() for row in rows} - {""}
                for name, rows in splits.items()
            }
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = sorted(sets[left] & sets[right])
                if overlap:
                    violations.append(
                        {
                            "type": key_name,
                            "field": field,
                            "splits": [left, right],
                            "count": len(overlap),
                            "examples": overlap[:5],
                        }
                    )
    return {"violation_count": sum(item["count"] for item in violations), "violations": violations}


def _benchmark_candidate_reject_reason(
    row: dict[str, Any],
    *,
    excluded: dict[str, set[str]],
    eligible_tiers: set[str] | None = None,
) -> str:
    eligible_tiers = eligible_tiers or {"main"}
    automatic_tier = _automatic_split_tier(row)
    if automatic_tier not in eligible_tiers:
        return f"tier_not_eligible:{automatic_tier}"
    if _truthy(row.get("is_inverse")) or str(row.get("direction") or "forward").strip().lower() == "inverse":
        return "inverse_direction"
    if _truthy(row.get("fallback")):
        return "fallback_record"
    if "accepted" in row and not _truthy(row.get("accepted")):
        return "not_accepted"
    if automatic_tier == "main" and "benchmark_eligible" in row and not _truthy(row.get("benchmark_eligible")):
        return "not_benchmark_eligible"
    if automatic_tier == "extended" and "training_eligible" in row and not _truthy(row.get("training_eligible")):
        return "extended_not_training_eligible"
    if _truthy(row.get("manual_review_required")):
        return "preexisting_manual_review_required"
    if not _sample_id(row):
        return "missing_sample_id"
    source_ids = _row_identity_values(row, ("source_disjoint_group_id", "raw_source_id", "source_id"))
    if not source_ids:
        return "missing_source_id"
    if source_ids & excluded["source"]:
        return "source_seen_in_prior_split"
    pair_ids = _row_identity_values(row, ("inverse_pair_group_id", "pair_group_id"))
    if pair_ids & excluded["pair"]:
        return "pair_seen_in_prior_split"
    if _sample_id(row) in excluded["sample"]:
        return "sample_seen_in_prior_split"
    return ""


def _identity_sets(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    source: set[str] = set()
    pair: set[str] = set()
    sample: set[str] = set()
    for row in rows:
        source.update(_row_identity_values(row, ("source_disjoint_group_id", "raw_source_id", "source_id")))
        pair.update(_row_identity_values(row, ("inverse_pair_group_id", "pair_group_id")))
        sample_id = _sample_id(row)
        if sample_id:
            sample.add(sample_id)
    return {"source": source, "pair": pair, "sample": sample}


def _row_identity_values(row: dict[str, Any], fields: Iterable[str]) -> set[str]:
    return {str(row.get(field) or "").strip() for field in fields} - {""}


def _read_many_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(Path(path)))
    return rows


def _local_candidates_by_sample(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        sample_id = _first_text(row, ("sample_id", "query_sample_id"))
        video = _first_text(row, ("video", "path", "video_path"))
        if not sample_id or not video or (sample_id, video) in seen:
            continue
        seen.add((sample_id, video))
        by_sample[sample_id].append(dict(row))
    return dict(by_sample)


def _formal_human_review_row(row: dict[str, Any], *, review_round: int) -> dict[str, Any]:
    review = _human_review_row(row)
    automatic_tier = _automatic_split_tier(row)
    review.update(
        {
            "dataset": _dataset(row),
            "source_disjoint_group_id": _first_text(
                row, ("source_disjoint_group_id", "raw_source_id", "source_id")
            ),
            "pair_group_id": _pair_id(row),
            "automatic_split_tier": automatic_tier,
            "promotion_review_required": automatic_tier == "extended",
            "automatic_diagnostic_reason": row.get("diagnostic_reason"),
            "review_round": int(review_round),
            "reviewer_id": "",
        }
    )
    return review


def _collate_reviews(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rounds: dict[str, dict[int, dict[str, tuple[bool, str]]]] = defaultdict(dict)
    tiers: dict[str, str] = {}
    decisions: Counter[str] = Counter()
    for row in rows:
        sample_id = _sample_id(row)
        if not sample_id:
            continue
        review = row.get("review") if isinstance(row.get("review"), dict) else row
        decision = str(review.get("decision") or "unreviewed").strip().lower()
        decisions[decision] += 1
        review_round = int(row.get("review_round") or review.get("review_round") or 1)
        tiers[sample_id] = str(row.get("automatic_split_tier") or "main").strip().lower()
        rounds[sample_id][review_round] = {
            "base": _review_pass_status(review),
            "promotion": _review_pass_status(review, require_extended_promotion=True),
        }

    result: dict[str, dict[str, Any]] = {}
    repeated = 0
    agreements = 0
    disagreements = 0
    for sample_id, values in rounds.items():
        primary = values.get(1) or values[min(values)]
        passed, reason = primary["base"]
        promotion_passed, promotion_reason = primary["promotion"]
        effective_key = "promotion" if tiers.get(sample_id) == "extended" else "base"
        if 2 in values:
            repeated += 1
            if values[2][effective_key][0] == primary[effective_key][0]:
                agreements += 1
            else:
                disagreements += 1
                passed = False
                reason = "repeat_review_disagreement"
                promotion_passed = False
                promotion_reason = "repeat_review_disagreement"
        result[sample_id] = {
            "passed": passed,
            "reason": reason,
            "promotion_passed": promotion_passed,
            "promotion_reason": promotion_reason,
            "automatic_split_tier": tiers.get(sample_id, "main"),
            "rounds": sorted(values),
        }
    return result, {
        "reviewed_sample_count": len(result),
        "decision_counts": dict(sorted(decisions.items())),
        "repeat_review_count": repeated,
        "repeat_review_agreement_count": agreements,
        "repeat_review_disagreement_count": disagreements,
        "repeat_review_agreement_rate": agreements / repeated if repeated else None,
    }


def _review_pass_status(
    review: dict[str, Any], *, require_extended_promotion: bool = False
) -> tuple[bool, str]:
    decision = str(review.get("decision") or "unreviewed").strip().lower()
    if decision not in {"pass", "passed", "accept", "accepted"}:
        return False, f"human_{decision or 'unreviewed'}"
    checks = BASE_BENCHMARK_REVIEW_CHECKS
    if require_extended_promotion:
        checks += EXTENDED_PROMOTION_REVIEW_CHECKS
    missing = [key for key in checks if review.get(key) is not True]
    if missing:
        return False, "human_check_failed_or_missing:" + ",".join(missing)
    return True, "passed"


def _normalize_eligible_tiers(values: Iterable[str]) -> set[str]:
    if isinstance(values, str):
        values = _parse_strings(values)
    tiers = {str(value).strip().lower() for value in values if str(value).strip()}
    unsupported = tiers - {"main", "extended"}
    if unsupported:
        raise ValueError(f"unsupported benchmark eligible tiers: {sorted(unsupported)}")
    if not tiers:
        raise ValueError("eligible_tiers cannot be empty")
    return tiers


def _automatic_split_tier(row: dict[str, Any]) -> str:
    return str(row.get("automatic_split_tier") or row.get("split_tier") or "main").strip().lower()


def _benchmark_output_row(
    row: dict[str, Any], *, human_verified_negatives: bool = False
) -> dict[str, Any]:
    output = dict(row)
    automatic_tier = _automatic_split_tier(row)
    output["automatic_split_tier"] = automatic_tier
    output["automatic_benchmark_eligible"] = (
        _truthy(row.get("benchmark_eligible")) if "benchmark_eligible" in row else automatic_tier == "main"
    )
    if automatic_tier == "extended":
        output["automatic_diagnostic_reason"] = row.get("diagnostic_reason")
        output["split_tier"] = "main"
        output["benchmark_eligible"] = True
        output["human_verified_benchmark_eligible"] = True
        output["benchmark_promotion"] = "human_verified_extended"
    else:
        output["human_verified_benchmark_eligible"] = False
        output["benchmark_promotion"] = "not_required"
    if human_verified_negatives:
        verified_local: list[dict[str, Any]] = []
        for item in row.get("local_same_source_candidates", []):
            if not isinstance(item, dict):
                continue
            verified = dict(item)
            verified["pre_review_verification_status"] = item.get("verification_status")
            verified["verification_status"] = "human_verified"
            verified["satisfies_edit"] = False
            verified["manual_review_required"] = False
            verified["verified_by_benchmark_review"] = True
            verified_local.append(verified)
        output["local_same_source_candidates"] = verified_local
    return output


def _select_balanced_benchmark(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    max_speech_ratio: float,
    max_dataset_ratio: float,
    max_per_source: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    speech_limit = max(0, math.floor(target_count * max_speech_ratio))
    dataset_limit = max(1, math.floor(target_count * max_dataset_ratio))
    remaining = sorted(rows, key=lambda row: _stable_row_key(row, random_seed))
    selected: list[dict[str, Any]] = []
    dataset_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    speech_count = 0
    while remaining and len(selected) < target_count:
        eligible = [
            row
            for row in remaining
            if dataset_counts[_dataset(row)] < dataset_limit
            and source_counts[_primary_source_id(row)] < max_per_source
            and (not _is_speech(row) or speech_count < speech_limit)
        ]
        if not eligible:
            break
        row = min(
            eligible,
            key=lambda item: (
                -int(_has_strict_local_negative(item)),
                dataset_counts[_dataset(item)],
                subtype_counts[_subtype(item)],
                _stable_row_key(item, random_seed),
            ),
        )
        selected.append(row)
        remaining.remove(row)
        dataset_counts[_dataset(row)] += 1
        subtype_counts[_subtype(row)] += 1
        source_counts[_primary_source_id(row)] += 1
        speech_count += int(_is_speech(row))
    return selected


def _has_strict_local_negative(row: dict[str, Any]) -> bool:
    collections = (
        row.get("audio_delta_hard_negatives"),
        row.get("hard_negatives"),
        row.get("local_same_source_candidates"),
    )
    for values in collections:
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            negative_type = str(item.get("negative_type") or item.get("type") or "").strip().lower()
            status = str(item.get("verification_status") or "").strip().lower()
            if (
                negative_type == "local_same_source"
                and _truthy(item.get("same_source", True))
                and item.get("satisfies_edit") is False
                and status in {"auto_verified", "human_verified", "verified"}
            ):
                return True
    return False


def _stable_row_key(row: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}|{_sample_id(row)}".encode("utf-8")).hexdigest()


def _sample_id(row: dict[str, Any]) -> str:
    return _first_text(row, ("sample_id", "proposal_id", "clip_id"))


def _pair_id(row: dict[str, Any]) -> str:
    return _first_text(row, ("inverse_pair_group_id", "pair_group_id"))


def _primary_source_id(row: dict[str, Any]) -> str:
    return _first_text(row, ("source_disjoint_group_id", "raw_source_id", "source_id")) or _sample_id(row)


def _dataset(row: dict[str, Any]) -> str:
    value = _first_text(row, ("dataset", "dataset_name", "source_dataset"))
    if not value and isinstance(row.get("quality"), dict):
        value = _first_text(row["quality"], ("dataset", "dataset_name"))
    return value or "unknown"


def _is_speech(row: dict[str, Any]) -> bool:
    return _subtype(row).strip().lower() in {"speech", "speech_topic", "speech_topic_in_video_context"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human_review_row(row: dict[str, Any]) -> dict[str, Any]:
    hard_negatives: list[dict[str, Any]] = []
    for key in ("audio_delta_hard_negatives", "hard_negatives", "local_same_source_candidates"):
        values = row.get(key)
        if isinstance(values, list):
            hard_negatives.extend(item for item in values if isinstance(item, dict))
    return {
        "sample_id": _first_text(row, ("sample_id", "proposal_id", "clip_id")),
        "reference_video": row.get("reference_video"),
        "target_video": row.get("target_video"),
        "edit_text": row.get("edit_text"),
        "b_subtype": row.get("b_subtype") or row.get("audio_delta_type"),
        "hard_negatives": hard_negatives,
        "review": {
            "edit_audio_only": None,
            "reference_does_not_satisfy_edit": None,
            "target_satisfies_edit": None,
            "video_only_cannot_identify_target": None,
            "hard_negatives_do_not_satisfy_edit": None,
            "audio_change_clearly_audible": None,
            "video_context_preserved": None,
            "not_asr_or_transcript_only": None,
            "decision": "unreviewed",
            "notes": "",
        },
    }


def _validation_markdown(summary: dict[str, Any]) -> str:
    selected = summary["selected_config"]
    lines = [
        "# Validation-only Model Selection",
        "",
        f"Selection rule: {summary['selection_rule']}",
        f"Selected: steps={selected['steps']}, lr={selected['learning_rate']:.6g}, batch={selected['batch_size']}.",
        "",
        "| Rank | Selected | Steps | LR | Batch | Seeds | R@1 mean | R@1 std | R@1 SE | Beats-ref mean |",
        "|---:|:---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        is_selected = (
            row["steps"] == selected["steps"]
            and row["learning_rate"] == selected["learning_rate"]
            and row["batch_size"] == selected["batch_size"]
        )
        lines.append(
            f"| {row['validation_rank']} | {'yes' if is_selected else ''} | {row['steps']} | "
            f"{row['learning_rate']:.6g} | {row['batch_size']} | "
            f"{','.join(str(seed) for seed in row['seeds'])} | {row['R@1_mean']:.4f} | {row['R@1_std']:.4f} | "
            f"{row['R@1_se']:.4f} | {row['target_beats_reference_mean']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _final_comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Audio-CVR Final Test Results",
        "",
        "| Mode | Model | R@1 | R@5 | R@10 | Target beats reference | Target-ref gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in sorted(summary["mode_summary"].items()):
        lines.append(
            f"| {mode} | Adapter | {_mean_std(metrics['R@1'])} | {_mean_std(metrics['R@5'])} | {_mean_std(metrics['R@10'])} | "
            f"{_mean_std(metrics['target_beats_reference'])} | {_mean_std(metrics['target_ref_gap'])} |"
        )
        lines.append(
            f"| {mode} | Base E5 | {_mean_std(metrics['base_R@1'])} | {_mean_std(metrics['base_R@5'])} | "
            f"{_mean_std(metrics['base_R@10'])} | {_mean_std(metrics['base_target_beats_reference'])} | "
            f"{_mean_std(metrics['base_target_ref_gap'])} |"
        )
    return "\n".join(lines) + "\n"


def _audio_gain_markdown(summary: dict[str, Any]) -> str:
    paired = summary["paired_statistics"]
    lines = [
        "# Audio Necessity: Paired Statistical Analysis",
        "",
        f"Primary comparison: `{paired['mode_b']} - {paired['mode_a']}`.",
        "",
        "| Metric | Mean difference | 95% paired bootstrap CI | Randomization p |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in paired["query_level_statistics"].items():
        low, high = values["bootstrap_95_ci"]
        lines.append(
            f"| {metric} | {values['difference_mean']:.6f} | [{low:.6f}, {high:.6f}] | "
            f"{values['paired_randomization_p']:.6g} |"
        )
    return "\n".join(lines) + "\n"


def _paired_comparisons_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Prespecified Paired Comparisons",
        "",
        "All differences are query-paired and averaged across the required final seeds.",
        "",
        "| Comparison | Metric | Difference | 95% bootstrap CI | Randomization p | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, comparison in summary.get("paired_comparisons", {}).items():
        for metric, values in comparison.get("query_level_statistics", {}).items():
            low, high = values["bootstrap_95_ci"]
            lines.append(
                f"| {name} | {metric} | {values['difference_mean']:.6f} | [{low:.6f}, {high:.6f}] | "
                f"{values['paired_randomization_p']:.6g} | {values.get('paired_randomization_p_holm', 1.0):.6g} |"
            )
    return "\n".join(lines) + "\n"


def _write_config_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{row['steps']}\t{row['learning_rate']:.12g}\t{row['batch_size']}\n" for row in rows),
        encoding="utf-8",
    )


def _mean_std(value: dict[str, float]) -> str:
    return f"{value['mean']:.4f} +/- {value['std']:.4f}"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _subtype(row: dict[str, Any]) -> str:
    return str(row.get("b_subtype") or row.get("audio_delta_type") or "unknown")


def _canonical_subtype(row: dict[str, Any]) -> str:
    return _canonical_subtype_name(_subtype(row))


def _canonical_subtype_name(value: Any) -> str:
    subtype = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if subtype in {"speech", "speech_topic", "speech_audio_content", "speech_topic_in_video_context"}:
        return "speech_topic_in_video_context"
    if subtype in {"music", "musical_event"}:
        return "music"
    if subtype in {"sound", "sound_event", "audio_event", "audio"}:
        return "sound_event"
    return "unknown"


def _normalized_dataset(row: dict[str, Any]) -> str:
    explicit = _dataset(row).strip().lower().replace("-", "_")
    text = " ".join(
        str(value or "").lower()
        for value in (
            explicit,
            row.get("reference_video"),
            row.get("target_video"),
            row.get("source_id"),
        )
    )
    aliases = (
        ("vgg_monoaudio", "vgg_monoaudio"),
        ("vggsound", "vggsound"),
        ("voxceleb", "voxceleb"),
        ("worldsense", "worldsense"),
        ("daily_omni", "daily_omni"),
        ("hdtf", "hdtf"),
        ("avatar", "avatar"),
    )
    for token, name in aliases:
        if token in text:
            return name
    return explicit or "unknown"


def _stable_source_id(row: dict[str, Any]) -> str:
    explicit = _first_text(row, ("source_disjoint_group_id", "raw_source_id", "source_id", "source_group_id"))
    if explicit:
        return explicit
    path = _first_text(row, ("reference_video", "target_video"))
    stem = Path(path).stem
    stem = re.sub(r"(?:__|[_-])(?:single|clip|segment|part)[_-]?\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?:__|[_-])\d{1,6}$", "", stem)
    dataset = _normalized_dataset(row)
    if stem:
        return f"{dataset}:{stem}"
    digest = hashlib.sha256(
        f"{dataset}|{row.get('reference_video')}|{row.get('target_video')}".encode("utf-8")
    ).hexdigest()[:20]
    return f"derived_source:{digest}"


def _stable_sample_id(row: dict[str, Any]) -> str:
    explicit = _first_text(row, ("sample_id", "proposal_id", "clip_id"))
    if explicit:
        return explicit
    digest = hashlib.sha256(
        "|".join(
            (
                _first_text(row, ("reference_video",)),
                _first_text(row, ("target_video",)),
                str(row.get("edit_text") or row.get("audio_only_edit_text") or "").strip(),
            )
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"audiocvr_{digest}"


def _stable_pair_group_id(row: dict[str, Any]) -> str:
    explicit = _first_text(row, ("inverse_pair_group_id", "pair_group_id"))
    if explicit:
        return explicit
    pair = sorted(
        (
            _normalize_media_identity(_first_text(row, ("reference_video",))),
            _normalize_media_identity(_first_text(row, ("target_video",))),
        )
    )
    digest = hashlib.sha256(f"{_stable_source_id(row)}|{pair[0]}|{pair[1]}".encode("utf-8")).hexdigest()[:24]
    return f"pair_{digest}"


def _normalized_reference_target_key(row: dict[str, Any]) -> str:
    reference = _normalize_media_identity(_first_text(row, ("reference_video", "reference_clip_path")))
    target = _normalize_media_identity(_first_text(row, ("target_video", "target_clip_path")))
    return f"{reference}|{target}" if reference and target else ""


def _normalize_media_identity(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _numeric_field(row: dict[str, Any], key: str) -> float:
    sources = (
        row,
        row.get("quality") if isinstance(row.get("quality"), dict) else {},
        row.get("final_omni_verification") if isinstance(row.get("final_omni_verification"), dict) else {},
        row.get("audio_delta_analysis") if isinstance(row.get("audio_delta_analysis"), dict) else {},
    )
    for source in sources:
        if key not in source:
            continue
        try:
            return min(1.0, max(0.0, float(source.get(key) or 0.0)))
        except (TypeError, ValueError):
            continue
    return 0.0


def _diagnostic_reasons(row: dict[str, Any]) -> set[str]:
    value = row.get("diagnostic_reason")
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if str(value or "").strip():
        return {str(value).strip()}
    return set()


def _existing_verifier_complete(row: dict[str, Any]) -> bool:
    return all(
        isinstance(row.get(key), dict) and bool(row.get(key))
        for key in ("audio_only_verification", "video_only_shortcut", "full_av_consistency")
    )


def _existing_verifier_explicit_failure(row: dict[str, Any]) -> str:
    audio = row.get("audio_only_verification") if isinstance(row.get("audio_only_verification"), dict) else None
    video = row.get("video_only_shortcut") if isinstance(row.get("video_only_shortcut"), dict) else None
    full_av = row.get("full_av_consistency") if isinstance(row.get("full_av_consistency"), dict) else None
    if audio is not None and "accept" in audio and not _truthy(audio.get("accept")):
        return "legacy_audio_only_reject"
    if audio is not None and "reference_satisfies_edit" in audio and _truthy(audio.get("reference_satisfies_edit")):
        return "legacy_reference_satisfies_edit"
    if audio is not None and "target_satisfies_edit" in audio and not _truthy(audio.get("target_satisfies_edit")):
        return "legacy_target_does_not_satisfy_edit"
    if video is not None and (
        _truthy(video.get("can_identify_target_without_audio")) or _truthy(video.get("visual_shortcut_risk"))
    ):
        return "legacy_video_only_shortcut"
    if full_av is not None and "accept" in full_av and not _truthy(full_av.get("accept")):
        return "legacy_full_av_reject"
    if full_av is not None and "audio_edit_still_valid" in full_av and not _truthy(full_av.get("audio_edit_still_valid")):
        return "legacy_audio_edit_invalid_full_av"
    return ""


def _existing_min_stage_confidence(row: dict[str, Any]) -> float:
    values: list[float] = []
    for key in ("audio_only_verification", "video_only_shortcut", "full_av_consistency"):
        payload = row.get(key) if isinstance(row.get(key), dict) else {}
        if "confidence" in payload:
            try:
                values.append(float(payload.get("confidence") or 0.0))
            except (TypeError, ValueError):
                pass
    return min(values) if values else 0.0


def _resolve_media_path(media_root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else media_root / path
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"media file does not exist: {resolved}")
    return resolved


def _automatic_audio_proposal(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("audio_only_proposal", "audio_delta_analysis", "audio_delta"):
        value = row.get(key)
        if isinstance(value, dict) and value:
            return dict(value)
    return {
        "difference_type": "speech" if _canonical_subtype(row) == "speech_topic_in_video_context" else "audio_event",
        "b_subtype": _canonical_subtype(row),
        "reference_audio_content": str(row.get("old_audio") or "").strip(),
        "target_audio_content": str(row.get("new_audio") or "").strip(),
        "edit_text": str(row.get("edit_text") or "").strip(),
    }


def _transcript_like_text(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    patterns = (
        r"\bfrom saying\s+['\"]?.+?['\"]?\s+to saying\s+['\"]?.+?['\"]?(?:\b|$)",
        r"\bchange (?:the )?(?:voice|speech) from saying\b",
        r"\breplace (?:the )?(?:sentence|phrase|words)\b",
        r"\bverbatim\b|\btranscript\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite(item) for item in value)
    if isinstance(value, float):
        return not math.isfinite(value)
    return False


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_named_ints(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected NAME=COUNT, got: {item}")
        name, raw_count = item.split("=", 1)
        canonical = _canonical_subtype_name(name)
        if canonical == "unknown":
            raise ValueError(f"unsupported subtype target: {name}")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"subtype target must be non-negative: {item}")
        result[canonical] = count
    if not result:
        raise ValueError("subtype target mapping must not be empty")
    return result


def _parse_mode_comparison(value: str) -> tuple[str, str]:
    parts = [item.strip() for item in str(value).split(":")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"comparison must be MODE_B:MODE_A, got: {value}")
    return parts[0], parts[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
