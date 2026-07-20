from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
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


def prepare_paper_splits(*, split_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_root = Path(split_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train = _read_jsonl(source_root / "train.jsonl")
    val = _read_jsonl(source_root / "val.jsonl")
    test_all = _read_jsonl(source_root / "test_main.jsonl")
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

    freeze = subparsers.add_parser("finalize-benchmark")
    freeze.add_argument("--candidate-path", required=True)
    freeze.add_argument("--review-path", action="append", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.add_argument("--exclude-path", action="append", default=[])
    freeze.add_argument("--target-count", type=int, default=150)
    freeze.add_argument("--minimum-count", type=int, default=100)
    freeze.add_argument("--max-speech-ratio", type=float, default=0.35)
    freeze.add_argument("--max-dataset-ratio", type=float, default=0.60)
    freeze.add_argument("--min-strict-local-coverage", type=float, default=0.0)
    freeze.add_argument("--random-seed", type=int, default=20260719)
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
    elif args.command == "finalize-benchmark":
        result = finalize_benchmark(
            candidate_path=args.candidate_path,
            review_paths=args.review_path,
            output_dir=args.output_dir,
            exclude_paths=args.exclude_path,
            target_count=args.target_count,
            minimum_count=args.minimum_count,
            max_speech_ratio=args.max_speech_ratio,
            max_dataset_ratio=args.max_dataset_ratio,
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
    random_seed: int,
) -> list[dict[str, Any]]:
    speech_limit = max(0, math.floor(target_count * max_speech_ratio))
    dataset_limit = max(1, math.floor(target_count * max_dataset_ratio))
    remaining = sorted(rows, key=lambda row: _stable_row_key(row, random_seed))
    selected: list[dict[str, Any]] = []
    dataset_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    speech_count = 0
    while remaining and len(selected) < target_count:
        eligible = [
            row
            for row in remaining
            if dataset_counts[_dataset(row)] < dataset_limit
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


def _parse_mode_comparison(value: str) -> tuple[str, str]:
    parts = [item.strip() for item in str(value).split(":")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"comparison must be MODE_B:MODE_A, got: {value}")
    return parts[0], parts[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
