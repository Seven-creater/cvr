from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CONDITIONS = ("exact", "transcoded", "temporal", "spatial")
MODES = ("V_T", "V_A_T")
METHODS = ("base_e5", "adapter")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _metric_summary(
    with_rows: Sequence[dict[str, Any]],
    masked_rows: Sequence[dict[str, Any]],
    method: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prefix = "base" if method == "base_e5" else "adapter"
    with_lookup = {str(row["sample_id"]): row for row in with_rows}
    masked_lookup = {str(row["sample_id"]): row for row in masked_rows}
    sample_ids = sorted(with_lookup)
    if sample_ids != sorted(masked_lookup):
        raise ValueError(f"with/masked sample IDs differ for method={method}")
    with_values = [with_lookup[sample_id] for sample_id in sample_ids]
    masked_values = [masked_lookup[sample_id] for sample_id in sample_ids]
    with_rank = np.asarray(
        [row[f"{prefix}_target_rank"] for row in with_values], dtype=np.int64
    )
    masked_rank = np.asarray(
        [row[f"{prefix}_target_rank"] for row in masked_values], dtype=np.int64
    )
    target_score = np.asarray(
        [row[f"{prefix}_target_score"] for row in with_values], dtype=np.float64
    )
    reference_score = np.asarray(
        [row[f"{prefix}_reference_score"] for row in with_values], dtype=np.float64
    )
    reference_rank = np.asarray(
        [
            row.get(f"{prefix}_reference_rank")
            if row.get(f"{prefix}_reference_rank") is not None
            else (
                1
                if bool((row.get(f"{prefix}_top1") or {}).get("is_reference"))
                else np.nan
            )
            for row in with_values
        ],
        dtype=np.float64,
    )
    arrays = {
        "with_correct": (with_rank == 1).astype(np.float64),
        "masked_correct": (masked_rank == 1).astype(np.float64),
        "target_beats_reference": (target_score > reference_score).astype(np.float64),
        "target_reference_gap": target_score - reference_score,
        "with_rank": with_rank.astype(np.float64),
        "masked_rank": masked_rank.astype(np.float64),
        "reference_rank": reference_rank,
    }
    summary = {
        "query_count": len(sample_ids),
        "with_reference": {
            "R@1": float(np.mean(with_rank <= 1)),
            "R@5": float(np.mean(with_rank <= 5)),
            "R@10": float(np.mean(with_rank <= 10)),
            "MRR": float(np.mean(1.0 / with_rank)),
            "mean_rank": float(np.mean(with_rank)),
            "median_rank": float(np.median(with_rank)),
            "target_beats_reference": float(np.mean(arrays["target_beats_reference"])),
            "target_reference_gap": float(np.mean(arrays["target_reference_gap"])),
            "reference_rank_mean": (
                float(np.nanmean(reference_rank))
                if np.isfinite(reference_rank).any()
                else None
            ),
            "top1_reference_rate": float(
                np.mean(
                    [
                        bool((row.get(f"{prefix}_top1") or {}).get("is_reference"))
                        for row in with_values
                    ]
                )
            ),
        },
        "masked_reference": {
            "R@1": float(np.mean(masked_rank <= 1)),
            "R@5": float(np.mean(masked_rank <= 5)),
            "R@10": float(np.mean(masked_rank <= 10)),
            "MRR": float(np.mean(1.0 / masked_rank)),
            "mean_rank": float(np.mean(masked_rank)),
            "median_rank": float(np.median(masked_rank)),
        },
        "reference_induced_R@1_drop": float(
            np.mean(arrays["masked_correct"] - arrays["with_correct"])
        ),
    }
    arrays["sample_ids"] = np.asarray(sample_ids)
    return summary, arrays


def _paired_test(
    first: np.ndarray,
    second: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    if difference.ndim != 1 or len(difference) == 0:
        raise ValueError("paired comparison requires a non-empty vector")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    randomized = np.empty(iterations, dtype=np.float64)
    chunk = 500
    for start in range(0, iterations, chunk):
        size = min(chunk, iterations - start)
        indices = rng.integers(0, len(difference), size=(size, len(difference)))
        bootstrap[start : start + size] = difference[indices].mean(axis=1)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(difference)))
        randomized[start : start + size] = (difference[None, :] * signs).mean(axis=1)
    observed = float(difference.mean())
    return {
        "mean_difference": observed,
        "bootstrap_95_ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "paired_randomization_p_two_sided": float(
            (np.sum(np.abs(randomized) >= abs(observed)) + 1) / (iterations + 1)
        ),
        "iterations": iterations,
    }


def _mcnemar(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(~first & second))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(first_only, second_only)
        tail = sum(
            math.comb(discordant, value) for value in range(lower + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "p_two_sided": float(p_value),
    }


def _holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, values[name] * (total - rank)))
        adjusted[name] = running
    return adjusted


def summarize_e5(
    *,
    evaluation_root: Path,
    output_dir: Path,
    seeds: Sequence[int],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    summaries: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    per_query: dict[
        tuple[str, str, str, int], dict[str, np.ndarray]
    ] = {}
    output_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for mode in MODES:
            for run_seed in seeds:
                root = evaluation_root / f"seed_{run_seed}" / condition / mode
                with_rows = _load_jsonl(
                    root / "with_reference" / "per_query_scores.jsonl"
                )
                masked_rows = _load_jsonl(
                    root / "masked_reference" / "per_query_scores.jsonl"
                )
                for method in METHODS:
                    summary, arrays = _metric_summary(with_rows, masked_rows, method)
                    summaries[method][condition].setdefault(mode, {})[
                        str(run_seed)
                    ] = summary
                    per_query[(method, condition, mode, run_seed)] = arrays
                    for index, sample_id in enumerate(arrays["sample_ids"]):
                        output_rows.append(
                            {
                                "model": method,
                                "condition": condition,
                                "mode": mode,
                                "seed": run_seed,
                                "sample_id": str(sample_id),
                                "with_correct_at_1": bool(
                                    arrays["with_correct"][index]
                                ),
                                "masked_correct_at_1": bool(
                                    arrays["masked_correct"][index]
                                ),
                                "target_beats_reference": bool(
                                    arrays["target_beats_reference"][index]
                                ),
                                "target_reference_gap": float(
                                    arrays["target_reference_gap"][index]
                                ),
                                "reference_rank": (
                                    float(arrays["reference_rank"][index])
                                    if np.isfinite(arrays["reference_rank"][index])
                                    else None
                                ),
                            }
                        )

    mean_std: dict[str, Any] = {}
    for method in METHODS:
        mean_std[method] = {}
        for condition in CONDITIONS:
            mean_std[method][condition] = {}
            for mode in MODES:
                seed_rows = summaries[method][condition][mode]
                aggregate: dict[str, Any] = {}
                for section, metrics in (
                    ("with_reference", ("R@1", "R@5", "R@10", "MRR", "target_beats_reference", "target_reference_gap", "top1_reference_rate")),
                    ("masked_reference", ("R@1", "R@5", "R@10", "MRR")),
                ):
                    aggregate[section] = {}
                    for metric in metrics:
                        values = np.asarray(
                            [seed_rows[str(item)][section][metric] for item in seeds],
                            dtype=np.float64,
                        )
                        aggregate[section][metric] = {
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=0)),
                        }
                values = np.asarray(
                    [
                        seed_rows[str(item)]["reference_induced_R@1_drop"]
                        for item in seeds
                    ],
                    dtype=np.float64,
                )
                aggregate["reference_induced_R@1_drop"] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
                mean_std[method][condition][mode] = aggregate

    comparisons: dict[str, Any] = {}
    for method in METHODS:
        method_seeds = (seeds[0],) if method == "base_e5" else seeds

        def averaged(condition: str, mode: str, field: str) -> np.ndarray:
            values = [
                per_query[(method, condition, mode, run_seed)][field]
                for run_seed in method_seeds
            ]
            return np.stack(values).mean(axis=0)

        for condition in CONDITIONS:
            for mode in MODES:
                with_correct = averaged(condition, mode, "with_correct")
                masked_correct = averaged(condition, mode, "masked_correct")
                name = f"{method}_{condition}_{mode}_reference_drop"
                comparisons[name] = _paired_test(
                    masked_correct,
                    with_correct,
                    iterations=iterations,
                    seed=seed + len(comparisons),
                )
                if method == "base_e5":
                    comparisons[name]["mcnemar"] = _mcnemar(
                        masked_correct, with_correct
                    )
            vat = averaged(condition, "V_A_T", "with_correct")
            vt = averaged(condition, "V_T", "with_correct")
            name = f"{method}_{condition}_audio_gain_R@1"
            comparisons[name] = _paired_test(
                vat, vt, iterations=iterations, seed=seed + len(comparisons)
            )
        for condition in CONDITIONS[1:]:
            for mode in MODES:
                perturbed = averaged(condition, mode, "with_correct")
                exact = averaged("exact", mode, "with_correct")
                name = f"{method}_{condition}_{mode}_R@1_vs_exact"
                comparisons[name] = _paired_test(
                    perturbed,
                    exact,
                    iterations=iterations,
                    seed=seed + len(comparisons),
                )

    adjusted = _holm(
        {
            name: payload["paired_randomization_p_two_sided"]
            for name, payload in comparisons.items()
        }
    )
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "e5_reference_ladder_mean_std.json", mean_std)
    _atomic_json(output_dir / "e5_paired_comparisons.json", comparisons)
    _atomic_jsonl(output_dir / "e5_reference_ladder_per_query.jsonl", output_rows)
    summary = {
        "model_family": "E5-Omni",
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "query_count": 1000,
        "selection_uses_test_metrics": False,
        "score_matrix_reused_for_masking": True,
    }
    _atomic_json(output_dir / "e5_reference_ladder_summary.json", summary)
    return summary


def summarize_imagebind(
    *,
    exact_evaluation: Path,
    variant_evaluation_root: Path,
    output_dir: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    output_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        root = (
            exact_evaluation
            if condition == "exact"
            else variant_evaluation_root / condition
        )
        rows = _load_jsonl(root / "per_query_results.jsonl")
        by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if str(row["mode"]) in MODES:
                by_mode[str(row["mode"])].append(row)
        by_condition[condition] = {}
        for mode in MODES:
            ordered = sorted(by_mode[mode], key=lambda row: str(row["sample_id"]))
            sample_ids = np.asarray([str(row["sample_id"]) for row in ordered])
            with_rank = np.asarray(
                [row["with_reference_rank"] for row in ordered], dtype=np.int64
            )
            masked_rank = np.asarray(
                [row["without_reference_rank"] for row in ordered], dtype=np.int64
            )
            values = {
                "sample_ids": sample_ids,
                "with_correct": (with_rank == 1).astype(np.float64),
                "masked_correct": (masked_rank == 1).astype(np.float64),
                "target_beats_reference": np.asarray(
                    [row["target_beats_reference"] for row in ordered],
                    dtype=np.float64,
                ),
                "target_reference_gap": np.asarray(
                    [row["target_reference_gap"] for row in ordered],
                    dtype=np.float64,
                ),
                "reference_rank": np.asarray(
                    [row["reference_rank"] for row in ordered], dtype=np.float64
                ),
            }
            arrays[(condition, mode)] = values
            by_condition[condition][mode] = {
                "query_count": len(ordered),
                "with_reference": {
                    "R@1": float(np.mean(with_rank <= 1)),
                    "R@5": float(np.mean(with_rank <= 5)),
                    "R@10": float(np.mean(with_rank <= 10)),
                    "MRR": float(np.mean(1.0 / with_rank)),
                    "target_beats_reference": float(
                        values["target_beats_reference"].mean()
                    ),
                    "target_reference_gap": float(
                        values["target_reference_gap"].mean()
                    ),
                    "reference_rank_mean": float(values["reference_rank"].mean()),
                },
                "masked_reference": {
                    "R@1": float(np.mean(masked_rank <= 1)),
                    "R@5": float(np.mean(masked_rank <= 5)),
                    "R@10": float(np.mean(masked_rank <= 10)),
                    "MRR": float(np.mean(1.0 / masked_rank)),
                },
                "reference_induced_R@1_drop": float(
                    np.mean(values["masked_correct"] - values["with_correct"])
                ),
            }
            for index, sample_id in enumerate(sample_ids):
                output_rows.append(
                    {
                        "model": "ImageBind-Huge",
                        "condition": condition,
                        "mode": mode,
                        "sample_id": str(sample_id),
                        "with_correct_at_1": bool(values["with_correct"][index]),
                        "masked_correct_at_1": bool(values["masked_correct"][index]),
                        "target_beats_reference": bool(
                            values["target_beats_reference"][index]
                        ),
                        "target_reference_gap": float(
                            values["target_reference_gap"][index]
                        ),
                        "reference_rank": float(values["reference_rank"][index]),
                    }
                )
    comparisons: dict[str, Any] = {}
    for condition in CONDITIONS:
        for mode in MODES:
            value = arrays[(condition, mode)]
            name = f"ImageBind_{condition}_{mode}_reference_drop"
            comparisons[name] = _paired_test(
                value["masked_correct"],
                value["with_correct"],
                iterations=iterations,
                seed=seed + len(comparisons),
            )
            comparisons[name]["mcnemar"] = _mcnemar(
                value["masked_correct"], value["with_correct"]
            )
        name = f"ImageBind_{condition}_audio_gain_R@1"
        comparisons[name] = _paired_test(
            arrays[(condition, "V_A_T")]["with_correct"],
            arrays[(condition, "V_T")]["with_correct"],
            iterations=iterations,
            seed=seed + len(comparisons),
        )
    for condition in CONDITIONS[1:]:
        for mode in MODES:
            name = f"ImageBind_{condition}_{mode}_R@1_vs_exact"
            comparisons[name] = _paired_test(
                arrays[(condition, mode)]["with_correct"],
                arrays[("exact", mode)]["with_correct"],
                iterations=iterations,
                seed=seed + len(comparisons),
            )
    adjusted = _holm(
        {
            name: payload["paired_randomization_p_two_sided"]
            for name, payload in comparisons.items()
        }
    )
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "imagebind_reference_ladder.json", by_condition)
    _atomic_json(output_dir / "imagebind_paired_comparisons.json", comparisons)
    _atomic_jsonl(
        output_dir / "imagebind_reference_ladder_per_query.jsonl", output_rows
    )
    summary = {
        "model": "ImageBind-Huge",
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "query_count": 1000,
        "selection_uses_test_metrics": False,
        "score_matrix_reused_for_masking": True,
    }
    _atomic_json(output_dir / "imagebind_reference_ladder_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio-CVR source-identity ladder")
    commands = parser.add_subparsers(dest="command", required=True)
    e5 = commands.add_parser("summarize-e5")
    e5.add_argument("--evaluation-root", required=True)
    e5.add_argument("--output-dir", required=True)
    e5.add_argument("--seeds", default="13,23,42,71,101")
    e5.add_argument("--iterations", type=int, default=20000)
    e5.add_argument("--seed", type=int, default=20260724)
    imagebind = commands.add_parser("summarize-imagebind")
    imagebind.add_argument("--exact-evaluation", required=True)
    imagebind.add_argument("--variant-evaluation-root", required=True)
    imagebind.add_argument("--output-dir", required=True)
    imagebind.add_argument("--iterations", type=int, default=20000)
    imagebind.add_argument("--seed", type=int, default=20260724)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "summarize-e5":
        value = summarize_e5(
            evaluation_root=Path(args.evaluation_root),
            output_dir=Path(args.output_dir),
            seeds=[int(value) for value in args.seeds.split(",") if value],
            iterations=args.iterations,
            seed=args.seed,
        )
    elif args.command == "summarize-imagebind":
        value = summarize_imagebind(
            exact_evaluation=Path(args.exact_evaluation),
            variant_evaluation_root=Path(args.variant_evaluation_root),
            output_dir=Path(args.output_dir),
            iterations=args.iterations,
            seed=args.seed,
        )
    else:
        raise ValueError(args.command)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
