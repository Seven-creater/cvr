from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    selected = rows[0]
    summary = {
        "selection_split": "validation_only",
        "selection_rule": "max mean R@1; then max target_beats_reference; then min R@1 std; then fewer steps",
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
    _write_config_tsv(output_root / "selected_config.tsv", rows[:1])
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
        "primary_mode_subtypes": _subtype_result_summary(
            [runs[(seed, primary_mode)]["summary"] for seed in seeds]
        ),
        "hard_negative_summary": hard_negative,
        "audit_path": str(output_root / "audit.json"),
    }
    _write_json(output_root / "audit.json", audit)
    _write_json(output_root / "per_seed_results.json", {"rows": per_seed})
    _write_json(output_root / "test_main_mean_std.json", summary)
    _write_json(output_root / "error_breakdown.json", error_breakdown)
    (output_root / "test_main_comparison.md").write_text(_final_comparison_markdown(summary), encoding="utf-8")
    (output_root / "audio_gain_summary.md").write_text(_audio_gain_markdown(summary), encoding="utf-8")
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

    validation = subparsers.add_parser("summarize-validation")
    validation.add_argument("--input-root", action="append", required=True)
    validation.add_argument("--output-dir", required=True)
    validation.add_argument("--required-seeds", default="13")
    validation.add_argument("--top-n", type=int, default=6)

    final = subparsers.add_parser("aggregate-final")
    final.add_argument("--input-root", required=True)
    final.add_argument("--output-dir", required=True)
    final.add_argument("--required-seeds", default="13,23,42,71,101")
    final.add_argument("--primary-mode", default=PRIMARY_MODE)
    final.add_argument("--reference-mode", default=REFERENCE_MODE)
    final.add_argument("--bootstrap-samples", type=int, default=20_000)
    final.add_argument("--permutation-samples", type=int, default=20_000)
    final.add_argument("--random-seed", type=int, default=20260718)

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
    elif args.command == "summarize-validation":
        result = summarize_validation(
            input_roots=args.input_root,
            output_dir=args.output_dir,
            required_seeds=_parse_ints(args.required_seeds),
            top_n=args.top_n,
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


def _human_review_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": _first_text(row, ("sample_id", "proposal_id", "clip_id")),
        "reference_video": row.get("reference_video"),
        "target_video": row.get("target_video"),
        "edit_text": row.get("edit_text"),
        "b_subtype": row.get("b_subtype") or row.get("audio_delta_type"),
        "hard_negatives": row.get("audio_delta_hard_negatives") or row.get("hard_negatives") or [],
        "review": {
            "edit_audio_only": None,
            "reference_does_not_satisfy_edit": None,
            "target_satisfies_edit": None,
            "video_only_cannot_identify_target": None,
            "hard_negatives_do_not_satisfy_edit": None,
            "decision": "unreviewed",
            "notes": "",
        },
    }


def _validation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Validation-only Model Selection",
        "",
        f"Selection rule: {summary['selection_rule']}",
        "",
        "| Rank | Steps | LR | Batch | Seeds | R@1 mean | R@1 std | Beats-ref mean |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['validation_rank']} | {row['steps']} | {row['learning_rate']:.6g} | {row['batch_size']} | "
            f"{','.join(str(seed) for seed in row['seeds'])} | {row['R@1_mean']:.4f} | {row['R@1_std']:.4f} | "
            f"{row['target_beats_reference_mean']:.4f} |"
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
