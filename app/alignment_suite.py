from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.artifacts import append_jsonl, ensure_runs_dir, write_run_artifacts
from app.backends import FileRetrievalBackend
from app.backends.base import load_json
from app.backends.file_backend import _normalize_retrieval_scores
from app.config import load_yaml
from app.controller import ScriptedController, resolve_scripted_policy
from app.demo import resolve_config_path
from app.eval import compute_recall_at_k, error_labels, parse_recall_ks, query_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict alignment suite for replay evaluation")
    parser.add_argument("--config", required=True, help="real.yaml config path")
    parser.add_argument(
        "--profiles",
        default="single-round-fixed,fixed,adaptive",
        help="Comma-separated scripted controller profiles",
    )
    parser.add_argument("--fixed-video-weight", type=float, default=0.7)
    parser.add_argument("--fixed-audio-weight", type=float, default=0.3)
    parser.add_argument("--fixed-object-focus", default="none")
    parser.add_argument("--fixed-temporal-focus", default="global")
    parser.add_argument("--fixed-topk", type=int, default=5)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--write-per-query-artifacts", action="store_true")
    parser.add_argument("--recall-k", default="1,3,5")
    parser.add_argument("--output-dir", help="Optional output directory for batch files and reports")
    return parser.parse_args()


def resolve_backend_from_config(config_path: str) -> tuple[FileRetrievalBackend, dict[str, Any]]:
    config = load_yaml(config_path)
    backend = FileRetrievalBackend(
        candidates_path=resolve_config_path(config_path, config["candidates_path"]),
        queries_path=resolve_config_path(config_path, config["queries_path"]),
        retrieval_scores_path=resolve_config_path(config_path, config.get("retrieval_scores_path")),
    )
    return backend, config


def load_pack_stats(config_path: str) -> dict[str, Any]:
    stats_path = Path(config_path).resolve().parent / "stats.json"
    if not stats_path.exists():
        return {}
    payload = load_json(stats_path)
    return payload if isinstance(payload, dict) else {}


def build_data_audit(backend: FileRetrievalBackend, retrieval_scores_path: str | None, pack_stats: dict[str, Any]) -> dict[str, Any]:
    normalized_scores = (
        _normalize_retrieval_scores(load_json(retrieval_scores_path))
        if retrieval_scores_path
        else {}
    )
    queries = backend.list_queries()
    expected_scored_candidates = max(0, len(backend._candidates) - 1)  # noqa: SLF001
    score_coverages = [len(items) for items in normalized_scores.values()]
    warnings: list[str] = []
    if pack_stats.get("uses_target_aware_filtering") is True:
        warnings.append("pack uses target-aware query filtering")
    if pack_stats.get("uses_rollout_filtering") is True:
        warnings.append("pack selection depends on scripted rollout simulation")
    if pack_stats.get("uses_heuristic_retrieval_scores") is True:
        warnings.append("retrieval scores are heuristic, not external frozen model outputs")

    return {
        "query_count": len(queries),
        "candidate_count": len(backend._candidates),  # noqa: SLF001
        "score_query_count": len(normalized_scores),
        "expected_scored_candidates_per_query": expected_scored_candidates,
        "score_coverage_min": min(score_coverages) if score_coverages else 0,
        "score_coverage_max": max(score_coverages) if score_coverages else 0,
        "score_coverage_avg": round(sum(score_coverages) / max(1, len(score_coverages)), 2) if score_coverages else 0.0,
        "full_coverage_queries": sum(1 for coverage in score_coverages if coverage == expected_scored_candidates),
        "uses_target_aware_filtering": pack_stats.get("uses_target_aware_filtering"),
        "uses_rollout_filtering": pack_stats.get("uses_rollout_filtering"),
        "uses_heuristic_retrieval_scores": pack_stats.get("uses_heuristic_retrieval_scores"),
        "filter_mode": pack_stats.get("filter_mode"),
        "query_selection_mode": pack_stats.get("query_selection_mode"),
        "selection_strategy": pack_stats.get("selection_strategy"),
        "warnings": warnings,
    }


def summarize_rows(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    avg_rounds = sum(len(row.get("rounds", [])) for row in rows) / max(1, total)
    avg_tool_calls = sum(len(row.get("tool_history", [])) for row in rows) / max(1, total)
    first_round_recall, any_round_recall = compute_recall_at_k(rows, ks)

    type_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        type_buckets[query_type(row)].append(row)

    type_summary = {
        bucket: {
            "count": len(items),
            "success_rate": round(
                sum(1 for item in items if item.get("success") is True) / max(1, len(items)),
                3,
            ),
        }
        for bucket, items in sorted(type_buckets.items())
    }

    failed_cases = []
    error_counter: Counter[str] = Counter()
    for row in rows:
        labels = error_labels(row)
        error_counter.update(labels)
        if labels:
            failed_cases.append(
                {
                    "query_id": row.get("query", {}).get("query_id"),
                    "type": query_type(row),
                    "target": row.get("query", {}).get("target_video_id"),
                    "final": row.get("final_candidate_id"),
                    "errors": labels,
                }
            )

    return {
        "runs": total,
        "success_rate": round(successes / max(1, total), 3),
        "avg_rounds": round(avg_rounds, 2),
        "avg_tool_calls": round(avg_tool_calls, 2),
        "first_round_top1": round(first_round_recall.get(1, 0.0), 3),
        "first_round_recall": {f"R@{k}": round(v, 3) for k, v in first_round_recall.items()},
        "any_round_recall": {f"R@{k}": round(v, 3) for k, v in any_round_recall.items()},
        "type_breakdown": type_summary,
        "error_breakdown": dict(error_counter),
        "failed_cases": failed_cases,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Strict Alignment Suite",
        "",
        "## Data Audit",
        "",
    ]
    audit = summary["data_audit"]
    for key in (
        "query_count",
        "candidate_count",
        "score_query_count",
        "expected_scored_candidates_per_query",
        "score_coverage_min",
        "score_coverage_max",
        "score_coverage_avg",
        "full_coverage_queries",
        "filter_mode",
        "query_selection_mode",
        "selection_strategy",
        "uses_target_aware_filtering",
        "uses_rollout_filtering",
        "uses_heuristic_retrieval_scores",
    ):
        lines.append(f"- {key}: `{audit.get(key)}`")
    lines.extend(["", "### Warnings", ""])
    warnings = audit.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    lines.extend(["", "## Profile Results", ""])
    for profile in summary["profiles"]:
        metrics = profile["metrics"]
        lines.extend(
            [
                f"### {profile['profile']}",
                "",
                f"- planner: `{profile['planner_name']}`",
                f"- planner_metadata: `{profile['planner_metadata']}`",
                f"- runs: `{metrics['runs']}`",
                f"- success_rate: `{metrics['success_rate']}`",
                f"- avg_rounds: `{metrics['avg_rounds']}`",
                f"- avg_tool_calls: `{metrics['avg_tool_calls']}`",
                f"- first_round_top1: `{metrics['first_round_top1']}`",
                f"- first_round_recall: `{metrics['first_round_recall']}`",
                f"- any_round_recall: `{metrics['any_round_recall']}`",
                f"- type_breakdown: `{metrics['type_breakdown']}`",
                f"- error_breakdown: `{metrics['error_breakdown'] or 'none'}`",
                f"- batch_jsonl: `{profile['batch_jsonl']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = str(Path(args.config).resolve())
    backend, config = resolve_backend_from_config(config_path)
    pack_stats = load_pack_stats(config_path)
    retrieval_scores_path = resolve_config_path(config_path, config.get("retrieval_scores_path"))
    data_audit = build_data_audit(backend, retrieval_scores_path, pack_stats)

    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    queries = backend.list_queries()
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ensure_runs_dir() / f"alignment-suite-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ks = parse_recall_ks(args.recall_k)

    profile_summaries: list[dict[str, Any]] = []
    for profile in profiles:
        controller = ScriptedController(
            backend=backend,
            policy=resolve_scripted_policy(
                profile=profile,
                fixed_video_weight=args.fixed_video_weight,
                fixed_audio_weight=args.fixed_audio_weight,
                fixed_object_focus=args.fixed_object_focus,
                fixed_temporal_focus=args.fixed_temporal_focus,
                fixed_topk=args.fixed_topk,
            ),
        )

        traces = []
        failures: list[dict[str, str]] = []
        for query in queries:
            try:
                trace = controller.run(query.query_id)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                failures.append({"query_id": query.query_id, "error": str(exc)})
                continue
            trace.query.target_video_id = query.target_video_id
            trace.success = (
                query.target_video_id == trace.final_candidate_id
                if query.target_video_id
                else None
            )
            traces.append(trace)
            if args.write_per_query_artifacts:
                write_run_artifacts(trace, prefix=f"{profile}-{query.query_id}")

        jsonl_path = append_jsonl(traces, path=output_dir / f"{profile}.jsonl")
        rows = [trace.to_dict() for trace in traces]
        metrics = summarize_rows(rows, ks)
        profile_summaries.append(
            {
                "profile": profile,
                "planner_name": controller.name,
                "planner_metadata": controller.policy.to_trace_metadata(),
                "batch_jsonl": str(jsonl_path),
                "metrics": metrics,
                "runtime_failures": failures,
            }
        )

    summary = {
        "config_path": config_path,
        "data_audit": data_audit,
        "profiles": profile_summaries,
    }

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
