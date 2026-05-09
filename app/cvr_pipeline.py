from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.avigate_agent import run_cvr_agent_case
from app.avigate_official import retrieve_videos_from_text_official
from app.cvr_fusion import FusedHit, fuse_video_hits, fused_hits_to_retrieval_hits
from app.cvr_query_builder import CVRTriplet, build_cvr_query_views
from app.e5_omni_index import E5TargetIndex, retrieve_e5_videos
from app.e5_omni_runtime import E5OmniRuntime
from app.omni_checker import OmniChecker
from app.retrieval_types import RetrievalHit


def run_e5_only_eval(
    *,
    e5_runtime: E5OmniRuntime,
    e5_index: E5TargetIndex,
    triplets: list[CVRTriplet],
    recall_ks: tuple[int, ...],
    topk: int,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    recall_ks = _normalize_recall_ks(recall_ks)
    hit_counts = {k: 0 for k in recall_ks}
    trace_lines: list[str] = []

    for run_index, triplet in enumerate(triplets, start=1):
        _emit(progress, f"[e5] start {run_index}/{len(triplets)}: {triplet.sample_id}")
        views = build_cvr_query_views(triplet)
        query_embedding = e5_runtime.encode_video_text_query(
            video_path=views.reference_video,
            text=views.e5_text_query,
        )
        hits = retrieve_e5_videos(query_embedding=query_embedding, index=e5_index, topk=max(max(recall_ks), topk))
        target_rank = _rank_of_target(hits, triplet.sample_id)
        for k in recall_ks:
            if target_rank is not None and target_rank <= k:
                hit_counts[k] += 1
        trace_lines.append(
            json.dumps(
                {
                    "mode": "cvr-e5-only",
                    "sample_id": triplet.sample_id,
                    "target_video_id": triplet.sample_id,
                    "target_rank": target_rank,
                    "query_views": asdict(views),
                    "topk_hits": [hit.to_dict() for hit in hits[:topk]],
                },
                ensure_ascii=False,
            )
        )

    summary = _summary_from_counts(
        mode="cvr-e5-only",
        runs=len(triplets),
        recall_ks=recall_ks,
        hit_counts=hit_counts,
    )
    _write_json(output_root / "summary.json", summary)
    _write_text(output_root / "traces.jsonl", "\n".join(trace_lines) + ("\n" if trace_lines else ""))
    return summary


def run_cvr_fusion_eval(
    *,
    avigate_runtime: Any,
    e5_runtime: E5OmniRuntime,
    e5_index: E5TargetIndex,
    triplets: list[CVRTriplet],
    recall_ks: tuple[int, ...],
    avigate_topk: int,
    e5_topk: int,
    fused_topk: int,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    recall_ks = _normalize_recall_ks(recall_ks)
    avigate_counts = {k: 0 for k in recall_ks}
    e5_counts = {k: 0 for k in recall_ks}
    fused_counts = {k: 0 for k in recall_ks}
    trace_lines: list[str] = []

    for run_index, triplet in enumerate(triplets, start=1):
        _emit(progress, f"[cvr-fusion] start {run_index}/{len(triplets)}: {triplet.sample_id}")
        views = build_cvr_query_views(triplet)
        avigate_hits, e5_hits, fused_hits = retrieve_cvr_candidates(
            avigate_runtime=avigate_runtime,
            e5_runtime=e5_runtime,
            e5_index=e5_index,
            triplet=triplet,
            avigate_topk=max(max(recall_ks), avigate_topk),
            e5_topk=max(max(recall_ks), e5_topk),
            fused_topk=max(max(recall_ks), fused_topk),
        )
        _update_counts(avigate_counts, _rank_of_target(avigate_hits, triplet.sample_id), recall_ks)
        _update_counts(e5_counts, _rank_of_target(e5_hits, triplet.sample_id), recall_ks)
        _update_counts(fused_counts, _rank_of_target_fused(fused_hits, triplet.sample_id), recall_ks)
        trace_lines.append(
            json.dumps(
                {
                    "mode": "cvr-fusion",
                    "sample_id": triplet.sample_id,
                    "target_video_id": triplet.sample_id,
                    "query_views": asdict(views),
                    "avigate_hits": [hit.to_dict() for hit in avigate_hits[:avigate_topk]],
                    "e5_hits": [hit.to_dict() for hit in e5_hits[:e5_topk]],
                    "fused_hits": [hit.to_dict() for hit in fused_hits[:fused_topk]],
                },
                ensure_ascii=False,
            )
        )

    summary = {
        "mode": "cvr-fusion",
        "runs": len(triplets),
        "avigate_recall": _metrics_from_counts(avigate_counts, len(triplets), recall_ks),
        "e5_recall": _metrics_from_counts(e5_counts, len(triplets), recall_ks),
        "fused_recall": _metrics_from_counts(fused_counts, len(triplets), recall_ks),
        "avigate_topk": int(avigate_topk),
        "e5_topk": int(e5_topk),
        "fused_topk": int(fused_topk),
    }
    _write_json(output_root / "summary.json", summary)
    _write_text(output_root / "traces.jsonl", "\n".join(trace_lines) + ("\n" if trace_lines else ""))
    return summary


def run_cvr_agent_eval(
    *,
    avigate_runtime: Any,
    e5_runtime: E5OmniRuntime,
    e5_index: E5TargetIndex,
    checker: OmniChecker,
    triplets: list[CVRTriplet],
    recall_ks: tuple[int, ...],
    avigate_topk: int,
    e5_topk: int,
    fused_topk: int,
    rerank_window: int,
    omni_concurrency: int,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    traces_path = output_root / "traces.jsonl"
    traces_path.write_text("", encoding="utf-8")
    recall_ks = _normalize_recall_ks(recall_ks)
    source_counts = {
        "avigate_recall": {k: 0 for k in recall_ks},
        "e5_recall": {k: 0 for k in recall_ks},
        "fused_recall": {k: 0 for k in recall_ks},
        "final_recall": {k: 0 for k in recall_ks},
    }
    total_omni_calls = 0
    fallback_runs = 0
    final_top1_correct = 0

    summary: dict[str, Any] = {}
    for run_index, triplet in enumerate(triplets, start=1):
        _emit(progress, f"[cvr-agent] start {run_index}/{len(triplets)}: {triplet.sample_id}")
        avigate_hits, e5_hits, fused_hits = retrieve_cvr_candidates(
            avigate_runtime=avigate_runtime,
            e5_runtime=e5_runtime,
            e5_index=e5_index,
            triplet=triplet,
            avigate_topk=avigate_topk,
            e5_topk=e5_topk,
            fused_topk=fused_topk,
        )
        fused_retrieval_hits = fused_hits_to_retrieval_hits(fused_hits)
        trace = run_cvr_agent_case(
            sample_id=triplet.sample_id,
            query_text=build_cvr_query_views(triplet).avigate_text_query,
            reference_video_path=triplet.reference_video,
            edit_text=triplet.edit_text,
            reference_caption=triplet.reference_caption,
            runtime=avigate_runtime,
            checker=checker,
            target_video_id=triplet.sample_id,
            avigate_hits=avigate_hits,
            e5_hits=e5_hits,
            fused_hits=fused_retrieval_hits,
            fused_evidence=[hit.to_dict() for hit in fused_hits],
            omni_concurrency=omni_concurrency,
            rerank_window=rerank_window,
            progress=progress,
        )
        _update_counts(source_counts["avigate_recall"], _rank_of_target(avigate_hits, triplet.sample_id), recall_ks)
        _update_counts(source_counts["e5_recall"], _rank_of_target(e5_hits, triplet.sample_id), recall_ks)
        _update_counts(source_counts["fused_recall"], _rank_of_target(fused_retrieval_hits, triplet.sample_id), recall_ks)
        _update_counts(source_counts["final_recall"], _rank_of_dict_target(trace["reranked_hits"], triplet.sample_id), recall_ks)
        if trace["final_result"].get("video_id") == triplet.sample_id:
            final_top1_correct += 1
        total_omni_calls += int(trace["omni_calls"])
        if trace.get("fallback_used"):
            fallback_runs += 1
        with traces_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

        summary = _agent_summary(
            runs=run_index,
            recall_ks=recall_ks,
            source_counts=source_counts,
            final_top1_correct=final_top1_correct,
            total_omni_calls=total_omni_calls,
            fallback_runs=fallback_runs,
            avigate_topk=avigate_topk,
            e5_topk=e5_topk,
            fused_topk=fused_topk,
            rerank_window=rerank_window,
        )
        _write_json(output_root / "summary.json", summary)

    return summary


def retrieve_cvr_candidates(
    *,
    avigate_runtime: Any,
    e5_runtime: E5OmniRuntime,
    e5_index: E5TargetIndex,
    triplet: CVRTriplet,
    avigate_topk: int,
    e5_topk: int,
    fused_topk: int,
) -> tuple[list[RetrievalHit], list[RetrievalHit], list[FusedHit]]:
    views = build_cvr_query_views(triplet)
    avigate_hits = retrieve_videos_from_text_official(views.avigate_text_query, avigate_runtime, topk=avigate_topk)
    query_embedding = e5_runtime.encode_video_text_query(
        video_path=views.reference_video,
        text=views.e5_text_query,
    )
    e5_hits = retrieve_e5_videos(query_embedding=query_embedding, index=e5_index, topk=e5_topk)
    fused_hits = fuse_video_hits(avigate_hits=avigate_hits, e5_hits=e5_hits, topk=fused_topk)
    return avigate_hits, e5_hits, fused_hits


def write_cvr_comparison(
    *,
    output_dir: str | Path,
    baseline_summary: dict[str, Any],
    e5_summary: dict[str, Any],
    fusion_summary: dict[str, Any],
    agent_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    rows = [
        {"method": "AVIGATE baseline", **_metric_row(baseline_summary.get("avigate_recall", baseline_summary))},
        {"method": "e5-omni only", **_metric_row(e5_summary.get("recall", e5_summary))},
        {"method": "AVIGATE + e5 fusion", **_metric_row(fusion_summary.get("fused_recall", fusion_summary))},
    ]
    if agent_summary is not None:
        rows.append(
            {
                "method": "AVIGATE + e5 fusion + Qwen2.5-Omni Agent",
                **_metric_row(agent_summary.get("final_recall", agent_summary)),
            }
        )
    comparison = {"output_dir": str(root), "rows": rows}
    _write_json(root / "comparison.json", comparison)
    _write_text(root / "comparison.md", _comparison_markdown(comparison))
    return comparison


def run_avigate_selected_baseline(
    *,
    avigate_runtime: Any,
    triplets: list[CVRTriplet],
    recall_ks: tuple[int, ...],
    topk: int,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    recall_ks = _normalize_recall_ks(recall_ks)
    hit_counts = {k: 0 for k in recall_ks}
    trace_lines: list[str] = []
    for run_index, triplet in enumerate(triplets, start=1):
        _emit(progress, f"[avigate] start {run_index}/{len(triplets)}: {triplet.sample_id}")
        views = build_cvr_query_views(triplet)
        hits = retrieve_videos_from_text_official(views.avigate_text_query, avigate_runtime, topk=max(max(recall_ks), topk))
        target_rank = _rank_of_target(hits, triplet.sample_id)
        _update_counts(hit_counts, target_rank, recall_ks)
        trace_lines.append(
            json.dumps(
                {
                    "mode": "cvr-avigate-baseline",
                    "sample_id": triplet.sample_id,
                    "target_video_id": triplet.sample_id,
                    "target_rank": target_rank,
                    "query_text": views.avigate_text_query,
                    "topk_hits": [hit.to_dict() for hit in hits[:topk]],
                },
                ensure_ascii=False,
            )
        )
    summary = _summary_from_counts(
        mode="cvr-avigate-baseline",
        runs=len(triplets),
        recall_ks=recall_ks,
        hit_counts=hit_counts,
    )
    summary["avigate_recall"] = summary.pop("recall")
    _write_json(output_root / "summary.json", summary)
    _write_text(output_root / "traces.jsonl", "\n".join(trace_lines) + ("\n" if trace_lines else ""))
    return summary


def _agent_summary(
    *,
    runs: int,
    recall_ks: tuple[int, ...],
    source_counts: dict[str, dict[int, int]],
    final_top1_correct: int,
    total_omni_calls: int,
    fallback_runs: int,
    avigate_topk: int,
    e5_topk: int,
    fused_topk: int,
    rerank_window: int,
) -> dict[str, Any]:
    summary = {
        "mode": "cvr-agent",
        "runs": runs,
        "final_top1_accuracy": round(final_top1_correct / max(1, runs), 4),
        "avg_omni_calls": round(total_omni_calls / max(1, runs), 4),
        "fallback_rate": round(fallback_runs / max(1, runs), 4),
        "avigate_topk": int(avigate_topk),
        "e5_topk": int(e5_topk),
        "fused_topk": int(fused_topk),
        "rerank_window": int(rerank_window),
    }
    for key, counts in source_counts.items():
        summary[key] = _metrics_from_counts(counts, runs, recall_ks)
    return summary


def _summary_from_counts(
    *,
    mode: str,
    runs: int,
    recall_ks: tuple[int, ...],
    hit_counts: dict[int, int],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "runs": runs,
        "recall": _metrics_from_counts(hit_counts, runs, recall_ks),
    }


def _metrics_from_counts(hit_counts: dict[int, int], runs: int, recall_ks: tuple[int, ...]) -> dict[str, float]:
    return {f"R@{k}": round(hit_counts[k] / max(1, runs), 4) for k in recall_ks}


def _metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in ("R@1", "R@5", "R@10")}


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Full CVR Comparison",
        "",
        f"- output_dir: `{comparison['output_dir']}`",
        "",
        "| Method | R@1 | R@5 | R@10 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(f"| {row['method']} | {_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} |")
    return "\n".join(lines) + "\n"


def _normalize_recall_ks(raw: tuple[int, ...]) -> tuple[int, ...]:
    ks = tuple(sorted({int(k) for k in raw if int(k) > 0}))
    if not ks:
        raise ValueError("recall_ks must contain at least one positive value")
    return ks


def _rank_of_target(hits: list[RetrievalHit], target_video_id: str) -> int | None:
    for hit in hits:
        if hit.video_id == target_video_id:
            return int(hit.rank)
    return None


def _rank_of_target_fused(hits: list[FusedHit], target_video_id: str) -> int | None:
    for hit in hits:
        if hit.video_id == target_video_id:
            return int(hit.rank)
    return None


def _rank_of_dict_target(hits: list[dict], target_video_id: str) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if hit.get("video_id") == target_video_id:
            return index
    return None


def _update_counts(counts: dict[int, int], rank: int | None, recall_ks: tuple[int, ...]) -> None:
    if rank is None:
        return
    for k in recall_ks:
        if rank <= k:
            counts[k] += 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
