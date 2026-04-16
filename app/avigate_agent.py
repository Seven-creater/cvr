from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.avigate_official import (
    retrieve_texts_from_video_official,
    retrieve_videos_from_text_official,
)
from app.omni_checker import CheckerResult, OmniChecker
from app.retrieval_types import RetrievalHit


@dataclass(frozen=True, slots=True)
class CandidateInspection:
    rank: int
    candidate: RetrievalHit
    checker: CheckerResult

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "candidate": self.candidate.to_dict(),
            "checker_result": self.checker.to_dict(),
        }


def run_official_agent_partial_eval(
    *,
    mode: str,
    runtime: Any,
    checker: OmniChecker,
    sample_size: int,
    topk: int = 10,
    max_iter: int = 3,
    submit_threshold: float = 0.7,
    recall_ks: tuple[int, ...] = (1, 5, 10),
    output_dir: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    if mode not in {"t2v", "v2t"}:
        raise ValueError("mode must be 't2v' or 'v2t'")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    recall_ks = tuple(sorted({int(k) for k in recall_ks if int(k) > 0}))
    if not recall_ks:
        raise ValueError("recall_ks must contain at least one positive integer")

    rows = runtime.text_rows if mode == "t2v" else runtime.video_rows
    total = min(sample_size, len(rows))
    if total <= 0:
        raise ValueError("no rows available for partial eval")

    summary_path: Path | None = None
    traces_path: Path | None = None
    if output_dir:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "summary.json"
        traces_path = output_root / "traces.jsonl"
        traces_path.write_text("", encoding="utf-8")

    round1_hits = {k: 0 for k in recall_ks}
    final_hits = {k: 0 for k in recall_ks}
    total_rounds = 0
    total_checker_calls = 0
    followup_runs = 0
    submit_top1 = 0
    submit_top2 = 0

    for run_index, row in enumerate(rows[:total], start=1):
        label = row.text if mode == "t2v" else row.video_id
        _emit_progress(progress, f"[{mode}] start {run_index}/{total}: {label}")

        if mode == "t2v":
            trace = run_t2v_official_agent_case(
                query_text=row.text,
                runtime=runtime,
                checker=checker,
                topk=topk,
                max_iter=max_iter,
                submit_threshold=submit_threshold,
                progress=progress,
            )
            _update_t2v_recall_counts(
                trace=trace,
                target_video_id=row.video_id,
                ks=recall_ks,
                round1_hits=round1_hits,
                final_hits=final_hits,
            )
        else:
            trace = run_v2t_official_agent_case(
                query_video_id=row.video_id,
                runtime=runtime,
                checker=checker,
                topk=topk,
                max_iter=max_iter,
                submit_threshold=submit_threshold,
                progress=progress,
            )
            _update_v2t_recall_counts(
                trace=trace,
                target_text_ids=set(runtime.target_text_ids(row.video_id)),
                ks=recall_ks,
                round1_hits=round1_hits,
                final_hits=final_hits,
            )

        total_rounds += len(trace["iterations"])
        checker_calls = sum(len(iteration.get("new_checked_candidates", [])) for iteration in trace["iterations"])
        total_checker_calls += checker_calls
        if any(iteration.get("action") != "submit" for iteration in trace["iterations"][:-1]):
            followup_runs += 1

        final_rank = int(trace["final_result"]["rank_in_final_search"])
        if final_rank == 1:
            submit_top1 += 1
        elif final_rank == 2:
            submit_top2 += 1

        if traces_path is not None:
            with traces_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

        summary = _build_partial_eval_summary(
            mode=mode,
            runs=run_index,
            recall_ks=recall_ks,
            round1_hits=round1_hits,
            final_hits=final_hits,
            total_rounds=total_rounds,
            total_checker_calls=total_checker_calls,
            followup_runs=followup_runs,
            submit_top1=submit_top1,
            submit_top2=submit_top2,
        )
        if summary_path is not None:
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        _emit_progress(
            progress,
            f"[{mode}] done {run_index}/{total}: final_rank={final_rank} checker_calls={checker_calls}",
        )

    result = {"summary": summary}
    if summary_path is not None:
        result["summary_path"] = str(summary_path)
    if traces_path is not None:
        result["traces_path"] = str(traces_path)
    return result


def run_t2v_official_agent_case(
    *,
    query_text: str,
    runtime: Any,
    checker: OmniChecker,
    topk: int = 10,
    max_iter: int = 3,
    submit_threshold: float = 0.7,
    progress: Callable[[str], None] | None = None,
) -> dict:
    current_query = query_text
    iterations: list[dict] = []

    for iter_index in range(1, max_iter + 1):
        _emit_progress(progress, f"[t2v] iter {iter_index}/{max_iter} query={current_query}")
        hits = retrieve_videos_from_text_official(current_query, runtime, topk=topk)
        inspected = _inspect_t2v_hits(
            query_text=current_query,
            runtime=runtime,
            checker=checker,
            hits=hits,
            ranks=[1, 2],
            progress=progress,
        )
        best = _choose_best_inspection(inspected)
        rewritten_query = best.checker.rewrite_suggestion.strip() if best else ""

        if best and best.checker.is_match and best.checker.confidence >= submit_threshold:
            iterations.append(
                {
                    "iter": iter_index,
                    "query_text": current_query,
                    "retrieval_hits": [hit.to_dict() for hit in hits],
                    "checked_candidates": [item.to_dict() for item in inspected],
                    "new_checked_candidates": [item.to_dict() for item in inspected],
                    "action": "submit",
                }
            )
            return {
                "mode": "t2v-agent",
                "query_text": query_text,
                "final_action": "submit",
                "iterations": iterations,
                "final_result": {
                    "video_id": best.candidate.video_id,
                    "rank_in_final_search": best.rank,
                    "total_iters": iter_index,
                },
            }

        if iter_index < max_iter and rewritten_query and rewritten_query != current_query:
            iterations.append(
                {
                    "iter": iter_index,
                    "query_text": current_query,
                    "retrieval_hits": [hit.to_dict() for hit in hits],
                    "checked_candidates": [item.to_dict() for item in inspected],
                    "new_checked_candidates": [item.to_dict() for item in inspected],
                    "action": "retry",
                    "next_query": rewritten_query,
                }
            )
            current_query = rewritten_query
            continue

        chosen = best or CandidateInspection(rank=1, candidate=hits[0], checker=_default_checker_result())
        iterations.append(
            {
                "iter": iter_index,
                "query_text": current_query,
                "retrieval_hits": [hit.to_dict() for hit in hits],
                "checked_candidates": [item.to_dict() for item in inspected],
                "new_checked_candidates": [item.to_dict() for item in inspected],
                "action": "submit",
            }
        )
        return {
            "mode": "t2v-agent",
            "query_text": query_text,
            "final_action": "submit",
            "iterations": iterations,
            "final_result": {
                "video_id": chosen.candidate.video_id,
                "rank_in_final_search": chosen.rank,
                "total_iters": iter_index,
            },
        }

    raise RuntimeError("t2v official agent exhausted without submission")


def run_v2t_official_agent_case(
    *,
    query_video_id: str,
    runtime: Any,
    checker: OmniChecker,
    topk: int = 10,
    max_iter: int = 3,
    submit_threshold: float = 0.7,
    min_inspected_before_submit: int = 6,
    progress: Callable[[str], None] | None = None,
) -> dict:
    inspected_by_rank: dict[int, CandidateInspection] = {}
    iterations: list[dict] = []

    for iter_index in range(1, max_iter + 1):
        _emit_progress(progress, f"[v2t] iter {iter_index}/{max_iter} video_id={query_video_id}")
        hits = retrieve_texts_from_video_official(query_video_id, runtime, topk=topk)
        inspect_cap = min(max(1, min_inspected_before_submit), min(topk, len(hits)))
        pending_ranks = [rank for rank in range(1, min(topk, len(hits)) + 1) if rank not in inspected_by_rank]
        current_ranks = pending_ranks[:2] or [1]
        new_items = _inspect_v2t_hits(
            query_video_id=query_video_id,
            runtime=runtime,
            checker=checker,
            hits=hits,
            ranks=current_ranks,
            progress=progress,
        )
        for item in new_items:
            inspected_by_rank[item.rank] = item

        inspected = [inspected_by_rank[rank] for rank in sorted(inspected_by_rank)]
        best = _choose_best_inspection(inspected)
        accepted = _choose_first_confident_match(inspected, submit_threshold)
        inspected_enough = len(inspected_by_rank) >= inspect_cap

        if accepted is not None and inspected_enough:
            iterations_action = "submit"
            final_rank = accepted.rank
        elif iter_index < max_iter and len(inspected_by_rank) < inspect_cap:
            iterations_action = "inspect_more"
            final_rank = None
        else:
            iterations_action = "submit"
            final_rank = _fallback_v2t_rank(inspected_by_rank, hits)

        iteration_payload = {
            "iter": iter_index,
            "query_video_id": query_video_id,
            "query_video_path": runtime.video_rows[runtime._video_index[query_video_id]].video_path,
            "retrieval_hits": [hit.to_dict() for hit in hits],
            "checked_candidates": [item.to_dict() for item in inspected],
            "new_checked_candidates": [item.to_dict() for item in new_items],
            "action": iterations_action,
        }

        if iterations_action == "inspect_more":
            iterations.append(iteration_payload)
            continue

        iterations.append(iteration_payload)
        chosen = inspected_by_rank.get(final_rank) or CandidateInspection(rank=1, candidate=hits[0], checker=_default_checker_result())
        return {
            "mode": "v2t-agent",
            "query_video_id": query_video_id,
            "query_video_path": runtime.video_rows[runtime._video_index[query_video_id]].video_path,
            "final_action": "submit",
            "iterations": iterations,
            "final_result": {
                "text_id": chosen.candidate.text_id,
                "rank_in_final_search": chosen.rank,
                "total_iters": iter_index,
            },
        }

    raise RuntimeError("v2t official agent exhausted without submission")


def _inspect_t2v_hits(
    *,
    query_text: str,
    runtime: Any,
    checker: OmniChecker,
    hits: list[RetrievalHit],
    ranks: list[int],
    progress: Callable[[str], None] | None = None,
) -> list[CandidateInspection]:
    inspected: list[CandidateInspection] = []
    for rank in ranks:
        if rank < 1 or rank > len(hits):
            continue
        hit = hits[rank - 1]
        candidate_video = runtime.video_rows[runtime._video_index[hit.video_id or ""]]
        _emit_progress(progress, f"[t2v] inspect rank={rank} video_id={candidate_video.video_id}")
        result = checker.inspect_t2v(query_text, candidate_video, rank=rank, score=hit.score)
        inspected.append(CandidateInspection(rank=rank, candidate=hit, checker=result))
    return inspected


def _inspect_v2t_hits(
    *,
    query_video_id: str,
    runtime: Any,
    checker: OmniChecker,
    hits: list[RetrievalHit],
    ranks: list[int],
    progress: Callable[[str], None] | None = None,
) -> list[CandidateInspection]:
    inspected: list[CandidateInspection] = []
    query_video = runtime.video_rows[runtime._video_index[query_video_id]]
    for rank in ranks:
        if rank < 1 or rank > len(hits):
            continue
        hit = hits[rank - 1]
        candidate_text = runtime.text_rows[runtime._text_index[hit.text_id or ""]]
        _emit_progress(progress, f"[v2t] inspect rank={rank} text_id={candidate_text.text_id}")
        result = checker.inspect_v2t(query_video, candidate_text, rank=rank, score=hit.score)
        inspected.append(CandidateInspection(rank=rank, candidate=hit, checker=result))
    return inspected


def _build_partial_eval_summary(
    *,
    mode: str,
    runs: int,
    recall_ks: tuple[int, ...],
    round1_hits: dict[int, int],
    final_hits: dict[int, int],
    total_rounds: int,
    total_checker_calls: int,
    followup_runs: int,
    submit_top1: int,
    submit_top2: int,
) -> dict:
    return {
        "runs": runs,
        "round1_recall": {f"R@{k}": round(round1_hits[k] / runs, 4) for k in recall_ks},
        "final_recall": {f"R@{k}": round(final_hits[k] / runs, 4) for k in recall_ks},
        "avg_rounds": round(total_rounds / runs, 4),
        "avg_checker_calls": round(total_checker_calls / runs, 4),
        "retry_rate": round(followup_runs / runs, 4),
        "submit_top1_rate": round(submit_top1 / runs, 4),
        "submit_top2_rate": round(submit_top2 / runs, 4),
        "mode": mode,
    }


def _update_t2v_recall_counts(
    *,
    trace: dict,
    target_video_id: str,
    ks: tuple[int, ...],
    round1_hits: dict[int, int],
    final_hits: dict[int, int],
) -> None:
    retrieval_hits = trace["iterations"][0]["retrieval_hits"]
    for k in ks:
        top_hits = retrieval_hits[:k]
        if any(hit.get("video_id") == target_video_id for hit in top_hits):
            round1_hits[k] += 1

    final_result = trace["final_result"]
    if final_result.get("video_id") != target_video_id:
        return
    final_rank = int(final_result["rank_in_final_search"])
    for k in ks:
        if final_rank <= k:
            final_hits[k] += 1


def _update_v2t_recall_counts(
    *,
    trace: dict,
    target_text_ids: set[str],
    ks: tuple[int, ...],
    round1_hits: dict[int, int],
    final_hits: dict[int, int],
) -> None:
    retrieval_hits = trace["iterations"][0]["retrieval_hits"]
    for k in ks:
        top_hits = retrieval_hits[:k]
        if any(hit.get("text_id") in target_text_ids for hit in top_hits):
            round1_hits[k] += 1

    final_result = trace["final_result"]
    if final_result.get("text_id") not in target_text_ids:
        return
    final_rank = int(final_result["rank_in_final_search"])
    for k in ks:
        if final_rank <= k:
            final_hits[k] += 1


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _fallback_v2t_rank(inspected_by_rank: dict[int, CandidateInspection], hits: list[RetrievalHit]) -> int:
    if 1 in inspected_by_rank:
        return 1
    if inspected_by_rank:
        return min(inspected_by_rank)
    return 1 if hits else 0


def _choose_first_confident_match(
    inspections: list[CandidateInspection],
    submit_threshold: float,
) -> CandidateInspection | None:
    qualified = [
        item
        for item in inspections
        if item.checker.is_match and item.checker.confidence >= submit_threshold
    ]
    if not qualified:
        return None
    return min(qualified, key=lambda item: item.rank)


def _choose_best_inspection(inspections: list[CandidateInspection]) -> CandidateInspection | None:
    if not inspections:
        return None
    return max(
        inspections,
        key=lambda item: (
            int(item.checker.is_match),
            item.checker.confidence,
            item.checker.visual_match + item.checker.audio_match,
            -item.rank,
        ),
    )


def _default_checker_result() -> CheckerResult:
    return CheckerResult(
        is_match=False,
        confidence=0.0,
        visual_match=0.0,
        audio_match=0.0,
        main_events=[],
        missing_elements=["uninspected_candidate"],
        reason="candidate was not inspected",
        rewrite_suggestion="",
    )
