from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def run_t2v_official_agent_case(
    *,
    query_text: str,
    runtime: Any,
    checker: OmniChecker,
    topk: int = 10,
    max_iter: int = 3,
    submit_threshold: float = 0.7,
) -> dict:
    current_query = query_text
    iterations: list[dict] = []

    for iter_index in range(1, max_iter + 1):
        hits = retrieve_videos_from_text_official(current_query, runtime, topk=topk)
        inspected = _inspect_t2v_hits(
            query_text=current_query,
            runtime=runtime,
            checker=checker,
            hits=hits,
            ranks=[1, 2],
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
) -> dict:
    inspected_by_rank: dict[int, CandidateInspection] = {}
    iterations: list[dict] = []

    for iter_index in range(1, max_iter + 1):
        hits = retrieve_texts_from_video_official(query_video_id, runtime, topk=topk)
        pending_ranks = [rank for rank in range(1, min(topk, len(hits)) + 1) if rank not in inspected_by_rank]
        current_ranks = pending_ranks[:2] or [1]
        new_items = _inspect_v2t_hits(
            query_video_id=query_video_id,
            runtime=runtime,
            checker=checker,
            hits=hits,
            ranks=current_ranks,
        )
        for item in new_items:
            inspected_by_rank[item.rank] = item

        inspected = [inspected_by_rank[rank] for rank in sorted(inspected_by_rank)]
        best = _choose_best_inspection(inspected)

        if best and best.checker.is_match and best.checker.confidence >= submit_threshold:
            iterations_action = "submit"
            final_rank = best.rank
        elif iter_index < max_iter and len(inspected_by_rank) < min(topk, len(hits)):
            iterations_action = "inspect_more"
            final_rank = None
        else:
            iterations_action = "submit"
            final_rank = (best or CandidateInspection(rank=1, candidate=hits[0], checker=_default_checker_result())).rank

        iteration_payload = {
            "iter": iter_index,
            "query_video_id": query_video_id,
            "query_video_path": runtime.video_rows[runtime._video_index[query_video_id]].video_path,
            "retrieval_hits": [hit.to_dict() for hit in hits],
            "checked_candidates": [item.to_dict() for item in inspected],
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
) -> list[CandidateInspection]:
    inspected: list[CandidateInspection] = []
    for rank in ranks:
        if rank < 1 or rank > len(hits):
            continue
        hit = hits[rank - 1]
        candidate_video = runtime.video_rows[runtime._video_index[hit.video_id or ""]]
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
) -> list[CandidateInspection]:
    inspected: list[CandidateInspection] = []
    query_video = runtime.video_rows[runtime._video_index[query_video_id]]
    for rank in ranks:
        if rank < 1 or rank > len(hits):
            continue
        hit = hits[rank - 1]
        candidate_text = runtime.text_rows[runtime._text_index[hit.text_id or ""]]
        result = checker.inspect_v2t(query_video, candidate_text, rank=rank, score=hit.score)
        inspected.append(CandidateInspection(rank=rank, candidate=hit, checker=result))
    return inspected


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
