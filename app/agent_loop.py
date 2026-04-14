from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from app.omni_checker import CheckerResult, OmniChecker
from app.retriever import FeatureRetriever, RetrievalHit, normalize_weights

Mode = Literal["t2v", "v2t"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class RetrievalParams:
    alpha_visual: float = 0.8
    alpha_audio: float = 0.2
    topk: int = 10

    def normalized(self, audio_available: bool) -> "RetrievalParams":
        alpha_visual, alpha_audio = normalize_weights(self.alpha_visual, self.alpha_audio, audio_available)
        topk = 10 if int(self.topk) >= 10 else 5
        return RetrievalParams(alpha_visual=alpha_visual, alpha_audio=alpha_audio, topk=topk)

    def to_dict(self) -> dict:
        return {
            "alpha_visual": round(self.alpha_visual, 4),
            "alpha_audio": round(self.alpha_audio, 4),
            "topk": int(self.topk),
        }


@dataclass(frozen=True, slots=True)
class ControllerAction:
    action: str
    submit_rank: int | None
    rewritten_query: str
    alpha_visual: float
    alpha_audio: float
    topk: int
    inspect_ranks_next: list[int]
    reason: str

    def normalized(self, audio_available: bool) -> "ControllerAction":
        alpha_visual, alpha_audio = normalize_weights(self.alpha_visual, self.alpha_audio, audio_available)
        topk = 10 if int(self.topk) >= 10 else 5
        inspect = [rank for rank in self.inspect_ranks_next if rank in (1, 2)]
        if not inspect:
            inspect = [1]
        return ControllerAction(
            action=self.action if self.action in ("submit", "retry") else "retry",
            submit_rank=self.submit_rank if self.submit_rank in (1, 2) else None,
            rewritten_query=self.rewritten_query,
            alpha_visual=alpha_visual,
            alpha_audio=alpha_audio,
            topk=topk,
            inspect_ranks_next=inspect[:2],
            reason=self.reason,
        )

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "submit_rank": self.submit_rank,
            "rewritten_query": self.rewritten_query,
            "alpha_visual": round(self.alpha_visual, 4),
            "alpha_audio": round(self.alpha_audio, 4),
            "topk": int(self.topk),
            "inspect_ranks_next": list(self.inspect_ranks_next),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CandidateInspection:
    rank: int
    candidate: RetrievalHit
    checker: CheckerResult

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "candidate": self.candidate.to_dict(),
            "checker": self.checker.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ControllerStep:
    state_text: str
    action: ControllerAction

    def to_dict(self) -> dict:
        return {"state_text": self.state_text, "action_json": self.action.to_dict()}


@dataclass(slots=True)
class RoundTrace:
    round_index: int
    mode: Mode
    query_repr: str
    retrieval_params: RetrievalParams
    retrieval_results: list[dict]
    checker_outputs: list[dict] = field(default_factory=list)
    controller_steps: list[dict] = field(default_factory=list)
    final_action: dict | None = None

    def to_dict(self) -> dict:
        return {
            "round_index": self.round_index,
            "mode": self.mode,
            "query_repr": self.query_repr,
            "retrieval_params": self.retrieval_params.to_dict(),
            "retrieval_results": list(self.retrieval_results),
            "checker_outputs": list(self.checker_outputs),
            "controller_steps": list(self.controller_steps),
            "final_action": self.final_action,
        }


def build_state_text(
    *,
    mode: Mode,
    query_repr: str,
    round_index: int,
    params: RetrievalParams,
    retrieval_results: list[RetrievalHit],
    inspections: list[CandidateInspection],
    retry_count: int,
) -> str:
    lines = [
        f"mode: {mode}",
        f"query: {query_repr}",
        f"round: {round_index}",
        f"retry_count: {retry_count}",
        f"params: alpha_visual={params.alpha_visual:.2f}, alpha_audio={params.alpha_audio:.2f}, topk={params.topk}",
        "retrieval:",
    ]
    for hit in retrieval_results[: min(5, len(retrieval_results))]:
        item_id = hit.video_id if mode == "t2v" else hit.text_id
        text = hit.text if hit.text else ""
        lines.append(f"- rank={hit.rank} item={item_id} score={hit.score:.4f} text={text}")
    lines.append("inspections:")
    if not inspections:
        lines.append("- none")
    for inspection in inspections:
        checker = inspection.checker
        lines.append(
            f"- rank={inspection.rank} match={checker.is_match} conf={checker.confidence:.2f} "
            f"visual={checker.visual_match:.2f} audio={checker.audio_match:.2f} "
            f"missing={checker.missing_elements} rewrite={checker.rewrite_suggestion}"
        )
    return "\n".join(lines)


def choose_best_inspected(inspections: list[CandidateInspection]) -> CandidateInspection | None:
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


def decide_action(
    *,
    mode: Mode,
    query_repr: str,
    round_index: int,
    max_rounds: int,
    params: RetrievalParams,
    retrieval_results: list[RetrievalHit],
    inspections: list[CandidateInspection],
    retry_count: int,
    audio_available: bool,
) -> ControllerStep:
    state_text = build_state_text(
        mode=mode,
        query_repr=query_repr,
        round_index=round_index,
        params=params,
        retrieval_results=retrieval_results,
        inspections=inspections,
        retry_count=retry_count,
    )
    best = choose_best_inspected(inspections)
    top2 = next((item for item in inspections if item.rank == 2), None)

    if best is not None and best.checker.is_match and best.checker.confidence >= 0.65:
        action = ControllerAction(
            action="submit",
            submit_rank=best.rank,
            rewritten_query=query_repr,
            alpha_visual=params.alpha_visual,
            alpha_audio=params.alpha_audio,
            topk=params.topk,
            inspect_ranks_next=[1],
            reason="best inspected candidate is good enough",
        )
        return ControllerStep(state_text=state_text, action=action.normalized(audio_available))

    if top2 is None:
        action = ControllerAction(
            action="retry",
            submit_rank=None,
            rewritten_query=query_repr,
            alpha_visual=params.alpha_visual,
            alpha_audio=params.alpha_audio,
            topk=params.topk,
            inspect_ranks_next=[1, 2],
            reason="inspect top2 before making a final decision",
        )
        return ControllerStep(state_text=state_text, action=action.normalized(audio_available))

    if round_index < max_rounds:
        rewritten_query = query_repr
        if mode == "t2v" and best is not None and best.checker.rewrite_suggestion.strip():
            rewritten_query = best.checker.rewrite_suggestion.strip()
        alpha_visual = params.alpha_visual
        alpha_audio = params.alpha_audio
        if audio_available and best is not None and best.checker.audio_match < best.checker.visual_match:
            alpha_visual, alpha_audio = 0.6, 0.4
        elif audio_available and best is not None and best.checker.visual_match < best.checker.audio_match:
            alpha_visual, alpha_audio = 0.85, 0.15
        action = ControllerAction(
            action="retry",
            submit_rank=None,
            rewritten_query=rewritten_query,
            alpha_visual=alpha_visual,
            alpha_audio=alpha_audio,
            topk=10,
            inspect_ranks_next=[1, 2],
            reason="retry with retuned parameters and optional rewrite",
        )
        return ControllerStep(state_text=state_text, action=action.normalized(audio_available))

    submit_rank = best.rank if best is not None else 1
    action = ControllerAction(
        action="submit",
        submit_rank=submit_rank,
        rewritten_query=query_repr,
        alpha_visual=params.alpha_visual,
        alpha_audio=params.alpha_audio,
        topk=params.topk,
        inspect_ranks_next=[1],
        reason="reached max rounds, submit best inspected candidate",
    )
    return ControllerStep(state_text=state_text, action=action.normalized(audio_available))


def rank_of_target_t2v(results: list[RetrievalHit], target_video_id: str | None) -> int | None:
    if not target_video_id:
        return None
    for hit in results:
        if hit.video_id == target_video_id:
            return hit.rank
    return None


def rank_of_target_v2t(results: list[RetrievalHit], target_text_ids: list[str] | None) -> int | None:
    if not target_text_ids:
        return None
    targets = set(target_text_ids)
    for hit in results:
        if hit.text_id in targets:
            return hit.rank
    return None


def compute_offline_reward(rank_before: int | None, rank_after: int | None, rounds_used: int) -> float | None:
    if rank_after is None:
        return None
    reward = 0.0
    if rank_before is not None and rank_before > rank_after:
        reward += min(1.0, (rank_before - rank_after) / 10.0)
    if rank_after == 1:
        reward += 1.0
    elif rank_after <= 5:
        reward += 0.5
    elif rank_after <= 10:
        reward += 0.25
    reward -= 0.05 * max(0, rounds_used - 1)
    return round(reward, 4)


def inspect_requested_ranks(
    *,
    mode: Mode,
    checker: OmniChecker,
    retriever: FeatureRetriever,
    query_text: str | None,
    query_video_id: str | None,
    retrieval_results: list[RetrievalHit],
    requested_ranks: list[int],
    existing: list[CandidateInspection],
) -> list[CandidateInspection]:
    known = {item.rank for item in existing}
    updated = list(existing)
    for rank in requested_ranks:
        if rank in known or rank < 1 or rank > len(retrieval_results):
            continue
        hit = retrieval_results[rank - 1]
        if mode == "t2v":
            candidate_video = retriever.get_video_row(hit.video_id)
            result = checker.inspect_t2v(query_text or "", candidate_video, rank=rank, score=hit.score)
        else:
            query_video = retriever.get_video_row(query_video_id or "")
            candidate_text = retriever.get_text_row(hit.text_id or "")
            result = checker.inspect_v2t(query_video, candidate_text, rank=rank, score=hit.score)
        updated.append(CandidateInspection(rank=rank, candidate=hit, checker=result))
        if len(updated) >= 2:
            break
    return sorted(updated, key=lambda item: item.rank)


def run_agent_case(
    *,
    mode: Mode,
    retriever: FeatureRetriever,
    checker: OmniChecker,
    query_text: str | None = None,
    query_video_id: str | None = None,
    target_video_id: str | None = None,
    target_text_ids: list[str] | None = None,
    initial_params: RetrievalParams | None = None,
    max_rounds: int = 3,
) -> dict[str, Any]:
    if mode == "t2v" and not query_text:
        raise ValueError("t2v mode requires query_text")
    if mode == "v2t" and not query_video_id:
        raise ValueError("v2t mode requires query_video_id")

    params = (initial_params or RetrievalParams()).normalized(retriever.audio_available)
    current_query_text = query_text or ""
    retry_count = 0
    round_traces: list[RoundTrace] = []
    checker_call_count = 0
    rank_before: int | None = None
    rank_after: int | None = None
    final_submission: RetrievalHit | None = None

    for round_index in range(1, max_rounds + 1):
        if mode == "t2v":
            results = retriever.retrieve_t2v(
                query_text=current_query_text,
                alpha_visual=params.alpha_visual,
                alpha_audio=params.alpha_audio,
                topk=params.topk,
            )
            if round_index == 1:
                rank_before = rank_of_target_t2v(results, target_video_id)
            query_repr = current_query_text
        else:
            results = retriever.retrieve_v2t(
                query_video_id or "",
                alpha_visual=params.alpha_visual,
                alpha_audio=params.alpha_audio,
                topk=params.topk,
            )
            if round_index == 1:
                rank_before = rank_of_target_v2t(results, target_text_ids)
            query_repr = f"video:{query_video_id}"

        inspections = inspect_requested_ranks(
            mode=mode,
            checker=checker,
            retriever=retriever,
            query_text=current_query_text if mode == "t2v" else None,
            query_video_id=query_video_id if mode == "v2t" else None,
            retrieval_results=results,
            requested_ranks=[1],
            existing=[],
        )
        checker_call_count += len(inspections)

        steps: list[ControllerStep] = []
        step = decide_action(
            mode=mode,
            query_repr=query_repr,
            round_index=round_index,
            max_rounds=max_rounds,
            params=params,
            retrieval_results=results,
            inspections=inspections,
            retry_count=retry_count,
            audio_available=retriever.audio_available,
        )
        steps.append(step)

        if 2 in step.action.inspect_ranks_next and len(inspections) < 2:
            updated = inspect_requested_ranks(
                mode=mode,
                checker=checker,
                retriever=retriever,
                query_text=current_query_text if mode == "t2v" else None,
                query_video_id=query_video_id if mode == "v2t" else None,
                retrieval_results=results,
                requested_ranks=step.action.inspect_ranks_next,
                existing=inspections,
            )
            checker_call_count += len(updated) - len(inspections)
            inspections = updated
            step = decide_action(
                mode=mode,
                query_repr=query_repr,
                round_index=round_index,
                max_rounds=max_rounds,
                params=params,
                retrieval_results=results,
                inspections=inspections,
                retry_count=retry_count,
                audio_available=retriever.audio_available,
            )
            steps.append(step)

        final_step = steps[-1]
        round_traces.append(
            RoundTrace(
                round_index=round_index,
                mode=mode,
                query_repr=query_repr,
                retrieval_params=params,
                retrieval_results=[item.to_dict() for item in results],
                checker_outputs=[item.to_dict() for item in inspections],
                controller_steps=[item.to_dict() for item in steps],
                final_action=final_step.action.to_dict(),
            )
        )

        if final_step.action.action == "submit" or round_index == max_rounds:
            submit_rank = final_step.action.submit_rank or 1
            final_submission = results[submit_rank - 1]
            rank_after = (
                rank_of_target_t2v(results, target_video_id)
                if mode == "t2v"
                else rank_of_target_v2t(results, target_text_ids)
            )
            break

        retry_count += 1
        params = RetrievalParams(
            alpha_visual=final_step.action.alpha_visual,
            alpha_audio=final_step.action.alpha_audio,
            topk=final_step.action.topk,
        ).normalized(retriever.audio_available)
        if mode == "t2v" and final_step.action.rewritten_query.strip():
            current_query_text = final_step.action.rewritten_query.strip()

    return {
        "mode": mode,
        "planner_name": "heuristic-loop-agent",
        "planner_metadata": {
            "target_visible_during_runtime": False,
            "success_computed_offline": True,
            "max_rounds": max_rounds,
        },
        "query_text": query_text,
        "query_video_id": query_video_id,
        "rounds": [item.to_dict() for item in round_traces],
        "checker_call_count": checker_call_count,
        "final_submission": final_submission.to_dict() if final_submission else None,
        "target_rank_before": rank_before,
        "target_rank_after": rank_after,
        "offline_reward": compute_offline_reward(rank_before, rank_after, len(round_traces)),
        "success": rank_after == 1 if rank_after is not None else None,
        "created_at": utc_now_iso(),
    }


def summarize_agent_traces(traces: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    total = len(traces)
    round1 = {f"R@{k}": 0.0 for k in ks}
    final = {f"R@{k}": 0.0 for k in ks}
    total_rounds = 0
    total_checker_calls = 0
    retry_count = 0
    submit_top1 = 0
    submit_top2 = 0

    for trace in traces:
        total_rounds += len(trace.get("rounds", []))
        total_checker_calls += int(trace.get("checker_call_count", 0))
        if len(trace.get("rounds", [])) > 1:
            retry_count += 1
        final_submission = trace.get("final_submission") or {}
        if final_submission.get("rank") == 1:
            submit_top1 += 1
        if final_submission.get("rank") == 2:
            submit_top2 += 1
        for target, bucket in (("target_rank_before", round1), ("target_rank_after", final)):
            rank_value = trace.get(target)
            if rank_value is None:
                continue
            for k in ks:
                if rank_value <= k:
                    bucket[f"R@{k}"] += 1.0

    def _normalize(values: dict[str, float]) -> dict[str, float]:
        return {key: round(value / max(1, total), 4) for key, value in values.items()}

    return {
        "runs": total,
        "round1_recall": _normalize(round1),
        "final_recall": _normalize(final),
        "avg_rounds": round(total_rounds / max(1, total), 4),
        "avg_checker_calls": round(total_checker_calls / max(1, total), 4),
        "retry_rate": round(retry_count / max(1, total), 4),
        "submit_top1_rate": round(submit_top1 / max(1, total), 4),
        "submit_top2_rate": round(submit_top2 / max(1, total), 4),
    }
