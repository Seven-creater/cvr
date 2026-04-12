from __future__ import annotations

from app.backends.base import RetrievalBackend
from app.schemas import RetrievalParams, RunTrace
from app.tools import AgentSessionState, SessionTools


class ScriptedController:
    name = "scripted"

    def __init__(self, backend: RetrievalBackend) -> None:
        self.backend = backend

    def run(self, query_id: str) -> RunTrace:
        query = self.backend.get_query(query_id)
        trace = RunTrace(query=query, planner_name=self.name)
        state = AgentSessionState(query=query, trace=trace)
        tools = SessionTools(self.backend, state)

        while not state.is_finished():
            params = self._choose_params(state)
            retrieval = tools.call_tool(
                "retrieve_candidates",
                {
                    "video_weight": params.video_weight,
                    "audio_weight": params.audio_weight,
                    "object_focus": params.object_focus,
                    "temporal_focus": params.temporal_focus,
                    "topk": params.topk,
                },
            )
            candidates = retrieval["candidates"][:2]
            primary_candidate_id = candidates[0]["candidate_id"]
            primary_compare = None
            best_candidate_id = None
            best_compare = None

            for candidate in candidates:
                candidate_id = candidate["candidate_id"]
                tools.call_tool("inspect_candidate", {"candidate_id": candidate_id})
                compare = tools.call_tool("compare_to_request", {"candidate_id": candidate_id})
                if candidate_id == primary_candidate_id:
                    primary_compare = compare
                if best_compare is None or compare["confidence"] > best_compare["confidence"]:
                    best_candidate_id = candidate_id
                    best_compare = compare

            assert (
                best_candidate_id is not None
                and best_compare is not None
                and primary_compare is not None
            )

            current_round = state.current_round()
            should_submit = (
                not primary_compare["missing"]
                or current_round.round_index >= state.max_rounds
            )
            if should_submit:
                explanation = self._build_explanation(best_candidate_id, best_compare, current_round.round_index)
                tools.call_tool(
                    "submit_best_candidate",
                    {
                        "candidate_id": best_candidate_id,
                        "explanation": explanation,
                    },
                )
            else:
                current_round.decision = "retry"
                current_round.notes = self._retry_reason(best_compare)

        return trace

    def _choose_params(self, state: AgentSessionState) -> RetrievalParams:
        query = state.query
        current_round_index = len(state.trace.rounds) + 1
        params = RetrievalParams(topk=5)

        needs_audio = bool(query.required_audio_tags)
        if needs_audio:
            params.video_weight = 0.95 if current_round_index == 1 else 0.45
            params.audio_weight = 0.05 if current_round_index == 1 else 0.55

        if query.required_objects:
            if current_round_index == 1:
                params.object_focus = "none"
            else:
                params.object_focus = query.required_objects[0]

        if query.required_temporal:
            params.temporal_focus = "global" if current_round_index == 1 else query.required_temporal

        if current_round_index == state.max_rounds and not query.required_objects and query.required_temporal:
            params.audio_weight = max(params.audio_weight, 0.45)
            params.video_weight = 1.0 - params.audio_weight

        return RetrievalParams(
            video_weight=params.video_weight,
            audio_weight=params.audio_weight,
            object_focus=params.object_focus,
            temporal_focus=params.temporal_focus,
            topk=params.topk,
        )

    def _retry_reason(self, comparison: dict[str, object]) -> str:
        missing = comparison.get("missing", [])
        conflicts = comparison.get("conflicts", [])
        return f"retry because missing={missing}, conflicts={conflicts}"

    def _build_explanation(
        self,
        candidate_id: str,
        comparison: dict[str, object],
        round_index: int,
    ) -> str:
        satisfied = ", ".join(comparison.get("satisfied", [])) or "none"
        missing = ", ".join(comparison.get("missing", [])) or "none"
        return (
            f"Submitted {candidate_id} at round {round_index}. "
            f"Satisfied: {satisfied}. Missing: {missing}."
        )
