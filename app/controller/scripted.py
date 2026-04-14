from __future__ import annotations

from dataclasses import dataclass

from app.backends.base import RetrievalBackend
from app.schemas import RetrievalParams, RunTrace
from app.tools import AgentSessionState, SessionTools


@dataclass(slots=True)
class ScriptedPolicy:
    name: str
    max_rounds: int
    adaptive_params: bool
    fixed_params: RetrievalParams

    def to_trace_metadata(self) -> dict[str, object]:
        return {
            "profile": self.name,
            "max_rounds": self.max_rounds,
            "adaptive_params": self.adaptive_params,
            "fixed_params": self.fixed_params.to_dict(),
            "target_visible_during_runtime": False,
            "success_computed_offline": True,
        }


def resolve_scripted_policy(
    profile: str = "adaptive",
    *,
    fixed_video_weight: float = 0.7,
    fixed_audio_weight: float = 0.3,
    fixed_object_focus: str = "none",
    fixed_temporal_focus: str = "global",
    fixed_topk: int = 5,
) -> ScriptedPolicy:
    fixed_params = RetrievalParams(
        video_weight=fixed_video_weight,
        audio_weight=fixed_audio_weight,
        object_focus=fixed_object_focus,
        temporal_focus=fixed_temporal_focus,
        topk=fixed_topk,
    )

    if profile == "adaptive":
        return ScriptedPolicy(
            name="adaptive",
            max_rounds=3,
            adaptive_params=True,
            fixed_params=fixed_params,
        )
    if profile == "fixed":
        return ScriptedPolicy(
            name="fixed",
            max_rounds=3,
            adaptive_params=False,
            fixed_params=fixed_params,
        )
    if profile == "single-round-fixed":
        return ScriptedPolicy(
            name="single-round-fixed",
            max_rounds=1,
            adaptive_params=False,
            fixed_params=fixed_params,
        )
    raise ValueError(f"unsupported scripted profile: {profile}")


class ScriptedController:
    def __init__(
        self,
        backend: RetrievalBackend,
        policy: ScriptedPolicy | None = None,
    ) -> None:
        self.backend = backend
        self.policy = policy or resolve_scripted_policy()
        self.name = f"scripted:{self.policy.name}"

    def run(self, query_id: str) -> RunTrace:
        query = self.backend.get_query(query_id)
        runtime_query = query.without_target()
        trace = RunTrace(
            query=runtime_query,
            planner_name=self.name,
            planner_metadata=self.policy.to_trace_metadata(),
        )
        state = AgentSessionState(
            query=runtime_query,
            trace=trace,
            max_rounds=self.policy.max_rounds,
        )
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
            if not candidates:
                raise RuntimeError("retrieve_candidates returned no candidates")
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
                candidate_priority = self._priority_tuple(query, candidate, compare)
                best_priority = (
                    self._priority_tuple(
                        query,
                        {"audio_score": -1.0, "video_score": -1.0, "score": -1.0},
                        best_compare,
                    )
                    if best_compare is not None
                    else None
                )
                if best_compare is None or candidate_priority > best_priority:
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
                explanation = self._build_explanation(
                    best_candidate_id,
                    best_compare,
                    current_round.round_index,
                )
                tools.call_tool(
                    "submit_best_candidate",
                    {
                        "candidate_id": best_candidate_id,
                        "explanation": explanation,
                    },
                )
            else:
                current_round.decision = "retry"
                current_round.notes = self._retry_reason(primary_compare, best_compare)

        return trace

    def _choose_params(self, state: AgentSessionState) -> RetrievalParams:
        if not self.policy.adaptive_params:
            params = self.policy.fixed_params
            return RetrievalParams(
                video_weight=params.video_weight,
                audio_weight=params.audio_weight,
                object_focus=params.object_focus,
                temporal_focus=params.temporal_focus,
                topk=params.topk,
            )

        query = state.query
        current_round_index = len(state.trace.rounds) + 1
        params = RetrievalParams(topk=self.policy.fixed_params.topk)

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

    def _retry_reason(
        self,
        primary_comparison: dict[str, object],
        best_comparison: dict[str, object],
    ) -> str:
        primary_missing = primary_comparison.get("missing", [])
        primary_conflicts = primary_comparison.get("conflicts", [])
        best_candidate_gap = primary_comparison != best_comparison
        return (
            f"retry because top-1 missing={primary_missing}, "
            f"top-1 conflicts={primary_conflicts}, "
            f"better_inspected_candidate={best_candidate_gap}"
        )

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

    def _priority_tuple(
        self,
        query,
        candidate: dict[str, object],
        comparison: dict[str, object],
    ) -> tuple[float, float, float, float]:
        modality_score = float(candidate.get("video_score", 0.0))
        if query.required_audio_tags:
            modality_score = float(candidate.get("audio_score", 0.0))
        return (
            float(comparison.get("confidence", 0.0)),
            modality_score,
            float(candidate.get("score", 0.0)),
            -float(len(comparison.get("missing", []))),
        )
