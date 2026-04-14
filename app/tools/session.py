from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.backends.base import RetrievalBackend
from app.schemas import QueryCase, RetrievalParams, RoundRecord, RunTrace, ToolCallTrace


def _preview(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return {"items": value[:2], "count": len(value)}
    return {"value": value}


@dataclass(slots=True)
class AgentSessionState:
    query: QueryCase
    trace: RunTrace
    max_rounds: int = 3
    max_inspects_per_round: int = 2
    final_submitted: bool = False

    def current_round(self) -> RoundRecord | None:
        return self.trace.rounds[-1] if self.trace.rounds else None

    def ensure_round(self) -> RoundRecord:
        current = self.current_round()
        if current is None or current.decision in {"retry", "submitted"}:
            if len(self.trace.rounds) >= self.max_rounds:
                raise RuntimeError("maximum rounds exceeded")
            current = RoundRecord(round_index=len(self.trace.rounds) + 1)
            self.trace.rounds.append(current)
        return current

    def is_finished(self) -> bool:
        return self.final_submitted

    def state_summary(self, backend: RetrievalBackend) -> str:
        source = backend.inspect_candidate(self.query.source_video_id)
        round_bits: list[str] = []
        for item in self.trace.rounds:
            compared = ", ".join(
                f"{candidate}:{result.confidence:.2f}"
                for candidate, result in item.comparisons.items()
            ) or "none"
            round_bits.append(
                f"round={item.round_index}; decision={item.decision}; "
                f"retrieved={item.retrieved_candidates}; compared={compared}"
            )
        summary = "\n".join(round_bits) if round_bits else "no rounds yet"
        return (
            f"query_id={self.query.query_id}\n"
            f"source_video={self.query.source_video_id}\n"
            f"source_summary={source.summary}\n"
            f"edit_instruction={self.query.edit_instruction}\n"
            f"required_audio={self.query.required_audio_tags}\n"
            f"required_objects={self.query.required_objects}\n"
            f"required_temporal={self.query.required_temporal}\n"
            f"history=\n{summary}"
        )


class SessionTools:
    def __init__(self, backend: RetrievalBackend, state: AgentSessionState) -> None:
        self.backend = backend
        self.state = state

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "retrieve_candidates",
                "description": "Retrieve top-k candidates with explicit audio/video weights. This must be the first tool called in round 1.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_weight": {"type": "number"},
                        "audio_weight": {"type": "number"},
                        "object_focus": {"type": "string"},
                        "temporal_focus": {"type": "string"},
                        "topk": {"type": "integer"},
                    },
                    "required": ["video_weight", "audio_weight", "object_focus", "temporal_focus", "topk"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "inspect_candidate",
                "description": "Inspect one candidate returned in the current round. At most two inspections are allowed per round.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "compare_to_request",
                "description": "Check whether a candidate satisfies the edit request relative to the source video.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "submit_best_candidate",
                "description": "Submit the final candidate when you are confident, or when you have exhausted the third round.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["candidate_id", "explanation"],
                    "additionalProperties": False,
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "retrieve_candidates":
            result = self.retrieve_candidates(arguments)
        elif name == "inspect_candidate":
            result = self.inspect_candidate(arguments)
        elif name == "compare_to_request":
            result = self.compare_to_request(arguments)
        elif name == "submit_best_candidate":
            result = self.submit_best_candidate(arguments)
        else:
            raise KeyError(f"unknown tool: {name}")

        round_index = self.state.current_round().round_index if self.state.current_round() else 0
        self.state.trace.tool_history.append(
            ToolCallTrace(
                tool_name=name,
                arguments=arguments,
                result_preview=_preview(result),
                round_index=round_index,
            )
        )
        return result

    def retrieve_candidates(self, arguments: dict[str, Any]) -> dict[str, Any]:
        current = self.state.ensure_round()
        if current.retrieval_params is not None:
            current.decision = "retry"
            current = self.state.ensure_round()

        params = RetrievalParams(
            video_weight=float(arguments["video_weight"]),
            audio_weight=float(arguments["audio_weight"]),
            object_focus=arguments["object_focus"],
            temporal_focus=arguments["temporal_focus"],
            topk=int(arguments["topk"]),
        )
        current.retrieval_params = params
        candidates = self.backend.retrieve_candidates(
            query=self.state.query,
            params=params,
            round_index=current.round_index,
        )
        current.retrieved_candidates = [item.candidate_id for item in candidates]
        current.inspected_candidates.clear()
        current.comparisons.clear()
        current.decision = "inspecting"
        return {
            "round_index": current.round_index,
            "params": params.to_dict(),
            "candidates": [item.to_dict() for item in candidates],
        }

    def inspect_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        current = self.state.current_round()
        if current is None or not current.retrieved_candidates:
            raise RuntimeError("retrieve_candidates must be called before inspect_candidate")
        candidate_id = arguments["candidate_id"]
        if candidate_id not in current.retrieved_candidates:
            raise RuntimeError("candidate must come from the current retrieval set")
        if len(current.inspected_candidates) >= self.state.max_inspects_per_round:
            raise RuntimeError("inspect limit reached for this round")
        if candidate_id not in current.inspected_candidates:
            current.inspected_candidates.append(candidate_id)
        inspection = self.backend.inspect_candidate(candidate_id)
        return inspection.to_dict()

    def compare_to_request(self, arguments: dict[str, Any]) -> dict[str, Any]:
        current = self.state.current_round()
        if current is None or not current.retrieved_candidates:
            raise RuntimeError("retrieve_candidates must be called before compare_to_request")
        candidate_id = arguments["candidate_id"]
        comparison = self.backend.compare_to_request(self.state.query, candidate_id)
        current.comparisons[candidate_id] = comparison
        return comparison.to_dict()

    def submit_best_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        candidate_id = arguments["candidate_id"]
        explanation = arguments["explanation"].strip()
        current = self.state.current_round()
        if current is None:
            raise RuntimeError("cannot submit before any retrieval")
        current.decision = "submitted"
        current.notes = explanation
        self.state.trace.final_candidate_id = candidate_id
        self.state.trace.final_explanation = explanation
        self.state.final_submitted = True
        return {
            "status": "submitted",
            "candidate_id": candidate_id,
            "explanation": explanation,
        }

    def dump_tool_result(self, tool_name: str, payload: dict[str, Any]) -> str:
        return json.dumps({"tool_name": tool_name, "payload": payload}, ensure_ascii=False)
