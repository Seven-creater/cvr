from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.backends.base import RetrievalBackend
from app.schemas import RunTrace
from app.tools import AgentSessionState, SessionTools

SYSTEM_PROMPT = """You are a concise retrieval controller.

You must solve the query by calling tools.
Rules:
- Begin with retrieve_candidates.
- Inspect at most 2 candidates per round.
- Prefer submit_best_candidate when a candidate satisfies the request.
- If audio cues matter, increase audio_weight when early results miss the audio requirement.
- If object cues matter, set object_focus to the required object on retry.
- If temporal cues matter, set temporal_focus to the required temporal cue on retry.
- Never exceed 3 retrieval rounds.
- Keep explanations short and concrete.
"""


@dataclass(slots=True)
class OpenAIResponsesController:
    backend: RetrievalBackend
    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 600

    def __post_init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI planner")
        self.client = OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def run(self, query_id: str) -> RunTrace:
        query = self.backend.get_query(query_id)
        runtime_query = query.without_target()
        trace = RunTrace(
            query=runtime_query,
            planner_name=self.name,
            planner_metadata={
                "profile": "openai",
                "target_visible_during_runtime": False,
                "success_computed_offline": True,
            },
        )
        state = AgentSessionState(query=runtime_query, trace=trace)
        tools = SessionTools(self.backend, state)
        user_input = (
            "Solve this composed retrieval query.\n\n"
            f"{state.state_summary(self.backend)}\n"
            "Use the tools to retrieve, inspect, compare, and submit the best candidate."
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=user_input,
            tools=tools.tool_specs(),
            max_output_tokens=self.max_output_tokens,
            max_tool_calls=20,
        )

        while not state.is_finished():
            tool_calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]
            if not tool_calls:
                text = getattr(response, "output_text", "") or ""
                if not text:
                    raise RuntimeError("model finished without a tool call or final text")
                if state.current_round() is not None and state.current_round().retrieved_candidates:
                    fallback_id = state.current_round().retrieved_candidates[0]
                    tools.call_tool(
                        "submit_best_candidate",
                        {
                            "candidate_id": fallback_id,
                            "explanation": text.strip()[:400],
                        },
                    )
                    break
                raise RuntimeError("model ended before any candidate was available")

            tool_outputs: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                arguments = json.loads(tool_call.arguments)
                result = tools.call_tool(tool_call.name, arguments)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": tools.dump_tool_result(tool_call.name, result),
                    }
                )
                if state.is_finished():
                    break

            if state.is_finished():
                break

            response = self.client.responses.create(
                model=self.model,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=tools.tool_specs(),
                max_output_tokens=self.max_output_tokens,
                max_tool_calls=20,
            )

        return trace
