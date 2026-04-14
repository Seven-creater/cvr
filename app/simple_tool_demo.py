from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

from app.backends.base import RetrievalBackend
from app.demo import build_backend
from app.schemas import QueryCase, RetrievalParams, RoundRecord, RunTrace, ToolCallTrace, utc_now_iso

SYSTEM_PROMPT = """You are a simple video retrieval assistant.

You can use tools to solve a composed retrieval request.
Recommended workflow:
1. Call retrieve_candidates first.
2. Inspect and compare one or two candidates.
3. If results are not good enough, call retrieve_candidates again with adjusted weights or focus.
4. When ready, call submit_best_candidate.

Do not invent any candidate details. Use only tool outputs.
Keep your final answer short.
"""


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    payload = {
        "role": "assistant",
        "content": message.content or "",
    }
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]
    return payload


@dataclass(slots=True)
class SimpleToolEnvironment:
    backend: RetrievalBackend
    query: QueryCase
    planner_name: str
    original_query: QueryCase = field(init=False)
    runtime_query: QueryCase = field(init=False)
    trace: RunTrace = field(init=False)
    finished: bool = field(init=False, default=False)
    latest_round: RoundRecord | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.original_query = self.query
        self.runtime_query = self.query.without_target()
        self.trace = RunTrace(
            query=self.runtime_query,
            planner_name=self.planner_name,
            planner_metadata={
                "profile": "simple-chat-demo",
                "target_visible_during_runtime": False,
                "success_computed_offline": True,
            },
        )
        self.finished = False
        self.latest_round: RoundRecord | None = None

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_candidates",
                    "description": "Retrieve top-k candidates with configurable video/audio weights and optional object/temporal focus.",
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
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "inspect_candidate",
                    "description": "Inspect one retrieved candidate.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                        },
                        "required": ["candidate_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_to_request",
                    "description": "Compare one candidate against the user request and report satisfied and missing requirements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                        },
                        "required": ["candidate_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_best_candidate",
                    "description": "Submit the final candidate when you are done.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["candidate_id", "explanation"],
                    },
                },
            },
        ]

    def user_prompt(self) -> str:
        source = self.backend.inspect_candidate(self.runtime_query.source_video_id)
        return (
            f"query_id={self.runtime_query.query_id}\n"
            f"source_video_id={self.runtime_query.source_video_id}\n"
            f"source_summary={source.summary}\n"
            f"edit_instruction={self.runtime_query.edit_instruction}\n"
            f"required_audio_tags={self.runtime_query.required_audio_tags}\n"
            f"required_objects={self.runtime_query.required_objects}\n"
            f"required_temporal={self.runtime_query.required_temporal}\n"
            "Please solve the task by calling tools."
        )

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "retrieve_candidates":
            result = self._retrieve_candidates(arguments)
        elif name == "inspect_candidate":
            result = self._inspect_candidate(arguments)
        elif name == "compare_to_request":
            result = self._compare_to_request(arguments)
        elif name == "submit_best_candidate":
            result = self._submit_best_candidate(arguments)
        else:
            raise KeyError(f"unknown tool: {name}")

        round_index = self.latest_round.round_index if self.latest_round is not None else 0
        self.trace.tool_history.append(
            ToolCallTrace(
                tool_name=name,
                arguments=arguments,
                result_preview=result if isinstance(result, dict) else {"value": result},
                round_index=round_index,
                created_at=utc_now_iso(),
            )
        )
        return result

    def _retrieve_candidates(self, arguments: dict[str, Any]) -> dict[str, Any]:
        params = RetrievalParams(
            video_weight=float(arguments["video_weight"]),
            audio_weight=float(arguments["audio_weight"]),
            object_focus=str(arguments["object_focus"]),
            temporal_focus=str(arguments["temporal_focus"]),
            topk=int(arguments["topk"]),
        )
        current = RoundRecord(
            round_index=len(self.trace.rounds) + 1,
            retrieval_params=params,
            decision="inspecting",
        )
        candidates = self.backend.retrieve_candidates(
            query=self.runtime_query,
            params=params,
            round_index=current.round_index,
        )
        current.retrieved_candidates = [item.candidate_id for item in candidates]
        self.trace.rounds.append(current)
        self.latest_round = current
        return {
            "round_index": current.round_index,
            "params": params.to_dict(),
            "candidates": [item.to_dict() for item in candidates],
        }

    def _inspect_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        current = self._require_round()
        candidate_id = str(arguments["candidate_id"])
        if candidate_id not in current.retrieved_candidates:
            raise RuntimeError("candidate must come from the latest retrieved set")
        if candidate_id not in current.inspected_candidates:
            current.inspected_candidates.append(candidate_id)
        return self.backend.inspect_candidate(candidate_id).to_dict()

    def _compare_to_request(self, arguments: dict[str, Any]) -> dict[str, Any]:
        current = self._require_round()
        candidate_id = str(arguments["candidate_id"])
        if candidate_id not in current.retrieved_candidates:
            raise RuntimeError("candidate must come from the latest retrieved set")
        result = self.backend.compare_to_request(self.runtime_query, candidate_id)
        current.comparisons[candidate_id] = result
        return result.to_dict()

    def _submit_best_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        current = self._require_round()
        candidate_id = str(arguments["candidate_id"])
        explanation = str(arguments["explanation"]).strip()
        current.decision = "submitted"
        current.notes = explanation
        self.trace.final_candidate_id = candidate_id
        self.trace.final_explanation = explanation
        self.finished = True
        return {
            "status": "submitted",
            "candidate_id": candidate_id,
            "accepted": True,
        }

    def finalize(self) -> RunTrace:
        self.trace.query.target_video_id = self.original_query.target_video_id
        if self.trace.final_candidate_id is not None and self.original_query.target_video_id is not None:
            self.trace.success = self.trace.final_candidate_id == self.original_query.target_video_id
        return self.trace

    def _require_round(self) -> RoundRecord:
        if self.latest_round is None:
            raise RuntimeError("retrieve_candidates must be called first")
        return self.latest_round


def run_conversation(
    client: Any,
    env: SimpleToolEnvironment,
    model: str,
    max_turns: int,
    temperature: float,
    max_tokens: int,
) -> RunTrace:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": env.user_prompt()},
    ]

    print("\n" + "=" * 60)
    print(f"QUERY: {env.runtime_query.query_id}")
    print(env.runtime_query.edit_instruction)
    print("=" * 60)

    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn} ---")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=env.tool_schemas(),
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message
        messages.append(_assistant_message_to_dict(message))
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments or "{}")
                print(f"  [Tool Call] {function_name}({function_args})")
                result = env.execute_tool(function_name, function_args)
                preview = result
                if isinstance(result, dict) and "candidates" in result:
                    preview = {
                        "round_index": result.get("round_index"),
                        "candidate_ids": [
                            item["candidate_id"]
                            for item in result.get("candidates", [])[:3]
                        ],
                    }
                print(f"  [Tool Result] {json.dumps(preview, ensure_ascii=False)}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                if env.finished:
                    break
            if env.finished:
                break
        else:
            final_text = message.content or ""
            print("\n  [Assistant]\n" + final_text)
            if env.finished:
                break

    trace = env.finalize()
    print("\n" + "-" * 60)
    print(f"Final candidate: {trace.final_candidate_id}")
    print(f"Target: {trace.query.target_video_id}")
    print(f"Success: {trace.success}")
    print("-" * 60)
    return trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal multi-turn tool-calling demo for retrieval")
    parser.add_argument("--mode", choices=["mock", "real"], required=True)
    parser.add_argument("--config", help="YAML config for real mode")
    parser.add_argument("--query-id", help="Run one query only")
    parser.add_argument("--max-queries", type=int, default=1)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    backend = build_backend(args.mode, args.config)
    queries = [backend.get_query(args.query_id)] if args.query_id else backend.list_queries()[: args.max_queries]
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    for query in queries:
        env = SimpleToolEnvironment(
            backend=backend,
            query=query,
            planner_name=f"simple-chat:{args.model}",
        )
        run_conversation(
            client=client,
            env=env,
            model=args.model,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
