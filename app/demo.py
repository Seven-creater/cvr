from __future__ import annotations

import argparse
import os

from app.artifacts import append_jsonl, write_run_artifacts
from app.backends import FileRetrievalBackend, MockRetrievalBackend
from app.config import load_yaml
from app.controller import OpenAIResponsesController, ScriptedController


def build_backend(mode: str, config_path: str | None):
    if mode == "mock":
        return MockRetrievalBackend()
    if mode == "real":
        if not config_path:
            raise ValueError("--config is required in real mode")
        config = load_yaml(config_path)
        return FileRetrievalBackend(
            candidates_path=config["candidates_path"],
            queries_path=config["queries_path"],
            retrieval_scores_path=config.get("retrieval_scores_path"),
        )
    raise ValueError(f"unsupported mode: {mode}")


def build_controller(planner: str, backend, model: str):
    if planner == "openai":
        return OpenAIResponsesController(backend=backend, model=model)
    return ScriptedController(backend=backend)


def resolve_default_planner(explicit: str | None) -> str:
    if explicit:
        return explicit
    return "openai" if os.environ.get("OPENAI_API_KEY") else "scripted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reflective CVR agent demo")
    parser.add_argument("--mode", choices=["mock", "real"], required=True)
    parser.add_argument("--config", help="YAML config for real mode")
    parser.add_argument("--planner", choices=["scripted", "openai"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--query-id", help="Run one query only")
    parser.add_argument("--output-prefix", help="Optional artifact prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planner_name = resolve_default_planner(args.planner)
    backend = build_backend(args.mode, args.config)
    controller = build_controller(planner_name, backend, args.model)
    queries = (
        [backend.get_query(args.query_id)]
        if args.query_id
        else backend.list_queries()
    )

    traces = []
    for query in queries:
        trace = controller.run(query.query_id)
        traces.append(trace)
        paths = write_run_artifacts(
            trace=trace,
            prefix=f"{args.output_prefix}-{query.query_id}" if args.output_prefix else None,
        )
        print(
            f"[done] query={query.query_id} planner={trace.planner_name} "
            f"final={trace.final_candidate_id} success={trace.success} "
            f"json={paths['json']} md={paths['md']}"
        )

    jsonl_path = append_jsonl(traces)
    print(f"[batch] wrote {jsonl_path}")


if __name__ == "__main__":
    main()
