from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from app.artifacts import append_jsonl, write_run_artifacts
from app.backends import FileRetrievalBackend, MockRetrievalBackend
from app.config import load_yaml
from app.controller import ScriptedController


def resolve_config_path(config_path: str, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)

    config_dir = Path(config_path).resolve().parent
    direct_candidate = (config_dir / path).resolve()
    if direct_candidate.exists():
        return str(direct_candidate)

    project_candidate = (config_dir.parent / path).resolve()
    return str(project_candidate)


def build_backend(mode: str, config_path: str | None):
    if mode == "mock":
        return MockRetrievalBackend()
    if mode == "real":
        if not config_path:
            raise ValueError("--config is required in real mode")
        config = load_yaml(config_path)
        return FileRetrievalBackend(
            candidates_path=resolve_config_path(config_path, config["candidates_path"]),
            queries_path=resolve_config_path(config_path, config["queries_path"]),
            retrieval_scores_path=resolve_config_path(config_path, config.get("retrieval_scores_path")),
        )
    raise ValueError(f"unsupported mode: {mode}")


def openai_available() -> bool:
    return importlib.util.find_spec("openai") is not None


def build_controller(planner: str, backend, model: str):
    if planner == "openai":
        if not openai_available():
            raise RuntimeError("planner=openai requires the openai package to be installed")
        from app.controller import OpenAIResponsesController

        return OpenAIResponsesController(backend=backend, model=model)
    return ScriptedController(backend=backend)


def resolve_default_planner(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("OPENAI_API_KEY") and openai_available():
        return "openai"
    return "scripted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reflective CVR agent demo")
    parser.add_argument("--mode", choices=["mock", "real"], required=True)
    parser.add_argument("--config", help="YAML config for real mode")
    parser.add_argument("--planner", choices=["scripted", "openai"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--query-id", help="Run one query only")
    parser.add_argument("--max-queries", type=int, help="Run at most N queries")
    parser.add_argument("--jsonl-output", help="Optional output path for the batch jsonl")
    parser.add_argument("--output-prefix", help="Optional artifact prefix")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining queries if one query fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planner_name = resolve_default_planner(args.planner)
    backend = build_backend(args.mode, args.config)
    controller = build_controller(planner_name, backend, args.model)
    queries = [backend.get_query(args.query_id)] if args.query_id else backend.list_queries()
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    traces = []
    failures: list[tuple[str, str]] = []
    for query in queries:
        try:
            trace = controller.run(query.query_id)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((query.query_id, str(exc)))
            print(f"[error] query={query.query_id} message={exc}")
            continue
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

    jsonl_path = append_jsonl(traces, path=args.jsonl_output)
    print(f"[batch] wrote {jsonl_path}")
    if failures:
        print(f"[summary] failures={len(failures)}")
        for query_id, message in failures[:10]:
            print(f"  - {query_id}: {message}")


if __name__ == "__main__":
    main()
