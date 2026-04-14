from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from app.artifacts import append_jsonl, write_run_artifacts
from app.backends import FileRetrievalBackend, MockRetrievalBackend
from app.config import load_yaml
from app.controller import ScriptedController, resolve_scripted_policy


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


def build_controller(
    planner: str,
    backend,
    model: str,
    controller_profile: str,
    fixed_video_weight: float,
    fixed_audio_weight: float,
    fixed_object_focus: str,
    fixed_temporal_focus: str,
    fixed_topk: int,
):
    if planner == "openai":
        if not openai_available():
            raise RuntimeError("planner=openai requires the openai package to be installed")
        from app.controller import OpenAIResponsesController

        return OpenAIResponsesController(backend=backend, model=model)
    policy = resolve_scripted_policy(
        profile=controller_profile,
        fixed_video_weight=fixed_video_weight,
        fixed_audio_weight=fixed_audio_weight,
        fixed_object_focus=fixed_object_focus,
        fixed_temporal_focus=fixed_temporal_focus,
        fixed_topk=fixed_topk,
    )
    return ScriptedController(backend=backend, policy=policy)


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
    parser.add_argument(
        "--controller-profile",
        choices=["adaptive", "fixed", "single-round-fixed"],
        default="adaptive",
        help="Scripted controller profile. Ignored when planner=openai.",
    )
    parser.add_argument("--fixed-video-weight", type=float, default=0.7)
    parser.add_argument("--fixed-audio-weight", type=float, default=0.3)
    parser.add_argument("--fixed-object-focus", default="none")
    parser.add_argument("--fixed-temporal-focus", default="global")
    parser.add_argument("--fixed-topk", type=int, default=5)
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
    controller = build_controller(
        planner_name,
        backend,
        args.model,
        args.controller_profile,
        args.fixed_video_weight,
        args.fixed_audio_weight,
        args.fixed_object_focus,
        args.fixed_temporal_focus,
        args.fixed_topk,
    )
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
        trace.query.target_video_id = query.target_video_id
        trace.success = (
            query.target_video_id == trace.final_candidate_id
            if query.target_video_id
            else None
        )
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
