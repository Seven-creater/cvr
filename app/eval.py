from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.avigate_agent import (
    run_official_agent_partial_eval,
    run_t2v_official_agent_case,
    run_v2t_official_agent_case,
)
from app.avigate_official import (
    AvigateRuntimeConfig,
    evaluate_avigate_official,
    load_avigate_runtime,
    retrieve_texts_from_video_official,
    retrieve_videos_from_text_official,
)
from app.omni_checker import OpenAIOmniChecker
from app.retrieval_types import parse_topk_values


def build_avigate_runtime(args: argparse.Namespace):
    config = AvigateRuntimeConfig(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint,
        data_json_path=args.data_json,
        test_csv_path=args.split_csv,
        video_root=args.video_root,
        audio_root=args.audio_root,
        clip_weight_path=args.clip_weight,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size_val=args.batch_size_val,
        max_words=args.max_words,
        max_frames=args.max_frames,
        sim_header=args.sim_header,
        cross_num_hidden_layers=args.cross_num_hidden_layers,
        audio_query_layers=args.audio_query_layers,
        temperature=args.temperature,
    )
    return load_avigate_runtime(config)


def command_avigate_baseline(args: argparse.Namespace) -> None:
    runtime = build_avigate_runtime(args)
    metrics = evaluate_avigate_official(runtime, tuple(parse_topk_values(args.topk)))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_avigate_t2v_case(args: argparse.Namespace) -> None:
    runtime = build_avigate_runtime(args)
    hits = retrieve_videos_from_text_official(args.query_text, runtime, topk=args.topk_value)
    print(
        json.dumps(
            {
                "mode": "t2v",
                "query_text": args.query_text,
                "results": [hit.to_dict() for hit in hits],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_avigate_v2t_case(args: argparse.Namespace) -> None:
    runtime = build_avigate_runtime(args)
    hits = retrieve_texts_from_video_official(args.query_video_id, runtime, topk=args.topk_value)
    print(
        json.dumps(
            {
                "mode": "v2t",
                "query_video_id": args.query_video_id,
                "results": [hit.to_dict() for hit in hits],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_avigate_t2v_agent_case(args: argparse.Namespace) -> None:
    runtime = build_avigate_runtime(args)
    checker = OpenAIOmniChecker(
        base_url=args.checker_base_url,
        api_key=args.checker_api_key,
        model=args.checker_model,
        timeout_seconds=args.checker_timeout_seconds,
    )
    trace = run_t2v_official_agent_case(
        query_text=args.query_text,
        runtime=runtime,
        checker=checker,
        topk=args.topk_value,
        rerank_window=args.rerank_window,
        max_iter=args.max_iter,
        submit_threshold=args.submit_threshold,
    )
    print(json.dumps(trace, ensure_ascii=False, indent=2))


def command_avigate_v2t_agent_case(args: argparse.Namespace) -> None:
    runtime = build_avigate_runtime(args)
    checker = OpenAIOmniChecker(
        base_url=args.checker_base_url,
        api_key=args.checker_api_key,
        model=args.checker_model,
        timeout_seconds=args.checker_timeout_seconds,
    )
    trace = run_v2t_official_agent_case(
        query_video_id=args.query_video_id,
        runtime=runtime,
        checker=checker,
        topk=args.topk_value,
        max_iter=args.max_iter,
        submit_threshold=args.submit_threshold,
    )
    print(json.dumps(trace, ensure_ascii=False, indent=2))


def command_avigate_agent_partial_eval(args: argparse.Namespace) -> None:
    runtime = build_avigate_runtime(args)
    checker = OpenAIOmniChecker(
        base_url=args.checker_base_url,
        api_key=args.checker_api_key,
        model=args.checker_model,
        timeout_seconds=args.checker_timeout_seconds,
    )
    result = run_official_agent_partial_eval(
        mode=args.mode,
        runtime=runtime,
        checker=checker,
        sample_size=args.sample_size,
        start_index=args.start_index,
        topk=args.topk_value,
        max_iter=args.max_iter,
        submit_threshold=args.submit_threshold,
        rerank_window=args.rerank_window,
        recall_ks=tuple(parse_topk_values(args.topk)),
        output_dir=args.output_dir,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def command_avigate_agent_merge(args: argparse.Namespace) -> None:
    input_dirs = [Path(raw_path) for raw_path in args.inputs]
    summaries = []
    trace_lines: list[str] = []
    mode: str | None = None

    for input_dir in input_dirs:
        summary_path = input_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found under {input_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if mode is None:
            mode = str(summary.get("mode"))
        elif str(summary.get("mode")) != mode:
            raise ValueError("all merged summaries must have the same mode")
        summaries.append(summary)

        traces_path = input_dir / "traces.jsonl"
        if traces_path.exists():
            lines = [line for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            trace_lines.extend(lines)

    if not summaries or mode is None:
        raise ValueError("no summaries to merge")

    total_runs = sum(int(summary["runs"]) for summary in summaries)
    if total_runs <= 0:
        raise ValueError("merged run count must be positive")

    merged = {
        "runs": total_runs,
        "round1_recall": _merge_metric_dicts(summaries, "round1_recall", total_runs),
        "final_recall": _merge_metric_dicts(summaries, "final_recall", total_runs),
        "final_top1_accuracy": _merge_scalar_metric(summaries, "final_top1_accuracy", total_runs),
        "avg_omni_calls": _merge_scalar_metric(summaries, "avg_omni_calls", total_runs),
        "audio_off_rate": _merge_scalar_metric(summaries, "audio_off_rate", total_runs),
        "fallback_rate": _merge_scalar_metric(summaries, "fallback_rate", total_runs),
        "mode": mode,
        "input_dirs": [str(path) for path in input_dirs],
    }
    if mode == "t2v":
        merged["query_rewrite_rate"] = _merge_scalar_metric(summaries, "query_rewrite_rate", total_runs)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "traces.jsonl").write_text("\n".join(trace_lines) + ("\n" if trace_lines else ""), encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, indent=2))


def _merge_metric_dicts(summaries: list[dict], key: str, total_runs: int) -> dict:
    metric_names = set()
    for summary in summaries:
        metric_names.update(summary.get(key, {}).keys())
    merged = {}
    for metric_name in sorted(metric_names):
        weighted_total = 0.0
        for summary in summaries:
            runs = int(summary["runs"])
            weighted_total += float(summary.get(key, {}).get(metric_name, 0.0)) * runs
        merged[metric_name] = round(weighted_total / total_runs, 4)
    return merged


def _merge_scalar_metric(summaries: list[dict], key: str, total_runs: int) -> float:
    weighted_total = 0.0
    for summary in summaries:
        runs = int(summary["runs"])
        weighted_total += float(summary.get(key, 0.0)) * runs
    return round(weighted_total / total_runs, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AVIGATE official retrieval evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    avigate_shared = argparse.ArgumentParser(add_help=False)
    avigate_shared.add_argument("--model-dir", required=True)
    avigate_shared.add_argument("--checkpoint", required=True)
    avigate_shared.add_argument("--data-json", required=True)
    avigate_shared.add_argument("--split-csv", required=True)
    avigate_shared.add_argument("--video-root", required=True)
    avigate_shared.add_argument("--audio-root", required=True)
    avigate_shared.add_argument("--clip-weight", required=True)
    avigate_shared.add_argument("--cache-dir")
    avigate_shared.add_argument("--device", default="cuda")
    avigate_shared.add_argument("--batch-size-val", type=int, default=100)
    avigate_shared.add_argument("--max-words", type=int, default=32)
    avigate_shared.add_argument("--max-frames", type=int, default=12)
    avigate_shared.add_argument("--sim-header", default="seqTransf")
    avigate_shared.add_argument("--cross-num-hidden-layers", type=int, default=4)
    avigate_shared.add_argument("--audio-query-layers", type=int, default=4)
    avigate_shared.add_argument("--temperature", type=float, default=1.0)

    avigate_baseline = subparsers.add_parser("avigate-baseline", parents=[avigate_shared])
    avigate_baseline.add_argument("--topk", default="1,5,10")

    avigate_t2v_case = subparsers.add_parser("avigate-t2v-case", parents=[avigate_shared])
    avigate_t2v_case.add_argument("--query-text", required=True)
    avigate_t2v_case.add_argument("--topk-value", type=int, default=10)

    avigate_v2t_case = subparsers.add_parser("avigate-v2t-case", parents=[avigate_shared])
    avigate_v2t_case.add_argument("--query-video-id", required=True)
    avigate_v2t_case.add_argument("--topk-value", type=int, default=10)

    agent_shared = argparse.ArgumentParser(add_help=False, parents=[avigate_shared])
    agent_shared.add_argument("--topk-value", type=int, default=10)
    agent_shared.add_argument("--rerank-window", type=int, default=5)
    agent_shared.add_argument("--max-iter", type=int, default=3)
    agent_shared.add_argument("--submit-threshold", type=float, default=0.7)
    agent_shared.add_argument("--checker-base-url", required=True)
    agent_shared.add_argument("--checker-api-key", required=True)
    agent_shared.add_argument("--checker-model", required=True)
    agent_shared.add_argument("--checker-timeout-seconds", type=float, default=180.0)

    avigate_t2v_agent_case = subparsers.add_parser("avigate-t2v-agent-case", parents=[agent_shared])
    avigate_t2v_agent_case.add_argument("--query-text", required=True)

    avigate_v2t_agent_case = subparsers.add_parser("avigate-v2t-agent-case", parents=[agent_shared])
    avigate_v2t_agent_case.add_argument("--query-video-id", required=True)

    avigate_agent_partial_eval = subparsers.add_parser("avigate-agent-partial-eval", parents=[agent_shared])
    avigate_agent_partial_eval.add_argument("--mode", required=True, choices=("t2v", "v2t"))
    avigate_agent_partial_eval.add_argument("--start-index", type=int, default=0)
    avigate_agent_partial_eval.add_argument("--sample-size", type=int, required=True)
    avigate_agent_partial_eval.add_argument("--output-dir", required=True)
    avigate_agent_partial_eval.add_argument("--topk", default="1,5,10")

    avigate_agent_merge = subparsers.add_parser("avigate-agent-merge")
    avigate_agent_merge.add_argument("--output-dir", required=True)
    avigate_agent_merge.add_argument("inputs", nargs="+")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "avigate-baseline":
        command_avigate_baseline(args)
        return
    if args.command == "avigate-t2v-case":
        command_avigate_t2v_case(args)
        return
    if args.command == "avigate-t2v-agent-case":
        command_avigate_t2v_agent_case(args)
        return
    if args.command == "avigate-v2t-agent-case":
        command_avigate_v2t_agent_case(args)
        return
    if args.command == "avigate-agent-partial-eval":
        command_avigate_agent_partial_eval(args)
        return
    if args.command == "avigate-agent-merge":
        command_avigate_agent_merge(args)
        return
    command_avigate_v2t_case(args)


if __name__ == "__main__":
    main()
