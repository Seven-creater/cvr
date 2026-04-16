from __future__ import annotations

import argparse
import json
import sys

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
        topk=args.topk_value,
        max_iter=args.max_iter,
        submit_threshold=args.submit_threshold,
        recall_ks=tuple(parse_topk_values(args.topk)),
        output_dir=args.output_dir,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


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
    avigate_agent_partial_eval.add_argument("--sample-size", type=int, required=True)
    avigate_agent_partial_eval.add_argument("--output-dir", required=True)
    avigate_agent_partial_eval.add_argument("--topk", default="1,5,10")
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
    command_avigate_v2t_case(args)


if __name__ == "__main__":
    main()
