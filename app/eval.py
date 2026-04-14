from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agent_loop import RetrievalParams, run_agent_case, summarize_agent_traces
from app.omni_checker import MockOmniChecker, OpenAIOmniChecker
from app.retriever import FeatureRetriever, build_text_encoder, parse_topk_values


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_checker(args: argparse.Namespace):
    if args.checker_kind == "mock":
        return MockOmniChecker()
    return OpenAIOmniChecker(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.omni_model,
        timeout_seconds=args.timeout_seconds,
    )


def build_retriever(args: argparse.Namespace, *, for_agent: bool) -> FeatureRetriever:
    encoder = None
    if for_agent:
        encoder = build_text_encoder(
            args.text_encoder_kind,
            dim=args.text_encoder_dim,
            command=args.text_encoder_command,
        )
    return FeatureRetriever.from_feature_dir(
        feature_dir=args.feature_dir,
        split_csv_path=args.split_csv,
        text_encoder=encoder,
    )


def command_baseline(args: argparse.Namespace) -> None:
    retriever = build_retriever(args, for_agent=False)
    metrics = retriever.evaluate_bidirectional(
        parse_topk_values(args.topk),
        alpha_visual=args.alpha_visual,
        alpha_audio=args.alpha_audio,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_agent_case(args: argparse.Namespace) -> None:
    retriever = build_retriever(args, for_agent=(args.mode == "t2v"))
    checker = build_checker(args)
    trace = run_agent_case(
        mode=args.mode,
        retriever=retriever,
        checker=checker,
        query_text=args.query_text,
        query_video_id=args.query_video_id,
        target_video_id=args.target_video_id,
        target_text_ids=args.target_text_ids.split(",") if args.target_text_ids else None,
        initial_params=RetrievalParams(
            alpha_visual=args.alpha_visual,
            alpha_audio=args.alpha_audio,
            topk=args.topk_value,
        ),
        max_rounds=args.max_rounds,
    )
    print(json.dumps(trace, ensure_ascii=False, indent=2))


def command_agent_batch(args: argparse.Namespace) -> None:
    retriever = build_retriever(args, for_agent=(args.mode == "t2v"))
    checker = build_checker(args)
    traces: list[dict] = []
    initial = RetrievalParams(
        alpha_visual=args.alpha_visual,
        alpha_audio=args.alpha_audio,
        topk=args.topk_value,
    )
    if args.mode == "t2v":
        rows = retriever.text_rows[: args.limit] if args.limit else retriever.text_rows
        for row in rows:
            traces.append(
                run_agent_case(
                    mode="t2v",
                    retriever=retriever,
                    checker=checker,
                    query_text=row.text,
                    target_video_id=row.video_id,
                    initial_params=initial,
                    max_rounds=args.max_rounds,
                )
            )
    else:
        rows = retriever.video_rows[: args.limit] if args.limit else retriever.video_rows
        for row in rows:
            traces.append(
                run_agent_case(
                    mode="v2t",
                    retriever=retriever,
                    checker=checker,
                    query_video_id=row.video_id,
                    target_text_ids=retriever.target_text_ids(row.video_id),
                    initial_params=initial,
                    max_rounds=args.max_rounds,
                )
            )
    summary = summarize_agent_traces(traces, parse_topk_values(args.report_topk))
    summary["mode"] = args.mode
    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "traces.jsonl", traces)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_json={output_dir / 'summary.json'}")
    print(f"traces_jsonl={output_dir / 'traces.jsonl'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature retriever baseline and loop agent evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--feature-dir", required=True)
    baseline.add_argument("--split-csv")
    baseline.add_argument("--topk", default="1,5,10")
    baseline.add_argument("--alpha-visual", type=float, default=0.8)
    baseline.add_argument("--alpha-audio", type=float, default=0.2)

    for name in ("agent-case", "agent-batch"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--mode", choices=("t2v", "v2t"), required=True)
        sub.add_argument("--feature-dir", required=True)
        sub.add_argument("--split-csv")
        sub.add_argument("--alpha-visual", type=float, default=0.8)
        sub.add_argument("--alpha-audio", type=float, default=0.2)
        sub.add_argument("--topk-value", type=int, choices=(5, 10), default=10)
        sub.add_argument("--max-rounds", type=int, default=3)
        sub.add_argument("--checker-kind", choices=("mock", "openai"), default="mock")
        sub.add_argument("--base-url", default="http://localhost:8091/v1")
        sub.add_argument("--api-key", default="EMPTY")
        sub.add_argument("--omni-model", default="Qwen2.5-Omni-7B")
        sub.add_argument("--timeout-seconds", type=float, default=120.0)
        sub.add_argument("--text-encoder-kind", choices=("none", "hashing", "subprocess"), default="none")
        sub.add_argument("--text-encoder-dim", type=int, default=8)
        sub.add_argument("--text-encoder-command")

    agent_case = subparsers.choices["agent-case"]
    agent_case.add_argument("--query-text")
    agent_case.add_argument("--query-video-id")
    agent_case.add_argument("--target-video-id")
    agent_case.add_argument("--target-text-ids")

    agent_batch = subparsers.choices["agent-batch"]
    agent_batch.add_argument("--limit", type=int)
    agent_batch.add_argument("--report-topk", default="1,5,10")
    agent_batch.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "baseline":
        command_baseline(args)
        return
    if args.command == "agent-case":
        command_agent_case(args)
        return
    command_agent_batch(args)


if __name__ == "__main__":
    main()
