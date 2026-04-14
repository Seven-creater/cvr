from __future__ import annotations

import argparse

from app.bandit_data import BanditRewardConfig, trace_to_bandit_samples, write_bandit_samples
from app.controller import ScriptedController, resolve_scripted_policy
from app.demo import build_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export contextual bandit samples from scripted retrieval rollouts"
    )
    parser.add_argument("--mode", choices=["mock", "real"], required=True)
    parser.add_argument("--config", help="YAML config for real mode")
    parser.add_argument(
        "--controller-profile",
        choices=["adaptive", "fixed", "single-round-fixed"],
        default="adaptive",
    )
    parser.add_argument("--fixed-video-weight", type=float, default=0.7)
    parser.add_argument("--fixed-audio-weight", type=float, default=0.3)
    parser.add_argument("--fixed-object-focus", default="none")
    parser.add_argument("--fixed-temporal-focus", default="global")
    parser.add_argument("--fixed-topk", type=int, default=5)
    parser.add_argument("--query-id", help="Run one query only")
    parser.add_argument("--max-queries", type=int, help="Run at most N queries")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--step-penalty", type=float, default=0.05)
    parser.add_argument("--rank-gain-scale", type=float, default=1.0)
    parser.add_argument("--top1-bonus", type=float, default=1.0)
    parser.add_argument("--top5-bonus", type=float, default=0.5)
    parser.add_argument("--top10-bonus", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = build_backend(args.mode, args.config)
    policy = resolve_scripted_policy(
        profile=args.controller_profile,
        fixed_video_weight=args.fixed_video_weight,
        fixed_audio_weight=args.fixed_audio_weight,
        fixed_object_focus=args.fixed_object_focus,
        fixed_temporal_focus=args.fixed_temporal_focus,
        fixed_topk=args.fixed_topk,
    )
    controller = ScriptedController(backend=backend, policy=policy)
    reward_config = BanditRewardConfig(
        step_penalty=args.step_penalty,
        rank_gain_scale=args.rank_gain_scale,
        top1_bonus=args.top1_bonus,
        top5_bonus=args.top5_bonus,
        top10_bonus=args.top10_bonus,
    )

    queries = [backend.get_query(args.query_id)] if args.query_id else backend.list_queries()
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    all_samples = []
    final_successes = 0
    for query in queries:
        trace = controller.run(query.query_id)
        trace.query.target_video_id = query.target_video_id
        trace.success = (
            trace.final_candidate_id == query.target_video_id
            if query.target_video_id
            else None
        )
        samples = trace_to_bandit_samples(
            backend=backend,
            trace=trace,
            original_query=query,
            reward_config=reward_config,
        )
        all_samples.extend(samples)
        final_successes += int(bool(trace.success))

    output_path = write_bandit_samples(args.output, all_samples)
    avg_reward = (
        sum(sample.reward for sample in all_samples) / len(all_samples)
        if all_samples
        else 0.0
    )
    success_rate = final_successes / len(queries) if queries else 0.0
    print(
        f"[bandit-export] wrote={output_path} queries={len(queries)} "
        f"samples={len(all_samples)} avg_reward={avg_reward:.4f} "
        f"success_rate={success_rate:.4f} policy={policy.name}"
    )


if __name__ == "__main__":
    main()
