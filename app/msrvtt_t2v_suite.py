from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.artifacts import ensure_runs_dir
from app.backends.base import overlap_score
from app.controller.scripted import resolve_scripted_policy
from app.prepare_msrvtt_replay import (
    AUDIO_LEXICON,
    OBJECT_LEXICON,
    SCENE_LEXICON,
    TEMPORAL_LEXICON,
    build_candidate_rows,
    detect_tags,
    load_json,
    load_split_ids,
    tokenize,
)
from app.schemas import CandidateMetadata, RetrievalParams, TextQueryCase, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standard MSR-VTT text-to-video suite")
    parser.add_argument("--msrvtt-json", required=True, help="Path to MSRVTT_data.json")
    parser.add_argument("--split-csv", required=True, help="MSRVTT_JSFUSION_test.csv path")
    parser.add_argument(
        "--profiles",
        default="single-round-fixed,fixed,adaptive",
        help="Comma-separated scripted controller profiles",
    )
    parser.add_argument("--fixed-video-weight", type=float, default=0.7)
    parser.add_argument("--fixed-audio-weight", type=float, default=0.3)
    parser.add_argument("--fixed-object-focus", default="none")
    parser.add_argument("--fixed-temporal-focus", default="global")
    parser.add_argument("--fixed-topk", type=int, default=10)
    parser.add_argument("--max-queries", type=int, help="Optional cap on text queries")
    parser.add_argument("--output-dir", help="Optional output directory")
    return parser.parse_args()


def build_text_queries(
    msrvtt_json_path: str | Path,
    split_csv_path: str | Path,
    max_queries: int | None = None,
) -> list[TextQueryCase]:
    raw = load_json(msrvtt_json_path)
    split_ids = load_split_ids(split_csv_path)
    queries: list[TextQueryCase] = []

    for index, item in enumerate(raw.get("sentences", [])):
        video_id = item.get("video_id")
        caption = (item.get("caption") or item.get("text") or "").strip()
        if not video_id or not caption:
            continue
        if split_ids is not None and video_id not in split_ids:
            continue

        temporal_tags = detect_tags(caption, TEMPORAL_LEXICON)
        queries.append(
            TextQueryCase(
                query_id=f"t2v_{index:06d}",
                text=caption,
                target_video_id=video_id,
                required_audio_tags=detect_tags(caption, AUDIO_LEXICON),
                required_objects=detect_tags(caption, OBJECT_LEXICON),
                required_temporal=temporal_tags[0] if temporal_tags else None,
                scene_tags=detect_tags(caption, SCENE_LEXICON),
                notes="standard MSR-VTT caption query",
            )
        )
        if max_queries is not None and len(queries) >= max_queries:
            break

    return queries


def query_type(query: TextQueryCase) -> str:
    active = [
        name
        for name, flag in (
            ("audio", bool(query.required_audio_tags)),
            ("object", bool(query.required_objects)),
            ("temporal", bool(query.required_temporal)),
        )
        if flag
    ]
    if not active:
        return "other"
    if len(active) == 1:
        return active[0]
    return "+".join(active)


def compare_candidate(query: TextQueryCase, candidate: CandidateMetadata) -> dict[str, Any]:
    satisfied: list[str] = []
    missing: list[str] = []

    token_overlap = len(
        tokenize(query.text) & tokenize(candidate.summary + " " + candidate.caption + " " + candidate.asr)
    ) / max(1, len(tokenize(query.text)))
    scene_overlap = overlap_score(query.scene_tags, candidate.scene_tags) if query.scene_tags else 0.0
    object_overlap = overlap_score(query.required_objects, candidate.visual_objects)
    audio_overlap = overlap_score(query.required_audio_tags, candidate.audio_tags)
    temporal_overlap = (
        1.0
        if query.required_temporal and query.required_temporal in candidate.temporal_tags
        else 0.0
    ) if query.required_temporal else 0.0

    if query.required_audio_tags:
        if audio_overlap >= 1.0:
            satisfied.append("required-audio")
        else:
            missing.append("required-audio")
    if query.required_objects:
        if object_overlap >= 1.0:
            satisfied.append("required-object")
        else:
            missing.append("required-object")
    if query.required_temporal:
        if temporal_overlap >= 1.0:
            satisfied.append("required-temporal")
        else:
            missing.append("required-temporal")
    if query.scene_tags:
        if scene_overlap >= 0.5:
            satisfied.append("scene-match")
        else:
            missing.append("scene-match")

    confidence = (
        0.50 * token_overlap
        + 0.15 * scene_overlap
        + 0.15 * object_overlap
        + 0.15 * audio_overlap
        + 0.05 * temporal_overlap
    )
    confidence = max(0.0, min(1.0, confidence))
    return {
        "candidate_id": candidate.video_id,
        "satisfied": satisfied,
        "missing": missing,
        "conflicts": [],
        "confidence": round(confidence, 4),
    }


def retrieve_candidates(
    query: TextQueryCase,
    candidates: list[CandidateMetadata],
    params: RetrievalParams,
) -> list[dict[str, Any]]:
    instruction_tokens = tokenize(query.text)
    results: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_tokens = tokenize(candidate.summary + " " + candidate.caption + " " + candidate.asr)
        token_overlap = len(instruction_tokens & candidate_tokens) / max(1, len(instruction_tokens))
        scene_overlap = overlap_score(query.scene_tags, candidate.scene_tags) if query.scene_tags else 0.0
        object_overlap = overlap_score(query.required_objects, candidate.visual_objects)
        audio_overlap = overlap_score(query.required_audio_tags, candidate.audio_tags)
        temporal_overlap = (
            1.0
            if query.required_temporal and query.required_temporal in candidate.temporal_tags
            else 0.0
        ) if query.required_temporal else 0.0

        video_score = (
            0.60 * token_overlap
            + 0.20 * scene_overlap
            + 0.15 * object_overlap
            + 0.05 * temporal_overlap
        )
        if params.object_focus != "none":
            video_score += 0.20 * float(params.object_focus in candidate.visual_objects)
        if params.temporal_focus != "global":
            video_score += 0.20 * float(params.temporal_focus in candidate.temporal_tags)

        combined = params.video_weight * video_score + params.audio_weight * (
            0.70 * audio_overlap + 0.30 * token_overlap
        )
        results.append(
            {
                "candidate_id": candidate.video_id,
                "score": round(combined, 4),
                "video_score": round(video_score, 4),
                "audio_score": round(audio_overlap, 4),
                "summary": candidate.summary,
                "audio_tags": list(candidate.audio_tags),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def target_rank(ranked_candidates: list[dict[str, Any]], target_video_id: str) -> int | None:
    for index, item in enumerate(ranked_candidates, start=1):
        if item["candidate_id"] == target_video_id:
            return index
    return None


def choose_params(query: TextQueryCase, profile: str, fixed_params: RetrievalParams, round_index: int) -> RetrievalParams:
    if profile != "adaptive":
        return RetrievalParams(
            video_weight=fixed_params.video_weight,
            audio_weight=fixed_params.audio_weight,
            object_focus=fixed_params.object_focus,
            temporal_focus=fixed_params.temporal_focus,
            topk=fixed_params.topk,
        )

    video_weight = 0.7
    audio_weight = 0.3
    object_focus = "none"
    temporal_focus = "global"

    if query.required_audio_tags and round_index > 1:
        video_weight = 0.45
        audio_weight = 0.55
    if query.required_objects and round_index > 1:
        object_focus = query.required_objects[0]
    if query.required_temporal and round_index > 1:
        temporal_focus = query.required_temporal

    return RetrievalParams(
        video_weight=video_weight,
        audio_weight=audio_weight,
        object_focus=object_focus,
        temporal_focus=temporal_focus,
        topk=fixed_params.topk,
    )


def rank_recall(rows: list[dict[str, Any]], rank_key: str, ks: list[int]) -> dict[str, float]:
    total = len(rows)
    values: dict[str, float] = {}
    for k in ks:
        hits = 0
        for row in rows:
            rank = row.get(rank_key)
            if rank is not None and rank <= k:
                hits += 1
        values[f"R@{k}"] = round(hits / max(1, total), 3)
    return values


def run_query(
    query: TextQueryCase,
    candidates: list[CandidateMetadata],
    profile: str,
    fixed_params: RetrievalParams,
    max_rounds: int,
) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    tool_calls = 0
    final_ranking: list[dict[str, Any]] = []
    final_round_index = 0

    for round_index in range(1, max_rounds + 1):
        params = choose_params(query, profile, fixed_params, round_index)
        ranking = retrieve_candidates(query, candidates, params)
        final_ranking = ranking
        final_round_index = round_index
        top_candidates = ranking[:2]
        comparisons = {
            item["candidate_id"]: compare_candidate(
                query,
                next(candidate for candidate in candidates if candidate.video_id == item["candidate_id"]),
            )
            for item in top_candidates
        }
        tool_calls += 1 + len(top_candidates) * 2
        top1_compare = comparisons[top_candidates[0]["candidate_id"]] if top_candidates else {"missing": ["empty"]}
        decision = "submit"
        notes = "top-1 satisfies query constraints"
        if round_index < max_rounds and top1_compare.get("missing"):
            decision = "retry"
            notes = f"retry because top-1 missing={top1_compare.get('missing', [])}"

        rounds.append(
            {
                "round_index": round_index,
                "retrieval_params": params.to_dict(),
                "retrieved_candidates": [item["candidate_id"] for item in ranking],
                "top_candidates": top_candidates,
                "comparisons": comparisons,
                "decision": decision,
                "notes": notes,
            }
        )
        if decision == "submit":
            break

    first_rank = target_rank(
        [{"candidate_id": candidate_id} for candidate_id in rounds[0]["retrieved_candidates"]],
        query.target_video_id,
    )
    final_rank = target_rank(
        [{"candidate_id": candidate_id} for candidate_id in rounds[-1]["retrieved_candidates"]],
        query.target_video_id,
    )
    return {
        "query": query.to_dict(),
        "planner_name": f"scripted:{profile}",
        "planner_metadata": {
            "profile": profile,
            "target_visible_during_runtime": False,
            "success_computed_offline": True,
            "standard_protocol": True,
        },
        "rounds": rounds,
        "tool_history_count": tool_calls,
        "first_round_target_rank": first_rank,
        "final_round_target_rank": final_rank,
        "success": final_rank == 1,
        "created_at": utc_now_iso(),
    }


def summarize_profile(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    total = len(rows)
    success_rate = sum(1 for row in rows if row.get("success") is True) / max(1, total)
    avg_rounds = sum(len(row.get("rounds", [])) for row in rows) / max(1, total)
    avg_tool_calls = sum(row.get("tool_history_count", 0) for row in rows) / max(1, total)
    first_round_recall = rank_recall(rows, "first_round_target_rank", ks)
    final_round_recall = rank_recall(rows, "final_round_target_rank", ks)

    type_buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = query_type(TextQueryCase(**row["query"]))
        type_buckets.setdefault(bucket, []).append(row)

    type_breakdown = {
        bucket: {
            "count": len(items),
            "success_rate": round(
                sum(1 for item in items if item.get("success") is True) / max(1, len(items)),
                3,
            ),
        }
        for bucket, items in sorted(type_buckets.items())
    }

    return {
        "runs": total,
        "success_rate": round(success_rate, 3),
        "avg_rounds": round(avg_rounds, 2),
        "avg_tool_calls": round(avg_tool_calls, 2),
        "first_round_top1": first_round_recall.get("R@1", 0.0),
        "final_round_top1": final_round_recall.get("R@1", 0.0),
        "first_round_recall": first_round_recall,
        "final_round_recall": final_round_recall,
        "type_breakdown": type_breakdown,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Standard MSR-VTT T2V Suite",
        "",
        "## Data Audit",
        "",
    ]
    for key, value in summary["data_audit"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Profiles", ""])
    for profile in summary["profiles"]:
        metrics = profile["metrics"]
        lines.extend(
            [
                f"### {profile['profile']}",
                "",
                f"- planner: `{profile['planner_name']}`",
                f"- runs: `{metrics['runs']}`",
                f"- success_rate: `{metrics['success_rate']}`",
                f"- avg_rounds: `{metrics['avg_rounds']}`",
                f"- avg_tool_calls: `{metrics['avg_tool_calls']}`",
                f"- first_round_recall: `{metrics['first_round_recall']}`",
                f"- final_round_recall: `{metrics['final_round_recall']}`",
                f"- type_breakdown: `{metrics['type_breakdown']}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    candidates = build_candidate_rows(args.msrvtt_json, args.split_csv)
    queries = build_text_queries(args.msrvtt_json, args.split_csv, args.max_queries)
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    output_dir = Path(args.output_dir) if args.output_dir else ensure_runs_dir() / f"msrvtt-t2v-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_policy = resolve_scripted_policy(
        profile="fixed",
        fixed_video_weight=args.fixed_video_weight,
        fixed_audio_weight=args.fixed_audio_weight,
        fixed_object_focus=args.fixed_object_focus,
        fixed_temporal_focus=args.fixed_temporal_focus,
        fixed_topk=max(10, args.fixed_topk),
    )
    ks = [1, 5, 10]

    profile_summaries = []
    for profile in profiles:
        policy = resolve_scripted_policy(
            profile=profile,
            fixed_video_weight=args.fixed_video_weight,
            fixed_audio_weight=args.fixed_audio_weight,
            fixed_object_focus=args.fixed_object_focus,
            fixed_temporal_focus=args.fixed_temporal_focus,
            fixed_topk=max(10, args.fixed_topk),
        )
        rows = [
            run_query(
                query=query,
                candidates=candidates,
                profile=profile,
                fixed_params=policy.fixed_params,
                max_rounds=policy.max_rounds,
            )
            for query in queries
        ]
        jsonl_path = output_dir / f"{profile}.jsonl"
        write_jsonl(jsonl_path, rows)
        profile_summaries.append(
            {
                "profile": profile,
                "planner_name": f"scripted:{profile}",
                "planner_metadata": {
                    "profile": profile,
                    "target_visible_during_runtime": False,
                    "success_computed_offline": True,
                    "standard_protocol": True,
                },
                "batch_jsonl": str(jsonl_path),
                "metrics": summarize_profile(rows, ks),
                "runtime_failures": [],
            }
        )

    summary = {
        "protocol": "standard-msrvtt-t2v",
        "data_audit": {
            "dataset": "MSR-VTT",
            "split_csv": str(Path(args.split_csv).resolve()),
            "candidate_count": len(candidates),
            "query_count": len(queries),
            "uses_target_aware_filtering": False,
            "uses_rollout_filtering": False,
            "uses_heuristic_retrieval_scores": True,
            "query_selection_mode": "standard-caption-queries",
            "selection_strategy": "all-captions-from-split",
        },
        "profiles": profile_summaries,
    }

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
