from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.backends.base import RetrievalBackend
from app.schemas import (
    BanditActionRecord,
    BanditSample,
    QueryCase,
    RetrievalCandidate,
    RetrievalParams,
    RoundRecord,
    RunTrace,
)


@dataclass(slots=True)
class BanditRewardConfig:
    step_penalty: float = 0.05
    rank_gain_scale: float = 1.0
    top1_bonus: float = 1.0
    top5_bonus: float = 0.5
    top10_bonus: float = 0.25


def candidate_rank(candidate_ids: list[str], target_video_id: str | None) -> int | None:
    if not target_video_id:
        return None
    try:
        return candidate_ids.index(target_video_id) + 1
    except ValueError:
        return None


def rank_score(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def build_action_menu(query: QueryCase, topk: int) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    def add_retry(action_id: str, description: str, params: RetrievalParams) -> None:
        payload = {
            "action_id": action_id,
            "action_type": "retry",
            "description": description,
            "retrieval_params": params.to_dict(),
        }
        if payload not in actions:
            actions.append(payload)

    add_retry(
        "retry_balanced",
        "Retry with balanced visual/audio fusion.",
        RetrievalParams(0.7, 0.3, "none", "global", topk),
    )
    add_retry(
        "retry_audio_boost",
        "Retry with stronger audio weight.",
        RetrievalParams(0.45, 0.55, "none", "global", topk),
    )
    add_retry(
        "retry_visual_boost",
        "Retry with stronger visual weight.",
        RetrievalParams(0.95, 0.05, "none", "global", topk),
    )

    if query.required_objects:
        add_retry(
            "retry_object_focus",
            f"Retry with object focus on {query.required_objects[0]}.",
            RetrievalParams(0.7, 0.3, query.required_objects[0], "global", topk),
        )

    if query.required_temporal:
        add_retry(
            "retry_temporal_focus",
            f"Retry with temporal focus on {query.required_temporal}.",
            RetrievalParams(0.7, 0.3, "none", query.required_temporal, topk),
        )

    if query.required_audio_tags and query.required_temporal:
        add_retry(
            "retry_audio_temporal",
            "Retry with stronger audio weight and temporal focus.",
            RetrievalParams(0.45, 0.55, "none", query.required_temporal, topk),
        )

    actions.append(
        {
            "action_id": "submit_top1",
            "action_type": "submit",
            "description": "Submit the current top-1 candidate.",
            "candidate_strategy": "top1",
        }
    )
    actions.append(
        {
            "action_id": "submit_best_compared",
            "action_type": "submit",
            "description": "Submit the inspected candidate with the best comparison confidence.",
            "candidate_strategy": "best_compared",
        }
    )
    return actions


def _history_brief(trace: RunTrace, upto_round_index: int) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for round_record in trace.rounds[:upto_round_index]:
        history.append(
            {
                "round_index": round_record.round_index,
                "decision": round_record.decision,
                "retrieval_params": (
                    round_record.retrieval_params.to_dict()
                    if round_record.retrieval_params is not None
                    else None
                ),
                "retrieved_candidates": list(round_record.retrieved_candidates),
                "missing_summary": {
                    candidate_id: list(result.missing)
                    for candidate_id, result in round_record.comparisons.items()
                },
            }
        )
    return history


def _retrieve_snapshot(
    backend: RetrievalBackend,
    query: QueryCase,
    round_record: RoundRecord | None,
    topk: int,
) -> list[dict[str, Any]]:
    if round_record is None or round_record.retrieval_params is None:
        return []
    candidates = backend.retrieve_candidates(
        query=query,
        params=round_record.retrieval_params,
        round_index=round_record.round_index,
    )
    return [item.to_dict() for item in candidates[:topk]]


def build_bandit_state(
    backend: RetrievalBackend,
    trace: RunTrace,
    query: QueryCase,
    round_record: RoundRecord | None,
    *,
    topk: int = 5,
) -> dict[str, Any]:
    source = backend.inspect_candidate(query.source_video_id)
    current_candidates = _retrieve_snapshot(backend, query, round_record, topk)
    inspected_candidates = []
    comparison_feedback: dict[str, Any] = {}

    if round_record is not None:
        for candidate_id in round_record.inspected_candidates:
            inspected_candidates.append(backend.inspect_candidate(candidate_id).to_dict())
        comparison_feedback = {
            candidate_id: result.to_dict()
            for candidate_id, result in round_record.comparisons.items()
        }

    return {
        "query_id": query.query_id,
        "source_video_id": query.source_video_id,
        "target_visible_during_runtime": False,
        "source_summary": source.summary,
        "edit_instruction": query.edit_instruction,
        "required_audio_tags": list(query.required_audio_tags),
        "required_objects": list(query.required_objects),
        "required_temporal": query.required_temporal,
        "current_round_index": round_record.round_index if round_record is not None else 0,
        "history": _history_brief(trace, round_record.round_index - 1 if round_record is not None else 0),
        "current_retrieval_params": (
            round_record.retrieval_params.to_dict()
            if round_record is not None and round_record.retrieval_params is not None
            else None
        ),
        "current_retrieved_candidates": current_candidates,
        "current_inspections": inspected_candidates,
        "current_comparisons": comparison_feedback,
    }


def format_observation_text(state: dict[str, Any], available_actions: list[dict[str, Any]]) -> str:
    current_candidates = state.get("current_retrieved_candidates", [])
    candidate_lines = []
    for item in current_candidates[:3]:
        candidate_lines.append(
            f"- {item['candidate_id']}: score={item['score']:.4f}, "
            f"video={item['video_score']:.4f}, audio={item['audio_score']:.4f}, "
            f"audio_tags={item['audio_tags']}, summary={item['summary']}"
        )
    if not candidate_lines:
        candidate_lines.append("- none")

    comparison_lines = []
    for candidate_id, result in state.get("current_comparisons", {}).items():
        comparison_lines.append(
            f"- {candidate_id}: satisfied={result['satisfied']}, "
            f"missing={result['missing']}, conflicts={result['conflicts']}, "
            f"confidence={result['confidence']}"
        )
    if not comparison_lines:
        comparison_lines.append("- none")

    action_lines = [
        f"- {item['action_id']}: {item['description']}"
        for item in available_actions
    ]

    return (
        f"query_id={state['query_id']}\n"
        f"source_video_id={state['source_video_id']}\n"
        f"source_summary={state['source_summary']}\n"
        f"edit_instruction={state['edit_instruction']}\n"
        f"required_audio_tags={state['required_audio_tags']}\n"
        f"required_objects={state['required_objects']}\n"
        f"required_temporal={state['required_temporal']}\n"
        f"current_round_index={state['current_round_index']}\n"
        f"current_retrieval_params={state['current_retrieval_params']}\n"
        "current_candidates=\n"
        + "\n".join(candidate_lines)
        + "\ncomparison_feedback=\n"
        + "\n".join(comparison_lines)
        + "\navailable_actions=\n"
        + "\n".join(action_lines)
    )


def _best_compared_candidate(round_record: RoundRecord | None) -> str | None:
    if round_record is None or not round_record.comparisons:
        return None
    best_candidate_id = None
    best_key = None
    for candidate_id, result in round_record.comparisons.items():
        key = (
            result.confidence,
            -len(result.missing),
            -len(result.conflicts),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_candidate_id = candidate_id
    return best_candidate_id


def _retry_action_id(action_menu: list[dict[str, Any]], params: RetrievalParams) -> str:
    params_dict = params.to_dict()
    for item in action_menu:
        if item["action_type"] == "retry" and item.get("retrieval_params") == params_dict:
            return item["action_id"]
    return "retry_custom"


def _submit_action(
    round_record: RoundRecord,
    retrieved_candidates: list[dict[str, Any]],
    final_candidate_id: str | None,
) -> tuple[str, str]:
    if final_candidate_id and retrieved_candidates:
        if final_candidate_id == retrieved_candidates[0]["candidate_id"]:
            return "submit_top1", "top1"
        if final_candidate_id == _best_compared_candidate(round_record):
            return "submit_best_compared", "best_compared"
    return "submit_custom", "custom"


def _terminal_bonus(
    final_candidate_id: str | None,
    target_video_id: str | None,
    current_rank: int | None,
    reward_config: BanditRewardConfig,
) -> float:
    if final_candidate_id and target_video_id and final_candidate_id == target_video_id:
        return reward_config.top1_bonus
    if current_rank is not None and current_rank <= 5:
        return reward_config.top5_bonus
    if current_rank is not None and current_rank <= 10:
        return reward_config.top10_bonus
    return 0.0


def trace_to_bandit_samples(
    backend: RetrievalBackend,
    trace: RunTrace,
    original_query: QueryCase,
    reward_config: BanditRewardConfig | None = None,
) -> list[BanditSample]:
    reward_config = reward_config or BanditRewardConfig()
    runtime_query = original_query.without_target()
    samples: list[BanditSample] = []
    default_topk = 5

    if not trace.rounds:
        return samples

    first_round = trace.rounds[0]
    first_candidates = _retrieve_snapshot(backend, runtime_query, first_round, default_topk)
    first_rank = candidate_rank(
        [item["candidate_id"] for item in first_candidates],
        original_query.target_video_id,
    )
    pre_actions = [
        item
        for item in build_action_menu(runtime_query, first_round.retrieval_params.topk)
        if item["action_type"] == "retry"
    ]
    pre_state = build_bandit_state(backend, trace, runtime_query, None, topk=default_topk)
    pre_reward = reward_config.rank_gain_scale * rank_score(first_rank) - reward_config.step_penalty
    pre_action = BanditActionRecord(
        action_id=_retry_action_id(pre_actions, first_round.retrieval_params),
        action_type="retry",
        description="Initial retrieval action.",
        retrieval_params=first_round.retrieval_params.to_dict(),
    )
    samples.append(
        BanditSample(
            query_id=runtime_query.query_id,
            source_video_id=runtime_query.source_video_id,
            round_index=0,
            planner_name=trace.planner_name,
            observation_text=format_observation_text(pre_state, pre_actions),
            state=pre_state,
            available_actions=pre_actions,
            action=pre_action,
            reward=round(pre_reward, 4),
            reward_breakdown={
                "step_penalty": reward_config.step_penalty,
                "prev_target_rank": None,
                "next_target_rank": first_rank,
                "rank_gain": round(rank_score(first_rank), 4),
            },
            done=False,
            final_success=trace.success,
        )
    )

    for index, round_record in enumerate(trace.rounds, start=1):
        retrieved_candidates = _retrieve_snapshot(backend, runtime_query, round_record, default_topk)
        current_rank = candidate_rank(
            [item["candidate_id"] for item in retrieved_candidates],
            original_query.target_video_id,
        )
        action_menu = build_action_menu(runtime_query, round_record.retrieval_params.topk)
        state = build_bandit_state(
            backend,
            trace,
            runtime_query,
            round_record,
            topk=default_topk,
        )

        if round_record.decision == "submitted":
            action_id, candidate_strategy = _submit_action(
                round_record,
                retrieved_candidates,
                trace.final_candidate_id,
            )
            bonus = _terminal_bonus(
                trace.final_candidate_id,
                original_query.target_video_id,
                current_rank,
                reward_config,
            )
            reward = round(bonus, 4)
            sample = BanditSample(
                query_id=runtime_query.query_id,
                source_video_id=runtime_query.source_video_id,
                round_index=index,
                planner_name=trace.planner_name,
                observation_text=format_observation_text(state, action_menu),
                state=state,
                available_actions=action_menu,
                action=BanditActionRecord(
                    action_id=action_id,
                    action_type="submit",
                    description="Submit after inspecting the current retrieval results.",
                    candidate_id=trace.final_candidate_id,
                    candidate_strategy=candidate_strategy,
                    explanation=trace.final_explanation,
                ),
                reward=reward,
                reward_breakdown={
                    "terminal_bonus": reward,
                    "terminal_target_rank": current_rank,
                    "submitted_candidate_id": trace.final_candidate_id,
                    "success": trace.success,
                },
                done=True,
                final_success=trace.success,
            )
            samples.append(sample)
            continue

        next_round = trace.rounds[index]
        next_candidates = _retrieve_snapshot(backend, runtime_query, next_round, default_topk)
        next_rank = candidate_rank(
            [item["candidate_id"] for item in next_candidates],
            original_query.target_video_id,
        )
        gain = reward_config.rank_gain_scale * (
            rank_score(next_rank) - rank_score(current_rank)
        )
        reward = round(gain - reward_config.step_penalty, 4)
        sample = BanditSample(
            query_id=runtime_query.query_id,
            source_video_id=runtime_query.source_video_id,
            round_index=index,
            planner_name=trace.planner_name,
            observation_text=format_observation_text(state, action_menu),
            state=state,
            available_actions=action_menu,
            action=BanditActionRecord(
                action_id=_retry_action_id(action_menu, next_round.retrieval_params),
                action_type="retry",
                description="Retry with adjusted retrieval parameters.",
                retrieval_params=next_round.retrieval_params.to_dict(),
                explanation=round_record.notes,
            ),
            reward=reward,
            reward_breakdown={
                "step_penalty": reward_config.step_penalty,
                "prev_target_rank": current_rank,
                "next_target_rank": next_rank,
                "rank_gain": round(gain, 4),
            },
            done=False,
            final_success=trace.success,
        )
        samples.append(sample)

    return samples


def write_bandit_samples(path: str | Path, samples: list[BanditSample]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return output_path
