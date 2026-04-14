from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.backends.base import heuristic_compare, overlap_score
from app.schemas import CandidateMetadata, QueryCase

AUDIO_LEXICON = {
    "music": ("music", "musician", "song", "sing", "singing", "band", "concert"),
    "guitar": ("guitar",),
    "piano": ("piano",),
    "drums": ("drum", "drums", "drummer"),
    "speech": ("talk", "talking", "speaking", "speech", "speaks", "says"),
    "laughter": ("laugh", "laughing", "giggle"),
    "cheering": ("cheer", "cheering", "cheers", "crowd"),
    "applause": ("applause", "clap", "clapping"),
    "engine": ("engine", "motor", "car", "truck", "vehicle"),
    "barking": ("bark", "barking"),
}

OBJECT_LEXICON = {
    "dog": ("dog", "puppy"),
    "cat": ("cat", "kitten"),
    "car": ("car", "vehicle", "automobile"),
    "truck": ("truck",),
    "bicycle": ("bicycle", "bike", "cyclist"),
    "guitar": ("guitar",),
    "piano": ("piano",),
    "ball": ("ball", "football", "soccer", "basketball"),
    "horse": ("horse",),
    "boat": ("boat", "ship"),
    "bird": ("bird",),
    "person": ("man", "woman", "person", "people", "boy", "girl", "child"),
}

SCENE_LEXICON = {
    "park": ("park", "playground", "field"),
    "street": ("street", "road", "sidewalk", "traffic"),
    "indoor": ("indoor", "inside", "room", "house", "home"),
    "outdoor": ("outdoor", "outside"),
    "kitchen": ("kitchen",),
    "concert": ("concert", "stage", "performance", "band", "musician"),
    "sports": ("soccer", "basketball", "football", "stadium", "game", "sport"),
    "water": ("water", "river", "lake", "ocean", "beach", "pool"),
    "animal": ("dog", "cat", "horse", "bird"),
}

TEMPORAL_LEXICON = {
    "opening": ("opening", "start", "starts", "begin", "beginning", "first"),
    "middle": ("middle", "then", "afterward"),
    "late": ("late", "later", "end", "ending", "finally"),
}

def tokenize(text: str) -> set[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {token for token in cleaned.split() if token}


def detect_tags(text: str, lexicon: dict[str, tuple[str, ...]]) -> list[str]:
    tokens = tokenize(text)
    tags = []
    for tag, triggers in lexicon.items():
        if any(trigger in tokens for trigger in triggers):
            tags.append(tag)
    return sorted(tags)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_split_ids(path: str | Path | None) -> set[str] | None:
    if not path:
        return None

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return set()

    header = [cell.strip().lower() for cell in rows[0]]
    has_header = "video_id" in header or "id" in header
    ids: set[str] = set()
    if has_header:
        key_index = header.index("video_id") if "video_id" in header else header.index("id")
        for row in rows[1:]:
            if row and len(row) > key_index and row[key_index].strip():
                ids.add(row[key_index].strip())
        return ids

    for row in rows:
        if row and row[0].strip():
            ids.add(row[0].strip())
    return ids


@dataclass(slots=True)
class ReplayPack:
    candidates: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    retrieval_scores: dict[str, dict[str, dict[str, Any]]]
    config: dict[str, Any]
    stats: dict[str, Any]


@dataclass(slots=True)
class QueryDiscriminability:
    accepted: bool
    target_rank: int
    target_score: float
    second_score: float
    score_margin: float
    strong_match_count: int
    round1_target_rank: int
    round2_target_rank: int
    round3_target_rank: int
    retry_favorable: bool
    agent_success: bool
    retry_rounds: int
    target_visible_rounds: int
    final_candidate_id: str | None
    hardness: float


@dataclass(slots=True)
class FilterPolicy:
    name: str
    min_target_margin: float
    max_strong_matches: int
    generation_rank_cutoff: int
    max_target_deficit: float
    round1_rank_cutoff: int
    round2_rank_cutoff: int
    round3_rank_cutoff: int
    max_source_uses: int
    max_target_uses: int
    prefer_retry_candidates: bool
    prefer_agent_failure_candidates: bool
    balance_outcomes: bool


@dataclass(slots=True)
class AcceptedQueryRecord:
    selection_priority: float
    query_type: str
    source: CandidateMetadata
    target: CandidateMetadata
    value: str
    discriminability: QueryDiscriminability
    outcome_bucket: str


SELECTION_MODES = ("target-aware", "unbiased")


FILTER_PRESETS: dict[str, FilterPolicy] = {
    "loose": FilterPolicy(
        name="loose",
        min_target_margin=0.02,
        max_strong_matches=3,
        generation_rank_cutoff=1,
        max_target_deficit=0.0,
        round1_rank_cutoff=1,
        round2_rank_cutoff=1,
        round3_rank_cutoff=1,
        max_source_uses=8,
        max_target_uses=8,
        prefer_retry_candidates=False,
        prefer_agent_failure_candidates=False,
        balance_outcomes=False,
    ),
    "medium-hard": FilterPolicy(
        name="medium-hard",
        min_target_margin=0.04,
        max_strong_matches=2,
        generation_rank_cutoff=1,
        max_target_deficit=0.0,
        round1_rank_cutoff=1,
        round2_rank_cutoff=1,
        round3_rank_cutoff=1,
        max_source_uses=5,
        max_target_uses=5,
        prefer_retry_candidates=False,
        prefer_agent_failure_candidates=False,
        balance_outcomes=False,
    ),
    "strict": FilterPolicy(
        name="strict",
        min_target_margin=0.06,
        max_strong_matches=1,
        generation_rank_cutoff=1,
        max_target_deficit=0.0,
        round1_rank_cutoff=1,
        round2_rank_cutoff=1,
        round3_rank_cutoff=1,
        max_source_uses=4,
        max_target_uses=4,
        prefer_retry_candidates=False,
        prefer_agent_failure_candidates=False,
        balance_outcomes=False,
    ),
    "hard": FilterPolicy(
        name="hard",
        min_target_margin=0.01,
        max_strong_matches=4,
        generation_rank_cutoff=2,
        max_target_deficit=0.03,
        round1_rank_cutoff=2,
        round2_rank_cutoff=1,
        round3_rank_cutoff=1,
        max_source_uses=3,
        max_target_uses=3,
        prefer_retry_candidates=True,
        prefer_agent_failure_candidates=False,
        balance_outcomes=False,
    ),
    "balanced-hard": FilterPolicy(
        name="balanced-hard",
        min_target_margin=0.0,
        max_strong_matches=5,
        generation_rank_cutoff=4,
        max_target_deficit=0.12,
        round1_rank_cutoff=5,
        round2_rank_cutoff=5,
        round3_rank_cutoff=5,
        max_source_uses=4,
        max_target_uses=4,
        prefer_retry_candidates=True,
        prefer_agent_failure_candidates=False,
        balance_outcomes=True,
    ),
    "agent-hard": FilterPolicy(
        name="agent-hard",
        min_target_margin=0.0,
        max_strong_matches=5,
        generation_rank_cutoff=3,
        max_target_deficit=0.15,
        round1_rank_cutoff=5,
        round2_rank_cutoff=5,
        round3_rank_cutoff=5,
        max_source_uses=3,
        max_target_uses=3,
        prefer_retry_candidates=True,
        prefer_agent_failure_candidates=True,
        balance_outcomes=False,
    ),
}


def resolve_filter_policy(
    difficulty_preset: str | None,
    min_target_margin: float,
    max_strong_matches: int,
) -> FilterPolicy:
    if difficulty_preset:
        preset = FILTER_PRESETS[difficulty_preset]
        return FilterPolicy(
            name=preset.name,
            min_target_margin=preset.min_target_margin,
            max_strong_matches=preset.max_strong_matches,
            generation_rank_cutoff=preset.generation_rank_cutoff,
            max_target_deficit=preset.max_target_deficit,
            round1_rank_cutoff=preset.round1_rank_cutoff,
            round2_rank_cutoff=preset.round2_rank_cutoff,
            round3_rank_cutoff=preset.round3_rank_cutoff,
            max_source_uses=preset.max_source_uses,
            max_target_uses=preset.max_target_uses,
            prefer_retry_candidates=preset.prefer_retry_candidates,
            prefer_agent_failure_candidates=preset.prefer_agent_failure_candidates,
            balance_outcomes=preset.balance_outcomes,
        )
    return FilterPolicy(
        name="custom",
        min_target_margin=float(min_target_margin),
        max_strong_matches=int(max_strong_matches),
        generation_rank_cutoff=1,
        max_target_deficit=0.0,
        round1_rank_cutoff=1,
        round2_rank_cutoff=1,
        round3_rank_cutoff=1,
        max_source_uses=6,
        max_target_uses=6,
        prefer_retry_candidates=False,
        prefer_agent_failure_candidates=False,
        balance_outcomes=False,
    )


def build_candidate_rows(
    msrvtt_json_path: str | Path,
    split_csv_path: str | Path | None = None,
    max_candidates: int | None = None,
) -> list[CandidateMetadata]:
    raw = load_json(msrvtt_json_path)
    split_ids = load_split_ids(split_csv_path)
    video_meta: dict[str, dict[str, Any]] = {}
    captions_by_video: dict[str, list[str]] = defaultdict(list)

    for item in raw.get("videos", []):
        video_id = item.get("video_id") or item.get("id")
        if not video_id:
            continue
        if split_ids is not None and video_id not in split_ids:
            continue
        video_meta[video_id] = item

    for item in raw.get("sentences", []):
        video_id = item.get("video_id")
        caption = item.get("caption") or item.get("text")
        if not video_id or not caption:
            continue
        if split_ids is not None and video_id not in split_ids:
            continue
        captions_by_video[video_id].append(caption.strip())
        video_meta.setdefault(video_id, {})

    ordered_video_ids = sorted(video_meta)
    if max_candidates is not None:
        ordered_video_ids = ordered_video_ids[:max_candidates]

    candidates: list[CandidateMetadata] = []
    for video_id in ordered_video_ids:
        captions = captions_by_video.get(video_id, [])
        caption = captions[0] if captions else ""
        summary = " ".join(captions[:2]).strip() or caption or video_id
        joined_text = " ".join(captions)
        audio_tags = detect_tags(joined_text, AUDIO_LEXICON)
        visual_objects = detect_tags(joined_text, OBJECT_LEXICON)
        scene_tags = detect_tags(joined_text, SCENE_LEXICON)
        temporal_tags = detect_tags(joined_text, TEMPORAL_LEXICON)
        if not temporal_tags:
            temporal_tags = ["global"]

        candidates.append(
            CandidateMetadata(
                video_id=video_id,
                title=caption[:80] or video_id,
                summary=summary,
                caption=caption or summary,
                asr="",
                audio_tags=audio_tags,
                visual_objects=visual_objects,
                scene_tags=scene_tags,
                temporal_tags=temporal_tags,
            )
        )
    return candidates


def similarity_score(source: CandidateMetadata, target: CandidateMetadata) -> float:
    source_tokens = tokenize(source.summary + " " + source.caption)
    target_tokens = tokenize(target.summary + " " + target.caption)
    token_overlap = len(source_tokens & target_tokens) / max(1, len(source_tokens | target_tokens))
    scene_overlap = len(set(source.scene_tags) & set(target.scene_tags))
    object_overlap = len(set(source.visual_objects) & set(target.visual_objects))
    return (0.5 * token_overlap) + (0.3 * scene_overlap) + (0.2 * object_overlap)


def make_query(
    query_id: str,
    source: CandidateMetadata,
    target: CandidateMetadata,
    query_type: str,
    required_value: str,
) -> QueryCase:
    preserve_tags = sorted(set(source.scene_tags) & set(target.scene_tags))
    if not preserve_tags:
        preserve_tags = sorted(set(source.scene_tags or target.scene_tags))[:2]
    scene_text = preserve_tags[0] if preserve_tags else "scene"

    if query_type == "audio":
        instruction = f"Find a similar {scene_text} clip, but with clear {required_value} audio."
        return QueryCase(
            query_id=query_id,
            source_video_id=source.video_id,
            edit_instruction=instruction,
            target_video_id=target.video_id,
            preserve_tags=preserve_tags,
            required_audio_tags=[required_value],
            notes="auto-generated from MSR-VTT captions",
        )
    if query_type == "object":
        article = "an" if required_value[:1].lower() in "aeiou" else "a"
        instruction = f"Keep the same {scene_text} scene, but focus on {article} {required_value}."
        return QueryCase(
            query_id=query_id,
            source_video_id=source.video_id,
            edit_instruction=instruction,
            target_video_id=target.video_id,
            preserve_tags=preserve_tags,
            required_objects=[required_value],
            notes="auto-generated from MSR-VTT captions",
        )

    instruction = f"Find a similar {scene_text} clip where the key event happens in the {required_value} part."
    return QueryCase(
        query_id=query_id,
        source_video_id=source.video_id,
        edit_instruction=instruction,
        target_video_id=target.video_id,
        preserve_tags=preserve_tags,
        required_temporal=required_value,
        notes="auto-generated from MSR-VTT captions",
    )


def generation_priority_score(
    query: QueryCase,
    source: CandidateMetadata,
    candidate: CandidateMetadata,
) -> tuple[float, dict[str, Any]]:
    candidate_scene_values = [*candidate.scene_tags, *candidate.visual_objects]
    preserve_ratio = overlap_score(query.preserve_tags, candidate_scene_values) if query.preserve_tags else 1.0
    object_ratio = overlap_score(query.required_objects, candidate.visual_objects) if query.required_objects else 0.0
    audio_ratio = overlap_score(query.required_audio_tags, candidate.audio_tags) if query.required_audio_tags else 0.0
    temporal_ratio = (
        1.0 if query.required_temporal and query.required_temporal in candidate.temporal_tags else 0.0
    ) if query.required_temporal else 0.0
    source_scene_overlap = len(set(source.scene_tags) & set(candidate.scene_tags)) / max(1, len(source.scene_tags) or 1)
    source_object_overlap = len(set(source.visual_objects) & set(candidate.visual_objects)) / max(1, len(source.visual_objects) or 1)
    token_overlap = len(tokenize(query.edit_instruction) & tokenize(candidate.summary + " " + candidate.caption)) / max(
        1, len(tokenize(query.edit_instruction))
    )
    required_signal = max(audio_ratio, object_ratio, temporal_ratio)
    source_required_signal = max(
        overlap_score(query.required_audio_tags, source.audio_tags) if query.required_audio_tags else 0.0,
        overlap_score(query.required_objects, source.visual_objects) if query.required_objects else 0.0,
        (
            1.0
            if query.required_temporal and query.required_temporal in source.temporal_tags
            else 0.0
        ) if query.required_temporal else 0.0,
    )
    novelty_gain = max(0.0, required_signal - source_required_signal)
    source_alignment = 0.6 * source_scene_overlap + 0.4 * source_object_overlap
    score = (
        0.40 * required_signal
        + 0.20 * preserve_ratio
        + 0.10 * token_overlap
        + 0.15 * novelty_gain
        + 0.15 * source_alignment
    )
    if required_signal == 0.0:
        score -= 0.20
    if query.required_objects and any(item in source.visual_objects for item in query.required_objects):
        score -= 0.10

    strong_match = (
        required_signal >= 1.0
        and preserve_ratio >= 0.5
        and token_overlap >= 0.10
    )
    signals = {
        "preserve_ratio": round(preserve_ratio, 4),
        "required_signal": round(required_signal, 4),
        "token_overlap": round(token_overlap, 4),
        "novelty_gain": round(novelty_gain, 4),
        "source_alignment": round(source_alignment, 4),
        "strong_match": strong_match,
    }
    return round(score, 4), signals


def score_candidate_for_query(
    query: QueryCase,
    source: CandidateMetadata,
    candidate: CandidateMetadata,
) -> dict[str, Any]:
    instruction_tokens = tokenize(query.edit_instruction)
    scene_overlap = len(set(query.preserve_tags) & set(candidate.scene_tags)) / max(1, len(query.preserve_tags) or 1)
    source_scene_overlap = len(set(source.scene_tags) & set(candidate.scene_tags)) / max(1, len(source.scene_tags) or 1)
    source_object_overlap = len(set(source.visual_objects) & set(candidate.visual_objects)) / max(1, len(source.visual_objects) or 1)
    token_overlap = len(instruction_tokens & tokenize(candidate.summary + " " + candidate.caption)) / max(1, len(instruction_tokens))

    required_audio = max(
        [1.0 if tag in candidate.audio_tags else 0.0 for tag in query.required_audio_tags] or [0.0]
    )
    required_object = max(
        [1.0 if tag in candidate.visual_objects else 0.0 for tag in query.required_objects] or [0.0]
    )
    required_temporal = (
        1.0 if query.required_temporal and query.required_temporal in candidate.temporal_tags else 0.0
    )

    video_score = round(
        0.35 * scene_overlap
        + 0.25 * source_scene_overlap
        + 0.20 * source_object_overlap
        + 0.20 * token_overlap
        + 0.20 * required_object
        + 0.10 * required_temporal,
        4,
    )
    audio_score = round(0.70 * required_audio + 0.30 * token_overlap, 4)

    return {
        "video_score": video_score,
        "audio_score": audio_score,
        "object_scores": {tag: float(tag in candidate.visual_objects) for tag in query.required_objects},
        "temporal_scores": (
            {query.required_temporal: required_temporal}
            if query.required_temporal
            else {}
        ),
    }


def scripted_round_params(query: QueryCase, round_index: int) -> dict[str, Any]:
    params = {
        "video_weight": 0.7,
        "audio_weight": 0.3,
        "object_focus": "none",
        "temporal_focus": "global",
    }
    if query.required_audio_tags:
        if round_index == 1:
            params["video_weight"] = 0.95
            params["audio_weight"] = 0.05
        else:
            params["video_weight"] = 0.45
            params["audio_weight"] = 0.55
    if query.required_objects and round_index > 1:
        params["object_focus"] = query.required_objects[0]
    if query.required_temporal and round_index > 1:
        params["temporal_focus"] = query.required_temporal
    return params


def build_ranked_candidates_for_query(
    query: QueryCase,
    source: CandidateMetadata,
    candidates: list[CandidateMetadata],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.video_id == source.video_id:
            continue
        score_payload = score_candidate_for_query(query, source, candidate)
        video_score = float(score_payload["video_score"])
        audio_score = float(score_payload["audio_score"])

        if params["object_focus"] != "none":
            video_score += float(score_payload["object_scores"].get(params["object_focus"], 0.0))
        if params["temporal_focus"] != "global":
            video_score += float(score_payload["temporal_scores"].get(params["temporal_focus"], 0.0))

        combined = params["video_weight"] * video_score + params["audio_weight"] * audio_score
        ranked.append(
            {
                "candidate_id": candidate.video_id,
                "score": round(combined, 4),
                "video_score": round(video_score, 4),
                "audio_score": round(audio_score, 4),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def rank_candidates_for_query(
    query: QueryCase,
    source: CandidateMetadata,
    candidates: list[CandidateMetadata],
    params: dict[str, Any],
) -> list[tuple[float, str]]:
    return [
        (float(item["score"]), str(item["candidate_id"]))
        for item in build_ranked_candidates_for_query(query, source, candidates, params)
    ]


def scripted_priority_tuple(
    query: QueryCase,
    candidate: dict[str, Any],
    comparison,
) -> tuple[float, float, float, float]:
    modality_score = float(candidate.get("audio_score", 0.0)) if query.required_audio_tags else float(
        candidate.get("video_score", 0.0)
    )
    return (
        float(comparison.confidence),
        modality_score,
        float(candidate.get("score", 0.0)),
        -float(len(comparison.missing)),
    )


def simulate_scripted_rollout(
    query: QueryCase,
    source: CandidateMetadata,
    target: CandidateMetadata,
    candidates: list[CandidateMetadata],
    max_rounds: int = 3,
) -> dict[str, Any]:
    candidate_map = {item.video_id: item for item in candidates}
    round_rankings = [
        build_ranked_candidates_for_query(query, source, candidates, scripted_round_params(query, round_index))
        for round_index in range(1, max_rounds + 1)
    ]
    round_target_ranks: list[int] = []
    for ranked in round_rankings:
        order = [str(item["candidate_id"]) for item in ranked]
        rank = order.index(target.video_id) + 1 if target.video_id in order else len(order) + 1
        round_target_ranks.append(rank)

    target_visible_rounds = sum(1 for rank in round_target_ranks if rank <= 2)
    final_candidate_id: str | None = None
    rounds_used = 0
    retry_rounds = 0

    for round_index, ranked in enumerate(round_rankings, start=1):
        rounds_used = round_index
        inspected = ranked[:2]
        if not inspected:
            break

        primary_compare = None
        best_candidate = None
        best_compare = None
        for candidate in inspected:
            candidate_id = str(candidate["candidate_id"])
            compare = heuristic_compare(query, source, candidate_map[candidate_id])
            if primary_compare is None:
                primary_compare = compare
            if best_compare is None or scripted_priority_tuple(query, candidate, compare) > scripted_priority_tuple(
                query,
                best_candidate,
                best_compare,
            ):
                best_candidate = candidate
                best_compare = compare

        if primary_compare is None or best_candidate is None:
            break
        if not primary_compare.missing or round_index >= max_rounds:
            final_candidate_id = str(best_candidate["candidate_id"])
            break
        retry_rounds += 1

    return {
        "round1_target_rank": round_target_ranks[0] if len(round_target_ranks) >= 1 else len(candidates) + 1,
        "round2_target_rank": round_target_ranks[1] if len(round_target_ranks) >= 2 else len(candidates) + 1,
        "round3_target_rank": round_target_ranks[2] if len(round_target_ranks) >= 3 else len(candidates) + 1,
        "target_visible_rounds": target_visible_rounds,
        "final_candidate_id": final_candidate_id,
        "agent_success": final_candidate_id == target.video_id if final_candidate_id else False,
        "rounds_used": rounds_used,
        "retry_rounds": retry_rounds,
    }


def classify_outcome_bucket(discriminability: QueryDiscriminability) -> str:
    if not discriminability.agent_success:
        return "failure"
    if discriminability.retry_rounds > 0:
        return "retry_success"
    return "direct_success"


def assess_query_discriminability(
    query: QueryCase,
    source: CandidateMetadata,
    target: CandidateMetadata,
    candidates: list[CandidateMetadata],
    policy: FilterPolicy,
) -> QueryDiscriminability:
    scored: list[tuple[float, str, dict[str, Any]]] = []
    strong_match_count = 0
    for candidate in candidates:
        if candidate.video_id == source.video_id:
            continue
        score, signals = generation_priority_score(query, source, candidate)
        scored.append((score, candidate.video_id, signals))
        if signals["strong_match"]:
            strong_match_count += 1

    scored.sort(key=lambda item: item[0], reverse=True)
    ranks = [video_id for _, video_id, _ in scored]
    if target.video_id not in ranks:
        return QueryDiscriminability(
            accepted=False,
            target_rank=len(ranks) + 1,
            target_score=0.0,
            second_score=0.0,
            score_margin=0.0,
            strong_match_count=strong_match_count,
            round1_target_rank=len(ranks) + 1,
            round2_target_rank=len(ranks) + 1,
            round3_target_rank=len(ranks) + 1,
            retry_favorable=False,
            agent_success=False,
            retry_rounds=0,
            target_visible_rounds=0,
            final_candidate_id=None,
            hardness=0.0,
        )

    target_rank = ranks.index(target.video_id) + 1
    target_score = next(score for score, video_id, _ in scored if video_id == target.video_id)
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    top_score = scored[0][0] if scored else 0.0
    score_margin = round(target_score - second_score, 4) if target_rank == 1 else round(target_score - top_score, 4)
    rollout = simulate_scripted_rollout(query, source, target, candidates)
    round1_target_rank = int(rollout["round1_target_rank"])
    round2_target_rank = int(rollout["round2_target_rank"])
    round3_target_rank = int(rollout["round3_target_rank"])
    retry_favorable = (
        round1_target_rank > 1
        and round1_target_rank <= policy.round1_rank_cutoff
        and round2_target_rank == 1
    )

    target_margin_ok = (
        score_margin >= policy.min_target_margin
        if target_rank == 1
        else abs(score_margin) <= policy.max_target_deficit
    )
    accepted = (
        target_rank <= policy.generation_rank_cutoff
        and target_margin_ok
        and strong_match_count <= policy.max_strong_matches
        and round1_target_rank <= policy.round1_rank_cutoff
        and round2_target_rank <= policy.round2_rank_cutoff
        and round3_target_rank <= policy.round3_rank_cutoff
    )

    hardness = (
        0.30 * float(retry_favorable)
        + 0.20 * float(round1_target_rank > 1)
        + 0.15 * min(1.0, strong_match_count / max(1, policy.max_strong_matches))
        + 0.15 * max(0.0, 1.0 - min(1.0, max(score_margin, 0.0) / max(1e-6, policy.min_target_margin or 0.01)))
        + 0.10 * (1.0 - (float(rollout["target_visible_rounds"]) / 3.0))
        + 0.10 * float(not rollout["agent_success"])
    )
    return QueryDiscriminability(
        accepted=accepted,
        target_rank=target_rank,
        target_score=target_score,
        second_score=second_score,
        score_margin=score_margin,
        strong_match_count=strong_match_count,
        round1_target_rank=round1_target_rank,
        round2_target_rank=round2_target_rank,
        round3_target_rank=round3_target_rank,
        retry_favorable=retry_favorable,
        agent_success=bool(rollout["agent_success"]),
        retry_rounds=int(rollout["retry_rounds"]),
        target_visible_rounds=int(rollout["target_visible_rounds"]),
        final_candidate_id=rollout["final_candidate_id"],
        hardness=round(hardness, 4),
    )


def append_query_record(
    record: AcceptedQueryRecord,
    policy: FilterPolicy,
    queries: list[QueryCase],
    type_counts: dict[str, int],
    type_limits: dict[str, int],
    source_counts: dict[str, int],
    target_counts: dict[str, int],
    filter_stats: dict[str, int],
    selected_keys: set[tuple[str, str, str, str]],
) -> bool:
    record_key = (record.query_type, record.source.video_id, record.target.video_id, record.value)
    if record_key in selected_keys:
        return False
    if type_counts[record.query_type] >= type_limits[record.query_type]:
        filter_stats["rejected_type_quota"] += 1
        return False
    if source_counts[record.source.video_id] >= policy.max_source_uses:
        filter_stats["rejected_source_quota"] += 1
        return False
    if target_counts[record.target.video_id] >= policy.max_target_uses:
        filter_stats["rejected_target_quota"] += 1
        return False

    query_id = f"msrvtt_{record.query_type}_{len(queries):05d}"
    queries.append(make_query(query_id, record.source, record.target, record.query_type, record.value))
    type_counts[record.query_type] += 1
    source_counts[record.source.video_id] += 1
    target_counts[record.target.video_id] += 1
    selected_keys.add(record_key)
    return True


def balanced_bucket_targets(type_limit: int) -> dict[str, int]:
    failure_target = max(1, type_limit // 3)
    retry_target = max(1, type_limit // 3)
    direct_target = max(1, type_limit - failure_target - retry_target)
    return {
        "failure": failure_target,
        "retry_success": retry_target,
        "direct_success": direct_target,
    }


def pair_selection_priority(
    source: CandidateMetadata,
    target: CandidateMetadata,
    query_type: str,
    value: str,
) -> float:
    similarity = similarity_score(source, target)
    preserve_overlap = len(set(source.scene_tags) & set(target.scene_tags)) / max(
        1,
        len(set(source.scene_tags) | set(target.scene_tags)) or 1,
    )
    novelty = 0.0
    if query_type == "audio":
        novelty = float(value in target.audio_tags and value not in source.audio_tags)
    elif query_type == "object":
        novelty = float(value in target.visual_objects and value not in source.visual_objects)
    elif query_type == "temporal":
        novelty = float(value in target.temporal_tags and value not in source.temporal_tags)

    return round(0.55 * similarity + 0.30 * novelty + 0.15 * preserve_overlap, 4)


def select_balanced_query_rows(
    accepted_records: list[AcceptedQueryRecord],
    max_queries: int,
    type_limits: dict[str, int],
    filter_stats: dict[str, int],
    policy: FilterPolicy,
) -> tuple[list[QueryCase], dict[str, int]]:
    queries: list[QueryCase] = []
    type_counts = {"audio": 0, "object": 0, "temporal": 0}
    source_counts: dict[str, int] = defaultdict(int)
    target_counts: dict[str, int] = defaultdict(int)
    selected_keys: set[tuple[str, str, str, str]] = set()
    selection_stats = {
        "selected_failure_queries": 0,
        "selected_retry_success_queries": 0,
        "selected_direct_success_queries": 0,
    }
    bucket_alias = {
        "failure": "selected_failure_queries",
        "retry_success": "selected_retry_success_queries",
        "direct_success": "selected_direct_success_queries",
    }
    type_order = ("audio", "object", "temporal")
    grouped: dict[str, dict[str, list[AcceptedQueryRecord]]] = {
        query_type: {"failure": [], "retry_success": [], "direct_success": []}
        for query_type in type_order
    }
    for record in accepted_records:
        grouped[record.query_type][record.outcome_bucket].append(record)

    for query_type in type_order:
        for bucket in grouped[query_type]:
            grouped[query_type][bucket].sort(key=lambda item: item.selection_priority, reverse=True)

    for query_type in type_order:
        targets = balanced_bucket_targets(type_limits[query_type])
        for bucket in ("failure", "retry_success", "direct_success"):
            added = 0
            for record in grouped[query_type][bucket]:
                if len(queries) >= max_queries or type_counts[query_type] >= type_limits[query_type]:
                    break
                if added >= targets[bucket]:
                    break
                if append_query_record(
                    record=record,
                    policy=policy,
                    queries=queries,
                    type_counts=type_counts,
                    type_limits=type_limits,
                    source_counts=source_counts,
                    target_counts=target_counts,
                    filter_stats=filter_stats,
                    selected_keys=selected_keys,
                ):
                    added += 1
                    selection_stats[bucket_alias[bucket]] += 1

    for query_type in type_order:
        merged = sorted(
            (
                record
                for bucket in ("failure", "retry_success", "direct_success")
                for record in grouped[query_type][bucket]
            ),
            key=lambda item: item.selection_priority,
            reverse=True,
        )
        for record in merged:
            if len(queries) >= max_queries or type_counts[query_type] >= type_limits[query_type]:
                break
            if append_query_record(
                record=record,
                policy=policy,
                queries=queries,
                type_counts=type_counts,
                type_limits=type_limits,
                source_counts=source_counts,
                target_counts=target_counts,
                filter_stats=filter_stats,
                selected_keys=selected_keys,
            ):
                selection_stats[bucket_alias[record.outcome_bucket]] += 1

    return queries, selection_stats


def build_query_rows(
    candidates: list[CandidateMetadata],
    max_queries: int = 100,
    seed: int = 13,
    policy: FilterPolicy | None = None,
    selection_mode: str = "target-aware",
) -> tuple[list[QueryCase], dict[str, int]]:
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"unsupported selection_mode: {selection_mode}")
    policy = policy or FILTER_PRESETS["medium-hard"]
    random.seed(seed)
    pairs: list[tuple[float, str, CandidateMetadata, CandidateMetadata, str]] = []
    for source in candidates:
        for target in candidates:
            if source.video_id == target.video_id:
                continue
            shared_scene = set(source.scene_tags) & set(target.scene_tags)
            if not shared_scene:
                continue

            added_audio = sorted(set(target.audio_tags) - set(source.audio_tags))
            added_objects = sorted(set(target.visual_objects) - set(source.visual_objects))
            added_temporal = sorted(
                {tag for tag in target.temporal_tags if tag != "global"}
                - {tag for tag in source.temporal_tags if tag != "global"}
            )
            similarity = similarity_score(source, target)

            if added_audio:
                pairs.append((similarity, "audio", source, target, added_audio[0]))
            if added_objects:
                pairs.append((similarity, "object", source, target, added_objects[0]))
            if added_temporal:
                pairs.append((similarity, "temporal", source, target, added_temporal[0]))

    pairs.sort(key=lambda item: item[0], reverse=True)

    accepted_records: list[AcceptedQueryRecord] = []
    used_triplets: set[tuple[str, str, str, str]] = set()
    type_limits = {
        "audio": max(1, max_queries // 3),
        "object": max(1, max_queries // 3),
        "temporal": max(1, max_queries // 3),
    }
    filter_stats = {
        "rejected_duplicate": 0,
        "rejected_type_quota": 0,
        "rejected_ambiguous": 0,
        "rejected_source_quota": 0,
        "rejected_target_quota": 0,
    }

    for _, query_type, source, target, value in pairs:
        key = (query_type, source.video_id, target.video_id, value)
        if key in used_triplets:
            filter_stats["rejected_duplicate"] += 1
            continue

        query_id = f"draft_{query_type}_{len(accepted_records):05d}"
        candidate_query = make_query(query_id, source, target, query_type, value)
        if selection_mode == "target-aware":
            discriminability = assess_query_discriminability(
                query=candidate_query,
                source=source,
                target=target,
                candidates=candidates,
                policy=policy,
            )
            if not discriminability.accepted:
                filter_stats["rejected_ambiguous"] += 1
                continue

            retry_bonus = 0.5 if policy.prefer_retry_candidates and discriminability.retry_favorable else 0.0
            failure_bonus = 0.8 if policy.prefer_agent_failure_candidates and not discriminability.agent_success else 0.0
            hidden_bonus = 0.15 * max(0, 3 - discriminability.target_visible_rounds)
            selection_priority = (
                failure_bonus
                + retry_bonus
                + hidden_bonus
                + discriminability.hardness
                + (1.0 - min(1.0, similarity_score(source, target)))
            )
            outcome_bucket = classify_outcome_bucket(discriminability)
        else:
            discriminability = QueryDiscriminability(
                accepted=True,
                target_rank=0,
                target_score=0.0,
                second_score=0.0,
                score_margin=0.0,
                strong_match_count=0,
                round1_target_rank=0,
                round2_target_rank=0,
                round3_target_rank=0,
                retry_favorable=False,
                agent_success=False,
                retry_rounds=0,
                target_visible_rounds=0,
                final_candidate_id=None,
                hardness=0.0,
            )
            selection_priority = pair_selection_priority(source, target, query_type, value)
            outcome_bucket = "direct_success"
        accepted_records.append(
            AcceptedQueryRecord(
                selection_priority=selection_priority,
                query_type=query_type,
                source=source,
                target=target,
                value=value,
                discriminability=discriminability,
                outcome_bucket=outcome_bucket,
            )
        )
        used_triplets.add(key)

    accepted_records.sort(key=lambda item: item.selection_priority, reverse=True)

    if policy.balance_outcomes and selection_mode == "target-aware":
        queries, selection_stats = select_balanced_query_rows(
            accepted_records=accepted_records,
            max_queries=max_queries,
            type_limits=type_limits,
            filter_stats=filter_stats,
            policy=policy,
        )
        return queries, {**filter_stats, **selection_stats}

    queries: list[QueryCase] = []
    type_counts = {"audio": 0, "object": 0, "temporal": 0}
    source_counts: dict[str, int] = defaultdict(int)
    target_counts: dict[str, int] = defaultdict(int)
    selected_keys: set[tuple[str, str, str, str]] = set()
    for record in accepted_records:
        if len(queries) >= max_queries:
            break
        append_query_record(
            record=record,
            policy=policy,
            queries=queries,
            type_counts=type_counts,
            type_limits=type_limits,
            source_counts=source_counts,
            target_counts=target_counts,
            filter_stats=filter_stats,
            selected_keys=selected_keys,
        )

    return queries, filter_stats


def build_retrieval_scores(
    candidates: list[CandidateMetadata],
    queries: list[QueryCase],
) -> dict[str, dict[str, dict[str, Any]]]:
    candidate_map = {item.video_id: item for item in candidates}
    scores: dict[str, dict[str, dict[str, Any]]] = {}

    for query in queries:
        source = candidate_map[query.source_video_id]
        query_scores: dict[str, dict[str, Any]] = {}

        for candidate in candidates:
            if candidate.video_id == query.source_video_id:
                continue
            query_scores[candidate.video_id] = score_candidate_for_query(query, source, candidate)

        scores[query.query_id] = query_scores

    return scores


def summarize_selected_queries(
    candidates: list[CandidateMetadata],
    queries: list[QueryCase],
    policy: FilterPolicy,
) -> dict[str, Any]:
    candidate_map = {item.video_id: item for item in candidates}
    discriminability_rows = [
        assess_query_discriminability(
            query=query,
            source=candidate_map[query.source_video_id],
            target=candidate_map[query.target_video_id],
            candidates=candidates,
            policy=policy,
        )
        for query in queries
    ]
    total = len(discriminability_rows)
    failure_count = sum(1 for item in discriminability_rows if not item.agent_success)
    retry_success_count = sum(1 for item in discriminability_rows if item.agent_success and item.retry_rounds > 0)
    direct_success_count = sum(1 for item in discriminability_rows if item.agent_success and item.retry_rounds == 0)
    retry_count = sum(1 for item in discriminability_rows if item.retry_rounds > 0)
    avg_visible_rounds = (
        sum(item.target_visible_rounds for item in discriminability_rows) / total
        if total
        else 0.0
    )
    return {
        "predicted_agent_failures": failure_count,
        "predicted_agent_successes": total - failure_count,
        "predicted_retry_success_queries": retry_success_count,
        "predicted_direct_success_queries": direct_success_count,
        "predicted_retry_queries": retry_count,
        "predicted_target_visible_rounds_avg": round(avg_visible_rounds, 2),
    }


def prepare_replay_pack(
    msrvtt_json_path: str | Path,
    split_csv_path: str | Path | None,
    output_dir: str | Path,
    max_candidates: int | None,
    max_queries: int,
    seed: int,
    difficulty_preset: str | None,
    min_target_margin: float,
    max_strong_matches: int,
    selection_mode: str = "target-aware",
) -> ReplayPack:
    output_dir = Path(output_dir)
    policy = resolve_filter_policy(
        difficulty_preset=difficulty_preset,
        min_target_margin=min_target_margin,
        max_strong_matches=max_strong_matches,
    )
    candidates = build_candidate_rows(
        msrvtt_json_path=msrvtt_json_path,
        split_csv_path=split_csv_path,
        max_candidates=max_candidates,
    )
    queries, filter_stats = build_query_rows(
        candidates,
        max_queries=max_queries,
        seed=seed,
        policy=policy,
        selection_mode=selection_mode,
    )
    retrieval_scores = build_retrieval_scores(candidates, queries)
    rollout_stats = summarize_selected_queries(candidates, queries, policy)

    candidates_payload = [item.to_dict() for item in candidates]
    queries_payload = [item.to_dict() for item in queries]
    config = {
        "candidates_path": "candidates.json",
        "queries_path": "queries.json",
        "retrieval_scores_path": "retrieval_scores.json",
    }
    stats = {
        "candidate_count": len(candidates_payload),
        "query_count": len(queries_payload),
        "audio_queries": sum(1 for item in queries if item.required_audio_tags),
        "object_queries": sum(1 for item in queries if item.required_objects),
        "temporal_queries": sum(1 for item in queries if item.required_temporal),
        "filter_mode": policy.name,
        "query_selection_mode": selection_mode,
        "min_target_margin": policy.min_target_margin,
        "max_strong_matches": policy.max_strong_matches,
        "generation_rank_cutoff": policy.generation_rank_cutoff,
        "max_target_deficit": policy.max_target_deficit,
        "round1_rank_cutoff": policy.round1_rank_cutoff,
        "round2_rank_cutoff": policy.round2_rank_cutoff,
        "round3_rank_cutoff": policy.round3_rank_cutoff,
        "max_source_uses": policy.max_source_uses,
        "max_target_uses": policy.max_target_uses,
        "prefer_retry_candidates": policy.prefer_retry_candidates,
        "prefer_agent_failure_candidates": policy.prefer_agent_failure_candidates,
        "balance_outcomes": policy.balance_outcomes if selection_mode == "target-aware" else False,
        "uses_target_aware_filtering": selection_mode == "target-aware",
        "uses_rollout_filtering": selection_mode == "target-aware",
        "uses_heuristic_retrieval_scores": True,
        "full_candidate_catalog": True,
        "discriminability_scorer": "generation_v2_decoupled",
        "retrieval_scorer": "heuristic_replay_full_catalog",
        "rollout_scorer": "scripted_controller_v1",
        "selection_strategy": (
            "balanced_outcome_mix"
            if policy.balance_outcomes and selection_mode == "target-aware"
            else ("pair_similarity_only" if selection_mode == "unbiased" else "priority_only")
        ),
        "expected_scored_candidates_per_query": max(0, len(candidates_payload) - 1),
        **rollout_stats,
        **filter_stats,
    }

    write_json(output_dir / "candidates.json", candidates_payload)
    write_json(output_dir / "queries.json", queries_payload)
    write_json(output_dir / "retrieval_scores.json", retrieval_scores)
    with (output_dir / "real.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    return ReplayPack(
        candidates=candidates_payload,
        queries=queries_payload,
        retrieval_scores=retrieval_scores,
        config=config,
        stats=stats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a pilot replay pack from raw MSR-VTT annotations")
    parser.add_argument("--msrvtt-json", required=True, help="Path to MSRVTT_data.json")
    parser.add_argument("--split-csv", help="Optional split csv such as MSRVTT_train.9k.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for candidates/queries/scores")
    parser.add_argument("--max-candidates", type=int, help="Optional cap on candidate videos")
    parser.add_argument("--max-queries", type=int, default=90, help="Number of auto-generated replay queries")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--difficulty-preset",
        choices=sorted(FILTER_PRESETS),
        help="Optional preset for replay filtering difficulty",
    )
    parser.add_argument("--min-target-margin", type=float, default=0.04)
    parser.add_argument("--max-strong-matches", type=int, default=1)
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default="target-aware",
        help="Whether query selection may use target-aware filtering or must avoid it entirely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack = prepare_replay_pack(
        msrvtt_json_path=args.msrvtt_json,
        split_csv_path=args.split_csv,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        max_queries=args.max_queries,
        seed=args.seed,
        difficulty_preset=args.difficulty_preset,
        min_target_margin=args.min_target_margin,
        max_strong_matches=args.max_strong_matches,
        selection_mode=args.selection_mode,
    )
    print(json.dumps(pack.stats, ensure_ascii=False, indent=2))
    print(f"config={Path(args.output_dir) / 'real.yaml'}")


if __name__ == "__main__":
    main()
