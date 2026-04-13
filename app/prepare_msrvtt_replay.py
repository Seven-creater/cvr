from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
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


def query_priority_score(
    query: QueryCase,
    source: CandidateMetadata,
    candidate: CandidateMetadata,
) -> tuple[float, dict[str, Any]]:
    comparison = heuristic_compare(query, source, candidate)
    preserve_ratio = overlap_score(query.preserve_tags, [*candidate.scene_tags, *candidate.visual_objects]) if query.preserve_tags else 1.0
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
    score = (
        0.45 * comparison.confidence
        + 0.20 * preserve_ratio
        + 0.20 * required_signal
        + 0.10 * token_overlap
        + 0.05 * (0.5 * source_scene_overlap + 0.5 * source_object_overlap)
    )
    if comparison.conflicts:
        score -= 0.10 * len(comparison.conflicts)
    return round(score, 4), comparison.to_dict()


def assess_query_discriminability(
    query: QueryCase,
    source: CandidateMetadata,
    target: CandidateMetadata,
    candidates: list[CandidateMetadata],
    min_target_margin: float,
    max_strong_matches: int,
) -> QueryDiscriminability:
    scored: list[tuple[float, str, dict[str, Any]]] = []
    strong_match_count = 0
    for candidate in candidates:
        if candidate.video_id == source.video_id:
            continue
        score, comparison = query_priority_score(query, source, candidate)
        scored.append((score, candidate.video_id, comparison))
        if not comparison["missing"] and not comparison["conflicts"]:
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
        )

    target_rank = ranks.index(target.video_id) + 1
    target_score = next(score for score, video_id, _ in scored if video_id == target.video_id)
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    top_score = scored[0][0] if scored else 0.0
    score_margin = round(target_score - second_score, 4) if target_rank == 1 else round(target_score - top_score, 4)
    accepted = target_rank == 1 and (strong_match_count <= max_strong_matches or score_margin >= min_target_margin)
    return QueryDiscriminability(
        accepted=accepted,
        target_rank=target_rank,
        target_score=target_score,
        second_score=second_score,
        score_margin=score_margin,
        strong_match_count=strong_match_count,
    )


def build_query_rows(
    candidates: list[CandidateMetadata],
    max_queries: int = 100,
    seed: int = 13,
    min_target_margin: float = 0.04,
    max_strong_matches: int = 1,
) -> tuple[list[QueryCase], dict[str, int]]:
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

    queries: list[QueryCase] = []
    used_triplets: set[tuple[str, str, str, str]] = set()
    type_counts = {"audio": 0, "object": 0, "temporal": 0}
    type_limits = {
        "audio": max(1, max_queries // 3),
        "object": max(1, max_queries // 3),
        "temporal": max(1, max_queries // 3),
    }
    filter_stats = {
        "rejected_duplicate": 0,
        "rejected_type_quota": 0,
        "rejected_ambiguous": 0,
    }

    for _, query_type, source, target, value in pairs:
        if len(queries) >= max_queries:
            break
        key = (query_type, source.video_id, target.video_id, value)
        if key in used_triplets:
            filter_stats["rejected_duplicate"] += 1
            continue
        if type_counts[query_type] >= type_limits[query_type]:
            filter_stats["rejected_type_quota"] += 1
            continue

        query_id = f"msrvtt_{query_type}_{len(queries):05d}"
        candidate_query = make_query(query_id, source, target, query_type, value)
        discriminability = assess_query_discriminability(
            query=candidate_query,
            source=source,
            target=target,
            candidates=candidates,
            min_target_margin=min_target_margin,
            max_strong_matches=max_strong_matches,
        )
        if not discriminability.accepted:
            filter_stats["rejected_ambiguous"] += 1
            continue

        queries.append(candidate_query)
        used_triplets.add(key)
        type_counts[query_type] += 1

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
        source_scene = set(source.scene_tags)
        source_objects = set(source.visual_objects)
        instruction_tokens = tokenize(query.edit_instruction)

        for candidate in candidates:
            if candidate.video_id == query.source_video_id:
                continue
            scene_overlap = len(set(query.preserve_tags) & set(candidate.scene_tags)) / max(1, len(query.preserve_tags) or 1)
            source_scene_overlap = len(source_scene & set(candidate.scene_tags)) / max(1, len(source_scene) or 1)
            source_object_overlap = len(source_objects & set(candidate.visual_objects)) / max(1, len(source_objects) or 1)
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

            query_scores[candidate.video_id] = {
                "video_score": video_score,
                "audio_score": audio_score,
                "object_scores": {tag: float(tag in candidate.visual_objects) for tag in query.required_objects},
                "temporal_scores": (
                    {query.required_temporal: required_temporal}
                    if query.required_temporal
                    else {}
                ),
            }

        scores[query.query_id] = query_scores

    return scores


def prepare_replay_pack(
    msrvtt_json_path: str | Path,
    split_csv_path: str | Path | None,
    output_dir: str | Path,
    max_candidates: int | None,
    max_queries: int,
    seed: int,
    min_target_margin: float,
    max_strong_matches: int,
) -> ReplayPack:
    output_dir = Path(output_dir)
    candidates = build_candidate_rows(
        msrvtt_json_path=msrvtt_json_path,
        split_csv_path=split_csv_path,
        max_candidates=max_candidates,
    )
    queries, filter_stats = build_query_rows(
        candidates,
        max_queries=max_queries,
        seed=seed,
        min_target_margin=min_target_margin,
        max_strong_matches=max_strong_matches,
    )
    retrieval_scores = build_retrieval_scores(candidates, queries)

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
        "min_target_margin": min_target_margin,
        "max_strong_matches": max_strong_matches,
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
    parser.add_argument("--min-target-margin", type=float, default=0.04)
    parser.add_argument("--max-strong-matches", type=int, default=1)
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
        min_target_margin=args.min_target_margin,
        max_strong_matches=args.max_strong_matches,
    )
    print(json.dumps(pack.stats, ensure_ascii=False, indent=2))
    print(f"config={Path(args.output_dir) / 'real.yaml'}")


if __name__ == "__main__":
    main()
