from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.schemas import CandidateMetadata, CompareResult, InspectionRecord, QueryCase, RetrievalCandidate, RetrievalParams

TOKEN_RE = re.compile(r"[a-z0-9]+")
APP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_ROOT.parent


def resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    direct = (PROJECT_ROOT / candidate).resolve()
    if direct.exists():
        return direct
    return (APP_ROOT / candidate).resolve()


def load_json(path: str | Path) -> Any:
    with resolve_data_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_tokens(*parts: str) -> set[str]:
    merged = " ".join(part.lower() for part in parts if part)
    return set(TOKEN_RE.findall(merged))


def overlap_score(required: list[str], observed: list[str]) -> float:
    if not required:
        return 0.0
    required_set = {item.lower() for item in required}
    observed_set = {item.lower() for item in observed}
    hits = len(required_set & observed_set)
    return hits / max(1, len(required_set))


def describe_reason(
    preserve_score: float,
    object_score: float,
    temporal_score: float,
    audio_score: float,
) -> str:
    return (
        f"preserve={preserve_score:.2f}, object={object_score:.2f}, "
        f"temporal={temporal_score:.2f}, audio={audio_score:.2f}"
    )


def heuristic_compare(query: QueryCase, source: CandidateMetadata, candidate: CandidateMetadata) -> CompareResult:
    satisfied: list[str] = []
    missing: list[str] = []
    conflicts: list[str] = []
    candidate_scene_values = [*candidate.scene_tags, *candidate.visual_objects]
    source_scene_values = [*source.scene_tags, *source.visual_objects]
    candidate_scene_set = {value.lower() for value in candidate_scene_values}
    candidate_object_set = {value.lower() for value in candidate.visual_objects}
    candidate_audio_set = {value.lower() for value in candidate.audio_tags}
    candidate_temporal_set = {value.lower() for value in candidate.temporal_tags}

    preserve_hits = [tag for tag in query.preserve_tags if tag.lower() in candidate_scene_set]
    if len(preserve_hits) >= max(1, len(query.preserve_tags) // 2):
        satisfied.append("preserve-scene")
    else:
        missing.append("preserve-scene")

    object_hits = [tag for tag in query.required_objects if tag.lower() in candidate_object_set]
    if query.required_objects:
        if len(object_hits) == len(query.required_objects):
            satisfied.append("required-object")
        else:
            missing.append("required-object")

    audio_hits = [tag for tag in query.required_audio_tags if tag.lower() in candidate_audio_set]
    if query.required_audio_tags:
        if len(audio_hits) == len(query.required_audio_tags):
            satisfied.append("required-audio")
        else:
            missing.append("required-audio")

    if query.required_temporal:
        if query.required_temporal.lower() in candidate_temporal_set:
            satisfied.append("required-temporal")
        else:
            missing.append("required-temporal")

    if "required-object" in satisfied and any(
        object_name.lower() in {value.lower() for value in source.visual_objects}
        for object_name in query.required_objects
    ):
        conflicts.append("object-change-too-small")

    preserve_ratio = overlap_score(query.preserve_tags, candidate_scene_values) if query.preserve_tags else 1.0
    object_ratio = overlap_score(query.required_objects, candidate.visual_objects) if query.required_objects else 1.0
    audio_ratio = overlap_score(query.required_audio_tags, candidate.audio_tags) if query.required_audio_tags else 1.0
    temporal_ratio = (
        1.0 if not query.required_temporal else float(query.required_temporal.lower() in candidate_temporal_set)
    )
    source_overlap = len({value.lower() for value in source_scene_values} & candidate_scene_set) / max(
        1, len({value.lower() for value in source_scene_values})
    )
    instruction_tokens = normalize_tokens(query.edit_instruction)
    candidate_tokens = normalize_tokens(candidate.summary, candidate.caption, candidate.asr)
    instruction_overlap = len(instruction_tokens & candidate_tokens) / max(1, len(instruction_tokens))

    weighted_components: list[tuple[float, float]] = [
        (preserve_ratio, 0.35 if query.preserve_tags else 0.15),
        (instruction_overlap, 0.15),
        (source_overlap, 0.10),
    ]
    if query.required_objects:
        weighted_components.append((object_ratio, 0.20))
    if query.required_audio_tags:
        weighted_components.append((audio_ratio, 0.25))
    if query.required_temporal:
        weighted_components.append((temporal_ratio, 0.20))

    weight_sum = sum(weight for _, weight in weighted_components)
    confidence = sum(score * weight for score, weight in weighted_components) / max(1e-6, weight_sum)
    if conflicts:
        confidence -= 0.10 * len(conflicts)
    confidence = max(0.0, min(1.0, confidence))
    return CompareResult(
        candidate_id=candidate.video_id,
        satisfied=satisfied,
        missing=missing,
        conflicts=conflicts,
        confidence=round(confidence, 4),
    )


class RetrievalBackend(ABC):
    @abstractmethod
    def list_queries(self) -> list[QueryCase]:
        raise NotImplementedError

    @abstractmethod
    def get_query(self, query_id: str) -> QueryCase:
        raise NotImplementedError

    @abstractmethod
    def get_candidate(self, candidate_id: str) -> CandidateMetadata:
        raise NotImplementedError

    @abstractmethod
    def retrieve_candidates(
        self,
        query: QueryCase,
        params: RetrievalParams,
        round_index: int,
    ) -> list[RetrievalCandidate]:
        raise NotImplementedError

    def inspect_candidate(self, candidate_id: str) -> InspectionRecord:
        candidate = self.get_candidate(candidate_id)
        return InspectionRecord(
            candidate_id=candidate.video_id,
            title=candidate.title,
            summary=candidate.summary,
            caption=candidate.caption,
            asr=candidate.asr,
            audio_tags=list(candidate.audio_tags),
            visual_objects=list(candidate.visual_objects),
            temporal_tags=list(candidate.temporal_tags),
        )

    def compare_to_request(self, query: QueryCase, candidate_id: str) -> CompareResult:
        source = self.get_candidate(query.source_video_id)
        candidate = self.get_candidate(candidate_id)
        return heuristic_compare(query, source, candidate)
