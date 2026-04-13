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

    preserve_hits = [tag for tag in query.preserve_tags if tag.lower() in {*(value.lower() for value in candidate.scene_tags), *(value.lower() for value in candidate.visual_objects)}]
    if len(preserve_hits) >= max(1, len(query.preserve_tags) // 2):
        satisfied.append("preserve-scene")
    else:
        missing.append("preserve-scene")

    object_hits = [tag for tag in query.required_objects if tag.lower() in {value.lower() for value in candidate.visual_objects}]
    if query.required_objects:
        if len(object_hits) == len(query.required_objects):
            satisfied.append("required-object")
        else:
            missing.append("required-object")

    audio_hits = [tag for tag in query.required_audio_tags if tag.lower() in {value.lower() for value in candidate.audio_tags}]
    if query.required_audio_tags:
        if len(audio_hits) == len(query.required_audio_tags):
            satisfied.append("required-audio")
        else:
            missing.append("required-audio")

    if query.required_temporal:
        if query.required_temporal.lower() in {value.lower() for value in candidate.temporal_tags}:
            satisfied.append("required-temporal")
        else:
            missing.append("required-temporal")

    if "required-object" in satisfied and any(
        object_name.lower() in {value.lower() for value in source.visual_objects}
        for object_name in query.required_objects
    ):
        conflicts.append("object-change-too-small")

    max_checks = 1 + int(bool(query.required_objects)) + int(bool(query.required_audio_tags)) + int(bool(query.required_temporal))
    confidence = len(satisfied) / max(1, max_checks)
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
