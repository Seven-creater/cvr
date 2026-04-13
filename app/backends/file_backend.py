from __future__ import annotations

from pathlib import Path
from typing import Any

from app.backends.base import RetrievalBackend, describe_reason, load_json, normalize_tokens, overlap_score
from app.schemas import CandidateMetadata, QueryCase, RetrievalCandidate, RetrievalParams


def _normalize_query_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("queries", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            items = []
            for key, value in payload.items():
                item = dict(value)
                item.setdefault("query_id", key)
                items.append(item)
            return items
    raise ValueError("unsupported query file format")


def _normalize_candidate_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("candidates", "videos", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            items = []
            for key, value in payload.items():
                item = dict(value)
                item.setdefault("video_id", key)
                items.append(item)
            return items
    raise ValueError("unsupported candidate file format")


def _normalize_score_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_score": float(item.get("video_score", item.get("score", 0.0))),
        "audio_score": float(item.get("audio_score", item.get("audio", 0.0))),
        "object_scores": dict(item.get("object_scores", item.get("objects", {})) or {}),
        "temporal_scores": dict(item.get("temporal_scores", item.get("temporal", {})) or {}),
    }


def _normalize_retrieval_scores(payload: Any) -> dict[str, dict[str, dict[str, Any]]]:
    if not payload:
        return {}

    if isinstance(payload, dict):
        if any(key in payload for key in ("results", "retrieval_scores", "queries", "items")):
            for key in ("results", "retrieval_scores", "queries", "items"):
                value = payload.get(key)
                if value is not None:
                    return _normalize_retrieval_scores(value)

        normalized: dict[str, dict[str, dict[str, Any]]] = {}
        for query_id, value in payload.items():
            if isinstance(value, dict):
                normalized[query_id] = {}
                for candidate_id, score_payload in value.items():
                    if isinstance(score_payload, dict):
                        normalized[query_id][candidate_id] = _normalize_score_payload(score_payload)
            elif isinstance(value, list):
                normalized[query_id] = {}
                for score_payload in value:
                    if not isinstance(score_payload, dict):
                        continue
                    candidate_id = (
                        score_payload.get("candidate_id")
                        or score_payload.get("video_id")
                        or score_payload.get("id")
                    )
                    if candidate_id is None:
                        continue
                    normalized[query_id][candidate_id] = _normalize_score_payload(score_payload)
        if normalized:
            return normalized

    if isinstance(payload, list):
        normalized = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            query_id = item.get("query_id") or item.get("id")
            if query_id is None:
                continue
            normalized[query_id] = {}
            results = (
                item.get("results")
                or item.get("ranked_candidates")
                or item.get("candidates")
                or item.get("items")
                or []
            )
            for score_payload in results:
                if not isinstance(score_payload, dict):
                    continue
                candidate_id = (
                    score_payload.get("candidate_id")
                    or score_payload.get("video_id")
                    or score_payload.get("id")
                )
                if candidate_id is None:
                    continue
                normalized[query_id][candidate_id] = _normalize_score_payload(score_payload)
        return normalized

    raise ValueError("unsupported retrieval score format")


class FileRetrievalBackend(RetrievalBackend):
    def __init__(
        self,
        candidates_path: str | Path,
        queries_path: str | Path,
        retrieval_scores_path: str | Path | None = None,
    ) -> None:
        candidates = _normalize_candidate_items(load_json(candidates_path))
        queries = _normalize_query_items(load_json(queries_path))
        candidate_rows = [CandidateMetadata.from_dict(item) for item in candidates]
        query_rows = [QueryCase.from_dict(item) for item in queries]
        self._candidates = {item.video_id: item for item in candidate_rows}
        self._queries = {item.query_id: item for item in query_rows}
        self._retrieval_scores = (
            _normalize_retrieval_scores(load_json(retrieval_scores_path))
            if retrieval_scores_path
            else {}
        )

    def list_queries(self) -> list[QueryCase]:
        return list(self._queries.values())

    def get_query(self, query_id: str) -> QueryCase:
        return self._queries[query_id]

    def get_candidate(self, candidate_id: str) -> CandidateMetadata:
        return self._candidates[candidate_id]

    def retrieve_candidates(
        self,
        query: QueryCase,
        params: RetrievalParams,
        round_index: int,
    ) -> list[RetrievalCandidate]:
        query_scores = self._retrieval_scores.get(query.query_id, {})
        if query_scores:
            ranked: list[RetrievalCandidate] = []
            for candidate_id, score_payload in query_scores.items():
                candidate = self.get_candidate(candidate_id)
                video_score = float(score_payload.get("video_score", 0.0))
                audio_score = float(score_payload.get("audio_score", 0.0))
                object_scores = score_payload.get("object_scores", {})
                temporal_scores = score_payload.get("temporal_scores", {})

                if params.object_focus != "none":
                    video_score += float(object_scores.get(params.object_focus, 0.0))
                if params.temporal_focus != "global":
                    video_score += float(temporal_scores.get(params.temporal_focus, 0.0))

                combined = params.video_weight * video_score + params.audio_weight * audio_score
                ranked.append(
                    RetrievalCandidate(
                        candidate_id=candidate_id,
                        score=round(combined, 4),
                        video_score=round(video_score, 4),
                        audio_score=round(audio_score, 4),
                        summary=candidate.summary,
                        audio_tags=list(candidate.audio_tags),
                        reason=f"file-score(video={video_score:.2f}, audio={audio_score:.2f})",
                    )
                )

            ranked.sort(key=lambda item: item.score, reverse=True)
            return ranked[: params.topk]

        source = self.get_candidate(query.source_video_id)
        instruction_tokens = normalize_tokens(query.edit_instruction)
        required_temporal = (
            [query.required_temporal.lower()] if query.required_temporal else []
        )
        source_tags = set(source.scene_tags + source.visual_objects)
        results: list[RetrievalCandidate] = []

        for candidate in self._candidates.values():
            if candidate.video_id == query.source_video_id:
                continue

            candidate_visual = set(candidate.scene_tags + candidate.visual_objects)
            preserve_score = overlap_score(query.preserve_tags, list(candidate_visual))
            source_similarity = len(source_tags & candidate_visual) / max(1, len(source_tags))
            object_score = overlap_score(query.required_objects, candidate.visual_objects)
            temporal_score = overlap_score(required_temporal, [item.lower() for item in candidate.temporal_tags])
            audio_score = overlap_score(query.required_audio_tags, candidate.audio_tags)
            token_overlap = len(
                instruction_tokens
                & normalize_tokens(candidate.summary, candidate.caption, candidate.asr)
            ) / max(1, len(instruction_tokens))

            video_score = (
                0.50 * preserve_score
                + 0.30 * source_similarity
                + 0.20 * token_overlap
            )
            if params.object_focus != "none":
                video_score += 0.20 * float(params.object_focus.lower() in {value.lower() for value in candidate.visual_objects})
            if params.temporal_focus != "global":
                video_score += 0.20 * float(params.temporal_focus.lower() in {value.lower() for value in candidate.temporal_tags})

            total_score = params.video_weight * video_score + params.audio_weight * audio_score
            total_score += 0.03 * round_index

            results.append(
                RetrievalCandidate(
                    candidate_id=candidate.video_id,
                    score=round(total_score, 4),
                    video_score=round(video_score, 4),
                    audio_score=round(audio_score, 4),
                    summary=candidate.summary,
                    audio_tags=list(candidate.audio_tags),
                    reason=describe_reason(
                        preserve_score=preserve_score,
                        object_score=object_score,
                        temporal_score=temporal_score,
                        audio_score=audio_score,
                    ),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: params.topk]
