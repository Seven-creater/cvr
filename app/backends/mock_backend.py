from __future__ import annotations

from pathlib import Path

from app.backends.base import RetrievalBackend, describe_reason, load_json, normalize_tokens, overlap_score
from app.schemas import CandidateMetadata, QueryCase, RetrievalCandidate, RetrievalParams


class MockRetrievalBackend(RetrievalBackend):
    def __init__(
        self,
        candidates_path: str | Path = "data/mock/candidates.json",
        queries_path: str | Path = "data/mock/queries.json",
    ) -> None:
        candidates = load_json(candidates_path)
        queries = load_json(queries_path)
        self._candidates = {
            item["video_id"]: CandidateMetadata.from_dict(item) for item in candidates
        }
        self._queries = {
            item["query_id"]: QueryCase.from_dict(item) for item in queries
        }

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
        source = self.get_candidate(query.source_video_id)
        source_tags = set(source.scene_tags + source.visual_objects)
        instruction_tokens = normalize_tokens(query.edit_instruction)
        focus_tokens = {params.object_focus.lower()} if params.object_focus != "none" else set()
        required_temporal = (
            {query.required_temporal.lower()}
            if query.required_temporal
            else set()
        )
        results: list[RetrievalCandidate] = []

        for candidate in self._candidates.values():
            if candidate.video_id == query.source_video_id:
                continue

            candidate_visual = set(candidate.scene_tags + candidate.visual_objects)
            preserve_score = overlap_score(query.preserve_tags, list(candidate_visual))
            source_similarity = len(source_tags & candidate_visual) / max(1, len(source_tags))
            object_score = overlap_score(query.required_objects, candidate.visual_objects)
            temporal_score = overlap_score(list(required_temporal), [item.lower() for item in candidate.temporal_tags])
            audio_score = overlap_score(query.required_audio_tags, candidate.audio_tags)
            token_overlap = len(
                instruction_tokens
                & normalize_tokens(candidate.summary, candidate.caption, candidate.asr)
            ) / max(1, len(instruction_tokens))

            video_score = (
                0.45 * preserve_score
                + 0.35 * source_similarity
                + 0.20 * token_overlap
            )

            if params.object_focus != "none" and params.object_focus.lower() in {
                item.lower() for item in candidate.visual_objects
            }:
                video_score += 0.35

            if params.temporal_focus != "global" and params.temporal_focus.lower() in {
                item.lower() for item in candidate.temporal_tags
            }:
                video_score += 0.30

            if focus_tokens and focus_tokens & {value.lower() for value in candidate.visual_objects}:
                video_score += 0.15

            total_score = params.video_weight * video_score + params.audio_weight * (
                0.70 * audio_score + 0.30 * token_overlap
            )
            total_score += 0.05 * round_index

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
