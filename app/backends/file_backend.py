from __future__ import annotations

from pathlib import Path

from app.backends.base import RetrievalBackend, describe_reason, load_json, normalize_tokens, overlap_score
from app.schemas import CandidateMetadata, QueryCase, RetrievalCandidate, RetrievalParams


class FileRetrievalBackend(RetrievalBackend):
    def __init__(
        self,
        candidates_path: str | Path,
        queries_path: str | Path,
        retrieval_scores_path: str | Path | None = None,
    ) -> None:
        candidates = load_json(candidates_path)
        queries = load_json(queries_path)
        self._candidates = {
            item["video_id"]: CandidateMetadata.from_dict(item) for item in candidates
        }
        self._queries = {
            item["query_id"]: QueryCase.from_dict(item) for item in queries
        }
        self._retrieval_scores = (
            load_json(retrieval_scores_path) if retrieval_scores_path else {}
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
