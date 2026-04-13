from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.backends import FileRetrievalBackend
from app.backends.base import load_json
from app.backends.file_backend import _normalize_retrieval_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate file-backed retrieval data")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = FileRetrievalBackend(
        candidates_path=args.candidates,
        queries_path=args.queries,
        retrieval_scores_path=args.scores,
    )

    queries = backend.list_queries()
    candidate_ids = {item.video_id for item in [backend.get_candidate(q.source_video_id) for q in queries]}
    print(f"queries={len(queries)}")
    print(f"unique_source_videos={len(candidate_ids)}")

    all_candidate_ids = set()
    missing_targets: list[str] = []
    duplicate_query_ids: list[str] = []
    seen_query_ids: set[str] = set()

    for query in queries:
        if query.query_id in seen_query_ids:
            duplicate_query_ids.append(query.query_id)
        seen_query_ids.add(query.query_id)
        source = backend.get_candidate(query.source_video_id)
        all_candidate_ids.add(source.video_id)
        print(
            json.dumps(
                {
                    "query_id": query.query_id,
                    "source_video_id": query.source_video_id,
                    "target_video_id": query.target_video_id,
                    "source_title": source.title,
                    "required_audio_tags": query.required_audio_tags,
                    "required_objects": query.required_objects,
                    "required_temporal": query.required_temporal,
                },
                ensure_ascii=False,
            )
        )

        if query.target_video_id:
            try:
                target = backend.get_candidate(query.target_video_id)
                all_candidate_ids.add(target.video_id)
            except KeyError:
                missing_targets.append(query.query_id)

    raw_scores = load_json(args.scores) if args.scores else {}
    normalized_scores = _normalize_retrieval_scores(raw_scores) if raw_scores else {}
    missing_score_queries = [query.query_id for query in queries if args.scores and query.query_id not in normalized_scores]
    dangling_score_candidates: list[str] = []
    for query_id, ranked in normalized_scores.items():
        for candidate_id in ranked:
            if candidate_id not in backend._candidates:  # noqa: SLF001
                dangling_score_candidates.append(f"{query_id}:{candidate_id}")

    print(f"candidate_catalog={len(backend._candidates)}")
    print(f"score_queries={len(normalized_scores)}")
    print(f"duplicate_query_ids={duplicate_query_ids or 'none'}")
    print(f"missing_target_rows={missing_targets or 'none'}")
    print(f"queries_without_scores={missing_score_queries or 'none'}")
    print(f"dangling_score_candidates={dangling_score_candidates or 'none'}")

    print("validation=ok")


if __name__ == "__main__":
    main()
