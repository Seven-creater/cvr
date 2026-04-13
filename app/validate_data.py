from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.backends import FileRetrievalBackend


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

    for query in queries:
        source = backend.get_candidate(query.source_video_id)
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
            backend.get_candidate(query.target_video_id)

    print("validation=ok")


if __name__ == "__main__":
    main()
