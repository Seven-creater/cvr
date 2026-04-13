import json
import tempfile
import unittest
from pathlib import Path

from app.backends import FileRetrievalBackend
from app.eval import resolve_input_paths
from app.schemas import RetrievalParams


class FileBackendFormatTests(unittest.TestCase):
    def test_backend_accepts_nested_and_list_score_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates_path = root / "candidates.json"
            queries_path = root / "queries.json"
            scores_path = root / "scores.json"

            candidates_payload = {
                "videos": [
                    {
                        "id": "src1",
                        "name": "Source clip",
                        "description": "A dog runs in the park.",
                        "caption": "Dog in park",
                        "transcript": "Come here dog",
                        "audio_events": ["birds"],
                        "objects": ["dog", "trees"],
                        "scene_labels": ["park"],
                        "time_tags": ["global"],
                    },
                    {
                        "id": "cand1",
                        "name": "Target clip",
                        "description": "A cat runs in the park with birds chirping.",
                        "caption": "Cat in park",
                        "transcript": "Look at that cat",
                        "audio_events": ["birds", "chirping"],
                        "objects": ["cat", "trees"],
                        "scene_labels": ["park"],
                        "time_tags": ["global"],
                    },
                ]
            }
            queries_payload = {
                "queries": [
                    {
                        "id": "q1",
                        "source_id": "src1",
                        "instruction": "Keep the park scene but focus on a cat.",
                        "target_id": "cand1",
                        "preserve": ["park"],
                        "objects": ["cat"],
                    }
                ]
            }
            scores_payload = [
                {
                    "query_id": "q1",
                    "results": [
                        {"video_id": "cand1", "score": 0.7, "audio": 0.2},
                    ],
                }
            ]

            candidates_path.write_text(json.dumps(candidates_payload), encoding="utf-8")
            queries_path.write_text(json.dumps(queries_payload), encoding="utf-8")
            scores_path.write_text(json.dumps(scores_payload), encoding="utf-8")

            backend = FileRetrievalBackend(
                candidates_path=candidates_path,
                queries_path=queries_path,
                retrieval_scores_path=scores_path,
            )
            query = backend.get_query("q1")
            self.assertEqual(query.required_objects, ["cat"])
            results = backend.retrieve_candidates(query, params=RetrievalParams(), round_index=1)
            self.assertEqual(results[0].candidate_id, "cand1")

    def test_eval_resolve_input_paths_supports_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a.jsonl"
            second = root / "b.jsonl"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                paths = resolve_input_paths("*.jsonl")
            finally:
                os.chdir(old_cwd)
            self.assertEqual({path.name for path in paths}, {"a.jsonl", "b.jsonl"})


if __name__ == "__main__":
    unittest.main()
