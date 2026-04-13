import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.backends import FileRetrievalBackend
from app.demo import build_backend
from app.prepare_msrvtt_replay import prepare_replay_pack


class PrepareMsrvttReplayTests(unittest.TestCase):
    def test_prepare_replay_pack_generates_backend_compatible_files(self) -> None:
        raw = {
            "videos": [
                {"video_id": "video1"},
                {"video_id": "video2"},
                {"video_id": "video3"},
                {"video_id": "video4"},
            ],
            "sentences": [
                {"video_id": "video1", "caption": "A band performs on a concert stage while people listen."},
                {"video_id": "video1", "caption": "Musicians play music indoors."},
                {"video_id": "video2", "caption": "A band performs on a concert stage while the crowd cheers and applauds."},
                {"video_id": "video2", "caption": "People clap during the indoor concert."},
                {"video_id": "video3", "caption": "A dog runs in the park near trees."},
                {"video_id": "video3", "caption": "People watch the dog outside."},
                {"video_id": "video4", "caption": "A cat runs in the park near trees."},
                {"video_id": "video4", "caption": "People watch the cat outside."},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "MSRVTT_data.json"
            split_path = root / "split.csv"
            out_dir = root / "prepared"

            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            with split_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id"])
                writer.writerow(["video1"])
                writer.writerow(["video2"])
                writer.writerow(["video3"])
                writer.writerow(["video4"])

            pack = prepare_replay_pack(
                msrvtt_json_path=raw_path,
                split_csv_path=split_path,
                output_dir=out_dir,
                max_candidates=None,
                max_queries=6,
                seed=7,
            )

            self.assertGreaterEqual(pack.stats["candidate_count"], 4)
            self.assertGreaterEqual(pack.stats["query_count"], 2)
            self.assertTrue((out_dir / "candidates.json").exists())
            self.assertTrue((out_dir / "queries.json").exists())
            self.assertTrue((out_dir / "retrieval_scores.json").exists())
            self.assertTrue((out_dir / "real.yaml").exists())
            self.assertEqual(pack.config["candidates_path"], "candidates.json")

            backend = FileRetrievalBackend(
                candidates_path=out_dir / "candidates.json",
                queries_path=out_dir / "queries.json",
                retrieval_scores_path=out_dir / "retrieval_scores.json",
            )
            self.assertGreaterEqual(len(backend.list_queries()), 1)
            built_backend = build_backend("real", str(out_dir / "real.yaml"))
            self.assertGreaterEqual(len(built_backend.list_queries()), 1)


if __name__ == "__main__":
    unittest.main()
