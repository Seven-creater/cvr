import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.backends import FileRetrievalBackend
from app.demo import build_backend
from app.prepare_msrvtt_replay import assess_query_discriminability, make_query, prepare_replay_pack
from app.schemas import CandidateMetadata


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
                min_target_margin=0.04,
                max_strong_matches=1,
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

    def test_discriminability_rejects_ambiguous_queries(self) -> None:
        source = CandidateMetadata(
            video_id="src1",
            title="source",
            summary="A dog runs in the park.",
            caption="Dog in park.",
            asr="",
            audio_tags=[],
            visual_objects=["dog", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        target = CandidateMetadata(
            video_id="target1",
            title="target",
            summary="A cat runs in the park.",
            caption="Cat in park.",
            asr="",
            audio_tags=[],
            visual_objects=["cat", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        distractor = CandidateMetadata(
            video_id="target2",
            title="distractor",
            summary="A cat moves through the park.",
            caption="Cat outdoors in park.",
            asr="",
            audio_tags=[],
            visual_objects=["cat", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        query = make_query("q1", source, target, "object", "cat")
        result = assess_query_discriminability(
            query=query,
            source=source,
            target=target,
            candidates=[source, target, distractor],
            min_target_margin=0.04,
            max_strong_matches=1,
        )
        self.assertFalse(result.accepted)
        self.assertGreaterEqual(result.strong_match_count, 2)


if __name__ == "__main__":
    unittest.main()
