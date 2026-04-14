import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.compare_msrvtt import build_comparison_payload
from app.msrvtt_t2v_suite import build_text_queries, compare_candidate, summarize_profile
from app.schemas import CandidateMetadata, TextQueryCase


class MsrvttT2VSuiteTests(unittest.TestCase):
    def test_build_text_queries_uses_standard_caption_protocol(self) -> None:
        payload = {
            "sentences": [
                {"video_id": "video1", "caption": "A band plays music on stage."},
                {"video_id": "video2", "caption": "A dog runs in the park."},
            ],
            "videos": [
                {"video_id": "video1"},
                {"video_id": "video2"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "MSRVTT_data.json"
            split_path = root / "MSRVTT_JSFUSION_test.csv"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            with split_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id"])
                writer.writerow(["video1"])
                writer.writerow(["video2"])

            queries = build_text_queries(json_path, split_path)

        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0].target_video_id, "video1")
        self.assertEqual(queries[0].required_audio_tags, ["music"])

    def test_compare_candidate_detects_missing_audio(self) -> None:
        query = TextQueryCase(
            query_id="q1",
            text="A band plays music on stage.",
            target_video_id="video1",
            required_audio_tags=["music"],
            scene_tags=["concert"],
        )
        candidate = CandidateMetadata(
            video_id="video2",
            title="Quiet stage",
            summary="A band stands on stage before the show begins.",
            caption="A band waits on stage.",
            asr="",
            audio_tags=["speech"],
            visual_objects=["person"],
            scene_tags=["concert"],
            temporal_tags=["global"],
        )

        result = compare_candidate(query, candidate)
        self.assertIn("required-audio", result["missing"])
        self.assertIn("scene-match", result["satisfied"])

    def test_summarize_profile_emits_final_round_metrics(self) -> None:
        rows = [
            {
                "query": TextQueryCase("q1", "text", "video1").to_dict(),
                "rounds": [{"round_index": 1}],
                "tool_history_count": 5,
                "first_round_target_rank": 2,
                "final_round_target_rank": 1,
                "success": True,
            },
            {
                "query": TextQueryCase("q2", "text", "video2").to_dict(),
                "rounds": [{"round_index": 1}],
                "tool_history_count": 5,
                "first_round_target_rank": 6,
                "final_round_target_rank": 4,
                "success": False,
            },
        ]

        metrics = summarize_profile(rows, [1, 5, 10])
        self.assertEqual(metrics["first_round_recall"]["R@1"], 0.0)
        self.assertEqual(metrics["final_round_recall"]["R@1"], 0.5)
        self.assertEqual(metrics["final_round_recall"]["R@5"], 1.0)

    def test_compare_payload_prefers_final_round_for_standard_suite(self) -> None:
        payload = {
            "profiles": [
                {
                    "profile": "adaptive",
                    "planner_name": "scripted:adaptive",
                    "metrics": {
                        "success_rate": 0.5,
                        "avg_rounds": 1.5,
                        "avg_tool_calls": 6.0,
                        "first_round_top1": 0.3,
                        "first_round_recall": {"R@1": 0.3, "R@5": 0.7, "R@10": 1.0},
                        "final_round_recall": {"R@1": 0.4, "R@5": 0.8, "R@10": 1.0},
                        "type_breakdown": {},
                    },
                }
            ]
        }

        comparison = build_comparison_payload(
            payload,
            profiles=["adaptive"],
            method_label="Ours",
            paper_reference="avigate_paper",
            include_reproduction=False,
        )
        self.assertEqual(comparison["ours_rows"][1]["name"], "Ours (adaptive, Final-round)")
        self.assertEqual(comparison["ours_rows"][1]["t2v"]["R@1"], 40.0)


if __name__ == "__main__":
    unittest.main()
