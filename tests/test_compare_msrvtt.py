import unittest

from app.compare_msrvtt import build_comparison_payload, render_markdown
from app.eval import summarize_rows


class CompareMsrvttTests(unittest.TestCase):
    def test_summarize_rows_emits_structured_metrics(self) -> None:
        rows = [
            {
                "success": True,
                "query": {
                    "query_id": "q1",
                    "target_video_id": "cand1",
                    "required_audio_tags": ["music"],
                    "required_objects": [],
                    "required_temporal": None,
                },
                "final_candidate_id": "cand1",
                "rounds": [
                    {"retrieved_candidates": ["cand1", "cand2", "cand3", "cand4", "cand5", "cand6"]},
                ],
                "tool_history": [{}, {}, {}],
            },
            {
                "success": False,
                "query": {
                    "query_id": "q2",
                    "target_video_id": "cand7",
                    "required_audio_tags": [],
                    "required_objects": ["cat"],
                    "required_temporal": None,
                },
                "final_candidate_id": "cand8",
                "rounds": [
                    {
                        "retrieved_candidates": ["cand2", "cand3", "cand4", "cand5", "cand6", "cand7"],
                        "comparisons": {},
                    },
                    {
                        "retrieved_candidates": ["cand8", "cand2", "cand3", "cand4", "cand5", "cand6"],
                        "comparisons": {},
                    },
                ],
                "tool_history": [{}, {}, {}, {}],
            },
        ]

        summary = summarize_rows(rows, [1, 5, 10])
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["first_round_recall"]["R@5"], 0.5)
        self.assertEqual(summary["first_round_recall"]["R@10"], 1.0)
        self.assertEqual(summary["any_round_recall"]["R@1"], 0.5)
        self.assertIn("wrong_candidate", summary["error_breakdown"])

    def test_compare_payload_scales_ours_to_percent(self) -> None:
        payload = {
            "profiles": [
                {
                    "profile": "adaptive",
                    "planner_name": "scripted:adaptive",
                    "metrics": {
                        "success_rate": 0.222,
                        "avg_rounds": 1.39,
                        "avg_tool_calls": 7.94,
                        "first_round_top1": 0.244,
                        "first_round_recall": {"R@1": 0.244, "R@5": 1.0, "R@10": 1.0},
                        "any_round_recall": {"R@1": 0.589, "R@5": 1.0, "R@10": 1.0},
                        "type_breakdown": {
                            "audio": {"count": 30, "success_rate": 0.067},
                        },
                    },
                }
            ]
        }

        comparison = build_comparison_payload(
            payload,
            profiles=["adaptive"],
            method_label="Ours",
            paper_reference="avigate_paper",
            include_reproduction=True,
        )
        round1_row = comparison["ours_rows"][0]
        anyround_row = comparison["ours_rows"][1]
        self.assertEqual(round1_row["t2v"]["R@1"], 24.4)
        self.assertEqual(anyround_row["t2v"]["R@1"], 58.9)
        markdown = render_markdown(comparison)
        self.assertIn("AVIGATE (Paper)", markdown)
        self.assertIn("Ours (adaptive, Round-1)", markdown)
        self.assertIn("T2V R@10", markdown)


if __name__ == "__main__":
    unittest.main()
