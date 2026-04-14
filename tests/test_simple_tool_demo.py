import unittest

from app.backends import MockRetrievalBackend
from app.simple_tool_demo import SimpleToolEnvironment


class SimpleToolDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = MockRetrievalBackend()
        self.query = self.backend.get_query("q_audio_cheer")
        self.env = SimpleToolEnvironment(
            backend=self.backend,
            query=self.query,
            planner_name="simple-chat:test",
        )

    def test_runtime_query_hides_target(self) -> None:
        self.assertIsNone(self.env.trace.query.target_video_id)
        self.assertFalse(self.env.trace.planner_metadata["target_visible_during_runtime"])

    def test_tool_flow_finalizes_success_offline(self) -> None:
        retrieval = self.env.execute_tool(
            "retrieve_candidates",
            {
                "video_weight": 0.45,
                "audio_weight": 0.55,
                "object_focus": "none",
                "temporal_focus": "global",
                "topk": 5,
            },
        )
        best_id = retrieval["candidates"][0]["candidate_id"]
        self.env.execute_tool("inspect_candidate", {"candidate_id": best_id})
        self.env.execute_tool("compare_to_request", {"candidate_id": best_id})
        self.env.execute_tool(
            "submit_best_candidate",
            {"candidate_id": best_id, "explanation": "best match"},
        )

        trace = self.env.finalize()
        self.assertEqual(trace.final_candidate_id, self.query.target_video_id)
        self.assertTrue(trace.success)
        self.assertEqual(trace.query.target_video_id, self.query.target_video_id)


if __name__ == "__main__":
    unittest.main()
