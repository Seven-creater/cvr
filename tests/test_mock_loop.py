import unittest

from app.backends import MockRetrievalBackend
from app.controller import ScriptedController, resolve_scripted_policy


class MockLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = MockRetrievalBackend()
        self.controller = ScriptedController(self.backend)

    def test_all_mock_queries_finish_within_three_rounds(self) -> None:
        for query in self.backend.list_queries():
            trace = self.controller.run(query.query_id)
            self.assertLessEqual(len(trace.rounds), 3, query.query_id)
            self.assertEqual(trace.final_candidate_id, query.target_video_id, query.query_id)
            self.assertIsNone(trace.query.target_video_id)
            self.assertIsNone(trace.success, query.query_id)

    def test_audio_query_increases_audio_weight_after_retry(self) -> None:
        trace = self.controller.run("q_audio_cheer")
        self.assertGreaterEqual(len(trace.rounds), 2)
        first_params = trace.rounds[0].retrieval_params
        second_params = trace.rounds[1].retrieval_params
        self.assertLess(first_params.audio_weight, second_params.audio_weight)
        self.assertEqual(trace.final_candidate_id, "cand_arena_cheer")

    def test_object_query_retries_with_object_focus(self) -> None:
        trace = self.controller.run("q_object_cat")
        self.assertGreaterEqual(len(trace.rounds), 2)
        self.assertEqual(trace.rounds[0].decision, "retry")
        self.assertEqual(trace.rounds[1].retrieval_params.object_focus, "cat")
        self.assertEqual(trace.final_candidate_id, "cand_park_cat")

    def test_fixed_profile_keeps_params_constant(self) -> None:
        controller = ScriptedController(
            self.backend,
            policy=resolve_scripted_policy(
                profile="fixed",
                fixed_video_weight=0.7,
                fixed_audio_weight=0.3,
                fixed_object_focus="none",
                fixed_temporal_focus="global",
            ),
        )
        trace = controller.run("q_audio_cheer")
        self.assertGreaterEqual(len(trace.rounds), 1)
        first_params = trace.rounds[0].retrieval_params
        self.assertAlmostEqual(first_params.video_weight, 0.7)
        self.assertAlmostEqual(first_params.audio_weight, 0.3)
        for round_row in trace.rounds[1:]:
            self.assertEqual(round_row.retrieval_params.to_dict(), first_params.to_dict())
        self.assertEqual(trace.planner_metadata["profile"], "fixed")
        self.assertFalse(trace.planner_metadata["adaptive_params"])
        self.assertFalse(trace.planner_metadata["target_visible_during_runtime"])
        self.assertTrue(trace.planner_metadata["success_computed_offline"])

    def test_single_round_fixed_profile_stops_after_one_round(self) -> None:
        controller = ScriptedController(
            self.backend,
            policy=resolve_scripted_policy(profile="single-round-fixed"),
        )
        trace = controller.run("q_audio_cheer")
        self.assertEqual(len(trace.rounds), 1)
        self.assertEqual(trace.planner_metadata["profile"], "single-round-fixed")


if __name__ == "__main__":
    unittest.main()
