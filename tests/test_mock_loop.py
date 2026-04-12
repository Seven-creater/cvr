import unittest

from app.backends import MockRetrievalBackend
from app.controller import ScriptedController


class MockLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = MockRetrievalBackend()
        self.controller = ScriptedController(self.backend)

    def test_all_mock_queries_finish_within_three_rounds(self) -> None:
        for query in self.backend.list_queries():
            trace = self.controller.run(query.query_id)
            self.assertLessEqual(len(trace.rounds), 3, query.query_id)
            self.assertEqual(trace.final_candidate_id, query.target_video_id, query.query_id)
            self.assertTrue(trace.success, query.query_id)

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


if __name__ == "__main__":
    unittest.main()

