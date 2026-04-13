import unittest

from app.backends import FileRetrievalBackend
from app.controller import ScriptedController


class RealBackendReplayTests(unittest.TestCase):
    def test_file_backed_replay(self) -> None:
        backend = FileRetrievalBackend(
            candidates_path="data/real_sample/candidates.json",
            queries_path="data/real_sample/queries.json",
            retrieval_scores_path="data/real_sample/retrieval_scores.json",
        )
        controller = ScriptedController(backend)
        for query in backend.list_queries():
            trace = controller.run(query.query_id)
            self.assertEqual(trace.final_candidate_id, query.target_video_id, query.query_id)
            self.assertTrue(trace.success, query.query_id)
            self.assertLessEqual(len(trace.rounds), 3, query.query_id)


if __name__ == "__main__":
    unittest.main()
