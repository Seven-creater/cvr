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
        trace = controller.run("real_q_audio")
        self.assertEqual(trace.final_candidate_id, "real_band_cheer")
        self.assertTrue(trace.success)
        self.assertLessEqual(len(trace.rounds), 3)


if __name__ == "__main__":
    unittest.main()

