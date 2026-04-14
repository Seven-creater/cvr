import json
import tempfile
import unittest
from pathlib import Path

from app.bandit_data import trace_to_bandit_samples, write_bandit_samples
from app.backends import MockRetrievalBackend
from app.controller import ScriptedController, resolve_scripted_policy


class BanditDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = MockRetrievalBackend()
        self.controller = ScriptedController(
            backend=self.backend,
            policy=resolve_scripted_policy("adaptive"),
        )

    def _run_trace(self, query_id: str):
        query = self.backend.get_query(query_id)
        trace = self.controller.run(query_id)
        trace.query.target_video_id = query.target_video_id
        trace.success = (
            trace.final_candidate_id == query.target_video_id
            if query.target_video_id
            else None
        )
        return query, trace

    def test_bandit_samples_hide_target_from_state(self) -> None:
        query, trace = self._run_trace("q_audio_cheer")
        samples = trace_to_bandit_samples(self.backend, trace, query)
        self.assertGreaterEqual(len(samples), 2)
        for sample in samples:
            self.assertNotIn("target_video_id", sample.state)
            self.assertFalse(sample.state["target_visible_during_runtime"])

    def test_retry_reward_is_positive_when_target_rank_improves(self) -> None:
        query, trace = self._run_trace("q_audio_cheer")
        samples = trace_to_bandit_samples(self.backend, trace, query)
        retry_samples = [
            sample for sample in samples
            if sample.action.action_type == "retry" and sample.round_index > 0
        ]
        self.assertEqual(len(retry_samples), 1)
        self.assertEqual(retry_samples[0].action.action_id, "retry_audio_boost")
        self.assertGreater(retry_samples[0].reward, 0.0)
        self.assertEqual(retry_samples[0].reward_breakdown["prev_target_rank"], 2)
        self.assertEqual(retry_samples[0].reward_breakdown["next_target_rank"], 1)

    def test_submit_sample_carries_offline_success(self) -> None:
        query, trace = self._run_trace("q_temporal_piano")
        samples = trace_to_bandit_samples(self.backend, trace, query)
        terminal = samples[-1]
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.action.action_type, "submit")
        self.assertTrue(terminal.final_success)
        self.assertEqual(terminal.reward_breakdown["success"], True)

    def test_write_bandit_samples_outputs_jsonl(self) -> None:
        query, trace = self._run_trace("q_object_cat")
        samples = trace_to_bandit_samples(self.backend, trace, query)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = write_bandit_samples(Path(tmpdir) / "samples.jsonl", samples)
            rows = output_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(rows), len(samples))
        first_row = json.loads(rows[0])
        self.assertEqual(first_row["query_id"], query.query_id)
        self.assertEqual(first_row["action"]["action_type"], "retry")
        self.assertFalse(first_row["state"]["target_visible_during_runtime"])


if __name__ == "__main__":
    unittest.main()
