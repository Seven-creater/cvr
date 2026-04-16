from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import dataclass

import numpy as np

from app.avigate_agent import (
    run_official_agent_partial_eval,
    run_t2v_official_agent_case,
    run_v2t_official_agent_case,
)
from app.omni_checker import MockOmniChecker
from app.retrieval_types import TextRow, VideoRow


@dataclass
class FakeRuntime:
    text_rows: list[TextRow]
    video_rows: list[VideoRow]
    text_score_map: dict[str, np.ndarray]
    video_score_map: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        self._video_index = {row.video_id: index for index, row in enumerate(self.video_rows)}
        self._text_index = {row.text_id: index for index, row in enumerate(self.text_rows)}

    def score_text_query(self, query_text: str) -> np.ndarray:
        return self.text_score_map[query_text]

    def score_video_query(self, video_id: str) -> np.ndarray:
        return self.video_score_map[video_id]

    def target_text_ids(self, video_id: str) -> list[str]:
        return [row.text_id for row in self.text_rows if row.video_id == video_id]


class AvigateAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video_rows = [
            VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
            VideoRow(video_id="video2", video_path="/tmp/video2.mp4"),
            VideoRow(video_id="video3", video_path="/tmp/video3.mp4"),
        ]
        self.text_rows = [
            TextRow(text_id="t1", video_id="video1", text="a person is cooking"),
            TextRow(text_id="t2", video_id="video2", text="a dog is running"),
            TextRow(text_id="t3", video_id="video3", text="a person fixes a computer"),
        ]

    def test_t2v_agent_rewrites_then_submits(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={
                "cook query": np.asarray([0.1, 0.9, 0.2], dtype=np.float32),
                "better cook query": np.asarray([0.95, 0.2, 0.1], dtype=np.float32),
            },
            video_score_map={},
        )
        checker = MockOmniChecker(
            t2v_results={
                "cook query::video2": {
                    "is_match": False,
                    "confidence": 0.2,
                    "visual_match": 0.2,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["cooking"],
                    "reason": "wrong activity",
                    "rewrite_suggestion": "better cook query",
                },
                "cook query::video3": {
                    "is_match": False,
                    "confidence": 0.1,
                    "visual_match": 0.1,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["cooking"],
                    "reason": "still wrong",
                    "rewrite_suggestion": "better cook query",
                },
                "better cook query::video1": {
                    "is_match": True,
                    "confidence": 0.91,
                    "visual_match": 0.9,
                    "audio_match": 0.7,
                    "main_events": ["cooking"],
                    "missing_elements": [],
                    "reason": "matches well",
                    "rewrite_suggestion": "",
                },
            }
        )

        trace = run_t2v_official_agent_case(
            query_text="cook query",
            runtime=runtime,
            checker=checker,
            topk=3,
            max_iter=3,
        )

        self.assertEqual("submit", trace["final_action"])
        self.assertEqual("video1", trace["final_result"]["video_id"])
        self.assertEqual("retry", trace["iterations"][0]["action"])
        self.assertEqual("better cook query", trace["iterations"][0]["next_query"])
        self.assertEqual("submit", trace["iterations"][1]["action"])

    def test_v2t_agent_inspects_more_then_submits(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={},
            video_score_map={
                "video3": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            },
        )
        checker = MockOmniChecker(
            v2t_results={
                "video3::t1": {
                    "is_match": False,
                    "confidence": 0.1,
                    "visual_match": 0.2,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["computer"],
                    "reason": "wrong text",
                    "rewrite_suggestion": "",
                },
                "video3::t2": {
                    "is_match": False,
                    "confidence": 0.2,
                    "visual_match": 0.2,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["computer"],
                    "reason": "still wrong",
                    "rewrite_suggestion": "",
                },
                "video3::t3": {
                    "is_match": True,
                    "confidence": 0.88,
                    "visual_match": 0.9,
                    "audio_match": 0.6,
                    "main_events": ["computer repair"],
                    "missing_elements": [],
                    "reason": "correct text",
                    "rewrite_suggestion": "",
                },
            }
        )

        trace = run_v2t_official_agent_case(
            query_video_id="video3",
            runtime=runtime,
            checker=checker,
            topk=3,
            max_iter=3,
        )

        self.assertEqual("submit", trace["final_action"])
        self.assertEqual("t3", trace["final_result"]["text_id"])
        self.assertEqual("inspect_more", trace["iterations"][0]["action"])
        self.assertEqual("submit", trace["iterations"][1]["action"])

    def test_v2t_agent_prefers_earlier_confident_match(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={},
            video_score_map={
                "video1": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            },
        )
        checker = MockOmniChecker(
            v2t_results={
                "video1::t1": {
                    "is_match": True,
                    "confidence": 0.82,
                    "visual_match": 0.8,
                    "audio_match": 0.5,
                    "main_events": ["cooking"],
                    "missing_elements": [],
                    "reason": "correct enough",
                    "rewrite_suggestion": "",
                },
                "video1::t2": {
                    "is_match": True,
                    "confidence": 0.97,
                    "visual_match": 0.9,
                    "audio_match": 0.7,
                    "main_events": ["running"],
                    "missing_elements": [],
                    "reason": "overconfident false positive",
                    "rewrite_suggestion": "",
                },
            }
        )

        trace = run_v2t_official_agent_case(
            query_video_id="video1",
            runtime=runtime,
            checker=checker,
            topk=3,
            max_iter=3,
        )

        self.assertEqual("submit", trace["final_action"])
        self.assertEqual("t1", trace["final_result"]["text_id"])
        self.assertEqual(1, trace["final_result"]["rank_in_final_search"])

    def test_partial_eval_writes_t2v_summary_and_traces(self) -> None:
        runtime = FakeRuntime(
            text_rows=[
                TextRow(text_id="t1", video_id="video1", text="cook query"),
                TextRow(text_id="t2", video_id="video2", text="run query"),
            ],
            video_rows=self.video_rows,
            text_score_map={
                "cook query": np.asarray([0.9, 0.2, 0.1], dtype=np.float32),
                "run query": np.asarray([0.6, 0.5, 0.4], dtype=np.float32),
            },
            video_score_map={},
        )
        checker = MockOmniChecker(
            t2v_results={
                "cook query::video1": {
                    "is_match": True,
                    "confidence": 0.91,
                    "visual_match": 0.9,
                    "audio_match": 0.6,
                    "main_events": ["cooking"],
                    "missing_elements": [],
                    "reason": "correct",
                    "rewrite_suggestion": "",
                },
                "cook query::video2": {
                    "is_match": False,
                    "confidence": 0.1,
                    "visual_match": 0.1,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["cooking"],
                    "reason": "wrong",
                    "rewrite_suggestion": "",
                },
                "run query::video1": {
                    "is_match": False,
                    "confidence": 0.2,
                    "visual_match": 0.2,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["running"],
                    "reason": "wrong",
                    "rewrite_suggestion": "",
                },
                "run query::video2": {
                    "is_match": True,
                    "confidence": 0.82,
                    "visual_match": 0.8,
                    "audio_match": 0.5,
                    "main_events": ["running"],
                    "missing_elements": [],
                    "reason": "correct",
                    "rewrite_suggestion": "",
                },
            }
        )
        progress_messages: list[str] = []
        runs_dir = Path(__file__).resolve().parents[1] / "runs"
        with tempfile.TemporaryDirectory(dir=runs_dir) as temp_dir:
            result = run_official_agent_partial_eval(
                mode="t2v",
                runtime=runtime,
                checker=checker,
                sample_size=2,
                topk=3,
                max_iter=2,
                submit_threshold=0.7,
                recall_ks=(1, 2),
                output_dir=temp_dir,
                progress=progress_messages.append,
            )

            summary = result["summary"]
            self.assertEqual(2, summary["runs"])
            self.assertEqual({"R@1": 0.5, "R@2": 1.0}, summary["round1_recall"])
            self.assertEqual({"R@1": 0.5, "R@2": 1.0}, summary["final_recall"])
            self.assertEqual(0.0, summary["retry_rate"])
            self.assertEqual(0.5, summary["submit_top1_rate"])
            self.assertEqual(0.5, summary["submit_top2_rate"])

            summary_path = Path(result["summary_path"])
            traces_path = Path(result["traces_path"])
            self.assertTrue(summary_path.exists())
            self.assertTrue(traces_path.exists())
            self.assertEqual(summary, json.loads(summary_path.read_text(encoding="utf-8")))
            self.assertEqual(2, len(traces_path.read_text(encoding="utf-8").strip().splitlines()))
            self.assertTrue(any("start 1/2" in message for message in progress_messages))
            self.assertTrue(any("done 2/2" in message for message in progress_messages))

    def test_partial_eval_counts_v2t_followup_runs(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=[self.video_rows[2]],
            text_score_map={},
            video_score_map={
                "video3": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            },
        )
        checker = MockOmniChecker(
            v2t_results={
                "video3::t1": {
                    "is_match": False,
                    "confidence": 0.1,
                    "visual_match": 0.1,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["computer"],
                    "reason": "wrong",
                    "rewrite_suggestion": "",
                },
                "video3::t2": {
                    "is_match": False,
                    "confidence": 0.2,
                    "visual_match": 0.1,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["computer"],
                    "reason": "still wrong",
                    "rewrite_suggestion": "",
                },
                "video3::t3": {
                    "is_match": True,
                    "confidence": 0.88,
                    "visual_match": 0.9,
                    "audio_match": 0.6,
                    "main_events": ["computer repair"],
                    "missing_elements": [],
                    "reason": "correct",
                    "rewrite_suggestion": "",
                },
            }
        )

        result = run_official_agent_partial_eval(
            mode="v2t",
            runtime=runtime,
            checker=checker,
            sample_size=1,
            topk=3,
            max_iter=3,
            submit_threshold=0.7,
            recall_ks=(1, 3),
        )

        summary = result["summary"]
        self.assertEqual({"R@1": 0.0, "R@3": 1.0}, summary["round1_recall"])
        self.assertEqual({"R@1": 0.0, "R@3": 1.0}, summary["final_recall"])
        self.assertEqual(1.0, summary["retry_rate"])
        self.assertEqual(0.0, summary["submit_top1_rate"])
        self.assertEqual(0.0, summary["submit_top2_rate"])
        self.assertEqual(3.0, summary["avg_checker_calls"])


if __name__ == "__main__":
    unittest.main()
