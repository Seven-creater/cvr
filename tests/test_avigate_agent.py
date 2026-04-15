from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from app.avigate_agent import run_t2v_official_agent_case, run_v2t_official_agent_case
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


if __name__ == "__main__":
    unittest.main()
