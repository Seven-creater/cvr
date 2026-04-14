from __future__ import annotations

import unittest

from app.omni_checker import CheckerResult, build_t2v_user_content, build_v2t_user_content
from app.retriever import TextRow, VideoRow


class OmniCheckerTests(unittest.TestCase):
    def test_t2v_prompt_does_not_leak_labels(self) -> None:
        content = build_t2v_user_content(
            "a man plays guitar on stage",
            VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
            rank=1,
            score=0.9,
        )
        prompt = content[0]["text"]
        self.assertNotIn("target_video_id", prompt)
        self.assertNotIn("ground truth", prompt.lower())
        self.assertEqual(content[1]["video_url"]["url"], "/tmp/video1.mp4")

    def test_v2t_prompt_does_not_leak_labels(self) -> None:
        content = build_v2t_user_content(
            VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
            TextRow(text_id="t1", video_id="video1", text="a man plays guitar on stage"),
            rank=1,
            score=0.9,
        )
        prompt = content[0]["text"]
        self.assertNotIn("target", prompt.lower())

    def test_checker_result_round_trip(self) -> None:
        result = CheckerResult.from_dict(
            {
                "is_match": True,
                "confidence": 0.7,
                "visual_match": 0.8,
                "audio_match": 0.3,
                "main_events": ["guitar"],
                "missing_elements": [],
                "reason": "ok",
                "rewrite_suggestion": "same",
            }
        )
        self.assertTrue(result.to_dict()["is_match"])
