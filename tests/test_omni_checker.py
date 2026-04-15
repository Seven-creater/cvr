from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.omni_checker import (
    CheckerResult,
    _complete_payload,
    _fallback_payload,
    _materialize_video_url,
    build_t2v_user_content,
    build_v2t_user_content,
)
from app.retriever import TextRow, VideoRow


class OmniCheckerTests(unittest.TestCase):
    def test_t2v_prompt_does_not_leak_labels(self) -> None:
        content = build_t2v_user_content(
            "a man plays guitar on stage",
            VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
            rank=1,
            score=0.9,
        )
        prompt = content[1]["text"]
        self.assertNotIn("target_video_id", prompt)
        self.assertNotIn("ground truth", prompt.lower())
        self.assertEqual(content[0]["video_url"]["url"], "/tmp/video1.mp4")

    def test_v2t_prompt_does_not_leak_labels(self) -> None:
        content = build_v2t_user_content(
            VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
            TextRow(text_id="t1", video_id="video1", text="a man plays guitar on stage"),
            rank=1,
            score=0.9,
        )
        prompt = content[1]["text"]
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

    def test_checker_result_tolerates_non_numeric_fields(self) -> None:
        result = CheckerResult.from_dict(
            {
                "is_match": "true",
                "confidence": "0.7",
                "visual_match": "The person is cooking.",
                "audio_match": None,
                "main_events": ["cooking"],
                "missing_elements": [],
                "reason": "free text",
                "rewrite_suggestion": "same",
            }
        )
        self.assertTrue(result.is_match)
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.visual_match, 0.0)
        self.assertEqual(result.audio_match, 0.0)

    def test_fallback_payload_wraps_unstructured_text(self) -> None:
        payload = _fallback_payload("plain answer")
        result = CheckerResult.from_dict(payload)
        self.assertFalse(result.is_match)
        self.assertEqual(result.reason, "plain answer")
        self.assertIn("unstructured_response", result.missing_elements)

    def test_complete_payload_marks_missing_fields(self) -> None:
        payload = _complete_payload({"is_match": True})
        result = CheckerResult.from_dict(payload)
        self.assertTrue(result.is_match)
        self.assertEqual(result.reason, "incomplete_json_response")
        self.assertIn("missing_field:confidence", result.missing_elements)
        self.assertIn("missing_field:rewrite_suggestion", result.missing_elements)

    def test_local_video_path_is_encoded_as_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "clip.mp4"
            path.write_bytes(b"fake-mp4-bytes")
            encoded = _materialize_video_url(str(path))
        self.assertTrue(encoded.startswith("data:video/mp4;base64,"))
