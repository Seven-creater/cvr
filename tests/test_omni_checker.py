from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.omni_checker import (
    RetrievalHints,
    T2VQueryUnderstanding,
    VideoDescription,
    _materialize_video_url,
    build_t2v_query_understanding_user_content,
    build_t2v_rerank_user_content,
    build_v2t_rerank_user_content,
    build_video_description_user_content,
)
from app.retrieval_types import VideoRow


class OmniCheckerTests(unittest.TestCase):
    def test_t2v_query_prompt_uses_plain_text_only(self) -> None:
        content = build_t2v_query_understanding_user_content("a man plays guitar on stage")
        self.assertEqual("text", content[0]["type"])
        prompt = content[0]["text"]
        self.assertNotIn("ground truth", prompt.lower())
        self.assertIn("Original query", prompt)

    def test_video_description_prompt_keeps_video_url(self) -> None:
        content = build_video_description_user_content(
            VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
        )
        self.assertEqual("/tmp/video1.mp4", content[0]["video_url"]["url"])
        self.assertIn("describe this video", content[1]["text"].lower())

    def test_query_understanding_normalizes_audio_relevance(self) -> None:
        result = T2VQueryUnderstanding.from_dict(
            {
                "retrieval_text": "better cooking scene",
                "summary": "person cooks",
                "main_events": ["cooking"],
                "objects": ["pan"],
                "scene": "kitchen",
                "audio_cues": ["sizzling"],
                "audio_relevance": "invalid-value",
                "reason": "fallback to unknown",
            },
            original_query_text="person cooks food",
        )
        hints = RetrievalHints.from_query_understanding("person cooks food", result)
        self.assertEqual("unknown", result.audio_relevance)
        self.assertEqual("better cooking scene", hints.query_text_override)
        self.assertEqual("on", hints.audio_mode)

    def test_rerank_prompts_embed_structured_context(self) -> None:
        query_understanding = T2VQueryUnderstanding(
            retrieval_text="cook in kitchen",
            summary="person cooks",
            main_events=["stirs food"],
            objects=["pot"],
            scene="kitchen",
            audio_cues=["sizzling"],
            audio_relevance="helpful",
            reason="clear cooking intent",
        )
        t2v_content = build_t2v_rerank_user_content(
            query_understanding,
            [{"video_id": "video1", "original_rank": 1, "video_description": {"summary": "person cooks"}}],
        )
        self.assertIn("video1", t2v_content[0]["text"])
        self.assertIn("cook in kitchen", t2v_content[0]["text"])

        video_description = VideoDescription(
            summary="person repairs a computer",
            main_events=["opens computer case"],
            objects=["desktop computer"],
            scene="desk",
            audio_cues=[],
            audio_relevance="irrelevant",
        )
        v2t_content = build_v2t_rerank_user_content(
            video_description,
            [{"text_id": "t1", "text": "a person opens a computer"}],
        )
        self.assertIn("t1", v2t_content[0]["text"])
        self.assertIn("person repairs a computer", v2t_content[0]["text"])

    def test_local_video_path_is_encoded_as_data_url(self) -> None:
        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-omni-checker-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            path = tmp_dir / "clip.mp4"
            path.write_bytes(b"fake-mp4-bytes")
            encoded = _materialize_video_url(str(path))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        self.assertTrue(encoded.startswith("data:video/mp4;base64,"))


if __name__ == "__main__":
    unittest.main()
