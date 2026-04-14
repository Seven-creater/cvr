from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.msrvtt_retrieval import (
    evaluate_bidirectional,
    load_msrvtt_dataset,
    parse_topk_values,
    retrieve_texts_from_video,
    retrieve_videos_from_text,
)


FAKE_MSRVTT = {
    "videos": [
        {"video_id": "video1"},
        {"video_id": "video2"},
    ],
    "sentences": [
        {"video_id": "video1", "caption": "a man plays guitar on stage"},
        {"video_id": "video1", "caption": "a musician performs with a guitar"},
        {"video_id": "video2", "caption": "a dog runs through the park"},
        {"video_id": "video2", "caption": "a puppy is running outside"},
    ],
}


class MsrvttRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.json_path = self.root / "MSRVTT_data.json"
        self.split_path = self.root / "MSRVTT_JSFUSION_test.csv"
        self.json_path.write_text(json.dumps(FAKE_MSRVTT, ensure_ascii=False), encoding="utf-8")
        self.split_path.write_text("video_id\nvideo1\nvideo2\n", encoding="utf-8")
        self.dataset = load_msrvtt_dataset(self.json_path, self.split_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_text_to_video_returns_expected_top1(self) -> None:
        hits = retrieve_videos_from_text("guitar performance on stage", self.dataset, topk=2)
        self.assertEqual(hits[0].video_id, "video1")
        self.assertEqual(len(hits), 2)

    def test_video_to_text_returns_same_video_captions_first(self) -> None:
        hits = retrieve_texts_from_video("video2", self.dataset, topk=2)
        self.assertTrue(all(hit.video_id == "video2" for hit in hits))

    def test_bidirectional_metrics_are_perfect_on_easy_toy_data(self) -> None:
        metrics = evaluate_bidirectional(self.dataset, ks=[1, 5, 10])
        self.assertEqual(metrics["t2v"]["R@1"], 1.0)
        self.assertEqual(metrics["v2t"]["R@1"], 1.0)

    def test_parse_topk_values_normalizes_duplicates(self) -> None:
        self.assertEqual(parse_topk_values("10,1,5,5"), [1, 5, 10])


if __name__ == "__main__":
    unittest.main()
