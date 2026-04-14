from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.retriever import FeatureRetriever, parse_topk_values


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        write_jsonl(
            root / "video_rows.jsonl",
            [
                {"video_id": "video1", "video_path": "/tmp/video1.mp4"},
                {"video_id": "video2", "video_path": "/tmp/video2.mp4"},
            ],
        )
        write_jsonl(
            root / "text_rows.jsonl",
            [
                {"text_id": "t1", "video_id": "video1", "text": "guitar stage music"},
                {"text_id": "t2", "video_id": "video2", "text": "dog park running"},
            ],
        )
        np.save(root / "text_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        np.save(root / "video_visual_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        np.save(root / "video_audio_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        self.retriever = FeatureRetriever.from_feature_dir(root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_retrieve_t2v_with_cached_query(self) -> None:
        hits = self.retriever.retrieve_t2v(text_id="t1", topk=2)
        self.assertEqual(hits[0].video_id, "video1")
        self.assertEqual(hits[1].video_id, "video2")

    def test_retrieve_v2t_with_video_query(self) -> None:
        hits = self.retriever.retrieve_v2t("video2", topk=2)
        self.assertEqual(hits[0].text_id, "t2")

    def test_evaluate_bidirectional(self) -> None:
        metrics = self.retriever.evaluate_bidirectional([1, 5, 10])
        self.assertEqual(metrics["t2v"]["R@1"], 1.0)
        self.assertEqual(metrics["v2t"]["R@1"], 1.0)

    def test_parse_topk_values_normalizes_duplicates(self) -> None:
        self.assertEqual(parse_topk_values("10,1,5,5"), [1, 5, 10])
