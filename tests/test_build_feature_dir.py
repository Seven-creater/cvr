from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.build_feature_dir import build_feature_dir, build_rows
from app.retriever import load_jsonl


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


class BuildFeatureDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.video_root = self.root / "videos"
        self.audio_root = self.root / "audios"
        self.video_root.mkdir()
        self.audio_root.mkdir()

        (self.video_root / "video1.mp4").write_bytes(b"video-1")
        (self.video_root / "video2.mp4").write_bytes(b"video-2")
        (self.audio_root / "video1.wav").write_bytes(b"audio-1")

        self.msrvtt_json = self.root / "MSRVTT_data.json"
        payload = {
            "videos": [
                {"video_id": "video1"},
                {"video_id": "video2"},
            ],
            "sentences": [
                {"video_id": "video1", "caption": "person plays guitar", "sen_id": 101},
                {"video_id": "video1", "caption": "stage performance", "sen_id": 102},
                {"video_id": "video2", "caption": "dog runs in park", "sen_id": 201},
            ],
        }
        self.msrvtt_json.write_text(json.dumps(payload), encoding="utf-8")

        self.split_csv = self.root / "split.csv"
        write_csv(self.split_csv, [["video_id"], ["video1"]])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_rows_applies_split_and_media_paths(self) -> None:
        text_rows, video_rows = build_rows(
            msrvtt_json_path=self.msrvtt_json,
            split_csv_path=self.split_csv,
            video_root=self.video_root,
            audio_root=self.audio_root,
        )

        self.assertEqual([row["video_id"] for row in video_rows], ["video1"])
        self.assertEqual(len(text_rows), 2)
        self.assertEqual(text_rows[0]["text_id"], "101")
        self.assertTrue(video_rows[0]["video_path"].endswith("video1.mp4"))
        self.assertTrue(video_rows[0]["audio_path"].endswith("video1.wav"))

    def test_build_feature_dir_writes_rows_and_embeddings(self) -> None:
        output_dir = self.root / "features"
        text_embeddings = self.root / "text_in.npy"
        video_visual_embeddings = self.root / "video_visual_in.npy"
        video_audio_embeddings = self.root / "video_audio_in.npy"

        np.save(text_embeddings, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        np.save(video_visual_embeddings, np.asarray([[1.0, 0.0]], dtype=np.float32))
        np.save(video_audio_embeddings, np.asarray([[0.3, 0.7]], dtype=np.float32))

        summary = build_feature_dir(
            msrvtt_json_path=self.msrvtt_json,
            output_dir=output_dir,
            split_csv_path=self.split_csv,
            video_root=self.video_root,
            audio_root=self.audio_root,
            text_embeddings_in=text_embeddings,
            video_visual_embeddings_in=video_visual_embeddings,
            video_audio_embeddings_in=video_audio_embeddings,
        )

        self.assertEqual(summary.video_count, 1)
        self.assertEqual(summary.text_count, 2)
        self.assertTrue((output_dir / "text_embeddings.npy").exists())
        self.assertTrue((output_dir / "video_visual_embeddings.npy").exists())
        self.assertTrue((output_dir / "video_audio_embeddings.npy").exists())
        self.assertEqual(len(load_jsonl(output_dir / "text_rows.jsonl")), 2)
        self.assertEqual(len(load_jsonl(output_dir / "video_rows.jsonl")), 1)

    def test_build_feature_dir_rejects_embedding_length_mismatch(self) -> None:
        output_dir = self.root / "features_bad"
        text_embeddings = self.root / "text_bad.npy"
        np.save(text_embeddings, np.asarray([[1.0, 0.0]], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "text_embeddings row count does not match text_rows.jsonl"):
            build_feature_dir(
                msrvtt_json_path=self.msrvtt_json,
                output_dir=output_dir,
                split_csv_path=self.split_csv,
                video_root=self.video_root,
                text_embeddings_in=text_embeddings,
            )
