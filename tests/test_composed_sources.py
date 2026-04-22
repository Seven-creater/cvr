from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from app.composed_sources import prepare_source_datasets


class ComposedSourcesTests(unittest.TestCase):
    def test_prepare_source_datasets_scans_media_when_no_parquet_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daily = root / "raw_datasets" / "daily_omni"
            world = root / "raw_datasets" / "worldsense"
            daily.mkdir(parents=True)
            world.mkdir(parents=True)
            (daily / "daily_a.mp4").write_bytes(b"video")
            (world / "world_b.webm").write_bytes(b"video")

            result = prepare_source_datasets(root=root, clip_limit=1)

            self.assertEqual(2, result["row_count"])
            self.assertEqual(2, result["clip_count"])
            self.assertEqual(1, result["pilot_clip_count"])
            rows_path = Path(result["source_rows_path"])
            clips_path = Path(result["source_clips_all_path"])
            self.assertTrue(rows_path.exists())
            self.assertTrue(clips_path.exists())
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({"daily_omni", "worldsense"}, {row["dataset"] for row in rows})

    def test_prepare_source_datasets_selects_balanced_pilot_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daily = root / "raw_datasets" / "daily_omni"
            world = root / "raw_datasets" / "worldsense"
            daily.mkdir(parents=True)
            world.mkdir(parents=True)
            for index in range(4):
                (daily / f"daily_{index}.mp4").write_bytes(b"video")
                (world / f"world_{index}.mp4").write_bytes(b"video")

            result = prepare_source_datasets(root=root, clip_limit=4)

            pilot_path = Path(result["source_clips_pilot_path"])
            clips = [json.loads(line) for line in pilot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(4, len(clips))
            self.assertEqual({"daily_omni": 2, "worldsense": 2}, _count_by_dataset(clips))

    def test_prepare_source_datasets_materializes_embedded_media_from_parquet_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daily = root / "raw_datasets" / "daily_omni"
            world = root / "raw_datasets" / "worldsense"
            daily.mkdir(parents=True)
            world.mkdir(parents=True)
            parquet_path = daily / "train-00000.parquet"
            parquet_path.write_bytes(b"placeholder")

            fake_rows = [
                {
                    "video_id": "sample-one",
                    "video": {"path": "sample-one.mp4", "bytes": b"mp4-bytes"},
                    "audio": {"path": "sample-one.wav", "bytes": b"wav-bytes"},
                    "question": "What sound is present?",
                    "answer": "cat meow",
                }
            ]

            with mock.patch("app.composed_sources._read_parquet_rows", return_value=fake_rows):
                result = prepare_source_datasets(root=root, clip_limit=5)

            rows = [
                json.loads(line)
                for line in Path(result["source_rows_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, len(rows))
            self.assertTrue(Path(rows[0]["video_path"]).exists())
            self.assertTrue(Path(rows[0]["audio_path"]).exists())
            self.assertEqual("What sound is present?", rows[0]["text_fields"]["question"])

            clips = [
                json.loads(line)
                for line in Path(result["source_clips_all_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, len(clips))
            self.assertEqual(["daily_omni"], [clip["dataset"] for clip in clips])

    def test_prepare_source_datasets_extracts_zips_and_resolves_relative_media(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daily = root / "raw_datasets" / "daily_omni"
            world = root / "raw_datasets" / "worldsense"
            daily.mkdir(parents=True)
            world.mkdir(parents=True)
            archive_path = world / "videos.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("videos/sample.mp4", b"video")
            parquet_path = world / "data.parquet"
            parquet_path.write_bytes(b"placeholder")
            fake_rows = [{"video_id": "sample", "video": "./videos/sample.mp4", "question": "What is shown?"}]

            with mock.patch("app.composed_sources._read_parquet_rows", return_value=fake_rows):
                result = prepare_source_datasets(root=root, clip_limit=5)

            rows = [
                json.loads(line)
                for line in Path(result["source_rows_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, len(rows))
            self.assertTrue(Path(rows[0]["video_path"]).exists())
            self.assertIn("_extracted", rows[0]["video_path"])
            self.assertEqual(1, result["dataset_counts"]["worldsense"]["archives"])


if __name__ == "__main__":
    unittest.main()


def _count_by_dataset(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        dataset = record["dataset"]
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts
