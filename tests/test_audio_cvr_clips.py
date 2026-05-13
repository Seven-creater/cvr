from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.audio_cvr_clips import build_audio_cvr_clips
from app.composed_data import ensure_layout


class AudioCvrClipsTests(unittest.TestCase):
    def test_builds_default_10s_clips_and_skips_sources_with_too_few_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            raw = root / "raw" / "daily_omni"
            raw.mkdir(parents=True)
            long_video = raw / "long.mp4"
            short_video = raw / "short.mp4"
            long_video.write_bytes(b"video")
            short_video.write_bytes(b"video")

            def fake_probe(path: Path) -> dict:
                return {
                    "has_video": True,
                    "has_audio": True,
                    "duration_seconds": 30.0 if Path(path).name == "long.mp4" else 15.0,
                }

            with mock.patch("app.audio_cvr_clips.probe_media", side_effect=fake_probe):
                summary = build_audio_cvr_clips(
                    root=root,
                    datasets=["daily_omni"],
                    clip_seconds=10,
                    min_clip_seconds=8,
                    max_clip_seconds=12,
                    dry_run=True,
                )

            self.assertEqual(1, summary["source_video_count"])
            self.assertEqual(3, summary["segment_count"])
            self.assertEqual({"daily_omni": 1}, summary["source_counts"])
            self.assertEqual({"too_few_segments:1": 1}, summary["skipped_counts"])

            records = [
                json.loads(line)
                for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([10.0, 10.0, 10.0], [record["duration_seconds"] for record in records])
            self.assertTrue(records[0]["output_path"].startswith("clips/audio_cvr_8_12s/daily_omni_long_"))

    def test_rejects_clip_seconds_outside_8_12_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            with self.assertRaises(ValueError):
                build_audio_cvr_clips(root=root, clip_seconds=6, min_clip_seconds=8, max_clip_seconds=12, dry_run=True)


if __name__ == "__main__":
    unittest.main()
