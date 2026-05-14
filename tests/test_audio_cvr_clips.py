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
            self.assertEqual({"too_few_segments:1": 1}, summary["skipped_counts_by_dataset"]["daily_omni"])

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

    def test_scans_known_server_dataset_layouts_instead_of_only_video_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            raw = root / "raw"
            video_paths = [
                raw / "daily_omni" / "video" / "daily.mp4",
                raw / "hdtf" / "videos" / "hdtf_video.mp4",
                raw / "hdtf" / "clips" / "hdtf_clip.mp4",
                raw / "avatar" / "avatar_root.mp4",
                raw / "avatar" / "video" / "avatar_video.mp4",
                raw / "vggsound" / "scratch" / "class_a" / "vgg.mp4",
                raw / "vgg_monoaudio" / "inter_class" / "mixed" / "mono.mp4",
                raw / "worldsense" / "videos" / "world.mp4",
                raw / "VoxCeleb" / "vox.mp4",
            ]
            ignored_paths = [
                raw / "daily_omni" / "other" / "ignored.mp4",
                raw / "vgg_monoaudio" / "intra_class" / "ignored.mp4",
            ]
            for path in video_paths + ignored_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")

            def fake_probe(path: Path) -> dict:
                return {"has_video": True, "has_audio": True, "duration_seconds": 20.0}

            datasets = ["daily_omni", "hdtf", "avatar", "vggsound", "vgg_monoaudio", "worldsense", "VoxCeleb"]
            with mock.patch("app.audio_cvr_clips.probe_media", side_effect=fake_probe):
                summary = build_audio_cvr_clips(
                    root=root,
                    datasets=datasets,
                    exclude_datasets=["VoxCeleb"],
                    clip_seconds=10,
                    min_clip_seconds=8,
                    max_clip_seconds=12,
                    dry_run=True,
                )

            self.assertEqual(
                {
                    "daily_omni": 1,
                    "hdtf": 2,
                    "avatar": 2,
                    "vggsound": 1,
                    "vgg_monoaudio": 1,
                    "worldsense": 1,
                },
                summary["discovered_video_counts"],
            )
            self.assertNotIn("VoxCeleb", summary["dataset_names"])
            self.assertEqual(8, summary["source_video_count"])
            self.assertEqual(16, summary["segment_count"])
            self.assertTrue(any(path.endswith("raw/vggsound/scratch") for path in summary["dataset_scan_roots"]["vggsound"]))
            self.assertTrue(any(path.endswith("raw/vgg_monoaudio/inter_class/mixed") for path in summary["dataset_scan_roots"]["vgg_monoaudio"]))

    def test_reports_avatar_single_clip_sources_as_too_few_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            raw = root / "raw" / "avatar"
            raw.mkdir(parents=True)
            for index in range(3):
                (raw / f"avatar_{index}.mp4").write_bytes(b"video")

            def fake_probe(path: Path) -> dict:
                return {"has_video": True, "has_audio": True, "duration_seconds": 10.0}

            with mock.patch("app.audio_cvr_clips.probe_media", side_effect=fake_probe):
                summary = build_audio_cvr_clips(
                    root=root,
                    datasets=["avatar"],
                    clip_seconds=10,
                    min_clip_seconds=8,
                    max_clip_seconds=12,
                    min_clips_per_source=2,
                    dry_run=True,
                )

            self.assertEqual({"avatar": 3}, summary["discovered_video_counts"])
            self.assertEqual(0, summary["source_video_count"])
            self.assertEqual({"too_few_segments:1": 3}, summary["skipped_counts_by_dataset"]["avatar"])

    def test_6_9_tail_segments_make_10s_sources_pairable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            raw = root / "raw" / "avatar"
            raw.mkdir(parents=True)
            source = raw / "avatar.mp4"
            source.write_bytes(b"video")

            def fake_probe(path: Path) -> dict:
                return {"has_video": True, "has_audio": True, "duration_seconds": 10.0}

            with mock.patch("app.audio_cvr_clips.probe_media", side_effect=fake_probe):
                summary = build_audio_cvr_clips(
                    root=root,
                    output_root=root / "clips" / "audio_cvr_6_9s",
                    datasets=["avatar"],
                    clip_seconds=8,
                    min_clip_seconds=6,
                    max_clip_seconds=9,
                    min_clips_per_source=2,
                    include_tail_segment=True,
                    dry_run=True,
                )

            records = [
                json.loads(line)
                for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, summary["source_video_count"])
            self.assertEqual(2, summary["segment_count"])
            self.assertEqual([0.0, 2.0], [record["start_seconds"] for record in records])
            self.assertEqual([8.0, 10.0], [record["end_seconds"] for record in records])
            self.assertTrue(summary["include_tail_segment"])
            self.assertTrue(summary["manifest_path"].endswith("audio_cvr_6_9s_clips.jsonl"))

    def test_dataset_video_root_override_excludes_hdtf_short_clip_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            raw = root / "raw" / "hdtf"
            video = raw / "videos" / "long.mp4"
            short_clip = raw / "clips" / "short.mp4"
            video.parent.mkdir(parents=True)
            short_clip.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            short_clip.write_bytes(b"video")

            def fake_probe(path: Path) -> dict:
                return {"has_video": True, "has_audio": True, "duration_seconds": 30.0}

            with mock.patch("app.audio_cvr_clips.probe_media", side_effect=fake_probe):
                summary = build_audio_cvr_clips(
                    root=root,
                    datasets=["hdtf"],
                    dataset_video_roots={"hdtf": ("videos",)},
                    clip_seconds=8,
                    min_clip_seconds=6,
                    max_clip_seconds=9,
                    include_tail_segment=True,
                    dry_run=True,
                )

            self.assertEqual({"hdtf": 1}, summary["discovered_video_counts"])
            self.assertEqual(1, summary["source_video_count"])
            self.assertEqual(4, summary["segment_count"])
            self.assertTrue(all(path.endswith("raw/hdtf/videos") for path in summary["dataset_scan_roots"]["hdtf"]))

    def test_clip_extraction_writes_temp_file_then_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            raw = root / "raw" / "daily_omni" / "video"
            raw.mkdir(parents=True)
            source = raw / "daily.mp4"
            source.write_bytes(b"video")

            def fake_probe(path: Path) -> dict:
                return {"has_video": True, "has_audio": True, "duration_seconds": 20.0}

            def fake_run(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                Path(command[-1]).write_bytes(b"complete clip")

            with mock.patch("app.audio_cvr_clips.probe_media", side_effect=fake_probe), mock.patch(
                "app.audio_cvr_clips.subprocess.run", side_effect=fake_run
            ):
                summary = build_audio_cvr_clips(
                    root=root,
                    datasets=["daily_omni"],
                    clip_seconds=10,
                    min_clip_seconds=8,
                    max_clip_seconds=12,
                )

            records = [
                json.loads(line)
                for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            output_paths = [root / record["output_path"] for record in records]
            self.assertEqual(2, len(output_paths))
            self.assertTrue(all(path.read_bytes() == b"complete clip" for path in output_paths))
            self.assertFalse(list(output_paths[0].parent.glob("*.tmp.*.mp4")))


if __name__ == "__main__":
    unittest.main()
