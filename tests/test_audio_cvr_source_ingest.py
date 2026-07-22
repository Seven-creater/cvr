from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.audio_cvr_source_ingest import (
    DatasetSpec,
    _download_dataset,
    _load_jsonl,
    assess_pilot_yield,
    extend_frozen_test,
    parse_avqa_video_identity,
    prepare_mirror_sources,
    prepare_stratified_clip_pilot,
)


class AudioCvrSourceIngestTests(unittest.TestCase):
    def test_partial_mirror_download_keeps_materialized_media_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "avqa_videos"
            destination.mkdir()
            (destination / "usable.mp4").write_bytes(b"v" * 8192)
            spec = DatasetSpec("avqa_videos", "example/avqa", "test", "avqa")

            with mock.patch("app.audio_cvr_source_ingest.shutil.which", return_value="hf"), mock.patch(
                "app.audio_cvr_source_ingest.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["hf", "download"]),
            ):
                summary = _download_dataset(
                    spec,
                    destination,
                    hf_endpoint="https://hf-mirror.com",
                    resume=True,
                    allow_partial=True,
                )

            self.assertEqual("partial_after_download_error", summary["status"])
            self.assertEqual(1, summary["materialized_media_count"])
            self.assertTrue((destination / ".audio_cvr_download_partial.json").exists())
            self.assertFalse((destination / ".audio_cvr_download_complete").exists())

    def test_jsonl_resume_preserves_and_truncates_only_incomplete_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.jsonl"
            path.write_bytes(b'{"sample_id":"ok"}\n{"sample_id":"partial')

            self.assertEqual([{"sample_id": "ok"}], _load_jsonl(path))
            self.assertEqual(b'{"sample_id":"ok"}\n', path.read_bytes())
            backups = list(path.parent.glob("progress.jsonl.incomplete_tail.*"))
            self.assertEqual(1, len(backups))
            self.assertEqual(b'{"sample_id":"partial', backups[0].read_bytes())

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    @staticmethod
    def _row(sample_id: str, subtype: str, source_id: str, *, dataset: str = "avqa_videos") -> dict:
        return {
            "sample_id": sample_id,
            "proposal_id": sample_id,
            "source_disjoint_group_id": source_id,
            "pair_group_id": f"pair_{sample_id}",
            "reference_video": f"clips/{sample_id}_ref.mp4",
            "target_video": f"clips/{sample_id}_tgt.mp4",
            "edit_text": "replace quiet ambience with applause" if subtype == "sound_event" else "replace piano music with guitar music",
            "b_subtype": subtype,
            "dataset": dataset,
            "accepted": True,
            "fallback": False,
            "manual_review_required": False,
            "split_tier": "main",
            "audio_delta_strength": 0.8,
            "video_context_strength": 0.7,
        }

    def test_avqa_identity_removes_only_final_time_suffix(self) -> None:
        self.assertEqual("youtube_id_with_under-score", parse_avqa_video_identity("youtube_id_with_under-score_000123"))
        self.assertEqual("youtube_id_with_under-score", parse_avqa_video_identity("youtube_id_with_under-score_-12.5"))
        self.assertEqual("youtube_id_with_under-score", parse_avqa_video_identity("youtube_id_with_under-score"))

    def test_extend_frozen_test_preserves_existing_rows_and_exact_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing_path = root / "test150.jsonl"
            candidate_path = root / "candidates.jsonl"
            output_dir = root / "output"
            existing = [
                self._row("old_sound", "sound_event", "old_source_sound", dataset="avatar"),
                self._row("old_music", "music", "old_source_music", dataset="avatar"),
            ]
            candidates = [
                self._row("new_sound", "sound_event", "new_source_sound"),
                self._row("new_music", "music", "new_source_music"),
                self._row("reserve_sound", "sound_event", "reserve_source_sound"),
            ]
            self._write_jsonl(existing_path, existing)
            self._write_jsonl(candidate_path, candidates)

            summary = extend_frozen_test(
                existing_test_path=existing_path,
                candidate_path=candidate_path,
                output_dir=output_dir,
                target_count=4,
                sound_event_target=2,
                music_target=2,
            )

            frozen = [json.loads(line) for line in (output_dir / "test_main_4.jsonl").read_text(encoding="utf-8").splitlines()]
            reserve = [json.loads(line) for line in (output_dir / "test1000_reserve_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["old_sound", "old_music"], [row["sample_id"] for row in frozen[:2]])
            self.assertEqual({"sound_event": 2, "music": 2}, summary["subtype_counts"])
            self.assertEqual(0, summary["audit"]["violation_count"])
            self.assertEqual(["reserve_sound"], [row["sample_id"] for row in reserve])

    def test_extend_frozen_test_rejects_source_overlap_and_shortage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing_path = root / "test150.jsonl"
            candidate_path = root / "candidates.jsonl"
            existing = [
                self._row("old_sound", "sound_event", "shared_source", dataset="avatar"),
                self._row("old_music", "music", "old_music_source", dataset="avatar"),
            ]
            candidates = [self._row("overlap_sound", "sound_event", "shared_source")]
            self._write_jsonl(existing_path, existing)
            self._write_jsonl(candidate_path, candidates)

            with self.assertRaisesRegex(ValueError, "not enough eligible"):
                extend_frozen_test(
                    existing_test_path=existing_path,
                    candidate_path=candidate_path,
                    output_dir=root / "output",
                    target_count=4,
                    sound_event_target=2,
                    music_target=2,
                )

    def test_assess_pilot_yield_distinguishes_go_borderline_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing_path = root / "test150.jsonl"
            self._write_jsonl(existing_path, [self._row("old", "sound_event", "old_source", dataset="avatar")])

            cases = (
                ("go", 4, 2, "GO"),
                ("borderline", 3, 1, "BORDERLINE"),
                ("fail", 2, 0, "FAIL"),
            )
            for name, sound_count, music_count, expected in cases:
                run_root = root / name
                main_rows = [self._row(f"{name}_s{i}", "sound_event", f"{name}_ss{i}") for i in range(sound_count)]
                main_rows += [self._row(f"{name}_m{i}", "music", f"{name}_ms{i}") for i in range(music_count)]
                ranked_rows = [dict(row, accepted=True) for row in main_rows]
                while len(ranked_rows) < 10:
                    ranked_rows.append({"proposal_id": f"rejected_{name}_{len(ranked_rows)}", "accepted": False})
                self._write_jsonl(run_root / "b_main_audio_cvr_triplets.jsonl", main_rows)
                self._write_jsonl(run_root / "b_ranked_single_source_pairs.jsonl", ranked_rows)

                summary = assess_pilot_yield(
                    run_root=run_root,
                    existing_test_path=existing_path,
                    output_dir=run_root / "assessment",
                    requested_candidates=10,
                    full_candidate_target=100,
                    min_total=6,
                    min_sound_event=4,
                    min_music=2,
                    borderline_total=4,
                    borderline_sound_event=3,
                    borderline_music=1,
                )

                self.assertEqual(expected, summary["decision"])
                self.assertTrue((run_root / "assessment" / "pilot_assessment.md").exists())
                self.assertFalse(summary["selection_uses_model_scores"])

    def test_stratified_clip_pilot_is_balanced_deterministic_and_preserves_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clips: list[dict] = []
            groups: list[dict] = []
            for dataset in ("avqa_videos", "existing_vggsound", "avscapbench"):
                for group_index in range(4):
                    group_id = f"{dataset}_group_{group_index}"
                    clip_ids = [f"{group_id}_clip_{clip_index}" for clip_index in range(2)]
                    groups.append({"group_id": group_id, "dataset": dataset, "candidate_clip_ids": clip_ids})
                    clips.extend(
                        {"clip_id": clip_id, "group_id": group_id, "dataset": dataset, "output_path": f"clips/{clip_id}.mp4"}
                        for clip_id in clip_ids
                    )
            annotated_clip = "avqa_videos_group_3_clip_0"
            self._write_jsonl(root / "clips.jsonl", clips)
            self._write_jsonl(root / "groups.jsonl", groups)
            self._write_jsonl(root / "annotations.jsonl", [{"clip_id": annotated_clip}])

            first = prepare_stratified_clip_pilot(
                clips_manifest_path=root / "clips.jsonl",
                clip_groups_path=root / "groups.jsonl",
                existing_annotations_path=root / "annotations.jsonl",
                output_dir=root / "pilot_a",
                datasets=["avqa_videos", "existing_vggsound", "avscapbench"],
                groups_per_dataset=2,
                seed=7,
            )
            second = prepare_stratified_clip_pilot(
                clips_manifest_path=root / "clips.jsonl",
                clip_groups_path=root / "groups.jsonl",
                existing_annotations_path=root / "annotations.jsonl",
                output_dir=root / "pilot_b",
                datasets=["avqa_videos", "existing_vggsound", "avscapbench"],
                groups_per_dataset=2,
                seed=7,
            )

            selected_clips = _load_jsonl(root / "pilot_a" / "pilot_clips_to_annotate.jsonl")
            selected_ids = {row["clip_id"] for row in selected_clips}
            self.assertIn(annotated_clip, selected_ids)
            self.assertEqual(6, first["pilot_group_count"])
            self.assertEqual(12, first["pilot_clip_count"])
            self.assertEqual(
                {"avqa_videos": 2, "existing_vggsound": 2, "avscapbench": 2},
                first["selected_dataset_group_counts"],
            )
            self.assertEqual(first["selected_dataset_group_counts"], second["selected_dataset_group_counts"])
            self.assertEqual(
                (root / "pilot_a" / "pilot_clips_to_annotate.jsonl").read_text(encoding="utf-8"),
                (root / "pilot_b" / "pilot_clips_to_annotate.jsonl").read_text(encoding="utf-8"),
            )
            self.assertFalse(first["selection_uses_model_scores"])

    def test_source_ingest_journals_each_decision_and_resumes_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            run_root = Path(temp_dir) / "run"
            source = root / "raw" / "vggsound" / "scratch" / "event.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"video-with-audio")
            media = {"has_video": True, "has_audio": True, "duration_seconds": 10.0}

            with mock.patch("app.audio_cvr_source_ingest.probe_media", return_value=media):
                first = prepare_mirror_sources(
                    root=root,
                    run_root=run_root,
                    datasets=["existing_vggsound"],
                    source_targets={"existing_vggsound": 1},
                    skip_download=True,
                    materialize_mode="copy",
                )
                resumed = prepare_mirror_sources(
                    root=root,
                    run_root=run_root,
                    datasets=["existing_vggsound"],
                    source_targets={"existing_vggsound": 1},
                    skip_download=True,
                    materialize_mode="copy",
                    resume=True,
                )

            progress_rows = [
                json.loads(line)
                for line in (run_root / "source_ingest.progress.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, first["selected_source_count"])
            self.assertEqual(1, resumed["selected_source_count"])
            self.assertEqual(1, len(progress_rows))
            self.assertEqual("selected", progress_rows[0]["decision"])

            with self.assertRaisesRegex(ValueError, "pass --resume"):
                prepare_mirror_sources(
                    root=root,
                    run_root=run_root,
                    datasets=["existing_vggsound"],
                    source_targets={"existing_vggsound": 1},
                    skip_download=True,
                    materialize_mode="copy",
                )


if __name__ == "__main__":
    unittest.main()
