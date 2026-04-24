from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.composed_data import (
    _accepted_sample_from_record,
    _action_evidence_score,
    _compose_reject_reason,
    _detect_primary_difference,
    _evidence_from_annotations,
    _difference_strength_score,
    _difference_priority_order,
    _effective_pair_quality,
    _build_pair_candidates,
    _build_proposal_id,
    _finalize_pair_verification,
    _has_intraclip_difference_conflict,
    _judge_accepts,
    _non_speech_audio_event_score,
    _pair_record_acceptance_issues,
    _pair_context_score,
    _pair_verification_counts,
    _select_final_accepted_records,
    _build_fallback_edit_text,
    _speech_evidence_score,
    _speech_specificity_score,
    _source_context,
    _target_uniqueness_score,
    annotate_clips,
    build_seeded_pair_slice,
    build_ffmpeg_extract_command,
    detective_annotate_clips,
    discover_raw_sources,
    ensure_layout,
    extract_clips,
    index_raw_sources,
    main as composed_data_main,
    plan_detective_event_clips,
    propose_group_pairs,
    propose_pairs,
    propose_seeded_pairs,
    validate_pilot_dataset,
)
from app.composed_omni import ALLOWED_DIFFERENCE_TYPES


class ComposedDataTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_ensure_layout_creates_expected_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = ensure_layout(temp_dir)
            for name in ("raw", "clips", "metadata", "captions", "pairs", "splits", "reports", "caches"):
                self.assertTrue(paths[name].exists(), name)
                self.assertTrue(paths[name].is_dir(), name)

    def test_discover_raw_sources_reads_raw_datasets_children(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "raw_datasets" / "daily_omni").mkdir(parents=True)
            (root / "raw_datasets" / "worldsense").mkdir(parents=True)
            discovered = discover_raw_sources(root)
            self.assertEqual(
                [("daily_omni", root / "raw_datasets" / "daily_omni"), ("worldsense", root / "raw_datasets" / "worldsense")],
                discovered,
            )

    def test_index_raw_sources_writes_jsonl_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw_datasets" / "daily_omni"
            source.mkdir(parents=True)
            (source / "a.mp4").write_bytes(b"x")
            (source / "nested").mkdir()
            (source / "nested" / "b.webm").write_bytes(b"y")

            summary = index_raw_sources(root=root, sources=[("daily_omni", source)])
            raw_index = root / "metadata" / "raw_assets.jsonl"
            report = root / "reports" / "raw_assets_summary.md"

            self.assertEqual(2, summary["asset_count"])
            self.assertTrue(raw_index.exists())
            self.assertTrue(report.exists())
            records = [json.loads(line) for line in raw_index.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(2, len(records))
            self.assertEqual({"daily_omni"}, {record["dataset"] for record in records})

    def test_build_ffmpeg_extract_command_preserves_audio_when_available(self) -> None:
        command = build_ffmpeg_extract_command(
            source_path="/tmp/input.mp4",
            output_path="/tmp/output.mp4",
            start_seconds=1.25,
            end_seconds=5.0,
            overwrite=True,
        )
        self.assertEqual("ffmpeg", command[0])
        self.assertIn("0:a?", command)
        self.assertEqual("/tmp/output.mp4", command[-1])

    def test_extract_clips_resolves_source_asset_ids_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw_datasets" / "daily_omni"
            source.mkdir(parents=True)
            video = source / "source.mp4"
            video.write_bytes(b"x")
            index_raw_sources(root=root, sources=[("daily_omni", source)])

            raw_index = root / "metadata" / "raw_assets.jsonl"
            asset = json.loads(raw_index.read_text(encoding="utf-8").splitlines()[0])
            plan_path = root / "metadata" / "clip_plan.jsonl"
            plan_path.write_text(
                json.dumps(
                    {
                        "clip_id": "daily_omni_ref_0001",
                        "source_asset_id": asset["asset_id"],
                        "start_seconds": 0,
                        "end_seconds": 4.5,
                        "role": "reference",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch("app.composed_data.subprocess.run") as run_mock:
                summary = extract_clips(root=root, plan_path=plan_path, overwrite=True)

            manifest_path = root / "metadata" / "clips.jsonl"
            self.assertEqual(1, summary["clip_count"])
            self.assertTrue(manifest_path.exists())
            run_mock.assert_called_once()
            record = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("daily_omni_ref_0001", record["clip_id"])
            self.assertTrue(record["output_path"].endswith("clips/daily_omni_ref_0001.mp4"))

    def test_plan_detective_event_clips_writes_clip_plan_and_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            video = root / "raw_datasets" / "worldsense" / "source.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"x")
            source_clips_path = root / "metadata" / "source_clips_all.jsonl"
            self._write_jsonl(
                source_clips_path,
                [
                    {
                        "clip_id": "worldsense_source",
                        "source_path": str(video),
                        "output_path": "raw_datasets/worldsense/source.mp4",
                        "dataset": "worldsense",
                        "source_row_ids": ["row_a"],
                        "text_fields": {"video_caption": "a jazz band performs on stage"},
                    }
                ],
            )

            with mock.patch(
                "app.composed_data.probe_media",
                return_value={
                    "duration_seconds": 22.0,
                    "has_audio": True,
                    "has_video": True,
                    "width": 640,
                    "height": 360,
                    "fps": 25.0,
                },
            ):
                summary = plan_detective_event_clips(
                    root=root,
                    source_clips_path=source_clips_path,
                    max_source_videos=1,
                    segment_seconds=8.0,
                )

            plan_path = Path(summary["clip_plan_output_path"])
            groups_path = Path(summary["clip_groups_output_path"])
            plan_records = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            group_records = [json.loads(line) for line in groups_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(3, summary["planned_clip_count"])
            self.assertEqual(1, summary["group_count"])
            self.assertEqual(3, len(plan_records))
            self.assertEqual("same_source_video", group_records[0]["group_reason"])
            self.assertEqual([record["clip_id"] for record in plan_records], group_records[0]["candidate_clip_ids"])

    def test_annotate_clips_writes_complete_annotations_with_mock_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            clip_one = root / "clips" / "clip_a.mp4"
            clip_two = root / "clips" / "clip_b.mp4"
            clip_one.write_bytes(b"a")
            clip_two.write_bytes(b"b")
            manifest_path = root / "metadata" / "clips.jsonl"
            self._write_jsonl(
                manifest_path,
                [
                    {
                        "clip_id": "clip_a",
                        "source_asset_id": "asset_a",
                        "output_path": "clips/clip_a.mp4",
                        "dataset": "daily_omni",
                        "source_row_ids": ["row_a"],
                        "text_fields": {"question": "What is the cat doing?"},
                        "start_seconds": 0.0,
                        "end_seconds": 4.0,
                        "duration_seconds": 4.0,
                    },
                    {
                        "clip_id": "clip_b",
                        "source_asset_id": "asset_b",
                        "output_path": "clips/clip_b.mp4",
                        "start_seconds": 2.0,
                        "end_seconds": 6.0,
                        "duration_seconds": 4.0,
                    },
                ],
            )

            annotation_outputs = [
                (
                    {
                        "summary": "one orange cat resting on a sofa",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 1},
                        "actions": ["resting"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                    {"provider": "mock", "clip_id": "clip_a"},
                ),
                (
                    {
                        "summary": "two orange cats resting on a sofa",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 2},
                        "actions": ["resting"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["soft meow"],
                        "modalities": ["visual", "audio"],
                    },
                    {"provider": "mock", "clip_id": "clip_b"},
                ),
            ]

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.annotate_clip.side_effect = annotation_outputs
                summary = annotate_clips(
                    root=root,
                    clips_manifest_path=manifest_path,
                    output_path=root / "captions" / "clip_annotations.jsonl",
                    base_url="http://127.0.0.1:8092/v1",
                    api_key="EMPTY",
                    model="captioner-model",
                )

            output_path = root / "captions" / "clip_annotations.jsonl"
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(2, summary["clip_count"])
            self.assertEqual(2, summary["annotated_count"])
            self.assertEqual(0, summary["fallback_count"])
            self.assertEqual(2, len(records))
            self.assertEqual({"clip_a", "clip_b"}, {record["clip_id"] for record in records})
            for record in records:
                self.assertIn("summary", record)
                self.assertIn("subjects", record)
                self.assertIn("object_counts", record)
                self.assertIn("actions", record)
                self.assertIn("scene", record)
                self.assertIn("attributes", record)
                self.assertIn("on_screen_text", record)
                self.assertIn("speech", record)
                self.assertIn("audio_events", record)
                self.assertIn("modalities", record)
                self.assertIn("source_asset_id", record)
                self.assertIn("fallback_used", record)
                self.assertIn("raw_model_output", record)
                self.assertFalse(record["fallback_used"])
            records_by_id = {record["clip_id"]: record for record in records}
            self.assertEqual("daily_omni", records_by_id["clip_a"]["dataset"])
            self.assertEqual(["row_a"], records_by_id["clip_a"]["source_row_ids"])
            self.assertEqual({"question": "What is the cat doing?"}, records_by_id["clip_a"]["text_fields"])

    def test_annotate_clips_marks_fallback_without_batch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "clip_a.mp4").write_bytes(b"a")
            (root / "clips" / "clip_b.mp4").write_bytes(b"b")
            manifest_path = root / "metadata" / "clips.jsonl"
            self._write_jsonl(
                manifest_path,
                [
                    {
                        "clip_id": "clip_a",
                        "source_asset_id": "asset_a",
                        "output_path": "clips/clip_a.mp4",
                        "start_seconds": 0.0,
                        "end_seconds": 4.0,
                        "duration_seconds": 4.0,
                    },
                    {
                        "clip_id": "clip_b",
                        "source_asset_id": "asset_b",
                        "output_path": "clips/clip_b.mp4",
                        "start_seconds": 1.0,
                        "end_seconds": 5.0,
                        "duration_seconds": 4.0,
                    },
                ],
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.annotate_clip.side_effect = [
                    ValueError("clip annotation missing fields: ['summary']"),
                    (
                        {
                            "summary": "a person claps in a quiet room",
                            "subjects": ["person"],
                            "object_counts": {"person": 1},
                            "actions": ["clapping"],
                            "scene": "studio",
                            "attributes": ["indoor"],
                            "on_screen_text": [],
                            "speech": [],
                            "audio_events": ["clap"],
                            "modalities": ["visual", "audio"],
                        },
                        {"provider": "mock", "clip_id": "clip_b"},
                    ),
                ]
                summary = annotate_clips(
                    root=root,
                    clips_manifest_path=manifest_path,
                    output_path=root / "captions" / "clip_annotations.jsonl",
                    base_url="http://127.0.0.1:8092/v1",
                    api_key="EMPTY",
                    model="captioner-model",
                )

            records = [
                json.loads(line)
                for line in (root / "captions" / "clip_annotations.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records_by_id = {record["clip_id"]: record for record in records}
            self.assertEqual(2, summary["clip_count"])
            self.assertEqual(1, summary["fallback_count"])
            self.assertTrue(records_by_id["clip_a"]["fallback_used"])
            self.assertEqual("annotation_fallback", records_by_id["clip_a"]["fallback_reason"])
            self.assertEqual(["visual"], records_by_id["clip_a"]["modalities"])
            self.assertFalse(records_by_id["clip_b"]["fallback_used"])

    def test_detective_annotate_clips_writes_trajectory_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "clip_a.mp4").write_bytes(b"a")
            manifest_path = root / "metadata" / "clips.jsonl"
            self._write_jsonl(
                manifest_path,
                [
                    {
                        "clip_id": "clip_a",
                        "source_asset_id": "asset_a",
                        "output_path": "clips/clip_a.mp4",
                        "dataset": "daily_omni",
                    }
                ],
            )

            detective_output = (
                {
                    "summary": "a person plays guitar on a small stage",
                    "subjects": ["person", "guitar"],
                    "object_counts": {"person": 1, "guitar": 1},
                    "actions": ["playing guitar"],
                    "scene": "small stage",
                    "attributes": ["indoor"],
                    "on_screen_text": [],
                    "speech": [],
                    "audio_events": ["guitar music"],
                    "modalities": ["visual", "audio"],
                    "storyline": ["person sits with guitar", "person plays music"],
                    "visible_text": [],
                    "speakers_and_transcript": [],
                    "detective_notes": ["audio confirms guitar performance"],
                    "detective_trajectory": [{"stage": "observer"}, {"stage": "detective_final"}],
                },
                {"provider": "mock", "mode": "detective"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.annotate_clip_detective.return_value = detective_output
                summary = detective_annotate_clips(
                    root=root,
                    clips_manifest_path=manifest_path,
                    output_path=root / "captions" / "clip_annotations_detective.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            records = [
                json.loads(line)
                for line in (root / "captions" / "clip_annotations_detective.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual("detective", summary["annotation_mode"])
            self.assertEqual(1, summary["annotated_count"])
            self.assertEqual(["person sits with guitar", "person plays music"], records[0]["storyline"])
            self.assertEqual([{"stage": "observer"}, {"stage": "detective_final"}], records[0]["detective_trajectory"])
            self.assertEqual("daily_omni", records[0]["dataset"])

    def test_detective_annotate_clips_falls_back_to_single_pass_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "clip_a.mp4").write_bytes(b"a")
            manifest_path = root / "metadata" / "clips.jsonl"
            self._write_jsonl(
                manifest_path,
                [{"clip_id": "clip_a", "source_asset_id": "asset_a", "output_path": "clips/clip_a.mp4"}],
            )

            single_pass_output = (
                {
                    "summary": "a person claps in a studio",
                    "subjects": ["person"],
                    "object_counts": {"person": 1},
                    "actions": ["clapping"],
                    "scene": "studio",
                    "attributes": ["indoor"],
                    "on_screen_text": [],
                    "speech": [],
                    "audio_events": ["clap"],
                    "modalities": ["visual", "audio"],
                },
                {"provider": "mock", "mode": "single_pass"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.annotate_clip_detective.side_effect = ValueError("bad detective json")
                client_cls.return_value.annotate_clip.return_value = single_pass_output
                summary = detective_annotate_clips(
                    root=root,
                    clips_manifest_path=manifest_path,
                    output_path=root / "captions" / "clip_annotations_detective.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            records = [
                json.loads(line)
                for line in (root / "captions" / "clip_annotations_detective.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(0, summary["fallback_count"])
            self.assertEqual(1, summary["detective_to_single_pass_count"])
            self.assertFalse(records[0]["fallback_used"])
            self.assertTrue(records[0]["detective_fallback_used"])
            self.assertEqual("detective_to_single_pass", records[0]["detective_fallback_reason"])
            self.assertEqual("a person claps in a studio", records[0]["summary"])
            self.assertIn("single_pass_fallback", [item.get("stage") for item in records[0]["detective_trajectory"]])

    def test_detective_annotate_cli_does_not_read_group_pair_options(self) -> None:
        argv = [
            "composed_data.py",
            "detective-annotate-clips",
            "--root",
            "/tmp/root",
            "--clips-manifest-path",
            "/tmp/clips.jsonl",
            "--base-url",
            "http://127.0.0.1:8093/v1",
            "--api-key",
            "EMPTY",
            "--model",
            "qwen3-omni",
        ]
        with mock.patch("sys.argv", argv), mock.patch("builtins.print"), mock.patch(
            "app.composed_data.detective_annotate_clips",
            return_value={"ok": True},
        ) as detective_mock:
            composed_data_main()

        self.assertNotIn("max_accepted_pairs", detective_mock.call_args.kwargs)
        self.assertEqual(180.0, detective_mock.call_args.kwargs["timeout_seconds"])

    def test_propose_pairs_outputs_schema_compliant_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("clip_ref.mp4", "clip_target.mp4", "clip_neg1.mp4", "clip_neg2.mp4"):
                (root / "clips" / name).write_bytes(b"x")

            raw_index_path = root / "metadata" / "raw_assets.jsonl"
            self._write_jsonl(
                raw_index_path,
                [
                    {
                        "asset_id": "asset_ref",
                        "dataset": "daily_omni",
                        "path": str(root / "raw_datasets" / "daily_omni" / "ref.mp4"),
                    },
                    {
                        "asset_id": "asset_target",
                        "dataset": "daily_omni",
                        "path": str(root / "raw_datasets" / "daily_omni" / "target.mp4"),
                    },
                    {
                        "asset_id": "asset_neg1",
                        "dataset": "worldsense",
                        "path": str(root / "raw_datasets" / "worldsense" / "neg1.mp4"),
                    },
                    {
                        "asset_id": "asset_neg2",
                        "dataset": "worldsense",
                        "path": str(root / "raw_datasets" / "worldsense" / "neg2.mp4"),
                    },
                ],
            )

            annotations_path = root / "captions" / "clip_annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "clip_ref",
                        "output_path": "clips/clip_ref.mp4",
                        "summary": "one orange cat resting on a sofa in a living room",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 1},
                        "actions": ["resting"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["quiet room"],
                        "modalities": ["visual", "audio"],
                        "source_asset_id": "asset_ref",
                        "fallback_used": False,
                        "raw_model_output": {"provider": "mock"},
                    },
                    {
                        "clip_id": "clip_target",
                        "output_path": "clips/clip_target.mp4",
                        "summary": "two orange cats resting on a sofa in a living room",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 2},
                        "actions": ["resting"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["quiet room"],
                        "modalities": ["visual", "audio"],
                        "source_asset_id": "asset_target",
                        "fallback_used": False,
                        "raw_model_output": {"provider": "mock"},
                    },
                    {
                        "clip_id": "clip_neg1",
                        "output_path": "clips/clip_neg1.mp4",
                        "summary": "one orange cat stretching on a sofa in a living room",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 1},
                        "actions": ["stretching"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["quiet room"],
                        "modalities": ["visual", "audio"],
                        "source_asset_id": "asset_neg1",
                        "fallback_used": False,
                        "raw_model_output": {"provider": "mock"},
                    },
                    {
                        "clip_id": "clip_neg2",
                        "output_path": "clips/clip_neg2.mp4",
                        "summary": "one orange cat resting on a sofa in a living room with a bell sound",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 1},
                        "actions": ["resting"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["bell ringing"],
                        "modalities": ["visual", "audio"],
                        "source_asset_id": "asset_neg2",
                        "fallback_used": False,
                        "raw_model_output": {"provider": "mock"},
                    },
                ],
            )

            def fake_propose_pair(*, reference_annotation, target_annotation, hard_negative_candidates, heuristic_pair=None):
                difference = {
                    "type": "object_count",
                    "from": f"{reference_annotation['object_counts'].get('cat', 0)} cat",
                    "to": f"{target_annotation['object_counts'].get('cat', 0)} cat",
                    "description": "the cat count changes while the scene stays the same",
                }
                return (
                    {
                        "edit_text": "change one orange cat into two orange cats",
                        "modalities": ["visual", "audio"],
                        "reference_caption": reference_annotation["summary"],
                        "target_caption": target_annotation["summary"],
                        "difference": difference,
                        "proposal_reason": f"same context with {len(hard_negative_candidates)} nearby negatives",
                    },
                    {"provider": "mock"},
                )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.propose_pair.side_effect = fake_propose_pair
                summary = propose_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    raw_index_path=raw_index_path,
                    output_path=root / "pairs" / "pilot_candidates.jsonl",
                    base_url="http://127.0.0.1:8092/v1",
                    api_key="EMPTY",
                    model="instruct-model",
                )

            output_path = root / "pairs" / "pilot_candidates.jsonl"
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(summary["proposal_count"], 1)
            self.assertEqual(summary["proposal_count"], len(records))
            self.assertEqual(0, summary["fallback_count"])
            for record in records:
                self.assertIn(record["difference"]["type"], ALLOWED_DIFFERENCE_TYPES)
                self.assertNotEqual(record["reference_video"], record["target_video"])
                self.assertNotIn(record["target_video"], record["hard_negatives"])
                self.assertGreaterEqual(len(record["hard_negatives"]), 2)
                self.assertIn("same_context_score", record["quality"])
                self.assertIn("edit_match_score", record["quality"])
                self.assertIn("target_uniqueness_score", record["quality"])
                self.assertIn("platform", record["source"])
                self.assertIn("url", record["source"])
                self.assertIn("license_note", record["source"])

    def test_propose_group_pairs_accepts_only_judged_group_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("clip_ref.mp4", "clip_target.mp4", "clip_neg1.mp4", "clip_neg2.mp4"):
                (root / "clips" / name).write_bytes(b"x")
            annotations_path = root / "captions" / "detective_annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "clip_ref",
                        "output_path": "clips/clip_ref.mp4",
                        "summary": "one orange cat resting on a sofa in a living room",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 1},
                        "actions": ["resting"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["quiet room"],
                        "modalities": ["visual", "audio"],
                        "storyline": ["one cat rests on the sofa"],
                        "visible_text": [],
                        "speakers_and_transcript": [],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_target",
                        "output_path": "clips/clip_target.mp4",
                        "summary": "two orange cats resting on a sofa in a living room",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 2},
                        "actions": ["resting"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["quiet room"],
                        "modalities": ["visual", "audio"],
                        "storyline": ["two cats rest on the sofa"],
                        "visible_text": [],
                        "speakers_and_transcript": [],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_neg1",
                        "output_path": "clips/clip_neg1.mp4",
                        "summary": "one orange cat stretching on a sofa in a living room",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 1},
                        "actions": ["stretching"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["quiet room"],
                        "modalities": ["visual", "audio"],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_neg2",
                        "output_path": "clips/clip_neg2.mp4",
                        "summary": "one orange cat resting on a sofa with a bell sound",
                        "subjects": ["cat"],
                        "object_counts": {"cat": 1},
                        "actions": ["resting"],
                        "scene": "living room",
                        "attributes": ["orange"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["bell ringing"],
                        "modalities": ["visual", "audio"],
                        "fallback_used": False,
                    },
                ],
            )
            groups_path = root / "metadata" / "clip_groups.jsonl"
            self._write_jsonl(
                groups_path,
                [
                    {
                        "group_id": "group_cat_room",
                        "dataset": "daily_omni",
                        "group_reason": "same_source_video",
                        "source_clip_ids": ["source_cat"],
                        "candidate_clip_ids": ["clip_ref", "clip_target", "clip_neg1", "clip_neg2"],
                        "group_tags": ["cat", "sofa"],
                    }
                ],
            )

            def fake_propose_pair(*, reference_annotation, target_annotation, hard_negative_candidates, heuristic_pair=None):
                return (
                    {
                        "edit_text": "change one orange cat into two orange cats",
                        "modalities": ["visual", "audio"],
                        "reference_caption": reference_annotation["summary"],
                        "target_caption": target_annotation["summary"],
                        "difference": {
                            "type": "object_count",
                            "from": "one cat",
                            "to": "two cats",
                            "description": "the cat count changes",
                        },
                        "proposal_reason": "same room and same action with a count change",
                    },
                    {"provider": "mock"},
                )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.propose_pair.side_effect = fake_propose_pair
                client_cls.return_value.judge_pair.return_value = (
                    {
                        "reference_satisfies_edit": False,
                        "target_satisfies_edit": True,
                        "single_main_difference": True,
                        "same_context_score": 0.82,
                        "edit_match_score": 0.91,
                        "target_uniqueness_score": 0.86,
                        "audio_required": False,
                        "hard_negative_quality": "good",
                        "accept": True,
                        "reject_reason": "",
                    },
                    {"provider": "mock-judge"},
                )
                client_cls.return_value.verify_pair_difference.return_value = (
                    {
                        "caption_delta": {
                            "caption_equivalent": False,
                            "has_concrete_difference": True,
                            "difference_matches_edit": True,
                            "concrete_differences": ["one cat becomes two cats"],
                            "reason": "cat count changes",
                        },
                        "edit_projection": {
                            "projected_target_caption": "two orange cats resting on a sofa in a living room",
                            "target_matches_projection": True,
                            "score": 0.92,
                            "missing_requirements": [],
                            "reason": "projected caption matches target",
                        },
                        "edit_necessity": {
                            "edit_needed": True,
                            "reference_satisfies_edit": False,
                            "target_satisfies_edit": True,
                            "score": 0.9,
                            "reason": "reference only has one cat",
                        },
                    },
                    {"provider": "mock-verification"},
                )
                summary = propose_group_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    clip_groups_path=groups_path,
                    output_path=root / "pairs" / "judged_pair_proposals.jsonl",
                    accepted_output_path=root / "pairs" / "accepted_pairs.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            accepted_records = [
                json.loads(line)
                for line in (root / "pairs" / "accepted_pairs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(summary["accepted_count"], 1)
            self.assertEqual(summary["accepted_count"], len(accepted_records))
            self.assertTrue(all(record["judge"]["target_satisfies_edit"] for record in accepted_records))
            self.assertTrue(all(record["verification"]["caption_delta"]["difference_matches_edit"] for record in accepted_records))
            self.assertTrue(all(record["group_id"] == "group_cat_room" for record in accepted_records))

    def test_propose_group_pairs_rejects_caption_equivalent_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("clip_ref.mp4", "clip_target.mp4", "clip_neg1.mp4", "clip_neg2.mp4"):
                (root / "clips" / name).write_bytes(b"x")
            annotations_path = root / "captions" / "detective_annotations.jsonl"
            annotations = [
                {
                    "clip_id": "clip_ref",
                    "output_path": "clips/clip_ref.mp4",
                    "summary": "a man writes on a paper at a desk",
                    "subjects": ["man", "paper"],
                    "object_counts": {"man": 1, "paper": 1},
                    "actions": ["writing"],
                    "scene": "desk",
                    "attributes": ["indoor"],
                    "on_screen_text": ["formula"],
                    "speech": [],
                    "audio_events": ["speech"],
                    "modalities": ["visual", "audio"],
                    "fallback_used": False,
                },
                {
                    "clip_id": "clip_target",
                    "output_path": "clips/clip_target.mp4",
                    "summary": "a man writes on a paper at the same desk",
                    "subjects": ["man", "paper", "pen"],
                    "object_counts": {"man": 1, "paper": 1, "pen": 1},
                    "actions": ["writing"],
                    "scene": "desk",
                    "attributes": ["indoor"],
                    "on_screen_text": ["formula"],
                    "speech": [],
                    "audio_events": ["speech"],
                    "modalities": ["visual", "audio"],
                    "fallback_used": False,
                },
                {
                    "clip_id": "clip_neg1",
                    "output_path": "clips/clip_neg1.mp4",
                    "summary": "a man reads a paper at a desk",
                    "subjects": ["man", "paper"],
                    "object_counts": {"man": 1, "paper": 1},
                    "actions": ["reading"],
                    "scene": "desk",
                    "attributes": ["indoor"],
                    "on_screen_text": ["formula"],
                    "speech": [],
                    "audio_events": ["speech"],
                    "modalities": ["visual", "audio"],
                    "fallback_used": False,
                },
                {
                    "clip_id": "clip_neg2",
                    "output_path": "clips/clip_neg2.mp4",
                    "summary": "a woman writes on a paper at a desk",
                    "subjects": ["woman", "paper"],
                    "object_counts": {"woman": 1, "paper": 1},
                    "actions": ["writing"],
                    "scene": "desk",
                    "attributes": ["indoor"],
                    "on_screen_text": ["formula"],
                    "speech": [],
                    "audio_events": ["speech"],
                    "modalities": ["visual", "audio"],
                    "fallback_used": False,
                },
            ]
            self._write_jsonl(annotations_path, annotations)
            groups_path = root / "metadata" / "clip_groups.jsonl"
            self._write_jsonl(
                groups_path,
                [
                    {
                        "group_id": "group_desk",
                        "dataset": "daily_omni",
                        "group_reason": "same_source_video",
                        "source_clip_ids": ["source_desk"],
                        "candidate_clip_ids": ["clip_ref", "clip_target", "clip_neg1", "clip_neg2"],
                        "group_tags": ["desk"],
                    }
                ],
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.propose_pair.return_value = (
                    {
                        "edit_text": "add a pen to the writing scene",
                        "modalities": ["visual"],
                        "reference_caption": "a man writes on a paper at a desk",
                        "target_caption": "a man writes on a paper at the same desk",
                        "difference": {
                            "type": "object_presence",
                            "from": "no pen",
                            "to": "pen present",
                            "description": "a pen is visible",
                        },
                        "proposal_reason": "same desk scene",
                    },
                    {"provider": "mock"},
                )
                client_cls.return_value.judge_pair.return_value = (
                    {
                        "reference_satisfies_edit": False,
                        "target_satisfies_edit": True,
                        "single_main_difference": True,
                        "same_context_score": 0.9,
                        "edit_match_score": 0.9,
                        "target_uniqueness_score": 0.8,
                        "audio_required": False,
                        "hard_negative_quality": "good",
                        "accept": True,
                        "reject_reason": "",
                    },
                    {"provider": "mock-judge"},
                )
                client_cls.return_value.verify_pair_difference.return_value = (
                    {
                        "caption_delta": {
                            "caption_equivalent": True,
                            "has_concrete_difference": False,
                            "difference_matches_edit": False,
                            "concrete_differences": [],
                            "reason": "the captions describe the same writing content",
                        },
                        "edit_projection": {
                            "projected_target_caption": "a man writes on a paper with a pen visible",
                            "target_matches_projection": False,
                            "score": 0.4,
                            "missing_requirements": ["pen visibility"],
                            "reason": "target caption does not add the edit",
                        },
                        "edit_necessity": {
                            "edit_needed": False,
                            "reference_satisfies_edit": False,
                            "target_satisfies_edit": False,
                            "score": 0.3,
                            "reason": "the edit is not supported by the target",
                        },
                    },
                    {"provider": "mock-verification"},
                )
                summary = propose_group_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    clip_groups_path=groups_path,
                    output_path=root / "pairs" / "judged_pair_proposals.jsonl",
                    accepted_output_path=root / "pairs" / "accepted_pairs.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            records = [
                json.loads(line)
                for line in (root / "pairs" / "judged_pair_proposals.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(summary["proposal_count"], 1)
            self.assertEqual(0, summary["accepted_count"])
            self.assertGreaterEqual(summary["verification_counts"]["caption_equivalent_reject_count"], 1)
            self.assertTrue(any("equivalent" in record["judge"]["reject_reason"] for record in records))

    def test_propose_group_pairs_cli_passes_max_accepted_pairs(self) -> None:
        argv = [
            "composed_data.py",
            "propose-group-pairs",
            "--root",
            "/tmp/root",
            "--clip-annotations-path",
            "/tmp/annotations.jsonl",
            "--clip-groups-path",
            "/tmp/groups.jsonl",
            "--base-url",
            "http://127.0.0.1:8093/v1",
            "--api-key",
            "EMPTY",
            "--model",
            "qwen3-omni",
            "--max-accepted-pairs",
            "7",
        ]
        with mock.patch("sys.argv", argv), mock.patch("builtins.print"), mock.patch(
            "app.composed_data.propose_group_pairs",
            return_value={"ok": True},
        ) as propose_mock:
            composed_data_main()

        self.assertEqual(7, propose_mock.call_args.kwargs["max_accepted_pairs"])

    def test_build_seeded_pair_slice_filters_split_and_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            videos = []
            for name in ("ref_a.mp4", "tgt_a.mp4", "ref_b.mp4", "tgt_b.mp4", "ref_val.mp4", "tgt_val.mp4"):
                path = root / "clips" / name
                path.write_bytes(b"x")
                videos.append(path)

            pair_seeds_path = root / "metadata" / "webvid_covr_pair_seeds.jsonl"
            self._write_jsonl(
                pair_seeds_path,
                [
                    {
                        "pair_seed_id": "seed_train_a",
                        "dataset": "webvid_covr",
                        "split": "train",
                        "reference_video_path": str(root / "clips" / "ref_a.mp4"),
                        "target_video_path": str(root / "clips" / "tgt_a.mp4"),
                    },
                    {
                        "pair_seed_id": "seed_train_b",
                        "dataset": "webvid_covr",
                        "split": "train",
                        "reference_video_path": str(root / "clips" / "ref_b.mp4"),
                        "target_video_path": str(root / "clips" / "tgt_b.mp4"),
                    },
                    {
                        "pair_seed_id": "seed_val",
                        "dataset": "webvid_covr",
                        "split": "validation",
                        "reference_video_path": str(root / "clips" / "ref_val.mp4"),
                        "target_video_path": str(root / "clips" / "tgt_val.mp4"),
                    },
                ],
            )
            source_clips_path = root / "metadata" / "source_clips_all.jsonl"
            self._write_jsonl(
                source_clips_path,
                [
                    {
                        "clip_id": "ref_a",
                        "source_path": str(root / "clips" / "ref_a.mp4"),
                        "output_path": str(root / "clips" / "ref_a.mp4"),
                        "dataset": "webvid_covr",
                    },
                    {
                        "clip_id": "tgt_a",
                        "source_path": str(root / "clips" / "tgt_a.mp4"),
                        "output_path": str(root / "clips" / "tgt_a.mp4"),
                        "dataset": "webvid_covr",
                    },
                    {
                        "clip_id": "ref_b",
                        "source_path": str(root / "clips" / "ref_b.mp4"),
                        "output_path": str(root / "clips" / "ref_b.mp4"),
                        "dataset": "webvid_covr",
                    },
                    {
                        "clip_id": "tgt_b",
                        "source_path": str(root / "clips" / "tgt_b.mp4"),
                        "output_path": str(root / "clips" / "tgt_b.mp4"),
                        "dataset": "webvid_covr",
                    },
                    {
                        "clip_id": "ref_val",
                        "source_path": str(root / "clips" / "ref_val.mp4"),
                        "output_path": str(root / "clips" / "ref_val.mp4"),
                        "dataset": "webvid_covr",
                    },
                    {
                        "clip_id": "tgt_val",
                        "source_path": str(root / "clips" / "tgt_val.mp4"),
                        "output_path": str(root / "clips" / "tgt_val.mp4"),
                        "dataset": "webvid_covr",
                    },
                ],
            )

            summary = build_seeded_pair_slice(
                pair_seeds_path=pair_seeds_path,
                source_clips_path=source_clips_path,
                output_seeds_path=root / "runs" / "slice_seeds.jsonl",
                output_clips_path=root / "runs" / "slice_clips.jsonl",
                split="train",
                max_seed_rows=1,
                seed_offset=1,
            )

            seeds = [json.loads(line) for line in (root / "runs" / "slice_seeds.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            clips = [json.loads(line) for line in (root / "runs" / "slice_clips.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["selected_seed_count"])
            self.assertEqual(["seed_train_b"], [seed["pair_seed_id"] for seed in seeds])
            self.assertEqual({"ref_b", "tgt_b"}, {clip["clip_id"] for clip in clips})
            self.assertTrue(all(clip["split"] == "train" for clip in clips))

    def test_propose_seeded_pairs_accepts_visual_webvid_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4", "val_neg.mp4"):
                (root / "clips" / name).write_bytes(b"x")

            annotations_path = root / "captions" / "webvid_annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "clip_ref",
                        "output_path": "clips/ref.mp4",
                        "source_path": str(root / "clips" / "ref.mp4"),
                        "dataset": "webvid_covr",
                        "split": "train",
                        "summary": "a red car parked outside a house",
                        "subjects": ["car"],
                        "object_counts": {"car": 1},
                        "actions": ["parked"],
                        "scene": "outside house",
                        "attributes": ["red"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                        "storyline": ["a red car is parked outside"],
                        "visible_text": [],
                        "speakers_and_transcript": [],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_target",
                        "output_path": "clips/target.mp4",
                        "source_path": str(root / "clips" / "target.mp4"),
                        "dataset": "webvid_covr",
                        "split": "train",
                        "summary": "a blue car parked outside a house",
                        "subjects": ["car"],
                        "object_counts": {"car": 1},
                        "actions": ["parked"],
                        "scene": "outside house",
                        "attributes": ["blue"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                        "storyline": ["a blue car is parked outside"],
                        "visible_text": [],
                        "speakers_and_transcript": [],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_neg1",
                        "output_path": "clips/neg1.mp4",
                        "source_path": str(root / "clips" / "neg1.mp4"),
                        "dataset": "webvid_covr",
                        "split": "train",
                        "summary": "a green car parked outside a house",
                        "subjects": ["car"],
                        "object_counts": {"car": 1},
                        "actions": ["parked"],
                        "scene": "outside house",
                        "attributes": ["green"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                        "storyline": ["a green car is parked outside"],
                        "visible_text": [],
                        "speakers_and_transcript": [],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_neg2",
                        "output_path": "clips/neg2.mp4",
                        "source_path": str(root / "clips" / "neg2.mp4"),
                        "dataset": "webvid_covr",
                        "split": "train",
                        "summary": "a red bus parked outside a house",
                        "subjects": ["bus"],
                        "object_counts": {"bus": 1},
                        "actions": ["parked"],
                        "scene": "outside house",
                        "attributes": ["red"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                        "storyline": ["a bus is parked outside"],
                        "visible_text": [],
                        "speakers_and_transcript": [],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_val_neg",
                        "output_path": "clips/val_neg.mp4",
                        "source_path": str(root / "clips" / "val_neg.mp4"),
                        "dataset": "webvid_covr",
                        "split": "validation",
                        "summary": "a yellow car parked outside a house",
                        "subjects": ["car"],
                        "object_counts": {"car": 1},
                        "actions": ["parked"],
                        "scene": "outside house",
                        "attributes": ["yellow"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                        "storyline": ["a yellow car is parked outside"],
                        "visible_text": [],
                        "speakers_and_transcript": [],
                        "fallback_used": False,
                    },
                ],
            )
            seeds_path = root / "metadata" / "webvid_covr_seed_slice.jsonl"
            self._write_jsonl(
                seeds_path,
                [
                    {
                        "pair_seed_id": "seed_pair_1",
                        "dataset": "webvid_covr",
                        "split": "train",
                        "reference_video_path": str(root / "clips" / "ref.mp4"),
                        "target_video_path": str(root / "clips" / "target.mp4"),
                        "txt1": "a red car parked outside",
                        "txt2": "a blue car parked outside",
                        "edit": "change the car from red to blue",
                        "sim_txt": 0.85,
                        "sim_vid": 0.77,
                        "scores": {"clip": 0.6},
                        "person_prob": 0.1,
                    }
                ],
            )

            def fake_propose_pair(*, reference_annotation, target_annotation, hard_negative_candidates, heuristic_pair=None):
                self.assertEqual("change the car from red to blue", heuristic_pair["seed_metadata"]["edit"])
                self.assertTrue(all(candidate.get("split") == "train" for candidate in hard_negative_candidates))
                return (
                    {
                        "edit_text": "change the car from red to blue",
                        "modalities": ["visual"],
                        "reference_caption": reference_annotation["summary"],
                        "target_caption": target_annotation["summary"],
                        "difference": {
                            "type": "attribute",
                            "from": "red car",
                            "to": "blue car",
                            "description": "the car color changes from red to blue",
                        },
                        "proposal_reason": "same setting with a color change",
                    },
                    {"provider": "mock"},
                )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.propose_pair.side_effect = fake_propose_pair
                client_cls.return_value.judge_pair.return_value = (
                    {
                        "reference_satisfies_edit": False,
                        "target_satisfies_edit": True,
                        "single_main_difference": True,
                        "same_context_score": 0.8,
                        "edit_match_score": 0.9,
                        "target_uniqueness_score": 0.88,
                        "audio_required": False,
                        "hard_negative_quality": "good",
                        "accept": True,
                        "reject_reason": "",
                    },
                    {"provider": "mock-judge"},
                )
                client_cls.return_value.verify_pair_difference.return_value = (
                    {
                        "caption_delta": {
                            "caption_equivalent": False,
                            "has_concrete_difference": True,
                            "difference_matches_edit": True,
                            "concrete_differences": ["red car becomes blue car"],
                            "reason": "car color changes",
                        },
                        "edit_projection": {
                            "projected_target_caption": "a blue car parked outside a house",
                            "target_matches_projection": True,
                            "score": 0.93,
                            "missing_requirements": [],
                            "reason": "projection matches target",
                        },
                        "edit_necessity": {
                            "edit_needed": True,
                            "reference_satisfies_edit": False,
                            "target_satisfies_edit": True,
                            "score": 0.9,
                            "reason": "reference is red and target is blue",
                        },
                    },
                    {"provider": "mock-verification"},
                )
                summary = propose_seeded_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_seeds_path=seeds_path,
                    output_path=root / "pairs" / "judged_pair_proposals.jsonl",
                    accepted_output_path=root / "pairs" / "accepted_pairs.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            accepted_records = [
                json.loads(line)
                for line in (root / "pairs" / "accepted_pairs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual("webvid_covr_seed_pair", accepted_records[0]["group_reason"])
            self.assertEqual("webvid_covr_seed_pair", accepted_records[0]["source_context"]["relation"])
            self.assertEqual("seed_pair_1", accepted_records[0]["seed_metadata"]["pair_seed_id"])
            self.assertEqual("webvid_covr", accepted_records[0]["source"]["platform"])
            self.assertTrue(all("val_neg" not in path for path in accepted_records[0]["hard_negatives"]))

    def test_propose_seeded_pairs_rejects_webvid_speech_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4"):
                (root / "clips" / name).write_bytes(b"x")

            annotations_path = root / "captions" / "webvid_annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "clip_ref",
                        "output_path": "clips/ref.mp4",
                        "source_path": str(root / "clips" / "ref.mp4"),
                        "dataset": "webvid_covr",
                        "split": "train",
                        "summary": "a person speaks about red cars",
                        "subjects": ["person"],
                        "object_counts": {"person": 1},
                        "actions": ["speaking"],
                        "scene": "studio",
                        "attributes": [],
                        "on_screen_text": ["red cars"],
                        "speech": ["red cars are popular"],
                        "audio_events": [],
                        "modalities": ["visual", "audio"],
                        "storyline": ["speaker discusses red cars"],
                        "visible_text": ["red cars"],
                        "speakers_and_transcript": ["speaker: red cars are popular"],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_target",
                        "output_path": "clips/target.mp4",
                        "source_path": str(root / "clips" / "target.mp4"),
                        "dataset": "webvid_covr",
                        "split": "train",
                        "summary": "a person speaks about blue cars",
                        "subjects": ["person"],
                        "object_counts": {"person": 1},
                        "actions": ["speaking"],
                        "scene": "studio",
                        "attributes": [],
                        "on_screen_text": ["blue cars"],
                        "speech": ["blue cars are popular"],
                        "audio_events": [],
                        "modalities": ["visual", "audio"],
                        "storyline": ["speaker discusses blue cars"],
                        "visible_text": ["blue cars"],
                        "speakers_and_transcript": ["speaker: blue cars are popular"],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_neg1",
                        "output_path": "clips/neg1.mp4",
                        "source_path": str(root / "clips" / "neg1.mp4"),
                        "dataset": "webvid_covr",
                        "split": "train",
                        "summary": "a person gestures in a studio",
                        "subjects": ["person"],
                        "object_counts": {"person": 1},
                        "actions": ["gesturing"],
                        "scene": "studio",
                        "attributes": [],
                        "on_screen_text": [],
                        "speech": ["cars are expensive"],
                        "audio_events": [],
                        "modalities": ["visual", "audio"],
                        "storyline": ["speaker gestures"],
                        "visible_text": [],
                        "speakers_and_transcript": ["speaker: cars are expensive"],
                        "fallback_used": False,
                    },
                    {
                        "clip_id": "clip_neg2",
                        "output_path": "clips/neg2.mp4",
                        "source_path": str(root / "clips" / "neg2.mp4"),
                        "dataset": "webvid_covr",
                        "split": "train",
                        "summary": "a person sits in a studio",
                        "subjects": ["person"],
                        "object_counts": {"person": 1},
                        "actions": ["sitting"],
                        "scene": "studio",
                        "attributes": [],
                        "on_screen_text": [],
                        "speech": ["cars are available"],
                        "audio_events": [],
                        "modalities": ["visual", "audio"],
                        "storyline": ["speaker sits"],
                        "visible_text": [],
                        "speakers_and_transcript": ["speaker: cars are available"],
                        "fallback_used": False,
                    },
                ],
            )
            seeds_path = root / "metadata" / "webvid_covr_seed_slice.jsonl"
            self._write_jsonl(
                seeds_path,
                [
                    {
                        "pair_seed_id": "seed_pair_speech",
                        "dataset": "webvid_covr",
                        "split": "train",
                        "reference_video_path": str(root / "clips" / "ref.mp4"),
                        "target_video_path": str(root / "clips" / "target.mp4"),
                        "txt1": "a person speaks about red cars",
                        "txt2": "a person speaks about blue cars",
                        "edit": "change the speech from red cars to blue cars",
                        "sim_txt": 0.9,
                        "sim_vid": 0.8,
                    }
                ],
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.propose_pair.return_value = (
                    {
                        "edit_text": "change the speech from red cars to blue cars",
                        "modalities": ["audio"],
                        "reference_caption": "a person speaks about red cars",
                        "target_caption": "a person speaks about blue cars",
                        "difference": {
                            "type": "speech",
                            "from": "red cars are popular",
                            "to": "blue cars are popular",
                            "description": "speech changes topic",
                        },
                        "proposal_reason": "speech content changes",
                    },
                    {"provider": "mock"},
                )
                client_cls.return_value.judge_pair.return_value = (
                    {
                        "reference_satisfies_edit": False,
                        "target_satisfies_edit": True,
                        "single_main_difference": True,
                        "same_context_score": 0.9,
                        "edit_match_score": 0.9,
                        "target_uniqueness_score": 0.85,
                        "audio_required": True,
                        "hard_negative_quality": "good",
                        "accept": True,
                        "reject_reason": "",
                    },
                    {"provider": "mock-judge"},
                )
                client_cls.return_value.verify_pair_difference.return_value = (
                    {
                        "caption_delta": {
                            "caption_equivalent": False,
                            "has_concrete_difference": True,
                            "difference_matches_edit": True,
                            "concrete_differences": ["red cars becomes blue cars"],
                            "reason": "speech content changes",
                        },
                        "edit_projection": {
                            "projected_target_caption": "a person speaks about blue cars",
                            "target_matches_projection": True,
                            "score": 0.95,
                            "missing_requirements": [],
                            "reason": "projection matches target",
                        },
                        "edit_necessity": {
                            "edit_needed": True,
                            "reference_satisfies_edit": False,
                            "target_satisfies_edit": True,
                            "score": 0.92,
                            "reason": "reference says red cars and target says blue cars",
                        },
                    },
                    {"provider": "mock-verification"},
                )
                summary = propose_seeded_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_seeds_path=seeds_path,
                    output_path=root / "pairs" / "judged_pair_proposals.jsonl",
                    accepted_output_path=root / "pairs" / "accepted_pairs.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            records = [
                json.loads(line)
                for line in (root / "pairs" / "judged_pair_proposals.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(0, summary["accepted_count"])
            self.assertEqual("speech", records[0]["difference"]["type"])
            self.assertIn("webvid_covr only allows visual difference types", records[0]["judge"]["reject_reason"])

    def test_build_seeded_pair_slice_cli_passes_arguments(self) -> None:
        argv = [
            "composed_data.py",
            "build-seeded-pair-slice",
            "--pair-seeds-path",
            "/tmp/seeds.jsonl",
            "--source-clips-path",
            "/tmp/source_clips.jsonl",
            "--output-seeds-path",
            "/tmp/output_seeds.jsonl",
            "--output-clips-path",
            "/tmp/output_clips.jsonl",
            "--split",
            "train",
            "--max-seed-rows",
            "11",
            "--seed-offset",
            "3",
        ]
        with mock.patch("sys.argv", argv), mock.patch("builtins.print"), mock.patch(
            "app.composed_data.build_seeded_pair_slice",
            return_value={"ok": True},
        ) as build_mock:
            composed_data_main()

        self.assertEqual("train", build_mock.call_args.kwargs["split"])
        self.assertEqual(11, build_mock.call_args.kwargs["max_seed_rows"])
        self.assertEqual(3, build_mock.call_args.kwargs["seed_offset"])

    def test_propose_seeded_pairs_cli_passes_max_accepted_pairs(self) -> None:
        argv = [
            "composed_data.py",
            "propose-seeded-pairs",
            "--root",
            "/tmp/root",
            "--clip-annotations-path",
            "/tmp/annotations.jsonl",
            "--pair-seeds-path",
            "/tmp/pair_seeds.jsonl",
            "--base-url",
            "http://127.0.0.1:8093/v1",
            "--api-key",
            "EMPTY",
            "--model",
            "qwen3-omni",
            "--max-accepted-pairs",
            "9",
        ]
        with mock.patch("sys.argv", argv), mock.patch("builtins.print"), mock.patch(
            "app.composed_data.propose_seeded_pairs",
            return_value={"ok": True},
        ) as propose_mock:
            composed_data_main()

        self.assertEqual(9, propose_mock.call_args.kwargs["max_accepted_pairs"])

    def test_compose_reject_reason_includes_threshold_failures(self) -> None:
        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.81,
            "edit_match_score": 0.62,
            "target_uniqueness_score": 0.58,
            "audio_required": True,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }

        reason = _compose_reject_reason(judge)

        self.assertIn("edit_match_score 0.620 is below 0.75", reason)
        self.assertIn("target_uniqueness_score 0.580 is below 0.70", reason)
        self.assertIn("the model judge did not accept the pair", reason)

    def test_verification_can_override_low_judge_edit_score(self) -> None:
        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.82,
            "edit_match_score": 0.395,
            "target_uniqueness_score": 0.86,
            "audio_required": False,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.92,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }

        quality = _effective_pair_quality(
            judge,
            verification,
            {"same_context_score": 0.82, "target_uniqueness_score": 0.86},
        )

        self.assertEqual(0.9, quality["edit_match_score"])
        self.assertFalse(_judge_accepts(judge))
        self.assertTrue(_judge_accepts(judge, verification, quality))

    def test_verification_override_still_rejects_equivalent_captions(self) -> None:
        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.95,
            "target_uniqueness_score": 0.9,
            "audio_required": False,
            "hard_negative_quality": "good",
            "accept": True,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": True,
                "has_concrete_difference": False,
                "difference_matches_edit": False,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.9,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }

        quality = _effective_pair_quality(judge, verification, {})

        self.assertFalse(_judge_accepts(judge, verification, quality))

    def test_finalize_pair_verification_writes_passed_diagnostics(self) -> None:
        verification = _finalize_pair_verification(
            {
                "caption_delta": {
                    "caption_equivalent": False,
                    "has_concrete_difference": True,
                    "difference_matches_edit": True,
                },
                "edit_projection": {
                    "target_matches_projection": True,
                    "score": 0.9,
                },
                "edit_necessity": {
                    "edit_needed": True,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "score": 0.85,
                },
            }
        )

        self.assertTrue(verification["passed"])
        self.assertEqual([], verification["failures"])

    def test_difference_strength_gate_blocks_weak_changes(self) -> None:
        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.9,
            "target_uniqueness_score": 0.9,
            "audio_required": False,
            "hard_negative_quality": "good",
            "accept": True,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.9,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }

        quality = _effective_pair_quality(
            judge,
            verification,
            {
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.2,
            },
        )

        self.assertFalse(_judge_accepts(judge, verification, quality))
        self.assertIn("difference_strength_score", _compose_reject_reason(judge, verification, quality))

    def test_action_evidence_gate_blocks_weak_action_override(self) -> None:
        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.6,
            "target_uniqueness_score": 0.9,
            "audio_required": False,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.9,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }

        weak_quality = _effective_pair_quality(
            judge,
            verification,
            {
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "difference_type": "action",
                "action_evidence_score": 0.62,
            },
        )
        strong_quality = dict(weak_quality)
        strong_quality["action_evidence_score"] = 0.74

        self.assertFalse(_judge_accepts(judge, verification, weak_quality))
        self.assertIn("action_evidence_score", _compose_reject_reason(judge, verification, weak_quality))
        self.assertTrue(_judge_accepts(judge, verification, strong_quality))

    def test_visual_near_duplicate_blocks_visual_difference_override(self) -> None:
        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.6,
            "target_uniqueness_score": 0.9,
            "audio_required": False,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.92,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }
        quality = _effective_pair_quality(
            judge,
            verification,
            {
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "visual_near_duplicate_score": 0.998,
                "difference_type": "attribute",
            },
        )

        self.assertFalse(_judge_accepts(judge, verification, quality))
        self.assertIn("visual_near_duplicate_score", _compose_reject_reason(judge, verification, quality))

    def test_visual_near_duplicate_allows_speech_difference(self) -> None:
        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.6,
            "target_uniqueness_score": 0.9,
            "audio_required": True,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.92,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }
        quality = _effective_pair_quality(
            judge,
            verification,
            {
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "visual_near_duplicate_score": 0.998,
                "difference_type": "speech",
                "has_audio_modality": 1.0,
                "speech_evidence_score": 0.88,
                "speech_specificity_score": 0.82,
                "speech_transcript_backed": 1.0,
            },
        )

        self.assertTrue(_judge_accepts(judge, verification, quality))

    def test_speech_gate_blocks_generic_speaking_to_camera(self) -> None:
        reference = {"speech": ["a man speaks to camera"]}
        target = {"speech": ["a man talks to camera in a forest"]}

        self.assertLess(_speech_evidence_score(reference, target), 0.75)
        self.assertLess(_speech_specificity_score(reference, target), 0.70)

        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.6,
            "target_uniqueness_score": 0.9,
            "audio_required": True,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.9,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }
        quality = _effective_pair_quality(
            judge,
            verification,
            {
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "difference_type": "speech",
                "has_audio_modality": 1.0,
                "speech_evidence_score": _speech_evidence_score(reference, target),
                "speech_specificity_score": _speech_specificity_score(reference, target),
            },
        )

        self.assertFalse(_judge_accepts(judge, verification, quality))
        self.assertIn("speech_evidence_score", _compose_reject_reason(judge, verification, quality))

    def test_speech_gate_requires_transcript_backing_for_specific_content(self) -> None:
        reference = {"speech": ["today I am introducing the old forest and its wildlife habitat"]}
        target = {"speech": ["the old forest is scheduled to be cut down next week"]}

        self.assertLess(_speech_evidence_score(reference, target), 0.75)

        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.6,
            "target_uniqueness_score": 0.9,
            "audio_required": True,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.9,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }
        quality = _effective_pair_quality(
            judge,
            verification,
            {
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "difference_type": "speech",
                "has_audio_modality": 1.0,
                "speech_evidence_score": _speech_evidence_score(reference, target),
                "speech_specificity_score": _speech_specificity_score(reference, target),
                "speech_transcript_backed": 0.0,
            },
        )

        self.assertFalse(_judge_accepts(judge, verification, quality))
        self.assertIn("transcript evidence", _compose_reject_reason(judge, verification, quality))

    def test_speech_gate_accepts_specific_transcript_delta(self) -> None:
        reference = {
            "speakers_and_transcript": [
                {"speaker": "narrator", "content": "Today I am introducing the old forest and its wildlife habitat."}
            ]
        }
        target = {
            "speakers_and_transcript": [
                {"speaker": "narrator", "content": "The old forest is scheduled to be cut down next week."}
            ]
        }

        self.assertGreaterEqual(_speech_evidence_score(reference, target), 0.75)
        self.assertGreaterEqual(_speech_specificity_score(reference, target), 0.70)

        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.6,
            "target_uniqueness_score": 0.9,
            "audio_required": True,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.9,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }
        quality = _effective_pair_quality(
            judge,
            verification,
            {
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": _speech_evidence_score(reference, target),
                "difference_type": "speech",
                "has_audio_modality": 1.0,
                "speech_evidence_score": _speech_evidence_score(reference, target),
                "speech_specificity_score": _speech_specificity_score(reference, target),
                "speech_transcript_backed": 1.0,
            },
        )

        self.assertTrue(_judge_accepts(judge, verification, quality))

    def test_audio_event_gate_rejects_speech_only_events(self) -> None:
        reference = {"audio_events": ["speech", "narration"]}
        target = {"audio_events": ["voiceover", "talking"]}

        self.assertEqual(0.0, _non_speech_audio_event_score(reference, target))

        judge = {
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "single_main_difference": True,
            "same_context_score": 0.9,
            "edit_match_score": 0.6,
            "target_uniqueness_score": 0.9,
            "audio_required": True,
            "hard_negative_quality": "good",
            "accept": False,
            "reject_reason": "",
        }
        verification = {
            "caption_delta": {
                "caption_equivalent": False,
                "has_concrete_difference": True,
                "difference_matches_edit": True,
            },
            "edit_projection": {
                "target_matches_projection": True,
                "score": 0.9,
            },
            "edit_necessity": {
                "edit_needed": True,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "score": 0.9,
            },
        }
        quality = _effective_pair_quality(
            judge,
            verification,
            {
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "difference_type": "audio_event",
                "non_speech_audio_event_score": _non_speech_audio_event_score(reference, target),
            },
        )

        self.assertFalse(_judge_accepts(judge, verification, quality))
        self.assertIn("non_speech_audio_event_score", _compose_reject_reason(judge, verification, quality))

    def test_audio_event_gate_accepts_non_speech_audio_delta(self) -> None:
        reference = {"audio_events": ["quiet forest ambience"]}
        target = {"audio_events": ["chainsaw noise", "machine buzzing"]}

        self.assertGreaterEqual(_non_speech_audio_event_score(reference, target), 0.70)

    def test_audio_event_gate_accepts_timeline_audio_delta_without_top_level_audio_events(self) -> None:
        reference = {"events": [{"audio": "quiet forest ambience"}]}
        target = {"events": [{"audio": "chainsaw noise and machine buzzing"}]}

        self.assertGreaterEqual(_non_speech_audio_event_score(reference, target), 0.70)

    def test_audio_event_gate_accepts_summary_audio_delta_without_structured_audio_fields(self) -> None:
        reference = {
            "summary": "a robot stands on a platform with a low electronic hum in the background",
            "detective_notes": ["background electronic hum remains constant"],
        }
        target = {
            "summary": "the same robot stands on the platform while dramatic orchestral music swells",
            "detective_notes": ["dramatic orchestral music replaces the previous hum"],
        }

        self.assertGreaterEqual(_non_speech_audio_event_score(reference, target), 0.70)

    def test_audio_event_gate_rejects_speech_only_absence_phrase(self) -> None:
        reference = {
            "summary": "a man speaks in a forest",
            "audio_events": [],
        }
        target = {
            "summary": "The audio track contains only speech; no background music or ambient noise is present.",
            "audio_events": [],
        }

        self.assertEqual(0.0, _non_speech_audio_event_score(reference, target))

    def test_fallback_audio_event_edit_text_uses_add_remove_for_absence(self) -> None:
        self.assertEqual(
            "add low-frequency electronic hum to the audio",
            _build_fallback_edit_text(
                {
                    "type": "audio_event",
                    "from": "no distinctive audio event",
                    "to": "low-frequency electronic hum",
                }
            ),
        )
        self.assertEqual(
            "remove low-frequency electronic hum from the audio",
            _build_fallback_edit_text(
                {
                    "type": "audio_event",
                    "from": "low-frequency electronic hum",
                    "to": "no distinctive audio event",
                }
            ),
        )

    def test_intraclip_audio_event_conflict_detects_from_to_in_target_caption(self) -> None:
        reference = {
            "summary": "a person writes on paper",
            "audio_events": ["low-frequency electronic hum"],
        }
        target = {
            "summary": "a person writes on paper",
            "audio_events": ["scratching sound"],
        }
        difference = {
            "type": "audio_event",
            "from": "low-frequency electronic hum",
            "to": "scratching sound",
        }

        self.assertTrue(
            _has_intraclip_difference_conflict(
                difference=difference,
                reference_caption="A person writes on paper with a low-frequency electronic hum.",
                target_caption="The audio changes from a low-frequency electronic hum to a scratching sound while the person writes.",
                reference_annotation=reference,
                target_annotation=target,
            )
        )

    def test_detect_primary_difference_uses_summary_audio_delta(self) -> None:
        reference = {
            "summary": "a robot stands on a platform with a low electronic hum in the background",
            "object_counts": {"robot": 1},
            "scene": "dark studio platform",
            "detective_notes": ["background electronic hum remains constant"],
        }
        target = {
            "summary": "the same robot stands on the platform while dramatic orchestral music swells",
            "object_counts": {"robot": 1},
            "scene": "dark studio platform",
            "detective_notes": ["dramatic orchestral music replaces the previous hum"],
        }

        difference = _detect_primary_difference(reference, target, priority_order=("audio_event", "object_presence", "scene"))

        self.assertIsNotNone(difference)
        self.assertEqual("audio_event", difference["type"])

    def test_pair_verification_counts_tracks_speech_audio_rejects(self) -> None:
        records = [
            {
                "accepted": False,
                "difference": {"type": "speech"},
                "quality": {"speech_evidence_score": 0.2, "speech_specificity_score": 0.3},
                "verification": {
                    "caption_delta": {
                        "caption_equivalent": False,
                        "has_concrete_difference": True,
                        "difference_matches_edit": True,
                    },
                    "edit_projection": {"target_matches_projection": True, "score": 0.9},
                    "edit_necessity": {
                        "edit_needed": True,
                        "reference_satisfies_edit": False,
                        "target_satisfies_edit": True,
                        "score": 0.9,
                    },
                },
            },
            {
                "accepted": False,
                "difference": {"type": "audio_event"},
                "quality": {"non_speech_audio_event_score": 0.0},
                "verification": {
                    "caption_delta": {
                        "caption_equivalent": False,
                        "has_concrete_difference": True,
                        "difference_matches_edit": True,
                    },
                    "edit_projection": {"target_matches_projection": True, "score": 0.9},
                    "edit_necessity": {
                        "edit_needed": True,
                        "reference_satisfies_edit": False,
                        "target_satisfies_edit": True,
                        "score": 0.9,
                    },
                },
            },
        ]

        counts = _pair_verification_counts(records)

        self.assertEqual(1, counts["speech_rejected_as_too_generic_count"])
        self.assertEqual(1, counts["speech_rejected_for_missing_transcript_count"])
        self.assertEqual(1, counts["audio_event_rejected_as_speech_only_count"])

    def test_accepted_sample_carries_speech_quality_fields(self) -> None:
        sample = _accepted_sample_from_record(
            {
                "proposal_id": "proposal__speech",
                "reference_video": "clips/ref.mp4",
                "target_video": "clips/target.mp4",
                "edit_text": "change speech from A to B",
                "modalities": ["audio"],
                "reference_caption": "ref",
                "target_caption": "target",
                "difference": {"type": "speech", "from": "A", "to": "B"},
                "hard_negatives": ["clips/neg.mp4"],
                "quality": {"difference_type": "speech"},
                "source": {"platform": "unknown", "url": "file:///tmp/target.mp4", "license_note": "internal"},
                "source_context": {"relation": "same_source_video"},
                "evidence": {},
                "judge": {},
                "verification": {},
                "speech_quality": {
                    "transcript_backed": True,
                    "evidence_score": 0.88,
                    "specificity_score": 0.9,
                    "audio_required": True,
                },
                "audio_event_quality": {},
                "transcript_backed": True,
                "group_id": "group_a",
                "group_reason": "same_source_video",
            },
            1,
        )

        self.assertEqual(True, sample["transcript_backed"])
        self.assertEqual(True, sample["speech_quality"]["transcript_backed"])
        self.assertEqual(0.88, sample["speech_quality"]["evidence_score"])

    def test_accepted_sample_carries_audio_event_quality_fields(self) -> None:
        sample = _accepted_sample_from_record(
            {
                "proposal_id": "proposal__audio",
                "reference_video": "clips/ref.mp4",
                "target_video": "clips/target.mp4",
                "edit_text": "change low electronic hum into orchestral music",
                "modalities": ["audio"],
                "reference_caption": "ref",
                "target_caption": "target",
                "difference": {"type": "audio_event", "from": "electronic hum", "to": "orchestral music"},
                "hard_negatives": ["clips/neg.mp4"],
                "quality": {"difference_type": "audio_event"},
                "source": {"platform": "unknown", "url": "file:///tmp/target.mp4", "license_note": "internal"},
                "source_context": {"relation": "same_source_video"},
                "evidence": {},
                "judge": {},
                "verification": {},
                "speech_quality": {},
                "audio_event_quality": {
                    "non_speech_score": 0.84,
                    "audio_required": True,
                },
                "transcript_backed": None,
                "group_id": "group_a",
                "group_reason": "same_source_video",
            },
            1,
        )

        self.assertEqual(0.84, sample["audio_event_quality"]["non_speech_score"])
        self.assertEqual(True, sample["audio_event_quality"]["audio_required"])

    def test_accepted_sample_carries_clip_ids(self) -> None:
        sample = _accepted_sample_from_record(
            {
                "proposal_id": "proposal__ids",
                "reference_clip_id": "clip_ref",
                "target_clip_id": "clip_target",
                "reference_video": "clips/ref.mp4",
                "target_video": "clips/target.mp4",
                "edit_text": "add a toy bin",
                "modalities": ["visual"],
                "reference_caption": "ref",
                "target_caption": "target",
                "difference": {"type": "object_presence", "from": "no toy bin", "to": "toy bin present"},
                "hard_negatives": ["clips/neg.mp4"],
                "quality": {"difference_type": "object_presence"},
                "source": {"platform": "unknown", "url": "file:///tmp/target.mp4", "license_note": "internal"},
                "source_context": {"relation": "same_source_video"},
                "evidence": {},
                "judge": {},
                "verification": {},
                "speech_quality": {},
                "audio_event_quality": {},
                "transcript_backed": None,
                "group_id": "group_a",
                "group_reason": "same_source_video",
            },
            1,
        )

        self.assertEqual("clip_ref", sample["reference_clip_id"])
        self.assertEqual("clip_target", sample["target_clip_id"])

    def test_pair_record_acceptance_issues_rejects_missing_target_and_intraclip_audio_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "ref.mp4").write_bytes(b"x")
            (root / "clips" / "neg.mp4").write_bytes(b"x")

            issues = _pair_record_acceptance_issues(
                root=root,
                record={
                    "reference_video": "clips/ref.mp4",
                    "target_video": "clips/missing.mp4",
                    "hard_negatives": ["clips/neg.mp4"],
                    "difference": {
                        "type": "audio_event",
                        "from": "low-frequency electronic hum",
                        "to": "scratching sound",
                    },
                    "reference_caption": "A person writes with a low-frequency electronic hum.",
                    "target_caption": "The audio changes from a low-frequency electronic hum to a scratching sound while writing.",
                },
                reference_annotation={"audio_events": ["low-frequency electronic hum"]},
                target_annotation={"audio_events": ["scratching sound"]},
            )

        self.assertTrue(any("target_video does not exist" in issue for issue in issues))
        self.assertTrue(any("single clip" in issue for issue in issues))

    def test_pair_record_acceptance_issues_rejects_speech_only_audio_event_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "ref.mp4").write_bytes(b"x")
            (root / "clips" / "target.mp4").write_bytes(b"x")

            issues = _pair_record_acceptance_issues(
                root=root,
                record={
                    "reference_video": "clips/ref.mp4",
                    "target_video": "clips/target.mp4",
                    "hard_negatives": [],
                    "difference": {
                        "type": "audio_event",
                        "from": "no distinctive audio event",
                        "to": "The audio track contains only speech; no background music or ambient noise is present.",
                    },
                    "reference_caption": "A man talks in a forest.",
                    "target_caption": "A man talks in a forest.",
                },
                reference_annotation={"audio_events": []},
                target_annotation={"audio_events": []},
            )

        self.assertTrue(any("speech-only" in issue for issue in issues))

    def test_select_final_accepted_records_dedupes_repeated_group_audio_events(self) -> None:
        base_record = {
            "accepted": True,
            "group_id": "group_audio",
            "source_context": {"relation": "same_source_video"},
            "modalities": ["audio"],
            "reference_caption": "ref",
            "target_caption": "target",
            "hard_negatives": ["clips/neg.mp4"],
            "source": {"platform": "unknown", "url": "file:///tmp/target.mp4", "license_note": "internal"},
            "evidence": {},
            "judge": {},
            "verification": {"passed": True},
            "speech_quality": {},
            "audio_event_quality": {},
            "transcript_backed": None,
            "group_reason": "same_source_video",
        }
        records = [
            {
                **base_record,
                "proposal_id": "proposal__audio_a",
                "reference_video": "clips/ref_a.mp4",
                "target_video": "clips/target_a.mp4",
                "edit_text": "add a low-frequency electronic hum",
                "difference": {"type": "audio_event", "from": "no distinctive audio event", "to": "low-frequency electronic hum"},
                "quality": {
                    "difference_type": "audio_event",
                    "difference_strength_score": 0.9,
                    "same_context_score": 0.9,
                    "target_uniqueness_score": 0.8,
                    "edit_match_score": 0.9,
                },
            },
            {
                **base_record,
                "proposal_id": "proposal__audio_b",
                "reference_video": "clips/ref_b.mp4",
                "target_video": "clips/target_b.mp4",
                "edit_text": "introduce a low-frequency electronic hum",
                "difference": {"type": "audio_event", "from": "no distinctive audio event", "to": "low-frequency electronic hum"},
                "quality": {
                    "difference_type": "audio_event",
                    "difference_strength_score": 0.85,
                    "same_context_score": 0.9,
                    "target_uniqueness_score": 0.79,
                    "edit_match_score": 0.9,
                },
            },
            {
                **base_record,
                "proposal_id": "proposal__action",
                "group_id": "group_action",
                "reference_video": "clips/action_ref.mp4",
                "target_video": "clips/action_target.mp4",
                "edit_text": "change the action from standing to waving",
                "difference": {"type": "action", "from": "standing", "to": "waving"},
                "quality": {
                    "difference_type": "action",
                    "difference_strength_score": 0.84,
                    "same_context_score": 0.88,
                    "target_uniqueness_score": 0.82,
                    "edit_match_score": 0.86,
                },
            },
            {
                **base_record,
                "proposal_id": "proposal__object",
                "group_id": "group_object",
                "modalities": ["visual"],
                "reference_video": "clips/object_ref.mp4",
                "target_video": "clips/object_target.mp4",
                "edit_text": "add a toy bin",
                "difference": {"type": "object_presence", "from": "no toy bin", "to": "1 toy bin"},
                "quality": {
                    "difference_type": "object_presence",
                    "difference_strength_score": 0.83,
                    "same_context_score": 0.87,
                    "target_uniqueness_score": 0.81,
                    "edit_match_score": 0.85,
                },
            },
        ]

        accepted = _select_final_accepted_records(records, max_accepted_pairs=4)

        self.assertEqual(3, len(accepted))
        self.assertEqual(1, sum(1 for record in accepted if record["difference"]["type"] == "audio_event"))
        self.assertIn("action", {record["difference"]["type"] for record in accepted})
        self.assertIn("object_presence", {record["difference"]["type"] for record in accepted})

    def test_difference_strength_scores_concrete_object_changes(self) -> None:
        reference = {
            "object_counts": {"cat": 1},
            "actions": ["sitting"],
            "events": [{"visual": "one cat sits on a sofa", "audio": "", "objects": ["cat"], "actions": ["sitting"]}],
        }
        target = {
            "object_counts": {"cat": 2},
            "actions": ["sitting"],
            "events": [{"visual": "two cats sit on the same sofa", "audio": "", "objects": ["cat"], "actions": ["sitting"]}],
        }
        difference = {
            "type": "object_count",
            "from": "1 cat",
            "to": "2 cat",
            "description": "the count of cat changes from 1 to 2",
        }

        score = _difference_strength_score(
            reference_annotation=reference,
            target_annotation=target,
            primary_difference=difference,
            changed_types=["object_count"],
        )

        self.assertGreaterEqual(score, 0.65)

    def test_detect_primary_difference_uses_timeline_action_evidence(self) -> None:
        reference = {
            "object_counts": {"dog": 1},
            "actions": ["moving"],
            "events": [{"visual": "the dog walks across the park", "actions": ["walking"]}],
            "storyline": ["the dog walks across the park"],
        }
        target = {
            "object_counts": {"dog": 1},
            "actions": ["moving"],
            "events": [{"visual": "the dog jumps over a low bar", "actions": ["jumping"]}],
            "storyline": ["the dog jumps over a low bar"],
        }

        difference = _detect_primary_difference(reference, target, priority_order=("action", "object_presence"))

        self.assertIsNotNone(difference)
        self.assertEqual("action", difference["type"])
        self.assertEqual("walking", difference["from"])
        self.assertEqual("jumping", difference["to"])

    def test_action_evidence_score_requires_timeline_support(self) -> None:
        reference = {"actions": ["running"]}
        target = {"actions": ["jumping"]}
        reference_with_timeline = {
            "actions": ["running"],
            "events": [{"visual": "the dog runs across the park", "actions": ["running"]}],
        }
        target_with_timeline = {
            "actions": ["jumping"],
            "storyline": ["the dog jumps over a low bar"],
        }

        self.assertLess(_action_evidence_score(reference, target), 0.65)
        self.assertGreaterEqual(_action_evidence_score(reference_with_timeline, target_with_timeline), 0.65)

    def test_evidence_from_annotations_carries_action_and_timeline_fields(self) -> None:
        reference = {
            "summary": "A game character runs along a cliff edge.",
            "actions": ["running"],
            "storyline": ["the character runs along a cliff edge"],
            "audio_events": ["ambient hum"],
        }
        target = {
            "summary": "A game character is launched from the cliff and glides forward.",
            "actions": ["launched", "gliding"],
            "events": [{"visual": "the character is launched from the cliff", "actions": ["launched"]}],
            "audio_events": ["ambient hum", "whoosh"],
        }

        evidence = _evidence_from_annotations(reference, target)

        self.assertEqual("A game character runs along a cliff edge.", evidence["reference_summary"])
        self.assertEqual("A game character is launched from the cliff and glides forward.", evidence["target_summary"])
        self.assertEqual(["running"], evidence["reference_actions"])
        self.assertEqual(["launched", "gliding"], evidence["target_actions"])
        self.assertEqual("running -> launched; gliding", evidence["action_change"])
        self.assertEqual(["the character runs along a cliff edge"], evidence["reference_timeline_evidence"])
        self.assertTrue(any("launched" in item for item in evidence["target_timeline_evidence"]))

    def test_target_uniqueness_allows_close_negatives_with_different_edit(self) -> None:
        reference = {
            "clip_id": "ref",
            "summary": "a presenter speaks at a studio desk",
            "subjects": ["presenter"],
            "object_counts": {"presenter": 1},
            "actions": ["speaking"],
            "scene": "studio desk",
            "attributes": ["indoor"],
            "on_screen_text": [],
        }
        target = {
            "clip_id": "target",
            "summary": "a presenter speaks at the same studio desk with a laptop visible",
            "subjects": ["presenter"],
            "object_counts": {"presenter": 1, "laptop": 1},
            "actions": ["speaking"],
            "scene": "studio desk",
            "attributes": ["indoor"],
            "on_screen_text": [],
        }
        close_negative = {
            "clip_id": "negative",
            "summary": "a presenter speaks at the same studio desk with a poster visible",
            "subjects": ["presenter"],
            "object_counts": {"presenter": 1, "poster": 1},
            "actions": ["speaking"],
            "scene": "studio desk",
            "attributes": ["indoor"],
            "on_screen_text": [],
        }
        duplicate_target = {
            "clip_id": "duplicate",
            "summary": "a presenter speaks at the same studio desk with another laptop visible",
            "subjects": ["presenter"],
            "object_counts": {"presenter": 1, "laptop": 1},
            "actions": ["speaking"],
            "scene": "studio desk",
            "attributes": ["indoor"],
            "on_screen_text": [],
        }
        difference = {
            "type": "object_presence",
            "from": "no laptop",
            "to": "1 laptop",
            "description": "laptop appears in the target clip",
        }

        with_different_edit = _target_uniqueness_score(
            reference_annotation=reference,
            target_annotation=target,
            annotations=[reference, target, close_negative],
            primary_difference=difference,
        )
        with_duplicate_edit = _target_uniqueness_score(
            reference_annotation=reference,
            target_annotation=target,
            annotations=[reference, target, duplicate_target],
            primary_difference=difference,
        )

        self.assertGreaterEqual(with_different_edit, 0.70)
        self.assertLess(with_duplicate_edit, 0.70)

    def test_detect_primary_difference_prefers_speech_in_high_context_order(self) -> None:
        reference = {
            "object_counts": {"person": 1},
            "actions": ["speaking"],
            "audio_events": ["speech"],
            "attributes": ["studio"],
            "scene": "studio desk shot",
            "speakers_and_transcript": [
                {"speaker": "host", "content": "Welcome to the show where we introduce today's camera setup."}
            ],
            "visible_text": ["episode 1"],
        }
        target = {
            "object_counts": {"person": 1},
            "actions": ["speaking"],
            "audio_events": ["speech"],
            "attributes": ["studio"],
            "scene": "studio desk shot",
            "speakers_and_transcript": [
                {"speaker": "host", "content": "Today we review the camera lens and compare the autofocus performance."}
            ],
            "visible_text": ["episode 1"],
        }

        difference = _detect_primary_difference(
            reference,
            target,
            priority_order=("object_count", "speech", "audio_event", "visible_text", "object_presence", "action"),
        )

        self.assertIsNotNone(difference)
        self.assertEqual("speech", difference["type"])

    def test_high_context_priority_prefers_object_change_over_visible_text(self) -> None:
        reference = {
            "object_counts": {"person": 1},
            "actions": ["speaking"],
            "audio_events": ["speech"],
            "attributes": ["studio"],
            "scene": "studio desk shot",
            "visible_text": ["speaker name"],
        }
        target = {
            "object_counts": {"person": 1, "toy bin": 1},
            "actions": ["speaking"],
            "audio_events": ["speech"],
            "attributes": ["studio"],
            "scene": "studio desk shot",
            "visible_text": [],
        }

        difference = _detect_primary_difference(
            reference,
            target,
            priority_order=_difference_priority_order(same_context_score=0.9),
        )

        self.assertIsNotNone(difference)
        self.assertEqual("object_presence", difference["type"])

    def test_high_context_priority_keeps_object_change_ahead_of_audio_event(self) -> None:
        reference = {
            "object_counts": {"person": 1},
            "actions": ["speaking"],
            "audio_events": ["low electronic hum"],
            "attributes": ["studio"],
            "scene": "studio desk shot",
            "visible_text": [],
        }
        target = {
            "object_counts": {"person": 1, "laptop": 1},
            "actions": ["speaking"],
            "audio_events": ["scratching sound"],
            "attributes": ["studio"],
            "scene": "studio desk shot",
            "visible_text": [],
        }

        difference = _detect_primary_difference(
            reference,
            target,
            priority_order=_difference_priority_order(same_context_score=0.9),
        )

        self.assertIsNotNone(difference)
        self.assertEqual("object_presence", difference["type"])

    def test_pair_candidates_keep_low_context_pairs_with_available_negatives(self) -> None:
        annotations = [
            {
                "clip_id": "ref",
                "output_path": "clips/ref.mp4",
                "summary": "a woman speaks at a podium",
                "subjects": ["woman"],
                "object_counts": {"woman": 1, "podium": 1},
                "actions": ["speaking"],
                "scene": "formal hall",
                "attributes": ["formal"],
                "on_screen_text": [],
                "speech": [],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
            {
                "clip_id": "target",
                "output_path": "clips/target.mp4",
                "summary": "a man speaks in a blue studio",
                "subjects": ["man"],
                "object_counts": {"man": 1},
                "actions": ["speaking"],
                "scene": "blue studio",
                "attributes": ["blue"],
                "on_screen_text": [],
                "speech": [],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
            {
                "clip_id": "neg1",
                "output_path": "clips/neg1.mp4",
                "summary": "a beaker is heated in a laboratory",
                "subjects": ["beaker"],
                "object_counts": {"beaker": 1},
                "actions": ["heating"],
                "scene": "laboratory",
                "attributes": ["transparent"],
                "on_screen_text": [],
                "speech": [],
                "audio_events": [],
                "modalities": ["visual"],
            },
            {
                "clip_id": "neg2",
                "output_path": "clips/neg2.mp4",
                "summary": "a musician plays a tuba on stage",
                "subjects": ["musician"],
                "object_counts": {"musician": 1, "tuba": 1},
                "actions": ["playing"],
                "scene": "stage",
                "attributes": ["musical"],
                "on_screen_text": [],
                "speech": [],
                "audio_events": ["music"],
                "modalities": ["visual", "audio"],
            },
        ]

        candidates = _build_pair_candidates(root=Path("/tmp/composed"), annotations=annotations)

        self.assertGreaterEqual(len(candidates), 1)
        self.assertGreaterEqual(len(candidates[0]["hard_negative_paths"]), 2)

    def test_pair_candidates_filter_low_context_cross_dataset_pairs(self) -> None:
        annotations = [
            {
                "clip_id": "daily_a",
                "dataset": "daily_omni",
                "output_path": "clips/daily_a.mp4",
                "summary": "a person holds a handmade bookmark at a table",
                "subjects": ["person", "bookmark"],
                "object_counts": {"person": 1, "bookmark": 1},
                "actions": ["holding"],
                "scene": "craft table",
                "attributes": ["decorative"],
                "on_screen_text": [],
                "speech": [],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
            {
                "clip_id": "world_a",
                "dataset": "worldsense",
                "output_path": "clips/world_a.mp4",
                "summary": "a jazz band performs on a wooden stage",
                "subjects": ["band"],
                "object_counts": {"band": 1, "tuba": 1},
                "actions": ["performing"],
                "scene": "stage",
                "attributes": ["musical"],
                "on_screen_text": [],
                "speech": [],
                "audio_events": ["music"],
                "modalities": ["visual", "audio"],
            },
            {
                "clip_id": "daily_b",
                "dataset": "daily_omni",
                "output_path": "clips/daily_b.mp4",
                "summary": "a person displays several handmade bookmarks at a table",
                "subjects": ["person", "bookmark"],
                "object_counts": {"person": 1, "bookmark": 4},
                "actions": ["displaying"],
                "scene": "craft table",
                "attributes": ["decorative"],
                "on_screen_text": [],
                "speech": [],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
            {
                "clip_id": "daily_c",
                "dataset": "daily_omni",
                "output_path": "clips/daily_c.mp4",
                "summary": "a person shows colorful keychains at a craft table",
                "subjects": ["person", "keychain"],
                "object_counts": {"person": 1, "keychain": 3},
                "actions": ["showing"],
                "scene": "craft table",
                "attributes": ["colorful"],
                "on_screen_text": [],
                "speech": [],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
        ]

        candidates = _build_pair_candidates(root=Path("/tmp/composed"), annotations=annotations)

        self.assertGreaterEqual(len(candidates), 1)
        self.assertTrue(all(candidate["source_context"]["relation"] != "cross_dataset" for candidate in candidates))

    def test_pair_candidates_retarget_primary_difference_for_type_diversity(self) -> None:
        annotations = [
            {
                "clip_id": "ref",
                "source_path": "/data/source.mp4",
                "source_clip": {"start_seconds": 0.0, "end_seconds": 8.0},
                "output_path": "clips/ref.mp4",
                "summary": "a presenter speaks at a studio desk",
                "subjects": ["presenter"],
                "object_counts": {"presenter": 1},
                "actions": ["speaking"],
                "scene": "studio desk",
                "attributes": ["indoor"],
                "on_screen_text": [],
                "speakers_and_transcript": [
                    {"speaker": "presenter", "content": "Welcome to the lesson about writing formulas on the board."}
                ],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
            {
                "clip_id": "target",
                "source_path": "/data/source.mp4",
                "source_clip": {"start_seconds": 8.0, "end_seconds": 16.0},
                "output_path": "clips/target.mp4",
                "summary": "a presenter speaks at the same studio desk with a laptop visible",
                "subjects": ["presenter"],
                "object_counts": {"presenter": 1, "laptop": 1},
                "actions": ["speaking"],
                "scene": "studio desk",
                "attributes": ["indoor"],
                "on_screen_text": [],
                "speakers_and_transcript": [
                    {"speaker": "presenter", "content": "Use the coupon code today to save money on this course."}
                ],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
            {
                "clip_id": "neg1",
                "source_path": "/data/source.mp4",
                "source_clip": {"start_seconds": 16.0, "end_seconds": 24.0},
                "output_path": "clips/neg1.mp4",
                "summary": "a presenter speaks at the studio desk",
                "subjects": ["presenter"],
                "object_counts": {"presenter": 1},
                "actions": ["speaking"],
                "scene": "studio desk",
                "attributes": ["indoor"],
                "on_screen_text": [],
                "speakers_and_transcript": [
                    {"speaker": "presenter", "content": "Welcome to the lesson about writing formulas on the board."}
                ],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
            {
                "clip_id": "neg2",
                "source_path": "/data/source.mp4",
                "source_clip": {"start_seconds": 24.0, "end_seconds": 32.0},
                "output_path": "clips/neg2.mp4",
                "summary": "a presenter speaks at the studio desk with a poster visible",
                "subjects": ["presenter"],
                "object_counts": {"presenter": 1, "poster": 1},
                "actions": ["speaking"],
                "scene": "studio desk",
                "attributes": ["indoor"],
                "on_screen_text": [],
                "speakers_and_transcript": [
                    {"speaker": "presenter", "content": "Thanks for watching this studio lesson until the end."}
                ],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
        ]

        candidates = _build_pair_candidates(root=Path("/tmp/composed"), annotations=annotations)
        proposal_ids = [candidate["proposal_id"] for candidate in candidates]
        primary_types = [candidate["primary_difference"]["type"] for candidate in candidates]

        self.assertEqual(len(proposal_ids), len(set(proposal_ids)))
        self.assertIn("object_presence", primary_types)
        self.assertIn("speech", primary_types)

    def test_pair_context_uses_temporal_proximity_for_same_source_video(self) -> None:
        left = {
            "source_path": "/data/video.mp4",
            "source_clip": {"start_seconds": 0.0, "end_seconds": 8.0},
        }
        adjacent = {
            "source_path": "/data/video.mp4",
            "source_clip": {"start_seconds": 8.0, "end_seconds": 16.0},
        }
        distant = {
            "source_path": "/data/video.mp4",
            "source_clip": {"start_seconds": 80.0, "end_seconds": 88.0},
        }

        adjacent_context = _source_context(left, adjacent)
        distant_context = _source_context(left, distant)

        self.assertEqual("same_source_video", adjacent_context["relation"])
        self.assertEqual("adjacent_or_overlapping", adjacent_context["temporal_relation"])
        self.assertEqual(0.9, _pair_context_score(semantic_context_score=0.05, source_context=adjacent_context))
        self.assertEqual("distant", distant_context["temporal_relation"])
        self.assertEqual(0.45, _pair_context_score(semantic_context_score=0.05, source_context=distant_context))

    def test_propose_pairs_marks_fallback_when_model_call_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("clip_ref.mp4", "clip_target.mp4", "clip_neg1.mp4", "clip_neg2.mp4"):
                (root / "clips" / name).write_bytes(b"x")

            annotations_path = root / "captions" / "clip_annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "clip_ref",
                        "output_path": "clips/clip_ref.mp4",
                        "summary": "one dog running in a park",
                        "subjects": ["dog"],
                        "object_counts": {"dog": 1},
                        "actions": ["running"],
                        "scene": "park",
                        "attributes": ["brown"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["barking"],
                        "modalities": ["visual", "audio"],
                        "source_asset_id": "asset_ref",
                        "fallback_used": False,
                        "raw_model_output": {"provider": "mock"},
                    },
                    {
                        "clip_id": "clip_target",
                        "output_path": "clips/clip_target.mp4",
                        "summary": "one dog jumping in a park",
                        "subjects": ["dog"],
                        "object_counts": {"dog": 1},
                        "actions": ["jumping"],
                        "scene": "park",
                        "attributes": ["brown"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["barking"],
                        "modalities": ["visual", "audio"],
                        "source_asset_id": "asset_target",
                        "fallback_used": False,
                        "raw_model_output": {"provider": "mock"},
                    },
                    {
                        "clip_id": "clip_neg1",
                        "output_path": "clips/clip_neg1.mp4",
                        "summary": "one dog sitting in a park",
                        "subjects": ["dog"],
                        "object_counts": {"dog": 1},
                        "actions": ["sitting"],
                        "scene": "park",
                        "attributes": ["brown"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["barking"],
                        "modalities": ["visual", "audio"],
                        "source_asset_id": "asset_neg1",
                        "fallback_used": False,
                        "raw_model_output": {"provider": "mock"},
                    },
                    {
                        "clip_id": "clip_neg2",
                        "output_path": "clips/clip_neg2.mp4",
                        "summary": "one dog running in a park with loud music",
                        "subjects": ["dog"],
                        "object_counts": {"dog": 1},
                        "actions": ["running"],
                        "scene": "park",
                        "attributes": ["brown"],
                        "on_screen_text": [],
                        "speech": [],
                        "audio_events": ["music"],
                        "modalities": ["visual", "audio"],
                        "source_asset_id": "asset_neg2",
                        "fallback_used": False,
                        "raw_model_output": {"provider": "mock"},
                    },
                ],
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client_cls.return_value.propose_pair.side_effect = RuntimeError("mock proposal failure")
                summary = propose_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    output_path=root / "pairs" / "pilot_candidates.jsonl",
                    base_url="http://127.0.0.1:8092/v1",
                    api_key="EMPTY",
                    model="instruct-model",
                )

            records = [
                json.loads(line)
                for line in (root / "pairs" / "pilot_candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(summary["proposal_count"], 1)
            self.assertEqual(summary["proposal_count"], summary["fallback_count"])
            self.assertTrue(all(record["fallback_used"] for record in records))
            self.assertTrue(all(record["difference"]["type"] in ALLOWED_DIFFERENCE_TYPES for record in records))
            self.assertTrue(all(record["reference_video"] not in record["hard_negatives"] for record in records))
            self.assertTrue(all(record["target_video"] not in record["hard_negatives"] for record in records))

    def test_validate_pilot_dataset_builds_gallery_from_targets_and_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4"):
                (root / "clips" / name).write_bytes(b"x")

            pilot_dir = root / "pilot_10"
            reports_dir = pilot_dir / "reports"
            reports_dir.mkdir(parents=True)
            pilot_path = pilot_dir / "pilot_10.jsonl"
            pilot_path.write_text(
                json.dumps(
                    {
                        "sample_id": "covr_pilot_0001",
                        "proposal_id": _build_proposal_id("clips/ref.mp4", "clips/target.mp4"),
                        "reference_video": "clips/ref.mp4",
                        "target_video": "clips/target.mp4",
                        "edit_text": "change one cat into two cats",
                        "modalities": ["visual", "audio"],
                        "reference_caption": "one cat on a sofa",
                        "target_caption": "two cats on a sofa",
                        "difference": {
                            "type": "object_count",
                            "from": "one cat",
                            "to": "two cats",
                        },
                        "hard_negatives": ["clips/neg1.mp4", "clips/neg2.mp4"],
                        "quality": {
                            "same_context_score": 0.9,
                            "edit_match_score": 0.8,
                            "target_uniqueness_score": 0.7,
                        },
                        "source_context": {
                            "relation": "same_dataset",
                            "score": 0.9,
                        },
                        "source": {
                            "platform": "bilibili",
                            "url": "https://example.com/video",
                            "license_note": "internal research pilot only",
                        },
                        "verification": {
                            "caption_delta": {
                                "caption_equivalent": False,
                                "has_concrete_difference": True,
                                "difference_matches_edit": True,
                                "concrete_differences": ["one cat becomes two cats"],
                                "reason": "cat count changes",
                            },
                            "edit_projection": {
                                "projected_target_caption": "two cats on a sofa",
                                "target_matches_projection": True,
                                "score": 0.9,
                                "missing_requirements": [],
                                "reason": "projection matches",
                            },
                            "edit_necessity": {
                                "edit_needed": True,
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": True,
                                "score": 0.88,
                                "reason": "reference has one cat",
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self._write_jsonl(
                pilot_dir / "judged_pair_proposals.jsonl",
                [
                    {
                        "proposal_id": _build_proposal_id("clips/ref.mp4", "clips/target.mp4"),
                        "accepted": True,
                        "verification": {
                            "caption_delta": {
                                "caption_equivalent": False,
                                "has_concrete_difference": True,
                                "difference_matches_edit": True,
                            },
                            "edit_projection": {
                                "target_matches_projection": True,
                                "score": 0.9,
                            },
                            "edit_necessity": {
                                "edit_needed": True,
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": True,
                                "score": 0.88,
                            },
                        },
                    },
                    {
                        "proposal_id": "proposal_rejected_equivalent",
                        "accepted": False,
                        "verification": {
                            "caption_delta": {
                                "caption_equivalent": True,
                                "has_concrete_difference": False,
                                "difference_matches_edit": False,
                            },
                            "edit_projection": {
                                "target_matches_projection": False,
                                "score": 0.2,
                            },
                            "edit_necessity": {
                                "edit_needed": False,
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": False,
                                "score": 0.1,
                            },
                        },
                    },
                ],
            )

            gallery_path = pilot_dir / "gallery.jsonl"
            report_path = reports_dir / "pilot_review.md"
            summary = validate_pilot_dataset(
                root=root,
                pilot_jsonl_path=pilot_path,
                gallery_output_path=gallery_path,
                report_output_path=report_path,
            )

            gallery_records = [json.loads(line) for line in gallery_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(3, len(gallery_records))
            self.assertEqual(1, summary["sample_count"])
            self.assertEqual({"same_dataset": 1}, summary["source_context_counts"])
            self.assertEqual(
                {"same_context_min": 0.9, "same_context_avg": 0.9, "same_context_max": 0.9},
                summary["quality_summary"],
            )
            self.assertTrue(report_path.exists())
            self.assertEqual(1, summary["verification_counts"]["caption_equivalent_reject_count"])
            self.assertEqual(1, summary["verification_counts"]["accepted_after_verification_count"])
            self.assertEqual(0, summary["speech_audio_quality_counts"]["speech_count"])
            self.assertEqual(0, summary["speech_audio_quality_counts"]["non_speech_audio_event_count"])
            self.assertFalse(summary["automated_acceptance"]["non_speech_audio_samples_at_least_1"])
            self.assertTrue(summary["automated_acceptance"]["speech_samples_all_have_evidence"])
            self.assertTrue(summary["automated_acceptance"]["speech_samples_all_transcript_backed"])
            self.assertIn("caption_equivalent_reject_count", report_path.read_text(encoding="utf-8"))
            self.assertIn("Speech / Audio Quality Counts", report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {"clips/target.mp4", "clips/neg1.mp4", "clips/neg2.mp4"},
                {record["video_path"] for record in gallery_records},
            )

    def test_validate_pilot_dataset_rejects_duplicate_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4", "other_target.mp4"):
                (root / "clips" / name).write_bytes(b"x")

            pilot_path = root / "pilot.jsonl"
            records = []
            duplicated_proposal_id = _build_proposal_id("clips/ref.mp4", "clips/target.mp4")
            for index, target in enumerate(("target.mp4", "other_target.mp4"), start=1):
                records.append(
                    {
                        "sample_id": f"covr_pilot_{index:04d}",
                        "proposal_id": duplicated_proposal_id,
                        "reference_video": "clips/ref.mp4",
                        "target_video": f"clips/{target}",
                        "edit_text": "change one cat into two cats",
                        "modalities": ["visual", "audio"],
                        "reference_caption": "one cat on a sofa",
                        "target_caption": "two cats on a sofa",
                        "difference": {"type": "object_count", "from": "one cat", "to": "two cats"},
                        "hard_negatives": ["clips/neg1.mp4", "clips/neg2.mp4"],
                        "quality": {
                            "same_context_score": 0.9,
                            "edit_match_score": 0.8,
                            "target_uniqueness_score": 0.7,
                        },
                        "source": {
                            "platform": "daily_omni",
                            "url": "file:///tmp/video.mp4",
                            "license_note": "internal research pilot only",
                        },
                    }
                )
            self._write_jsonl(pilot_path, records)

            with self.assertRaisesRegex(ValueError, "duplicate proposal_id"):
                validate_pilot_dataset(
                    root=root,
                    pilot_jsonl_path=pilot_path,
                    gallery_output_path=root / "gallery.jsonl",
                    report_output_path=root / "pilot_review.md",
                )

    def test_validate_pilot_dataset_rejects_reference_in_hard_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4"):
                (root / "clips" / name).write_bytes(b"x")

            pilot_path = root / "pilot.jsonl"
            record = {
                "sample_id": "covr_pilot_0001",
                "proposal_id": _build_proposal_id("clips/ref.mp4", "clips/target.mp4"),
                "reference_video": "clips/ref.mp4",
                "target_video": "clips/target.mp4",
                "edit_text": "change one cat into two cats",
                "modalities": ["visual", "audio"],
                "reference_caption": "one cat on a sofa",
                "target_caption": "two cats on a sofa",
                "difference": {"type": "object_count", "from": "one cat", "to": "two cats"},
                "hard_negatives": ["clips/ref.mp4", "clips/neg1.mp4"],
                "quality": {
                    "same_context_score": 0.9,
                    "edit_match_score": 0.8,
                    "target_uniqueness_score": 0.7,
                },
                "source": {
                    "platform": "daily_omni",
                    "url": "file:///tmp/video.mp4",
                    "license_note": "internal research pilot only",
                },
            }
            self._write_jsonl(pilot_path, [record])

            with self.assertRaisesRegex(ValueError, "reference_video cannot appear in hard_negatives"):
                validate_pilot_dataset(
                    root=root,
                    pilot_jsonl_path=pilot_path,
                    gallery_output_path=root / "gallery.jsonl",
                    report_output_path=root / "pilot_review.md",
                )


if __name__ == "__main__":
    unittest.main()
