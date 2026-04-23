from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.composed_data import (
    _compose_reject_reason,
    _detect_primary_difference,
    _difference_strength_score,
    _difference_priority_order,
    _effective_pair_quality,
    _build_pair_candidates,
    _build_proposal_id,
    _judge_accepts,
    _pair_context_score,
    _source_context,
    annotate_clips,
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

    def test_detect_primary_difference_prefers_speech_in_high_context_order(self) -> None:
        reference = {
            "object_counts": {"person": 1},
            "actions": ["speaking"],
            "audio_events": ["speech"],
            "attributes": ["studio"],
            "scene": "studio desk shot",
            "speech": ["welcome to the show"],
            "visible_text": ["episode 1"],
        }
        target = {
            "object_counts": {"person": 1},
            "actions": ["speaking"],
            "audio_events": ["speech"],
            "attributes": ["studio"],
            "scene": "studio desk shot",
            "speech": ["today we review the camera"],
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
                "speech": ["welcome to the lesson"],
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
                "speech": ["use the coupon code today"],
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
                "speech": ["welcome to the lesson"],
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
                "speech": ["thanks for watching"],
                "audio_events": ["speech"],
                "modalities": ["visual", "audio"],
            },
        ]

        candidates = _build_pair_candidates(root=Path("/tmp/composed"), annotations=annotations)
        proposal_ids = [candidate["proposal_id"] for candidate in candidates]

        self.assertEqual(len(proposal_ids), len(set(proposal_ids)))
        self.assertTrue(any(candidate["primary_difference"]["type"] == "speech" for candidate in candidates))

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
            self.assertIn("caption_equivalent_reject_count", report_path.read_text(encoding="utf-8"))
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
