from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.composed_data import (
    _accepted_sample_from_record,
    _action_evidence_score,
    _annotation_prompt_view,
    _apply_post_vace_semantic_verdict,
    _compose_reject_reason,
    _detect_primary_difference,
    _evidence_from_annotations,
    _difference_strength_score,
    _difference_priority_order,
    _effective_pair_quality,
    _edit_text_quality_payload,
    _build_pair_candidates,
    _build_proposal_id,
    _finalize_pair_verification,
    _has_intraclip_difference_conflict,
    _judge_accepts,
    _known_pair_generation_issues,
    _audio_event_independent_evidence_gate,
    _non_speech_audio_event_score,
    _observable_difference_gate,
    _pair_record_acceptance_issues,
    _pair_context_score,
    _pair_verification_counts,
    _maybe_reorient_candidate_for_model_fields,
    _model_difference_prefers_reverse_direction,
    _prepare_record_for_acceptance,
    _repair_pair_model_fields,
    _select_final_accepted_records,
    _build_fallback_edit_text,
    _speech_evidence_score,
    _speech_specificity_score,
    _source_context,
    _target_uniqueness_score,
    _video_edit_risk_assessment,
    _video_edit_plan_lint,
    _audit_src_ref_image_candidate,
    annotate_clips,
    build_manual_review_bundle,
    build_ffmpeg_extract_command,
    detective_annotate_clips,
    discover_raw_sources,
    ensure_layout,
    extract_clips,
    index_raw_sources,
    main as composed_data_main,
    plan_audio_edits,
    plan_detective_event_clips,
    plan_stable_omni_clips,
    cache_reference_understandings,
    plan_src_ref_images,
    select_src_ref_images,
    plan_video_masks,
    plan_video_edits,
    propose_group_pairs,
    propose_pairs,
    validate_known_pairs,
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

    def test_plan_stable_omni_clips_uses_cache_and_enforces_window_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw_datasets" / "daily_omni"
            source.mkdir(parents=True)
            video = source / "source.mp4"
            video.write_bytes(b"x")
            index_raw_sources(root=root, sources=[("daily_omni", source)])
            cache_path = root / "caches" / "stable_cache.jsonl"

            with mock.patch(
                "app.composed_data.probe_media",
                return_value={
                    "duration_seconds": 20.0,
                    "has_audio": True,
                    "has_video": True,
                    "width": 640,
                    "height": 360,
                    "fps": 25.0,
                },
            ):
                first = plan_stable_omni_clips(
                    root=root,
                    cache_path=cache_path,
                    max_source_videos=1,
                    min_clip_seconds=5.0,
                    max_clip_seconds=8.0,
                )
                second = plan_stable_omni_clips(
                    root=root,
                    cache_path=cache_path,
                    max_source_videos=1,
                    min_clip_seconds=5.0,
                    max_clip_seconds=8.0,
                )

            records = [
                json.loads(line)
                for line in Path(second["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, first["clip_plan_count"])
            self.assertEqual(1, second["cache_hits"])
            self.assertGreaterEqual(records[0]["duration_seconds"], 5.0)
            self.assertLessEqual(records[0]["duration_seconds"], 8.0)
            self.assertIn("stable_clip_selection", records[0])

    def test_plan_stable_omni_clips_persists_each_cache_record_before_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw_datasets" / "daily_omni"
            source.mkdir(parents=True)
            (source / "a.mp4").write_bytes(b"a")
            (source / "b.mp4").write_bytes(b"b")
            index_raw_sources(root=root, sources=[("daily_omni", source)])
            output_path = root / "metadata" / "stable_plan.jsonl"
            cache_path = root / "caches" / "stable_cache.jsonl"

            with mock.patch(
                "app.composed_data.probe_media",
                return_value={
                    "duration_seconds": 20.0,
                    "has_audio": True,
                    "has_video": True,
                    "width": 640,
                    "height": 360,
                    "fps": 25.0,
                },
            ), mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client = client_cls.return_value
                client.select_stable_clip_window.side_effect = [
                    (
                        {
                            "start_sec": 2.0,
                            "end_sec": 8.0,
                            "stability_score": 0.95,
                            "camera_motion": "static",
                            "main_subjects": ["robot"],
                            "visible_text_risk": False,
                            "recommended_for_vace": True,
                            "reason": "stable single subject",
                        },
                        {"provider": "mock"},
                    ),
                    KeyboardInterrupt(),
                ]

                with self.assertRaises(KeyboardInterrupt):
                    plan_stable_omni_clips(
                        root=root,
                        output_path=output_path,
                        cache_path=cache_path,
                        max_source_videos=2,
                        min_clip_seconds=5.0,
                        max_clip_seconds=8.0,
                        base_url="http://127.0.0.1:8093/v1",
                        api_key="EMPTY",
                        model="omni",
                    )

            cache_records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            plan_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, len(cache_records))
            self.assertEqual(1, len(plan_records))
            self.assertEqual("stable single subject", cache_records[0]["selection"]["reason"])

            with mock.patch(
                "app.composed_data.probe_media",
                return_value={
                    "duration_seconds": 20.0,
                    "has_audio": True,
                    "has_video": True,
                    "width": 640,
                    "height": 360,
                    "fps": 25.0,
                },
            ):
                resumed = plan_stable_omni_clips(
                    root=root,
                    output_path=output_path,
                    cache_path=cache_path,
                    max_source_videos=1,
                    min_clip_seconds=5.0,
                    max_clip_seconds=8.0,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="omni",
                )
            self.assertEqual(1, resumed["cache_hits"])

    def test_cache_reference_understandings_writes_stable_edit_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "robot_ref",
                        "output_path": "clips/robot_ref.mp4",
                        "summary": "a black and gold robotic action figure rotates on a platform",
                        "subjects": ["robotic action figure", "platform"],
                        "object_counts": {"robotic action figure": 1, "platform": 1},
                        "actions": ["rotating"],
                        "scene": "dark studio",
                    }
                ],
            )

            summary = cache_reference_understandings(root=root, clip_annotations_path=annotations_path)
            records = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["understanding_count"])
            self.assertEqual("robot_ref", records[0]["clip_id"])
            self.assertTrue(records[0]["stable_edit_targets"])
            self.assertEqual("robot body", records[0]["stable_edit_targets"][0]["target"])

    def test_cache_reference_understandings_skips_failed_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "empty_fallback",
                        "output_path": "clips/empty_fallback.mp4",
                        "summary": "",
                        "subjects": [],
                        "actions": [],
                        "scene": "",
                        "fallback_used": True,
                        "detective_fallback_reason": "detective_and_single_pass_failed",
                    },
                    {
                        "clip_id": "usable",
                        "output_path": "clips/usable.mp4",
                        "summary": "a red tote bag on a table",
                        "subjects": ["tote bag"],
                        "object_counts": {"tote bag": 1},
                        "actions": ["resting"],
                        "scene": "tabletop",
                    },
                ],
            )

            summary = cache_reference_understandings(root=root, clip_annotations_path=annotations_path)
            records = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["understanding_count"])
            self.assertEqual(1, summary["skipped_unusable_annotation_count"])
            self.assertEqual(["usable"], [record["clip_id"] for record in records])

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

    def test_annotation_prompt_view_truncates_long_fields(self) -> None:
        annotation = {
            "clip_id": "clip_long",
            "output_path": "clips/clip_long.mp4",
            "summary": "s" * 1200,
            "subjects": [f"subject {idx}" for idx in range(20)],
            "object_counts": {"subject": 1},
            "actions": ["walking"],
            "scene": "room",
            "attributes": [],
            "on_screen_text": [],
            "speech": ["speech " + "x" * 500],
            "audio_events": [],
            "modalities": ["visual"],
            "storyline": ["story " + "y" * 500 for _ in range(10)],
            "events": [{"description": "event " + "z" * 500, "irrelevant": "drop me"}],
            "visible_text": [],
            "speakers_and_transcript": ["speaker " + "t" * 500],
            "uncertainties": [],
        }

        prompt = _annotation_prompt_view(annotation)

        self.assertLessEqual(len(prompt["summary"]), 700)
        self.assertEqual(8, len(prompt["subjects"]))
        self.assertLessEqual(len(prompt["speech"][0]), 180)
        self.assertEqual(6, len(prompt["storyline"]))
        self.assertLessEqual(len(prompt["storyline"][0]), 220)
        self.assertEqual(["description"], list(prompt["events"][0].keys()))

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

    def test_annotate_clips_supports_concurrent_requests_and_preserves_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("clip_a.mp4", "clip_b.mp4", "clip_c.mp4"):
                (root / "clips" / name).write_bytes(name.encode("utf-8"))
            manifest_path = root / "metadata" / "clips.jsonl"
            self._write_jsonl(
                manifest_path,
                [
                    {"clip_id": "clip_a", "source_asset_id": "asset_a", "output_path": "clips/clip_a.mp4"},
                    {"clip_id": "clip_b", "source_asset_id": "asset_b", "output_path": "clips/clip_b.mp4"},
                    {"clip_id": "clip_c", "source_asset_id": "asset_c", "output_path": "clips/clip_c.mp4"},
                ],
            )

            class FakeClient:
                calls: list[str] = []

                def __init__(self, **_: object) -> None:
                    pass

                def annotate_clip(self, *, clip_path: str) -> tuple[dict[str, object], dict[str, object]]:
                    clip_id = Path(clip_path).stem
                    FakeClient.calls.append(clip_id)
                    return (
                        {
                            "summary": f"{clip_id} summary",
                            "subjects": [clip_id],
                            "object_counts": {clip_id: 1},
                            "actions": [],
                            "scene": "test scene",
                            "attributes": [],
                            "on_screen_text": [],
                            "speech": [],
                            "audio_events": [],
                            "modalities": ["visual"],
                        },
                        {"provider": "fake", "clip_id": clip_id},
                    )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", FakeClient):
                summary = annotate_clips(
                    root=root,
                    clips_manifest_path=manifest_path,
                    output_path=root / "captions" / "clip_annotations.jsonl",
                    base_url="http://127.0.0.1:8092/v1",
                    api_key="EMPTY",
                    model="captioner-model",
                    concurrency=2,
                )

            records = [
                json.loads(line)
                for line in (root / "captions" / "clip_annotations.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(3, summary["annotated_count"])
            self.assertEqual(2, summary["concurrency"])
            self.assertEqual(["clip_a", "clip_b", "clip_c"], [record["clip_id"] for record in records])
            self.assertEqual({"clip_a", "clip_b", "clip_c"}, set(FakeClient.calls))

    def test_detective_annotate_clips_persists_each_record_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "clip_a.mp4").write_bytes(b"a")
            (root / "clips" / "clip_b.mp4").write_bytes(b"b")
            manifest_path = root / "metadata" / "clips.jsonl"
            self._write_jsonl(
                manifest_path,
                [
                    {"clip_id": "clip_a", "source_asset_id": "asset_a", "output_path": "clips/clip_a.mp4"},
                    {"clip_id": "clip_b", "source_asset_id": "asset_b", "output_path": "clips/clip_b.mp4"},
                ],
            )
            output_path = root / "captions" / "clip_annotations_detective.jsonl"
            detective_output = (
                {
                    "summary": "a robot rotates in a studio",
                    "subjects": ["robot"],
                    "object_counts": {"robot": 1},
                    "actions": ["rotating"],
                    "scene": "studio",
                    "attributes": ["black and gold"],
                    "on_screen_text": [],
                    "speech": [],
                    "audio_events": [],
                    "modalities": ["visual"],
                    "storyline": ["robot rotates"],
                    "visible_text": [],
                    "speakers_and_transcript": [],
                    "detective_notes": [],
                    "detective_trajectory": [{"stage": "observer"}],
                    "uncertainties": [],
                },
                {"provider": "mock", "mode": "detective"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client = client_cls.return_value
                client.annotate_clip_detective.side_effect = [detective_output, KeyboardInterrupt()]
                with self.assertRaises(KeyboardInterrupt):
                    detective_annotate_clips(
                        root=root,
                        clips_manifest_path=manifest_path,
                        output_path=output_path,
                        base_url="http://127.0.0.1:8093/v1",
                        api_key="EMPTY",
                        model="qwen3-omni",
                        overwrite=True,
                    )
                self.assertEqual(2, client.annotate_clip_detective.call_count)

            partial_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(["clip_a"], [record["clip_id"] for record in partial_records])

            second_output = (
                {
                    "summary": "a tote bag rests on a table",
                    "subjects": ["tote bag"],
                    "object_counts": {"tote bag": 1},
                    "actions": [],
                    "scene": "room",
                    "attributes": ["red"],
                    "on_screen_text": [],
                    "speech": [],
                    "audio_events": [],
                    "modalities": ["visual"],
                    "storyline": ["bag on table"],
                    "visible_text": [],
                    "speakers_and_transcript": [],
                    "detective_notes": [],
                    "detective_trajectory": [{"stage": "observer"}],
                    "uncertainties": [],
                },
                {"provider": "mock", "mode": "detective"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client = client_cls.return_value
                client.annotate_clip_detective.return_value = second_output
                summary = detective_annotate_clips(
                    root=root,
                    clips_manifest_path=manifest_path,
                    output_path=output_path,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    overwrite=True,
                )
                self.assertEqual(1, client.annotate_clip_detective.call_count)

            final_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(2, summary["clip_count"])
            self.assertEqual(1, summary["reused_count"])
            self.assertEqual(1, summary["annotated_count"])
            self.assertEqual({"clip_a", "clip_b"}, {record["clip_id"] for record in final_records})

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
            verification_calls = client_cls.return_value.verify_pair_difference.call_args_list
            self.assertGreaterEqual(len(verification_calls), 1)
            for call in verification_calls:
                reference_clip_path = Path(call.kwargs["reference_clip_path"])
                target_clip_path = Path(call.kwargs["target_clip_path"])
                self.assertTrue(reference_clip_path.is_absolute())
                self.assertTrue(target_clip_path.is_absolute())
                self.assertTrue(reference_clip_path.exists())
                self.assertTrue(target_clip_path.exists())

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

    def test_finalize_pair_verification_rejects_order_only_difference(self) -> None:
        verification = _finalize_pair_verification(
            {
                "caption_delta": {
                    "caption_equivalent": False,
                    "has_concrete_difference": True,
                    "difference_matches_edit": True,
                    "concrete_differences": [
                        "Both clips show the same control room and space station shots, only in a different order."
                    ],
                    "reason": "The reference shows control room then space station, while the target reverses that sequence.",
                },
                "edit_projection": {
                    "target_matches_projection": True,
                    "score": 0.9,
                    "reason": "The same scenes are present, but their shot order differs.",
                },
                "edit_necessity": {
                    "edit_needed": True,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "score": 0.9,
                    "reason": "The videos share the same elements with different temporal order.",
                },
            }
        )

        self.assertFalse(verification["passed"])
        self.assertTrue(verification["caption_delta"]["caption_equivalent"])
        self.assertFalse(verification["caption_delta"]["has_concrete_difference"])
        self.assertFalse(verification["edit_necessity"]["edit_needed"])
        self.assertTrue(verification["edit_necessity"]["reference_satisfies_edit"])
        self.assertIn("same content appears in a different shot/order sequence", verification["caption_delta"]["reason"])

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

    def test_audio_event_independent_evidence_accepts_specific_absence_phrase(self) -> None:
        evidence = _audio_event_independent_evidence_gate(
            reference_annotation={"audio_events": []},
            target_annotation={"audio_events": ["whoosh"]},
            difference={"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(["whoosh"], evidence["target_evidence"])

    def test_synthetic_audio_evidence_corrects_contradictory_verification(self) -> None:
        record = {
            "source_type": "synthetic_edit",
            "reference_video": "clips/ref.mp4",
            "target_video": "clips/target.mp4",
            "edit_text": "add whoosh to the audio",
            "modalities": ["audio"],
            "difference": {"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
            "quality": {
                "same_context_score": 0.98,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "visual_near_duplicate_score": 0.99,
                "difference_type": "audio_event",
                "non_speech_audio_event_score": 0.7,
            },
            "generation": {
                "model": "ffmpeg-deterministic-audio",
                "model_route": "deterministic_overlay",
                "source_video": "clips/ref.mp4",
                "audio_edit_plan": {
                    "route": "deterministic_overlay",
                    "expected_event": "whoosh",
                    "audio_prompt": "whoosh",
                    "preserve_video": True,
                },
            },
            "judge": {
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "single_main_difference": True,
                "same_context_score": 0.98,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "hard_negative_quality": "good",
                "accept": True,
            },
            "verification": {
                "caption_delta": {
                    "caption_equivalent": True,
                    "has_concrete_difference": False,
                    "difference_matches_edit": False,
                    "concrete_differences": ["target contains whoosh, reference does not"],
                },
                "edit_projection": {"target_matches_projection": False, "score": 0.0},
                "edit_necessity": {
                    "edit_needed": False,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "score": 0.0,
                },
            },
        }

        prepared = _prepare_record_for_acceptance(
            record,
            reference_annotation={"audio_events": []},
            target_annotation={"audio_events": ["whoosh"]},
        )

        self.assertEqual(1.0, prepared["quality"]["audio_event_independent_evidence_passed"])
        self.assertTrue(prepared["verification"]["passed"])
        self.assertTrue(_judge_accepts(prepared["judge"], prepared["verification"], prepared["quality"]))

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

    def test_accepted_sample_carries_edit_text_and_observable_gate_fields(self) -> None:
        sample = _accepted_sample_from_record(
            {
                "proposal_id": "proposal__quality",
                "reference_video": "clips/ref.mp4",
                "target_video": "clips/target.mp4",
                "edit_text": "add a dollhouse",
                "modalities": ["visual"],
                "reference_caption": "a playroom",
                "target_caption": "a playroom with a dollhouse",
                "difference": {"type": "object_presence", "from": "no dollhouse", "to": "1 dollhouse"},
                "hard_negatives": ["clips/neg.mp4"],
                "quality": {"difference_type": "object_presence", "edit_text_quality_score": 1.0},
                "source": {"platform": "unknown", "url": "file:///tmp/target.mp4", "license_note": "internal"},
                "source_context": {"relation": "same_source_video"},
                "evidence": {},
                "judge": {},
                "verification": {},
                "edit_text_quality": {"score": 1.0, "bad_patterns": []},
                "observable_difference": {"passed": True, "supporting_fields": ["object_counts"]},
                "speech_quality": {},
                "audio_event_quality": {},
                "transcript_backed": None,
                "group_id": "group_a",
                "group_reason": "same_source_video",
            },
            1,
        )

        self.assertEqual(1.0, sample["edit_text_quality"]["score"])
        self.assertTrue(sample["observable_difference"]["passed"])

    def test_edit_text_quality_rejects_audio_event_visual_caption_leakage(self) -> None:
        quality = _edit_text_quality_payload(
            edit_text="add a woman with blonde hair and a nose ring speaking to the audio",
            difference={"type": "audio_event", "from": "no distinctive audio event", "to": "whoosh"},
            modalities=["audio"],
            reference_caption="A quiet room.",
            target_caption="A woman with blonde hair speaks in a room.",
        )

        self.assertFalse(quality["no_modality_leakage"])
        self.assertFalse(quality["matches_difference_type"])
        self.assertLess(quality["score"], 0.75)

    def test_repair_pair_model_fields_rewrites_malformed_object_presence_edit(self) -> None:
        repaired = _repair_pair_model_fields(
            model_fields={
                "edit_text": "change no man into 1 man",
                "modalities": ["visual"],
                "reference_caption": "an empty room",
                "target_caption": "a man appears in the room",
                "difference": {"type": "object_presence", "from": "no man", "to": "1 man"},
            },
            reference_annotation={"summary": "an empty room"},
            target_annotation={"summary": "a man appears in the room"},
        )

        self.assertEqual("add a man", repaired["edit_text"])

    def test_edit_text_quality_rejects_caption_like_target_copy(self) -> None:
        caption = "A man in a white shirt stands at a desk in a bright room and speaks to the camera."
        quality = _edit_text_quality_payload(
            edit_text=caption,
            difference={"type": "object_presence", "from": "no man", "to": "1 man"},
            modalities=["visual"],
            reference_caption="A bright room with a desk.",
            target_caption=caption,
        )

        self.assertFalse(quality["not_caption_like"])
        self.assertLess(quality["score"], 0.75)

    def test_observable_difference_gate_rejects_caption_only_visual_delta(self) -> None:
        gate = _observable_difference_gate(
            reference_annotation={
                "summary": "A toy robot rotates on a platform.",
                "object_counts": {"robot": 1, "platform": 1},
                "actions": ["rotating"],
                "visible_text": [],
            },
            target_annotation={
                "summary": "A black and gold robot figure rotates on a platform.",
                "object_counts": {"robot": 1, "platform": 1},
                "actions": ["rotating"],
                "visible_text": [],
            },
            difference={"type": "attribute", "from": "black and grey", "to": "black and gold"},
            visual_near_duplicate_score=0.996,
        )

        self.assertFalse(gate["passed"])
        self.assertEqual("high", gate["near_duplicate_risk"])

    def test_observable_difference_gate_rejects_human_group_presence_when_reference_already_has_people(self) -> None:
        gate = _observable_difference_gate(
            reference_annotation={
                "summary": "A busy control room with people working at desks, followed by a view of a space station.",
                "subjects": ["people", "control room"],
                "object_counts": {"people": 20, "desks": 8},
                "actions": ["working"],
                "visible_text": [],
            },
            target_annotation={
                "summary": "A space station in orbit, followed by a busy control room with personnel at work.",
                "subjects": ["personnel", "control room"],
                "object_counts": {"personnel": 20, "desks": 8},
                "actions": ["working"],
                "visible_text": [],
            },
            difference={
                "type": "object_presence",
                "from": "no control room personnel",
                "to": "20 control room personnel",
            },
            visual_near_duplicate_score=0.9,
        )

        self.assertFalse(gate["passed"])
        self.assertIn("reference already appears to contain", gate["failure_reason"])

    def test_observable_difference_gate_rejects_dollhouse_toy_house_alias(self) -> None:
        gate = _observable_difference_gate(
            reference_annotation={
                "summary": "A woman speaks in a classroom playroom with toys behind her.",
                "subjects": ["woman", "toy house"],
                "object_counts": {"toy house": 1, "teddy bear": 1},
                "actions": ["speaking"],
                "visible_text": [],
            },
            target_annotation={
                "summary": "A woman speaks in a room with a dollhouse and teddy bear behind her.",
                "subjects": ["woman", "dollhouse"],
                "object_counts": {"dollhouse": 1, "teddy bear": 1},
                "actions": ["speaking"],
                "visible_text": [],
            },
            difference={
                "type": "object_presence",
                "from": "no dollhouse",
                "to": "1 dollhouse",
            },
            visual_near_duplicate_score=0.52,
        )

        self.assertFalse(gate["passed"])
        self.assertIn("equivalent object", gate["failure_reason"])

    def test_observable_difference_gate_rejects_background_decor_without_frame_evidence(self) -> None:
        gate = _observable_difference_gate(
            reference_annotation={
                "summary": "A man and a woman argue in a living room.",
                "object_counts": {"woman": 1, "man": 1},
                "actions": ["arguing", "gesturing"],
                "storyline": ["The woman gestures while the man stands nearby."],
                "visible_text": ["MAKE WRONG"],
            },
            target_annotation={
                "summary": "A man enters a living room and interacts with a woman near a table.",
                "object_counts": {"woman": 1, "man": 1, "framed picture": 2},
                "actions": ["entering", "walking", "placing hands on"],
                "storyline": ["A man enters the room and places his hands on the woman's shoulders."],
                "visible_text": ["HUSTLE", "WORK"],
            },
            difference={
                "type": "object_presence",
                "from": "no framed picture",
                "to": "2 framed picture",
            },
            visual_near_duplicate_score=0.86,
        )

        self.assertFalse(gate["passed"])
        self.assertIn("background decor object lacks frame-level evidence", gate["failure_reason"])
        self.assertIn("competing stronger action difference", gate["failure_reason"])

    def test_observable_difference_gate_rejects_visible_text_with_competing_action_delta(self) -> None:
        gate = _observable_difference_gate(
            reference_annotation={
                "summary": "A man and a woman argue in a living room.",
                "actions": ["arguing", "gesturing", "turning away"],
                "visible_text": ["MAKE WRONG"],
            },
            target_annotation={
                "summary": "A man and a woman have a serious conversation in a living room.",
                "actions": ["talking", "listening", "gesturing"],
                "visible_text": ["PROTECT WHAT'S RIGHT"],
            },
            difference={
                "type": "visible_text",
                "from": "MAKE WRONG",
                "to": "PROTECT WHAT'S RIGHT",
            },
            visual_near_duplicate_score=0.93,
        )

        self.assertFalse(gate["passed"])
        self.assertIn("competing stronger action difference", gate["failure_reason"])

    def test_verification_edit_text_quality_check_blocks_override(self) -> None:
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
        }
        verification = {
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
            "edit_text_quality_check": {
                "not_caption_like": False,
                "matches_modality": True,
                "single_primary_difference": True,
                "reference_does_not_satisfy": True,
                "target_satisfies": True,
                "score": 0.4,
                "failure_reason": "caption-like edit",
            },
        }
        quality = {
            "same_context_score": 0.9,
            "edit_match_score": 0.9,
            "target_uniqueness_score": 0.9,
            "difference_strength_score": 0.8,
            "difference_type": "speech",
            "has_audio_modality": 1.0,
            "speech_transcript_backed": 1.0,
            "speech_evidence_score": 0.9,
            "speech_specificity_score": 0.9,
            "edit_text_quality_score": 1.0,
            "edit_text_is_imperative": 1.0,
            "edit_text_matches_difference_type": 1.0,
            "edit_text_single_change": 1.0,
            "edit_text_not_caption_like": 1.0,
            "edit_text_no_modality_leakage": 1.0,
        }

        self.assertFalse(_judge_accepts(judge, verification, quality))
        self.assertIn("edit_text_quality_check", _compose_reject_reason(judge, verification, quality))

    def test_prepare_record_recomputes_quality_after_local_verification_sync(self) -> None:
        record = {
            "edit_text": "add a dollhouse to the background",
            "modalities": ["visual"],
            "reference_caption": "A playroom with toy bins.",
            "target_caption": "A playroom with toy bins and a dollhouse.",
            "difference": {"type": "object_presence", "from": "no dollhouse", "to": "1 dollhouse"},
            "judge": {
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "single_main_difference": True,
                "same_context_score": 0.9,
                "edit_match_score": 0.4,
                "target_uniqueness_score": 0.9,
                "audio_required": False,
                "hard_negative_quality": "good",
                "accept": False,
            },
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
                "edit_text_quality_check": {
                    "not_caption_like": True,
                    "matches_modality": True,
                    "single_primary_difference": True,
                    "reference_does_not_satisfy": False,
                    "target_satisfies": False,
                    "score": 0.2,
                    "failure_reason": "model duplicated necessity check",
                },
            },
            "quality": {"same_context_score": 0.9, "edit_match_score": 0.4, "target_uniqueness_score": 0.9},
            "heuristic_quality": {
                "same_context_score": 0.9,
                "edit_match_score": 0.4,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.82,
                "difference_type": "object_presence",
                "edit_text_quality_score": 1.0,
                "edit_text_is_imperative": 1.0,
                "edit_text_matches_difference_type": 1.0,
                "edit_text_single_change": 1.0,
                "edit_text_not_caption_like": 1.0,
                "edit_text_no_modality_leakage": 1.0,
                "observable_difference_passed": 1.0,
            },
            "edit_text_quality": {
                "score": 1.0,
                "is_imperative_edit": True,
                "matches_difference_type": True,
                "single_change": True,
                "not_caption_like": True,
                "no_modality_leakage": True,
                "bad_patterns": [],
            },
            "observable_difference": {"passed": True, "supporting_fields": ["object_counts"]},
        }

        prepared = _prepare_record_for_acceptance(
            record,
            reference_annotation={"object_counts": {"toy bins": 2}},
            target_annotation={"object_counts": {"toy bins": 2, "dollhouse": 1}},
        )

        self.assertTrue(prepared["verification"]["passed"])
        self.assertEqual(0.9, prepared["quality"]["edit_match_score"])
        self.assertTrue(_judge_accepts(prepared["judge"], prepared["verification"], prepared["quality"]))

    def test_prepare_record_syncs_observable_difference_failure_into_verification(self) -> None:
        record = {
            "edit_text": "add 20 control room personnel to the scene",
            "modalities": ["visual"],
            "reference_caption": "A busy control room followed by a space station.",
            "target_caption": "A space station followed by a busy control room.",
            "difference": {
                "type": "object_presence",
                "from": "no control room personnel",
                "to": "20 control room personnel",
            },
            "judge": {
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "single_main_difference": True,
                "same_context_score": 0.9,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "audio_required": False,
                "hard_negative_quality": "good",
                "accept": True,
            },
            "verification": {
                "caption_delta": {
                    "caption_equivalent": False,
                    "has_concrete_difference": True,
                    "difference_matches_edit": True,
                    "reason": "The target has more personnel.",
                },
                "edit_projection": {
                    "target_matches_projection": True,
                    "score": 0.9,
                    "reason": "The target matches the projected edit.",
                },
                "edit_necessity": {
                    "edit_needed": True,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "score": 0.9,
                    "reason": "The reference lacks the edit.",
                },
            },
            "quality": {"same_context_score": 0.9, "edit_match_score": 0.9, "target_uniqueness_score": 0.9},
            "heuristic_quality": {
                "same_context_score": 0.9,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "difference_type": "object_presence",
                "edit_text_quality_score": 1.0,
                "observable_difference_passed": 0.0,
            },
            "edit_text_quality": {
                "score": 1.0,
                "is_imperative_edit": True,
                "matches_difference_type": True,
                "single_change": True,
                "not_caption_like": True,
                "no_modality_leakage": True,
                "bad_patterns": [],
            },
            "observable_difference": {
                "passed": False,
                "failure_reason": "reference already appears to contain control room personnel",
            },
        }

        prepared = _prepare_record_for_acceptance(
            record,
            reference_annotation={"summary": "A busy control room with people working at desks."},
            target_annotation={"summary": "A space station followed by a busy control room with personnel."},
        )

        self.assertFalse(prepared["verification"]["passed"])
        self.assertFalse(prepared["verification"]["caption_delta"]["has_concrete_difference"])
        self.assertFalse(prepared["verification"]["edit_necessity"]["edit_needed"])
        self.assertTrue(prepared["verification"]["edit_necessity"]["reference_satisfies_edit"])
        self.assertTrue(
            any("observable_difference gate failed" in failure for failure in prepared["verification"]["failures"])
        )
        self.assertFalse(_judge_accepts(prepared["judge"], prepared["verification"], prepared["quality"]))

    def test_prepare_record_keeps_local_bad_edit_text_rejected(self) -> None:
        record = {
            "edit_text": "A woman with blonde hair speaks in a room",
            "modalities": ["audio"],
            "reference_caption": "A quiet room.",
            "target_caption": "A woman with blonde hair speaks in a room.",
            "difference": {"type": "audio_event", "from": "none", "to": "whoosh"},
            "judge": {
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "single_main_difference": True,
                "same_context_score": 0.9,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "audio_required": True,
                "hard_negative_quality": "good",
                "accept": True,
            },
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
                "edit_text_quality_check": {"not_caption_like": True, "matches_modality": True, "score": 1.0},
            },
            "quality": {"same_context_score": 0.9, "edit_match_score": 0.9, "target_uniqueness_score": 0.9},
            "heuristic_quality": {
                "same_context_score": 0.9,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "difference_type": "audio_event",
                "non_speech_audio_event_score": 0.92,
                "edit_text_quality_score": 0.5,
                "edit_text_is_imperative": 1.0,
                "edit_text_matches_difference_type": 0.0,
                "edit_text_single_change": 1.0,
                "edit_text_not_caption_like": 0.0,
                "edit_text_no_modality_leakage": 0.0,
                "observable_difference_passed": 1.0,
            },
            "edit_text_quality": {
                "score": 0.5,
                "is_imperative_edit": True,
                "matches_difference_type": False,
                "single_change": True,
                "not_caption_like": False,
                "no_modality_leakage": False,
                "bad_patterns": ["audio_event edit_text contains visual subject"],
            },
            "observable_difference": {"passed": True, "supporting_fields": ["audio_events"]},
        }

        prepared = _prepare_record_for_acceptance(
            record,
            reference_annotation={"audio_events": []},
            target_annotation={"audio_events": ["whoosh"]},
        )

        self.assertFalse(prepared["verification"]["passed"])
        self.assertFalse(_judge_accepts(prepared["judge"], prepared["verification"], prepared["quality"]))
        self.assertIn("edit_text_quality_check", _compose_reject_reason(prepared["judge"], prepared["verification"], prepared["quality"]))

    def test_model_difference_direction_detects_reversed_reference_target(self) -> None:
        reference_annotation = {
            "summary": "A woman speaks in a room.",
            "object_counts": {"woman": 1, "room": 1},
            "actions": ["speaking"],
        }
        target_annotation = {
            "summary": "An empty room.",
            "object_counts": {"room": 1},
            "actions": [],
        }
        reversed_difference = {
            "type": "object_presence",
            "from": "no woman",
            "to": "1 woman",
            "description": "a woman appears",
        }
        forward_difference = {
            "type": "object_presence",
            "from": "1 woman",
            "to": "no woman",
            "description": "a woman disappears",
        }

        self.assertTrue(
            _model_difference_prefers_reverse_direction(
                difference=reversed_difference,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
        )
        self.assertFalse(
            _model_difference_prefers_reverse_direction(
                difference=forward_difference,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
        )

    def test_reorient_candidate_swaps_reference_and_target_for_reversed_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clips = root / "clips"
            clips.mkdir(parents=True)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4"):
                (clips / name).write_bytes(b"x")
            reference_annotation = {
                "clip_id": "ref",
                "output_path": "clips/ref.mp4",
                "summary": "A woman speaks in a room.",
                "object_counts": {"woman": 1, "room": 1},
                "subjects": ["woman"],
                "scene": "room",
                "actions": ["speaking"],
            }
            target_annotation = {
                "clip_id": "target",
                "output_path": "clips/target.mp4",
                "summary": "An empty room.",
                "object_counts": {"room": 1},
                "subjects": ["room"],
                "scene": "room",
                "actions": [],
            }
            neg1 = {
                "clip_id": "neg1",
                "output_path": "clips/neg1.mp4",
                "summary": "A man speaks in a room.",
                "object_counts": {"man": 1, "room": 1},
                "subjects": ["man"],
                "scene": "room",
                "actions": ["speaking"],
            }
            neg2 = {
                "clip_id": "neg2",
                "output_path": "clips/neg2.mp4",
                "summary": "A room with a chair.",
                "object_counts": {"chair": 1, "room": 1},
                "subjects": ["chair"],
                "scene": "room",
                "actions": [],
            }
            candidate = {
                "proposal_id": "proposal__forward",
                "reference_annotation": reference_annotation,
                "target_annotation": target_annotation,
                "primary_difference": {
                    "type": "object_presence",
                    "from": "1 woman",
                    "to": "no woman",
                    "description": "woman disappears",
                },
                "changed_difference_types": ["object_presence"],
                "quality": {"same_context_score": 0.9, "edit_match_score": 0.9, "target_uniqueness_score": 0.9},
                "source_context": {"relation": "same_source_video", "score": 0.9},
                "hard_negative_annotations": [neg1, neg2],
                "hard_negative_paths": ["clips/neg1.mp4", "clips/neg2.mp4"],
            }
            model_fields = {
                "edit_text": "add a woman to the scene",
                "modalities": ["visual"],
                "reference_caption": "An empty room.",
                "target_caption": "A woman speaks in a room.",
                "difference": {
                    "type": "object_presence",
                    "from": "no woman",
                    "to": "1 woman",
                    "description": "woman appears",
                },
                "proposal_reason": "model chose the reverse direction",
            }

            oriented, oriented_fields, swapped = _maybe_reorient_candidate_for_model_fields(
                root=root,
                candidate=candidate,
                model_fields=model_fields,
                annotations=[reference_annotation, target_annotation, neg1, neg2],
            )

        self.assertTrue(swapped)
        self.assertEqual("target", oriented["reference_annotation"]["clip_id"])
        self.assertEqual("ref", oriented["target_annotation"]["clip_id"])
        self.assertEqual("An empty room.", oriented_fields["reference_caption"])
        self.assertEqual("A woman speaks in a room.", oriented_fields["target_caption"])

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

    def test_prepare_record_rejects_audio_event_with_stronger_visible_text_delta(self) -> None:
        record = {
            "edit_text": "add whoosh to the audio",
            "modalities": ["audio"],
            "reference_caption": "A man speaks to camera with on-screen text.",
            "target_caption": "A man speaks to camera with different on-screen text and a whoosh.",
            "difference": {"type": "audio_event", "from": "no distinctive audio event", "to": "whoosh"},
            "judge": {
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "single_main_difference": True,
                "same_context_score": 0.9,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "audio_required": True,
                "hard_negative_quality": "good",
                "accept": True,
            },
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
            "quality": {"same_context_score": 0.9, "edit_match_score": 0.9, "target_uniqueness_score": 0.9},
            "heuristic_quality": {
                "same_context_score": 0.9,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "difference_type": "audio_event",
                "non_speech_audio_event_score": 0.92,
            },
        }

        prepared = _prepare_record_for_acceptance(
            record,
            reference_annotation={
                "audio_events": [],
                "visible_text": ["you're sending out mass emails"],
                "actions": ["speaking", "gesturing"],
            },
            target_annotation={
                "audio_events": ["whoosh"],
                "visible_text": ["here's the email subject line", "40% open rate"],
                "actions": ["speaking", "gesturing"],
            },
        )

        self.assertFalse(prepared["verification"]["passed"])
        self.assertFalse(_judge_accepts(prepared["judge"], prepared["verification"], prepared["quality"]))
        self.assertIn("competing stronger difference", _compose_reject_reason(prepared["judge"], prepared["verification"], prepared["quality"]))

    def test_prepare_record_rejects_audio_event_without_independent_audio_evidence(self) -> None:
        record = {
            "edit_text": "add whoosh to the audio",
            "modalities": ["audio"],
            "reference_caption": "A quiet visual scene.",
            "target_caption": "The same scene with a whoosh mentioned by caption only.",
            "difference": {"type": "audio_event", "from": "no distinctive audio event", "to": "whoosh"},
            "judge": {
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "single_main_difference": True,
                "same_context_score": 0.9,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "audio_required": True,
                "hard_negative_quality": "good",
                "accept": True,
            },
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
            "quality": {"same_context_score": 0.9, "edit_match_score": 0.9, "target_uniqueness_score": 0.9},
            "heuristic_quality": {
                "same_context_score": 0.9,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "difference_strength_score": 0.8,
                "difference_type": "audio_event",
                "non_speech_audio_event_score": 0.92,
            },
        }

        prepared = _prepare_record_for_acceptance(
            record,
            reference_annotation={"audio_events": [], "events": []},
            target_annotation={"audio_events": [], "events": []},
        )

        self.assertFalse(prepared["verification"]["passed"])
        self.assertFalse(_judge_accepts(prepared["judge"], prepared["verification"], prepared["quality"]))
        self.assertIn("audio_event lacks independent", _compose_reject_reason(prepared["judge"], prepared["verification"], prepared["quality"]))

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

    def test_select_final_accepted_records_dedupes_reused_target_video(self) -> None:
        base_record = {
            "accepted": True,
            "group_id": "group_text",
            "source_context": {"relation": "same_source_video"},
            "modalities": ["visual"],
            "target_video": "clips/shared_target.mp4",
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
            "quality": {
                "difference_type": "visible_text",
                "difference_strength_score": 0.85,
                "same_context_score": 0.9,
                "target_uniqueness_score": 0.8,
                "edit_match_score": 0.9,
            },
        }
        records = [
            {
                **base_record,
                "proposal_id": "proposal__text_a",
                "reference_video": "clips/ref_a.mp4",
                "reference_caption": "ref a",
                "edit_text": "change on-screen text from A to B",
                "difference": {"type": "visible_text", "from": "A", "to": "B"},
            },
            {
                **base_record,
                "proposal_id": "proposal__text_b",
                "reference_video": "clips/ref_b.mp4",
                "reference_caption": "ref b",
                "edit_text": "change on-screen text from C to D",
                "difference": {"type": "visible_text", "from": "C", "to": "D"},
                "quality": {
                    "difference_type": "visible_text",
                    "difference_strength_score": 0.8,
                    "same_context_score": 0.9,
                    "target_uniqueness_score": 0.8,
                    "edit_match_score": 0.9,
                },
            },
        ]

        accepted = _select_final_accepted_records(records, max_accepted_pairs=2)

        self.assertEqual(1, len(accepted))
        self.assertEqual("clips/shared_target.mp4", accepted[0]["target_video"])

    def test_select_final_accepted_records_keeps_distinct_synthetic_edits_with_same_delta(self) -> None:
        base_record = {
            "accepted": True,
            "source_type": "synthetic_edit",
            "group_id": "synthetic_robot_color",
            "source_context": {"relation": "synthetic_edit"},
            "modalities": ["visual"],
            "reference_caption": "a black and gold robot rotates on a platform",
            "target_caption": "a bright yellow robot rotates on the same platform",
            "hard_negatives": ["clips/neg.mp4"],
            "source": {"platform": "synthetic", "url": "file:///tmp/target.mp4", "license_note": "internal"},
            "evidence": {},
            "judge": {},
            "verification": {"passed": True},
            "speech_quality": {},
            "audio_event_quality": {},
            "transcript_backed": None,
            "group_reason": "synthetic_edit",
            "edit_text": "change robot body color from black and gold to bright yellow",
            "difference": {"type": "attribute", "from": "black and gold robot body", "to": "bright yellow robot body"},
            "quality": {
                "difference_type": "attribute",
                "difference_strength_score": 0.8,
                "same_context_score": 0.95,
                "target_uniqueness_score": 0.98,
                "edit_match_score": 0.9,
            },
        }
        records = [
            {
                **base_record,
                "proposal_id": "synthetic_visual_pair_plan_a",
                "reference_video": "clips/robot_seg_003.mp4",
                "target_video": "clips/synth_robot_seg_003_yellow.mp4",
            },
            {
                **base_record,
                "proposal_id": "synthetic_visual_pair_plan_b",
                "reference_video": "clips/robot_seg_004.mp4",
                "target_video": "clips/synth_robot_seg_004_yellow.mp4",
            },
        ]

        accepted = _select_final_accepted_records(records, max_accepted_pairs=5)

        self.assertEqual(2, len(accepted))
        self.assertEqual(
            {"clips/synth_robot_seg_003_yellow.mp4", "clips/synth_robot_seg_004_yellow.mp4"},
            {record["target_video"] for record in accepted},
        )
        self.assertEqual(2, len({record["sample_id"] for record in accepted}))

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

    def test_validate_known_pairs_writes_synthetic_generation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4", "src_video_for_vace.mp4", "mask.mp4", "raw.mp4"):
                (root / "clips" / name).write_bytes(b"x")
            (root / "review_inputs").mkdir()

            annotations_path = root / "captions" / "known_annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref",
                        "output_path": "clips/ref.mp4",
                        "summary": "a chair without a backpack",
                        "subjects": ["chair"],
                        "object_counts": {"chair": 1},
                        "actions": [],
                        "scene": "room",
                        "attributes": [],
                        "visible_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                    {
                        "clip_id": "target",
                        "output_path": "clips/target.mp4",
                        "summary": "a chair with a red backpack",
                        "subjects": ["chair", "red backpack"],
                        "object_counts": {"chair": 1, "red backpack": 1},
                        "actions": [],
                        "scene": "room",
                        "attributes": ["red"],
                        "visible_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                    {
                        "clip_id": "neg1",
                        "output_path": "clips/neg1.mp4",
                        "summary": "a chair with a blue backpack",
                        "subjects": ["chair", "blue backpack"],
                        "object_counts": {"chair": 1, "blue backpack": 1},
                        "actions": [],
                        "scene": "room",
                        "attributes": ["blue"],
                        "visible_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                    {
                        "clip_id": "neg2",
                        "output_path": "clips/neg2.mp4",
                        "summary": "a chair with a laptop",
                        "subjects": ["chair", "laptop"],
                        "object_counts": {"chair": 1, "laptop": 1},
                        "actions": [],
                        "scene": "room",
                        "attributes": [],
                        "visible_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                ],
            )
            known_pairs_path = root / "pairs" / "synthetic_candidate_pairs.jsonl"
            self._write_jsonl(
                known_pairs_path,
                [
                    {
                        "source_type": "synthetic_edit",
                        "reference_video": "clips/ref.mp4",
                        "target_video": "clips/target.mp4",
                        "edit_text": "add a red backpack on the chair",
                        "modalities": ["visual"],
                        "difference": {
                            "type": "object_presence",
                            "from": "no red backpack",
                            "to": "red backpack",
                            "description": "a red backpack is added to the chair",
                        },
                        "quality": {"visual_near_duplicate_score": 0.9},
                        "hard_negatives": ["clips/neg1.mp4", "clips/neg2.mp4"],
                        "generation": {
                            "model": "Wan2.1-VACE-1.3B",
                            "model_route": "vace_controlled",
                            "source_video": "clips/ref.mp4",
                            "prompt": "Only add a red backpack on the chair.",
                            "source_prompt": "a chair without a backpack",
                            "target_prompt": "a chair with a red backpack",
                            "preserve_tokens": ["chair", "room", "camera motion"],
                            "src_video_for_vace": "clips/src_video_for_vace.mp4",
                            "src_mask": "clips/mask.mp4",
                            "mask_semantics_version": 3,
                            "mask_polarity": "white_generate_black_preserve",
                            "mask_metrics": {"mask_coverage_ratio_avg": 0.12},
                            "review_inputs_dir": "review_inputs",
                            "duration_metrics": {
                                "raw_duration_drift_seconds": 0.0,
                                "target_duration_drift_seconds": 0.0,
                                "max_duration_drift_seconds": 0.5,
                                "duration_gate": {"passed": True},
                            },
                            "post_vace_verdict": {
                                "semantic_gate_required": False,
                                "semantic_gate_passed": True,
                            },
                            "postprocess": {
                                "audio_copied_from_reference": True,
                                "raw_generated_video": "clips/raw.mp4",
                            },
                            "seed": 1234,
                        },
                    }
                ],
            )
            verification = {
                "caption_delta": {
                    "caption_equivalent": False,
                    "has_concrete_difference": True,
                    "difference_matches_edit": True,
                },
                "edit_projection": {
                    "target_matches_projection": True,
                    "score": 0.95,
                },
                "edit_necessity": {
                    "edit_needed": True,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "score": 0.92,
                },
            }
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
            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client = client_cls.return_value
                client.judge_pair.return_value = (judge, {"provider": "mock"})
                client.verify_pair_difference.return_value = (verification, {"provider": "mock"})
                summary = validate_known_pairs(
                    root=root,
                    known_pairs_path=known_pairs_path,
                    clip_annotations_path=annotations_path,
                    output_path=root / "pairs" / "judged_synthetic_pair_proposals.jsonl",
                    accepted_output_path=root / "pairs" / "accepted_synthetic_pairs.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="instruct-model",
                    overwrite=True,
                )

            self.assertEqual(1, summary["accepted_count"])
            accepted_records = [
                json.loads(line)
                for line in (root / "pairs" / "accepted_synthetic_pairs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertRegex(accepted_records[0]["sample_id"], r"^covr_omni_synth_[0-9a-f]{8}$")
            self.assertEqual("synthetic_edit", accepted_records[0]["source_type"])
            self.assertEqual("Wan2.1-VACE-1.3B", accepted_records[0]["generation"]["model"])

    def test_validate_known_pairs_retries_verification_without_video_on_context_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4", "src_video_for_vace.mp4", "mask.mp4", "raw.mp4"):
                (root / "clips" / name).write_bytes(b"x")
            (root / "review_inputs").mkdir()

            annotations_path = root / "captions" / "known_annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref",
                        "output_path": "clips/ref.mp4",
                        "summary": "a chair without a backpack",
                        "subjects": ["chair"],
                        "object_counts": {"chair": 1},
                        "actions": [],
                        "scene": "room",
                        "attributes": [],
                        "visible_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                    {
                        "clip_id": "target",
                        "output_path": "clips/target.mp4",
                        "summary": "a chair with a red backpack",
                        "subjects": ["chair", "red backpack"],
                        "object_counts": {"chair": 1, "red backpack": 1},
                        "actions": [],
                        "scene": "room",
                        "attributes": ["red"],
                        "visible_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                    {
                        "clip_id": "neg1",
                        "output_path": "clips/neg1.mp4",
                        "summary": "a chair with a blue backpack",
                        "subjects": ["chair", "blue backpack"],
                        "object_counts": {"chair": 1, "blue backpack": 1},
                        "actions": [],
                        "scene": "room",
                        "attributes": ["blue"],
                        "visible_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                    {
                        "clip_id": "neg2",
                        "output_path": "clips/neg2.mp4",
                        "summary": "a chair with a laptop",
                        "subjects": ["chair", "laptop"],
                        "object_counts": {"chair": 1, "laptop": 1},
                        "actions": [],
                        "scene": "room",
                        "attributes": [],
                        "visible_text": [],
                        "speech": [],
                        "audio_events": [],
                        "modalities": ["visual"],
                    },
                ],
            )
            known_pairs_path = root / "pairs" / "synthetic_candidate_pairs.jsonl"
            self._write_jsonl(
                known_pairs_path,
                [
                    {
                        "source_type": "synthetic_edit",
                        "reference_video": "clips/ref.mp4",
                        "target_video": "clips/target.mp4",
                        "edit_text": "add a red backpack on the chair",
                        "modalities": ["visual"],
                        "difference": {
                            "type": "object_presence",
                            "from": "no red backpack",
                            "to": "red backpack",
                            "description": "a red backpack is added to the chair",
                        },
                        "quality": {"visual_near_duplicate_score": 0.9},
                        "hard_negatives": ["clips/neg1.mp4", "clips/neg2.mp4"],
                        "generation": {
                            "model": "Wan2.1-VACE-1.3B",
                            "model_route": "vace_controlled",
                            "source_video": "clips/ref.mp4",
                            "prompt": "Only add a red backpack on the chair.",
                            "target_prompt": "a chair with a red backpack",
                            "source_prompt": "a chair without a backpack",
                            "preserve_tokens": ["chair", "room", "camera motion"],
                            "src_video_for_vace": "clips/src_video_for_vace.mp4",
                            "src_mask": "clips/mask.mp4",
                            "mask_metrics": {"mask_coverage_ratio_avg": 0.12},
                            "review_inputs_dir": "review_inputs",
                            "duration_metrics": {
                                "raw_duration_drift_seconds": 0.0,
                                "target_duration_drift_seconds": 0.0,
                                "max_duration_drift_seconds": 0.5,
                                "duration_gate": {"passed": True},
                            },
                            "post_vace_verdict": {
                                "semantic_gate_required": False,
                                "semantic_gate_passed": True,
                            },
                            "postprocess": {
                                "audio_copied_from_reference": True,
                                "raw_generated_video": "clips/raw.mp4",
                            },
                            "seed": 1234,
                        },
                    }
                ],
            )
            verification = {
                "caption_delta": {
                    "caption_equivalent": False,
                    "has_concrete_difference": True,
                    "difference_matches_edit": True,
                },
                "edit_projection": {"target_matches_projection": True, "score": 0.95},
                "edit_necessity": {
                    "edit_needed": True,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "score": 0.92,
                },
            }
            judge = {
                "reference_satisfies_edit": False,
                "target_satisfies_edit": True,
                "single_main_difference": True,
                "same_context_score": 0.95,
                "edit_match_score": 0.9,
                "target_uniqueness_score": 0.9,
                "audio_required": False,
                "hard_negative_quality": "good",
                "accept": True,
                "reject_reason": "",
            }
            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client = client_cls.return_value
                client.judge_pair.return_value = (judge, {"provider": "mock"})
                client.verify_pair_difference.side_effect = [
                    RuntimeError("input length 21257 exceeds max_model_len 16384"),
                    (verification, {"provider": "mock-retry"}),
                ]
                summary = validate_known_pairs(
                    root=root,
                    known_pairs_path=known_pairs_path,
                    clip_annotations_path=annotations_path,
                    output_path=root / "pairs" / "judged_synthetic_pair_proposals.jsonl",
                    accepted_output_path=root / "pairs" / "accepted_synthetic_pairs.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="instruct-model",
                    overwrite=True,
                )

            self.assertEqual(2, client.verify_pair_difference.call_count)
            first_call = client.verify_pair_difference.call_args_list[0].kwargs
            retry_call = client.verify_pair_difference.call_args_list[1].kwargs
            self.assertEqual(Path(first_call["reference_clip_path"]).name, "ref.mp4")
            self.assertIsNone(retry_call["reference_clip_path"])
            self.assertIsNone(retry_call["target_clip_path"])
            judged = [
                json.loads(line)
                for line in (root / "pairs" / "judged_synthetic_pair_proposals.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(judged[0]["verification_annotation_only_retry_used"])
            self.assertIn("video_verification_error", judged[0]["raw_verification_output"])
            self.assertEqual({"provider": "mock-retry"}, judged[0]["raw_verification_output"]["annotation_only_retry"])

    def test_synthetic_sample_ids_are_stable_per_proposal_not_batch_index(self) -> None:
        first = _accepted_sample_from_record(
            {
                "source_type": "synthetic_edit",
                "proposal_id": "synthetic_visual_pair_plan_a",
                "reference_video": "clips/ref_a.mp4",
                "target_video": "clips/target_a.mp4",
                "edit_text": "change robot body color to yellow",
                "modalities": ["visual"],
                "difference": {"type": "attribute"},
                "reference_caption": "",
                "target_caption": "",
                "hard_negatives": [],
                "quality": {},
                "source": {},
                "judge": {},
                "verification": {},
            },
            1,
        )
        second = _accepted_sample_from_record(
            {
                "source_type": "synthetic_edit",
                "proposal_id": "synthetic_visual_pair_plan_b",
                "reference_video": "clips/ref_b.mp4",
                "target_video": "clips/target_b.mp4",
                "edit_text": "change robot body color to yellow",
                "modalities": ["visual"],
                "difference": {"type": "attribute"},
                "reference_caption": "",
                "target_caption": "",
                "hard_negatives": [],
                "quality": {},
                "source": {},
                "judge": {},
                "verification": {},
            },
            1,
        )

        self.assertRegex(first["sample_id"], r"^covr_omni_synth_[0-9a-f]{8}$")
        self.assertRegex(second["sample_id"], r"^covr_omni_synth_[0-9a-f]{8}$")
        self.assertNotEqual(first["sample_id"], second["sample_id"])

    def test_validate_known_pairs_rejects_missing_generation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4"):
                (root / "clips" / name).write_bytes(b"x")
            annotations_path = root / "captions" / "known_annotations.jsonl"
            base_annotation = {
                "summary": "a simple room",
                "subjects": ["chair"],
                "object_counts": {"chair": 1},
                "actions": [],
                "scene": "room",
                "attributes": [],
                "visible_text": [],
                "speech": [],
                "audio_events": [],
                "modalities": ["visual"],
            }
            self._write_jsonl(
                annotations_path,
                [
                    {"clip_id": "ref", "output_path": "clips/ref.mp4", **base_annotation},
                    {"clip_id": "target", "output_path": "clips/target.mp4", **base_annotation},
                    {"clip_id": "neg1", "output_path": "clips/neg1.mp4", **base_annotation},
                    {"clip_id": "neg2", "output_path": "clips/neg2.mp4", **base_annotation},
                ],
            )
            known_pairs_path = root / "pairs" / "synthetic_candidate_pairs.jsonl"
            self._write_jsonl(
                known_pairs_path,
                [
                    {
                        "source_type": "synthetic_edit",
                        "reference_video": "clips/ref.mp4",
                        "target_video": "clips/target.mp4",
                        "edit_text": "add a red backpack on the chair",
                        "modalities": ["visual"],
                        "difference": {
                            "type": "object_presence",
                            "from": "no red backpack",
                            "to": "red backpack",
                        },
                        "hard_negatives": ["clips/neg1.mp4", "clips/neg2.mp4"],
                    }
                ],
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient") as client_cls:
                client = client_cls.return_value
                client.judge_pair.return_value = (
                    {
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
                    },
                    {},
                )
                client.verify_pair_difference.return_value = (
                    {
                        "caption_delta": {
                            "caption_equivalent": False,
                            "has_concrete_difference": True,
                            "difference_matches_edit": True,
                        },
                        "edit_projection": {"target_matches_projection": True, "score": 0.95},
                        "edit_necessity": {
                            "edit_needed": True,
                            "reference_satisfies_edit": False,
                            "target_satisfies_edit": True,
                            "score": 0.92,
                        },
                    },
                    {},
                )
                summary = validate_known_pairs(
                    root=root,
                    known_pairs_path=known_pairs_path,
                    clip_annotations_path=annotations_path,
                    output_path=root / "pairs" / "judged_synthetic_pair_proposals.jsonl",
                    accepted_output_path=root / "pairs" / "accepted_synthetic_pairs.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="instruct-model",
                    overwrite=True,
                )

            self.assertEqual(0, summary["accepted_count"])
            judged = [
                json.loads(line)
                for line in (root / "pairs" / "judged_synthetic_pair_proposals.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("generation", judged[0]["judge"]["reject_reason"])

    def test_plan_video_edits_uses_omni_prompt_planner_and_excludes_audio_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a person holds a mobile phone at a desk",
                        "subjects": ["person", "mobile phone", "desk"],
                        "object_counts": {"person": 1, "mobile phone": 1, "desk": 1},
                        "actions": ["holding"],
                        "scene": "desk",
                    },
                    {
                        "clip_id": "ref_audio",
                        "output_path": "clips/ref_audio.mp4",
                        "summary": "a person jumps",
                        "subjects": ["person"],
                        "actions": ["jumping"],
                        "audio_events": [],
                    },
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "visual_1",
                        "reference_video": "clips/ref_visual.mp4",
                        "reference_caption": "a person holds a mobile phone at a desk",
                        "edit_text": "replace the mobile phone with a tablet",
                        "difference": {"type": "object_presence", "from": "mobile phone", "to": "tablet"},
                    },
                    {
                        "proposal_id": "audio_1",
                        "reference_video": "clips/ref_audio.mp4",
                        "edit_text": "add whoosh to the audio",
                        "difference": {"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
                    },
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertEqual("visual_1", plans[0]["plan_id"])
            self.assertEqual("heuristic_prompt_planner", plans[0]["planner"]["stage"])
            self.assertTrue(plans[0]["planner"]["fallback_used"])
            self.assertIn("mobile phone", plans[0]["source_prompt"])
            self.assertIn("tablet", plans[0]["target_prompt"])
            self.assertEqual(1, summary["skipped_by_type"]["audio_event"])

    def test_plan_video_edits_skips_unusable_reference_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "fallback_ref",
                        "output_path": "clips/fallback_ref.mp4",
                        "summary": "",
                        "subjects": [],
                        "actions": [],
                        "scene": "",
                        "fallback_used": True,
                        "detective_fallback_reason": "detective_and_single_pass_failed",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "proposal__fallback",
                        "reference_video": "clips/fallback_ref.mp4",
                        "edit_text": "change the robot body color from black and gold to bright yellow",
                        "difference": {"type": "attribute", "from": "black and gold", "to": "bright yellow"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual([], plans)
            self.assertEqual(1, summary["skipped_reasons"]["reference_annotation_unusable"])

    def test_plan_video_edits_rewrites_replacement_without_preserving_source_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a man sits beside a cup on a small table",
                        "subjects": ["man", "cup", "table"],
                        "object_counts": {"man": 1, "cup": 1, "table": 1},
                        "actions": ["sitting"],
                        "scene": "office",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "cup_to_bottle",
                        "reference_video": "clips/ref_visual.mp4",
                        "edit_text": "replace the cup with a bottle",
                        "difference": {"type": "object_presence", "from": "cup", "to": "bottle"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertNotIn("Add only", plans[0]["target_prompt"])
            self.assertIn("Replace only the cup with bottle", plans[0]["target_prompt"])
            self.assertIn("no cup is visible", plans[0]["target_prompt"].lower())
            self.assertNotIn("cup", [item.lower() for item in plans[0]["preserve_tokens"]])
            self.assertNotIn("cup", plans[0]["negative_prompt"].lower())
            self.assertTrue(plans[0]["plan_lint"]["passed"])

    def test_plan_video_edits_rewrites_removal_without_add_only_no(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a table has a cup beside a notebook",
                        "subjects": ["table", "cup", "notebook"],
                        "object_counts": {"table": 1, "cup": 1, "notebook": 1},
                        "actions": ["static"],
                        "scene": "desk",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "remove_cup",
                        "reference_video": "clips/ref_visual.mp4",
                        "edit_text": "remove the cup from the scene",
                        "difference": {"type": "object_presence", "from": "cup", "to": "no cup"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertNotIn("Add only no", plans[0]["target_prompt"])
            self.assertIn("Remove only the cup", plans[0]["target_prompt"])
            self.assertNotIn("cup", plans[0]["negative_prompt"].lower())
            self.assertTrue(plans[0]["plan_lint"]["passed"])

    def test_plan_video_edits_rewrites_safe_clothing_prompt_without_source_clothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "musician_ref",
                        "output_path": "clips/musician_ref.mp4",
                        "summary": "a man in a blue fedora and patterned shirt plays a ukulele",
                        "subjects": ["man"],
                        "object_counts": {"man": 1, "ukulele": 1, "microphone": 1},
                        "actions": ["playing ukulele"],
                        "scene": "brick wall",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "solid_black_shirt",
                        "reference_video": "clips/musician_ref.mp4",
                        "edit_text": "change the patterned shirt into a solid black shirt",
                        "difference": {
                            "type": "attribute",
                            "from": "patterned shirt",
                            "to": "solid black shirt",
                        },
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertIn("solid black shirt", plans[0]["target_prompt"])
            self.assertIn("blue fedora", plans[0]["target_prompt"])
            self.assertNotIn("patterned shirt", plans[0]["target_prompt"])
            self.assertNotIn("jacket", plans[0]["target_prompt"].lower())
            self.assertNotIn("Change only", plans[0]["target_prompt"])
            self.assertTrue(plans[0]["plan_lint"]["passed"])

    def test_plan_video_edits_repairs_background_replace_prompt_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "woman_ref.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "woman_ref",
                        "output_path": "clips/woman_ref.mp4",
                        "summary": "A woman with curly red hair and glasses speaks to the camera in a sunlit room.",
                        "subjects": ["woman"],
                        "object_counts": {"woman": 1, "glasses": 1, "window": 1, "door": 1},
                        "actions": ["speaking"],
                        "scene": "sunlit room",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "woman_lab",
                        "reference_video": "clips/woman_ref.mp4",
                        "reference_caption": "A woman with curly red hair and glasses speaks in a sunlit room.",
                        "edit_text": "change the background to a futuristic laboratory",
                        "difference": {"type": "scene", "from": "sunlit room background", "to": "futuristic laboratory background"},
                    }
                ],
            )

            fake_client = mock.Mock()
            fake_client.plan_video_edit.return_value = (
                {
                    "should_generate": True,
                    "source_prompt": "A woman with curly red hair and glasses speaks to the camera in a sunlit room.",
                    "target_prompt": (
                        "A woman with curly red hair and glasses speaks to the camera in a sunlit room "
                        "with a futuristic laboratory background."
                    ),
                    "edit_token": "futuristic laboratory background",
                    "preserve_tokens": ["woman", "curly red hair", "glasses", "speaking", "sunlit room", "lighting", "layout"],
                    "negative_prompt": "Do not change the woman, lighting, timing, layout, sunlit room, window, or door.",
                    "edit_region": "background",
                    "mask_query": "background",
                    "preserve_regions": ["woman", "window", "door"],
                    "model_route": "vace_controlled",
                    "reason": "Background can be changed while preserving the woman.",
                    "repaired_fields": [],
                },
                {"raw": "planner"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=fake_client):
                summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            plan = plans[0]
            target_key = plan["target_prompt"].lower()
            self.assertIn("futuristic laboratory", target_key)
            self.assertIn("stable frontal medium-close-up framing", target_key)
            self.assertNotIn("sunlit room", target_key)
            self.assertNotIn("window", target_key)
            self.assertNotIn("door", target_key)
            self.assertNotIn("lighting", [item.lower() for item in plan["preserve_tokens"]])
            self.assertNotIn("layout", [item.lower() for item in plan["preserve_tokens"]])
            self.assertNotIn("sunlit room", [item.lower() for item in plan["preserve_tokens"]])
            self.assertNotIn("window", [item.lower() for item in plan["preserve_regions"]])
            self.assertNotIn("door", [item.lower() for item in plan["preserve_regions"]])
            self.assertIn("camera framing", [item.lower() for item in plan["preserve_tokens"]])
            self.assertNotIn("sunlit room", plan["negative_prompt"].lower())
            self.assertNotIn("window", plan["negative_prompt"].lower())
            self.assertNotIn("door", plan["negative_prompt"].lower())
            self.assertFalse(any("layout exactly" in lock.lower() for lock in plan["visual_edit_risk"]["locks"]))
            self.assertFalse(any("lighting" in lock.lower() for lock in plan["visual_edit_risk"]["locks"]))
            self.assertFalse(any("layout" in lock.lower() for lock in plan["visual_edit_risk"]["locks"]))
            self.assertFalse(any("window" in lock.lower() or "door" in lock.lower() for lock in plan["visual_edit_risk"]["locks"]))
            self.assertFalse(plan["route_suitability"]["production_allowed"])
            self.assertFalse(plan["route_suitability"]["plain_masked_vace_production"])
            self.assertEqual("vace_bg_replace_composite_first_frame_mv2v", plan["route_suitability"]["recommended_route"])
            self.assertEqual("vace_bg_replace_composite_first_frame_mv2v", plan["background_replace_policy"]["recommended_route"])
            self.assertFalse(plan["background_replace_policy"]["plain_masked_vace_production"])
            self.assertTrue(plan["background_replace_policy"]["requires_composite_first_frame"])
            self.assertIn("target_prompt_rewritten_for_background_replace", plan["planner"]["repaired_fields"])
            self.assertIn("visual_edit_risk_locks_rewritten_for_background_replace", plan["planner"]["repaired_fields"])
            self.assertIn("preserve_regions_rewritten_for_background_replace", plan["planner"]["repaired_fields"])
            self.assertTrue(plan["plan_lint"]["passed"])

    def test_plan_video_edits_rejects_structural_clothing_after_model_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "musician_ref.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "musician_ref",
                        "output_path": "clips/musician_ref.mp4",
                        "summary": "a man in a blue fedora and patterned shirt plays a ukulele and sings",
                        "subjects": ["man"],
                        "object_counts": {"man": 1, "ukulele": 1, "microphone": 1},
                        "actions": ["playing ukulele", "singing"],
                        "scene": "brick wall",
                        "visible_text": ["S"],
                        "storyline": ["plays ukulele", "sings", "moves in place"],
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "seed",
                        "reference_video": "clips/musician_ref.mp4",
                        "reference_caption": "a man in a patterned shirt plays ukulele",
                        "edit_text": "exploration seed",
                        "difference": {"type": "attribute", "from": "reference video", "to": "visual edit"},
                    }
                ],
            )

            fake_client = mock.Mock()
            fake_client.plan_video_edit.return_value = (
                {
                    "should_generate": True,
                    "edit_text": "change the outfit into a black jacket",
                    "difference": {
                        "type": "attribute",
                        "from": "man wearing blue and white patterned shirt",
                        "to": "man wearing black jacket",
                    },
                    "source_prompt": "A man in a blue fedora and patterned shirt plays a ukulele and sings into a microphone against a brick wall.",
                    "target_prompt": "A man in a blue fedora and a black jacket plays a ukulele and sings into a microphone against a brick wall.",
                    "edit_token": "black jacket",
                    "preserve_tokens": ["man", "ukulele", "microphone", "camera motion", "lighting"],
                    "negative_prompt": "Do not change the man, ukulele, microphone, camera motion, lighting, timing, or visible text.",
                    "edit_region": "clothing",
                    "mask_query": "clothing",
                    "model_route": "vace_controlled",
                    "reason": "The clothing is visible and maskable.",
                    "repaired_fields": [],
                },
                {"raw": "planner"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=fake_client):
                summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=1,
                    planning_mode="exploration",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual([], plans)
            self.assertEqual(0, summary["plan_count"])
            self.assertGreaterEqual(summary["skipped_reasons"]["plan_lint_structural_clothing_tryon_required"], 1)
            self.assertNotIn("model_planner_revised_to_high_risk_high", summary["skipped_reasons"])

    def test_plan_video_edits_rejects_screen_text_object_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a person wipes a laptop displaying a webpage",
                        "subjects": ["person", "laptop"],
                        "object_counts": {"person": 1, "laptop": 1},
                        "actions": ["wiping"],
                        "scene": "wooden table",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "laptop_to_tablet",
                        "reference_video": "clips/ref_visual.mp4",
                        "edit_text": "replace the laptop with a tablet",
                        "difference": {"type": "object_presence", "from": "laptop", "to": "tablet"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["plan_lint_object_replacement_screen_or_visible_text_risk"])

    def test_plan_video_edits_rejects_replacement_when_source_object_not_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a man sits beside a small table in an office",
                        "subjects": ["man", "table"],
                        "object_counts": {"man": 1, "table": 1},
                        "actions": ["sitting"],
                        "scene": "office",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "cup_to_bottle",
                        "reference_video": "clips/ref_visual.mp4",
                        "edit_text": "replace the cup with a bottle",
                        "difference": {"type": "object_presence", "from": "cup", "to": "bottle"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["plan_lint_object_replacement_source_not_visible"])

    def test_plan_video_edits_rejects_removing_seated_support_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a young girl sits in a chair holding a toy",
                        "subjects": ["young girl", "chair"],
                        "object_counts": {"young girl": 1, "chair": 1},
                        "actions": ["sitting"],
                        "scene": "indoor room",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "remove_chair",
                        "reference_video": "clips/ref_visual.mp4",
                        "edit_text": "remove the chair from the scene",
                        "difference": {"type": "object_presence", "from": "chair", "to": "no chair"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["plan_lint_object_removal_breaks_seated_support"])

    def test_plan_video_edits_rejects_replacing_seated_support_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a young girl sits in a chair holding a toy",
                        "subjects": ["young girl", "chair"],
                        "object_counts": {"young girl": 1, "chair": 1},
                        "actions": ["sitting"],
                        "scene": "indoor room",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "replace_chair",
                        "reference_video": "clips/ref_visual.mp4",
                        "edit_text": "replace the chair with a stool",
                        "difference": {"type": "object_presence", "from": "chair", "to": "stool"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["plan_lint_object_replacement_breaks_support_contact"])

    def test_plan_video_edits_rejects_ambiguous_multi_instance_mask_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "two empty chairs are visible on a stage",
                        "subjects": ["stage", "chair"],
                        "object_counts": {"chair": 2, "stage": 1},
                        "actions": [],
                        "scene": "stage",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "replace_chair",
                        "reference_video": "clips/ref_visual.mp4",
                        "edit_text": "replace the chair with a stool",
                        "difference": {"type": "object_presence", "from": "chair", "to": "stool"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["plan_lint_ambiguous_multi_instance_mask_query"])

    def test_plan_video_edits_rejects_multi_scene_background_reference_before_mask_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "bg_ref.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "bg_ref",
                        "output_path": "clips/bg_ref.mp4",
                        "summary": "a man speaks across multiple scenes in different locations",
                        "subjects": ["man"],
                        "object_counts": {"man": 1},
                        "actions": ["speaking"],
                        "scene": "indoor room",
                        "stable_scene": "multiple scenes across different locations",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "bg_lab",
                        "reference_video": "clips/bg_ref.mp4",
                        "edit_text": "change the background to a futuristic laboratory",
                        "difference": {
                            "type": "scene",
                            "from": "original room background",
                            "to": "futuristic laboratory background",
                        },
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["vace_background_edit_multi_scene_reference"])

    def test_plan_video_edits_rejects_multi_subject_background_reference_before_mask_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "bg_ref.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "bg_ref",
                        "output_path": "clips/bg_ref.mp4",
                        "summary": "two speakers stand side by side in a studio",
                        "subjects": ["man with beard", "woman with glasses"],
                        "object_counts": {"man": 1, "woman": 1},
                        "actions": ["speaking"],
                        "scene": "studio wall",
                        "stable_scene": "studio wall",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "bg_lab",
                        "reference_video": "clips/bg_ref.mp4",
                        "edit_text": "change the background to a futuristic laboratory",
                        "difference": {
                            "type": "scene",
                            "from": "studio background",
                            "to": "futuristic laboratory background",
                        },
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["vace_background_edit_multi_subject_reference"])

    def test_video_edit_plan_lint_rejects_replacement_prompt_source_state_conflict(self) -> None:
        lint = _video_edit_plan_lint(
            source_prompt="A flutist is sitting on a chair on stage.",
            target_prompt=(
                "A flutist is sitting on a chair on stage. Replace only the chair with stool. "
                "The same shot shows stool in the original chair location; no chair is visible."
            ),
            edit_text="replace the chair with a stool",
            difference={"type": "object_presence", "from": "chair", "to": "stool"},
            edit_token="stool",
            preserve_tokens=["flutist", "stage"],
            negative_prompt="preserve the flutist and stage",
            reference_annotation={
                "summary": "A flutist is sitting on a chair on stage.",
                "subjects": ["flutist", "chair"],
                "object_counts": {"chair": 1},
                "actions": ["sitting"],
            },
            mask_query="chair",
        )

        self.assertIn("replacement_target_prompt_conflicts_with_source_state", lint["errors"])

    def test_video_edit_plan_lint_rejects_visible_text_or_logo_edit(self) -> None:
        lint = _video_edit_plan_lint(
            source_prompt="A hand reveals hologram text reading made in Slovenia.",
            target_prompt="A hand reveals hologram text reading made in Macedonia.",
            edit_text="made in macedonia",
            difference={"type": "attribute", "from": "made in Slovenia", "to": "made in Macedonia"},
            edit_token="made in macedonia",
            preserve_tokens=["hand", "hologram"],
            negative_prompt="preserve the hand and hologram",
            reference_annotation={"summary": "A hand reveals text.", "visible_text": ["MADE IN SLOVENIA"]},
        )

        self.assertIn("visible_text_or_logo_edit", lint["errors"])

    def test_plan_video_edits_uses_model_prompt_planner_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "ref_visual.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a person holds a mobile phone at a desk",
                        "subjects": ["person", "mobile phone", "desk"],
                        "object_counts": {"person": 1, "mobile phone": 1, "desk": 1},
                        "actions": ["holding"],
                        "scene": "desk",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "visual_1",
                        "reference_video": "clips/ref_visual.mp4",
                        "reference_caption": "a person holds a mobile phone at a desk",
                        "edit_text": "replace the mobile phone with a tablet",
                        "difference": {"type": "object_presence", "from": "mobile phone", "to": "tablet"},
                    }
                ],
            )

            fake_client = mock.Mock()
            fake_client.plan_video_edit.return_value = (
                {
                    "should_generate": True,
                    "source_prompt": "Omni source prompt: a person holds a phone at a desk.",
                    "target_prompt": "Omni target prompt: same shot, replace only the phone with a tablet. No mobile phone is visible.",
                    "edit_token": "tablet",
                    "preserve_tokens": ["person", "desk", "camera motion", "lighting"],
                    "negative_prompt": "Do not change the person, desk, camera motion, lighting, timing, or visible text.",
                    "edit_region": "hand-held object",
                    "model_route": "vace_controlled",
                    "reason": "The object is visible and localized.",
                    "repaired_fields": ["target_prompt"],
                },
                {"raw": "planner"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=fake_client) as client_cls:
                summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, summary["plan_count"])
            client_cls.assert_called_once()
            fake_client.plan_video_edit.assert_called_once()
            self.assertEqual("strongest_omni_prompt_planner", plans[0]["planner"]["stage"])
            self.assertFalse(plans[0]["planner"]["fallback_used"])
            self.assertEqual("qwen3-omni", plans[0]["planner"]["model"])
            self.assertEqual(["target_prompt"], plans[0]["planner"]["repaired_fields"])
            self.assertEqual("tablet", plans[0]["edit_token"])
            self.assertEqual("hand-held object", plans[0]["edit_region"])
            self.assertIn("replace only the phone", plans[0]["target_prompt"])
            self.assertEqual({"raw": "planner"}, plans[0]["raw_planner_output"])

    def test_plan_video_edits_rejects_clean_naked_object_insertion_for_vace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "robot_ref",
                        "output_path": "clips/robot_ref.mp4",
                        "summary": "a black and gold robotic action figure rotates on a reflective platform",
                        "subjects": ["robotic action figure", "platform"],
                        "object_counts": {"robotic action figure": 1, "platform": 1},
                        "actions": ["rotating"],
                        "scene": "dark studio",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "plant_insert",
                        "reference_video": "clips/robot_ref.mp4",
                        "reference_caption": "a robot rotates on a platform",
                        "edit_text": "add a medium green potted plant in the background",
                        "difference": {
                            "type": "object_presence",
                            "from": "no medium green potted plant",
                            "to": "medium green potted plant",
                        },
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([], plans)
            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["vace_rejects_tiny_or_naked_object_edit"])

    def test_plan_video_edits_uses_model_revised_safe_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "ref_visual.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a hand writes on a white paper on a desk",
                        "subjects": ["hand", "paper", "desk"],
                        "object_counts": {"hand": 1, "paper": 1, "desk": 1},
                        "actions": ["writing"],
                        "scene": "desk",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "visual_1",
                        "reference_video": "clips/ref_visual.mp4",
                        "reference_caption": "a hand writes on paper",
                        "edit_text": "add a robot action figure",
                        "difference": {"type": "object_presence", "from": "no robot action figure", "to": "robot action figure"},
                    }
                ],
            )

            fake_client = mock.Mock()
            fake_client.plan_video_edit.return_value = (
                {
                    "should_generate": True,
                    "edit_text": "change the paper color from white to pale blue",
                    "difference": {"type": "attribute", "from": "white paper", "to": "pale blue paper"},
                    "source_prompt": "A close-up video of a hand writing on white paper on a desk.",
                    "target_prompt": "The same close-up video of the same hand writing on the same desk, but the paper surface is pale blue while everything else stays unchanged.",
                    "edit_token": "pale blue paper",
                    "preserve_tokens": ["hand", "paper", "desk", "writing", "camera motion", "lighting"],
                    "negative_prompt": "Do not change the hand, paper, writing, desk, camera motion, lighting, timing, or visible text.",
                    "edit_region": "paper surface",
                    "model_route": "vace_controlled",
                    "reason": "The candidate object insertion is unsuitable, so the planner chose a safer large visible color edit.",
                    "repaired_fields": [],
                },
                {"raw": "planner"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=fake_client):
                summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, summary["plan_count"])
            self.assertEqual("change the paper color from white to pale blue", plans[0]["edit_text"])
            self.assertEqual({"type": "attribute", "from": "white paper", "to": "pale blue paper"}, plans[0]["difference"])
            self.assertEqual("pale blue paper", plans[0]["edit_token"])
            self.assertEqual("paper surface", plans[0]["edit_region"])
            self.assertEqual("vace_controlled", plans[0]["model_route"])

    def test_plan_video_edits_prefers_reference_attribute_ideation_over_action_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "robot_ref.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "robot_ref",
                        "output_path": "clips/robot_ref.mp4",
                        "summary": "a black and gold robotic action figure rotates on a reflective platform in a dark studio",
                        "subjects": ["robotic action figure", "platform"],
                        "object_counts": {"robotic action figure": 1, "platform": 1},
                        "actions": ["rotating"],
                        "scene": "dark studio",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "robot_action",
                        "reference_video": "clips/robot_ref.mp4",
                        "reference_caption": "a robot rotates on a platform",
                        "edit_text": "change the action from rotating to hovering",
                        "difference": {"type": "action", "from": "rotating", "to": "hovering"},
                    }
                ],
            )

            fake_client = mock.Mock()
            fake_client.plan_video_edit.return_value = (
                {
                    "should_generate": True,
                    "edit_text": "change the robot body color from black and gold to bright yellow",
                    "difference": {
                        "type": "attribute",
                        "from": "black and gold robot body",
                        "to": "bright yellow robot body",
                    },
                    "source_prompt": "A black and gold robotic action figure rotates on a reflective platform in a dark studio.",
                    "target_prompt": "The same robotic action figure rotates on the same platform in the same dark studio, but the robot body is bright yellow.",
                    "edit_token": "bright yellow robot body",
                    "preserve_tokens": ["yellow visor", "platform", "dark studio", "rotation", "camera motion", "lighting"],
                    "negative_prompt": "Do not change the platform, camera motion, lighting, timing, background, or visible text.",
                    "edit_region": "robot body",
                    "model_route": "vace_controlled",
                    "reason": "The robot body color is a safer VACE edit than changing the action.",
                    "repaired_fields": [],
                },
                {"raw": "planner"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=fake_client):
                summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["safe_visual_ideation_from_non_vace_candidate"])
            self.assertEqual("change the action from rotating to hovering", plans[0]["source_candidate_edit_text"])
            self.assertEqual("attribute", plans[0]["difference"]["type"])
            self.assertEqual("vace_controlled", plans[0]["model_route"])
            self.assertEqual("strongest_omni_prompt_planner", plans[0]["planner"]["stage"])
            self.assertFalse(plans[0]["planner"]["fallback_used"])
            self.assertEqual("existing_subject_attribute_edit", plans[0]["route_suitability"]["reason"])
            self.assertEqual("robot body", plans[0]["mask_query"])
            self.assertEqual("grounded_sam2_video_mask", plans[0]["mask_plan"])
            self.assertEqual("vace14b_masked_v2v", plans[0]["route"])
            self.assertEqual("to_be_generated", plans[0]["vace_inputs"]["src_mask"])
            self.assertTrue(plans[0]["validation_requirements"]["requires_mask"])

    def test_plan_video_edits_exploration_mode_generates_diverse_vace_families_from_one_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "robot_ref.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "robot_ref",
                        "output_path": "clips/robot_ref.mp4",
                        "summary": "a black and gold robotic action figure rotates in a dark studio with a cup beside it",
                        "subjects": ["robotic action figure", "cup", "platform"],
                        "object_counts": {"robotic action figure": 1, "cup": 1, "platform": 1},
                        "actions": ["rotating"],
                        "scene": "dark studio background",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "robot_candidate",
                        "reference_video": "clips/robot_ref.mp4",
                        "reference_caption": "a robot rotates in a studio",
                        "edit_text": "change the action from rotating to hovering",
                        "difference": {"type": "action", "from": "rotating", "to": "hovering"},
                    }
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=10,
                planning_mode="exploration",
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            families = {plan["exploration_family"] for plan in plans}
            self.assertEqual("exploration", summary["planning_mode"])
            self.assertGreaterEqual(summary["plan_count"], 5)
            self.assertIn("attribute_color", families)
            self.assertIn("attribute_material", families)
            self.assertIn("background_change", families)
            self.assertIn("style_lighting", families)
            self.assertIn("object_replacement", families)
            self.assertTrue(all(plan["model_route"] == "vace_controlled" for plan in plans))
            self.assertEqual({"clips/robot_ref.mp4"}, {plan["reference_video"] for plan in plans})

    def test_plan_video_edits_reuses_cached_omni_prompt_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "robot_ref.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "robot_ref",
                        "output_path": "clips/robot_ref.mp4",
                        "summary": "a black and gold robotic action figure rotates on a reflective platform",
                        "subjects": ["robotic action figure", "platform"],
                        "object_counts": {"robotic action figure": 1, "platform": 1},
                        "actions": ["rotating"],
                        "scene": "dark studio",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "robot_color",
                        "reference_video": "clips/robot_ref.mp4",
                        "reference_caption": "a robot rotates on a platform",
                        "edit_text": "change the robot body color from black and gold to bright yellow",
                        "difference": {
                            "type": "attribute",
                            "from": "black and gold robot body",
                            "to": "bright yellow robot body",
                        },
                    }
                ],
            )
            cache_path = root / "pairs" / "planner_cache.jsonl"
            planned_payload = {
                "should_generate": True,
                "edit_text": "change the robot body color from black and gold to bright yellow",
                "difference": {
                    "type": "attribute",
                    "from": "black and gold robot body",
                    "to": "bright yellow robot body",
                },
                "source_prompt": "A black and gold robotic action figure rotates on a platform.",
                "target_prompt": "The same robot rotates on the same platform, but the body is bright yellow.",
                "edit_token": "bright yellow robot body",
                "preserve_tokens": ["platform", "dark studio", "camera motion", "lighting"],
                "negative_prompt": "Do not change the platform, background, camera, lighting, timing, or visible text.",
                "edit_region": "robot body",
                "mask_query": "robot body",
                "model_route": "vace_controlled",
                "reason": "large existing attribute edit",
                "repaired_fields": [],
            }

            first_client = mock.Mock()
            first_client.plan_video_edit.return_value = (planned_payload, {"raw": "planner"})
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=first_client):
                first_summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    planner_cache_path=cache_path,
                )

            second_client = mock.Mock()
            second_client.plan_video_edit.side_effect = AssertionError("cache should avoid Omni call")
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=second_client):
                second_summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    planner_cache_path=cache_path,
                )

            plans = [
                json.loads(line)
                for line in Path(second_summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, first_summary["planner_cache_misses"])
            self.assertEqual(1, second_summary["planner_cache_hits"])
            second_client.plan_video_edit.assert_not_called()
            self.assertFalse(plans[0]["planner"]["fallback_used"])
            self.assertTrue(plans[0]["planner"]["cache_hit"])

    def test_plan_video_masks_builds_grounded_sam_manifest_for_vace_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "robot_ref.mp4").write_bytes(b"video")
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "robot_color",
                        "reference_video": "clips/robot_ref.mp4",
                        "edit_text": "change the robot body color from black and gold to bright yellow",
                        "difference": {"type": "attribute", "from": "black and gold robot body", "to": "bright yellow robot body"},
                        "model_route": "vace_controlled",
                        "edit_token": "bright yellow robot body",
                        "edit_region": "robot body",
                        "mask_query": "robot body",
                        "preserve_tokens": ["platform", "dark studio", "camera motion"],
                    }
                ],
            )

            summary = plan_video_masks(
                root=root,
                video_edit_plan_path=edit_plan_path,
                output_path=root / "pairs" / "video_mask_plan.jsonl",
                mask_manifest_path=root / "pairs" / "video_mask_manifest.jsonl",
            )
            mask_plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            mask_manifest = [
                json.loads(line)
                for line in Path(summary["mask_manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["mask_plan_count"])
            self.assertEqual("robot body", mask_plans[0]["mask_query"])
            self.assertEqual("edit_masked_region", mask_plans[0]["mask_mode"])
            self.assertEqual(0.02, mask_plans[0]["mask_gate"]["min_coverage_ratio"])
            self.assertEqual(0.65, mask_plans[0]["mask_gate"]["max_coverage_ratio"])
            self.assertEqual("SAM2.1_video_predictor", mask_plans[0]["toolchain"]["segmenter"])
            self.assertEqual("robot_color", mask_manifest[0]["plan_id"])
            self.assertTrue(mask_manifest[0]["mask_video"].endswith("robot_color_mask.mp4"))
            self.assertEqual(3, mask_manifest[0]["mask_semantics_version"])
            self.assertEqual("white_generate_black_preserve", mask_manifest[0]["mask_polarity"])

    def test_background_plan_masks_foreground_subject_for_inverse_background_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "man_ref.mp4").write_bytes(b"video")
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "lab_background",
                        "reference_video": "clips/man_ref.mp4",
                        "edit_text": "change the background to a futuristic laboratory",
                        "difference": {
                            "type": "scene",
                            "from": "original background",
                            "to": "futuristic laboratory background",
                        },
                        "model_route": "vace_controlled",
                        "edit_token": "futuristic laboratory background",
                        "edit_region": "background",
                        "mask_query": "background",
                        "reference_understanding": {"main_subjects": ["man with white beard"]},
                        "preserve_tokens": ["man with white beard", "camera motion"],
                    }
                ],
            )

            summary = plan_video_masks(
                root=root,
                video_edit_plan_path=edit_plan_path,
                output_path=root / "pairs" / "video_mask_plan.jsonl",
                mask_manifest_path=root / "pairs" / "video_mask_manifest.jsonl",
            )
            mask_plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual("man", mask_plans[0]["mask_query"])
            self.assertIn("man with white beard", mask_plans[0]["mask_query_candidates"])
            self.assertEqual("edit_background_inverse_subject", mask_plans[0]["mask_mode"])
            self.assertEqual(0.20, mask_plans[0]["mask_gate"]["min_coverage_ratio"])
            self.assertEqual(0.90, mask_plans[0]["mask_gate"]["max_coverage_ratio"])
            self.assertEqual(0.10, mask_plans[0]["mask_gate"]["min_detected_keyframe_box_coverage"])
            self.assertEqual(0.20, mask_plans[0]["mask_gate"]["min_background_editable_ratio"])
            self.assertEqual(0.20, mask_plans[0]["mask_gate"]["max_subject_overlap_ratio"])
            self.assertEqual(0.04, mask_plans[0]["mask_gate"]["min_foreground_subject_coverage_ratio"])
            self.assertEqual(0.70, mask_plans[0]["mask_gate"]["max_foreground_subject_coverage_ratio"])

    def test_background_family_with_garment_mask_uses_local_edit_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "robe_ref.mp4").write_bytes(b"video")
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "robe_color",
                        "reference_video": "clips/robe_ref.mp4",
                        "edit_text": "change the character's robe from red to blue",
                        "difference": {"type": "scene", "from": "red robe", "to": "blue robe"},
                        "model_route": "vace_controlled",
                        "exploration_family": "background_change",
                        "edit_token": "blue robe",
                        "edit_region": "character robe",
                        "mask_query": "robe",
                    }
                ],
            )

            summary = plan_video_masks(
                root=root,
                video_edit_plan_path=edit_plan_path,
                output_path=root / "pairs" / "video_mask_plan.jsonl",
                mask_manifest_path=root / "pairs" / "video_mask_manifest.jsonl",
            )
            mask_plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["mask_plan_count"])
            self.assertEqual("robe", mask_plans[0]["mask_query"])
            self.assertEqual("edit_masked_region", mask_plans[0]["mask_mode"])
            self.assertIn("character robe", mask_plans[0]["mask_query_candidates"])

    def test_clothing_plan_masks_clothing_even_if_query_is_person(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "shirt_ref.mp4").write_bytes(b"video")
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "shirt_color",
                        "reference_video": "clips/shirt_ref.mp4",
                        "edit_text": "change the patterned shirt to a solid black shirt",
                        "difference": {"type": "attribute", "from": "patterned shirt", "to": "solid black shirt"},
                        "model_route": "vace_controlled",
                        "exploration_family": "clothing_color",
                        "edit_token": "solid black shirt",
                        "edit_region": "shirt",
                        "mask_query": "man",
                        "preserve_tokens": ["face", "hands", "ukulele", "microphone"],
                    }
                ],
            )

            summary = plan_video_masks(
                root=root,
                video_edit_plan_path=edit_plan_path,
                output_path=root / "pairs" / "video_mask_plan.jsonl",
                mask_manifest_path=root / "pairs" / "video_mask_manifest.jsonl",
            )
            mask_plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["mask_plan_count"])
            self.assertEqual("shirt", mask_plans[0]["mask_query"])
            self.assertEqual(0.03, mask_plans[0]["mask_gate"]["min_coverage_ratio"])
            self.assertEqual(0.30, mask_plans[0]["mask_gate"]["max_coverage_ratio"])
            self.assertEqual(2, mask_plans[0]["mask_gate"]["min_protected_detections"])

    def test_plan_video_masks_skips_low_contrast_dark_clothing_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "shirt_ref.mp4").write_bytes(b"video")
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "black_to_navy_shirt",
                        "reference_video": "clips/shirt_ref.mp4",
                        "edit_text": "change the clothing color to deep navy blue",
                        "difference": {"type": "attribute", "from": "black shirt", "to": "deep navy blue shirt"},
                        "model_route": "vace_controlled",
                        "exploration_family": "clothing_color",
                        "edit_token": "shirt",
                        "edit_region": "man's shirt",
                        "mask_query": "shirt",
                    }
                ],
            )

            summary = plan_video_masks(
                root=root,
                video_edit_plan_path=edit_plan_path,
                output_path=root / "pairs" / "video_mask_plan.jsonl",
                mask_manifest_path=root / "pairs" / "video_mask_manifest.jsonl",
            )

            self.assertEqual(0, summary["mask_plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["low_contrast_dark_clothing_color_edit"])

    def test_plan_video_masks_skips_multi_subject_background_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "talk_show_ref.mp4").write_bytes(b"video")
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "background_two_speakers",
                        "reference_video": "clips/talk_show_ref.mp4",
                        "edit_text": "change the background to a futuristic laboratory",
                        "difference": {"type": "scene", "from": "original background", "to": "futuristic laboratory background"},
                        "model_route": "vace_controlled",
                        "exploration_family": "background_change",
                        "edit_token": "background",
                        "edit_region": "background",
                        "mask_query": "background",
                        "reference_understanding": {
                            "main_subjects": ["man with white beard and hat", "man with glasses"],
                            "stable_scene": "indoor studio with a green hexagonal wall and a dark room with a plant",
                        },
                    }
                ],
            )

            summary = plan_video_masks(
                root=root,
                video_edit_plan_path=edit_plan_path,
                output_path=root / "pairs" / "video_mask_plan.jsonl",
                mask_manifest_path=root / "pairs" / "video_mask_manifest.jsonl",
            )

            self.assertEqual(0, summary["mask_plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["multi_subject_background_mask_route_unsupported"])

    def test_plan_video_masks_skips_tiny_fullframe_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "cup_ref.mp4").write_bytes(b"video")
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "cup_bottle",
                        "reference_video": "clips/cup_ref.mp4",
                        "edit_text": "replace the cup with a bottle",
                        "difference": {"type": "object_presence", "from": "cup", "to": "bottle"},
                        "model_route": "vace_controlled",
                        "edit_token": "bottle",
                        "edit_region": "cup",
                        "mask_query": "cup",
                    }
                ],
            )

            summary = plan_video_masks(
                root=root,
                video_edit_plan_path=edit_plan_path,
                output_path=root / "pairs" / "video_mask_plan.jsonl",
                mask_manifest_path=root / "pairs" / "video_mask_manifest.jsonl",
            )

            self.assertEqual(0, summary["mask_plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["small_object_too_tiny_for_fullframe_vace"])

    def test_plan_src_ref_images_requires_references_for_object_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "phone_to_tablet",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "replace the phone with a tablet",
                        "difference": {"type": "object_presence", "from": "phone", "to": "tablet"},
                        "model_route": "vace_controlled",
                        "edit_token": "tablet",
                        "edit_region": "hand-held object",
                        "exploration_family": "object_replacement",
                    },
                    {
                        "plan_id": "robot_color",
                        "reference_video": "clips/robot.mp4",
                        "edit_text": "change robot body color from black and gold to bright yellow",
                        "difference": {"type": "attribute", "from": "black and gold", "to": "bright yellow"},
                        "model_route": "vace_controlled",
                        "edit_token": "bright yellow robot body",
                        "edit_region": "robot body",
                        "exploration_family": "attribute_color",
                    },
                ],
            )

            summary = plan_src_ref_images(
                root=root,
                video_edit_plan_path=edit_plan_path,
                image_root=root / "src_ref_images",
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertEqual("phone_to_tablet", plans[0]["plan_id"])
            self.assertTrue(plans[0]["required"])
            self.assertEqual("replacement_object", plans[0]["src_ref_role"])
            self.assertIn("tablet", plans[0]["image_prompts"][0])
            self.assertIn("matching the viewpoint and scale of a phone", plans[0]["image_prompts"][0])
            self.assertEqual(1, summary["skipped_reasons"]["src_ref_not_needed"])

    def test_plan_src_ref_images_uses_tabletop_prompt_for_cup_to_bottle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "cup_to_bottle",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "replace the cup with a bottle",
                        "difference": {"type": "object_presence", "from": "cup", "to": "bottle"},
                        "model_route": "vace_controlled",
                        "edit_token": "bottle",
                        "edit_region": "tabletop object",
                        "exploration_family": "object_replacement",
                    }
                ],
            )

            summary = plan_src_ref_images(
                root=root,
                video_edit_plan_path=edit_plan_path,
                image_root=root / "src_ref_images",
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertIn("small tabletop bottle", plans[0]["image_prompts"][0])
            self.assertIn("cup-sized proportion", plans[0]["image_prompts"][0])
            self.assertIn("replacing a cup on a table", plans[0]["image_prompts"][1])

    def test_plan_src_ref_images_sets_16x9_size_for_background_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "lab_background",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "change the background to a futuristic laboratory",
                        "difference": {"type": "scene", "from": "brick wall", "to": "futuristic laboratory"},
                        "model_route": "vace_controlled",
                        "edit_token": "futuristic laboratory",
                        "edit_region": "background",
                        "exploration_family": "background_change",
                    }
                ],
            )

            summary = plan_src_ref_images(
                root=root,
                video_edit_plan_path=edit_plan_path,
                image_root=root / "src_ref_images",
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertEqual("background_reference", plans[0]["src_ref_role"])
            self.assertEqual(1664, plans[0]["image_width"])
            self.assertEqual(928, plans[0]["image_height"])

    def test_plan_src_ref_images_skips_structural_black_jacket(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "black_jacket",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "change the outfit into a black jacket",
                        "difference": {"type": "attribute", "from": "patterned shirt", "to": "black jacket"},
                        "model_route": "vace_controlled",
                        "edit_token": "black jacket",
                        "edit_region": "clothing",
                        "exploration_family": "clothing_type",
                    }
                ],
            )

            summary = plan_src_ref_images(
                root=root,
                video_edit_plan_path=edit_plan_path,
                image_root=root / "src_ref_images",
            )
            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["structural_clothing_tryon_required"])
            self.assertEqual("", Path(summary["output_path"]).read_text(encoding="utf-8"))

    def test_plan_src_ref_images_keeps_safe_clothing_reference_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            edit_plan_path = root / "pairs" / "video_edit_plan.jsonl"
            self._write_jsonl(
                edit_plan_path,
                [
                    {
                        "plan_id": "solid_black_shirt",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "change the patterned shirt into a solid black shirt",
                        "difference": {"type": "attribute", "from": "patterned shirt", "to": "solid black shirt"},
                        "model_route": "vace_controlled",
                        "edit_token": "solid black shirt",
                        "edit_region": "clothing",
                        "exploration_family": "clothing_type",
                    }
                ],
            )

            summary = plan_src_ref_images(
                root=root,
                video_edit_plan_path=edit_plan_path,
                image_root=root / "src_ref_images",
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertEqual("clothing_reference", plans[0]["src_ref_role"])
            self.assertEqual("solid black shirt", plans[0]["target_object"])
            self.assertIn("cropped upper-body photo", plans[0]["image_prompts"][0])
            self.assertIn("solid black shirt", plans[0]["image_prompts"][0])
            self.assertNotIn("jacket", plans[0]["image_prompts"][0].lower())
            self.assertIn("garment silhouette clearly visible", plans[0]["image_prompts"][0])

    def test_select_src_ref_images_picks_existing_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            candidate_dir = root / "src_ref_images" / "phone_to_tablet"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "candidate_001.png").write_bytes(b"one")
            (candidate_dir / "candidate_002.png").write_bytes(b"two")
            plan_path = root / "pairs" / "src_ref_image_plan.jsonl"
            self._write_jsonl(
                plan_path,
                [
                    {
                        "plan_id": "phone_to_tablet",
                        "candidate_dir": str(candidate_dir),
                        "required": True,
                        "src_ref_role": "replacement_object",
                    }
                ],
            )

            summary = select_src_ref_images(root=root, src_ref_image_plan_path=plan_path, max_selected=1)
            records = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["selected_plan_count"])
            self.assertEqual("selected", records[0]["status"])
            self.assertEqual(1, len(records[0]["selected_src_ref_images"]))
            self.assertEqual(1, len(records[0]["rejected"]))

    def test_select_src_ref_images_caps_clothing_reference_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            candidate_dir = root / "src_ref_images" / "black_jacket"
            candidate_dir.mkdir(parents=True)
            for index in range(3):
                (candidate_dir / f"candidate_{index + 1:03d}.png").write_bytes(b"image")
            plan_path = root / "pairs" / "src_ref_image_plan.jsonl"
            self._write_jsonl(
                plan_path,
                [
                    {
                        "plan_id": "black_jacket",
                        "candidate_dir": str(candidate_dir),
                        "required": True,
                        "src_ref_role": "clothing_reference",
                        "target": "open black long-sleeved jacket over a black T-shirt",
                    }
                ],
            )

            summary = select_src_ref_images(root=root, src_ref_image_plan_path=plan_path, max_selected=3)
            records = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["selected_plan_count"])
            self.assertEqual(1, len(records[0]["selected_src_ref_images"]))
            self.assertEqual(2, len(records[0]["rejected"]))

    def test_select_src_ref_images_uses_omni_audit_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            candidate_dir = root / "src_ref_images" / "phone_to_tablet"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "candidate_001.png").write_bytes(b"one")
            (candidate_dir / "candidate_002.png").write_bytes(b"two")
            plan_path = root / "pairs" / "src_ref_image_plan.jsonl"
            self._write_jsonl(
                plan_path,
                [
                    {
                        "plan_id": "phone_to_tablet",
                        "candidate_dir": str(candidate_dir),
                        "required": True,
                        "src_ref_role": "replacement_object",
                    }
                ],
            )

            class FakeClient:
                def __init__(self, **kwargs: object) -> None:
                    self.kwargs = kwargs

                def audit_src_ref_images(
                    self,
                    *,
                    src_ref_plan: dict,
                    candidate_image_paths: list[str],
                    max_selected: int,
                ) -> tuple[dict, dict]:
                    self.src_ref_plan = src_ref_plan
                    self.candidate_image_paths = candidate_image_paths
                    self.max_selected = max_selected
                    return (
                        {
                            "selected_indices": [2],
                            "audit": [{"index": 2, "verdict": "select", "reason": "better angle"}],
                            "rejected": [{"index": 1, "reason": "flat view"}],
                            "reason": "candidate 2 best matches the replacement object",
                        },
                        {"raw": True},
                    )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", FakeClient):
                summary = select_src_ref_images(
                    root=root,
                    src_ref_image_plan_path=plan_path,
                    max_selected=1,
                    base_url="http://127.0.0.1:8093/v1",
                    model="qwen3-omni",
                )
            records = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["selected_plan_count"])
            self.assertEqual("omni_src_ref_image_audit", records[0]["selection_method"])
            self.assertTrue(records[0]["selected_src_ref_images"][0].endswith("candidate_002.png"))
            self.assertEqual("candidate 2 best matches the replacement object", records[0]["selection_reason"])
            self.assertIn("omni_audit", records[0])

    def test_select_src_ref_images_rejects_square_background_reference(self) -> None:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover - optional local dependency
            self.skipTest(f"Pillow unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            candidate_dir = root / "src_ref_images" / "lab_background"
            candidate_dir.mkdir(parents=True)
            Image.new("RGB", (1024, 1024), color=(20, 30, 40)).save(candidate_dir / "candidate_001.png")
            plan_path = root / "pairs" / "src_ref_image_plan.jsonl"
            self._write_jsonl(
                plan_path,
                [
                    {
                        "plan_id": "lab_background",
                        "candidate_dir": str(candidate_dir),
                        "required": True,
                        "src_ref_role": "background_reference",
                    }
                ],
            )

            summary = select_src_ref_images(root=root, src_ref_image_plan_path=plan_path, max_selected=1)
            records = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(0, summary["selected_plan_count"])
            self.assertEqual(1, summary["audit_rejected_count"])
            self.assertEqual("rejected_by_deterministic_audit", records[0]["status"])
            self.assertIn("background_src_ref_not_16x9", records[0]["candidate_audit"][0]["hard_reject_reasons"])

    def test_src_ref_background_candidate_audit_accepts_16x9_plate(self) -> None:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover - optional local dependency
            self.skipTest(f"Pillow unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate_001.png"
            Image.new("RGB", (1280, 720), color=(20, 30, 40)).save(candidate)

            audit = _audit_src_ref_image_candidate(candidate, {"src_ref_role": "background_reference"})

            self.assertTrue(audit["eligible"])
            self.assertEqual([], audit["hard_reject_reasons"])
            self.assertIn("background candidate is close to 16:9", audit["reasons"])

    def test_src_ref_clothing_candidate_audit_rejects_empty_product_jacket(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "empty_jacket_product_catalog.png"
            candidate.write_bytes(b"not-an-image")

            audit = _audit_src_ref_image_candidate(candidate, {"src_ref_role": "clothing_reference"})

            self.assertFalse(audit["eligible"])
            self.assertIn("clothing_src_ref_product_or_empty_jacket_artifact", audit["hard_reject_reasons"])

    def test_plan_video_edits_rejects_tiny_additive_attribute_revision_for_vace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "ref_visual.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_visual",
                        "output_path": "clips/ref_visual.mp4",
                        "summary": "a woman speaks to camera in a room",
                        "subjects": ["woman", "room"],
                        "object_counts": {"woman": 1},
                        "actions": ["speaking"],
                        "scene": "room",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "visual_1",
                        "reference_video": "clips/ref_visual.mp4",
                        "reference_caption": "a woman speaks to camera",
                        "edit_text": "change the attribute from no nose ring to nose ring",
                        "difference": {"type": "attribute", "from": "no nose ring", "to": "nose ring"},
                    }
                ],
            )

            fake_client = mock.Mock()
            fake_client.plan_video_edit.return_value = (
                {
                    "should_generate": True,
                    "edit_text": "add a nose ring to the woman",
                    "difference": {"type": "attribute", "from": "no nose ring", "to": "nose ring"},
                    "source_prompt": "A woman speaks to camera in the same room.",
                    "target_prompt": "The same woman speaks to camera in the same room with a small nose ring added.",
                    "edit_token": "nose ring",
                    "preserve_tokens": ["woman", "room", "camera motion", "lighting"],
                    "negative_prompt": "Do not change the woman, room, camera motion, lighting, timing, or visible text.",
                    "edit_region": "face, nose area",
                    "model_route": "tokenflow_style",
                    "reason": "A nose ring is a localized accessory addition.",
                    "repaired_fields": [],
                },
                {"raw": "planner"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=fake_client):
                summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([], plans)
            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["vace_rejects_tiny_or_naked_object_edit"])

    def test_plan_video_edits_ideates_safe_visual_edit_from_audio_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "ref_audio.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_audio",
                        "output_path": "clips/ref_audio.mp4",
                        "summary": "a black and gold robotic action figure rotates on a reflective platform in a dark studio",
                        "subjects": ["robotic action figure", "platform"],
                        "object_counts": {"robotic action figure": 1, "platform": 1},
                        "actions": ["rotating", "turning"],
                        "scene": "dark studio",
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "audio_1",
                        "reference_video": "clips/ref_audio.mp4",
                        "reference_caption": "a hand writes on paper",
                        "edit_text": "add whoosh to the audio",
                        "difference": {"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
                    }
                ],
            )

            fake_client = mock.Mock()
            fake_client.plan_video_edit.return_value = (
                {
                    "should_generate": True,
                    "edit_text": "change the robot body color from black and gold to bright yellow",
                    "difference": {
                        "type": "attribute",
                        "from": "black and gold robot body",
                        "to": "bright yellow robot body",
                    },
                    "source_prompt": "A black and gold robotic action figure rotates on a reflective platform in a dark studio.",
                    "target_prompt": "The same robotic action figure rotates on the same platform in the same dark studio, but the robot body is bright yellow.",
                    "edit_token": "bright yellow robot body",
                    "preserve_tokens": ["yellow visor", "platform", "dark studio", "rotation", "camera motion", "lighting"],
                    "negative_prompt": "Do not change the platform, camera motion, lighting, timing, background, or visible text.",
                    "edit_region": "robot body",
                    "model_route": "vace_controlled",
                    "reason": "The robot body color is a large existing attribute suitable for VACE.",
                    "repaired_fields": [],
                },
                {"raw": "planner"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=fake_client):
                summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, summary["plan_count"])
            self.assertEqual("add whoosh to the audio", plans[0]["source_candidate_edit_text"])
            self.assertEqual("change the robot body color from black and gold to bright yellow", plans[0]["edit_text"])
            self.assertEqual("attribute", plans[0]["difference"]["type"])
            self.assertEqual("vace_controlled", plans[0]["model_route"])
            self.assertTrue(plans[0]["visual_edit_risk"]["safe_visual_ideation_relaxed"])
            self.assertIn("multiple_actions", plans[0]["visual_edit_risk"]["relaxed_risk_reasons"])
            self.assertEqual(1, summary["skipped_reasons"]["safe_visual_ideation_from_unsupported_type"])
            call_kwargs = fake_client.plan_video_edit.call_args.kwargs
            self.assertEqual("change the robot body color from black and gold to bright yellow", call_kwargs["candidate"]["edit_text"])

    def test_plan_video_edits_keeps_visible_text_hard_risk_for_safe_ideation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            (root / "clips" / "ref_audio.mp4").write_bytes(b"video")
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref_audio",
                        "output_path": "clips/ref_audio.mp4",
                        "summary": "a black and gold robotic action figure rotates on a platform with visible chemical text",
                        "subjects": ["robotic action figure", "platform"],
                        "object_counts": {"robotic action figure": 1, "platform": 1},
                        "actions": ["rotating"],
                        "scene": "dark studio",
                        "visible_text": ["chemical formula"],
                        "on_screen_text": ["chemical formula"],
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "audio_1",
                        "reference_video": "clips/ref_audio.mp4",
                        "reference_caption": "a hand writes on paper",
                        "edit_text": "add whoosh to the audio",
                        "difference": {"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
                    }
                ],
            )

            fake_client = mock.Mock()
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=fake_client):
                summary = plan_video_edits(
                    root=root,
                    pair_candidates_path=candidates_path,
                    clip_annotations_path=annotations_path,
                    max_plans=5,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["safe_visual_ideation_from_unsupported_type"])
            self.assertEqual(1, summary["skipped_reasons"]["risk_visible_text_present"])
            fake_client.plan_video_edit.assert_not_called()

    def test_plan_video_edits_skips_high_risk_visible_text_and_motion_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "text_ref",
                        "output_path": "clips/text_ref.mp4",
                        "summary": "a person speaks while a lower-third caption is visible",
                        "subjects": ["person", "caption"],
                        "object_counts": {"person": 1},
                        "actions": ["speaking", "gesturing"],
                        "scene": "indoor room",
                        "visible_text": ["MAKE WRONG"],
                        "on_screen_text": ["MAKE WRONG"],
                    },
                    {
                        "clip_id": "motion_ref",
                        "output_path": "clips/motion_ref.mp4",
                        "summary": "a person runs, jumps, and then waves",
                        "subjects": ["person"],
                        "object_counts": {"person": 1},
                        "actions": ["running", "jumping", "waving"],
                        "scene": "outdoor track",
                    },
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "text_risky",
                        "reference_video": "clips/text_ref.mp4",
                        "edit_text": "add a pen-like device",
                        "difference": {"type": "object_presence", "from": "no pen-like device", "to": "pen-like device"},
                    },
                    {
                        "proposal_id": "motion_risky",
                        "reference_video": "clips/motion_ref.mp4",
                        "edit_text": "add a small backpack",
                        "difference": {"type": "object_presence", "from": "no backpack", "to": "backpack"},
                    },
                ],
            )

            summary = plan_video_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )

            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["risk_visible_text_present"])
            self.assertGreaterEqual(summary["skipped_reasons"]["risk_multiple_actions"], 1)

    def test_video_edit_risk_adds_text_and_motion_locks(self) -> None:
        risk = _video_edit_risk_assessment(
            {
                "summary": "a person speaks to camera",
                "subjects": ["person"],
                "actions": ["speaking", "gesturing"],
                "visible_text": ["hello"],
            },
            difference_type="object_presence",
        )

        self.assertFalse(risk["allow_generation"])
        self.assertIn("visible_text_present", risk["risk_reasons"])
        self.assertTrue(any("visible text" in lock for lock in risk["locks"]))
        self.assertTrue(any("motion timing" in lock for lock in risk["locks"]))

    def test_plan_audio_edits_only_allows_non_speech_audio_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref",
                        "output_path": "clips/ref.mp4",
                        "summary": "a person jumps across a small platform",
                        "actions": ["jumping"],
                        "audio_events": [],
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "audio_ok",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "add whoosh to the audio",
                        "difference": {"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
                    },
                    {
                        "proposal_id": "speech_bad",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "change the speech topic to marketing",
                        "difference": {"type": "speech", "from": "sports", "to": "marketing"},
                    },
                    {
                        "proposal_id": "audio_speech_bad",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "change the audio to only speech",
                        "difference": {"type": "audio_event", "from": "no distinctive audio event", "to": "only speech"},
                    },
                ],
            )

            summary = plan_audio_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertEqual("audio_ok", plans[0]["plan_id"])
            self.assertEqual("strongest_omni_audio_prompt_planner", plans[0]["planner"]["stage"])
            self.assertEqual("short_clip_reference_video_and_audio_understanding", plans[0]["planner"]["input"])
            self.assertEqual("whoosh", plans[0]["audio_edit_plan"]["expected_event"])
            self.assertEqual("visual_sync", plans[0]["audio_edit_plan"]["timing_strategy"])
            self.assertEqual("contextual_non_speech_audio_edit", plans[0]["route_suitability"]["reason"])
            self.assertEqual("S", plans[0]["route_suitability"]["priority"])
            self.assertEqual("whoosh", plans[0]["audio_reference_understanding"]["suggested_non_speech_audio_events"][0]["expected_event"])
            self.assertEqual(1, summary["skipped_by_type"]["speech"])
            self.assertEqual(2, summary["skipped_reasons"]["speech_content_or_speech_only_audio"])

    def test_plan_audio_edits_ideates_contextual_sound_from_action_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref",
                        "output_path": "clips/ref.mp4",
                        "summary": "a character is launched from a cliff and glides through the air",
                        "subjects": ["character", "cliff"],
                        "actions": ["launched", "gliding"],
                        "audio_events": [],
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "action_hint",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "change the action from running to launched",
                        "difference": {"type": "action", "from": "running", "to": "launched"},
                    }
                ],
            )

            summary = plan_audio_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(1, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["safe_audio_ideation_from_non_audio_candidate"])
            self.assertEqual("change the action from running to launched", plans[0]["source_candidate_edit_text"])
            self.assertEqual("audio_event", plans[0]["difference"]["type"])
            self.assertEqual("whoosh", plans[0]["audio_edit_plan"]["expected_event"])
            self.assertEqual("foleycrafter_temporal", plans[0]["audio_edit_plan"]["route"])
            self.assertEqual("visual_sync", plans[0]["route_suitability"]["timing_strategy"])
            self.assertEqual("S", plans[0]["route_suitability"]["priority"])

    def test_plan_audio_edits_rejects_event_already_present_in_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "captions" / "annotations.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "ref",
                        "output_path": "clips/ref.mp4",
                        "summary": "a person jumps with a whoosh sound",
                        "actions": ["jumping"],
                        "audio_events": ["whoosh"],
                    }
                ],
            )
            candidates_path = root / "pairs" / "candidates.jsonl"
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "proposal_id": "audio_dup",
                        "reference_video": "clips/ref.mp4",
                        "edit_text": "add whoosh to the audio",
                        "difference": {"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
                    }
                ],
            )

            summary = plan_audio_edits(
                root=root,
                pair_candidates_path=candidates_path,
                clip_annotations_path=annotations_path,
                max_plans=5,
            )
            plans = [
                json.loads(line)
                for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual([], plans)
            self.assertEqual(0, summary["plan_count"])
            self.assertEqual(1, summary["skipped_reasons"]["reference_already_has_expected_audio_event"])

    def test_build_manual_review_bundle_copies_videos_and_writes_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "clips").mkdir(parents=True)
            (root / "synthetic").mkdir(parents=True)
            (root / "src_ref").mkdir(parents=True)
            (root / "review_inputs_src").mkdir(parents=True)
            (root / "clips" / "ref.mp4").write_bytes(b"reference-video")
            (root / "synthetic" / "target.mp4").write_bytes(b"target-video")
            (root / "synthetic" / "src_video_for_vace.mp4").write_bytes(b"src-video")
            (root / "synthetic" / "mask.mp4").write_bytes(b"mask-video")
            (root / "synthetic" / "raw.mp4").write_bytes(b"raw-video")
            (root / "src_ref" / "jacket.png").write_bytes(b"src-ref")
            (root / "review_inputs_src" / "vace_prompt.txt").write_text("same shot target prompt", encoding="utf-8")
            (root / "review_inputs_src" / "preflight_report.json").write_text("{}", encoding="utf-8")
            (root / "review_inputs_src" / "duration_metrics.json").write_text("{}", encoding="utf-8")
            (root / "review_inputs_src" / "vace_command.json").write_text("{}", encoding="utf-8")
            (root / "review_inputs_src" / "reference_contact.jpg").write_bytes(b"reference-contact")
            (root / "review_inputs_src" / "mask_contact.jpg").write_bytes(b"mask-contact")
            (root / "review_inputs_src" / "src_video_contact.jpg").write_bytes(b"src-video-contact")
            (root / "review_inputs_src" / "raw_target_contact.jpg").write_bytes(b"raw-contact")
            (root / "review_inputs_src" / "target_contact.jpg").write_bytes(b"target-contact")
            pairs_path = root / "accepted.jsonl"
            self._write_jsonl(
                pairs_path,
                [
                    {
                        "sample_id": "sample_1",
                        "proposal_id": "proposal__abc",
                        "reference_video": "clips/ref.mp4",
                        "target_video": "synthetic/target.mp4",
                        "edit_text": "change the robot body color from black and gold to bright yellow",
                        "difference": {"type": "attribute", "from": "black and gold", "to": "bright yellow"},
                        "reference_caption": "A black and gold robot rotates on a platform.",
                        "target_caption": "A bright yellow robot rotates on the same platform.",
                        "verification": {"passed": True},
                        "observable_difference": {"passed": True},
                        "competing_difference": {"passed": True},
                        "generation": {
                            "model_route": "vace_controlled",
                            "src_video_for_vace": "synthetic/src_video_for_vace.mp4",
                            "src_mask": "synthetic/mask.mp4",
                            "src_ref_images": ["src_ref/jacket.png"],
                            "review_inputs_dir": "review_inputs_src",
                            "mask_metrics": {"mask_coverage_ratio_avg": 0.12},
                            "duration_metrics": {"duration_gate": {"passed": True}},
                            "vace_command": {"argv": ["python", "generate.py"]},
                            "post_vace_verdict": {"duration_gate_passed": True},
                            "postprocess": {
                                "audio_copied_from_reference": True,
                                "raw_generated_video": "synthetic/raw.mp4",
                            },
                        },
                    }
                ],
            )
            output_dir = root / "manual_review"

            summary = build_manual_review_bundle(
                root=root,
                pairs_path=pairs_path,
                output_dir=output_dir,
            )

            self.assertEqual(1, summary["bundle_count"])
            item_dir = output_dir / "0001_sample_1"
            self.assertTrue((item_dir / "reference.mp4").exists())
            self.assertTrue((item_dir / "target.mp4").exists())
            review_text = (item_dir / "review.md").read_text(encoding="utf-8")
            self.assertIn("change the robot body color", review_text)
            self.assertIn("A black and gold robot", review_text)
            self.assertIn("A bright yellow robot", review_text)
            self.assertIn("src_video_for_vace", review_text)
            self.assertIn("duration_metrics", review_text)
            self.assertTrue((item_dir / "src_video_for_vace.mp4").exists())
            self.assertTrue((item_dir / "mask.mp4").exists())
            self.assertTrue((item_dir / "raw_target.mp4").exists())
            self.assertTrue((item_dir / "src_ref_images" / "001_jacket_png").exists())
            self.assertTrue((item_dir / "review_inputs" / "vace_prompt.txt").exists())
            self.assertTrue((item_dir / "metadata.json").exists())
            self.assertTrue((item_dir / "semantic_evaluation_result.json").exists())
            metadata = json.loads((item_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["incomplete_review_bundle"])
            self.assertEqual([], metadata["review_bundle_issues"])
            self.assertIn("sample_1", (output_dir / "index.md").read_text(encoding="utf-8"))
            self.assertIn("complete", (output_dir / "index.md").read_text(encoding="utf-8"))

    def test_build_manual_review_bundle_marks_incomplete_visual_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "clips").mkdir(parents=True)
            (root / "synthetic").mkdir(parents=True)
            (root / "src_ref").mkdir(parents=True)
            (root / "clips" / "ref.mp4").write_bytes(b"reference-video")
            (root / "synthetic" / "target.mp4").write_bytes(b"target-video")
            (root / "synthetic" / "raw.mp4").write_bytes(b"raw-video")
            (root / "src_ref" / "jacket.png").write_bytes(b"src-ref")
            pairs_path = root / "accepted.jsonl"
            self._write_jsonl(
                pairs_path,
                [
                    {
                        "sample_id": "sample_incomplete",
                        "proposal_id": "proposal__missing_inputs",
                        "reference_video": "clips/ref.mp4",
                        "target_video": "synthetic/target.mp4",
                        "edit_text": "replace the cup with a bottle",
                        "difference": {"type": "object_presence", "from": "cup", "to": "bottle"},
                        "generation": {
                            "model_route": "vace_controlled",
                            "src_ref_requirements": {"required": True},
                            "src_ref_images": ["src_ref/jacket.png"],
                            "duration_metrics": {"duration_gate": {"passed": True}},
                            "post_vace_verdict": {"semantic_gate_required": True, "semantic_gate_passed": True},
                            "postprocess": {
                                "audio_copied_from_reference": True,
                                "raw_generated_video": "synthetic/raw.mp4",
                            },
                        },
                    }
                ],
            )

            summary = build_manual_review_bundle(
                root=root,
                pairs_path=pairs_path,
                output_dir=root / "manual_review",
            )

            self.assertEqual(1, summary["incomplete_review_bundle_count"])
            item_dir = root / "manual_review" / "0001_sample_incomplete"
            metadata = json.loads((item_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["incomplete_review_bundle"])
            self.assertTrue(any("missing src_mask" in issue for issue in metadata["review_bundle_issues"]))
            self.assertTrue(any("missing src_video_for_vace" in issue for issue in metadata["review_bundle_issues"]))
            self.assertTrue(any("missing review_inputs_dir" in issue for issue in metadata["review_bundle_issues"]))

    def test_pair_record_acceptance_issues_rejects_speech_difference_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"ref")
            (root / "target.mp4").write_bytes(b"target")
            issues = _pair_record_acceptance_issues(
                root=root,
                record={
                    "reference_video": "ref.mp4",
                    "target_video": "target.mp4",
                    "edit_text": "change the speech topic to marketing",
                    "difference": {"type": "speech", "from": "sales", "to": "marketing"},
                },
                reference_annotation={"speech": ["sales"]},
                target_annotation={"speech": ["marketing"]},
            )

            self.assertTrue(any("speech difference type is disabled" in issue for issue in issues))

    def test_synthetic_audio_rejects_visual_drift_and_missing_target_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"ref")
            (root / "target.mp4").write_bytes(b"target")
            with mock.patch(
                "app.composed_data.probe_media",
                return_value={"duration_seconds": 8.0, "has_audio": True, "has_video": True},
            ):
                issues = _pair_record_acceptance_issues(
                    root=root,
                    record={
                        "source_type": "synthetic_edit",
                        "reference_video": "ref.mp4",
                        "target_video": "target.mp4",
                        "edit_text": "add whoosh to the audio",
                        "difference": {"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
                        "quality": {"visual_near_duplicate_score": 0.80},
                        "source_context": {"relation": "synthetic_from_reference"},
                        "generation": {
                            "model_route": "deterministic_overlay",
                            "audio_edit_plan": {"route": "deterministic_overlay", "expected_event": "whoosh"},
                        },
                    },
                    reference_annotation={"audio_events": []},
                    target_annotation={"audio_events": ["quiet room"]},
                )

            self.assertTrue(any("audio synthetic target changed visual stream" in issue for issue in issues))
            self.assertTrue(any("target sound was not detected" in issue for issue in issues))

    def test_synthetic_visual_requires_reference_audio_remux_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"ref")
            (root / "target.mp4").write_bytes(b"target")
            with mock.patch(
                "app.composed_data.probe_media",
                return_value={"duration_seconds": 8.0, "has_audio": True, "has_video": True},
            ):
                issues = _pair_record_acceptance_issues(
                    root=root,
                    record={
                        "source_type": "synthetic_edit",
                        "reference_video": "ref.mp4",
                        "target_video": "target.mp4",
                        "edit_text": "add a red backpack",
                        "difference": {"type": "object_presence", "from": "no red backpack", "to": "red backpack"},
                        "quality": {"visual_near_duplicate_score": 0.90},
                        "source_context": {"relation": "synthetic_from_reference"},
                        "generation": {"model_route": "vace_controlled", "postprocess": {}},
                    },
                    reference_annotation={"object_counts": {"chair": 1}},
                    target_annotation={"object_counts": {"chair": 1, "red backpack": 1}},
                )

            self.assertTrue(any("audio_copied_from_reference=true" in issue for issue in issues))

    def test_synthetic_visual_rejects_missing_required_src_ref_and_duration_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"ref")
            (root / "target.mp4").write_bytes(b"target")
            with mock.patch(
                "app.composed_data.probe_media",
                return_value={"duration_seconds": 5.0, "has_audio": True, "has_video": True},
            ):
                issues = _pair_record_acceptance_issues(
                    root=root,
                    record={
                        "source_type": "synthetic_edit",
                        "reference_video": "ref.mp4",
                        "target_video": "target.mp4",
                        "edit_text": "replace the cup with a bottle",
                        "difference": {"type": "object_presence", "from": "cup", "to": "bottle"},
                        "quality": {"visual_near_duplicate_score": 0.90},
                        "source_context": {"relation": "synthetic_from_reference"},
                        "generation": {
                            "model_route": "vace_controlled",
                            "src_ref_requirements": {"required": True},
                            "src_ref_images": [],
                            "duration_metrics": {
                                "raw_duration_drift_seconds": 0.75,
                                "target_duration_drift_seconds": 0.8,
                                "max_duration_drift_seconds": 0.5,
                                "duration_gate": {"passed": False, "errors": ["target_duration_drift_seconds 0.800 > 0.500"]},
                            },
                            "postprocess": {"audio_copied_from_reference": True},
                        },
                    },
                    reference_annotation={"object_counts": {"cup": 1}},
                    target_annotation={"object_counts": {"bottle": 1}},
                )

            self.assertTrue(any("missing required src_ref_images" in issue for issue in issues))
            self.assertTrue(any("duration gate failed" in issue for issue in issues))
            self.assertTrue(any("raw_duration_drift_seconds" in issue for issue in issues))

    def test_synthetic_visual_requires_duration_gate_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"ref")
            (root / "target.mp4").write_bytes(b"target")
            with mock.patch(
                "app.composed_data.probe_media",
                return_value={"duration_seconds": 5.0, "has_audio": True, "has_video": True},
            ):
                issues = _pair_record_acceptance_issues(
                    root=root,
                    record={
                        "source_type": "synthetic_edit",
                        "reference_video": "ref.mp4",
                        "target_video": "target.mp4",
                        "edit_text": "change the robot body color to yellow",
                        "difference": {"type": "attribute", "from": "black robot", "to": "yellow robot"},
                        "quality": {"visual_near_duplicate_score": 0.90},
                        "source_context": {"relation": "synthetic_from_reference"},
                        "generation": {
                            "model": "wan-vace",
                            "source_video": "ref.mp4",
                            "model_route": "vace_controlled",
                            "prompt": "A yellow robot rotates.",
                            "source_prompt": "A black robot rotates.",
                            "target_prompt": "A yellow robot rotates.",
                            "preserve_tokens": ["robot", "rotation", "camera framing"],
                            "src_video_for_vace": "ref.mp4",
                            "src_mask": "target.mp4",
                            "mask_semantics_version": 3,
                            "mask_polarity": "white_generate_black_preserve",
                            "mask_metrics": {"mask_coverage_ratio_avg": 0.2},
                            "review_inputs_dir": ".",
                            "duration_metrics": {"duration_drift_seconds": 11.96},
                            "post_vace_verdict": {"semantic_gate_required": True, "semantic_gate_passed": True},
                            "postprocess": {"audio_copied_from_reference": True, "raw_generated_video": "target.mp4"},
                        },
                    },
                    reference_annotation={"object_counts": {"robot": 1}},
                    target_annotation={"object_counts": {"robot": 1}},
                )

            self.assertTrue(any("duration gate is required" in issue for issue in issues))

    def test_synthetic_visual_rejects_unpassed_post_vace_semantic_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"ref")
            (root / "target.mp4").write_bytes(b"target")
            with mock.patch(
                "app.composed_data.probe_media",
                return_value={"duration_seconds": 5.0, "has_audio": True, "has_video": True},
            ):
                issues = _pair_record_acceptance_issues(
                    root=root,
                    record={
                        "source_type": "synthetic_edit",
                        "reference_video": "ref.mp4",
                        "target_video": "target.mp4",
                        "edit_text": "change the outfit into a black jacket",
                        "difference": {"type": "attribute", "from": "patterned shirt", "to": "black jacket"},
                        "quality": {"visual_near_duplicate_score": 0.90},
                        "source_context": {"relation": "synthetic_from_reference"},
                        "generation": {
                            "model_route": "vace_controlled",
                            "post_vace_verdict": {
                                "semantic_gate_required": True,
                                "semantic_gate_passed": False,
                            },
                            "postprocess": {"audio_copied_from_reference": True},
                        },
                    },
                    reference_annotation={"object_counts": {"man": 1}},
                    target_annotation={"object_counts": {"man": 1}},
                )

            self.assertTrue(any("post-VACE semantic gate" in issue for issue in issues))

    def test_synthetic_visual_rejects_structural_clothing_vace_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"ref")
            (root / "target.mp4").write_bytes(b"target")
            with mock.patch(
                "app.composed_data.probe_media",
                return_value={"duration_seconds": 5.0, "has_audio": True, "has_video": True},
            ):
                issues = _pair_record_acceptance_issues(
                    root=root,
                    record={
                        "source_type": "synthetic_edit",
                        "reference_video": "ref.mp4",
                        "target_video": "target.mp4",
                        "edit_text": "change the outfit into a black jacket",
                        "difference": {"type": "attribute", "from": "patterned shirt", "to": "black jacket"},
                        "quality": {"visual_near_duplicate_score": 0.90},
                        "source_context": {"relation": "synthetic_from_reference"},
                        "generation": {
                            "model": "wan-vace",
                            "model_route": "vace_controlled",
                            "prompt": "same shot",
                            "source_prompt": "A man in a patterned shirt plays ukulele.",
                            "target_prompt": "A man in an open black long-sleeved jacket plays ukulele.",
                            "preserve_tokens": ["man", "ukulele"],
                            "src_video_for_vace": "ref.mp4",
                            "src_mask": "target.mp4",
                            "mask_metrics": {"mask_coverage_ratio_avg": 0.1},
                            "review_inputs_dir": ".",
                            "duration_metrics": {"duration_gate": {"passed": True}},
                            "post_vace_verdict": {"semantic_gate_required": True, "semantic_gate_passed": True},
                            "postprocess": {"audio_copied_from_reference": True, "raw_generated_video": "target.mp4"},
                        },
                    },
                    reference_annotation={"object_counts": {"man": 1}},
                    target_annotation={"object_counts": {"man": 1}},
                )
                issues.extend(
                    _known_pair_generation_issues(
                        {
                            "source_type": "synthetic_edit",
                            "edit_text": "change the outfit into a black jacket",
                            "difference": {"type": "attribute", "from": "patterned shirt", "to": "black jacket"},
                            "generation": {
                                "model": "wan-vace",
                                "model_route": "vace_controlled",
                                "prompt": "same shot",
                                "source_prompt": "A man in a patterned shirt plays ukulele.",
                                "target_prompt": "A man in an open black long-sleeved jacket plays ukulele.",
                                "preserve_tokens": ["man", "ukulele"],
                                "src_video_for_vace": "ref.mp4",
                                "src_mask": "target.mp4",
                                "mask_metrics": {"mask_coverage_ratio_avg": 0.1},
                                "review_inputs_dir": ".",
                                "duration_metrics": {"duration_gate": {"passed": True}},
                                "post_vace_verdict": {"semantic_gate_required": True, "semantic_gate_passed": True},
                                "postprocess": {"audio_copied_from_reference": True, "raw_generated_video": "target.mp4"},
                            },
                        }
                    )
                )

            self.assertTrue(any("try-on route" in issue for issue in issues))

    def test_synthetic_visual_rejects_plain_background_replacement_vace_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("ref.mp4", "target.mp4", "src_video.mp4", "src_mask.mp4"):
                (root / name).write_bytes(name.encode("utf-8"))

            record = {
                "source_type": "synthetic_edit",
                "reference_video": "ref.mp4",
                "target_video": "target.mp4",
                "edit_text": "change the background to a futuristic laboratory",
                "difference": {"type": "scene", "from": "sunlit room background", "to": "futuristic laboratory background"},
                "quality": {"visual_near_duplicate_score": 0.90},
                "source_context": {"relation": "synthetic_from_reference"},
                "generation": {
                    "model": "wan-vace",
                    "source_video": "ref.mp4",
                    "model_route": "vace_controlled",
                    "prompt": "A woman speaks in a futuristic laboratory.",
                    "source_prompt": "A woman speaks in a sunlit room.",
                    "target_prompt": "A woman speaks in a clean futuristic laboratory interior.",
                    "edit_region": "background",
                    "mask_query": "woman",
                    "preserve_tokens": ["woman", "face", "camera framing"],
                    "src_video_for_vace": "src_video.mp4",
                    "src_mask": "src_mask.mp4",
                    "mask_semantics_version": 3,
                    "mask_polarity": "white_generate_black_preserve",
                    "mask_metrics": {"mask_coverage_ratio_avg": 0.55},
                    "review_inputs_dir": ".",
                    "duration_metrics": {"duration_gate": {"passed": True}},
                    "post_vace_verdict": {"semantic_gate_required": True, "semantic_gate_passed": True},
                    "postprocess": {"audio_copied_from_reference": True, "raw_generated_video": "target.mp4"},
                },
            }
            with mock.patch(
                "app.composed_data.probe_media",
                return_value={"duration_seconds": 5.0, "has_audio": True, "has_video": True},
            ):
                issues = _pair_record_acceptance_issues(
                    root=root,
                    record=record,
                    reference_annotation={"object_counts": {"woman": 1}},
                    target_annotation={"object_counts": {"woman": 1}},
                )

            self.assertTrue(any("plain masked VACE is experiment-only" in issue for issue in issues))

            composite_record = copy.deepcopy(record)
            composite_record["generation"]["background_replace_route"] = "vace_bg_replace_composite_first_frame_mv2v"
            with mock.patch(
                "app.composed_data.probe_media",
                return_value={"duration_seconds": 5.0, "has_audio": True, "has_video": True},
            ):
                composite_issues = _pair_record_acceptance_issues(
                    root=root,
                    record=composite_record,
                    reference_annotation={"object_counts": {"woman": 1}},
                    target_annotation={"object_counts": {"woman": 1}},
                )

            self.assertFalse(any("plain masked VACE is experiment-only" in issue for issue in composite_issues))

    def test_post_vace_semantic_verdict_passes_strict_black_jacket_annotation(self) -> None:
        record = {
            "edit_text": "change the outfit into a black jacket",
            "target_caption": "A man wearing an open black long-sleeved jacket plays ukulele into a microphone.",
            "difference": {"type": "attribute", "from": "patterned shirt", "to": "black jacket"},
            "generation": {
                "post_vace_verdict": {
                    "semantic_gate_required": True,
                    "semantic_gate_passed": False,
                }
            },
            "verification": {
                "edit_projection": {
                    "reason": "The target shows an open black long-sleeved jacket and preserves the ukulele and microphone."
                }
            },
        }

        updated = _apply_post_vace_semantic_verdict(
            record,
            target_annotation={
                "summary": "A man in a blue fedora wears an open black jacket with long sleeves while playing ukulele.",
                "subjects": ["man"],
                "actions": ["playing ukulele", "singing"],
                "object_counts": {"ukulele": 1, "microphone": 1},
            },
        )

        verdict = updated["generation"]["post_vace_verdict"]
        self.assertTrue(verdict["semantic_gate_passed"])
        self.assertEqual("passed_semantic_gate", verdict["stage"])

    def test_post_vace_semantic_verdict_rejects_dark_shirt_as_black_jacket(self) -> None:
        record = {
            "edit_text": "change the outfit into a black jacket",
            "target_caption": "A man in a dark shirt plays a ukulele near a microphone.",
            "difference": {"type": "attribute", "from": "patterned shirt", "to": "black jacket"},
            "generation": {
                "post_vace_verdict": {
                    "semantic_gate_required": True,
                    "semantic_gate_passed": False,
                }
            },
        }

        updated = _apply_post_vace_semantic_verdict(
            record,
            target_annotation={
                "summary": "The man wears a dark shirt while playing ukulele.",
                "subjects": ["man"],
                "actions": ["playing ukulele"],
                "object_counts": {"ukulele": 1, "microphone": 1},
            },
        )

        verdict = updated["generation"]["post_vace_verdict"]
        self.assertFalse(verdict["semantic_gate_passed"])
        self.assertEqual("failed_semantic_gate", verdict["stage"])
        self.assertIn("target_annotation_missing_black_jacket", verdict["semantic_gate_errors"])
        self.assertIn("target_annotation_forbidden_marker:dark shirt", verdict["semantic_gate_errors"])

    def test_post_vace_semantic_verdict_rejects_shirt_edit_that_becomes_vest(self) -> None:
        record = {
            "edit_text": "change the patterned shirt to a solid black shirt",
            "difference": {"type": "attribute", "from": "patterned shirt", "to": "solid black shirt"},
            "generation": {
                "exploration_family": "clothing_color",
                "edit_token": "solid black shirt",
                "preserve_tokens": ["man", "ukulele", "microphone"],
                "post_vace_verdict": {
                    "semantic_gate_required": True,
                    "semantic_gate_passed": False,
                },
            },
        }

        updated = _apply_post_vace_semantic_verdict(
            record,
            target_annotation={
                "summary": "A man in a dark vest plays a ukulele near a microphone.",
                "subjects": ["man"],
                "actions": ["playing ukulele"],
                "object_counts": {"ukulele": 1, "microphone": 1},
            },
        )

        verdict = updated["generation"]["post_vace_verdict"]
        self.assertFalse(verdict["semantic_gate_passed"])
        self.assertEqual("clothing", verdict["semantic_gate_family"])
        self.assertIn("target_annotation_missing_target_clothing", verdict["semantic_gate_errors"])
        self.assertIn("target_annotation_forbidden_clothing_result:vest", verdict["semantic_gate_errors"])

    def test_post_vace_semantic_verdict_rejects_background_target_unchanged(self) -> None:
        record = {
            "edit_text": "change the background to a futuristic laboratory",
            "difference": {"type": "scene", "from": "brick wall background", "to": "futuristic laboratory background"},
            "generation": {
                "exploration_family": "background_change",
                "preserve_tokens": ["man", "ukulele", "microphone"],
                "post_vace_verdict": {
                    "semantic_gate_required": True,
                    "semantic_gate_passed": False,
                },
            },
        }

        updated = _apply_post_vace_semantic_verdict(
            record,
            target_annotation={
                "summary": "A blurry man performs in front of the same brick wall background.",
                "subjects": ["man"],
                "actions": ["performing"],
                "object_counts": {},
            },
        )

        verdict = updated["generation"]["post_vace_verdict"]
        self.assertFalse(verdict["semantic_gate_passed"])
        self.assertEqual("background", verdict["semantic_gate_family"])
        self.assertIn("target_annotation_missing_target_background", verdict["semantic_gate_errors"])
        self.assertIn("target_annotation_missing_preserved_object:ukulele", verdict["semantic_gate_errors"])

    def test_post_vace_semantic_verdict_rejects_background_overlay_and_original_room(self) -> None:
        record = {
            "edit_text": "change the background to a futuristic laboratory",
            "difference": {"type": "scene", "from": "original background", "to": "futuristic laboratory background"},
            "reference_caption": "A woman with curly red hair and glasses speaks in a sunlit room.",
            "generation": {
                "exploration_family": "background_change",
                "source_prompt": "A woman with curly red hair and glasses speaks to the camera in a sunlit room.",
                "preserve_tokens": ["woman", "curly red hair", "glasses", "speaking"],
                "preserve_regions": ["window", "door"],
                "post_vace_verdict": {
                    "semantic_gate_required": True,
                    "semantic_gate_passed": False,
                },
            },
        }

        updated = _apply_post_vace_semantic_verdict(
            record,
            target_annotation={
                "summary": (
                    "The same woman with curly red hair and glasses speaks in the original sunlit room; "
                    "the window and door are still visible under a blue overlay, with no futuristic laboratory present."
                ),
                "subjects": ["woman"],
                "actions": ["speaking"],
                "object_counts": {"glasses": 1, "window": 1, "door": 1},
            },
        )

        verdict = updated["generation"]["post_vace_verdict"]
        self.assertFalse(verdict["semantic_gate_passed"])
        self.assertEqual("failed_semantic_gate", verdict["stage"])
        self.assertIn("target_background_missing", verdict["semantic_gate_errors"])
        self.assertIn("original_background_retained", verdict["semantic_gate_errors"])
        self.assertIn("background_not_replaced_original_room_still_visible", verdict["semantic_gate_errors"])
        self.assertIn("futuristic_lab_only_blue_overlay", verdict["semantic_gate_errors"])
        self.assertIn("subject_preserved_but_edit_failed", verdict["semantic_gate_errors"])

    def test_post_vace_semantic_verdict_passes_background_lab_when_source_room_removed(self) -> None:
        record = {
            "edit_text": "change the background to a futuristic laboratory",
            "difference": {"type": "scene", "from": "sunlit room background", "to": "futuristic laboratory background"},
            "generation": {
                "exploration_family": "background_change",
                "source_prompt": "A woman with curly red hair and glasses speaks to the camera in a sunlit room.",
                "preserve_tokens": ["woman", "glasses", "speaking"],
                "post_vace_verdict": {
                    "semantic_gate_required": True,
                    "semantic_gate_passed": False,
                },
            },
        }

        updated = _apply_post_vace_semantic_verdict(
            record,
            target_annotation={
                "summary": (
                    "A woman with curly red hair and glasses speaks in front of a futuristic lab with benches, "
                    "glass equipment, and high tech wall panels."
                ),
                "scene": "futuristic laboratory",
                "subjects": ["woman"],
                "actions": ["speaking"],
                "object_counts": {"glasses": 1, "lab bench": 1},
            },
        )

        verdict = updated["generation"]["post_vace_verdict"]
        self.assertTrue(verdict["semantic_gate_passed"])
        self.assertEqual("passed_semantic_gate", verdict["stage"])

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
            self.assertEqual({"natural": 1}, summary["source_type_counts"])
            self.assertEqual({"natural:object_count": 1}, summary["source_type_difference_counts"])
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
            self.assertIn("Source Type Counts", report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {"clips/target.mp4", "clips/neg1.mp4", "clips/neg2.mp4"},
                {record["video_path"] for record in gallery_records},
            )

    def test_validate_pilot_dataset_allows_synthetic_plan_proposal_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("ref.mp4", "target.mp4", "neg.mp4"):
                (root / "clips" / name).write_bytes(b"x")

            pilot_path = root / "pilot.jsonl"
            self._write_jsonl(
                pilot_path,
                [
                    {
                        "sample_id": "covr_omni_synth_0001",
                        "proposal_id": "synthetic_audio_pair_0001",
                        "source_type": "synthetic_edit",
                        "reference_video": "clips/ref.mp4",
                        "target_video": "clips/target.mp4",
                        "edit_text": "add whoosh to the audio",
                        "modalities": ["audio"],
                        "reference_caption": "same visual clip without whoosh",
                        "target_caption": "same visual clip with whoosh",
                        "difference": {"type": "audio_event", "from": "no whoosh", "to": "whoosh"},
                        "hard_negatives": ["clips/neg.mp4"],
                        "quality": {
                            "same_context_score": 0.98,
                            "edit_match_score": 0.9,
                            "target_uniqueness_score": 0.9,
                            "difference_strength_score": 0.8,
                            "non_speech_audio_event_score": 0.9,
                        },
                        "source_context": {"relation": "synthetic_from_reference"},
                        "source": {
                            "platform": "synthetic",
                            "url": "file:///tmp/target.mp4",
                            "license_note": "internal research pilot only",
                        },
                        "generation": {
                            "model": "ffmpeg-deterministic-audio",
                            "model_route": "deterministic_overlay",
                            "source_video": "clips/ref.mp4",
                            "audio_edit_plan": {
                                "route": "deterministic_overlay",
                                "audio_prompt": "whoosh",
                                "expected_event": "whoosh",
                                "preserve_video": True,
                            },
                        },
                    }
                ],
            )

            summary = validate_pilot_dataset(
                root=root,
                pilot_jsonl_path=pilot_path,
                gallery_output_path=root / "gallery.jsonl",
                report_output_path=root / "review.md",
            )

            self.assertEqual(1, summary["sample_count"])
            self.assertEqual({"synthetic_edit": 1}, summary["source_type_counts"])

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
