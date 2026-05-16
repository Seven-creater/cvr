from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.audio_lines_single_source import (
    augment_b_inverse,
    build_b_splits,
    merge_line_results,
    prepare_existing_single_source_clips,
    split_audio_line_candidates,
    _inverse_b_line_edit_text,
)
from app.composed_data import (
    ensure_layout,
    propose_single_source_pairs,
    _b_line_edit_text_audio_only_issues,
    _is_transient_omni_exception,
    _single_source_pair_acceptance_issues,
    _single_source_final_verification_issues,
)


class AudioLinesSingleSourceTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_json_decode_errors_are_retried_as_transient_omni_failures(self) -> None:
        self.assertTrue(_is_transient_omni_exception(ValueError("JSONDecodeError: Expecting ',' delimiter")))

    def test_b_audio_review_final_issues_keep_quality_and_visual_boundaries(self) -> None:
        base = {
            "accept": True,
            "confidence": 0.9,
            "quality_score": 0.65,
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "observable_delta": True,
            "single_primary_delta": True,
            "text_or_ocr_driven": False,
            "segment_wide": False,
            "edit_text_accurate": True,
            "main_reject_reason": "",
            "evidence": ["target contains the requested speech"],
            "recommended_edit_text": "",
            "audio_primary": True,
            "visual_locked": True,
            "visual_too_different_for_B": False,
            "edit_text_audio_only": True,
        }
        model_fields = {"difference": {"type": "speech"}}
        self.assertNotIn(
            "final_omni_delta_not_segment_wide",
            _single_source_final_verification_issues(
                base,
                acceptance_profile="b_audio_review",
                audio_dataset_line="speech_audio_content",
                model_fields=model_fields,
            ),
        )

        low_quality = {**base, "quality_score": 0.59}
        self.assertIn(
            "final_omni_quality_score_below_threshold: 0.59 < 0.60",
            _single_source_final_verification_issues(
                low_quality,
                acceptance_profile="b_audio_review",
                audio_dataset_line="speech_audio_content",
                model_fields=model_fields,
            ),
        )

        visual_too_different = {**base, "visual_too_different_for_B": True, "audio_primary": False, "visual_locked": False}
        issues = _single_source_final_verification_issues(
            visual_too_different,
            acceptance_profile="b_audio_review",
            audio_dataset_line="speech_audio_content",
            model_fields=model_fields,
        )
        self.assertIn("final_omni_visual_too_different_for_B", issues)
        self.assertIn("final_omni_audio_not_primary", issues)

    def test_b_context_cvr_final_issues_reject_asr_degeneracy(self) -> None:
        base = {
            "accept": True,
            "confidence": 0.9,
            "quality_score": 0.72,
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "observable_delta": True,
            "single_primary_delta": True,
            "text_or_ocr_driven": False,
            "segment_wide": False,
            "edit_text_accurate": True,
            "main_reject_reason": "",
            "evidence": ["speech topic changes in the same clip context"],
            "recommended_edit_text": "",
            "audio_primary": True,
            "visual_locked": True,
            "visual_too_different_for_B": False,
            "edit_text_audio_only": True,
            "visual_context_preserved": True,
            "video_context_strength": 0.30,
            "asr_degeneracy_risk": 0.80,
            "not_asr_only": False,
        }
        issues = _single_source_final_verification_issues(
            base,
            acceptance_profile="b_audio_context_cvr",
            audio_dataset_line="speech_audio_content",
            model_fields={"difference": {"type": "speech"}},
        )
        self.assertIn("final_omni_video_context_too_weak: 0.30 < 0.45", issues)
        self.assertIn("final_omni_asr_degeneracy_risk_too_high: 0.80 > 0.55", issues)
        self.assertIn("final_omni_asr_only", issues)

    def test_b_context_cvr_local_gate_rejects_ami_auxiliary_source(self) -> None:
        issues = _single_source_pair_acceptance_issues(
            model_fields={
                "edit_text": "change the speech from discussing budget to discussing health",
                "modalities": ["audio"],
                "difference": {"type": "speech", "from": "budget", "to": "health"},
                "confidence": 0.9,
                "delta_temporal_extent": {"target_coverage": 0.8},
                "is_segment_wide_delta": True,
            },
            edit_text_quality={"score": 0.9},
            acceptance_profile="b_audio_context_cvr",
            audio_dataset_line="speech_audio_content",
            candidate_quality={
                "acceptance_profile": "b_audio_context_cvr",
                "video_context_strength": 0.7,
                "asr_degeneracy_risk": 0.2,
                "video_context_type": "meeting",
            },
            reference_annotation={"dataset": "ami_av", "speech": ["budget"]},
            target_annotation={"dataset": "ami_av", "speech": ["health"]},
        )
        self.assertIn("diagnostic_asr_auxiliary_source: AMI-AV is not accepted into main B-line", issues)

    def test_b_line_rejects_hollow_or_visual_leaky_audio_edit_text(self) -> None:
        bad_cases = [
            ("The speech content has been altered.", "speech", "generic audio placeholder"),
            ("change the speech from unintelligible speech to not transcribed speech", "speech", "hollow audio wording"),
            ("change the speech from discussing A to discussing B", "speech", "placeholder audio wording"),
            ("change the voice from saying \"A\" to saying \"B\"", "speech", "placeholder audio wording"),
            ("add target audio to the audio", "audio_event", "generic audio placeholder"),
            ("replace fishing reel sound; Two men are fishing near the river.", "audio_event", "visual clause in audio edit"),
            ("change the speech from discussing wide smile to discussing neutral smile", "speech", "visual wording smile"),
            ("change the speech from discussing hand gestures to discussing walking off-screen", "speech", "visual wording gesture"),
            ("add a SUBSCRIBE button and bell icon animation with a mouse click sound to the audio", "audio_event", "visual wording subscribe"),
            (
                "replace acoustic guitar music, fingerstyle, gentle melody with acoustic guitar music, fingerstyle, gentle melody, similar to reference",
                "audio_event",
                "weak audio_event delta",
            ),
            (
                "replace low electronic hum with low-frequency electronic hum",
                "audio_event",
                "audio_event endpoints too similar",
            ),
            (
                "change the speech from discussing imaging session and image processing workflows to discussing interesting as well and plan is to",
                "speech",
                "fragmentary speech wording",
            ),
        ]
        for edit_text, difference_type, expected in bad_cases:
            with self.subTest(edit_text=edit_text):
                issues = _b_line_edit_text_audio_only_issues(edit_text, difference_type)
                self.assertTrue(any(expected in issue for issue in issues), issues)

        self.assertEqual(
            [],
            _b_line_edit_text_audio_only_issues(
                "change the speech from discussing the bakery opening to discussing the mayor's remarks",
                "speech",
            ),
        )
        self.assertEqual(
            [],
            _b_line_edit_text_audio_only_issues(
                "change the speech from discussing imaging session and image processing workflows to discussing famous nebulas on the internet",
                "speech",
            ),
        )
        self.assertEqual(
            [],
            _b_line_edit_text_audio_only_issues("replace a continuous electronic hum with classical music", "audio_event"),
        )
        self.assertEqual(
            [],
            _b_line_edit_text_audio_only_issues("replace soft piano music with upbeat pop music", "audio_event"),
        )
        self.assertEqual([], _b_line_edit_text_audio_only_issues("add crowd cheering to the audio", "audio_event"))

    def test_prepare_existing_single_source_reconstructs_groups_and_reuses_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            folder = root / "clips" / "single_source" / "daily_source_001"
            folder.mkdir(parents=True)
            for index in range(1, 5):
                (folder / f"daily_source_001__single_{index:03d}.mp4").write_bytes(b"video")
            annotation_root = root / "runs" / "old"
            self._write_jsonl(
                annotation_root / "single_source_annotations.jsonl",
                [
                    {
                        "clip_id": "daily_source_001__single_001",
                        "output_path": "clips/single_source/daily_source_001/daily_source_001__single_001.mp4",
                        "summary": "speaker talks",
                        "speech": ["hello"],
                        "audio_events": [],
                        "modalities": ["audio", "visual"],
                    }
                ],
            )

            summary = prepare_existing_single_source_clips(
                root=root,
                single_source_root=root / "clips" / "single_source",
                run_root=root / "runs" / "audio_lines",
                annotation_search_roots=[annotation_root],
            )

            groups = [
                json.loads(line)
                for line in (root / "runs" / "audio_lines" / "single_source_clip_groups.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            annotations = [
                json.loads(line)
                for line in (root / "runs" / "audio_lines" / "single_source_annotations.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, summary["usable_group_count"])
            self.assertEqual(4, summary["segment_count"])
            self.assertEqual(["daily_source_001__single_001", "daily_source_001__single_002", "daily_source_001__single_003", "daily_source_001__single_004"], groups[0]["candidate_clip_ids"])
            self.assertEqual(1, len(annotations))
            self.assertEqual("daily_source_001__single_001", annotations[0]["clip_id"])
            self.assertTrue(summary["outputs"]["clips_to_annotate"].endswith("clips_to_annotate.jsonl"))

    def test_prepare_existing_single_source_can_force_fresh_limited_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for folder_index in range(1, 4):
                folder = root / "clips" / "single_source" / f"daily_source_{folder_index:03d}"
                folder.mkdir(parents=True)
                for index in range(1, 6):
                    (folder / f"daily_source_{folder_index:03d}__single_{index:03d}.mp4").write_bytes(b"video")
            annotation_root = root / "runs" / "old"
            self._write_jsonl(
                annotation_root / "single_source_annotations.jsonl",
                [
                    {
                        "clip_id": "daily_source_001__single_001",
                        "output_path": "clips/single_source/daily_source_001/daily_source_001__single_001.mp4",
                        "summary": "old reusable annotation",
                    }
                ],
            )

            summary = prepare_existing_single_source_clips(
                root=root,
                single_source_root=root / "clips" / "single_source",
                run_root=root / "runs" / "fresh_audio_lines",
                max_clips=9,
                annotation_search_roots=[annotation_root],
                reuse_annotations=False,
            )

            annotations_path = root / "runs" / "fresh_audio_lines" / "single_source_annotations.jsonl"
            clips_path = root / "runs" / "fresh_audio_lines" / "clips_to_annotate.jsonl"
            annotations = [line for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            clips = [line for line in clips_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(0, len(annotations))
            self.assertEqual(9, len(clips))
            self.assertFalse(summary["reuse_annotations"])
            self.assertEqual(9, summary["segment_count"])

    def test_prepare_existing_can_accept_two_clip_audio_cvr_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            folder = root / "clips" / "audio_cvr_8_12s" / "daily_source_001"
            folder.mkdir(parents=True)
            for index in range(1, 3):
                (folder / f"daily_source_001__single_{index:03d}.mp4").write_bytes(b"video")

            summary = prepare_existing_single_source_clips(
                root=root,
                single_source_root=root / "clips" / "audio_cvr_8_12s",
                run_root=root / "runs" / "audio_cvr",
                min_clips_per_folder=2,
                reuse_annotations=False,
            )

            groups = [
                json.loads(line)
                for line in (root / "runs" / "audio_cvr" / "single_source_clip_groups.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, summary["usable_group_count"])
            self.assertEqual(2, summary["segment_count"])
            self.assertEqual(2, summary["min_clips_per_folder"])
            self.assertEqual(["daily_source_001__single_001", "daily_source_001__single_002"], groups[0]["candidate_clip_ids"])

    def test_split_candidates_builds_a_and_b_lines_without_caption_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations = [
                {
                    "clip_id": "seg_1",
                    "output_path": "clips/seg_1.mp4",
                    "summary": "speaker discusses a fruit market without an overlay",
                    "subjects": ["speaker"],
                    "speech": ["I like apples from the morning market because they taste sweet today"],
                    "speakers_and_transcript": ["speaker: I like apples from the morning market because they taste sweet today"],
                    "audio_events": [],
                    "modalities": ["audio", "visual"],
                },
                {
                    "clip_id": "seg_2",
                    "output_path": "clips/seg_2.mp4",
                    "summary": "speaker discusses a fruit market with an overlay",
                    "subjects": ["speaker"],
                    "speech": ["I like oranges from the evening market because they taste bright today"],
                    "speakers_and_transcript": ["speaker: I like oranges from the evening market because they taste bright today"],
                    "audio_events": [],
                    "modalities": ["audio", "visual"],
                },
            ]
            candidates = [
                {
                    "candidate_id": "c1",
                    "proposal_id": "p1",
                    "reference_clip_id": "seg_1",
                    "target_clip_id": "seg_2",
                    "reference_video": "clips/seg_1.mp4",
                    "target_video": "clips/seg_2.mp4",
                    "difference": {"type": "object_presence", "from": "no overlay", "to": "overlay"},
                    "quality": {"same_context_score": 0.8},
                    "composite_score": 0.8,
                }
            ]
            annotations_path = root / "ann.jsonl"
            candidates_path = root / "cand.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            self._write_jsonl(annotations_path, annotations)
            self._write_jsonl(candidates_path, candidates)

            with mock.patch("app.audio_lines_single_source._pair_audio_anchor_score", return_value=(0.91, 0.05)):
                summary = split_audio_line_candidates(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    a_output_path=a_path,
                    b_output_path=b_path,
                    summary_path=root / "summary.json",
                )

            a_records = [json.loads(line) for line in a_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            b_records = [json.loads(line) for line in b_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["a_candidate_count"])
            self.assertEqual(1, summary["b_candidate_count"])
            self.assertEqual("visual_audio_anchor", a_records[0]["audio_dataset_line"])
            self.assertEqual("speech_audio_content", b_records[0]["audio_dataset_line"])
            self.assertEqual("speech", b_records[0]["difference"]["type"])
            self.assertNotIn("target_caption", a_records[0])

    def test_v4_strict_a_line_prefers_large_visual_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "ann.jsonl"
            candidates_path = root / "cand.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "anchor",
                        "output_path": "clips/anchor.mp4",
                        "summary": "news anchor in a studio speaking to camera",
                        "subjects": ["news anchor"],
                        "actions": ["speaking"],
                        "scene": "studio",
                        "attributes": ["desk", "blue backdrop"],
                    },
                    {
                        "clip_id": "flood",
                        "output_path": "clips/flood.mp4",
                        "summary": "aerial footage of flooded streets and buildings",
                        "subjects": ["flooded city"],
                        "actions": ["water flowing through streets"],
                        "scene": "outdoor flood aerial",
                        "attributes": ["water", "buildings"],
                    },
                    {
                        "clip_id": "anchor_bright",
                        "output_path": "clips/anchor_bright.mp4",
                        "summary": "news anchor in a studio speaking to camera",
                        "subjects": ["news anchor"],
                        "actions": ["speaking"],
                        "scene": "studio",
                        "attributes": ["slightly brighter backdrop"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "scene_ok",
                        "proposal_id": "scene_ok",
                        "reference_clip_id": "anchor",
                        "target_clip_id": "flood",
                        "reference_video": "clips/anchor.mp4",
                        "target_video": "clips/flood.mp4",
                        "difference": {"type": "scene", "from": "studio anchor", "to": "flood aerial"},
                    },
                    {
                        "candidate_id": "attribute_weak",
                        "proposal_id": "attribute_weak",
                        "reference_clip_id": "anchor",
                        "target_clip_id": "anchor_bright",
                        "reference_video": "clips/anchor.mp4",
                        "target_video": "clips/anchor_bright.mp4",
                        "difference": {"type": "attribute", "from": "blue backdrop", "to": "brighter blue backdrop"},
                    },
                    {
                        "candidate_id": "attribute_mislabeled_high",
                        "proposal_id": "attribute_mislabeled_high",
                        "reference_clip_id": "anchor",
                        "target_clip_id": "flood",
                        "reference_video": "clips/anchor.mp4",
                        "target_video": "clips/flood.mp4",
                        "difference": {"type": "attribute", "from": "studio anchor", "to": "flood aerial"},
                    },
                ],
            )

            with mock.patch("app.audio_lines_single_source._pair_audio_anchor_score", return_value=(0.93, 0.05)):
                summary = split_audio_line_candidates(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    a_output_path=a_path,
                    b_output_path=b_path,
                    summary_path=root / "summary.json",
                    audio_line_quality_profile="v4_strict",
                )

            a_records = [json.loads(line) for line in a_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual("v4_strict", summary["audio_line_quality_profile"])
            self.assertEqual({"scene_ok", "attribute_mislabeled_high"}, {record["candidate_id"] for record in a_records})
            self.assertGreaterEqual(a_records[0]["quality"]["visual_delta_strength"], 0.45)

    def test_v4_strict_b_line_requires_similar_visual_context_and_concrete_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "ann.jsonl"
            candidates_path = root / "cand.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "cricket_quiet",
                        "output_path": "clips/cricket_quiet.mp4",
                        "start_seconds": 0.0,
                        "summary": "cricket match broadcast showing players on the field",
                        "subjects": ["cricket players", "field"],
                        "actions": ["playing cricket"],
                        "scene": "sports broadcast stadium",
                        "audio_events": ["commentary"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "cricket_cheer",
                        "output_path": "clips/cricket_cheer.mp4",
                        "start_seconds": 6.0,
                        "summary": "cricket match broadcast showing players on the field",
                        "subjects": ["cricket players", "field"],
                        "actions": ["playing cricket"],
                        "scene": "sports broadcast stadium",
                        "audio_events": ["crowd cheering", "commentary"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "kitchen_hum",
                        "output_path": "clips/kitchen_hum.mp4",
                        "start_seconds": 12.0,
                        "summary": "close view of a kitchen counter and appliance",
                        "subjects": ["kitchen appliance"],
                        "actions": ["appliance running"],
                        "scene": "kitchen",
                        "audio_events": ["electronic hum"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "cheer_ok",
                        "proposal_id": "cheer_ok",
                        "reference_clip_id": "cricket_quiet",
                        "target_clip_id": "cricket_cheer",
                        "reference_video": "clips/cricket_quiet.mp4",
                        "target_video": "clips/cricket_cheer.mp4",
                        "difference": {"type": "audio_event", "from": "commentary", "to": "crowd cheering"},
                    },
                    {
                        "candidate_id": "visual_too_different",
                        "proposal_id": "visual_too_different",
                        "reference_clip_id": "cricket_quiet",
                        "target_clip_id": "kitchen_hum",
                        "reference_video": "clips/cricket_quiet.mp4",
                        "target_video": "clips/kitchen_hum.mp4",
                        "difference": {"type": "audio_event", "from": "commentary", "to": "electronic hum"},
                    },
                ],
            )

            summary = split_audio_line_candidates(
                root=root,
                clip_annotations_path=annotations_path,
                pair_candidates_path=candidates_path,
                a_output_path=a_path,
                b_output_path=b_path,
                summary_path=root / "summary.json",
                audio_line_quality_profile="v4_strict",
            )

            b_records = [json.loads(line) for line in b_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["b_candidate_count"])
            self.assertEqual("cheer_ok", b_records[0]["candidate_id"])
            self.assertEqual("audio_event", b_records[0]["difference"]["type"])
            self.assertGreaterEqual(b_records[0]["quality"]["visual_context_similarity"], 0.30)

    def test_v4_strict_b_line_mines_audio_first_pairs_from_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "ann.jsonl"
            candidates_path = root / "cand.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "daily_source__single_001",
                        "output_path": "clips/source_a/daily_source__single_001.mp4",
                        "start_seconds": 0.0,
                        "summary": "podium speech in a conference hall",
                        "subjects": ["speaker", "podium"],
                        "actions": ["speaking at podium"],
                        "scene": "conference hall",
                        "speech": ["the speaker explains the budget plan for local transportation improvements"],
                        "speakers_and_transcript": ["speaker: the speaker explains the budget plan for local transportation improvements"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "daily_source__single_002",
                        "output_path": "clips/source_a/daily_source__single_002.mp4",
                        "start_seconds": 6.0,
                        "summary": "podium speech in a conference hall",
                        "subjects": ["speaker", "podium"],
                        "actions": ["speaking at podium"],
                        "scene": "conference hall",
                        "speech": ["the speaker describes a public health program for neighborhood clinics"],
                        "speakers_and_transcript": ["speaker: the speaker describes a public health program for neighborhood clinics"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(candidates_path, [])

            summary = split_audio_line_candidates(
                root=root,
                clip_annotations_path=annotations_path,
                pair_candidates_path=candidates_path,
                a_output_path=a_path,
                b_output_path=b_path,
                summary_path=root / "summary.json",
                audio_line_quality_profile="v4_strict",
            )

            b_records = [json.loads(line) for line in b_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["b_audio_first_candidate_count"])
            self.assertEqual(1, summary["b_candidate_count"])
            self.assertEqual("audio_first_annotation_pair", b_records[0]["quality"]["candidate_source"])
            self.assertEqual("speech", b_records[0]["difference"]["type"])

    def test_v5_audio_primary_b_line_mines_speech_topic_pairs_without_transcript_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "ann.jsonl"
            candidates_path = root / "cand.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "live__single_001",
                        "output_path": "clips/live/live__single_001.mp4",
                        "start_seconds": 0.0,
                        "summary": "same livestream speaker at a desk",
                        "subjects": ["speaker", "desk"],
                        "actions": ["speaking to camera"],
                        "scene": "indoor livestream desk",
                        "speech": ["the speaker talks about budget planning and transportation funding"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "live__single_002",
                        "output_path": "clips/live/live__single_002.mp4",
                        "start_seconds": 6.0,
                        "summary": "same livestream speaker at a desk",
                        "subjects": ["speaker", "desk"],
                        "actions": ["speaking to camera"],
                        "scene": "indoor livestream desk",
                        "speech": ["the speaker talks about clinic staffing and public health services"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(candidates_path, [])

            summary = split_audio_line_candidates(
                root=root,
                clip_annotations_path=annotations_path,
                pair_candidates_path=candidates_path,
                a_output_path=a_path,
                b_output_path=b_path,
                summary_path=root / "summary.json",
                audio_line_quality_profile="v5_audio_primary",
                b_candidate_mode="audio_first",
            )

            b_records = [json.loads(line) for line in b_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual("v5_audio_primary", summary["audio_line_quality_profile"])
            self.assertEqual("audio_first", summary["b_candidate_mode"])
            self.assertEqual(1, summary["b_audio_first_candidate_count"])
            self.assertEqual(1, summary["b_candidate_count"])
            self.assertEqual("speech", b_records[0]["difference"]["type"])
            self.assertGreaterEqual(b_records[0]["quality"]["speech_evidence_score"], 0.45)

    def test_b_context_cvr_rejects_asr_only_talking_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "ann.jsonl"
            candidates_path = root / "cand.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "meeting__single_001",
                        "output_path": "clips/meeting/meeting__single_001.mp4",
                        "start_seconds": 0.0,
                        "summary": "static talking head in a zoom meeting",
                        "subjects": ["speaker"],
                        "actions": ["speaking to camera"],
                        "scene": "webinar meeting",
                        "speech": ["the speaker says the meeting starts at nine"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "meeting__single_002",
                        "output_path": "clips/meeting/meeting__single_002.mp4",
                        "start_seconds": 6.0,
                        "summary": "static talking head in a zoom meeting",
                        "subjects": ["speaker"],
                        "actions": ["speaking to camera"],
                        "scene": "webinar meeting",
                        "speech": ["the speaker says the budget review starts tomorrow"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(candidates_path, [])

            summary = split_audio_line_candidates(
                root=root,
                clip_annotations_path=annotations_path,
                pair_candidates_path=candidates_path,
                a_output_path=a_path,
                b_output_path=b_path,
                summary_path=root / "summary.json",
                audio_line_quality_profile="b_audio_context_cvr",
                b_candidate_mode="audio_first",
            )

            self.assertEqual(0, summary["b_candidate_count"])
            self.assertGreaterEqual(summary["reject_counts"].get("b_audio_first_speech_visual_gate_failed", 0), 1)

    def test_b_context_cvr_accepts_tutorial_speech_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "ann.jsonl"
            candidates_path = root / "cand.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "cook__single_001",
                        "output_path": "clips/cook/cook__single_001.mp4",
                        "start_seconds": 0.0,
                        "summary": "cooking tutorial host at a kitchen counter",
                        "subjects": ["host", "kitchen counter", "ingredients"],
                        "actions": ["explaining ingredients"],
                        "scene": "kitchen tutorial",
                        "speech": ["the host explains flour sugar and eggs for the cake batter"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "cook__single_002",
                        "output_path": "clips/cook/cook__single_002.mp4",
                        "start_seconds": 6.0,
                        "summary": "cooking tutorial host at a kitchen counter",
                        "subjects": ["host", "kitchen counter", "mixing bowl"],
                        "actions": ["explaining mixing steps"],
                        "scene": "kitchen tutorial",
                        "speech": ["the host explains how to whisk the batter until smooth"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(candidates_path, [])

            summary = split_audio_line_candidates(
                root=root,
                clip_annotations_path=annotations_path,
                pair_candidates_path=candidates_path,
                a_output_path=a_path,
                b_output_path=b_path,
                summary_path=root / "summary.json",
                audio_line_quality_profile="b_audio_context_cvr",
                b_candidate_mode="audio_first",
            )

            b_records = [json.loads(line) for line in b_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["b_candidate_count"])
            self.assertEqual("speech_topic_in_video_context", b_records[0]["quality"]["b_subtype"])
            self.assertEqual("tutorial_instruction", b_records[0]["quality"]["video_context_type"])
            self.assertLessEqual(b_records[0]["quality"]["asr_degeneracy_risk"], 0.55)

    def test_merge_b_line_outputs_context_buckets_and_caps_speech(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            b_shards = run_root / "b_shards"
            b_shards.mkdir(parents=True)
            accepted = []
            for index in range(1, 7):
                subtype = "speech_topic_in_video_context" if index <= 4 else "music"
                accepted.append(
                    {
                        "proposal_id": f"p{index}",
                        "reference_video": f"ref{index}.mp4",
                        "target_video": f"tgt{index}.mp4",
                        "edit_text": "change the tutorial narration" if subtype.startswith("speech") else "change the background music",
                        "difference": {"type": "speech" if subtype.startswith("speech") else "audio_event"},
                        "accepted": True,
                        "b_subtype": subtype,
                        "audio_delta_strength": 0.82,
                        "video_context_strength": 0.72,
                        "asr_degeneracy_risk": 0.18,
                        "visual_shortcut_risk": False,
                        "audio_only_verification": {"accept": True},
                        "video_only_shortcut": {"can_identify_target_without_audio": False},
                        "audio_only_solvability": 0.50,
                        "full_av_required": True,
                        "quality": {
                            "b_subtype": subtype,
                            "audio_delta_strength": 0.82,
                            "video_context_strength": 0.72,
                            "asr_degeneracy_risk": 0.18,
                        },
                    }
                )
            self._write_jsonl(b_shards / "accepted_progress_01.jsonl", accepted)
            self._write_jsonl(b_shards / "ranked_01.jsonl", accepted)

            summary = merge_line_results(run_root=run_root, target_a_count=0, target_b_count=5)

            selected = [
                json.loads(line)
                for line in (run_root / "b_speech_audio_content_triplets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            speech = [
                json.loads(line)
                for line in (run_root / "b_speech_context_triplets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            music = [
                json.loads(line)
                for line in (run_root / "b_music_triplets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            main = [
                json.loads(line)
                for line in (run_root / "b_main_audio_cvr_triplets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            extended = [
                json.loads(line)
                for line in (run_root / "b_extended_audio_cvr_triplets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(6, len(selected))
            self.assertEqual(4, len(speech))
            self.assertEqual(2, len(music))
            self.assertEqual(3, len(main))
            self.assertEqual(3, len(extended))
            self.assertLessEqual(sum(1 for record in main if record["b_subtype"] == "speech_topic_in_video_context"), 1)
            self.assertIn("b_context_cvr_summary_path", summary)
            self.assertEqual(3, summary["b_main_count"])

    def test_merge_b_line_can_keep_all_accepted_records_for_large_b_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            b_shards = run_root / "b_shards"
            b_shards.mkdir(parents=True)
            accepted = []
            for index in range(1, 7):
                accepted.append(
                    {
                        "proposal_id": f"b{index}",
                        "reference_video": f"ref{index}.mp4",
                        "target_video": f"tgt{index}.mp4",
                        "edit_text": f"change the speech from discussing topic {index} to discussing topic {index + 1}",
                        "difference": {"type": "speech"},
                        "accepted": True,
                        "b_subtype": "speech_topic_in_video_context",
                        "audio_delta_strength": 0.65,
                        "video_context_strength": 0.50,
                        "asr_degeneracy_risk": 0.40,
                        "visual_shortcut_risk": False,
                        "audio_only_solvability": 0.55,
                        "quality": {
                            "b_subtype": "speech_topic_in_video_context",
                            "audio_delta_strength": 0.65,
                            "video_context_strength": 0.50,
                            "asr_degeneracy_risk": 0.40,
                        },
                    }
                )
            self._write_jsonl(b_shards / "ranked_01.jsonl", accepted)

            summary = merge_line_results(run_root=run_root, target_a_count=0, target_b_count=2, keep_all_b=True)

            selected = [
                json.loads(line)
                for line in (run_root / "b_speech_audio_content_triplets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(6, len(selected))
            self.assertEqual(6, summary["b_exported_count"])
            self.assertTrue(summary["keep_all_b"])
            self.assertEqual({"extended": 6}, summary["b_split_tier_counts"])

    def test_merge_b_line_exports_main_extended_and_diagnostic_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            b_shards = run_root / "b_shards"
            b_shards.mkdir(parents=True)
            accepted = [
                {
                    "proposal_id": "main_music",
                    "reference_video": "ref1.mp4",
                    "target_video": "tgt1.mp4",
                    "edit_text": "replace quiet guitar music with upbeat piano music",
                    "difference": {"type": "audio_event"},
                    "accepted": True,
                    "b_subtype": "music",
                    "audio_delta_strength": 0.90,
                    "video_context_strength": 0.80,
                    "asr_degeneracy_risk": 0.10,
                    "visual_shortcut_risk": False,
                    "audio_only_verification": {"accept": True},
                    "video_only_shortcut": {"can_identify_target_without_audio": False},
                    "audio_only_solvability": 0.50,
                    "full_av_required": True,
                    "quality": {"b_subtype": "music"},
                },
                {
                    "proposal_id": "extended_speech",
                    "reference_video": "ref2.mp4",
                    "target_video": "tgt2.mp4",
                    "edit_text": "change the commentary from introducing players to describing the goal",
                    "difference": {"type": "speech"},
                    "accepted": True,
                    "b_subtype": "speech_topic_in_video_context",
                    "audio_delta_strength": 0.64,
                    "video_context_strength": 0.50,
                    "asr_degeneracy_risk": 0.45,
                    "visual_shortcut_risk": False,
                    "audio_only_verification": {"accept": True},
                    "video_only_shortcut": {"can_identify_target_without_audio": False},
                    "audio_only_solvability": 0.55,
                    "quality": {"b_subtype": "speech_topic_in_video_context"},
                },
                {
                    "proposal_id": "diagnostic_asr",
                    "reference_video": "ref3.mp4",
                    "target_video": "tgt3.mp4",
                    "edit_text": "change the voice from saying \"hello there\" to saying \"goodbye now\"",
                    "difference": {"type": "speech"},
                    "accepted": True,
                    "b_subtype": "speech_topic_in_video_context",
                    "audio_delta_strength": 0.90,
                    "video_context_strength": 0.30,
                    "asr_degeneracy_risk": 0.82,
                    "visual_shortcut_risk": False,
                    "audio_only_solvability": 0.92,
                    "speech_role": "asr_only",
                    "quality": {"b_subtype": "speech_topic_in_video_context"},
                },
            ]
            self._write_jsonl(b_shards / "ranked_01.jsonl", accepted)

            summary = merge_line_results(run_root=run_root, target_a_count=0, target_b_count=3, keep_all_b=True)

            all_rows = [json.loads(line) for line in (run_root / "b_all_audio_cvr_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            main_rows = [json.loads(line) for line in (run_root / "b_main_audio_cvr_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            extended_rows = [json.loads(line) for line in (run_root / "b_extended_audio_cvr_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            diagnostic_rows = [json.loads(line) for line in (run_root / "b_diagnostic_asr_risk_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(["main", "extended", "diagnostic"], [record["split_tier"] for record in all_rows])
            self.assertEqual(["main_music"], [record["proposal_id"] for record in main_rows])
            self.assertEqual(["extended_speech"], [record["proposal_id"] for record in extended_rows])
            self.assertEqual(["diagnostic_asr"], [record["proposal_id"] for record in diagnostic_rows])
            self.assertFalse(diagnostic_rows[0]["benchmark_eligible"])
            self.assertIn("asr_degeneracy_risk_high", diagnostic_rows[0]["diagnostic_reason"])
            self.assertEqual({"main": 1, "extended": 1, "diagnostic": 1}, summary["b_split_tier_counts"])

    def test_b_tier_thresholds_require_audio_only_and_guard_voxceleb_main(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            b_shards = run_root / "b_shards"
            b_shards.mkdir(parents=True)
            records = [
                {
                    "proposal_id": "main_loose_threshold",
                    "reference_video": "ref1.mp4",
                    "target_video": "tgt1.mp4",
                    "edit_text": "replace quiet ambience with crowd cheering",
                    "difference": {"type": "audio_event"},
                    "accepted": True,
                    "b_subtype": "sound_event",
                    "audio_delta_strength": 0.70,
                    "video_context_strength": 0.46,
                    "asr_degeneracy_risk": 0.54,
                    "visual_shortcut_risk": False,
                    "audio_only_verification": {"accept": True},
                    "video_only_shortcut": {"can_identify_target_without_audio": False},
                    "quality": {"b_subtype": "sound_event"},
                },
                {
                    "proposal_id": "no_audio_only_accept",
                    "reference_video": "ref2.mp4",
                    "target_video": "tgt2.mp4",
                    "edit_text": "replace quiet ambience with applause",
                    "difference": {"type": "audio_event"},
                    "accepted": True,
                    "b_subtype": "sound_event",
                    "audio_delta_strength": 0.90,
                    "video_context_strength": 0.80,
                    "asr_degeneracy_risk": 0.10,
                    "visual_shortcut_risk": False,
                    "video_only_shortcut": {"can_identify_target_without_audio": False},
                    "quality": {"b_subtype": "sound_event"},
                },
                {
                    "proposal_id": "voxceleb_guarded",
                    "dataset": "voxceleb",
                    "reference_video": "raw/voxceleb/a/ref.mp4",
                    "target_video": "raw/voxceleb/a/tgt.mp4",
                    "edit_text": "replace quiet speech with applause",
                    "difference": {"type": "audio_event"},
                    "accepted": True,
                    "b_subtype": "sound_event",
                    "audio_delta_strength": 0.90,
                    "video_context_strength": 0.60,
                    "asr_degeneracy_risk": 0.20,
                    "visual_shortcut_risk": False,
                    "audio_only_verification": {"accept": True},
                    "video_only_shortcut": {"can_identify_target_without_audio": False},
                    "quality": {"b_subtype": "sound_event"},
                },
                {
                    "proposal_id": "voxceleb_strong",
                    "dataset": "voxceleb",
                    "reference_video": "raw/voxceleb/b/ref.mp4",
                    "target_video": "raw/voxceleb/b/tgt.mp4",
                    "edit_text": "replace quiet room ambience with crowd cheering",
                    "difference": {"type": "audio_event"},
                    "accepted": True,
                    "b_subtype": "sound_event",
                    "audio_delta_strength": 0.92,
                    "video_context_strength": 0.72,
                    "asr_degeneracy_risk": 0.20,
                    "visual_shortcut_risk": False,
                    "audio_only_verification": {"accept": True},
                    "video_only_shortcut": {"can_identify_target_without_audio": False},
                    "quality": {"b_subtype": "sound_event"},
                },
            ]
            self._write_jsonl(b_shards / "ranked_01.jsonl", records)

            merge_line_results(run_root=run_root, target_a_count=0, target_b_count=4, keep_all_b=True)

            all_rows = [json.loads(line) for line in (run_root / "b_all_audio_cvr_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            by_id = {record["proposal_id"]: record for record in all_rows}
            self.assertEqual("main", by_id["main_loose_threshold"]["split_tier"])
            self.assertEqual("extended", by_id["no_audio_only_accept"]["split_tier"])
            self.assertIn("audio_only_verification_not_accepted", by_id["no_audio_only_accept"]["diagnostic_reason"])
            self.assertEqual("extended", by_id["voxceleb_guarded"]["split_tier"])
            self.assertIn("voxceleb_main_guard", by_id["voxceleb_guarded"]["diagnostic_reason"])
            self.assertEqual("main", by_id["voxceleb_strong"]["split_tier"])

    def test_merge_b_line_mines_typed_hard_negatives_and_missing_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            b_shards = run_root / "b_shards"
            b_shards.mkdir(parents=True)
            positive = {
                "proposal_id": "positive",
                "raw_source_id": "source_show_1",
                "reference_video": "ref.mp4",
                "target_video": "target.mp4",
                "edit_text": "change the speech from discussing budget planning to discussing health services",
                "difference": {"type": "speech"},
                "accepted": True,
                "b_subtype": "speech_topic_in_video_context",
                "audio_delta_strength": 0.65,
                "video_context_strength": 0.50,
                "asr_degeneracy_risk": 0.40,
                "visual_shortcut_risk": False,
                "audio_only_verification": {"accept": True},
                "video_only_shortcut": {"can_identify_target_without_audio": False},
                "quality": {"b_subtype": "speech_topic_in_video_context"},
            }
            ranked = [
                positive,
                {
                    "proposal_id": "visual_hard_candidate",
                    "raw_source_id": "source_show_1",
                    "target_video": "visual_hard.mp4",
                    "edit_text": "change the speech from discussing sports to discussing weather",
                    "difference": {"type": "speech"},
                    "accepted": False,
                    "b_subtype": "speech_topic_in_video_context",
                    "visual_context_similarity": 0.90,
                    "video_context_strength": 0.80,
                    "audio_delta_strength": 0.20,
                },
                {
                    "proposal_id": "audio_hard_candidate",
                    "raw_source_id": "source_show_2",
                    "target_video": "audio_hard.mp4",
                    "edit_text": "change the speech from discussing budget planning to discussing health services",
                    "difference": {"type": "speech"},
                    "accepted": False,
                    "b_subtype": "speech_topic_in_video_context",
                    "audio_delta_strength": 0.80,
                },
                {
                    "proposal_id": "asr_hard_candidate",
                    "raw_source_id": "source_show_3",
                    "target_video": "asr_hard.mp4",
                    "edit_text": "change the speech from discussing budget planning to discussing public health",
                    "difference": {"type": "speech"},
                    "accepted": False,
                    "b_subtype": "speech_topic_in_video_context",
                    "audio_delta_strength": 0.70,
                },
            ]
            self._write_jsonl(b_shards / "ranked_01.jsonl", ranked)

            merge_line_results(run_root=run_root, target_a_count=0, target_b_count=1, keep_all_b=True)

            row = [json.loads(line) for line in (run_root / "b_all_audio_cvr_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][0]
            negative_types = {item["type"] for item in row["audio_delta_hard_negatives"]}
            self.assertIn("reference_negative", negative_types)
            self.assertIn("visual_hard", negative_types)
            self.assertIn("audio_hard", negative_types)
            self.assertIn("asr_hard", negative_types)
            self.assertEqual({}, row["hard_negative_missing_reasons"])

            ranked_without_cross_source = [positive, ranked[1]]
            self._write_jsonl(b_shards / "ranked_01.jsonl", ranked_without_cross_source)
            merge_line_results(run_root=run_root, target_a_count=0, target_b_count=1, keep_all_b=True)
            row = [json.loads(line) for line in (run_root / "b_all_audio_cvr_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][0]
            self.assertIn("audio_hard", row["hard_negative_missing_reasons"])
            self.assertIn("asr_hard", row["hard_negative_missing_reasons"])

    def test_build_b_splits_is_source_and_pair_group_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            rows = []
            for index in range(1, 7):
                rows.append(
                    {
                        "proposal_id": f"main_{index}",
                        "accepted": True,
                        "split_tier": "main",
                        "benchmark_eligible": True,
                        "training_eligible": True,
                        "raw_source_id": f"source_{index}",
                        "pair_group_id": f"pair_{index}",
                        "inverse_pair_group_id": f"pair_{index}",
                        "reference_video": f"ref{index}.mp4",
                        "target_video": f"tgt{index}.mp4",
                        "edit_text": "replace quiet room ambience with crowd cheering",
                    }
                )
            self._write_jsonl(run_root / "b_main_audio_cvr_triplets.jsonl", rows)
            self._write_jsonl(run_root / "b_extended_audio_cvr_triplets.jsonl", [])
            self._write_jsonl(run_root / "b_diagnostic_asr_risk_triplets.jsonl", [{"proposal_id": "diag", "raw_source_id": "source_diag", "pair_group_id": "pair_diag"}])
            inverse = dict(rows[0])
            inverse.update({"proposal_id": "inverse_main_1", "is_inverse": True, "derived_from_inverse": True, "direction": "inverse"})
            self._write_jsonl(run_root / "b_inverse_accepted.jsonl", [inverse])

            summary = build_b_splits(run_root=run_root, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)

            self.assertEqual([], summary["leakage_violations"])
            test_main = [json.loads(line) for line in (run_root / "b_splits" / "test_main.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            pair_groups = [record["pair_group_id"] for record in test_main]
            self.assertEqual(len(pair_groups), len(set(pair_groups)))
            self.assertTrue(all(not record.get("is_inverse") for record in test_main))
            train = [json.loads(line) for line in (run_root / "b_splits" / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            inverse_diag = [json.loads(line) for line in (run_root / "b_splits" / "test_inverse_diagnostic.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, sum(1 for record in train + inverse_diag if record.get("is_inverse")))

    def test_a_omni_first_keeps_audio_anchor_pairs_for_omni_visual_judging(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations_path = root / "ann.jsonl"
            candidates_path = root / "cand.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "source__single_001",
                        "output_path": "clips/source/source__single_001.mp4",
                        "start_seconds": 0.0,
                        "summary": "a host speaking in a studio",
                        "subjects": ["host"],
                        "actions": ["speaking"],
                        "scene": "studio",
                        "attributes": ["desk"],
                    },
                    {
                        "clip_id": "source__single_002",
                        "output_path": "clips/source/source__single_002.mp4",
                        "start_seconds": 6.0,
                        "summary": "field footage of flooded streets",
                        "subjects": ["flooded streets"],
                        "actions": ["water moving"],
                        "scene": "outdoor flood scene",
                        "attributes": ["water", "buildings"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "speech_hint_but_visual_possible",
                        "proposal_id": "speech_hint_but_visual_possible",
                        "reference_clip_id": "source__single_001",
                        "target_clip_id": "source__single_002",
                        "reference_video": "clips/source/source__single_001.mp4",
                        "target_video": "clips/source/source__single_002.mp4",
                        "difference": {"type": "speech", "from": "studio speech", "to": "field report speech"},
                    }
                ],
            )

            with mock.patch("app.audio_lines_single_source._pair_audio_anchor_score", return_value=(0.92, 0.08)):
                summary = split_audio_line_candidates(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    a_output_path=a_path,
                    b_output_path=b_path,
                    summary_path=root / "summary.json",
                    audio_line_quality_profile="v5_audio_primary",
                    a_candidate_mode="omni_first",
                )

            a_records = [json.loads(line) for line in a_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual("omni_first", summary["a_candidate_mode"])
            self.assertEqual(1, summary["a_candidate_count"])
            self.assertEqual("visual_audio_anchor", a_records[0]["audio_dataset_line"])
            self.assertEqual("speech", a_records[0]["quality"]["visual_hint_difference_type"])
            self.assertEqual("v5_audio_primary", a_records[0]["audio_line_quality_profile"])

    def test_speech_audio_content_line_allows_speech_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "speaker says hello",
                        "speech": ["hello"],
                        "speakers_and_transcript": ["speaker: hello"],
                        "audio_events": [],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "speaker says goodbye",
                        "speech": ["goodbye"],
                        "speakers_and_transcript": ["speaker: goodbye"],
                        "audio_events": [],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "hello", "to": "goodbye"},
                        "quality": {"speech_transcript_backed": 1.0, "speech_evidence_score": 0.9, "speech_specificity_score": 0.9, "has_audio_modality": 1.0},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the speech from hello to goodbye",
                    "modalities": ["audio"],
                    "reference_caption": "speaker says hello",
                    "target_caption": "speaker says goodbye",
                    "difference": {"type": "speech", "from": "hello", "to": "goodbye", "description": "spoken content changes"},
                    "dominant_delta": {"type": "speech", "from": "hello", "to": "goodbye", "reason": "transcripts differ"},
                    "reference_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "talking head", "internal_transitions": []},
                    "target_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "talking head", "internal_transitions": []},
                    "delta_temporal_extent": {"reference": "hello is spoken", "target": "goodbye is spoken", "target_coverage": 0.9, "evidence": "target transcript says goodbye"},
                    "subject_roles": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": ""},
                    "is_segment_wide_delta": True,
                    "discarded_deltas": [],
                    "evidence": ["reference says hello; target says goodbye"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.9,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": True,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["the audible speech changes from hello to goodbye"],
                    "recommended_edit_text": "",
                    "audio_primary": True,
                    "visual_locked": True,
                    "visual_too_different_for_B": False,
                    "edit_text_audio_only": True,
                },
                {"raw": "final"},
            )
            client.refine_b_line_edit_text.return_value = (
                {
                    "refined_edit_text": "change the speech from discussing sports to discussing weather",
                    "edit_text_specificity_score": 0.9,
                    "reject_if_unspecific": False,
                    "edit_text_reject_reason": "",
                    "speech_or_audio_evidence": ["reference discusses sports", "target discusses weather"],
                },
                {"raw": "refine"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="exploration",
                    audio_dataset_line="speech_audio_content",
                )

            accepted = [json.loads(line) for line in (root / "pairs" / "accepted.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual("speech_audio_content", accepted[0]["audio_dataset_line"])
            self.assertEqual("speech", accepted[0]["difference"]["type"])

    def test_b_audio_blind_review_accepts_audio_only_verified_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4", "ref.wav", "tgt.wav"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"media")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "sports broadcast with the same field view",
                        "scene": "stadium broadcast",
                        "speech": ["commentary introduces the players"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "sports broadcast with the same field view",
                        "scene": "stadium broadcast",
                        "speech": ["commentary describes a goal"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "players", "to": "goal"},
                        "quality": {"visual_delta_strength": 0.1, "visual_context_similarity": 0.9},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_b_line_audio_only_pair.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "difference_type": "speech",
                    "b_subtype": "speech_topic",
                    "reference_audio_content": "commentary introduces the players",
                    "target_audio_content": "commentary describes a goal",
                    "edit_text": "change the commentary from introducing the players to describing the goal",
                    "audio_difference_specific": True,
                    "edit_text_audio_only": True,
                    "confidence": 0.91,
                    "evidence": ["reference introduces players", "target describes a goal"],
                },
                {"raw": "proposal"},
            )
            client.verify_b_line_audio_only_edit.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "audio_difference_specific": True,
                    "edit_text_audio_only": True,
                    "confidence": 0.92,
                    "evidence": ["target audio describes a goal"],
                },
                {"raw": "audio_verify"},
            )
            client.verify_b_line_full_av_consistency.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "visual_context_preserved": True,
                    "visual_shortcut_risk": False,
                    "audio_edit_still_valid": True,
                    "confidence": 0.85,
                    "evidence": ["full videos keep the same broadcast context"],
                },
                {"raw": "full_av"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client), \
                mock.patch(
                    "app.composed_data._extract_audio_only_cache",
                    side_effect=[root / "clips" / "ref.wav", root / "clips" / "tgt.wav"],
                ):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="b_audio_blind_review",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual("audio_only_blind_review", ranked[0]["candidate_stage"])
            self.assertEqual("change the commentary from introducing the players to describing the goal", ranked[0]["edit_text"])
            self.assertTrue(ranked[0]["audio_only_accept"])
            self.assertTrue(ranked[0]["full_av_consistency_accept"])
            client.propose_single_source_pair.assert_not_called()

    def test_b_audio_blind_review_rejects_full_av_visual_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4", "ref.wav", "tgt.wav"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"media")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {"clip_id": "seg_1", "output_path": "clips/seg_1.mp4", "summary": "same host in a tutorial", "scene": "kitchen tutorial", "speech": ["ingredients"], "modalities": ["audio", "visual"]},
                    {"clip_id": "seg_2", "output_path": "clips/seg_2.mp4", "summary": "same host in a tutorial", "scene": "kitchen tutorial", "speech": ["mixing"], "modalities": ["audio", "visual"]},
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "ingredients", "to": "mixing"},
                        "quality": {"visual_delta_strength": 0.1, "visual_context_similarity": 0.9},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_b_line_audio_only_pair.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "difference_type": "speech",
                    "b_subtype": "speech_topic",
                    "reference_audio_content": "ingredients",
                    "target_audio_content": "mixing",
                    "edit_text": "change the speech from discussing ingredients to discussing mixing",
                    "audio_difference_specific": True,
                    "edit_text_audio_only": True,
                    "confidence": 0.9,
                    "evidence": ["speech differs"],
                },
                {},
            )
            client.verify_b_line_audio_only_edit.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "audio_difference_specific": True,
                    "edit_text_audio_only": True,
                    "confidence": 0.9,
                    "evidence": ["target discusses mixing"],
                },
                {},
            )
            client.verify_b_line_full_av_consistency.return_value = (
                {
                    "accept": False,
                    "reject_reason": "target is identifiable from visible step change",
                    "visual_context_preserved": True,
                    "visual_shortcut_risk": True,
                    "audio_edit_still_valid": True,
                    "confidence": 0.8,
                    "evidence": ["visual cooking step reveals target"],
                },
                {},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client), \
                mock.patch(
                    "app.composed_data._extract_audio_only_cache",
                    side_effect=[root / "clips" / "ref.wav", root / "clips" / "tgt.wav"],
                ):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="b_audio_blind_review",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(0, summary["accepted_count"])
            self.assertIn("visual_shortcut_risk", ranked[0]["judge"]["reject_reason"])

    def test_b_audio_blind_review_v2_accepts_delta_then_video_only_checked_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4", "ref.wav", "tgt.wav", "ref_silent.mp4", "tgt_silent.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"media")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "same sports broadcast field view",
                        "scene": "stadium broadcast",
                        "speech": ["commentary introduces the players"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "same sports broadcast field view",
                        "scene": "stadium broadcast",
                        "speech": ["commentary describes a goal"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "players", "to": "goal"},
                        "quality": {"visual_delta_strength": 0.1, "visual_context_similarity": 0.9},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.analyze_b_line_audio_delta.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "audio_delta_type": "speech",
                    "b_subtype": "speech_topic",
                    "audio_delta_strength": 0.86,
                    "reference_audio_content": "commentary introducing the players",
                    "target_audio_content": "commentary describing a goal",
                    "audio_difference_specific": True,
                    "confidence": 0.91,
                    "evidence": ["reference introduces players", "target describes a goal"],
                },
                {"raw": "delta"},
            )
            client.generate_b_line_audio_edit_text.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "edit_text": "change the commentary from introducing the players to describing the goal",
                    "edit_text_audio_only": True,
                    "edit_text_specificity_score": 0.92,
                    "confidence": 0.9,
                    "evidence": ["specific speech-topic delta"],
                },
                {"raw": "edit"},
            )
            client.verify_b_line_audio_only_edit.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "audio_difference_specific": True,
                    "edit_text_audio_only": True,
                    "confidence": 0.92,
                    "evidence": ["target audio describes a goal"],
                },
                {"raw": "audio_verify"},
            )
            client.verify_b_line_video_only_shortcut.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "visual_context_preserved": True,
                    "visual_shortcut_risk": False,
                    "can_identify_target_without_audio": False,
                    "confidence": 0.86,
                    "evidence": ["silent videos preserve the same broadcast view"],
                },
                {"raw": "video_only"},
            )
            client.verify_b_line_full_av_consistency.return_value = (
                {
                    "accept": True,
                    "reject_reason": "",
                    "visual_context_preserved": True,
                    "visual_shortcut_risk": False,
                    "audio_edit_still_valid": True,
                    "confidence": 0.85,
                    "evidence": ["full videos keep the same broadcast context"],
                },
                {"raw": "full_av"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client), \
                mock.patch(
                    "app.composed_data._extract_audio_only_cache",
                    side_effect=[root / "clips" / "ref.wav", root / "clips" / "tgt.wav"],
                ), \
                mock.patch(
                    "app.composed_data._extract_video_only_cache",
                    side_effect=[root / "clips" / "ref_silent.mp4", root / "clips" / "tgt_silent.mp4"],
                ):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="b_audio_blind_review_v2",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual("change the commentary from introducing the players to describing the goal", ranked[0]["edit_text"])
            self.assertEqual(0.86, ranked[0]["audio_delta_strength"])
            self.assertFalse(ranked[0]["video_only_shortcut_risk"])
            self.assertEqual("speech", ranked[0]["difference"]["type"])
            client.propose_b_line_audio_only_pair.assert_not_called()
            client.propose_single_source_pair.assert_not_called()

    def test_b_audio_blind_review_v2_rejects_weak_audio_delta_and_video_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4", "ref.wav", "tgt.wav", "ref_silent.mp4", "tgt_silent.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"media")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {"clip_id": "seg_1", "output_path": "clips/seg_1.mp4", "summary": "same host holding a card", "scene": "desk demo", "speech": ["same talk"], "modalities": ["audio", "visual"]},
                    {"clip_id": "seg_2", "output_path": "clips/seg_2.mp4", "summary": "same host flips the card back", "scene": "desk demo", "speech": ["same talk"], "modalities": ["audio", "visual"]},
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "same talk", "to": "same talk"},
                        "quality": {"visual_delta_strength": 0.1, "visual_context_similarity": 0.9},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.analyze_b_line_audio_delta.return_value = (
                {
                    "accept": False,
                    "reject_reason": "audio difference is weak",
                    "audio_delta_type": "speech",
                    "b_subtype": "speech_topic",
                    "audio_delta_strength": 0.2,
                    "reference_audio_content": "same talk",
                    "target_audio_content": "same talk",
                    "audio_difference_specific": False,
                    "confidence": 0.4,
                    "evidence": ["both clips contain similar speech"],
                },
                {"raw": "delta"},
            )
            client.generate_b_line_audio_edit_text.return_value = (
                {
                    "accept": False,
                    "reject_reason": "no specific audio edit",
                    "edit_text": "change the speech from discussing the card front to discussing the card back",
                    "edit_text_audio_only": False,
                    "edit_text_specificity_score": 0.2,
                    "confidence": 0.3,
                    "evidence": [],
                },
                {"raw": "edit"},
            )
            client.verify_b_line_audio_only_edit.return_value = (
                {
                    "accept": False,
                    "reject_reason": "reference also satisfies edit",
                    "reference_satisfies_edit": True,
                    "target_satisfies_edit": False,
                    "audio_difference_specific": False,
                    "edit_text_audio_only": False,
                    "confidence": 0.3,
                    "evidence": [],
                },
                {},
            )
            client.verify_b_line_video_only_shortcut.return_value = (
                {
                    "accept": False,
                    "reject_reason": "card side reveals the target",
                    "visual_context_preserved": True,
                    "visual_shortcut_risk": True,
                    "can_identify_target_without_audio": True,
                    "confidence": 0.9,
                    "evidence": ["silent target shows the card back"],
                },
                {},
            )
            client.verify_b_line_full_av_consistency.return_value = (
                {
                    "accept": False,
                    "reject_reason": "visual shortcut",
                    "visual_context_preserved": True,
                    "visual_shortcut_risk": True,
                    "audio_edit_still_valid": False,
                    "confidence": 0.8,
                    "evidence": ["visual change is enough"],
                },
                {},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client), \
                mock.patch(
                    "app.composed_data._extract_audio_only_cache",
                    side_effect=[root / "clips" / "ref.wav", root / "clips" / "tgt.wav"],
                ), \
                mock.patch(
                    "app.composed_data._extract_video_only_cache",
                    side_effect=[root / "clips" / "ref_silent.mp4", root / "clips" / "tgt_silent.mp4"],
                ):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="b_audio_blind_review_v2",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(0, summary["accepted_count"])
            reject_reason = ranked[0]["judge"]["reject_reason"]
            self.assertIn("audio_delta_strength_below_threshold", reject_reason)
            self.assertIn("video_only_shortcut_risk", reject_reason)
            self.assertIn("visual wording card", reject_reason)

    def test_speech_audio_content_line_rewrites_visual_leaky_speech_edit_before_final_omni(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "same man at a desk discusses budgets",
                        "speech": ["budget planning"],
                        "speakers_and_transcript": ["speaker: budget planning"],
                        "audio_events": [],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "same man at a desk discusses health",
                        "speech": ["health policy"],
                        "speakers_and_transcript": ["speaker: health policy"],
                        "audio_events": [],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "man talking about budgets", "to": "man talking about health"},
                        "quality": {
                            "speech_transcript_backed": 1.0,
                            "speech_evidence_score": 0.9,
                            "speech_specificity_score": 0.9,
                            "has_audio_modality": 1.0,
                        },
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the man talking about budgets to the man talking about health",
                    "modalities": ["audio"],
                    "reference_caption": "same man at a desk discusses budgets",
                    "target_caption": "same man at a desk discusses health",
                    "difference": {
                        "type": "speech",
                        "from": "man talking about budgets",
                        "to": "man talking about health",
                        "description": "spoken topic changes",
                    },
                    "dominant_delta": {"type": "speech", "from": "budgets", "to": "health", "reason": "speech topic differs"},
                    "reference_state": {"main_speaker": "man", "inset_subjects": [], "product_overlay": "", "composition": "desk shot", "internal_transitions": []},
                    "target_state": {"main_speaker": "man", "inset_subjects": [], "product_overlay": "", "composition": "desk shot", "internal_transitions": []},
                    "delta_temporal_extent": {"reference": "budget is spoken", "target": "health is spoken", "target_coverage": 0.9, "evidence": "target speech says health"},
                    "subject_roles": {"main_speaker": "man", "inset_subjects": [], "product_overlay": ""},
                    "is_segment_wide_delta": True,
                    "discarded_deltas": ["minor pose change"],
                    "evidence": ["speech content changes"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.9,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": True,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["final verifier confirms the audible speech topic changes"],
                    "recommended_edit_text": "",
                    "audio_primary": True,
                    "visual_locked": True,
                    "visual_too_different_for_B": False,
                    "edit_text_audio_only": True,
                },
                {"raw": "final"},
            )
            client.refine_b_line_edit_text.return_value = (
                {
                    "refined_edit_text": "change the speech from discussing sports to discussing weather",
                    "edit_text_specificity_score": 0.9,
                    "reject_if_unspecific": False,
                    "edit_text_reject_reason": "",
                    "speech_or_audio_evidence": ["reference discusses sports", "target discusses weather"],
                },
                {"raw": "refine"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="exploration",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            final_call = client.verify_single_source_pair_final.call_args.kwargs
            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual("change the speech from discussing budgets to discussing health", ranked[0]["edit_text"])
            self.assertEqual("change the speech from discussing budgets to discussing health", final_call["model_fields"]["edit_text"])
            self.assertTrue(ranked[0]["raw_model_output"])
            self.assertEqual(
                "change the man talking about budgets to the man talking about health",
                ranked[0]["b_line_original_edit_text"],
            )

    def test_visual_audio_anchor_line_final_omni_can_rescue_visual_strength_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "news anchor in studio",
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "flood aerial footage while the news narration continues",
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "a1",
                        "proposal_id": "a1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "scene", "from": "studio anchor", "to": "flood aerial"},
                        "quality": {
                            "audio_line_quality_profile": "v4_strict",
                            "visual_delta_strength": 0.2,
                            "audio_anchor_score": 0.95,
                        },
                        "audio_dataset_line": "visual_audio_anchor",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the scene from a studio anchor shot to flood aerial footage",
                    "modalities": ["visual"],
                    "reference_caption": "news anchor in studio",
                    "target_caption": "flood aerial footage",
                    "difference": {"type": "scene", "from": "studio anchor", "to": "flood aerial", "description": "the visual scene changes"},
                    "dominant_delta": {"type": "scene", "from": "studio anchor", "to": "flood aerial", "reason": "large visual scene change"},
                    "reference_state": {"main_speaker": "anchor", "inset_subjects": [], "product_overlay": "", "composition": "studio", "internal_transitions": []},
                    "target_state": {"main_speaker": "", "inset_subjects": [], "product_overlay": "", "composition": "flood aerial", "internal_transitions": []},
                    "delta_temporal_extent": {"reference": "studio shot", "target": "flood aerial", "target_coverage": 0.9, "evidence": "target shows flood aerial footage"},
                    "subject_roles": {"main_speaker": "anchor", "inset_subjects": [], "product_overlay": ""},
                    "is_segment_wide_delta": True,
                    "discarded_deltas": [],
                    "evidence": ["target changes to flood aerial footage"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.9,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": True,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["final verifier sees a large visual scene change under continuous narration"],
                    "recommended_edit_text": "",
                    "large_visual_delta": True,
                    "audio_context_preserved": True,
                },
                {"raw": "final"},
            )
            client.refine_b_line_edit_text.return_value = (
                {
                    "refined_edit_text": "change the speech from discussing sports to discussing weather",
                    "edit_text_specificity_score": 0.9,
                    "reject_if_unspecific": False,
                    "edit_text_reject_reason": "",
                    "speech_or_audio_evidence": ["reference discusses sports", "target discusses weather"],
                },
                {"raw": "refine"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="audio_matters",
                    audio_dataset_line="visual_audio_anchor",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertTrue(ranked[0]["final_omni_accept"])
            self.assertFalse(ranked[0]["local_gate_passed"])
            self.assertEqual([], ranked[0]["single_source_pair_acceptance_issues"])

    def test_speech_audio_content_line_rejects_visual_difference_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "speaker at podium",
                        "speech": ["budget update"],
                        "speakers_and_transcript": ["speaker: budget update"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "speaker at podium",
                        "speech": ["health update"],
                        "speakers_and_transcript": ["speaker: health update"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "budget", "to": "health"},
                        "quality": {"audio_line_quality_profile": "v4_strict", "visual_context_similarity": 0.9, "visual_delta_strength": 0.1},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the shot from a podium view to a wider room view",
                    "modalities": ["visual"],
                    "reference_caption": "speaker at podium",
                    "target_caption": "speaker at podium",
                    "difference": {"type": "scene", "from": "podium view", "to": "wide room view", "description": "visual framing changes"},
                    "dominant_delta": {"type": "scene", "from": "podium view", "to": "wide room view", "reason": "model chose a visual change"},
                    "reference_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "podium", "internal_transitions": []},
                    "target_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "wide room", "internal_transitions": []},
                    "delta_temporal_extent": {"reference": "podium", "target": "wide room", "target_coverage": 0.9, "evidence": "visual framing"},
                    "subject_roles": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": ""},
                    "is_segment_wide_delta": True,
                    "discarded_deltas": [],
                    "evidence": ["visual framing changed"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="exploration",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(0, summary["accepted_count"])
            self.assertIn("scene is not allowed for speech_audio_content", ranked[0]["judge"]["reject_reason"])
            client.verify_single_source_pair_final.assert_not_called()

    def test_speech_audio_content_line_requires_final_omni_audio_primary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "same podium speaker",
                        "speech": ["budget update"],
                        "speakers_and_transcript": ["speaker: budget update"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "same podium speaker",
                        "speech": ["health update"],
                        "speakers_and_transcript": ["speaker: health update"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "budget", "to": "health"},
                        "quality": {"audio_line_quality_profile": "v4_strict", "visual_context_similarity": 0.9, "visual_delta_strength": 0.1},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the speech from discussing the budget to discussing health",
                    "modalities": ["audio"],
                    "reference_caption": "same podium speaker discusses budget",
                    "target_caption": "same podium speaker discusses health",
                    "difference": {"type": "speech", "from": "budget", "to": "health", "description": "spoken content changes"},
                    "dominant_delta": {"type": "speech", "from": "budget", "to": "health", "reason": "transcripts differ"},
                    "reference_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "podium", "internal_transitions": []},
                    "target_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "podium", "internal_transitions": []},
                    "delta_temporal_extent": {"reference": "budget is spoken", "target": "health is spoken", "target_coverage": 0.9, "evidence": "target transcript says health"},
                    "subject_roles": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": ""},
                    "is_segment_wide_delta": True,
                    "discarded_deltas": [],
                    "evidence": ["speech content changes"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.9,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": True,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["the speech changes, but the model does not mark audio as primary"],
                    "recommended_edit_text": "",
                    "audio_primary": False,
                    "visual_locked": True,
                    "visual_too_different_for_B": False,
                    "edit_text_audio_only": True,
                },
                {"raw": "final"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="exploration",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(0, summary["accepted_count"])
            self.assertIn("final_omni_audio_not_primary", ranked[0]["judge"]["reject_reason"])

    def test_b_audio_review_accepts_partial_clip_audio_delta_for_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "same host in the studio",
                        "speech": ["sports update"],
                        "speakers_and_transcript": ["host: sports update"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "same host in the studio",
                        "speech": ["weather update"],
                        "speakers_and_transcript": ["host: weather update"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "sports update", "to": "weather update"},
                        "quality": {"speech_transcript_backed": 1.0, "speech_evidence_score": 0.9, "speech_specificity_score": 0.9, "has_audio_modality": 1.0},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the speech from discussing sports to discussing weather",
                    "modalities": ["audio"],
                    "reference_caption": "same host in studio discusses sports",
                    "target_caption": "same host in studio discusses weather",
                    "difference": {"type": "speech", "from": "sports", "to": "weather", "description": "spoken topic changes"},
                    "dominant_delta": {"type": "speech", "from": "sports", "to": "weather", "reason": "speech topic differs"},
                    "reference_state": {"main_speaker": "host", "inset_subjects": [], "product_overlay": "", "composition": "studio host", "internal_transitions": []},
                    "target_state": {"main_speaker": "host", "inset_subjects": [], "product_overlay": "", "composition": "studio host", "internal_transitions": []},
                    "delta_temporal_extent": {"reference": "sports is spoken", "target": "weather is spoken", "target_coverage": 0.4, "evidence": "target includes a weather sentence"},
                    "subject_roles": {"main_speaker": "host", "inset_subjects": [], "product_overlay": ""},
                    "is_segment_wide_delta": False,
                    "discarded_deltas": ["minor hand movement"],
                    "evidence": ["target speech includes weather"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.65,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": False,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["the target contains the requested weather speech, but not for the entire clip"],
                    "recommended_edit_text": "",
                    "audio_primary": True,
                    "visual_locked": True,
                    "visual_too_different_for_B": False,
                    "edit_text_audio_only": True,
                },
                {"raw": "final"},
            )
            client.refine_b_line_edit_text.return_value = (
                {
                    "refined_edit_text": "change the speech from discussing sports to discussing weather",
                    "edit_text_specificity_score": 0.9,
                    "reject_if_unspecific": False,
                    "edit_text_reject_reason": "",
                    "speech_or_audio_evidence": ["reference discusses sports", "target discusses weather"],
                },
                {"raw": "refine"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="b_audio_review",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual([], ranked[0]["single_source_pair_acceptance_issues"])
            self.assertIn("final_omni_delta_not_segment_wide", ranked[0]["single_source_pair_review_required"])

    def test_b_audio_review_rescues_placeholder_speech_edit_with_speech_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {"clip_id": "seg_1", "output_path": "clips/seg_1.mp4", "summary": "same speaker", "speech": ["speaker talks"], "speakers_and_transcript": ["speaker talks"], "audio_events": ["speech"], "modalities": ["audio", "visual"]},
                    {"clip_id": "seg_2", "output_path": "clips/seg_2.mp4", "summary": "same speaker", "speech": ["speaker talks"], "speakers_and_transcript": ["speaker talks"], "audio_events": ["speech"], "modalities": ["audio", "visual"]},
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "speech", "to": "speech"},
                        "quality": {"speech_transcript_backed": 1.0, "speech_evidence_score": 0.9, "speech_specificity_score": 0.8, "has_audio_modality": 1.0},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the speech from discussing A to discussing B",
                    "modalities": ["audio"],
                    "reference_caption": "same speaker talks",
                    "target_caption": "same speaker talks",
                    "difference": {"type": "speech", "from": "A", "to": "B", "description": "spoken content changes"},
                    "dominant_delta": {"type": "speech", "from": "A", "to": "B", "reason": "speech topic differs"},
                    "reference_state": {},
                    "target_state": {},
                    "delta_temporal_extent": {"reference": "speech", "target": "speech", "target_coverage": 0.4, "evidence": "speech changes"},
                    "subject_roles": {},
                    "is_segment_wide_delta": False,
                    "discarded_deltas": [],
                    "evidence": ["speech content changes"],
                    "confidence": 0.72,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.8,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": False,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["target has the requested speech change"],
                    "recommended_edit_text": "",
                    "audio_primary": True,
                    "visual_locked": True,
                    "visual_too_different_for_B": False,
                    "edit_text_audio_only": True,
                },
                {"raw": "final"},
            )
            client.refine_b_line_edit_text.return_value = (
                {
                    "refined_edit_text": "change the speech from discussing A to discussing B",
                    "edit_text_specificity_score": 0.95,
                    "reject_if_unspecific": False,
                    "edit_text_reject_reason": "",
                    "speech_or_audio_evidence": ["speech changes"],
                },
                {"raw": "refine"},
            )
            client.refine_b_line_speech_content.return_value = (
                {
                    "reference_speech_content": "budget planning",
                    "target_speech_content": "health services",
                    "speech_transcription_confidence": 0.88,
                    "speech_language": "English",
                    "refined_edit_text": "change the speech from discussing budget planning to discussing health services",
                    "reject_if_still_unclear": False,
                    "speech_rewrite_reject_reason": "",
                },
                {"raw": "rewrite"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="b_audio_review",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertTrue(ranked[0]["speech_rewrite_used"])
            self.assertEqual(
                "change the speech from discussing budget planning to discussing health services",
                ranked[0]["edit_text"],
            )

    def test_b_audio_review_rejects_placeholder_speech_when_rewrite_is_unclear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {"clip_id": "seg_1", "output_path": "clips/seg_1.mp4", "summary": "same speaker", "speech": ["speech"], "speakers_and_transcript": ["speech"], "audio_events": ["speech"], "modalities": ["audio", "visual"]},
                    {"clip_id": "seg_2", "output_path": "clips/seg_2.mp4", "summary": "same speaker", "speech": ["speech"], "speakers_and_transcript": ["speech"], "audio_events": ["speech"], "modalities": ["audio", "visual"]},
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "speech", "to": "speech"},
                        "quality": {"speech_transcript_backed": 1.0, "speech_evidence_score": 0.9, "speech_specificity_score": 0.8, "has_audio_modality": 1.0},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the speech from discussing A to discussing B",
                    "modalities": ["audio"],
                    "reference_caption": "speaker talks",
                    "target_caption": "speaker talks",
                    "difference": {"type": "speech", "from": "A", "to": "B", "description": "spoken content changes"},
                    "dominant_delta": {"type": "speech", "from": "A", "to": "B", "reason": "speech topic differs"},
                    "reference_state": {},
                    "target_state": {},
                    "delta_temporal_extent": {"reference": "speech", "target": "speech", "target_coverage": 0.4, "evidence": "speech changes"},
                    "subject_roles": {},
                    "is_segment_wide_delta": False,
                    "discarded_deltas": [],
                    "evidence": ["speech content changes"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.8,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": False,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["target has the requested speech change"],
                    "recommended_edit_text": "",
                    "audio_primary": True,
                    "visual_locked": True,
                    "visual_too_different_for_B": False,
                    "edit_text_audio_only": True,
                },
                {"raw": "final"},
            )
            client.refine_b_line_edit_text.return_value = (
                {
                    "refined_edit_text": "change the speech from discussing A to discussing B",
                    "edit_text_specificity_score": 0.95,
                    "reject_if_unspecific": False,
                    "edit_text_reject_reason": "",
                    "speech_or_audio_evidence": ["speech changes"],
                },
                {"raw": "refine"},
            )
            client.refine_b_line_speech_content.return_value = (
                {
                    "reference_speech_content": "not clear enough",
                    "target_speech_content": "not clear enough",
                    "speech_transcription_confidence": 0.3,
                    "speech_language": "",
                    "refined_edit_text": "",
                    "reject_if_still_unclear": True,
                    "speech_rewrite_reject_reason": "speech is not clear enough",
                },
                {"raw": "rewrite"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="b_audio_review",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(0, summary["accepted_count"])
            self.assertIn("speech_rewrite_reject", ranked[0]["judge"]["reject_reason"])

    def test_b_audio_review_still_rejects_low_quality_visual_or_hollow_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {"clip_id": "seg_1", "output_path": "clips/seg_1.mp4", "summary": "speaker", "speech": ["budget"], "speakers_and_transcript": ["speaker: budget"], "audio_events": ["speech"], "modalities": ["audio", "visual"]},
                    {"clip_id": "seg_2", "output_path": "clips/seg_2.mp4", "summary": "speaker", "speech": ["budget"], "speakers_and_transcript": ["speaker: budget"], "audio_events": ["speech"], "modalities": ["audio", "visual"]},
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "budget", "to": "budget"},
                        "quality": {"speech_transcript_backed": 1.0, "speech_evidence_score": 0.9, "speech_specificity_score": 0.9, "has_audio_modality": 1.0},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the speech from discussing budget to discussing budget",
                    "modalities": ["audio"],
                    "reference_caption": "speaker says budget",
                    "target_caption": "speaker says budget",
                    "difference": {"type": "speech", "from": "budget", "to": "budget", "description": "same speech"},
                    "dominant_delta": {"type": "speech", "from": "budget", "to": "budget", "reason": "same speech"},
                    "reference_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "podium", "internal_transitions": []},
                    "target_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "podium", "internal_transitions": []},
                    "delta_temporal_extent": {"reference": "budget", "target": "budget", "target_coverage": 0.9, "evidence": "same speech"},
                    "subject_roles": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": ""},
                    "is_segment_wide_delta": True,
                    "discarded_deltas": [],
                    "evidence": ["same speech"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.9,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": True,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["even a permissive final verifier cannot rescue identical endpoints"],
                    "recommended_edit_text": "",
                    "audio_primary": True,
                    "visual_locked": True,
                    "visual_too_different_for_B": False,
                    "edit_text_audio_only": True,
                },
                {"raw": "final"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="b_audio_review",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(0, summary["accepted_count"])
            self.assertIn("edit_text_not_audio_only: identical audio endpoints", ranked[0]["judge"]["reject_reason"])

    def test_speech_audio_content_line_final_omni_can_rescue_local_visual_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "same podium speaker",
                        "speech": ["budget update"],
                        "speakers_and_transcript": ["speaker: budget update"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "same podium speaker",
                        "speech": ["health update"],
                        "speakers_and_transcript": ["speaker: health update"],
                        "audio_events": ["speech"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "c1",
                        "proposal_id": "p1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "budget", "to": "health"},
                        "quality": {
                            "audio_line_quality_profile": "v4_strict",
                            "visual_context_similarity": 0.2,
                            "visual_delta_strength": 0.8,
                        },
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.return_value = (
                {
                    "edit_text": "change the speech from discussing the budget to discussing health",
                    "modalities": ["audio"],
                    "reference_caption": "podium speaker discusses budget",
                    "target_caption": "podium speaker discusses health",
                    "difference": {"type": "speech", "from": "budget", "to": "health", "description": "spoken content changes"},
                    "dominant_delta": {"type": "speech", "from": "budget", "to": "health", "reason": "transcripts differ"},
                    "reference_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "podium", "internal_transitions": []},
                    "target_state": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": "", "composition": "podium", "internal_transitions": []},
                    "delta_temporal_extent": {"reference": "budget is spoken", "target": "health is spoken", "target_coverage": 0.9, "evidence": "target transcript says health"},
                    "subject_roles": {"main_speaker": "speaker", "inset_subjects": [], "product_overlay": ""},
                    "is_segment_wide_delta": True,
                    "discarded_deltas": [],
                    "evidence": ["speech content changes"],
                    "confidence": 0.9,
                    "accept": True,
                    "reject_reason": "",
                },
                {"raw": "ok"},
            )
            client.verify_single_source_pair_final.return_value = (
                {
                    "accept": True,
                    "confidence": 0.9,
                    "quality_score": 0.9,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "observable_delta": True,
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": True,
                    "edit_text_accurate": True,
                    "main_reject_reason": "",
                    "evidence": ["final verifier confirms the same visual context and the spoken topic change"],
                    "recommended_edit_text": "",
                    "audio_primary": True,
                    "visual_locked": True,
                    "visual_too_different_for_B": False,
                    "edit_text_audio_only": True,
                },
                {"raw": "final"},
            )

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                summary = propose_single_source_pairs(
                    root=root,
                    clip_annotations_path=annotations_path,
                    pair_candidates_path=candidates_path,
                    output_path=root / "pairs" / "ranked.jsonl",
                    accepted_output_path=root / "pairs" / "accepted.jsonl",
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                    acceptance_profile="exploration",
                    audio_dataset_line="speech_audio_content",
                )

            ranked = [json.loads(line) for line in (root / "pairs" / "ranked.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertTrue(ranked[0]["final_omni_accept"])
            self.assertFalse(ranked[0]["local_gate_passed"])
            self.assertEqual([], ranked[0]["single_source_pair_acceptance_issues"])

    def test_propose_single_source_pairs_does_not_cache_transient_omni_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            for name in ("seg_1.mp4", "seg_2.mp4"):
                path = root / "clips" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            annotations_path = root / "captions" / "single_source_annotations.jsonl"
            candidates_path = root / "pairs" / "single_source_pair_candidates.jsonl"
            ranked_path = root / "pairs" / "ranked.jsonl"
            self._write_jsonl(
                annotations_path,
                [
                    {
                        "clip_id": "seg_1",
                        "output_path": "clips/seg_1.mp4",
                        "summary": "cricket match with commentary",
                        "speech": ["commentary about the first batter"],
                        "speakers_and_transcript": ["commentator: first batter"],
                        "audio_events": ["commentary"],
                        "modalities": ["audio", "visual"],
                    },
                    {
                        "clip_id": "seg_2",
                        "output_path": "clips/seg_2.mp4",
                        "summary": "cricket match with crowd cheering",
                        "speech": ["commentary about the second batter"],
                        "speakers_and_transcript": ["commentator: second batter"],
                        "audio_events": ["crowd cheering", "commentary"],
                        "modalities": ["audio", "visual"],
                    },
                ],
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        "candidate_id": "b1",
                        "proposal_id": "b1",
                        "reference_clip_id": "seg_1",
                        "target_clip_id": "seg_2",
                        "reference_video": "clips/seg_1.mp4",
                        "target_video": "clips/seg_2.mp4",
                        "difference": {"type": "speech", "from": "first batter", "to": "second batter"},
                        "quality": {"audio_line_quality_profile": "v4_strict", "visual_context_similarity": 0.5},
                        "audio_dataset_line": "speech_audio_content",
                    }
                ],
            )
            client = mock.Mock()
            client.propose_single_source_pair.side_effect = ConnectionRefusedError("Connection refused")

            with mock.patch("app.composed_data.OpenAIComposedDataClient", return_value=client):
                with self.assertRaises(ConnectionRefusedError):
                    propose_single_source_pairs(
                        root=root,
                        clip_annotations_path=annotations_path,
                        pair_candidates_path=candidates_path,
                        output_path=ranked_path,
                        accepted_output_path=root / "pairs" / "accepted.jsonl",
                        base_url="http://127.0.0.1:8093/v1",
                        api_key="EMPTY",
                        model="qwen3-omni",
                        acceptance_profile="exploration",
                        audio_dataset_line="speech_audio_content",
                        omni_retries=1,
                        fail_on_transient_omni_errors=True,
                    )

            ranked_rows = [line for line in ranked_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([], ranked_rows)

    def test_merge_line_results_uses_progress_when_ranked_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "runs" / "audio_lines"
            self._write_jsonl(
                run_root / "a_shards" / "accepted_progress_01.jsonl",
                [
                    {
                        "proposal_id": "a1",
                        "accepted": True,
                        "edit_text": "add a red sign",
                        "reference_clip_id": "a_ref",
                        "target_clip_id": "a_target",
                    }
                ],
            )
            self._write_jsonl(
                run_root / "a_shards" / "rejected_progress_01.jsonl",
                [{"proposal_id": "a2", "accepted": False, "judge": {"reject_reason": "weak visual delta"}}],
            )
            self._write_jsonl(
                run_root / "b_shards" / "accepted_progress_01.jsonl",
                [
                    {
                        "proposal_id": "b1",
                        "accepted": True,
                        "edit_text": "change the spoken sentence",
                        "reference_clip_id": "b_ref",
                        "target_clip_id": "b_target",
                    }
                ],
            )

            summary = merge_line_results(run_root=run_root, target_a_count=8, target_b_count=8)

            a_exported = [json.loads(line) for line in (run_root / "a_visual_audio_anchor_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            b_exported = [json.loads(line) for line in (run_root / "b_speech_audio_content_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(2, summary["a_ranked_count"])
            self.assertEqual(1, summary["b_ranked_count"])
            self.assertEqual(["a1"], [record["proposal_id"] for record in a_exported])
            self.assertEqual(["b1"], [record["proposal_id"] for record in b_exported])

    def test_inverse_edit_text_generation_is_edit_type_aware(self) -> None:
        self.assertEqual(
            "change the speech from discussing the mayor's remarks to discussing the bakery opening",
            _inverse_b_line_edit_text("change the speech from discussing the bakery opening to discussing the mayor's remarks")["edit_text"],
        )
        self.assertEqual(
            "remove crowd cheering from the audio",
            _inverse_b_line_edit_text("add crowd cheering to the audio")["edit_text"],
        )
        self.assertEqual(
            "replace applause with quiet room ambience",
            _inverse_b_line_edit_text("replace quiet room ambience with applause")["edit_text"],
        )
        self.assertFalse(_inverse_b_line_edit_text("change the speech from discussing speaking to discussing speaking")["ok"])

    def test_augment_b_inverse_accepts_reverified_inverse_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            input_path = run_root / "b_main_audio_cvr_triplets.jsonl"
            self._write_jsonl(
                input_path,
                [
                    {
                        "proposal_id": "b_forward_1",
                        "candidate_id": "c1",
                        "accepted": True,
                        "split_tier": "main",
                        "reference_clip_id": "ref",
                        "target_clip_id": "tgt",
                        "reference_video": str(root / "clips" / "ref.mp4"),
                        "target_video": str(root / "clips" / "tgt.mp4"),
                        "edit_text": "change the speech from discussing the bakery opening to discussing the mayor's remarks",
                        "difference": {"type": "speech"},
                        "audio_only_reference_content": "speech about the bakery opening",
                        "audio_only_target_content": "speech about the mayor's remarks",
                        "b_subtype": "speech_topic_in_video_context",
                        "hard_negatives": ["clips/visual_hard.mp4", "clips/audio_hard.mp4", "clips/asr_hard.mp4"],
                        "quality": {"video_context_strength": 0.8, "asr_degeneracy_risk": 0.2},
                    }
                ],
            )
            client = mock.Mock()
            client.verify_b_line_audio_only_edit.return_value = (
                {
                    "accept": True,
                    "reference_satisfies_edit": False,
                    "target_satisfies_edit": True,
                    "audio_difference_specific": True,
                    "edit_text_audio_only": True,
                    "confidence": 0.91,
                    "evidence": ["the inverse target contains bakery-opening speech"],
                },
                {"raw": "audio"},
            )
            client.verify_b_line_video_only_shortcut.return_value = (
                {
                    "accept": True,
                    "visual_shortcut_risk": False,
                    "can_identify_target_without_audio": False,
                    "visual_context_preserved": True,
                    "confidence": 0.82,
                    "evidence": ["video-only view cannot identify the audio topic"],
                },
                {"raw": "video"},
            )
            client.verify_b_line_full_av_consistency.return_value = (
                {
                    "accept": True,
                    "visual_context_preserved": True,
                    "visual_shortcut_risk": False,
                    "audio_edit_still_valid": True,
                    "confidence": 0.87,
                    "evidence": ["full AV is consistent with the inverse audio edit"],
                },
                {"raw": "full"},
            )

            with mock.patch("app.audio_lines_single_source.OpenAIComposedDataClient", return_value=client), mock.patch(
                "app.audio_lines_single_source._extract_audio_only_cache",
                side_effect=lambda video_path, cache_dir, clip_id: cache_dir / f"{clip_id}.wav",
            ), mock.patch(
                "app.audio_lines_single_source._extract_video_only_cache",
                side_effect=lambda video_path, cache_dir, clip_id: cache_dir / f"{clip_id}.mp4",
            ):
                summary = augment_b_inverse(
                    run_root=run_root,
                    input_path=input_path,
                    root=root,
                    max_records=1,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            accepted = [json.loads(line) for line in (run_root / "b_inverse_accepted.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            train = [json.loads(line) for line in (run_root / "b_train_bidirectional_triplets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual(1, len(accepted))
            self.assertEqual(2, len(train))
            self.assertTrue(accepted[0]["is_inverse"])
            self.assertEqual("tgt", accepted[0]["reference_clip_id"])
            self.assertEqual("ref", accepted[0]["target_clip_id"])
            self.assertEqual(
                "change the speech from discussing the mayor's remarks to discussing the bakery opening",
                accepted[0]["edit_text"],
            )
            self.assertEqual("extended", accepted[0]["split_tier"])
            self.assertFalse(accepted[0]["benchmark_eligible"])
            self.assertTrue(accepted[0]["training_eligible"])
            self.assertEqual("inverse", accepted[0]["direction"])
            self.assertEqual("replace", accepted[0]["edit_type"])
            self.assertEqual("speech_topic", accepted[0]["audio_delta_type"])
            self.assertEqual("the mayor's remarks", accepted[0]["old_audio"])
            self.assertEqual("the bakery opening", accepted[0]["new_audio"])
            self.assertEqual(accepted[0]["pair_group_id"], accepted[0]["inverse_pair_group_id"])
            self.assertIn({"type": "reference_negative", "video": str(root / "clips" / "tgt.mp4")}, accepted[0]["audio_delta_hard_negatives"])
            self.assertEqual("clean_audio_delta", accepted[0]["shortcut_label"])
            self.assertEqual("forward", train[0]["direction"])
            self.assertEqual("inverse", train[1]["direction"])
            self.assertEqual("the bakery opening", train[0]["old_audio"])
            self.assertEqual("the mayor's remarks", train[0]["new_audio"])

    def test_augment_b_inverse_rejects_video_only_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            input_path = run_root / "b_main_audio_cvr_triplets.jsonl"
            self._write_jsonl(
                input_path,
                [
                    {
                        "proposal_id": "b_forward_2",
                        "accepted": True,
                        "reference_clip_id": "ref",
                        "target_clip_id": "tgt",
                        "reference_video": str(root / "ref.mp4"),
                        "target_video": str(root / "tgt.mp4"),
                        "edit_text": "add crowd cheering to the audio",
                        "difference": {"type": "audio_event"},
                    }
                ],
            )
            client = mock.Mock()
            client.verify_b_line_audio_only_edit.return_value = (
                {"accept": True, "reference_satisfies_edit": False, "target_satisfies_edit": True, "audio_difference_specific": True, "edit_text_audio_only": True, "confidence": 0.9},
                {},
            )
            client.verify_b_line_video_only_shortcut.return_value = (
                {"accept": False, "visual_shortcut_risk": True, "can_identify_target_without_audio": True, "visual_context_preserved": True, "confidence": 0.9, "reject_reason": "visual shortcut"},
                {},
            )
            client.verify_b_line_full_av_consistency.return_value = (
                {"accept": True, "visual_context_preserved": True, "visual_shortcut_risk": False, "audio_edit_still_valid": True, "confidence": 0.9},
                {},
            )

            with mock.patch("app.audio_lines_single_source.OpenAIComposedDataClient", return_value=client), mock.patch(
                "app.audio_lines_single_source._extract_audio_only_cache",
                side_effect=lambda video_path, cache_dir, clip_id: cache_dir / f"{clip_id}.wav",
            ), mock.patch(
                "app.audio_lines_single_source._extract_video_only_cache",
                side_effect=lambda video_path, cache_dir, clip_id: cache_dir / f"{clip_id}.mp4",
            ):
                summary = augment_b_inverse(
                    run_root=run_root,
                    input_path=input_path,
                    root=root,
                    max_records=1,
                    base_url="http://127.0.0.1:8093/v1",
                    api_key="EMPTY",
                    model="qwen3-omni",
                )

            rejected = [json.loads(line) for line in (run_root / "b_inverse_rejected.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(0, summary["accepted_count"])
            self.assertEqual(1, len(rejected))
            self.assertIn("inverse_video_only_shortcut_risk", rejected[0]["inverse_reject_reason"])


if __name__ == "__main__":
    unittest.main()
