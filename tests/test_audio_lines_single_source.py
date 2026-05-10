from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.audio_lines_single_source import (
    merge_line_results,
    prepare_existing_single_source_clips,
    split_audio_line_candidates,
)
from app.composed_data import ensure_layout, propose_single_source_pairs


class AudioLinesSingleSourceTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

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

    def test_split_candidates_builds_a_and_b_lines_without_caption_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ensure_layout(root)
            annotations = [
                {
                    "clip_id": "seg_1",
                    "output_path": "clips/seg_1.mp4",
                    "summary": "speaker discusses apples without an overlay",
                    "subjects": ["speaker"],
                    "speech": ["I like apples"],
                    "speakers_and_transcript": ["speaker: I like apples"],
                    "audio_events": [],
                    "modalities": ["audio", "visual"],
                },
                {
                    "clip_id": "seg_2",
                    "output_path": "clips/seg_2.mp4",
                    "summary": "speaker discusses oranges with an overlay",
                    "subjects": ["speaker"],
                    "speech": ["I like oranges"],
                    "speakers_and_transcript": ["speaker: I like oranges"],
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
            self.assertGreaterEqual(b_records[0]["quality"]["visual_context_similarity"], 0.18)

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

            accepted = [json.loads(line) for line in (root / "pairs" / "accepted.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual("speech_audio_content", accepted[0]["audio_dataset_line"])
            self.assertEqual("speech", accepted[0]["difference"]["type"])

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


if __name__ == "__main__":
    unittest.main()
