from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from app.e5_audio_delta_train import (
    DEFAULT_DATA_ROOT,
    DeterministicEncoder,
    EvalGalleryItem,
    build_splits,
    cache_embeddings,
    eval_adapter,
    load_audio_delta_records,
    prepare_omnicvr_records,
    prepare_records,
    run_ablations,
    run_loss_schedule,
    run_stability_grid,
    train_adapter,
    train_lora_plan,
    _batch_whitening_loss,
    _coral_loss,
    _false_negative_weights,
    _gallery_negative_recall_by_type,
    _hardness_weights,
    _import_torch,
    load_eval_gallery_items,
    _modality_tau,
    _multi_positive_loss,
    _quantile_negative_curriculum_weights,
    _scheduled_learning_rate,
    _scheduled_temperature,
    _training_profile_options,
    _AudioDeltaAdapter,
    _document_payload,
    _ensure_pyav_error_compat,
    _query_payload,
    _resolve_media_path,
    _save_embedding_npz_atomic,
    _skippable_media_encoding_error,
    _video_payload,
)
from app.audio_cvr_protocol_eval import mine_local_same_source, summarize_data, summarize_evals


class E5AudioDeltaTrainTests(unittest.TestCase):
    def test_batch_shape_and_ffmpeg_failures_are_isolated_as_media_errors(self) -> None:
        self.assertTrue(
            _skippable_media_encoding_error(
                ValueError("setting an array element with a sequence; detected an inhomogeneous shape")
            )
        )
        self.assertTrue(
            _skippable_media_encoding_error(
                subprocess.CalledProcessError(254, ["ffmpeg", "-i", "broken.mp4"])
            )
        )

    def test_embedding_npz_is_written_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache" / "eval_embeddings.npz"
            _save_embedding_npz_atomic(path, {"query": np.asarray([[1.0, 2.0]], dtype=np.float32)})

            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["query"], np.asarray([[1.0, 2.0]], dtype=np.float32))
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_pyav_error_compat_maps_legacy_name_to_ffmpeg_error(self) -> None:
        class FakeFFmpegError(OSError):
            pass

        fake_av = SimpleNamespace(error=SimpleNamespace(FFmpegError=FakeFFmpegError))
        selected = _ensure_pyav_error_compat(fake_av)

        self.assertIs(FakeFFmpegError, fake_av.AVError)
        self.assertTrue(str(selected).endswith("FakeFFmpegError"))

    def test_checkpoint_prefill_shards_resume_and_retry_transient_decode_failure(self) -> None:
        class CountingEncoder:
            def __init__(self, *, fail_once: bool = False, forbid_calls: bool = False) -> None:
                self.inner = DeterministicEncoder()
                self.fail_once = fail_once
                self.forbid_calls = forbid_calls
                self.calls = 0

            def encode_document(self, inputs):
                if self.forbid_calls:
                    raise AssertionError("all payloads should have been loaded from checkpoints")
                signatures = {
                    ("mapping", *(str(key) for key in sorted(item)))
                    if isinstance(item, dict)
                    else ("scalar", type(item).__name__)
                    for item in inputs
                }
                if len(signatures) != 1:
                    raise AssertionError(f"checkpoint batches must use one input shape: {signatures}")
                self.calls += len(inputs)
                if self.fail_once:
                    self.fail_once = False
                    raise BlockingIOError(11, "Resource temporarily unavailable")
                return self.inner.encode_document(inputs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [self._record("sample_1", source="source_a", pair="pair_a")]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)
            self._write_jsonl(
                records_dir / "eval_gallery.jsonl",
                [
                    {"gallery_id": "positive::sample_1", "video": "/tmp/sample_1_tgt.mp4", "kind": "positive"},
                    {"gallery_id": "reference::sample_1", "video": "/tmp/sample_1_ref.mp4", "kind": "reference_negative"},
                    {"gallery_id": "distractor::1", "video": "/tmp/distractor.mp4", "kind": "distractor"},
                ],
            )
            cache_dir = root / "cache"
            first = CountingEncoder(fail_once=True)
            cache_embeddings(
                records_dir=records_dir,
                output_dir=cache_dir,
                encoder=first,
                skip_train=True,
                checkpoint_embeddings=True,
                checkpoint_prefill_only=True,
                checkpoint_shard_index=0,
                checkpoint_shard_count=2,
                encoding_item_batch_size=3,
                encoding_retries=1,
                encoding_retry_wait_seconds=0,
            )
            second = CountingEncoder()
            cache_embeddings(
                records_dir=records_dir,
                output_dir=cache_dir,
                encoder=second,
                skip_train=True,
                checkpoint_embeddings=True,
                checkpoint_prefill_only=True,
                checkpoint_shard_index=1,
                checkpoint_shard_count=2,
                encoding_item_batch_size=3,
            )
            cache_embeddings(
                records_dir=records_dir,
                output_dir=cache_dir,
                encoder=CountingEncoder(forbid_calls=True),
                skip_train=True,
                checkpoint_embeddings=True,
            )

            self.assertGreater(first.calls, 0)
            self.assertGreater(second.calls, 0)
            self.assertTrue((cache_dir / "eval_embeddings.npz").exists())
            self.assertTrue((cache_dir / "checkpoint_prefill_shard_000_of_002.json").exists())
            self.assertTrue((cache_dir / "checkpoint_prefill_shard_001_of_002.json").exists())

    def test_persistent_gallery_failure_is_audited_and_masked(self) -> None:
        class BadGalleryEncoder:
            def __init__(self) -> None:
                self.inner = DeterministicEncoder()

            def encode_document(self, inputs):
                for item in inputs:
                    if isinstance(item, dict) and "bad_distractor.mp4" in str(item.get("video", "")):
                        raise BlockingIOError(11, "Failed initializing scaling graph")
                    if isinstance(item, dict) and "zero_frame.mp4" in str(item.get("video", "")):
                        raise ZeroDivisionError("division by zero")
                return self.inner.encode_document(inputs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [self._record("sample_1", source="source_a", pair="pair_a")]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)
            self._write_jsonl(
                records_dir / "eval_gallery.jsonl",
                [
                    {"gallery_id": "positive::sample_1", "video": "/tmp/sample_1_tgt.mp4", "kind": "positive"},
                    {"gallery_id": "reference::sample_1", "video": "/tmp/sample_1_ref.mp4", "kind": "reference_negative"},
                    {"gallery_id": "distractor::bad", "video": "/tmp/bad_distractor.mp4", "kind": "candidate"},
                    {"gallery_id": "distractor::zero", "video": "/tmp/zero_frame.mp4", "kind": "candidate"},
                    {"gallery_id": "distractor::good", "video": "/tmp/good_distractor.mp4", "kind": "candidate"},
                ],
            )
            cache_dir = root / "cache"
            failure_dir = root / "shared_failures"
            encoder = BadGalleryEncoder()
            prefill = cache_embeddings(
                records_dir=records_dir,
                output_dir=cache_dir,
                encoder=encoder,
                skip_train=True,
                checkpoint_embeddings=True,
                checkpoint_prefill_only=True,
                encoding_item_batch_size=8,
                encoding_retries=1,
                encoding_retry_wait_seconds=0,
                skip_persistent_encoding_failures=True,
                encoding_failure_dir=failure_dir,
            )
            summary = cache_embeddings(
                records_dir=records_dir,
                output_dir=cache_dir,
                encoder=encoder,
                skip_train=True,
                checkpoint_embeddings=True,
                skip_persistent_encoding_failures=True,
                encoding_failure_dir=failure_dir,
            )

            with np.load(cache_dir / "eval_embeddings.npz", allow_pickle=False) as data:
                self.assertEqual([1.0, 1.0, 0.0, 0.0, 1.0], data["gallery_valid_mask"].tolist())
                self.assertEqual([1.0, 1.0, 0.0, 0.0, 1.0], data["candidate_gallery_mask"][0].tolist())
            self.assertEqual(2, prefill["skipped_persistent_failure_count"])
            self.assertEqual(2, summary["eval"]["failed_gallery_count"])
            self.assertEqual(3, summary["eval"]["effective_gallery_count"])
            self.assertTrue((cache_dir / "eval_gallery_encoding_failures.jsonl").exists())

    def test_required_media_failure_excludes_same_query_across_modality_caches(self) -> None:
        class BadRequiredEncoder:
            def __init__(self) -> None:
                self.inner = DeterministicEncoder()

            def encode_document(self, inputs):
                for item in inputs:
                    if isinstance(item, dict) and "bad_tgt.mp4" in str(item.get("video", "")):
                        raise ZeroDivisionError("division by zero")
                return self.inner.encode_document(inputs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [
                self._record("good", source="source_good", pair="pair_good"),
                self._record("bad", source="source_bad", pair="pair_bad"),
            ]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)
            self._write_jsonl(
                records_dir / "eval_gallery.jsonl",
                [
                    {"gallery_id": "positive::good", "video": "/tmp/good_tgt.mp4", "kind": "positive"},
                    {"gallery_id": "reference::good", "video": "/tmp/good_ref.mp4", "kind": "reference_negative"},
                    {"gallery_id": "positive::bad", "video": "/tmp/bad_tgt.mp4", "kind": "positive"},
                    {"gallery_id": "reference::bad", "video": "/tmp/bad_ref.mp4", "kind": "reference_negative"},
                ],
            )
            failure_dir = root / "shared_failures"
            cache_on = root / "cache_on"
            encoder = BadRequiredEncoder()
            cache_embeddings(
                records_dir=records_dir,
                output_dir=cache_on,
                encoder=encoder,
                skip_train=True,
                checkpoint_embeddings=True,
                checkpoint_prefill_only=True,
                encoding_item_batch_size=8,
                encoding_retries=0,
                skip_persistent_encoding_failures=True,
                encoding_failure_dir=failure_dir,
            )
            summary_on = cache_embeddings(
                records_dir=records_dir,
                output_dir=cache_on,
                encoder=encoder,
                skip_train=True,
                checkpoint_embeddings=True,
                skip_persistent_encoding_failures=True,
                encoding_failure_dir=failure_dir,
            )
            cache_off = root / "cache_off"
            summary_off = cache_embeddings(
                records_dir=records_dir,
                output_dir=cache_off,
                encoder=encoder,
                skip_train=True,
                checkpoint_embeddings=True,
                video_audio_mode="off",
                skip_persistent_encoding_failures=True,
                encoding_failure_dir=failure_dir,
            )

            for cache_dir, summary in ((cache_on, summary_on), (cache_off, summary_off)):
                cached_records = load_audio_delta_records(cache_dir / "eval_records.jsonl")
                self.assertEqual(["good"], [record.sample_id for record in cached_records])
                self.assertEqual(1, summary["eval"]["excluded_record_count"])
                self.assertEqual(["bad"], summary["eval"]["excluded_sample_ids"])
                with np.load(cache_dir / "eval_embeddings.npz", allow_pickle=False) as data:
                    self.assertEqual((1, 4), data["candidate_gallery_mask"].shape)
                    self.assertEqual([1.0, 1.0, 0.0, 1.0], data["gallery_valid_mask"].tolist())
                self.assertTrue((cache_dir / "eval_excluded_encoding_failures.jsonl").exists())

    def test_prepare_omnicvr_and_per_query_reference_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            videos = root / "videos"
            videos.mkdir()
            for video_id in ("a.mp4", "b.mp4", "c.mp4", "d.mp4"):
                (videos / video_id).write_bytes(b"")
            annotations = root / "omnicvr.jsonl"
            self._write_jsonl(
                annotations,
                [
                    {
                        "source_id": "a.mp4",
                        "target_id": "b.mp4",
                        "instruction": "replace the bell with a drum",
                        "candidates": ["a.mp4", "b.mp4", "c.mp4"],
                    },
                    {
                        "source_id": "b.mp4",
                        "target_id": "c.mp4",
                        "instruction": "replace the drum with a whistle",
                        "candidates": ["b.mp4", "c.mp4", "d.mp4"],
                    },
                ],
            )

            summary = prepare_omnicvr_records(
                annotation_path=annotations,
                videos_dir=videos,
                output_dir=root / "records",
                query_count=2,
                expected_gallery_size=3,
            )
            self.assertEqual(2, summary["query_count"])
            self.assertEqual(4, summary["gallery_union_count"])
            self.assertFalse(summary["shared_candidate_pool"])
            eval_rows = [json.loads(line) for line in (root / "records" / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
            self._write_jsonl(root / "records" / "train.jsonl", eval_rows)

            cache_embeddings(records_dir=root / "records", output_dir=root / "cache", mock_encoder=True)
            with np.load(root / "cache" / "eval_embeddings.npz") as data:
                candidate_mask = np.asarray(data["candidate_gallery_mask"], dtype=bool)
                positive = np.asarray(data["positive_gallery_index"], dtype=np.int64)
                reference = np.asarray(data["reference_gallery_index"], dtype=np.int64)
            self.assertEqual((2, 4), candidate_mask.shape)
            self.assertTrue(np.all(candidate_mask.sum(axis=1) == 3))
            self.assertTrue(np.all(candidate_mask[np.arange(2), positive]))
            self.assertTrue(np.all(candidate_mask[np.arange(2), reference]))

            train_adapter(cache_dir=root / "cache", output_dir=root / "adapter", steps=1, batch_size=2, device="cpu")
            eval_summary = eval_adapter(
                cache_dir=root / "cache",
                adapter_dir=root / "adapter",
                output_dir=root / "eval_without_source",
                device="cpu",
                exclude_query_reference=True,
                save_topk=2,
            )
            self.assertEqual(4, eval_summary["gallery_count"])
            self.assertEqual(2, eval_summary["effective_gallery_count"])
            self.assertEqual(2, eval_summary["excluded_query_reference_count"])
            self.assertFalse(eval_summary["reference_in_gallery"])
            topk_rows = [
                json.loads(line)
                for line in (root / "eval_without_source" / "per_query_topk.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(not any(item["is_reference"] for item in row["base_topk"]) for row in topk_rows))
            self.assertTrue(all(not any(item["is_reference"] for item in row["adapter_topk"]) for row in topk_rows))

    def test_prepare_omnicvr_rejects_missing_source_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotations = root / "omnicvr.jsonl"
            self._write_jsonl(
                annotations,
                [{"source_id": "a.mp4", "target_id": "b.mp4", "instruction": "add a bell", "candidates": ["b.mp4"]}],
            )
            with self.assertRaisesRegex(ValueError, "source_id is absent"):
                prepare_omnicvr_records(
                    annotation_path=annotations,
                    videos_dir=root,
                    output_dir=root / "records",
                    query_count=1,
                    expected_gallery_size=1,
                    require_existing_media=False,
                )

    def test_prepare_loads_b_line_tier_outputs_and_preserves_training_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            dataset_run.mkdir()
            self._write_jsonl(
                dataset_run / "b_main_audio_cvr_triplets.jsonl",
                [self._record("main_1", source="source_a", pair="pair_a")],
            )
            self._write_jsonl(
                dataset_run / "b_extended_audio_cvr_triplets.jsonl",
                [self._record("extended_1", source="source_b", pair="pair_b", split_tier="extended")],
            )

            summary = prepare_records(
                run_root=dataset_run,
                output_dir=root / "records",
                max_train_records=2,
                max_eval_records=1,
            )
            records = load_audio_delta_records(root / "records" / "train.jsonl")

            self.assertEqual(2, summary["train_count"])
            self.assertEqual("replace", records[0].edit_type)
            self.assertEqual("the bakery opening", records[0].old_audio)
            self.assertEqual("the mayor's remarks", records[0].new_audio)
            self.assertEqual("reference_negative", records[0].hard_negatives[0]["type"])

    def test_prepare_can_build_eval_gallery_with_random_distractors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            dataset_run.mkdir()
            self._write_jsonl(
                dataset_run / "b_main_audio_cvr_triplets.jsonl",
                [
                    self._record("main_1", source="source_a", pair="pair_a"),
                    self._record("main_2", source="source_b", pair="pair_b", split_tier="extended"),
                ],
            )
            self._write_jsonl(
                dataset_run / "single_source_annotations.jsonl",
                [
                    {"clip_id": "d1", "output_path": "/tmp/distractor_001.mp4", "source_clip_id": "other_source_1"},
                    {"clip_id": "d2", "output_path": "/tmp/distractor_002.mp4", "source_clip_id": "other_source_2"},
                    {"clip_id": "d3", "output_path": "/tmp/distractor_003.mp4", "source_clip_id": "other_source_3"},
                    {"clip_id": "d4", "output_path": "/tmp/distractor_004.mp4", "source_clip_id": "other_source_4"},
                ],
            )

            summary = prepare_records(
                run_root=dataset_run,
                output_dir=root / "records",
                max_train_records=2,
                max_eval_records=1,
                eval_gallery_size=4,
                distractor_seed=7,
            )

            gallery = [json.loads(line) for line in (root / "records" / "eval_gallery.jsonl").read_text(encoding="utf-8").splitlines()]
            positive_indices = json.loads((root / "records" / "eval_gallery_positive_indices.json").read_text(encoding="utf-8"))

            self.assertEqual(4, summary["eval_gallery"]["gallery_count"])
            self.assertEqual(1, summary["eval_gallery"]["positive_count"])
            self.assertEqual(3, summary["eval_gallery"]["distractor_count"])
            self.assertEqual(4, len(gallery))
            self.assertEqual(1, len(positive_indices["positive_gallery_index"]))
            self.assertTrue(any(item["kind"] == "distractor" for item in gallery))

    def test_prepare_can_include_reference_negative_in_eval_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            dataset_run.mkdir()
            self._write_jsonl(dataset_run / "b_main_audio_cvr_triplets.jsonl", [self._record("main_1", source="source_a", pair="pair_a")])
            self._write_jsonl(
                dataset_run / "single_source_annotations.jsonl",
                [
                    {"clip_id": "d1", "output_path": "/tmp/distractor_001.mp4", "source_clip_id": "other_source_1"},
                    {"clip_id": "d2", "output_path": "/tmp/distractor_002.mp4", "source_clip_id": "other_source_2"},
                ],
            )

            summary = prepare_records(
                run_root=dataset_run,
                output_dir=root / "records",
                max_train_records=1,
                max_eval_records=1,
                eval_gallery_size=4,
                eval_gallery_include_reference_negative=True,
                distractor_seed=7,
            )

            gallery = [json.loads(line) for line in (root / "records" / "eval_gallery.jsonl").read_text(encoding="utf-8").splitlines()]
            indices = json.loads((root / "records" / "eval_gallery_positive_indices.json").read_text(encoding="utf-8"))

            self.assertEqual("pilot_only_random_distractor_gallery_with_reference_negative", summary["eval_protocol"])
            self.assertEqual(1, summary["eval_gallery"]["reference_negative_count"])
            self.assertEqual(1, len(indices["positive_gallery_index"]))
            self.assertEqual(1, len(indices["reference_gallery_index"]))
            self.assertTrue(any(item["kind"] == "reference_negative" for item in gallery))

    def test_prepare_can_build_typed_and_local_audio_cvr_galleries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            dataset_run.mkdir()
            record = self._record(
                "main_1",
                source="source_a",
                pair="pair_a",
                negatives=[
                    {"type": "reference_negative", "video": "/tmp/main_1_ref.mp4", "source_id": "source_a", "satisfies_edit": "false"},
                    {
                        "type": "visual_hard",
                        "video": "/tmp/main_1_visual.mp4",
                        "source_id": "source_b",
                        "satisfies_edit": "false",
                        "temporal_relation": "visual_hard_fallback",
                        "verification_status": "human_verified",
                    },
                    {
                        "type": "local_same_source",
                        "video": "/tmp/main_1_local.mp4",
                        "source_id": "source_a",
                        "satisfies_edit": "false",
                        "temporal_relation": "adjacent_after",
                        "verification_status": "human_verified",
                    },
                    {"type": "audio_hard", "video": "/tmp/main_1_audio.mp4", "source_id": "source_b", "satisfies_edit": "false"},
                    {"type": "asr_hard", "video": "/tmp/main_1_asr.mp4", "source_id": "source_c", "satisfies_edit": "true"},
                ],
            )
            self._write_jsonl(dataset_run / "b_main_audio_cvr_triplets.jsonl", [record])

            typed = prepare_records(
                run_root=dataset_run,
                output_dir=root / "typed_records",
                max_train_records=1,
                max_eval_records=1,
                eval_gallery_size=6,
                eval_gallery_protocol="typed_hardneg",
            )
            typed_gallery = [json.loads(line) for line in (root / "typed_records" / "eval_gallery.jsonl").read_text(encoding="utf-8").splitlines()]
            typed_kinds = {item["kind"] for item in typed_gallery}
            self.assertEqual("audio_cvr_typed_hardneg_gallery", typed["eval_protocol"])
            self.assertIn("reference_negative", typed_kinds)
            self.assertIn("visual_hard", typed_kinds)
            self.assertIn("audio_hard", typed_kinds)
            self.assertNotIn("asr_hard", typed_kinds)

            local = prepare_records(
                run_root=dataset_run,
                output_dir=root / "local_records",
                max_train_records=1,
                max_eval_records=1,
                eval_gallery_size=4,
                eval_gallery_protocol="local_same_source",
            )
            local_gallery = [json.loads(line) for line in (root / "local_records" / "eval_gallery.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual("audio_cvr_local_same_source_gallery", local["eval_protocol"])
            self.assertTrue(any(item["kind"] == "local_same_source" for item in local_gallery))
            self.assertFalse(any(item["kind"] == "audio_hard" for item in local_gallery))
            local_payloads = [item["source_payload"] for item in local_gallery if item["kind"] == "local_same_source"]
            self.assertTrue(any(payload.get("temporal_relation") == "adjacent_after" for payload in local_payloads))
            self.assertTrue(any(payload.get("verification_status") == "human_verified" for payload in local_payloads))
            self.assertFalse(any(item["kind"] == "local_same_source" and item["source_payload"].get("negative_type") == "visual_hard" for item in local_gallery))
            self.assertTrue(any(item["kind"] == "local_fallback_visual" for item in local_gallery))
            self.assertFalse(any(item["kind"] == "local_fallback_visual" and item["source_payload"].get("same_source") for item in local_gallery))

    def test_mine_local_same_source_candidates_from_clip_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            dataset_run.mkdir()
            record = self._record("main_1", source="source_a", pair="pair_a")
            record["reference_video"] = "/clips/source_a/source_a__single_001.mp4"
            record["target_video"] = "/clips/source_a/source_a__single_003.mp4"
            self._write_jsonl(dataset_run / "b_main_audio_cvr_triplets.jsonl", [record])
            self._write_jsonl(
                dataset_run / "single_source_annotations.jsonl",
                [
                    {"clip_id": "source_a__single_000", "output_path": "/clips/source_a/source_a__single_000.mp4", "source_clip_id": "source_a"},
                    {"clip_id": "source_a__single_001", "output_path": "/clips/source_a/source_a__single_001.mp4", "source_clip_id": "source_a"},
                    {"clip_id": "source_a__single_002", "output_path": "/clips/source_a/source_a__single_002.mp4", "source_clip_id": "source_a"},
                    {"clip_id": "source_a__single_003", "output_path": "/clips/source_a/source_a__single_003.mp4", "source_clip_id": "source_a"},
                    {"clip_id": "source_a__single_004", "output_path": "/clips/source_a/source_a__single_004.mp4", "source_clip_id": "source_a"},
                    {"clip_id": "source_b__single_000", "output_path": "/clips/source_b/source_b__single_000.mp4", "source_clip_id": "source_b"},
                ],
            )

            output_path = dataset_run / "b_main_local_same_source_candidates.jsonl"
            summary = mine_local_same_source(
                run_root=dataset_run,
                input_path=dataset_run / "b_main_audio_cvr_triplets.jsonl",
                output_path=output_path,
                max_per_query=5,
            )
            candidates = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertGreaterEqual(summary["strict_local_same_source_coverage"], 1.0)
            self.assertTrue(candidates)
            self.assertFalse(any(item["video"] == record["reference_video"] for item in candidates))
            self.assertFalse(any(item["video"] == record["target_video"] for item in candidates))
            self.assertTrue(all(item["negative_type"] == "local_same_source" for item in candidates))
            self.assertTrue(any(item["temporal_relation"] == "adjacent_after" for item in candidates))
            self.assertEqual("candidate_unverified", candidates[0]["verification_status"])
            self.assertTrue((dataset_run / "local_same_source_candidate_summary.json").exists())
            self.assertTrue((dataset_run / "local_same_source_coverage.md").exists())

    def test_mine_local_same_source_recovers_source_from_manifest_when_triplet_source_is_blank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            dataset_run.mkdir()
            record = self._record("main_1", source="", pair="pair_a")
            record["raw_source_id"] = ""
            record["source_clip_id"] = ""
            record["reference_video"] = "clips/audio_cvr_6_9s/source_a/source_a__single_001.mp4"
            record["target_video"] = "clips/audio_cvr_6_9s/source_a/source_a__single_003.mp4"
            self._write_jsonl(dataset_run / "b_main_audio_cvr_triplets.jsonl", [record])
            self._write_jsonl(
                dataset_run / "single_source_annotations.jsonl",
                [
                    {
                        "clip_id": "source_a__single_001",
                        "output_path": "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/audio_cvr_6_9s/source_a/source_a__single_001.mp4",
                        "source_clip_id": "source_a",
                    },
                    {
                        "clip_id": "source_a__single_002",
                        "output_path": "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/audio_cvr_6_9s/source_a/source_a__single_002.mp4",
                        "source_clip_id": "source_a",
                    },
                    {
                        "clip_id": "source_a__single_003",
                        "output_path": "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/audio_cvr_6_9s/source_a/source_a__single_003.mp4",
                        "source_clip_id": "source_a",
                    },
                ],
            )

            output_path = dataset_run / "b_main_local_same_source_candidates.jsonl"
            summary = mine_local_same_source(
                run_root=dataset_run,
                input_path=dataset_run / "b_main_audio_cvr_triplets.jsonl",
                output_path=output_path,
                max_per_query=5,
            )
            candidates = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(1.0, summary["strict_local_same_source_coverage"])
            self.assertEqual(1, len(candidates))
            self.assertEqual("source_a", candidates[0]["source_id"])
            self.assertTrue(candidates[0]["video"].endswith("source_a__single_002.mp4"))

    def test_prepare_can_use_mined_local_candidate_and_verified_galleries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            dataset_run.mkdir()
            record = self._record("main_1", source="source_a", pair="pair_a")
            self._write_jsonl(dataset_run / "b_main_audio_cvr_triplets.jsonl", [record])
            candidate = {
                "sample_id": "main_1",
                "type": "local_same_source",
                "negative_type": "local_same_source",
                "video": "/tmp/main_1_local.mp4",
                "source_id": "source_a",
                "satisfies_edit": "unknown",
                "verification_status": "candidate_unverified",
                "temporal_relation": "adjacent_after",
            }
            candidates_path = dataset_run / "local_candidates.jsonl"
            self._write_jsonl(candidates_path, [candidate])

            candidate_summary = prepare_records(
                run_root=dataset_run,
                output_dir=root / "candidate_records",
                max_train_records=1,
                max_eval_records=1,
                eval_gallery_size=4,
                eval_gallery_protocol="local_same_source_candidate",
                local_same_source_candidates_path=candidates_path,
            )
            candidate_gallery = [json.loads(line) for line in (root / "candidate_records" / "eval_gallery.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual("audio_cvr_local_same_source_candidate_gallery", candidate_summary["eval_protocol"])
            self.assertTrue(any(item["kind"] == "local_same_source" for item in candidate_gallery))
            self.assertTrue(any(item["source_payload"].get("verification_status") == "candidate_unverified" for item in candidate_gallery))

            verified_summary = prepare_records(
                run_root=dataset_run,
                output_dir=root / "verified_empty_records",
                max_train_records=1,
                max_eval_records=1,
                eval_gallery_size=4,
                eval_gallery_protocol="local_same_source_verified",
                local_same_source_candidates_path=candidates_path,
            )
            verified_empty_gallery = [json.loads(line) for line in (root / "verified_empty_records" / "eval_gallery.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual("audio_cvr_local_same_source_verified_gallery", verified_summary["eval_protocol"])
            self.assertFalse(any(item["kind"] == "local_same_source" for item in verified_empty_gallery))

            candidate["satisfies_edit"] = "false"
            candidate["verification_status"] = "human_verified"
            self._write_jsonl(candidates_path, [candidate])
            verified_summary = prepare_records(
                run_root=dataset_run,
                output_dir=root / "verified_records",
                max_train_records=1,
                max_eval_records=1,
                eval_gallery_size=4,
                eval_gallery_protocol="local_same_source_verified",
                local_same_source_candidates_path=candidates_path,
            )
            verified_gallery = [json.loads(line) for line in (root / "verified_records" / "eval_gallery.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual("audio_cvr_local_same_source_verified_gallery", verified_summary["eval_protocol"])
            self.assertTrue(any(item["kind"] == "local_same_source" for item in verified_gallery))

    def test_cache_train_and_eval_adapter_smoke_with_mock_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            train_rows = [
                self._record("sample_1", source="source_a", pair="pair_a"),
                self._record("sample_2", source="source_b", pair="pair_b", old_audio="quiet room ambience", new_audio="crowd cheering"),
            ]
            self._write_jsonl(records_dir / "train.jsonl", train_rows)
            self._write_jsonl(records_dir / "eval.jsonl", train_rows[:1])

            cache_summary = cache_embeddings(
                records_dir=records_dir,
                output_dir=root / "embedding_cache",
                mock_encoder=True,
                local_segments=2,
            )
            train_summary = train_adapter(
                cache_dir=root / "embedding_cache",
                output_dir=root / "adapter",
                steps=2,
                batch_size=2,
                device="cpu",
            )
            eval_summary = eval_adapter(
                cache_dir=root / "embedding_cache",
                adapter_dir=root / "adapter",
                output_dir=root / "eval",
                device="cpu",
            )

            self.assertEqual([2, 32], cache_summary["train"]["embedding_shape"])
            self.assertEqual([2, 2, 32], cache_summary["train"]["target_segments_shape"])
            self.assertTrue((root / "adapter" / "adapter.pt").exists())
            self.assertTrue((root / "adapter" / "loss_curve.jsonl").exists())
            self.assertEqual(2, train_summary["steps"])
            self.assertEqual(1, eval_summary["eval_count"])
            self.assertTrue(eval_summary["has_local_segments"])
            self.assertIn("by_audio_delta_type", eval_summary)
            self.assertTrue((root / "eval" / "comparison.md").exists())

    def test_low_rank_residual_adapter_starts_at_identity_and_round_trips(self) -> None:
        torch = _import_torch()
        model = _AudioDeltaAdapter(torch, 8, adapter_architecture="low_rank_residual", adapter_rank=2)
        value = torch.randn(3, 8)

        expected = torch.nn.functional.normalize(value, dim=-1)
        self.assertTrue(torch.allclose(expected, model.query(value), atol=1e-7))
        self.assertTrue(torch.allclose(expected, model.doc(value), atol=1e-7))
        self.assertEqual(2, model.adapter_rank)
        self.assertEqual(99, sum(parameter.numel() for parameter in model.parameters()))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [self._record("sample_1", source="source_a", pair="pair_a")]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)
            cache_embeddings(records_dir=records_dir, output_dir=root / "cache", mock_encoder=True)
            train_summary = train_adapter(
                cache_dir=root / "cache",
                output_dir=root / "adapter",
                steps=1,
                batch_size=1,
                device="cpu",
                adapter_architecture="low_rank_residual",
                adapter_rank=4,
            )
            eval_summary = eval_adapter(cache_dir=root / "cache", adapter_dir=root / "adapter", output_dir=root / "eval", device="cpu")

            self.assertEqual("low_rank_residual", train_summary["adapter_architecture"])
            self.assertEqual(4, train_summary["adapter_rank"])
            self.assertEqual(771, train_summary["trainable_parameter_count"])
            self.assertEqual("low_rank_residual", eval_summary["adapter_architecture"])
            self.assertEqual(4, eval_summary["adapter_rank"])
            adapter_row = next(row for row in eval_summary["rows"] if row["method"] == "audio_delta_adapter_global")
            self.assertIn("MRR", adapter_row)
            self.assertIn("mean_rank", adapter_row)
            self.assertIn("median_rank", adapter_row)
            self.assertIn("by_dataset_group", eval_summary)

    def test_eval_can_exclude_reference_from_same_cached_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [self._record("sample_1", source="source_a", pair="pair_a")]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)
            self._write_jsonl(
                records_dir / "eval_gallery.jsonl",
                [
                    {"gallery_id": "positive::sample_1", "video": "/tmp/sample_1_tgt.mp4", "raw_source_id": "source_a", "kind": "positive"},
                    {"gallery_id": "reference::sample_1", "video": "/tmp/sample_1_ref.mp4", "raw_source_id": "source_a", "kind": "reference_negative"},
                    {"gallery_id": "distractor::1", "video": "/tmp/distractor.mp4", "raw_source_id": "other", "kind": "distractor"},
                ],
            )
            cache_embeddings(records_dir=records_dir, output_dir=root / "cache", mock_encoder=True)
            train_adapter(cache_dir=root / "cache", output_dir=root / "adapter", steps=1, batch_size=1, device="cpu")

            summary = eval_adapter(
                cache_dir=root / "cache",
                adapter_dir=root / "adapter",
                output_dir=root / "eval_without_ref",
                device="cpu",
                exclude_gallery_kinds=("reference_negative",),
                save_topk=3,
            )

            self.assertEqual(3, summary["gallery_count"])
            self.assertEqual(2, summary["effective_gallery_count"])
            self.assertEqual(1, summary["excluded_gallery_count"])
            self.assertFalse(summary["reference_in_gallery"])
            scores = json.loads((root / "eval_without_ref" / "per_query_scores.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(np.isfinite(scores["base_reference_score"]))
            topk = json.loads((root / "eval_without_ref" / "per_query_topk.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(any(row["kind"] == "reference_negative" for row in topk["base_topk"][:2]))

    def test_eval_can_score_small_query_set_against_larger_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            eval_rows = [self._record("sample_1", source="source_a", pair="pair_a")]
            self._write_jsonl(records_dir / "train.jsonl", eval_rows)
            self._write_jsonl(records_dir / "eval.jsonl", eval_rows)
            self._write_jsonl(
                records_dir / "eval_gallery.jsonl",
                [
                    {"gallery_id": "distractor::1", "video": "/tmp/distractor_001.mp4", "raw_source_id": "other_source_1", "kind": "distractor"},
                    {"gallery_id": "positive::sample_1", "video": "/tmp/sample_1_tgt.mp4", "raw_source_id": "source_a", "kind": "positive"},
                    {"gallery_id": "distractor::2", "video": "/tmp/distractor_002.mp4", "raw_source_id": "other_source_2", "kind": "distractor"},
                ],
            )

            cache_embeddings(records_dir=records_dir, output_dir=root / "embedding_cache", mock_encoder=True, local_segments=1)
            train_adapter(cache_dir=root / "embedding_cache", output_dir=root / "adapter", steps=1, batch_size=1, device="cpu")
            eval_summary = eval_adapter(cache_dir=root / "embedding_cache", adapter_dir=root / "adapter", output_dir=root / "eval", device="cpu")

            self.assertEqual(1, eval_summary["eval_count"])
            self.assertEqual(3, eval_summary["gallery_count"])
            self.assertIn("R@1", eval_summary["rows"][0])
            self.assertTrue((root / "eval" / "comparison.md").read_text(encoding="utf-8").count("gallery_count"))

    def test_eval_large_gallery_without_local_segments_does_not_require_segment_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            eval_rows = [self._record("sample_1", source="source_a", pair="pair_a")]
            self._write_jsonl(records_dir / "train.jsonl", eval_rows)
            self._write_jsonl(records_dir / "eval.jsonl", eval_rows)
            self._write_jsonl(
                records_dir / "eval_gallery.jsonl",
                [
                    {"gallery_id": "positive::sample_1", "video": "/tmp/sample_1_tgt.mp4", "raw_source_id": "source_a", "kind": "positive"},
                    {"gallery_id": "reference::sample_1", "video": "/tmp/sample_1_ref.mp4", "raw_source_id": "source_a", "kind": "reference_negative"},
                    {"gallery_id": "distractor::1", "video": "/tmp/distractor_001.mp4", "raw_source_id": "other_source_1", "kind": "distractor"},
                ],
            )

            cache_embeddings(records_dir=records_dir, output_dir=root / "embedding_cache", mock_encoder=True, local_segments=0)
            train_adapter(cache_dir=root / "embedding_cache", output_dir=root / "adapter", steps=1, batch_size=1, device="cpu")
            eval_summary = eval_adapter(cache_dir=root / "embedding_cache", adapter_dir=root / "adapter", output_dir=root / "eval", device="cpu", save_topk=2)

            self.assertFalse(eval_summary["has_local_segments"])
            self.assertTrue(eval_summary["has_reference_gallery_index"])
            self.assertEqual(3, eval_summary["gallery_count"])
            self.assertIn("R@1", eval_summary["rows"][0])
            self.assertTrue((root / "eval" / "per_query_topk.jsonl").exists())
            self.assertTrue((root / "eval" / "per_query_scores.jsonl").exists())
            self.assertTrue((root / "eval" / "score_diagnostics.json").exists())
            self.assertTrue((root / "eval" / "adapter_geometry.json").exists())
            score_row = json.loads((root / "eval" / "per_query_scores.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNotNone(score_row["reference_gallery_index"])

    def test_cache_embeddings_can_reuse_old_cache_for_reference_negative_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_records = root / "old_records"
            old_records.mkdir()
            rows = [
                self._record("sample_1", source="source_a", pair="pair_a"),
                self._record("sample_2", source="source_b", pair="pair_b"),
            ]
            self._write_jsonl(old_records / "train.jsonl", rows)
            self._write_jsonl(old_records / "eval.jsonl", rows)
            self._write_jsonl(
                old_records / "eval_gallery.jsonl",
                [
                    {"gallery_id": "positive::sample_1", "video": "/tmp/sample_1_tgt.mp4", "raw_source_id": "source_a", "kind": "positive", "source_payload": {"sample_id": "sample_1"}},
                    {"gallery_id": "positive::sample_2", "video": "/tmp/sample_2_tgt.mp4", "raw_source_id": "source_b", "kind": "positive", "source_payload": {"sample_id": "sample_2"}},
                    {"gallery_id": "distractor::1", "video": "/tmp/distractor_001.mp4", "raw_source_id": "other_source_1", "kind": "distractor"},
                ],
            )
            old_cache = root / "old_cache"
            cache_embeddings(records_dir=old_records, output_dir=old_cache, mock_encoder=True, local_segments=0)

            new_records = root / "new_records"
            new_records.mkdir()
            self._write_jsonl(new_records / "train.jsonl", rows)
            self._write_jsonl(new_records / "eval.jsonl", rows)
            self._write_jsonl(
                new_records / "eval_gallery.jsonl",
                [
                    {"gallery_id": "positive::sample_2", "video": "/tmp/sample_2_tgt.mp4", "raw_source_id": "source_b", "kind": "positive", "source_payload": {"sample_id": "sample_2"}},
                    {"gallery_id": "reference::sample_1", "video": "/tmp/sample_1_ref.mp4", "raw_source_id": "source_a", "kind": "reference_negative", "source_payload": {"sample_id": "sample_1"}},
                    {"gallery_id": "positive::sample_1", "video": "/tmp/sample_1_tgt.mp4", "raw_source_id": "source_a", "kind": "positive", "source_payload": {"sample_id": "sample_1"}},
                    {"gallery_id": "distractor::1", "video": "/tmp/distractor_001.mp4", "raw_source_id": "other_source_1", "kind": "distractor"},
                    {"gallery_id": "reference::sample_2", "video": "/tmp/sample_2_ref.mp4", "raw_source_id": "source_b", "kind": "reference_negative", "source_payload": {"sample_id": "sample_2"}},
                ],
            )

            summary = cache_embeddings(records_dir=new_records, output_dir=root / "new_cache", reuse_cache_from=old_cache)
            data = dict(np.load(str(root / "new_cache" / "eval_embeddings.npz")))

            self.assertEqual(str(old_cache), summary["reuse_cache_from"])
            self.assertEqual([5, 32], list(data["gallery"].shape))
            self.assertEqual([2, 0], list(data["positive_gallery_index"]))
            self.assertEqual([1, 4], list(data["reference_gallery_index"]))

    def test_real_encoder_inputs_wrap_video_paths_as_multimodal_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            self._write_jsonl(records_dir / "train.jsonl", [self._record("sample_1", source="source_a", pair="pair_a")])
            self._write_jsonl(records_dir / "eval.jsonl", [self._record("sample_1", source="source_a", pair="pair_a")])
            encoder = _SpyEncoder()

            cache_embeddings(
                records_dir=records_dir,
                output_dir=root / "embedding_cache",
                encoder=encoder,
            )

            bare_video_strings = [item for item in encoder.inputs if isinstance(item, str) and item.endswith(".mp4")]
            self.assertEqual([], bare_video_strings)
            self.assertTrue(any(isinstance(item, dict) and item.get("video", "").endswith("_ref.mp4") and "text" in item for item in encoder.inputs))
            self.assertTrue(any(isinstance(item, dict) and item.get("video", "").endswith("_tgt.mp4") and "text" not in item for item in encoder.inputs))

    def test_video_payload_resolves_relative_clip_paths_under_default_data_root(self) -> None:
        payload = _video_payload("clips/audio_cvr_6_9s/example.mp4")

        self.assertEqual(str(Path(DEFAULT_DATA_ROOT) / "clips/audio_cvr_6_9s/example.mp4"), payload["video"])

    def test_train_lora_plan_is_dry_run_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = train_lora_plan(output_dir=Path(temp_dir) / "lora")

            self.assertEqual("dry_run_only", plan["status"])
            self.assertIn("q_proj", plan["default_target_modules"])
            self.assertTrue((Path(temp_dir) / "lora" / "lora_plan.json").exists())

    def test_build_splits_is_source_and_pair_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            self._write_jsonl(
                dataset_run / "b_all_audio_cvr_triplets.jsonl",
                [
                    self._record("s1_a", source="source_1", pair="pair_1"),
                    self._record("s1_b_inverse", source="source_1", pair="pair_1", direction="inverse"),
                    self._record("s2_a", source="source_2", pair="pair_2"),
                    self._record("s3_a", source="source_3", pair="pair_3", split_tier="diagnostic", shortcut_label="asr_like"),
                    self._record("s4_a", source="source_4", pair="pair_4"),
                ],
            )

            summary = build_splits(run_root=dataset_run, output_dir=root / "splits", train_ratio=0.5, val_ratio=0.25, seed=1)

            self.assertEqual([], summary["leakage_checks"]["raw_source_cross_split_leaks"])
            self.assertEqual([], summary["leakage_checks"]["pair_group_cross_split_leaks"])
            self.assertTrue(summary["leakage_checks"]["test_main_unique_pair_groups"])
            self.assertTrue((root / "splits" / "train.jsonl").exists())
            self.assertTrue((root / "splits" / "diagnostic.jsonl").exists())

    def test_run_ablations_writes_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            train_rows = [
                self._record("sample_1", source="source_a", pair="pair_a"),
                self._record("sample_2", source="source_b", pair="pair_b", old_audio="quiet room ambience", new_audio="crowd cheering"),
            ]
            self._write_jsonl(records_dir / "train.jsonl", train_rows)
            self._write_jsonl(records_dir / "eval.jsonl", train_rows)
            cache_embeddings(records_dir=records_dir, output_dir=root / "embedding_cache", mock_encoder=True, local_segments=1)

            summary = run_ablations(
                cache_dir=root / "embedding_cache",
                output_dir=root / "ablations",
                steps=1,
                batch_size=2,
                device="cpu",
            )

            self.assertGreaterEqual(len(summary["rows"]), 3)
            names = {row["ablation"] for row in summary["rows"]}
            self.assertIn("without_modality_temperature", names)
            self.assertIn("without_quantile_negative_curriculum", names)
            self.assertIn("without_batch_whitening", names)
            self.assertNotIn("C1_ref_delta", names)
            self.assertTrue((root / "ablations" / "comparison.md").exists())

    def test_run_loss_schedule_is_recipe_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [
                self._record("sample_1", source="source_a", pair="pair_a"),
                self._record("sample_2", source="source_b", pair="pair_b", old_audio="quiet room ambience", new_audio="crowd cheering"),
            ]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)
            cache_embeddings(records_dir=records_dir, output_dir=root / "embedding_cache", mock_encoder=True, local_segments=0)

            summary = run_loss_schedule(
                cache_dir=root / "embedding_cache",
                output_dir=root / "loss_schedule",
                steps=1,
                batch_size=2,
                device="cpu",
            )

            names = {row["name"] for row in summary["rows"]}
            self.assertEqual({"S1_e5_omni_recipe"}, names)
            self.assertEqual(0.0, summary["rows"][0]["lambda_ref"])
            self.assertEqual(0.0, summary["rows"][0]["lambda_delta"])
            self.assertEqual(0.0, summary["rows"][0]["lambda_hn"])
            self.assertFalse(summary["audio_delta_stage2_enabled"])
            self.assertTrue((root / "loss_schedule" / "loss_schedule_summary.json").exists())
            self.assertTrue((root / "loss_schedule" / "loss_schedule_comparison.md").exists())

    def test_focused_loss_schedule_keeps_recipe_baseline_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [
                self._record("sample_1", source="source_a", pair="pair_a"),
                self._record("sample_2", source="source_b", pair="pair_b"),
            ]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)
            cache_embeddings(records_dir=records_dir, output_dir=root / "embedding_cache", mock_encoder=True, local_segments=0)

            summary = run_loss_schedule(
                cache_dir=root / "embedding_cache",
                output_dir=root / "loss_schedule_focused",
                steps=1,
                batch_size=2,
                device="cpu",
                schedule_preset="focused",
            )

            names = {row["name"] for row in summary["rows"]}
            self.assertEqual({"S1_e5_omni_recipe"}, names)
            self.assertEqual(1, summary["rows"][0]["steps"])

    def test_early_stop_loss_schedule_does_not_reintroduce_task_losses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [
                self._record("sample_1", source="source_a", pair="pair_a"),
                self._record("sample_2", source="source_b", pair="pair_b"),
            ]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)
            cache_embeddings(records_dir=records_dir, output_dir=root / "embedding_cache", mock_encoder=True, local_segments=0)

            summary = run_loss_schedule(
                cache_dir=root / "embedding_cache",
                output_dir=root / "loss_schedule_early_stop",
                steps=2,
                batch_size=2,
                learning_rate=3e-4,
                device="cpu",
                schedule_preset="early_stop",
            )

            rows_by_name = {row["name"]: row for row in summary["rows"]}
            self.assertEqual({"S1_e5_omni_recipe"}, set(rows_by_name))
            self.assertEqual(2, rows_by_name["S1_e5_omni_recipe"]["steps"])
            self.assertEqual(0.0, rows_by_name["S1_e5_omni_recipe"]["lambda_ref"])
            self.assertEqual(0.0, rows_by_name["S1_e5_omni_recipe"]["lambda_delta"])
            comparison = (root / "loss_schedule_early_stop" / "loss_schedule_comparison.md").read_text(encoding="utf-8")
            self.assertIn("| Run | Eval | Steps | LR |", comparison)

    def test_gallery_negative_recall_reports_same_source_gallery_items(self) -> None:
        records = load_audio_delta_records_from_rows([self._record("sample_1", source="source_a", pair="pair_a")])
        items = [
            EvalGalleryItem("positive::sample_1", "target.mp4", "source_a", "positive", {"sample_id": "sample_1", "negative_type": ""}),
            EvalGalleryItem("reference::sample_1", "ref.mp4", "source_a", "reference_negative", {"sample_id": "sample_1", "negative_type": "reference_negative", "same_source": True}),
            EvalGalleryItem("local::sample_1", "local.mp4", "source_a", "local_same_source", {"sample_id": "sample_1", "negative_type": "local_same_source", "same_source": True}),
            EvalGalleryItem("visual::sample_1", "visual.mp4", "source_b", "visual_hard", {"sample_id": "sample_1", "negative_type": "visual_hard", "same_source": False}),
        ]
        scores = np.asarray([[0.9, 0.8, 0.95, 0.1]], dtype=np.float32)

        summary = _gallery_negative_recall_by_type(scores, items, records, positive_index=np.asarray([0]))

        self.assertEqual(1.0, summary["reference_negative"]["positive_beats_negative_rate"])
        self.assertEqual(0.0, summary["local_same_source"]["positive_beats_negative_rate"])
        self.assertEqual(0.5, summary["same_source_any"]["positive_beats_negative_rate"])
        self.assertEqual(1.0, summary["visual_hard"]["positive_beats_negative_rate"])

    def test_load_eval_gallery_items_flattens_nested_source_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "eval_gallery.jsonl"
            self._write_jsonl(
                path,
                [
                    {
                        "gallery_id": "local::sample_1",
                        "video": "local.mp4",
                        "raw_source_id": "source_a",
                        "kind": "local_same_source",
                        "source_payload": {
                            "gallery_id": "outer_local::sample_1",
                            "kind": "local_same_source",
                            "source_payload": {
                                "sample_id": "sample_1",
                                "negative_type": "local_same_source",
                                "same_source": True,
                            },
                        },
                    }
                ],
            )

            items = load_eval_gallery_items(path)

            self.assertEqual("sample_1", items[0].source_payload["sample_id"])
            self.assertEqual("local_same_source", items[0].source_payload["negative_type"])
            self.assertTrue(items[0].source_payload["same_source"])

    def test_stability_grid_reuses_cache_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [
                self._record("sample_1", source="source_a", pair="pair_a"),
                self._record("sample_2", source="source_b", pair="pair_b"),
            ]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows[:1])
            cache_embeddings(records_dir=records_dir, output_dir=root / "embedding_cache", mock_encoder=True, local_segments=0)

            summary = run_stability_grid(
                cache_dir=root / "embedding_cache",
                output_dir=root / "stability",
                steps_grid=(1,),
                learning_rate_grid=(1e-3,),
                batch_size=1,
                device="cpu",
            )

            self.assertEqual(1, len(summary["rows"]))
            self.assertTrue((root / "stability" / "stability_grid_summary.json").exists())
            self.assertTrue((root / "stability" / "stability_grid_comparison.md").exists())

    def test_v2_research_profile_logs_new_losses_and_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [
                self._record("sample_1", source="source_a", pair="pair_shared"),
                self._record("sample_2", source="source_b", pair="pair_shared", direction="inverse"),
            ]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows[:1])
            cache_embeddings(records_dir=records_dir, output_dir=root / "embedding_cache", mock_encoder=True, local_segments=2)

            summary = train_adapter(
                cache_dir=root / "embedding_cache",
                output_dir=root / "adapter",
                steps=3,
                batch_size=2,
                device="cpu",
                training_profile="v2_research",
                enable_memory_bank=True,
                memory_bank_size=8,
            )
            loss_rows = [json.loads(line) for line in (root / "adapter" / "loss_curve.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual("v2_research", summary["training_profile"])
            self.assertEqual("masked_dcl", summary["loss_options"]["contrastive_objective"])
            self.assertFalse(summary["loss_options"]["enable_hardness_weighting"])
            self.assertFalse(summary["loss_options"]["enable_multi_positive"])
            self.assertTrue(summary["loss_options"]["enable_batch_whitening"])
            self.assertTrue(summary["loss_options"]["enable_modality_temperature"])
            self.assertTrue(summary["loss_options"]["enable_coral_align"])
            self.assertTrue(summary["loss_options"]["enable_quantile_negative_curriculum"])
            self.assertTrue(summary["loss_options"]["enable_false_negative_filtering"])
            self.assertEqual(0.0, summary["loss_options"]["lambda_delta"])
            self.assertEqual(0.0, summary["loss_options"]["lambda_ref"])
            self.assertEqual(0.0, summary["loss_options"]["lambda_hn"])
            self.assertEqual(0.0, summary["loss_options"]["lambda_edit_type"])
            self.assertEqual(0.0, summary["loss_options"]["lambda_visual"])
            self.assertIn("loss_hw_hn", loss_rows[-1])
            self.assertIn("loss_multi_positive", loss_rows[-1])
            self.assertIn("loss_masked_dcl", loss_rows[-1])
            self.assertIn("loss_coral_align", loss_rows[-1])
            self.assertIn("loss_coral_query_target", loss_rows[-1])
            self.assertIn("loss_coral_doc_edit", loss_rows[-1])
            self.assertIn("loss_coral_delta_edit", loss_rows[-1])
            self.assertIn("loss_batch_whitening", loss_rows[-1])
            self.assertIn("loss_memory_bank", loss_rows[-1])
            self.assertIn("tau_text", loss_rows[-1])
            self.assertIn("effective_temperature_cvr", loss_rows[-1])
            self.assertIn("kept_negative_count", loss_rows[-1])
            self.assertIn("suspected_false_negative_count", loss_rows[-1])
            self.assertIn("temperature", loss_rows[-1])
            self.assertIn("whitening_enabled", loss_rows[-1])
            self.assertGreater(loss_rows[-1]["memory_bank_size"], 0)

    def test_e5_omni_recipe_and_v2_research_profiles_match(self) -> None:
        kwargs = dict(
            enable_hardness_weighting=None,
            enable_multi_positive=None,
            enable_coral_align=None,
            enable_memory_bank=None,
            enable_false_negative_filtering=None,
            enable_modality_temperature=None,
            enable_quantile_negative_curriculum=None,
            enable_batch_whitening=None,
            lambda_delta=None,
            lambda_hn=None,
            lambda_ref=None,
            lambda_edit_type=None,
            lambda_visual=None,
            lambda_hw_hn=None,
            lambda_multi_positive=None,
            lambda_coral_align=None,
            lambda_memory_bank=None,
            lambda_batch_whitening=None,
        )
        v2 = _training_profile_options(training_profile="v2_research", **kwargs)
        recipe = _training_profile_options(training_profile="e5_omni_recipe", **kwargs)
        self.assertEqual(v2, recipe)
        self.assertEqual("masked_dcl", recipe["contrastive_objective"])
        self.assertTrue(recipe["enable_batch_whitening"])
        self.assertEqual(0.0, recipe["lambda_delta"])
        self.assertEqual(0.0, recipe["lambda_ref"])
        self.assertEqual(0.0, recipe["lambda_hn"])

        ref_recipe = _training_profile_options(training_profile="e5_omni_recipe", **{**kwargs, "lambda_ref": 0.3, "lambda_delta": 0.5})
        self.assertEqual(0.0, ref_recipe["lambda_ref"])
        self.assertEqual(0.0, ref_recipe["lambda_delta"])
        self.assertEqual(0.0, ref_recipe["lambda_hn"])

    def test_cache_ffmpeg_mode_is_scoped_to_cache_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            self._write_jsonl(records_dir / "train.jsonl", [self._record("sample_1", source="source_a", pair="pair_a")])
            self._write_jsonl(records_dir / "eval.jsonl", [self._record("sample_1", source="source_a", pair="pair_a")])

            summary = cache_embeddings(
                records_dir=records_dir,
                output_dir=root / "embedding_cache",
                mock_encoder=True,
                local_segments=2,
                local_segment_mode="ffmpeg",
                local_segment_cache_dir=root / "local_cache",
            )

            self.assertEqual("ffmpeg", summary["local_segment_mode"])
            self.assertEqual([1, 2, 32], summary["train"]["target_segments_shape"])
            self.assertFalse(Path("/tmp/sample_1_ref.mp4").exists())

    def test_hardness_weights_prioritize_high_similarity_negatives(self) -> None:
        torch = _import_torch()
        records = [load_audio_delta_records_from_rows([self._record("sample_1", source="source_a", pair="pair_a")])[0]]
        neg_scores = torch.tensor([[0.1, 0.9, 0.2, 0.3]], dtype=torch.float32)
        active = torch.ones_like(neg_scores)

        weights = _hardness_weights(torch, records, neg_scores, active, temperature=0.07, weight_min=0.25, weight_max=4.0)

        self.assertGreater(float(weights[0, 1]), float(weights[0, 2]))
        self.assertGreaterEqual(float(weights.min()), 0.25)
        self.assertLessEqual(float(weights.max()), 4.0)

    def test_modality_temperature_clamps_and_can_disable_to_global_temperature(self) -> None:
        torch = _import_torch()
        model = _AudioDeltaAdapter(torch, 4, modality_temperature_init=0.5)

        enabled_tau = _modality_tau(
            torch,
            model,
            ("text", "audio", "video"),
            {
                "enable_modality_temperature": True,
                "modality_temperature_min": 0.005,
                "modality_temperature_max": 0.2,
            },
            fallback=0.07,
            device=torch.device("cpu"),
        )
        disabled_tau = _modality_tau(
            torch,
            model,
            ("text", "audio", "video"),
            {
                "enable_modality_temperature": False,
                "modality_temperature_min": 0.005,
                "modality_temperature_max": 0.2,
            },
            fallback=0.07,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(0.2, float(enabled_tau.detach()), places=6)
        self.assertAlmostEqual(0.07, float(disabled_tau.detach()), places=6)

    def test_quantile_negative_curriculum_masks_easy_negatives_after_warmup(self) -> None:
        torch = _import_torch()
        scores = torch.tensor([[0.1, 0.9, 0.2, 0.3]], dtype=torch.float32)
        active = torch.ones_like(scores)

        weights = _quantile_negative_curriculum_weights(
            torch,
            scores,
            active,
            enabled=True,
            step=10,
            total_steps=10,
            warmup_ratio=0.1,
            keep_ratio_start=1.0,
            keep_ratio_end=0.5,
            easy_weight=0.1,
        )

        self.assertEqual(1.0, float(weights[0, 1]))
        self.assertLess(float(weights[0, 0]), 1.0)
        self.assertGreaterEqual(float(weights.min()), 0.0)

    def test_multi_positive_and_coral_helpers_are_stable(self) -> None:
        torch = _import_torch()
        logits = torch.tensor([[4.0, 4.0, 0.1], [4.0, 4.0, 0.2], [0.1, 0.2, 4.0]], dtype=torch.float32)
        groups = torch.tensor([1, 1, 2], dtype=torch.int64)

        multi_loss = _multi_positive_loss(torch, logits, groups)
        coral_one = _coral_loss(torch, torch.ones(1, 4), torch.ones(1, 4))
        coral_many = _coral_loss(torch, torch.eye(4), torch.flip(torch.eye(4), dims=[0]))
        whiten_one = _batch_whitening_loss(torch, torch.ones(1, 4))
        whiten_many = _batch_whitening_loss(torch, torch.eye(4))

        self.assertLess(float(multi_loss), 0.2)
        self.assertEqual(0.0, float(coral_one))
        self.assertGreaterEqual(float(coral_many), 0.0)
        self.assertEqual(0.0, float(whiten_one))
        self.assertGreaterEqual(float(whiten_many), 0.0)

    def test_false_negative_filter_soft_weights_high_similarity(self) -> None:
        torch = _import_torch()
        record = load_audio_delta_records_from_rows(
            [
                self._record(
                    "sample_1",
                    source="source_a",
                    pair="pair_a",
                    negatives=[
                        {"type": "reference_negative", "video": "/tmp/ref.mp4"},
                        {"type": "visual_hard", "video": "/tmp/vh.mp4", "pair_group_id": "pair_a"},
                        {"type": "audio_hard", "video": "/tmp/ah.mp4"},
                    ],
                )
            ]
        )[0]
        scores = torch.tensor([[0.99, 0.99, 0.95, 0.1]], dtype=torch.float32)

        weights = _false_negative_weights(torch, [record], scores, threshold=0.92, soft_weight=0.15)

        self.assertEqual(1.0, float(weights[0, 0]))
        self.assertEqual(0.0, float(weights[0, 1]))
        self.assertEqual(0.15, round(float(weights[0, 2]), 2))

    def test_schedule_helpers(self) -> None:
        self.assertLess(_scheduled_learning_rate(base_lr=1.0, step=1, total_steps=10, warmup_steps=2, min_ratio=0.1), 1.0)
        self.assertAlmostEqual(0.07, _scheduled_temperature(step=1, total_steps=10, start=0.07, end=0.03))
        self.assertAlmostEqual(0.03, _scheduled_temperature(step=10, total_steps=10, start=0.07, end=0.03))

    def test_cache_embeddings_records_video_audio_mode_for_reusable_protocol_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [self._record("sample_1", source="source_a", pair="pair_a")]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)

            summary = cache_embeddings(
                records_dir=records_dir,
                output_dir=root / "embedding_cache",
                mock_encoder=True,
                video_audio_mode="off",
            )

            self.assertEqual("off", summary["runtime"]["video_audio_mode"])

    def test_cache_embeddings_can_skip_redundant_train_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            rows = [self._record("sample_1", source="source_a", pair="pair_a")]
            self._write_jsonl(records_dir / "train.jsonl", rows)
            self._write_jsonl(records_dir / "eval.jsonl", rows)

            summary = cache_embeddings(
                records_dir=records_dir,
                output_dir=root / "embedding_cache",
                mock_encoder=True,
                skip_train=True,
            )

            self.assertTrue(summary["skip_train"])
            self.assertTrue(summary["train"]["skipped"])
            self.assertFalse((root / "embedding_cache" / "train_embeddings.npz").exists())
            self.assertTrue((root / "embedding_cache" / "eval_embeddings.npz").exists())

    def test_query_input_modes_support_audio_necessity_protocol_payloads(self) -> None:
        record = load_audio_delta_records_from_rows([self._record("sample_1", source="source_a", pair="pair_a")])[0]

        composed = _query_payload(record, query_input_mode="composed")
        text_only = _query_payload(record, query_input_mode="text_only")
        video_only = _query_payload(record, query_input_mode="video_only")
        audio_only = _query_payload(record, query_input_mode="audio_only")
        audio_text = _query_payload(record, query_input_mode="audio_text")
        audio_document = _document_payload(record.target_video, document_input_mode="audio")

        self.assertIsInstance(composed, dict)
        self.assertIn("video", composed)
        self.assertIn("text", composed)
        self.assertIsInstance(text_only, str)
        self.assertIn("Edit the reference video", text_only)
        self.assertEqual({"video": record.reference_video}, video_only)
        self.assertEqual({"audio": str(Path(_resolve_media_path(record.reference_video)))}, audio_only)
        self.assertEqual({"audio": str(Path(_resolve_media_path(record.reference_video))), "text": text_only}, audio_text)
        self.assertEqual({"audio": str(Path(_resolve_media_path(record.target_video)))}, audio_document)

    def test_protocol_eval_summaries_are_reusable_beyond_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_run = root / "dataset_run"
            output_dir = root / "protocol_eval"
            self._write_jsonl(
                dataset_run / "b_main_audio_cvr_triplets.jsonl",
                [
                    self._record(
                        "sample_1",
                        source="source_a",
                        pair="pair_a",
                        negatives=[
                            {"type": "reference_negative", "video": "/tmp/sample_1_ref.mp4", "satisfies_edit": "false"},
                            {"type": "local_same_source", "video": "/tmp/sample_1_local.mp4", "source_id": "source_a", "satisfies_edit": "false", "verification_status": "human_verified", "temporal_relation": "adjacent_after"},
                            {"type": "visual_hard", "video": "/tmp/sample_1_vh.mp4", "satisfies_edit": "false"},
                        ],
                    )
                ],
            )
            self._write_jsonl(dataset_run / "b_extended_audio_cvr_triplets.jsonl", [])
            self._write_jsonl(dataset_run / "b_diagnostic_audio_cvr_triplets.jsonl", [])
            self._write_jsonl(dataset_run / "b_all_audio_cvr_triplets.jsonl", [])
            (dataset_run / "audio_necessity_eval_manifest.json").write_text("{}", encoding="utf-8")
            (dataset_run / "benchmark_quality_summary.json").write_text("{}", encoding="utf-8")

            data_summary = summarize_data(run_root=dataset_run, output_dir=output_dir, run_label="Full Audio-CVR Eval")

            self.assertEqual("Full Audio-CVR Eval", data_summary["run_label"])
            self.assertEqual(1, data_summary["tier_counts"]["main"])
            self.assertEqual(1.0, data_summary["hard_negative_coverage"]["local_same_source"]["coverage_rate"])
            self.assertTrue((output_dir / "data_quality_summary.md").exists())
            self.assertIn("Full Audio-CVR Eval", (output_dir / "data_quality_summary.md").read_text(encoding="utf-8"))

    def test_protocol_eval_can_aggregate_eval_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_dir = root / "eval_typed"
            eval_dir.mkdir(parents=True)
            summary = {
                "rows": [
                    {"method": "base_e5_global", "R@1": 0.2, "R@5": 1.0, "R@10": 1.0},
                    {"method": "audio_delta_adapter_global", "R@1": 0.4, "R@5": 1.0, "R@10": 1.0},
                ],
                "target_beats_reference": {
                    "base_e5": {"target_beats_reference_rate": 0.2, "target_minus_reference_mean": -0.1},
                    "audio_delta_adapter": {"target_beats_reference_rate": 0.4, "target_minus_reference_mean": 0.01},
                },
                "base_reference_rank_summary": {"median_rank": 1},
                "reference_rank_summary": {"median_rank": 2},
                "base_hard_negative_recall_by_type": {"visual_hard": {"positive_beats_negative_rate": 0.8}},
                "hard_negative_recall_by_type": {"visual_hard": {"positive_beats_negative_rate": 0.9}},
            }
            (eval_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self._write_jsonl(
                eval_dir / "per_query_scores.jsonl",
                [
                    {
                        "sample_id": "sample_1",
                        "adapter_top1": {"kind": "reference_negative", "is_target": False, "is_reference": True},
                    }
                ],
            )

            eval_summary = summarize_evals(output_dir=root / "out", evals=[f"typed_hardneg={eval_dir}"], run_label="Full Audio-CVR Eval")

            self.assertEqual("Full Audio-CVR Eval", eval_summary["run_label"])
            self.assertEqual(2, len(eval_summary["gallery_protocol_rows"]))
            self.assertEqual(1, eval_summary["topk_error_count"])
            self.assertTrue((root / "out" / "protocol_eval_summary.json").exists())
            self.assertTrue((root / "out" / "advisor_brief.md").exists())

    def _record(
        self,
        sample_id: str,
        *,
        source: str,
        pair: str,
        split_tier: str = "main",
        old_audio: str = "the bakery opening",
        new_audio: str = "the mayor's remarks",
        direction: str = "forward",
        shortcut_label: str = "clean_audio_delta",
        negatives: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "reference_video": f"/tmp/{sample_id}_ref.mp4",
            "target_video": f"/tmp/{sample_id}_tgt.mp4",
            "edit_text": f"change the speech from discussing {old_audio} to discussing {new_audio}",
            "edit_type": "replace",
            "audio_delta_type": "speech_topic",
            "old_audio": old_audio,
            "new_audio": new_audio,
            "direction": direction,
            "split_tier": split_tier,
            "raw_source_id": source,
            "pair_group_id": pair,
            "inverse_pair_group_id": pair,
            "shortcut_label": shortcut_label,
            "audio_delta_strength": 0.82,
            "video_context_strength": 0.72,
            "asr_degeneracy_risk": 0.20,
            "visual_shortcut_risk": 0.10,
            "full_av_required": True,
            "audio_delta_hard_negatives": negatives or [
                {"type": "reference_negative", "video": f"/tmp/{sample_id}_ref.mp4"},
                {"type": "visual_hard", "video": f"/tmp/{sample_id}_vh.mp4"},
                {"type": "audio_hard", "video": f"/tmp/{sample_id}_ah.mp4"},
                {"type": "asr_hard", "video": f"/tmp/{sample_id}_asr.mp4"},
            ],
        }

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class _SpyEncoder:
    def __init__(self) -> None:
        self.inputs: list[object] = []

    def encode_document(self, inputs: list[object]) -> list[list[float]]:
        self.inputs.extend(inputs)
        return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]


def load_audio_delta_records_from_rows(rows: list[dict[str, object]]):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "records.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return load_audio_delta_records(path)


if __name__ == "__main__":
    unittest.main()
