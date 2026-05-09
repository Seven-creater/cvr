from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.e5_cvr_eval import (
    E5CVRTriplet,
    _configure_video_processing,
    build_or_load_target_index,
    load_triplets_jsonl,
    prepare_reference_audio_triplets,
    run_eval_slice,
)


class FakeE5Encoder:
    def encode_document(self, inputs: list[object]) -> np.ndarray:
        rows = []
        for item in inputs:
            if isinstance(item, dict):
                text = str(item.get("text", ""))
                if "target two" in text:
                    rows.append([0.0, 1.0, 0.0])
                elif "target three" in text:
                    rows.append([0.0, 0.0, 1.0])
                else:
                    rows.append([1.0, 0.0, 0.0])
            else:
                rows.append(self._video_vector(str(item)))
        return np.asarray(rows, dtype=np.float32)

    def _video_vector(self, path: str) -> list[float]:
        if "target2" in path:
            return [0.0, 1.0, 0.0]
        if "target3" in path:
            return [0.0, 0.0, 1.0]
        return [1.0, 0.0, 0.0]


class CapturingE5Encoder(FakeE5Encoder):
    def __init__(self) -> None:
        self.calls: list[list[object]] = []

    def encode_document(self, inputs: list[object]) -> np.ndarray:
        self.calls.append(list(inputs))
        return super().encode_document(inputs)


class FailingE5Encoder:
    def encode_document(self, inputs: list[object]) -> np.ndarray:
        raise AssertionError("target index should have been loaded from cache")


class E5CVREvalTests(unittest.TestCase):
    def test_load_triplets_rejects_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "triplets.jsonl"
            path.write_text('{"sample_id": "sample1", "reference_video": "ref.mp4"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing target_video"):
                load_triplets_jsonl(path)

    def test_target_index_keeps_manifest_order_and_embedding_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            progress: list[str] = []
            index = build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1, "batch_size": 1},
                progress=progress.append,
            )
            loaded_progress: list[str] = []
            loaded = build_or_load_target_index(
                triplets=triplets,
                encoder=FailingE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1, "batch_size": 1},
                progress=loaded_progress.append,
            )

            self.assertEqual(["sample1", "sample2", "sample3"], [record.sample_id for record in index.records])
            self.assertEqual((3, 3), index.embeddings.shape)
            self.assertEqual((3, 3), loaded.embeddings.shape)
            self.assertTrue(any("target 1-1/3 start" in message for message in progress))
            self.assertTrue(any("target 3-3/3 done" in message for message in progress))
            self.assertTrue(any("loaded target index" in message for message in loaded_progress))
            self.assertTrue((root / "target_index" / "target_embeddings.npy").exists())
            self.assertTrue((root / "target_index" / "target_index.json").exists())

    def test_target_index_rejects_audio_mode_mismatch_in_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={
                    "model_path": "fake-e5",
                    "video_fps": 1,
                    "batch_size": 1,
                    "video_audio_mode": "off",
                    "load_audio_from_video": False,
                },
            )

            with self.assertRaisesRegex(ValueError, "do not reuse audio-off"):
                build_or_load_target_index(
                    triplets=triplets,
                    encoder=FailingE5Encoder(),
                    index_dir=root / "target_index",
                    runtime_info={
                        "model_path": "fake-e5",
                        "video_fps": 1,
                        "batch_size": 1,
                        "video_audio_mode": "on",
                        "load_audio_from_video": True,
                    },
                )

    def test_configure_video_processing_enables_audio_loading(self) -> None:
        class FakeProcessor:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(self, *args: object, **kwargs: object) -> object:
                self.calls.append(dict(kwargs))
                return kwargs

        class FakeModule:
            def __init__(self) -> None:
                self.processing_kwargs = {"video": {"fps": 99}, "chat_template": {"foo": "bar"}}
                self.processor = FakeProcessor()

        class FakeModel:
            def __init__(self) -> None:
                self.module = FakeModule()

            def __getitem__(self, index: int) -> FakeModule:
                return self.module

        model = FakeModel()
        patched = _configure_video_processing(model, max_pixels=123, fps=2, load_audio_from_video=True)

        self.assertEqual(
            {
                "max_pixels": 123,
                "do_sample_frames": True,
                "fps": 2,
                "load_audio_from_video": True,
                "use_audio_in_video": True,
            },
            model.module.processing_kwargs["video"],
        )
        self.assertEqual({"foo": "bar"}, model.module.processing_kwargs["chat_template"])
        self.assertTrue(patched)

        result = model.module.processor(videos_kwargs={"load_audio_from_video": True, "use_audio_in_video": True})
        self.assertEqual({"use_audio_in_video": True}, result["videos_kwargs"])

    def test_query_subset_uses_full_gallery_and_calculates_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            index = build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
            )

            progress: list[str] = []
            summary = run_eval_slice(
                triplets=triplets,
                target_index=index,
                encoder=FakeE5Encoder(),
                output_dir=root / "smoke20",
                sample_size=2,
                recall_ks=(1, 2, 3),
                topk_trace=3,
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
                progress=progress.append,
            )
            traces = [
                json.loads(line)
                for line in (root / "smoke20" / "traces.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(2, summary["query_count"])
            self.assertEqual(3, summary["gallery_count"])
            self.assertEqual({"R@1": 1.0, "R@2": 1.0, "R@3": 1.0}, summary["recall"])
            self.assertEqual(2, len(traces))
            self.assertEqual(1, traces[1]["target_rank"])
            self.assertNotIn("target_caption", traces[0])
            self.assertTrue(any("query 1/2 start" in message for message in progress))
            self.assertTrue(any("query 2/2 done rank=1" in message for message in progress))

    def test_query_mode_composed_passes_video_and_text_to_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            index = build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
            )
            encoder = CapturingE5Encoder()

            summary = run_eval_slice(
                triplets=triplets,
                target_index=index,
                encoder=encoder,
                output_dir=root / "composed",
                sample_size=1,
                recall_ks=(1, 5, 10),
                topk_trace=1,
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
                query_mode="composed",
            )

            self.assertIsInstance(encoder.calls[0][0], dict)
            self.assertIn("text", encoder.calls[0][0])
            self.assertEqual("composed", summary["query_mode"])
            self.assertTrue(summary["uses_edit_text_for_embedding"])

    def test_query_mode_video_only_passes_only_reference_video_to_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            index = build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
            )
            encoder = CapturingE5Encoder()

            summary = run_eval_slice(
                triplets=triplets,
                target_index=index,
                encoder=encoder,
                output_dir=root / "video_only",
                sample_size=1,
                recall_ks=(1, 5, 10),
                topk_trace=1,
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
                query_mode="video-only",
            )
            trace = json.loads((root / "video_only" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual([triplets[0].reference_video], encoder.calls[0])
            self.assertEqual("video-only", summary["query_mode"])
            self.assertEqual("reference_video", summary["query_input"])
            self.assertFalse(summary["uses_edit_text_for_embedding"])
            self.assertEqual("", summary["query_template"])
            self.assertEqual("video-only", trace["query_mode"])
            self.assertFalse(trace["query_used_text"])

    def test_reference_audio_muted_only_rewrites_reference_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            commands: list[list[str]] = []
            progress: list[str] = []

            def runner(command: list[str]) -> None:
                commands.append(command)
                Path(command[-1]).write_bytes(b"muted")

            def probe(path: Path) -> list[dict[str, str]]:
                return [{"codec_type": "video"}] if path.exists() else []

            prepared, summary = prepare_reference_audio_triplets(
                triplets=triplets,
                reference_audio_mode="muted",
                cache_dir=root / "muted_cache",
                output_dir=root / "run",
                command_runner=runner,
                stream_probe=probe,
                progress=progress.append,
            )

            self.assertEqual(3, len(prepared))
            self.assertNotEqual(triplets[0].reference_video, prepared[0].reference_video)
            self.assertEqual(triplets[0].target_video, prepared[0].target_video)
            self.assertIn("-map", commands[0])
            self.assertIn("0:v:0", commands[0])
            self.assertIn("-c:v", commands[0])
            self.assertIn("copy", commands[0])
            self.assertIn("-an", commands[0])
            self.assertEqual("reference_only", summary["audio_removed_scope"])
            self.assertTrue(summary["audio_removed"])
            self.assertEqual("strip", summary["reference_audio_transform"])
            self.assertEqual(3, summary["generated_count"])
            self.assertEqual(0, summary["reused_count"])
            self.assertTrue((root / "run" / "reference_muted_triplets.jsonl").exists())
            self.assertTrue((root / "run" / "reference_muted_media_manifest.jsonl").exists())
            self.assertTrue(any("reference-audio muted start" in message for message in progress))
            self.assertTrue(any("wrote reference muted triplets" in message for message in progress))

            def failing_runner(command: list[str]) -> None:
                raise AssertionError("muted reference should be reused from cache")

            reused, reused_summary = prepare_reference_audio_triplets(
                triplets=triplets,
                reference_audio_mode="muted",
                cache_dir=root / "muted_cache",
                output_dir=root / "run_reused",
                command_runner=failing_runner,
                stream_probe=probe,
            )

            self.assertEqual([item.reference_video for item in prepared], [item.reference_video for item in reused])
            self.assertEqual(0, reused_summary["generated_count"])
            self.assertEqual(3, reused_summary["reused_count"])

    def test_reference_audio_silent_keeps_audio_track_for_audio_on_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            commands: list[list[str]] = []

            def runner(command: list[str]) -> None:
                commands.append(command)
                Path(command[-1]).write_bytes(b"silent")

            def probe(path: Path) -> list[dict[str, str]]:
                return [{"codec_type": "video"}, {"codec_type": "audio"}] if path.exists() else []

            prepared, summary = prepare_reference_audio_triplets(
                triplets=triplets,
                reference_audio_mode="silent",
                cache_dir=root / "audio_cache",
                output_dir=root / "run",
                command_runner=runner,
                stream_probe=probe,
            )

            self.assertEqual(3, len(prepared))
            self.assertNotEqual(triplets[0].reference_video, prepared[0].reference_video)
            self.assertEqual(triplets[0].target_video, prepared[0].target_video)
            self.assertIn("anullsrc=channel_layout=stereo:sample_rate=16000", commands[0])
            self.assertIn("-c:a", commands[0])
            self.assertIn("aac", commands[0])
            self.assertEqual("silent", summary["reference_audio_mode"])
            self.assertEqual("silent", summary["reference_audio_transform"])
            self.assertTrue(summary["audio_removed"])
            self.assertTrue((root / "run" / "reference_silent_triplets.jsonl").exists())

    def test_composed_query_with_muted_reference_keeps_edit_text_and_original_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)

            def runner(command: list[str]) -> None:
                Path(command[-1]).write_bytes(b"muted")

            def probe(path: Path) -> list[dict[str, str]]:
                return [{"codec_type": "video"}] if path.exists() else []

            prepared, _ = prepare_reference_audio_triplets(
                triplets=triplets,
                reference_audio_mode="muted",
                cache_dir=root / "muted_cache",
                output_dir=root / "run",
                command_runner=runner,
                stream_probe=probe,
            )
            index = build_or_load_target_index(
                triplets=prepared,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
            )
            encoder = CapturingE5Encoder()

            summary = run_eval_slice(
                triplets=prepared,
                target_index=index,
                encoder=encoder,
                output_dir=root / "eval",
                sample_size=1,
                recall_ks=(1, 5, 10),
                topk_trace=1,
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
                query_mode="composed",
                reference_audio_mode="muted",
            )
            trace = json.loads((root / "eval" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])
            payload = encoder.calls[0][0]

            self.assertIsInstance(payload, dict)
            self.assertEqual(prepared[0].reference_video, payload["video"])
            self.assertNotEqual(triplets[0].reference_video, payload["video"])
            self.assertIn(triplets[0].edit_text, payload["text"])
            self.assertEqual(triplets[0].target_video, prepared[0].target_video)
            self.assertEqual(triplets[0].target_video, trace["target_video"])
            self.assertEqual("composed", summary["query_mode"])
            self.assertTrue(summary["uses_edit_text_for_embedding"])
            self.assertEqual("muted", summary["reference_audio_mode"])
            self.assertEqual("original", summary["target_audio_mode"])
            self.assertTrue(trace["query_used_text"])

    def test_summary_and_trace_mark_audio_enabled_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            runtime = {
                "model_path": "fake-e5",
                "video_fps": 1,
                "batch_size": 1,
                "video_audio_mode": "on",
                "load_audio_from_video": True,
                "use_audio_in_video": True,
                "processor_video_kwargs_sanitizer": True,
            }
            index = build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info=runtime,
            )

            summary = run_eval_slice(
                triplets=triplets,
                target_index=index,
                encoder=FakeE5Encoder(),
                output_dir=root / "eval",
                sample_size=1,
                recall_ks=(1, 5, 10),
                topk_trace=1,
                runtime_info=runtime,
            )
            trace = json.loads((root / "eval" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual("on", summary["video_audio_mode"])
            self.assertTrue(summary["load_audio_from_video"])
            self.assertTrue(summary["use_audio_in_video"])
            self.assertTrue(summary["processor_video_kwargs_sanitizer"])
            self.assertEqual("on", trace["video_audio_mode"])
            self.assertTrue(trace["load_audio_from_video"])
            self.assertTrue(trace["use_audio_in_video"])
            self.assertTrue(trace["processor_video_kwargs_sanitizer"])

    def test_trace_keeps_target_rank_and_topk_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = self._write_three_triplets(root)
            index = build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
            )

            run_eval_slice(
                triplets=triplets,
                target_index=index,
                encoder=FakeE5Encoder(),
                output_dir=root / "full3",
                sample_size=3,
                recall_ks=(1, 5, 10),
                topk_trace=2,
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
                reference_audio_mode="muted",
            )
            first_trace = json.loads((root / "full3" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual("sample1", first_trace["sample_id"])
            self.assertEqual(1, first_trace["target_rank"])
            self.assertIn("target_score", first_trace)
            self.assertEqual("muted", first_trace["reference_audio_mode"])
            self.assertEqual("original", first_trace["target_audio_mode"])
            self.assertEqual("reference_only", first_trace["audio_removed_scope"])
            self.assertTrue(first_trace["audio_removed"])
            self.assertEqual("sample1", first_trace["topk_hits"][0]["sample_id"])

    def _write_three_triplets(self, root: Path) -> list[E5CVRTriplet]:
        triplets = [
            E5CVRTriplet(
                sample_id="sample1",
                reference_video=str(root / "ref1.mp4"),
                target_video=str(root / "target1.mp4"),
                edit_text="make it target one",
                reference_caption="reference one",
            ),
            E5CVRTriplet(
                sample_id="sample2",
                reference_video=str(root / "ref2.mp4"),
                target_video=str(root / "target2.mp4"),
                edit_text="make it target two",
                reference_caption="reference two",
            ),
            E5CVRTriplet(
                sample_id="sample3",
                reference_video=str(root / "ref3.mp4"),
                target_video=str(root / "target3.mp4"),
                edit_text="make it target three",
                reference_caption="reference three",
            ),
        ]
        for triplet in triplets:
            Path(triplet.reference_video).write_bytes(b"ref")
            Path(triplet.target_video).write_bytes(b"target")
        return triplets


if __name__ == "__main__":
    unittest.main()
