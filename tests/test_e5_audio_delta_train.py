from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.e5_audio_delta_train import (
    DEFAULT_DATA_ROOT,
    build_splits,
    cache_embeddings,
    eval_adapter,
    load_audio_delta_records,
    prepare_records,
    run_ablations,
    train_adapter,
    train_lora_plan,
    _video_payload,
)


class E5AudioDeltaTrainTests(unittest.TestCase):
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
            self.assertTrue((root / "ablations" / "comparison.md").exists())

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
            "audio_delta_hard_negatives": [
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


if __name__ == "__main__":
    unittest.main()
