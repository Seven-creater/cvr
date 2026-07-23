from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.audio_cvr_external_baseline import (
    MODES,
    _metric_summary,
    _mode_embeddings,
    assemble_embeddings,
    audit_cache,
    build_delta_inventory,
    cache_imagebind,
    cache_imagebind_bundle,
    evaluate_embeddings,
    prepare_inventory,
    summarize_results,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _unit(index: int) -> np.ndarray:
    value = np.zeros(1024, dtype=np.float32)
    value[index] = 1.0
    return value


class _FakeEncoder:
    init_count = 0

    def __init__(self, *_args, **_kwargs):
        type(self).init_count += 1

    def encode_media(self, paths):
        values = []
        for path in paths:
            index = sum(path.read_bytes()) % 32
            values.append((_unit(index), _unit(index + 32)))
        return values

    def encode_text(self, texts):
        return [_unit((sum(text.encode("utf-8")) % 32) + 64) for text in texts]


class AudioCVRExternalBaselineTests(unittest.TestCase):
    def _fixture(self, root: Path, count: int = 2) -> tuple[Path, list[dict]]:
        rows = []
        for index in range(count):
            reference = root / f"ref_{index}.mp4"
            target = root / f"target_{index}.mp4"
            visual = root / f"visual_{index}.mp4"
            audio = root / f"audio_{index}.mp4"
            asr = root / f"asr_{index}.mp4"
            for offset, path in enumerate((reference, target, visual, audio, asr), start=1):
                path.write_bytes(bytes([index * 10 + offset]))
            rows.append(
                {
                    "sample_id": f"sample_{index}",
                    "reference_video": reference.name,
                    "target_video": target.name,
                    "edit_text": f"change sound {index}",
                    "b_subtype": "sound_event" if index == 0 else "music",
                    "dataset": "avatar",
                    "source_disjoint_group_id": f"source_{index}",
                    "pair_group_id": f"pair_{index}",
                    "hard_negatives": [
                        {"type": "visual_hard", "video": visual.name},
                        {"type": "audio_hard", "video": audio.name},
                        {"type": "asr_hard", "video": asr.name},
                    ],
                }
            )
        records = root / "records.jsonl"
        _write_jsonl(records, rows)
        return records, rows

    def _model_dir(self, root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"weights")
        return model

    def test_prepare_inventory_deduplicates_media_and_checks_inheritance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records, rows = self._fixture(root)
            rows[1]["target_video"] = rows[0]["target_video"]
            _write_jsonl(records, rows)
            inventory = root / "inventory"
            summary = prepare_inventory(records, inventory, [root], expected_count=2)
            self.assertEqual(9, summary["media_count"])
            self.assertEqual(2, summary["text_count"])
            inherited = root / "inherited.jsonl"
            _write_jsonl(inherited, [rows[0]])
            prepare_inventory(records, root / "inventory_final", [root], inherited_records=inherited)

    def test_prepare_inventory_rejects_missing_inherited_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records, _ = self._fixture(root, count=1)
            inherited = root / "inherited.jsonl"
            _write_jsonl(inherited, [{"sample_id": "missing"}])
            with self.assertRaisesRegex(ValueError, "do not inherit"):
                prepare_inventory(records, root / "inventory", [root], inherited_records=inherited)

    def test_content_cache_is_sharded_atomic_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records, _ = self._fixture(root)
            inventory = root / "inventory"
            prepare_inventory(records, inventory, [root])
            cache = root / "cache"
            model = self._model_dir(root)
            _FakeEncoder.init_count = 0
            with mock.patch("app.audio_cvr_external_baseline.ImageBindEncoder", _FakeEncoder):
                for shard in range(2):
                    cache_imagebind(inventory / "media_inventory.jsonl", cache, model, root, kind="media", shard_index=shard, shard_count=2, device="cpu", batch_size=2, retries=2)
                    cache_imagebind(inventory / "text_inventory.jsonl", cache, model, root, kind="text", shard_index=shard, shard_count=2, device="cpu", batch_size=2, retries=2)
                first_init_count = _FakeEncoder.init_count
                for shard in range(2):
                    summary = cache_imagebind(inventory / "media_inventory.jsonl", cache, model, root, kind="media", shard_index=shard, shard_count=2, device="cpu", batch_size=2, retries=2)
                    self.assertEqual(0, summary["encoded_count"])
                self.assertEqual(first_init_count, _FakeEncoder.init_count)
            audit = audit_cache(inventory, cache, root / "audit.json")
            self.assertTrue(audit["complete"])
            self.assertEqual(10, audit["media"]["complete_count"])
            self.assertEqual(2, audit["text"]["complete_count"])

    def test_bundle_loads_imagebind_once_for_media_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records, _ = self._fixture(root)
            inventory = root / "inventory"
            prepare_inventory(records, inventory, [root])
            cache = root / "cache"
            model = self._model_dir(root)
            _FakeEncoder.init_count = 0
            with mock.patch("app.audio_cvr_external_baseline.ImageBindEncoder", _FakeEncoder):
                summary = cache_imagebind_bundle(
                    inventory / "media_inventory.jsonl",
                    inventory / "text_inventory.jsonl",
                    cache,
                    model,
                    root,
                    shard_index=0,
                    shard_count=1,
                    device="cpu",
                    batch_size=2,
                    retries=2,
                )
            self.assertEqual(1, _FakeEncoder.init_count)
            self.assertEqual("both", summary["kind"])
            self.assertEqual(12, summary["encoded_count"])
            self.assertTrue(audit_cache(inventory, cache, root / "audit.json")["complete"])

    def test_delta_inventory_contains_only_new_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pre_records, pre_rows = self._fixture(root, count=1)
            pre = root / "pre"
            prepare_inventory(pre_records, pre, [root])
            extra_root = root / "extra"
            extra_root.mkdir()
            final_records, final_rows = self._fixture(extra_root, count=1)
            final_rows[0]["sample_id"] = "sample_new"
            final_rows[0]["edit_text"] = "replace the background sound with music"
            for key in ("reference_video", "target_video"):
                old = extra_root / final_rows[0][key]
                new = old.with_name(f"new_{old.name}")
                old.rename(new)
                final_rows[0][key] = new.name
            for negative in final_rows[0]["hard_negatives"]:
                old = extra_root / negative["video"]
                new = old.with_name(f"new_{old.name}")
                old.rename(new)
                negative["video"] = new.name
            _write_jsonl(final_records, final_rows)
            combined = root / "combined.jsonl"
            _write_jsonl(combined, pre_rows + final_rows)
            final = root / "final"
            prepare_inventory(combined, final, [root, extra_root], inherited_records=pre_records)
            summary = build_delta_inventory(pre, final, root / "delta")
            self.assertGreater(summary["media"]["delta_count"], 0)
            self.assertEqual(1, summary["text"]["delta_count"])

    def test_exact_reference_masking_masks_one_item_per_query(self) -> None:
        scores = np.asarray([[0.8, 0.1, 0.9, 0.0], [0.0, 0.7, 0.2, 0.8]], dtype=np.float32)
        positives = np.asarray([0, 1])
        references = np.asarray([2, 3])
        with_ref, _ = _metric_summary(scores, positives, references, mask_own_reference=False)
        without_ref, _ = _metric_summary(scores, positives, references, mask_own_reference=True)
        self.assertEqual(0.0, with_ref["R@1"])
        self.assertEqual(1.0, without_ref["R@1"])
        self.assertEqual(4, with_ref["effective_gallery_count_per_query"])
        self.assertEqual(3, without_ref["effective_gallery_count_per_query"])

    def test_seven_mode_formulas_are_normalized(self) -> None:
        data = {
            "query_vision": np.stack([_unit(0), _unit(1)]),
            "query_audio": np.stack([_unit(2), _unit(3)]),
            "query_text": np.stack([_unit(4), _unit(5)]),
            "gallery_vision": np.stack([_unit(0), _unit(1), _unit(2), _unit(3)]),
            "gallery_audio": np.stack([_unit(4), _unit(5), _unit(6), _unit(7)]),
        }
        for mode in MODES:
            query, gallery = _mode_embeddings(data, mode)
            np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1.0, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(gallery, axis=1), 1.0, atol=1e-6)

    def test_end_to_end_assembly_evaluation_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records, _ = self._fixture(root)
            inventory = root / "inventory"
            prepare_inventory(records, inventory, [root])
            cache = root / "cache"
            model = self._model_dir(root)
            with mock.patch("app.audio_cvr_external_baseline.ImageBindEncoder", _FakeEncoder):
                cache_imagebind(inventory / "media_inventory.jsonl", cache, model, root, kind="media", shard_index=0, shard_count=1, device="cpu", batch_size=2, retries=2)
                cache_imagebind(inventory / "text_inventory.jsonl", cache, model, root, kind="text", shard_index=0, shard_count=1, device="cpu", batch_size=2, retries=2)
            assembly = root / "assembly"
            summary = assemble_embeddings(records, inventory, cache, assembly, max_exclusion_rate=0.0)
            self.assertEqual(2, summary["valid_query_count"])
            self.assertEqual(4, summary["gallery_count"])
            evaluation = root / "evaluation"
            evaluate_embeddings(assembly, evaluation, topk=2)
            results = json.loads((evaluation / "seven_mode_results.json").read_text(encoding="utf-8"))
            self.assertEqual(set(MODES), set(results))
            statistics = root / "statistics"
            summarize_results(evaluation, statistics, iterations=100, seed=7)
            self.assertTrue((statistics / "paper_results.md").is_file())
            self.assertEqual("COMPLETE", json.loads((statistics / "statistics_summary.json").read_text(encoding="utf-8"))["state"])


if __name__ == "__main__":
    unittest.main()
