from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.audio_cvr_audiovlm2vec import (
    IMAGE_TOKEN,
    LowRankResidualAdapter,
    _audio_cvr_records,
    _embedding_item,
    _metric_summary,
    audit_embeddings,
    encode_vlm2vec,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _FakeEncoder:
    init_count = 0
    encode_count = 0

    def __init__(self, *_args, **_kwargs) -> None:
        type(self).init_count += 1

    def encode(self, rows):
        type(self).encode_count += len(rows)
        return [
            np.full(8, (int(row["embedding_key"][:2], 16) + 1) / 256.0, dtype=np.float32)
            for row in rows
        ]


class AudioCVRAudioVLM2VecTests(unittest.TestCase):
    def test_audio_records_make_exact_target_reference_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for index in range(2):
                reference = root / f"reference_{index}.mp4"
                target = root / f"target_{index}.mp4"
                reference.write_bytes(b"reference")
                target.write_bytes(b"target")
                rows.append(
                    {
                        "sample_id": f"sample_{index}",
                        "reference_video": reference.name,
                        "target_video": target.name,
                        "edit_text": f"change sound {index}",
                        "b_subtype": "sound_event",
                        "dataset": "avatar",
                        "source_disjoint_group_id": f"source_{index}",
                    }
                )
            source = root / "test.jsonl"
            _write_jsonl(source, rows)
            output = root / "prepared"
            summary = _audio_cvr_records(
                source,
                output,
                [root],
                split_name="test",
                expected_count=2,
            )
            records = [json.loads(line) for line in (output / "test_records.jsonl").read_text().splitlines()]
            gallery = [json.loads(line) for line in (output / "test_gallery.jsonl").read_text().splitlines()]
            self.assertEqual(4, summary["gallery_count"])
            self.assertEqual(["target", "target", "reference", "reference"], [row["kind"] for row in gallery])
            self.assertEqual([0, 1], [row["positive_index"] for row in records])
            self.assertEqual([2, 3], [row["reference_index"] for row in records])
            self.assertEqual([0, 1, 2, 3], records[0]["candidate_indices"])

    def test_reference_mask_changes_only_own_reference(self) -> None:
        records = [
            {"positive_index": 0, "reference_index": 2, "candidate_indices": [0, 1, 2, 3]},
            {"positive_index": 1, "reference_index": 3, "candidate_indices": [0, 1, 2, 3]},
        ]
        scores = np.asarray(
            [
                [0.8, 0.1, 0.9, 0.2],
                [0.1, 0.7, 0.3, 0.95],
            ],
            dtype=np.float32,
        )
        with_ref, _ = _metric_summary(scores, records, mask_reference=False)
        without_ref, _ = _metric_summary(scores, records, mask_reference=True)
        self.assertEqual(0.0, with_ref["R@1"])
        self.assertEqual(1.0, without_ref["R@1"])
        self.assertEqual(4, with_ref["candidate_count_min"])
        self.assertEqual(3, without_ref["candidate_count_min"])

    def test_prompts_keep_modes_scientifically_distinct(self) -> None:
        common = {
            "dataset": "audiocvr",
            "split": "test",
            "role": "query",
            "item_id": "sample",
            "media_path": "/tmp/video.mp4",
            "edit_text": "replace piano with guitar",
            "caption": "a man speaks over piano music",
        }
        visual_text = _embedding_item(mode="V_T", **common)
        audio_text = _embedding_item(mode="V_A_T", **common)
        self.assertIn(IMAGE_TOKEN, visual_text["text"])
        self.assertNotIn("Reference audio:", visual_text["text"])
        self.assertIn("Reference audio:", audio_text["text"])
        self.assertNotEqual(visual_text["embedding_key"], audio_text["embedding_key"])

    def test_embedding_cache_is_atomic_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = root / "inventory.jsonl"
            rows = [
                {
                    "embedding_key": f"{index:064x}",
                    "dataset": "audiocvr",
                    "split": "test",
                    "mode": "V_T",
                    "role": "query",
                    "item_id": str(index),
                    "media_path": str(root / "video.mp4"),
                    "text": "prompt",
                }
                for index in range(4)
            ]
            _write_jsonl(inventory, rows)
            cache = root / "cache"
            _FakeEncoder.init_count = 0
            _FakeEncoder.encode_count = 0
            with mock.patch("app.audio_cvr_audiovlm2vec.VLM2VecEncoder", _FakeEncoder):
                for shard in range(2):
                    encode_vlm2vec(
                        inventory_path=inventory,
                        cache_dir=cache,
                        base_model=root,
                        adapter_model=root,
                        shard_index=shard,
                        shard_count=2,
                        device="cpu",
                        batch_size=2,
                        retries=1,
                    )
                first_count = _FakeEncoder.encode_count
                for shard in range(2):
                    summary = encode_vlm2vec(
                        inventory_path=inventory,
                        cache_dir=cache,
                        base_model=root,
                        adapter_model=root,
                        shard_index=shard,
                        shard_count=2,
                        device="cpu",
                        batch_size=2,
                        retries=1,
                    )
                    self.assertEqual(0, summary["encoded_count"])
            self.assertEqual(4, first_count)
            self.assertTrue(audit_embeddings(inventory, cache, root / "audit.json")["complete"])

    def test_low_rank_residual_is_identity_at_initialization(self) -> None:
        import torch

        adapter = LowRankResidualAdapter(16, 4, "cpu")
        value = torch.randn(5, 16)
        expected = torch.nn.functional.normalize(value, p=2, dim=-1)
        self.assertTrue(torch.allclose(expected, adapter.query(value), atol=1e-6))
        self.assertTrue(torch.allclose(expected, adapter.document(value), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
