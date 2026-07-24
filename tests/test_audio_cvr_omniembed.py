from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.audio_cvr_omniembed import (
    CONDITIONS,
    MODES,
    _embedding_item,
    _message,
    _normalize_omnicvr,
    audit_cache,
    encode_inventory,
    evaluate,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _unit(index: int, dimension: int = 8) -> np.ndarray:
    value = np.zeros(dimension, dtype=np.float32)
    value[index] = 1.0
    return value


class _FakeEncoder:
    init_count = 0

    def __init__(self, **_kwargs) -> None:
        type(self).init_count += 1

    def encode(self, item: dict) -> np.ndarray:
        return _unit(int(item["item_id"].split("_")[-1]) % 8)


class AudioCVROmniEmbedTests(unittest.TestCase):
    def test_omnicvr_gallery_ids_are_resolved_to_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media = root / "video.mp4"
            media.write_bytes(b"video")
            gallery = [
                {
                    "gallery_id": f"gallery_{index}",
                    "video": str(media),
                }
                for index in range(2000)
            ]
            records = [
                {
                    "sample_id": f"query_{index}",
                    "reference_video": str(media),
                    "edit_text": "change the sound",
                    "positive_gallery_id": f"gallery_{index}",
                    "reference_gallery_id": f"gallery_{index + 1000}",
                    "candidate_gallery_ids": [
                        f"gallery_{index}",
                        f"gallery_{index + 1000}",
                    ],
                }
                for index in range(1000)
            ]
            records_path = root / "records.jsonl"
            gallery_path = root / "gallery.jsonl"
            _write_jsonl(records_path, records)
            _write_jsonl(gallery_path, gallery)

            normalized_records, normalized_gallery = _normalize_omnicvr(
                records_path,
                gallery_path,
                [root],
            )
            self.assertEqual(2000, len(normalized_gallery))
            self.assertEqual(0, normalized_records[0]["positive_index"])
            self.assertEqual(1000, normalized_records[0]["reference_index"])
            self.assertEqual([0, 1000], normalized_records[0]["candidate_indices"])

    def test_fixed_message_separates_audio_modes(self) -> None:
        item = {
            "mode": "V_T",
            "role": "query",
            "media_path": "reference.mp4",
            "edit_text": "replace piano with guitar",
        }
        message, use_audio = _message(item)
        self.assertFalse(use_audio)
        self.assertIn("same visual context", message[0]["content"][1]["text"])
        item["mode"] = "V_A_T"
        _, use_audio = _message(item)
        self.assertTrue(use_audio)

    def test_encoding_is_item_atomic_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media = root / "video.mp4"
            media.write_bytes(b"video")
            rows = [
                {
                    "embedding_key": f"key_{index}",
                    "dataset": "audiocvr",
                    "mode": "V_T",
                    "condition": "exact",
                    "role": "document",
                    "item_id": f"item_{index}",
                    "media_path": str(media),
                    "edit_text": "",
                }
                for index in range(4)
            ]
            inventory = root / "inventory.jsonl"
            _write_jsonl(inventory, rows)
            cache = root / "cache"
            encoder = _FakeEncoder()
            first = encode_inventory(
                inventory_path=inventory,
                cache_dir=cache,
                base_model=root,
                adapter_model=root,
                shard_index=0,
                shard_count=1,
                device="cpu",
                retries=2,
                torch_dtype="float32",
                attn_implementation="sdpa",
                encoder=encoder,
            )
            second = encode_inventory(
                inventory_path=inventory,
                cache_dir=cache,
                base_model=root,
                adapter_model=root,
                shard_index=0,
                shard_count=1,
                device="cpu",
                retries=2,
                torch_dtype="float32",
                attn_implementation="sdpa",
                encoder=encoder,
            )
            self.assertEqual(4, first["encoded_count"])
            self.assertEqual(4, second["reused_count"])
            self.assertTrue(
                audit_cache(
                    inventory_path=inventory,
                    cache_dir=cache,
                    output_path=root / "audit.json",
                )["complete"]
            )

    def test_evaluation_reuses_score_matrix_for_exact_masking_and_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            cache = root / "cache"
            media = root / "video.mp4"
            media.write_bytes(b"video")
            inventory_rows = []

            for dataset in ("audiocvr", "omnicvr"):
                records = []
                gallery = []
                for index in range(2):
                    sample_id = f"{dataset}_{index}"
                    records.append(
                        {
                            "sample_id": sample_id,
                            "reference_video": str(media),
                            "edit_text": f"change {index}",
                            "positive_index": index,
                            "reference_index": index + 2,
                            "candidate_indices": [0, 1, 2, 3],
                        }
                    )
                    gallery.append(
                        {
                            "gallery_id": f"{dataset}::target::{sample_id}",
                            "kind": "target",
                            "media_path": str(media),
                        }
                    )
                for index in range(2):
                    sample_id = f"{dataset}_{index}"
                    gallery.append(
                        {
                            "gallery_id": f"{dataset}::reference::{sample_id}",
                            "kind": "reference",
                            "media_path": str(media),
                        }
                    )
                _write_jsonl(records_dir / f"{dataset}_records.jsonl", records)
                _write_jsonl(records_dir / f"{dataset}_gallery.jsonl", gallery)

                conditions = CONDITIONS if dataset == "audiocvr" else ("exact",)
                for mode in MODES:
                    for index, record in enumerate(records):
                        item = _embedding_item(
                            dataset=dataset,
                            mode=mode,
                            condition="exact",
                            role="query",
                            item_id=record["sample_id"],
                            media_path=str(media),
                            edit_text=record["edit_text"],
                        )
                        inventory_rows.append(item)
                        path = cache / "items" / f"{item['embedding_key']}.npy"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(path, _unit(index))
                    for condition in conditions:
                        for index, gallery_row in enumerate(gallery):
                            item_condition = (
                                condition
                                if dataset == "audiocvr"
                                and gallery_row["kind"] == "reference"
                                else "exact"
                            )
                            item = _embedding_item(
                                dataset=dataset,
                                mode=mode,
                                condition=item_condition,
                                role="document",
                                item_id=gallery_row["gallery_id"],
                                media_path=str(media),
                            )
                            if not any(
                                old["embedding_key"] == item["embedding_key"]
                                for old in inventory_rows
                            ):
                                inventory_rows.append(item)
                                path = cache / "items" / f"{item['embedding_key']}.npy"
                                path.parent.mkdir(parents=True, exist_ok=True)
                                vector = _unit(index if index < 2 else index - 2)
                                np.save(path, vector)

            inventory = root / "inventory.jsonl"
            _write_jsonl(inventory, inventory_rows)
            summary = evaluate(
                records_dir=records_dir,
                inventory_path=inventory,
                cache_dir=cache,
                output_dir=root / "evaluation",
            )
            self.assertTrue(summary["masking_reuses_same_score_matrix"])
            results = json.loads(
                (root / "evaluation" / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(CONDITIONS), set(results["audiocvr"]["V_T"]))
            self.assertEqual(2, results["omnicvr"]["V_A_T"]["exact"]["with_reference"]["query_count"])


if __name__ == "__main__":
    unittest.main()
