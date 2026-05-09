from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.e5_cvr_eval import (
    E5CVRTriplet,
    TargetRecord,
    build_or_load_target_index,
    load_triplets_jsonl,
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
            index = build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
            )
            loaded = build_or_load_target_index(
                triplets=triplets,
                encoder=FakeE5Encoder(),
                index_dir=root / "target_index",
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
            )

            self.assertEqual(["sample1", "sample2", "sample3"], [record.sample_id for record in index.records])
            self.assertEqual((3, 3), index.embeddings.shape)
            self.assertEqual((3, 3), loaded.embeddings.shape)
            self.assertTrue((root / "target_index" / "target_embeddings.npy").exists())
            self.assertTrue((root / "target_index" / "target_index.json").exists())

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

            summary = run_eval_slice(
                triplets=triplets,
                target_index=index,
                encoder=FakeE5Encoder(),
                output_dir=root / "smoke20",
                sample_size=2,
                recall_ks=(1, 2, 3),
                topk_trace=3,
                runtime_info={"model_path": "fake-e5", "video_fps": 1},
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
            )
            first_trace = json.loads((root / "full3" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual("sample1", first_trace["sample_id"])
            self.assertEqual(1, first_trace["target_rank"])
            self.assertIn("target_score", first_trace)
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
