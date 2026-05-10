import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.audio_matters_natural import (
    AudioFeature,
    export_audio_matters_triplets,
    mine_audio_matters_candidates,
)


class AudioMattersNaturalTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _make_root(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        (root / "clips").mkdir(parents=True)
        for name in ("ref.mp4", "target.mp4", "neg1.mp4", "neg2.mp4"):
            (root / "clips" / name).write_bytes(name.encode("utf-8"))
        return temp_dir, root

    def _annotations(self) -> list[dict]:
        base = {
            "dataset": "daily_omni",
            "scene": "studio desk",
            "actions": ["speaking"],
            "audio_events": ["steady narration"],
            "speech": ["hello"],
        }
        return [
            {
                **base,
                "clip_id": "ref",
                "output_path": "clips/ref.mp4",
                "summary": "speaker at a desk",
                "object_counts": {"speaker": 1},
                "attributes": ["blue background"],
            },
            {
                **base,
                "clip_id": "target",
                "output_path": "clips/target.mp4",
                "summary": "speaker at a desk with a product",
                "object_counts": {"speaker": 1, "product": 1},
                "attributes": ["blue background"],
            },
            {
                **base,
                "clip_id": "neg1",
                "output_path": "clips/neg1.mp4",
                "summary": "speaker with a chart",
                "object_counts": {"speaker": 1, "chart": 1},
                "attributes": ["blue background"],
            },
            {
                **base,
                "clip_id": "neg2",
                "output_path": "clips/neg2.mp4",
                "summary": "speaker with a cup",
                "object_counts": {"speaker": 1, "cup": 1},
                "attributes": ["blue background"],
            },
        ]

    def test_mines_natural_audio_anchor_visual_candidate(self) -> None:
        temp_dir, root = self._make_root()
        with temp_dir:
            annotations_path = root / "annotations.jsonl"
            groups_path = root / "groups.jsonl"
            output_path = root / "audio_matters_mined_candidates.jsonl"
            report_path = root / "audio_matters_mining_report.md"
            self._write_jsonl(annotations_path, self._annotations())
            self._write_jsonl(
                groups_path,
                [
                    {
                        "group_id": "daily_source_a",
                        "group_reason": "same_source_video",
                        "candidate_clip_ids": ["ref", "target", "neg1", "neg2"],
                    }
                ],
            )

            def loader(path: Path) -> AudioFeature:
                if path.name in {"ref.mp4", "target.mp4"}:
                    return AudioFeature(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), 0.2, 8.0, 128000)
                if path.name == "neg1.mp4":
                    return AudioFeature(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), 0.2, 8.0, 128000)
                return AudioFeature(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), 0.2, 8.0, 128000)

            summary = mine_audio_matters_candidates(
                root=root,
                clip_annotations_path=annotations_path,
                clip_groups_path=groups_path,
                output_path=output_path,
                report_path=report_path,
                max_candidates=10,
                min_audio_anchor_score=0.95,
                audio_feature_loader=loader,
            )

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["selected_count"])
            self.assertEqual(1, len(records))
            self.assertEqual("ref", records[0]["reference_clip_id"])
            self.assertEqual("target", records[0]["target_clip_id"])
            self.assertEqual("object_presence", records[0]["difference"]["type"])
            self.assertTrue(records[0]["source_context"]["audio_anchor_required"])
            self.assertEqual("visual", records[0]["quality"]["edit_primary_modality"])
            self.assertNotIn("target_caption", records[0])
            self.assertIn("Audio-Matters Natural Candidate Mining Report", report_path.read_text(encoding="utf-8"))

    def test_rejects_low_audio_similarity(self) -> None:
        temp_dir, root = self._make_root()
        with temp_dir:
            annotations_path = root / "annotations.jsonl"
            groups_path = root / "groups.jsonl"
            output_path = root / "audio_matters_mined_candidates.jsonl"
            report_path = root / "audio_matters_mining_report.md"
            self._write_jsonl(annotations_path, self._annotations())
            self._write_jsonl(
                groups_path,
                [{"group_id": "g", "candidate_clip_ids": ["ref", "target", "neg1", "neg2"]}],
            )

            def loader(path: Path) -> AudioFeature:
                vectors = {
                    "ref.mp4": [1.0, 0.0, 0.0, 0.0],
                    "target.mp4": [0.0, 1.0, 0.0, 0.0],
                    "neg1.mp4": [0.0, 0.0, 1.0, 0.0],
                    "neg2.mp4": [0.0, 0.0, 0.0, 1.0],
                }
                vector = np.asarray(vectors[path.name], dtype=np.float32)
                return AudioFeature(vector, 0.2, 8.0, 128000)

            summary = mine_audio_matters_candidates(
                root=root,
                clip_annotations_path=annotations_path,
                clip_groups_path=groups_path,
                output_path=output_path,
                report_path=report_path,
                max_candidates=10,
                min_audio_anchor_score=0.95,
                audio_feature_loader=loader,
            )

            self.assertEqual(0, summary["selected_count"])
            self.assertEqual("", output_path.read_text(encoding="utf-8"))
            self.assertGreater(summary["rejection_counts"].get("low_audio_anchor_score", 0), 0)

    def test_exports_triplets_without_target_caption(self) -> None:
        temp_dir, root = self._make_root()
        with temp_dir:
            accepted_path = root / "accepted_audio_matters_pairs.jsonl"
            output_path = root / "audio_matters_triplets.jsonl"
            summary_path = root / "audio_matters_triplets_summary.json"
            self._write_jsonl(
                accepted_path,
                [
                    {
                        "proposal_id": "pair_a",
                        "reference_video": "clips/ref.mp4",
                        "target_video": "clips/target.mp4",
                        "edit_text": "add a product beside the speaker",
                        "reference_caption": "speaker at a desk",
                        "target_caption": "speaker at a desk with a product",
                        "difference": {"type": "object_presence"},
                        "accepted": True,
                        "reference_clip_id": "ref",
                        "target_clip_id": "target",
                        "heuristic_quality": {
                            "audio_anchor_score": 0.98,
                            "audio_anchor_type": "similar_or_same_natural_audio",
                        },
                    }
                ],
            )

            summary = export_audio_matters_triplets(
                root=root,
                accepted_pairs_path=accepted_path,
                output_path=output_path,
                summary_path=summary_path,
            )

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, summary["output_count"])
            self.assertEqual("pair_a", records[0]["sample_id"])
            self.assertTrue(Path(records[0]["reference_video"]).is_absolute())
            self.assertTrue(Path(records[0]["target_video"]).is_absolute())
            self.assertEqual(0.98, records[0]["audio_anchor_score"])
            self.assertNotIn("target_caption", records[0])
            self.assertFalse(json.loads(summary_path.read_text(encoding="utf-8"))["contains_target_caption"])


if __name__ == "__main__":
    unittest.main()
