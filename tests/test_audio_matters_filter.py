from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.audio_matters_filter import AudioFeature, filter_audio_matters_triplets
from app.e5_cvr_eval import E5CVRTriplet


class AudioMattersFilterTests(unittest.TestCase):
    def test_filters_visual_edit_with_similar_audio_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplets = [
                self._triplet(root, "keep", difference_type="object_presence", edit_text="add a red cup"),
                self._triplet(root, "audio", difference_type="audio_event", edit_text="add a red cup"),
                self._triplet(root, "text", difference_type="scene", edit_text="add background music"),
                self._triplet(root, "low", difference_type="attribute", edit_text="change the jacket to blue"),
            ]
            features = {
                "keep_ref.mp4": AudioFeature(np.asarray([1.0, 0.0], dtype=np.float32), rms=0.2, sample_count=16000),
                "keep_target.mp4": AudioFeature(np.asarray([1.0, 0.0], dtype=np.float32), rms=0.2, sample_count=16000),
                "audio_ref.mp4": AudioFeature(np.asarray([1.0, 0.0], dtype=np.float32), rms=0.2, sample_count=16000),
                "audio_target.mp4": AudioFeature(np.asarray([1.0, 0.0], dtype=np.float32), rms=0.2, sample_count=16000),
                "text_ref.mp4": AudioFeature(np.asarray([1.0, 0.0], dtype=np.float32), rms=0.2, sample_count=16000),
                "text_target.mp4": AudioFeature(np.asarray([1.0, 0.0], dtype=np.float32), rms=0.2, sample_count=16000),
                "low_ref.mp4": AudioFeature(np.asarray([1.0, 0.0], dtype=np.float32), rms=0.2, sample_count=16000),
                "low_target.mp4": AudioFeature(np.asarray([0.0, 1.0], dtype=np.float32), rms=0.2, sample_count=16000),
            }
            progress: list[str] = []

            summary = filter_audio_matters_triplets(
                triplets=triplets,
                triplets_jsonl=root / "triplets.jsonl",
                output_dir=root / "out",
                min_audio_anchor_score=0.85,
                audio_feature_loader=lambda path: features[Path(path).name],
                progress=progress.append,
            )
            accepted = [
                json.loads(line)
                for line in (root / "out" / "audio_matters_triplets.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rejected = [
                json.loads(line)
                for line in (root / "out" / "rejected_triplets.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(1, summary["accepted_count"])
            self.assertEqual(3, summary["rejected_count"])
            self.assertEqual("keep", accepted[0]["sample_id"])
            self.assertTrue(accepted[0]["audio_anchor_required"])
            self.assertEqual("visual", accepted[0]["edit_primary_modality"])
            self.assertFalse(accepted[0]["audio_primary_modality"])
            self.assertNotIn("target_caption", accepted[0])
            self.assertIn("non_visual_difference_type:audio_event", rejected[0]["audio_matters_reject_reasons"])
            self.assertTrue(any("edit_text_mentions_audio" in row["audio_matters_reject_reasons"] for row in rejected))
            self.assertTrue(any("audio_anchor_score_below_threshold" in row["audio_matters_reject_reasons"] for row in rejected))
            self.assertTrue(any("triplet 1/4 start" in message for message in progress))
            self.assertTrue(any("triplet 1/4 accepted" in message for message in progress))

    def test_rejects_missing_or_quiet_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triplet = self._triplet(root, "quiet", difference_type="action", edit_text="change the gesture to waving")

            summary = filter_audio_matters_triplets(
                triplets=[triplet],
                triplets_jsonl=root / "triplets.jsonl",
                output_dir=root / "out",
                min_audio_anchor_score=0.85,
                min_rms=0.1,
                audio_feature_loader=lambda _path: AudioFeature(
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    rms=0.001,
                    sample_count=16000,
                ),
            )
            rejected = json.loads((root / "out" / "rejected_triplets.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(0, summary["accepted_count"])
            self.assertIn("audio_too_quiet_or_missing", rejected["audio_matters_reject_reasons"])

    def _triplet(self, root: Path, sample_id: str, *, difference_type: str, edit_text: str) -> E5CVRTriplet:
        reference = root / f"{sample_id}_ref.mp4"
        target = root / f"{sample_id}_target.mp4"
        reference.write_bytes(b"ref")
        target.write_bytes(b"target")
        return E5CVRTriplet(
            sample_id=sample_id,
            reference_video=str(reference),
            target_video=str(target),
            edit_text=edit_text,
            reference_caption="reference caption",
            source="daily_omni",
            difference_type=difference_type,
        )


if __name__ == "__main__":
    unittest.main()
