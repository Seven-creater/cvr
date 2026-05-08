from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from app.composed_triplets import build_and_write_triplets, build_triplets, write_triplet_outputs


class ComposedTripletTests(unittest.TestCase):
    def test_builds_triplet_jsonl_csv_and_summary_without_target_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root / "00002_worldsense_case", source="worldsense", target_caption="SECRET target")
            self._write_sample(root / "00001_daily_case", source="daily_omni", edit_text="add a red cup")

            triplets, invalids, summary = build_triplets(root, expected_count=2)
            output_dir = root / "out"
            write_triplet_outputs(output_dir=output_dir, triplets=triplets, invalids=invalids, summary=summary)

            jsonl_lines = (output_dir / "triplets.jsonl").read_text(encoding="utf-8").splitlines()
            csv_text = (output_dir / "triplets.csv").read_text(encoding="utf-8")
            summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual([], invalids)
        self.assertEqual(2, len(jsonl_lines))
        self.assertEqual("00001_daily_case", json.loads(jsonl_lines[0])["sample_id"])
        self.assertIn("reference_video", jsonl_lines[0])
        self.assertIn("target_video", jsonl_lines[0])
        self.assertIn("edit_text", jsonl_lines[0])
        self.assertNotIn("target_caption", jsonl_lines[0])
        self.assertIn("sample_id,reference_video,target_video,edit_text", csv_text)
        self.assertEqual(2, summary_payload["valid_triplets"])
        self.assertEqual({"daily_omni": 1, "worldsense": 1, "unknown": 0}, summary_payload["dataset_counts"])

    def test_strict_command_fails_and_writes_invalid_samples_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root / "00001_good")
            self._write_sample(root / "00002_bad", include_target=False)
            output_dir = root / "out"
            args = argparse.Namespace(dataset_root=str(root), output_dir=str(output_dir), expected_count=2)

            with self.assertRaises(SystemExit):
                build_and_write_triplets(args)

            invalid_lines = (output_dir / "invalid_samples.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(1, len(invalid_lines))
        self.assertEqual("00002_bad", json.loads(invalid_lines[0])["sample_id"])
        self.assertIn("target.mp4", json.loads(invalid_lines[0])["reason"])

    def test_strict_command_fails_on_empty_edit_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root / "00001_bad", edit_text="")
            output_dir = root / "out"
            args = argparse.Namespace(dataset_root=str(root), output_dir=str(output_dir), expected_count=1)

            with self.assertRaises(SystemExit):
                build_and_write_triplets(args)

            invalid = json.loads((output_dir / "invalid_samples.jsonl").read_text(encoding="utf-8"))

        self.assertEqual("00001_bad", invalid["sample_id"])
        self.assertIn("edit_text.txt is empty", invalid["reason"])

    def test_reference_annotation_summary_is_caption_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root / "00001_case", reference_caption="", annotation_summary="fallback summary")

            triplets, invalids, _summary = build_triplets(root, expected_count=1)

        self.assertEqual([], invalids)
        self.assertEqual("fallback summary", triplets[0].reference_caption)

    def test_expected_count_mismatch_fails_after_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root / "00001_case")
            output_dir = root / "out"
            args = argparse.Namespace(dataset_root=str(root), output_dir=str(output_dir), expected_count=2)

            with self.assertRaises(SystemExit):
                build_and_write_triplets(args)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(1, summary["valid_triplets"])
        self.assertEqual(2, summary["expected_count"])

    def _write_sample(
        self,
        sample_dir: Path,
        *,
        source: str = "daily_omni",
        edit_text: str = "change one cat into two cats",
        reference_caption: str = "one cat on a sofa",
        target_caption: str = "two cats on a sofa",
        annotation_summary: str = "",
        include_target: bool = True,
    ) -> None:
        sample_dir.mkdir(parents=True)
        (sample_dir / "reference.mp4").write_bytes(b"ref")
        if include_target:
            (sample_dir / "target.mp4").write_bytes(b"target")
        (sample_dir / "edit_text.txt").write_text(edit_text + ("\n" if edit_text else ""), encoding="utf-8")
        (sample_dir / "info.json").write_text(
            json.dumps(
                {
                    "edit_text": edit_text,
                    "reference_caption": reference_caption,
                    "target_caption": target_caption,
                    "source": source,
                    "difference_type": "object_presence",
                    "accepted": True,
                    "final_omni_accept": True,
                    "final_omni_quality_score": 0.92,
                    "reference_clip_id": "ref-clip",
                    "target_clip_id": "target-clip",
                }
            ),
            encoding="utf-8",
        )
        if annotation_summary:
            (sample_dir / "reference_annotation.json").write_text(
                json.dumps({"summary": annotation_summary}),
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main()
