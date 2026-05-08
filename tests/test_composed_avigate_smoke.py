from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.composed_avigate_smoke import (
    load_composed_triplets,
    run_baseline,
    stage_triplets,
    write_comparison,
)
from app.retrieval_types import TextRow, VideoRow


@dataclass
class FakeRuntime:
    text_rows: list[TextRow]
    video_rows: list[VideoRow]
    score_map: dict[str, np.ndarray]
    audio_available: bool = True

    def score_text_query(self, query_text: str) -> np.ndarray:
        return self.score_map[query_text]


class ComposedAvigateSmokeTests(unittest.TestCase):
    def test_load_triplets_builds_caption_edit_query_without_target_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(
                root / "00001_sample",
                reference_caption="a person plays guitar",
                target_caption="SECRET target answer",
                edit_text="add a second musician",
            )

            triplets = load_composed_triplets(root, sample_size=1)

        self.assertEqual(1, len(triplets))
        self.assertEqual("00001_sample", triplets[0].sample_id)
        self.assertEqual("a person plays guitar. Edit: add a second musician.", triplets[0].query_text)
        self.assertNotIn("SECRET target answer", triplets[0].query_text)

    def test_stage_triplets_writes_avigate_compatible_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root / "00001_sample", reference_caption="one cat", edit_text="change one cat into two cats")
            triplet = load_composed_triplets(root, sample_size=1)[0]

            report = stage_triplets(
                [triplet],
                staged_root=root / "staged",
                extract_audio=False,
                link_mode="copy",
            )

            split_csv = (root / "staged" / "split.csv").read_text(encoding="utf-8")
            triplets_jsonl = (root / "staged" / "triplets.jsonl").read_text(encoding="utf-8")
            staged_video_exists = Path(report["video_root"], "00001_sample.mp4").exists()
            data_json = json.loads(Path(report["data_json"]).read_text(encoding="utf-8"))

        self.assertEqual(1, report["sample_count"])
        self.assertIn("video_id,sentence", split_csv)
        self.assertIn("00001_sample,one cat. Edit: change one cat into two cats.", split_csv)
        self.assertTrue(staged_video_exists)
        self.assertEqual({}, data_json)
        self.assertEqual("00001_sample", json.loads(triplets_jsonl)["sample_id"])

    def test_run_baseline_writes_traces_and_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            runtime = FakeRuntime(
                text_rows=[
                    TextRow(text_id="s1::caption::1", video_id="s1", text="query one"),
                    TextRow(text_id="s2::caption::1", video_id="s2", text="query two"),
                ],
                video_rows=[
                    VideoRow(video_id="s1", video_path="/tmp/s1.mp4"),
                    VideoRow(video_id="s2", video_path="/tmp/s2.mp4"),
                ],
                score_map={
                    "query one": np.asarray([0.2, 0.9], dtype=np.float32),
                    "query two": np.asarray([0.1, 0.8], dtype=np.float32),
                },
            )

            summary = run_baseline(runtime, recall_ks=(1, 2), topk=2, output_dir=output_root)
            traces = (output_root / "baseline_traces.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual({"R@1": 0.5, "R@2": 1.0}, summary["t2v"])
        self.assertEqual(2, len(traces))
        self.assertEqual(2, json.loads(traces[0])["target_rank"])
        self.assertEqual(1, json.loads(traces[1])["target_rank"])

    def test_write_comparison_includes_baseline_and_agent_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison = write_comparison(
                run_root=root,
                staged_root=root / "staged",
                baseline_summary={"runs": 20, "t2v": {"R@1": 0.1, "R@5": 0.4, "R@10": 0.6}},
                agent_summary={
                    "round1_recall": {"R@1": 0.1, "R@5": 0.4, "R@10": 0.6},
                    "final_recall": {"R@1": 0.2, "R@5": 0.45, "R@10": 0.65},
                },
                checker_model="qwen2.5-omni",
            )
            markdown = (root / "comparison.md").read_text(encoding="utf-8")

        self.assertEqual(20, comparison["sample_count"])
        self.assertIn("AVIGATE baseline", markdown)
        self.assertIn("AVIGATE+Qwen2.5-Omni Agent", markdown)
        self.assertIn("0.6500", markdown)

    def _write_sample(
        self,
        sample_dir: Path,
        *,
        reference_caption: str,
        edit_text: str,
        target_caption: str = "target caption",
    ) -> None:
        sample_dir.mkdir(parents=True)
        (sample_dir / "reference.mp4").write_bytes(b"ref")
        (sample_dir / "target.mp4").write_bytes(b"target")
        (sample_dir / "edit_text.txt").write_text(edit_text + "\n", encoding="utf-8")
        (sample_dir / "info.json").write_text(
            json.dumps(
                {
                    "reference_caption": reference_caption,
                    "target_caption": target_caption,
                    "source": "daily_omni",
                    "difference_type": "object_presence",
                    "accepted": True,
                    "target_clip_id": "clip-target",
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
