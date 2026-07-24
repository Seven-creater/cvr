from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from app.audio_cvr_vgg_monoaudio import prepare_candidates, reversible_pairs


FIELDS = (
    "file_name",
    "target_file",
    "paired_file",
    "target_position",
    "target_start_sec",
    "paired_start_sec",
    "label",
    "target_category",
    "paired_category",
    "paired_label",
)


class VggMonoAudioCandidateTest(unittest.TestCase):
    def test_reversible_pairs_require_same_visual_layout(self) -> None:
        rows = [
            {
                "target_file": "a",
                "paired_file": "b",
                "target_position": "left",
                "target_start_sec": "0",
                "paired_start_sec": "2",
                "media_path": __file__,
                "subset": "inter_class",
            },
            {
                "target_file": "b",
                "paired_file": "a",
                "target_position": "right",
                "target_start_sec": "2",
                "paired_start_sec": "0",
                "media_path": __file__,
                "subset": "inter_class",
            },
        ]
        self.assertEqual(len(reversible_pairs(rows)), 1)

    def test_prepare_excludes_human_and_limits_component_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            output = Path(temp) / "output"
            subset = root / "inter_class"
            (subset / "mixed").mkdir(parents=True)
            rows = [
                self._row("a", "b", "left", "dog barking", "Animal", "car engine", "Vehicle"),
                self._row("b", "a", "right", "car engine", "Vehicle", "dog barking", "Animal"),
                self._row("a", "c", "left", "dog barking", "Animal", "bell", "Other"),
                self._row("c", "a", "right", "bell", "Other", "dog barking", "Animal"),
                self._row("d", "e", "left", "male speech", "Human", "piano", "Music"),
                self._row("e", "d", "right", "piano", "Music", "male speech", "Human"),
            ]
            for row in rows:
                (subset / row["file_name"]).touch()
            with (subset / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            summary = prepare_candidates(
                root=root,
                output_dir=output,
                exclude_jsonl_paths=(),
                max_component_uses=1,
                max_candidates=10,
            )
            self.assertEqual(summary["reversible_same_layout_pair_count"], 3)
            self.assertEqual(summary["reversible_nonhuman_pair_count"], 2)
            self.assertEqual(summary["selected_count"], 1)
            self.assertEqual(summary["max_observed_component_uses"], 1)

    @staticmethod
    def _row(
        target: str,
        paired: str,
        position: str,
        label: str,
        category: str,
        paired_label: str,
        paired_category: str,
    ) -> dict[str, str]:
        return {
            "file_name": f"mixed/{target}_{paired}_{position}.mp4",
            "target_file": target,
            "paired_file": paired,
            "target_position": position,
            "target_start_sec": "0",
            "paired_start_sec": "0",
            "label": label,
            "target_category": category,
            "paired_category": paired_category,
            "paired_label": paired_label,
        }


if __name__ == "__main__":
    unittest.main()
