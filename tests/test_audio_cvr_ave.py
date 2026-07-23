from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.audio_cvr_ave import (
    AveAnnotation,
    boundary_candidate,
    extract_youtube_ids,
)


class AudioCvrAveTest(unittest.TestCase):
    def annotation(self, start: float, end: float) -> AveAnnotation:
        return AveAnnotation(
            category="Church bell",
            video_id="abcdefghijk",
            quality="good",
            event_start=start,
            event_end=end,
            source_path=Path("/tmp/abcdefghijk.mp4"),
        )

    def test_boundary_candidate_prefers_event_free_then_event_rich_window(self) -> None:
        candidate = boundary_candidate(self.annotation(6, 9), clip_seconds=6)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.reference_start, 0)
        self.assertEqual(candidate.target_start, 4)
        self.assertEqual(candidate.reference_event_overlap, 0)
        self.assertEqual(candidate.target_event_overlap, 3)
        self.assertEqual(candidate.tier, "high")

    def test_full_length_event_is_not_directional(self) -> None:
        self.assertIsNone(boundary_candidate(self.annotation(0, 10), clip_seconds=6))

    def test_middle_event_without_overlap_difference_is_rejected(self) -> None:
        self.assertIsNone(boundary_candidate(self.annotation(2, 8), clip_seconds=6))

    def test_extracts_video_ids_from_existing_source_names(self) -> None:
        rows = [
            {
                "raw_source_id": "avatar:avatar_ZwJZo1k9jlA_00028_deadbeef",
                "reference_video": "clips/vggsound__yJcmmwiMcZ_000030/source.mp4",
            },
            {"source_video": "/data/abcdefghijk.mp4"},
        ]
        self.assertEqual(
            extract_youtube_ids(rows),
            {"ZwJZo1k9jlA", "_yJcmmwiMcZ", "abcdefghijk"},
        )


if __name__ == "__main__":
    unittest.main()
