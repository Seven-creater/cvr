from __future__ import annotations

import unittest

import numpy as np

from app.audio_cvr_reference_ladder import _metric_summary, _paired_test


class AudioCVRReferenceLadderTests(unittest.TestCase):
    def test_metric_summary_uses_paired_masked_ranks(self) -> None:
        with_rows = [
            {
                "sample_id": "a",
                "base_target_rank": 2,
                "adapter_target_rank": 2,
                "base_reference_rank": 1,
                "adapter_reference_rank": 1,
                "base_target_score": 0.8,
                "adapter_target_score": 0.8,
                "base_reference_score": 0.9,
                "adapter_reference_score": 0.9,
                "base_top1": {"is_reference": True},
                "adapter_top1": {"is_reference": True},
            },
            {
                "sample_id": "b",
                "base_target_rank": 1,
                "adapter_target_rank": 1,
                "base_reference_rank": 2,
                "adapter_reference_rank": 2,
                "base_target_score": 0.9,
                "adapter_target_score": 0.9,
                "base_reference_score": 0.8,
                "adapter_reference_score": 0.8,
                "base_top1": {"is_reference": False},
                "adapter_top1": {"is_reference": False},
            },
        ]
        masked_rows = [
            {**with_rows[0], "adapter_target_rank": 1, "base_target_rank": 1},
            with_rows[1],
        ]
        summary, arrays = _metric_summary(with_rows, masked_rows, "adapter")
        self.assertEqual(0.5, summary["with_reference"]["R@1"])
        self.assertEqual(1.0, summary["masked_reference"]["R@1"])
        self.assertEqual(0.5, summary["reference_induced_R@1_drop"])
        np.testing.assert_array_equal([0.0, 1.0], arrays["with_correct"])
        np.testing.assert_array_equal([1.0, 1.0], arrays["masked_correct"])

    def test_paired_test_is_deterministic(self) -> None:
        first = np.asarray([1.0, 1.0, 0.0, 1.0])
        second = np.asarray([0.0, 1.0, 0.0, 0.0])
        first_result = _paired_test(first, second, iterations=1000, seed=17)
        second_result = _paired_test(first, second, iterations=1000, seed=17)
        self.assertEqual(first_result, second_result)
        self.assertEqual(0.5, first_result["mean_difference"])


if __name__ == "__main__":
    unittest.main()
