from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.audio_cvr_paper_experiment import (
    aggregate_final,
    finalize_benchmark,
    prepare_benchmark_review,
    prepare_paper_splits,
    score_fusion,
    summarize_validation,
)
from app.e5_audio_delta_train import _AudioDeltaAdapter, _import_torch


class AudioCVRPaperExperimentTests(unittest.TestCase):
    def test_prepare_paper_splits_preserves_assignment_and_filters_test_main(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_root = root / "source_splits"
            split_root.mkdir()
            self._write_jsonl(split_root / "train.jsonl", [self._row("train", "source_train", "pair_train")])
            self._write_jsonl(split_root / "val.jsonl", [self._row("val", "source_val", "pair_val")])
            self._write_jsonl(
                split_root / "test_main.jsonl",
                [
                    self._row("main", "source_test_a", "pair_test_a"),
                    self._row("extended", "source_test_b", "pair_test_b", tier="extended"),
                ],
            )

            summary = prepare_paper_splits(split_root=split_root, output_dir=root / "paper")

            self.assertEqual(2, summary["counts"]["test_all"])
            self.assertEqual(1, summary["counts"]["test_main"])
            self.assertEqual(0, summary["leakage"]["violation_count"])
            test_main = self._read_jsonl(root / "paper" / "test_main.jsonl")
            self.assertEqual("main", test_main[0]["sample_id"])
            review = self._read_jsonl(root / "paper" / "test_main_human_review.jsonl")
            self.assertEqual("unreviewed", review[0]["review"]["decision"])

    def test_prepare_paper_splits_rejects_source_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_root = root / "source_splits"
            split_root.mkdir()
            self._write_jsonl(split_root / "train.jsonl", [self._row("train", "same_source", "pair_train")])
            self._write_jsonl(split_root / "val.jsonl", [self._row("val", "same_source", "pair_val")])
            self._write_jsonl(split_root / "test_main.jsonl", [self._row("test", "source_test", "pair_test")])

            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare_paper_splits(split_root=split_root, output_dir=root / "paper")

    def test_validation_selection_uses_required_seed_means(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grid = root / "grid"
            for steps, seed, r1, beats in (
                (120, 13, 0.40, 0.45),
                (120, 23, 0.38, 0.44),
                (240, 13, 0.42, 0.40),
                (240, 23, 0.30, 0.39),
            ):
                run = grid / f"steps_{steps}" / f"seed_{seed}"
                self._write_train_summary(run / "adapter" / "train_summary.json", steps=steps, seed=seed)
                self._write_eval_summary(run / "eval" / "summary.json", r1=r1, beats=beats)

            summary = summarize_validation(
                input_roots=[grid], output_dir=root / "selection", required_seeds=[13, 23], top_n=2
            )

            self.assertEqual(120, summary["selected_config"]["steps"])
            self.assertEqual("validation_only", summary["selection_split"])
            self.assertTrue((root / "selection" / "selected_config.tsv").exists())

    def test_one_se_validation_rule_selects_earlier_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grid = root / "grid"
            for steps, seed, r1 in (
                (700, 13, 0.45),
                (700, 23, 0.45),
                (1000, 13, 0.60),
                (1000, 23, 0.40),
            ):
                run = grid / f"steps_{steps}" / f"seed_{seed}"
                self._write_train_summary(run / "adapter" / "train_summary.json", steps=steps, seed=seed)
                self._write_eval_summary(run / "eval" / "summary.json", r1=r1, beats=0.4)

            summary = summarize_validation(
                input_roots=[grid],
                output_dir=root / "selection",
                required_seeds=[13, 23],
                selection_rule="one_se_earliest",
            )

            self.assertEqual(700, summary["selected_config"]["steps"])
            self.assertEqual("one_se_earliest", summary["selection_rule_name"])
            self.assertEqual(2, summary["one_se_candidate_count"])

    def test_benchmark_review_and_freeze_are_source_disjoint_and_human_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates = []
            for index in range(18):
                row = self._row(f"sample_{index}", f"source_{index}", f"pair_{index}")
                row["dataset"] = "dataset_a" if index % 2 else "dataset_b"
                row["audio_delta_type"] = "speech_topic_in_video_context" if index % 4 == 0 else "sound_event"
                candidates.append(row)
            candidate_path = root / "candidates.jsonl"
            self._write_jsonl(candidate_path, candidates)
            old_train = root / "old_train.jsonl"
            self._write_jsonl(old_train, [self._row("old", "source_0", "pair_old")])

            prepared = prepare_benchmark_review(
                input_path=candidate_path,
                output_dir=root / "review",
                exclude_paths=[old_train],
                review_count=16,
                repeat_review_fraction=0.25,
                random_seed=9,
            )
            self.assertEqual(17, prepared["eligible_count"])
            review_rows = self._read_jsonl(root / "review" / "human_review_round1.jsonl")
            for row in review_rows:
                row["review"].update(
                    {
                        "edit_audio_only": True,
                        "reference_does_not_satisfy_edit": True,
                        "target_satisfies_edit": True,
                        "video_only_cannot_identify_target": True,
                        "hard_negatives_do_not_satisfy_edit": True,
                        "decision": "passed",
                    }
                )
            completed = root / "completed_review.jsonl"
            self._write_jsonl(completed, review_rows)

            frozen = finalize_benchmark(
                candidate_path=root / "review" / "benchmark_review_candidates.jsonl",
                review_paths=[completed],
                output_dir=root / "frozen",
                exclude_paths=[old_train],
                target_count=10,
                minimum_count=10,
                max_speech_ratio=0.40,
                max_dataset_ratio=0.60,
                random_seed=9,
            )

            self.assertEqual(10, frozen["final_count"])
            self.assertTrue(frozen["target_count_met"])
            self.assertEqual(0, frozen["leakage"]["violation_count"])
            self.assertTrue((root / "frozen" / "frozen_benchmark.sha256").exists())
            final_rows = self._read_jsonl(root / "frozen" / "test_main.jsonl")
            self.assertNotIn("source_0", {row["source_disjoint_group_id"] for row in final_rows})
            speech_count = sum(row["audio_delta_type"] == "speech_topic_in_video_context" for row in final_rows)
            self.assertLessEqual(speech_count, 4)

    def test_extended_benchmark_candidates_require_explicit_human_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main = self._row("main_sample", "source_main", "pair_main")
            extended = self._row("extended_sample", "source_extended", "pair_extended", tier="extended")
            extended.update(
                {
                    "benchmark_eligible": False,
                    "training_eligible": True,
                    "diagnostic_reason": ["main_speech_cap_exceeded"],
                    "audio_delta_type": "speech_topic_in_video_context",
                }
            )
            input_path = root / "candidates.jsonl"
            self._write_jsonl(input_path, [main, extended])
            local_path = root / "local_candidates.jsonl"
            self._write_jsonl(
                local_path,
                [
                    {
                        "sample_id": "extended_sample",
                        "negative_type": "local_same_source",
                        "video": "/extended_local.mp4",
                        "same_source": True,
                        "satisfies_edit": "unknown",
                        "verification_status": "candidate_unverified",
                    }
                ],
            )

            default_summary = prepare_benchmark_review(
                input_path=input_path,
                output_dir=root / "default_review",
                review_count=10,
            )
            self.assertEqual(1, default_summary["eligible_count"])

            prepared = prepare_benchmark_review(
                input_path=input_path,
                output_dir=root / "promotion_review",
                review_count=10,
                eligible_tiers=("main", "extended"),
                local_candidate_paths=(local_path,),
            )
            self.assertEqual(2, prepared["eligible_count"])
            review_rows = self._read_jsonl(root / "promotion_review" / "human_review_round1.jsonl")
            for row in review_rows:
                row["review"].update(
                    {
                        "edit_audio_only": True,
                        "reference_does_not_satisfy_edit": True,
                        "target_satisfies_edit": True,
                        "video_only_cannot_identify_target": True,
                        "hard_negatives_do_not_satisfy_edit": True,
                        "decision": "passed",
                    }
                )
                if row["automatic_split_tier"] == "extended":
                    row["review"].update(
                        {
                            "audio_change_clearly_audible": True,
                            "video_context_preserved": True,
                            "not_asr_or_transcript_only": True,
                        }
                    )
            completed = root / "completed.jsonl"
            self._write_jsonl(completed, review_rows)

            frozen = finalize_benchmark(
                candidate_path=root / "promotion_review" / "benchmark_review_candidates.jsonl",
                review_paths=[completed],
                output_dir=root / "frozen",
                target_count=2,
                minimum_count=2,
                max_speech_ratio=1.0,
                max_dataset_ratio=1.0,
                eligible_tiers=("main", "extended"),
            )

            self.assertEqual(1, frozen["human_promoted_extended_count"])
            self.assertEqual(1, frozen["strict_local_count"])
            rows = {row["sample_id"]: row for row in self._read_jsonl(root / "frozen" / "test_main.jsonl")}
            promoted = rows["extended_sample"]
            self.assertEqual("extended", promoted["automatic_split_tier"])
            self.assertEqual("main", promoted["split_tier"])
            self.assertTrue(promoted["human_verified_benchmark_eligible"])
            self.assertEqual("human_verified_extended", promoted["benchmark_promotion"])
            self.assertEqual(["main_speech_cap_exceeded"], promoted["automatic_diagnostic_reason"])
            self.assertEqual("human_verified", promoted["local_same_source_candidates"][0]["verification_status"])
            self.assertIs(promoted["local_same_source_candidates"][0]["satisfies_edit"], False)

    def test_frozen_benchmark_limits_queries_per_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates = [
                self._row("source_a_first", "source_a", "pair_a1"),
                self._row("source_a_second", "source_a", "pair_a2"),
                self._row("source_b_first", "source_b", "pair_b1"),
            ]
            candidate_path = root / "candidates.jsonl"
            self._write_jsonl(candidate_path, candidates)
            prepare_benchmark_review(
                input_path=candidate_path,
                output_dir=root / "review",
                review_count=3,
            )
            reviews = self._read_jsonl(root / "review" / "human_review_round1.jsonl")
            for row in reviews:
                row["review"].update(
                    {
                        "edit_audio_only": True,
                        "reference_does_not_satisfy_edit": True,
                        "target_satisfies_edit": True,
                        "video_only_cannot_identify_target": True,
                        "hard_negatives_do_not_satisfy_edit": True,
                        "decision": "passed",
                    }
                )
            review_path = root / "completed.jsonl"
            self._write_jsonl(review_path, reviews)

            summary = finalize_benchmark(
                candidate_path=root / "review" / "benchmark_review_candidates.jsonl",
                review_paths=[review_path],
                output_dir=root / "frozen",
                target_count=3,
                minimum_count=2,
                max_speech_ratio=1.0,
                max_dataset_ratio=1.0,
                max_per_source=1,
            )

            self.assertEqual(2, summary["final_count"])
            rows = self._read_jsonl(root / "frozen" / "test_main.jsonl")
            self.assertEqual(2, len({row["source_disjoint_group_id"] for row in rows}))

    def test_final_aggregation_reports_paired_audio_gain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final = root / "final"
            for seed in (13, 23):
                self._write_final_eval(final / f"seed_{seed}" / "eval_V_T", ranks=[2, 1, 3], gaps=[-0.1, 0.1, -0.2])
                self._write_final_eval(final / f"seed_{seed}" / "eval_V_A_T", ranks=[1, 1, 2], gaps=[0.1, 0.2, -0.05])

            summary = aggregate_final(
                input_root=final,
                output_dir=root / "stats",
                required_seeds=[13, 23],
                bootstrap_samples=200,
                permutation_samples=200,
                random_seed=7,
            )

            gain = summary["paired_statistics"]["query_level_statistics"]["R@1"]["difference_mean"]
            self.assertGreater(gain, 0.0)
            self.assertEqual(0, json.loads((root / "stats" / "audit.json").read_text())["violation_count"])
            self.assertTrue((root / "stats" / "audio_gain_summary.md").exists())
            self.assertTrue((root / "stats" / "error_breakdown.json").exists())

    def test_final_aggregation_supports_prespecified_multiple_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final = root / "final"
            for seed in (13, 23):
                self._write_final_eval(final / f"seed_{seed}" / "eval_V_only", ranks=[3, 1, 3], gaps=[-0.2, 0.1, -0.2])
                self._write_final_eval(final / f"seed_{seed}" / "eval_V_T", ranks=[2, 1, 3], gaps=[-0.1, 0.1, -0.2])
                self._write_final_eval(final / f"seed_{seed}" / "eval_V_A_T", ranks=[1, 1, 2], gaps=[0.1, 0.2, -0.05])

            summary = aggregate_final(
                input_root=final,
                output_dir=root / "stats",
                required_seeds=[13, 23],
                comparisons=[("V_A_T", "V_T"), ("V_A_T", "V_only"), ("V_T", "V_only")],
                bootstrap_samples=200,
                permutation_samples=200,
                random_seed=7,
            )

            self.assertEqual(3, len(summary["paired_comparisons"]))
            statistic = summary["paired_comparisons"]["V_A_T_minus_V_T"]["query_level_statistics"]["R@1"]
            self.assertIn("paired_randomization_p_holm", statistic)
            self.assertTrue((root / "stats" / "paired_comparisons.md").exists())

    def test_score_fusion_selects_alpha_on_paired_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_a = root / "cache_a"
            cache_b = root / "cache_b"
            query_a = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            query_b = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            gallery = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
            for cache, query in ((cache_a, query_a), (cache_b, query_b)):
                cache.mkdir()
                np.savez_compressed(
                    cache / "eval_embeddings.npz",
                    query=query,
                    gallery=gallery,
                    positive_gallery_index=np.asarray([0, 1], dtype=np.int64),
                    reference_gallery_index=np.asarray([2, 2], dtype=np.int64),
                )
                self._write_jsonl(
                    cache / "eval_records.jsonl",
                    [self._row("sample_0", "source_0", "pair_0"), self._row("sample_1", "source_1", "pair_1")],
                )
                self._write_jsonl(
                    cache / "eval_gallery.jsonl",
                    [
                        {"gallery_id": "p0", "video": "/p0.mp4", "raw_source_id": "source_0", "kind": "positive", "source_payload": {"sample_id": "sample_0"}},
                        {"gallery_id": "p1", "video": "/p1.mp4", "raw_source_id": "source_1", "kind": "positive", "source_payload": {"sample_id": "sample_1"}},
                        {"gallery_id": "r", "video": "/r.mp4", "raw_source_id": "other", "kind": "reference_negative", "source_payload": {"negative_type": "reference_negative"}},
                    ],
                )
            torch = _import_torch()
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            torch.save(_AudioDeltaAdapter(torch, 2).state_dict(), adapter_dir / "adapter.pt")

            summary = score_fusion(
                cache_a=cache_a,
                cache_b=cache_b,
                adapter_dir=adapter_dir,
                output_dir=root / "fusion",
                alpha_grid=[0.0, 0.5, 1.0],
                device="cpu",
            )

            self.assertEqual(1.0, summary["fusion"]["selected_alpha_cache_a"])
            self.assertTrue((root / "fusion" / "per_query_scores.jsonl").exists())

    @staticmethod
    def _row(sample_id: str, source: str, pair: str, *, tier: str = "main") -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "reference_video": f"/{sample_id}_ref.mp4",
            "target_video": f"/{sample_id}_target.mp4",
            "edit_text": "add a bell sound",
            "source_disjoint_group_id": source,
            "raw_source_id": source,
            "pair_group_id": pair,
            "inverse_pair_group_id": pair,
            "split_tier": tier,
            "direction": "forward",
            "audio_delta_type": "sound_event",
        }

    @staticmethod
    def _write_train_summary(path: Path, *, steps: int, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"steps": steps, "learning_rate": 0.0003, "batch_size": 8, "seed": seed}),
            encoding="utf-8",
        )

    @staticmethod
    def _write_eval_summary(path: Path, *, r1: float, beats: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "eval_count": 3,
            "gallery_count": 10,
            "rows": [
                {"method": "base_e5_global", "R@1": 0.1, "R@5": 0.5, "R@10": 1.0},
                {"method": "audio_delta_adapter_global", "R@1": r1, "R@5": 0.8, "R@10": 1.0},
            ],
            "target_beats_reference": {
                "base_e5": {"target_beats_reference_rate": 0.2},
                "audio_delta_adapter": {
                    "target_beats_reference_rate": beats,
                    "target_minus_reference_mean": -0.01,
                },
            },
            "gallery_negative_recall_by_type": {},
        }
        path.write_text(json.dumps(summary), encoding="utf-8")

    def _write_final_eval(self, eval_dir: Path, *, ranks: list[int], gaps: list[float]) -> None:
        hits = sum(rank <= 1 for rank in ranks) / len(ranks)
        self._write_eval_summary(eval_dir / "summary.json", r1=hits, beats=sum(gap > 0 for gap in gaps) / len(gaps))
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        summary["eval_count"] = len(ranks)
        summary["gallery_count"] = 10
        summary["target_beats_reference"]["audio_delta_adapter"]["target_minus_reference_mean"] = sum(gaps) / len(gaps)
        (eval_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        score_rows = [
            {
                "sample_id": f"sample_{index}",
                "adapter_target_rank": rank,
                "adapter_target_minus_reference": gap,
            }
            for index, (rank, gap) in enumerate(zip(ranks, gaps))
        ]
        self._write_jsonl(eval_dir / "per_query_scores.jsonl", score_rows)
        topk_rows = [
            {
                "sample_id": row["sample_id"],
                "adapter_target_rank": row["adapter_target_rank"],
                "adapter_topk": [{"kind": "reference_negative", "is_reference": True}],
            }
            for row in score_rows
        ]
        self._write_jsonl(eval_dir / "per_query_topk.jsonl", topk_rows)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
