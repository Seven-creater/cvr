from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.audio_cvr_paper_experiment import (
    _automatic_consensus_reject_reason,
    _automatic_review_decision,
    _select_disjoint_test_validation,
    _select_exact_benchmark_quota,
    aggregate_final,
    audit_training_splits,
    finalize_automatic_benchmark,
    finalize_benchmark,
    prepare_automatic_benchmark_review,
    prepare_benchmark_review,
    prepare_paper_splits,
    prepare_training_subset,
    score_fusion,
    summarize_validation,
)
from app.e5_audio_delta_train import _AudioDeltaAdapter, _import_torch


class AudioCVRPaperExperimentTests(unittest.TestCase):
    def test_prepare_training_subset_filters_non_speech_and_preserves_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = [
                self._automatic_row("sound", "source_sound", "pair_sound", subtype="sound_event"),
                self._automatic_row("music", "source_music", "pair_music", subtype="music"),
                self._automatic_row("speech", "source_speech", "pair_speech", subtype="speech_topic_in_video_context"),
            ]
            val = [self._automatic_row("val", "source_val", "pair_val", subtype="music")]
            test = [self._automatic_row("test", "source_test", "pair_test", subtype="sound_event")]
            self._write_jsonl(root / "train.jsonl", train)
            self._write_jsonl(root / "val.jsonl", val)
            self._write_jsonl(root / "test.jsonl", test)

            summary = prepare_training_subset(
                train_path=root / "train.jsonl",
                val_path=root / "val.jsonl",
                test_path=root / "test.jsonl",
                output_dir=root / "subset",
                expected_count=2,
            )

            self.assertEqual(2, summary["forward_count"])
            self.assertEqual(2, summary["unique_source_count"])
            self.assertEqual({"music": 1, "sound_event": 1}, summary["subtype_distribution"])
            self.assertFalse(summary["selection_uses_model_scores"])
            self.assertTrue(summary["ready_for_inverse_augmentation"])
            selected = [json.loads(line) for line in (root / "subset" / "train_non_speech_forward.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({"sound", "music"}, {row["sample_id"] for row in selected})

    def test_paper_splits_accept_frozen_test_main_150_and_audit_clean_training_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_root = root / "benchmark_v1"
            train = [self._automatic_row("train", "source_train", "pair_train", subtype="sound_event")]
            val = [self._automatic_row("val", "source_val", "pair_val", subtype="music")]
            test = [self._automatic_row("test", "source_test", "pair_test", subtype="sound_event")]
            for row, split in ((train[0], "train"), (val[0], "val"), (test[0], "test_main")):
                row["dataset_split"] = split
                row["split_tier"] = "main"
                row["direction"] = "forward"
                row["is_inverse"] = False
            self._write_jsonl(split_root / "train.jsonl", train)
            self._write_jsonl(split_root / "val.jsonl", val)
            self._write_jsonl(split_root / "test_main_150.jsonl", test)

            prepared = prepare_paper_splits(
                split_root=split_root,
                output_dir=root / "paper_splits",
            )
            audit = audit_training_splits(
                train_path=split_root / "train.jsonl",
                val_path=split_root / "val.jsonl",
                test_path=split_root / "test_main_150.jsonl",
                output_dir=root / "audit",
            )

            self.assertTrue(prepared["test_source_path"].endswith("test_main_150.jsonl"))
            self.assertEqual(0, audit["violation_count"])
            self.assertEqual(1, audit["train_forward_count"])
            self.assertTrue(audit["ready_for_training"])
            self.assertTrue((root / "audit" / "training_split_audit.md").exists())

    def test_automatic_review_pool_merges_and_deduplicates_without_using_legacy_asr_risk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main = self._automatic_row("main", "source_a", "pair_a", subtype="sound_event")
            duplicate = self._automatic_row(
                "duplicate",
                "source_a",
                "pair_a",
                subtype="sound_event",
                tier="extended",
            )
            duplicate["reference_video"] = main["reference_video"]
            duplicate["target_video"] = main["target_video"]
            duplicate["asr_degeneracy_risk"] = 0.45
            rejected = self._automatic_row("rejected", "source_b", "pair_b", subtype="music")
            rejected["audio_only_verification"]["accept"] = False
            first = root / "old.jsonl"
            second = root / "fresh.jsonl"
            self._write_jsonl(first, [main])
            self._write_jsonl(second, [duplicate, rejected])

            summary = prepare_automatic_benchmark_review(
                input_paths=[first, second],
                output_dir=root / "review",
                review_pool_targets={"sound_event": 10, "music": 10, "speech_topic_in_video_context": 10},
            )

            self.assertEqual(3, summary["input_count"])
            self.assertEqual(1, summary["deduplicated_count"])
            self.assertEqual(1, summary["duplicate_drop_counts"]["duplicate_reference_target"])
            self.assertEqual(1, summary["filter_rejection_counts"]["legacy_audio_only_reject"])
            rows = self._read_jsonl(root / "review" / "combined_pool_deduplicated.jsonl")
            self.assertEqual("main", rows[0]["sample_id"])
            self.assertTrue(rows[0]["legacy_asr_risk_advisory_only"])

    def test_automatic_speech_review_rejects_transcript_like_asr(self) -> None:
        row = self._automatic_row(
            "speech",
            "source_speech",
            "pair_speech",
            subtype="speech_topic_in_video_context",
        )
        row["edit_text"] = "change the spoken phrase from hello there to good morning"
        review = _automatic_review_decision(
            row,
            audio_verify=self._audio_review_payload(),
            video_verify=self._video_review_payload(),
            full_av=self._full_av_review_payload(),
            context=self._context_review_payload(
                speech_role="asr_only",
                transcript_like=True,
                asr_risk=0.90,
            ),
            review_pass_id=1,
            model="mock-omni",
        )

        self.assertEqual("reject", review["decision"])
        self.assertIn("speech_role:asr_only", review["reject_reasons"])
        self.assertIn("transcript_like", review["reject_reasons"])
        self.assertIn("asr_risk_above_0.35", review["reject_reasons"])

    def test_automatic_repeat_disagreement_never_enters_consensus(self) -> None:
        first = self._model_review("sample", subtype="sound_event")
        second = dict(first)
        second["review_pass_id"] = 2
        second["full_av_pass"] = False

        reason = _automatic_consensus_reject_reason(first, second, repeated=True)

        self.assertEqual("rejected_review_disagreement", reason)

    def test_automatic_selector_enforces_90_30_30_quota_and_source_uniqueness(self) -> None:
        rows: list[dict[str, object]] = []
        specs = (("sound_event", 90), ("music", 30), ("speech_topic_in_video_context", 30))
        datasets = ("avatar", "vggsound", "worldsense", "daily_omni")
        index = 0
        for subtype, count in specs:
            for _ in range(count):
                dataset = datasets[index % len(datasets)]
                row = self._automatic_row(
                    f"sample_{index}",
                    f"source_{index}",
                    f"pair_{index}",
                    subtype=subtype,
                    dataset=dataset,
                )
                row["automatic_review_pass1"] = self._model_review(
                    str(row["sample_id"]), subtype=subtype
                )
                row["min_stage_confidence"] = 0.9
                rows.append(row)
                index += 1

        selected, summary = _select_exact_benchmark_quota(
            rows,
            targets={"sound_event": 90, "music": 30, "speech_topic_in_video_context": 30},
            total_target=150,
            max_dataset_ratio=0.50,
            max_hdtf_ratio=0.15,
            max_voxceleb_ratio=0.05,
            max_per_source=1,
            random_seed=20260720,
        )

        self.assertEqual(150, len(selected))
        self.assertEqual(
            {"music": 30, "sound_event": 90, "speech_topic_in_video_context": 30},
            summary["selected_subtypes"],
        )
        self.assertEqual(150, len({row["raw_source_id"] for row in selected}))

    def test_joint_split_selection_reserves_validation_sources_before_test(self) -> None:
        rows = [
            self._automatic_row(
                "sound_shared", "source_shared", "pair_sound_shared", subtype="sound_event"
            ),
            self._automatic_row("sound_two", "source_two", "pair_sound_two", subtype="sound_event"),
            self._automatic_row(
                "sound_three", "source_three", "pair_sound_three", subtype="sound_event"
            ),
            self._automatic_row(
                "music_shared", "source_shared", "pair_music_shared", subtype="music"
            ),
        ]
        for row in rows:
            row["min_stage_confidence"] = 0.8
        rows[0]["min_stage_confidence"] = 1.0

        test, _, validation, _, order = _select_disjoint_test_validation(
            rows,
            test_targets={"sound_event": 2, "music": 0},
            validation_targets={"sound_event": 0, "music": 1},
            max_dataset_ratio=1.0,
            relaxed_dataset_ratio=1.0,
            max_hdtf_ratio=1.0,
            max_voxceleb_ratio=1.0,
            max_per_source=1,
            random_seed=13,
        )

        self.assertEqual("validation_reserved_before_test", order)
        self.assertEqual(2, len(test))
        self.assertEqual(1, len(validation))
        self.assertFalse(
            {str(row["raw_source_id"]) for row in test}
            & {str(row["raw_source_id"]) for row in validation}
        )

    def test_finalize_automatic_benchmark_rebuilds_disjoint_splits_and_asr_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows: list[dict[str, object]] = []
            subtypes = [
                "sound_event",
                "sound_event",
                "sound_event",
                "sound_event",
                "sound_event",
                "music",
                "music",
                "speech_topic_in_video_context",
                "speech_topic_in_video_context",
            ]
            for index, subtype in enumerate(subtypes):
                rows.append(
                    self._automatic_row(
                        f"sample_{index}",
                        f"source_{index}",
                        f"pair_{index}",
                        subtype=subtype,
                        dataset=("avatar", "vggsound", "worldsense", "daily_omni")[index % 4],
                    )
                )
            asr_row = self._automatic_row(
                "sample_asr",
                "source_asr",
                "pair_asr",
                subtype="speech_topic_in_video_context",
                dataset="hdtf",
            )
            rows.append(asr_row)
            combined = root / "combined.jsonl"
            candidates = root / "candidates.jsonl"
            pass1_path = root / "pass1.jsonl"
            pass2_path = root / "pass2.jsonl"
            self._write_jsonl(combined, rows)
            self._write_jsonl(candidates, rows)
            reviews = [
                self._model_review(str(row["sample_id"]), subtype=str(row["b_subtype"]))
                for row in rows[:-1]
            ]
            asr_review = self._model_review("sample_asr", subtype="speech_topic_in_video_context")
            asr_review.update(
                {
                    "decision": "reject",
                    "speech_role": "asr_only",
                    "transcript_like": True,
                    "recomputed_asr_risk": 0.9,
                    "reject_reasons": ["speech_role:asr_only", "transcript_like"],
                }
            )
            reviews.append(asr_review)
            self._write_jsonl(pass1_path, reviews)
            self._write_jsonl(pass2_path, [])

            manifest = finalize_automatic_benchmark(
                combined_pool_path=combined,
                candidate_path=candidates,
                pass1_review_paths=[pass1_path],
                pass2_review_paths=[pass2_path],
                output_dir=root / "benchmark",
                subtype_targets={"sound_event": 3, "music": 1, "speech_topic_in_video_context": 1},
                validation_targets={"sound_event": 1, "music": 1, "speech_topic_in_video_context": 1},
                repeat_review_fraction=0.0,
                max_dataset_ratio=0.80,
                relaxed_dataset_ratio=0.80,
                max_hdtf_ratio=0.20,
                max_voxceleb_ratio=0.20,
            )

            self.assertEqual(5, manifest["test_final_count"])
            self.assertEqual(0, manifest["leakage"]["violation_count"])
            self.assertEqual(1, len(self._read_jsonl(root / "benchmark" / "test_asr_diagnostic.jsonl")))
            self.assertTrue((root / "benchmark" / "frozen_benchmark.sha256").exists())
            self.assertTrue((root / "benchmark" / "automatic_review_summary.json").exists())
            self.assertTrue((root / "benchmark" / "benchmark_quality_report.md").exists())
            audit = json.loads((root / "benchmark" / "leakage_audit.json").read_text(encoding="utf-8"))
            self.assertFalse(audit["selection_uses_model_scores"])
            self.assertEqual(0, audit["duplicate_pair_count"])

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

    def test_validation_selection_keeps_adapter_ranks_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grid = root / "grid"
            for rank, r1 in ((16, 0.40), (32, 0.40)):
                run = grid / f"rank_{rank}" / "seed_13"
                self._write_train_summary(
                    run / "adapter" / "train_summary.json",
                    steps=100,
                    seed=13,
                    adapter_architecture="low_rank_residual",
                    adapter_rank=rank,
                )
                self._write_eval_summary(run / "eval" / "summary.json", r1=r1, beats=0.5)

            summary = summarize_validation(
                input_roots=[grid],
                output_dir=root / "selection",
                required_seeds=[13],
                selection_rule="one_se_earliest",
            )

            self.assertEqual(2, summary["configuration_count"])
            self.assertEqual(16, summary["selected_config"]["adapter_rank"])
            adapter_tsv = (root / "selection" / "selected_adapter_config.tsv").read_text(encoding="utf-8")
            self.assertTrue(adapter_tsv.startswith("low_rank_residual\t16\t100\t"))

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
            adapter = _AudioDeltaAdapter(torch, 2, adapter_architecture="low_rank_residual", adapter_rank=1)
            torch.save(adapter.state_dict(), adapter_dir / "adapter.pt")
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"adapter_architecture": "low_rank_residual", "adapter_rank": 1}),
                encoding="utf-8",
            )

            summary = score_fusion(
                cache_a=cache_a,
                cache_b=cache_b,
                adapter_dir=adapter_dir,
                output_dir=root / "fusion",
                alpha_grid=[0.0, 0.5, 1.0],
                device="cpu",
            )

            self.assertEqual(1.0, summary["fusion"]["selected_alpha_cache_a"])
            self.assertEqual("low_rank_residual", summary["adapter_architecture"])
            self.assertEqual(1, summary["adapter_rank"])
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
    def _automatic_row(
        sample_id: str,
        source: str,
        pair: str,
        *,
        subtype: str,
        tier: str = "main",
        dataset: str = "avatar",
    ) -> dict[str, object]:
        row = AudioCVRPaperExperimentTests._row(sample_id, source, pair, tier=tier)
        row.update(
            {
                "accepted": True,
                "fallback": False,
                "dataset": dataset,
                "b_subtype": subtype,
                "audio_delta_type": subtype,
                "audio_delta_strength": 0.9,
                "video_context_strength": 0.8,
                "asr_degeneracy_risk": 0.45,
                "audio_only_verification": AudioCVRPaperExperimentTests._audio_review_payload(),
                "video_only_shortcut": AudioCVRPaperExperimentTests._video_review_payload(),
                "full_av_consistency": AudioCVRPaperExperimentTests._full_av_review_payload(),
            }
        )
        return row

    @staticmethod
    def _audio_review_payload() -> dict[str, object]:
        return {
            "accept": True,
            "reference_satisfies_edit": False,
            "target_satisfies_edit": True,
            "audio_difference_specific": True,
            "edit_text_audio_only": True,
            "confidence": 0.9,
        }

    @staticmethod
    def _video_review_payload() -> dict[str, object]:
        return {
            "accept": True,
            "visual_context_preserved": True,
            "visual_shortcut_risk": False,
            "can_identify_target_without_audio": False,
            "confidence": 0.9,
        }

    @staticmethod
    def _full_av_review_payload() -> dict[str, object]:
        return {
            "accept": True,
            "visual_context_preserved": True,
            "visual_shortcut_risk": False,
            "audio_edit_still_valid": True,
            "confidence": 0.9,
        }

    @staticmethod
    def _context_review_payload(
        *,
        speech_role: str,
        transcript_like: bool = False,
        asr_risk: float = 0.1,
    ) -> dict[str, object]:
        return {
            "accept": True,
            "visual_context_preserved": True,
            "audio_edit_still_valid": True,
            "full_av_required": True,
            "speech_role": speech_role,
            "transcript_like": transcript_like,
            "recomputed_asr_risk": asr_risk,
            "video_context_strength": 0.8,
            "audio_only_solvability": 0.5,
            "confidence": 0.9,
        }

    @staticmethod
    def _model_review(sample_id: str, *, subtype: str) -> dict[str, object]:
        speech = subtype == "speech_topic_in_video_context"
        return {
            "sample_id": sample_id,
            "reviewer_type": "omni",
            "review_profile": "audiocvr_benchmark_review_v1",
            "review_pass_id": 1,
            "audio_only_pass": True,
            "video_only_pass": True,
            "full_av_pass": True,
            "speech_role": "contextual_speech" if speech else "not_speech",
            "transcript_like": False,
            "full_av_required": True,
            "recomputed_asr_risk": 0.1,
            "video_context_strength": 0.8,
            "audio_only_solvability": 0.5,
            "min_stage_confidence": 0.9,
            "decision": "pass",
            "reject_reasons": [],
        }

    @staticmethod
    def _write_train_summary(
        path: Path,
        *,
        steps: int,
        seed: int,
        adapter_architecture: str = "full_rank",
        adapter_rank: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "steps": steps,
                    "learning_rate": 0.0003,
                    "batch_size": 8,
                    "seed": seed,
                    "adapter_architecture": adapter_architecture,
                    "adapter_rank": adapter_rank,
                }
            ),
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
