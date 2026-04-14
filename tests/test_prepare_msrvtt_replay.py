import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.backends import FileRetrievalBackend
from app.demo import build_backend
from app.prepare_msrvtt_replay import (
    assess_query_discriminability,
    make_query,
    prepare_replay_pack,
    resolve_filter_policy,
    simulate_scripted_rollout,
)
from app.schemas import CandidateMetadata


class PrepareMsrvttReplayTests(unittest.TestCase):
    def test_prepare_replay_pack_generates_backend_compatible_files(self) -> None:
        raw = {
            "videos": [
                {"video_id": "video1"},
                {"video_id": "video2"},
                {"video_id": "video3"},
                {"video_id": "video4"},
            ],
            "sentences": [
                {"video_id": "video1", "caption": "A band performs on a concert stage while people listen."},
                {"video_id": "video1", "caption": "Musicians play music indoors."},
                {"video_id": "video2", "caption": "A band performs on a concert stage while the crowd cheers and applauds."},
                {"video_id": "video2", "caption": "People clap during the indoor concert."},
                {"video_id": "video3", "caption": "A dog runs in the park near trees."},
                {"video_id": "video3", "caption": "People watch the dog outside."},
                {"video_id": "video4", "caption": "A cat runs in the park near trees."},
                {"video_id": "video4", "caption": "People watch the cat outside."},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "MSRVTT_data.json"
            split_path = root / "split.csv"
            out_dir = root / "prepared"

            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            with split_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id"])
                writer.writerow(["video1"])
                writer.writerow(["video2"])
                writer.writerow(["video3"])
                writer.writerow(["video4"])

            pack = prepare_replay_pack(
                msrvtt_json_path=raw_path,
                split_csv_path=split_path,
                output_dir=out_dir,
                max_candidates=None,
                max_queries=6,
                seed=7,
                difficulty_preset=None,
                min_target_margin=0.04,
                max_strong_matches=1,
            )

            self.assertGreaterEqual(pack.stats["candidate_count"], 4)
            self.assertGreaterEqual(pack.stats["query_count"], 2)
            self.assertTrue((out_dir / "candidates.json").exists())
            self.assertTrue((out_dir / "queries.json").exists())
            self.assertTrue((out_dir / "retrieval_scores.json").exists())
            self.assertTrue((out_dir / "real.yaml").exists())
            self.assertEqual(pack.config["candidates_path"], "candidates.json")
            self.assertEqual(pack.stats["discriminability_scorer"], "generation_v2_decoupled")
            self.assertEqual(pack.stats["rollout_scorer"], "scripted_controller_v1")
            self.assertEqual(pack.stats["expected_scored_candidates_per_query"], pack.stats["candidate_count"] - 1)
            self.assertIn("predicted_agent_failures", pack.stats)
            self.assertIn("predicted_retry_queries", pack.stats)

            backend = FileRetrievalBackend(
                candidates_path=out_dir / "candidates.json",
                queries_path=out_dir / "queries.json",
                retrieval_scores_path=out_dir / "retrieval_scores.json",
            )
            self.assertGreaterEqual(len(backend.list_queries()), 1)
            built_backend = build_backend("real", str(out_dir / "real.yaml"))
            self.assertGreaterEqual(len(built_backend.list_queries()), 1)

    def test_discriminability_rejects_ambiguous_queries(self) -> None:
        source = CandidateMetadata(
            video_id="src1",
            title="source",
            summary="A dog runs in the park.",
            caption="Dog in park.",
            asr="",
            audio_tags=[],
            visual_objects=["dog", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        target = CandidateMetadata(
            video_id="target1",
            title="target",
            summary="A cat runs in the park.",
            caption="Cat in park.",
            asr="",
            audio_tags=[],
            visual_objects=["cat", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        distractor = CandidateMetadata(
            video_id="target2",
            title="distractor",
            summary="A cat moves through the park.",
            caption="Cat outdoors in park.",
            asr="",
            audio_tags=[],
            visual_objects=["cat", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        query = make_query("q1", source, target, "object", "cat")
        policy = resolve_filter_policy(
            difficulty_preset="strict",
            min_target_margin=0.04,
            max_strong_matches=1,
        )
        result = assess_query_discriminability(
            query=query,
            source=source,
            target=target,
            candidates=[source, target, distractor],
            policy=policy,
        )
        self.assertFalse(result.accepted)
        self.assertGreaterEqual(result.strong_match_count, 2)

    def test_filter_policy_preset_overrides_manual_values(self) -> None:
        policy = resolve_filter_policy(
            difficulty_preset="medium-hard",
            min_target_margin=0.99,
            max_strong_matches=99,
        )
        self.assertEqual(policy.name, "medium-hard")
        self.assertEqual(policy.min_target_margin, 0.04)
        self.assertEqual(policy.max_strong_matches, 2)
        self.assertEqual(policy.max_source_uses, 5)
        self.assertEqual(policy.max_target_uses, 5)

    def test_hard_policy_prefers_retry_candidates(self) -> None:
        policy = resolve_filter_policy(
            difficulty_preset="hard",
            min_target_margin=0.99,
            max_strong_matches=99,
        )
        self.assertEqual(policy.name, "hard")
        self.assertEqual(policy.generation_rank_cutoff, 2)
        self.assertEqual(policy.round1_rank_cutoff, 2)
        self.assertEqual(policy.round2_rank_cutoff, 1)
        self.assertTrue(policy.prefer_retry_candidates)

    def test_agent_hard_policy_prefers_agent_failures(self) -> None:
        policy = resolve_filter_policy(
            difficulty_preset="agent-hard",
            min_target_margin=0.99,
            max_strong_matches=99,
        )
        self.assertEqual(policy.name, "agent-hard")
        self.assertEqual(policy.generation_rank_cutoff, 3)
        self.assertEqual(policy.round1_rank_cutoff, 5)
        self.assertEqual(policy.round2_rank_cutoff, 5)
        self.assertEqual(policy.round3_rank_cutoff, 5)
        self.assertTrue(policy.prefer_retry_candidates)
        self.assertTrue(policy.prefer_agent_failure_candidates)

    def test_scripted_rollout_detects_hidden_target_failure(self) -> None:
        source = CandidateMetadata(
            video_id="src",
            title="source",
            summary="A dog runs in the park near trees.",
            caption="A dog runs in the park near trees.",
            asr="",
            audio_tags=[],
            visual_objects=["dog", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        target = CandidateMetadata(
            video_id="target",
            title="target",
            summary="A cat waits in the park.",
            caption="A cat waits in the park.",
            asr="",
            audio_tags=[],
            visual_objects=["cat"],
            scene_tags=["park"],
            temporal_tags=["global"],
        )
        distractor_a = CandidateMetadata(
            video_id="dist_a",
            title="distractor-a",
            summary="A cat and dog run together in the outdoor park near trees.",
            caption="A cat and dog run together in the outdoor park near trees.",
            asr="",
            audio_tags=[],
            visual_objects=["cat", "dog", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        distractor_b = CandidateMetadata(
            video_id="dist_b",
            title="distractor-b",
            summary="A cat plays with a dog in the outdoor park.",
            caption="A cat plays with a dog in the outdoor park.",
            asr="",
            audio_tags=[],
            visual_objects=["cat", "dog"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        query = make_query("q_hidden_target", source, target, "object", "cat")

        rollout = simulate_scripted_rollout(
            query=query,
            source=source,
            target=target,
            candidates=[source, target, distractor_a, distractor_b],
        )

        self.assertGreaterEqual(rollout["round1_target_rank"], 3)
        self.assertGreaterEqual(rollout["round2_target_rank"], 3)
        self.assertFalse(rollout["agent_success"])
        self.assertIn(rollout["final_candidate_id"], {"dist_a", "dist_b"})


if __name__ == "__main__":
    unittest.main()
