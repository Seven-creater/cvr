import os
import tempfile
import unittest
from pathlib import Path

from app.artifacts import append_jsonl, write_run_artifacts
from app.backends.base import PROJECT_ROOT, heuristic_compare
from app.schemas import CandidateMetadata, QueryCase, RunTrace


class ArtifactAndCompareTests(unittest.TestCase):
    def test_artifacts_write_to_repo_runs_when_cwd_changes(self) -> None:
        trace = RunTrace(
            query=QueryCase(
                query_id="artifact_test",
                source_video_id="src1",
                edit_instruction="Find a similar clip with cheering audio.",
                target_video_id="cand1",
                preserve_tags=["concert"],
                required_audio_tags=["cheering"],
            ),
            planner_name="scripted",
        )
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                paths = write_run_artifacts(trace, prefix="artifact-test")
                jsonl_path = append_jsonl([trace], path="runs/artifact-test.jsonl")
            finally:
                os.chdir(old_cwd)

        self.assertEqual(paths["json"].parent, (PROJECT_ROOT / "runs").resolve())
        self.assertEqual(paths["md"].parent, (PROJECT_ROOT / "runs").resolve())
        self.assertEqual(jsonl_path.parent, (PROJECT_ROOT / "runs").resolve())

        for path in (paths["json"], paths["md"], jsonl_path):
            if path.exists():
                path.unlink()

    def test_compare_confidence_prefers_better_preserve_and_instruction_match(self) -> None:
        query = QueryCase(
            query_id="q1",
            source_video_id="src1",
            edit_instruction="Find a similar outdoor park clip with clear cheering audio.",
            preserve_tags=["park", "outdoor"],
            required_audio_tags=["cheering"],
        )
        source = CandidateMetadata(
            video_id="src1",
            title="source",
            summary="People walk in an outdoor park.",
            caption="A crowd walks outside in a park.",
            asr="",
            audio_tags=["crowd"],
            visual_objects=["people", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        stronger = CandidateMetadata(
            video_id="cand_strong",
            title="strong",
            summary="A cheering crowd gathers in an outdoor park.",
            caption="People cheer loudly outside in the park.",
            asr="The crowd is cheering.",
            audio_tags=["cheering", "crowd"],
            visual_objects=["people", "trees"],
            scene_tags=["park", "outdoor"],
            temporal_tags=["global"],
        )
        weaker = CandidateMetadata(
            video_id="cand_weak",
            title="weak",
            summary="A cheering crowd gathers in a park.",
            caption="People cheer in the park.",
            asr="",
            audio_tags=["cheering"],
            visual_objects=["people"],
            scene_tags=["park"],
            temporal_tags=["global"],
        )

        stronger_result = heuristic_compare(query, source, stronger)
        weaker_result = heuristic_compare(query, source, weaker)
        self.assertEqual(stronger_result.missing, [])
        self.assertEqual(weaker_result.missing, [])
        self.assertGreater(stronger_result.confidence, weaker_result.confidence)


if __name__ == "__main__":
    unittest.main()
