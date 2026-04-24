from __future__ import annotations

import unittest
from pathlib import Path


class ScriptTests(unittest.TestCase):
    def test_omni_detective_script_uses_own_repo_root(self) -> None:
        script = Path("scripts/run_omni_detective_pilot.sh").read_text(encoding="utf-8")

        self.assertIn("REPO_ROOT=", script)
        self.assertIn('cd "$REPO_ROOT"', script)
        self.assertIn('export PYTHONPATH="$REPO_ROOT', script)
        self.assertNotIn("cd /data02/usr/wangqihao/Demo/test/cvr", script)
        self.assertNotIn("PYTHONPATH=/data02/usr/wangqihao/Demo/test/cvr", script)

    def test_omni_detective_script_has_gpu_resource_policy(self) -> None:
        script = Path("scripts/run_omni_detective_pilot.sh").read_text(encoding="utf-8")

        self.assertIn("MAX_GPUS", script)
        self.assertIn("GPU_IDS", script)
        self.assertIn("MODEL_STAGE", script)
        self.assertIn("one Omni model per run", script)
        self.assertIn("refusing to run with GPU_COUNT", script)

    def test_omni_detective_script_accepts_run_root_cli_override(self) -> None:
        script = Path("scripts/run_omni_detective_pilot.sh").read_text(encoding="utf-8")

        self.assertIn("--run-root", script)
        self.assertIn('RUN_ROOT="$2"', script)
        self.assertNotIn("omni_detective_pilot_20260422", script)
        self.assertIn("omni_detective_pilot", script)

    def test_video_edit_env_script_is_read_only_and_checks_wan_layout(self) -> None:
        script = Path("scripts/check_video_edit_env.sh").read_text(encoding="utf-8")

        self.assertIn("03_audio_vlm2vec_backbone", script)
        self.assertIn("Wan2.1-VACE-1.3B", script)
        self.assertIn("Wan2.1-VACE-14B", script)
        self.assertIn("huggingface_hub", script)
        self.assertIn("nvidia-smi", script)
        self.assertIn("never downloads or runs generation", script)
        self.assertNotIn("snapshot_download", script)
        self.assertNotIn("python -m app.composed_data", script)

    def test_wan_vace_download_script_targets_expected_models(self) -> None:
        script = Path("scripts/download_wan_vace_models.sh").read_text(encoding="utf-8")

        self.assertIn("Wan-AI/Wan2.1-VACE-1.3B", script)
        self.assertIn("Wan-AI/Wan2.1-VACE-14B", script)
        self.assertIn("Wan-Video/Wan2.1.git", script)
        self.assertIn("--model-size 1.3B|14B|both", script)
        self.assertIn("03_audio_vlm2vec_backbone", script)
        self.assertIn("snapshot_download", script)

    def test_synthetic_validation_script_runs_known_pair_validation(self) -> None:
        script = Path("scripts/run_synthetic_known_pairs_validation.sh").read_text(encoding="utf-8")

        self.assertIn("--run-root", script)
        self.assertIn("--known-pairs", script)
        self.assertIn("--clip-annotations", script)
        self.assertIn("validate-known-pairs", script)
        self.assertIn("judged_synthetic_pair_proposals.jsonl", script)
        self.assertIn("accepted_synthetic_pairs.jsonl", script)
        self.assertIn("synthetic_pilot_review.md", script)
        self.assertIn("one Omni model per run", script)
        self.assertIn("refusing to run with GPU_COUNT", script)


if __name__ == "__main__":
    unittest.main()
