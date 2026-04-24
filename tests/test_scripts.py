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


if __name__ == "__main__":
    unittest.main()
