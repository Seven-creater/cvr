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


if __name__ == "__main__":
    unittest.main()
