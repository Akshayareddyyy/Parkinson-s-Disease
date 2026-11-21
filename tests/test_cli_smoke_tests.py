"""
Unit tests for CLI entrypoints and smoke test scripts.
Verifies clean exit codes and expected status outputs when executing runners.
"""

import sys
import os
import unittest
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCLISmokeTests(unittest.TestCase):

    def test_root_smoke_test_runner(self):
        """Verifies code/smoke_test.py runs cleanly and reports all 5 steps passed."""
        script_path = os.path.join(ROOT_DIR, "code", "smoke_test.py")
        res = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=ROOT_DIR
        )

        self.assertEqual(res.returncode, 0, f"smoke_test.py failed with stderr: {res.stderr}")
        self.assertIn("ALL SMOKE TESTS PASSED SUCCESSFULLY!", res.stdout)
        self.assertIn("Dataset successfully loaded", res.stdout)
        self.assertIn("PennyLane version", res.stdout)

    def test_phase5_smoke_test_runner(self):
        """Verifies experiments/phase5/code/smoke_test.py passes quantum environment checks."""
        script_path = os.path.join(ROOT_DIR, "experiments", "phase5", "code", "smoke_test.py")
        res = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.join(ROOT_DIR, "experiments", "phase5", "code")
        )

        self.assertEqual(res.returncode, 0, f"phase5 smoke_test.py failed with stderr: {res.stderr}")
        self.assertIn("ALL UNIT & SMOKE TESTS PASSED CLEANLY WITH ZERO ERRORS!", res.stdout)
        self.assertIn("9-Qubit VQC Architecture", res.stdout)
        self.assertIn("9-Qubit PQC-R Architecture", res.stdout)

    def test_phase5_pre_execution_audit_runner(self):
        """Verifies experiments/phase5/code/pre_execution_audit.py audits hardware and checkpoints."""
        script_path = os.path.join(ROOT_DIR, "experiments", "phase5", "code", "pre_execution_audit.py")
        res = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.join(ROOT_DIR, "experiments", "phase5", "code")
        )

        self.assertEqual(res.returncode, 0, f"pre_execution_audit.py failed with stderr: {res.stderr}")
        self.assertIn("ALL 10 PRE-EXECUTION CHECKS PASSED WITH ZERO DEFECTS", res.stdout)
        self.assertIn("Canonical Dataset Integrity", res.stdout)


if __name__ == "__main__":
    unittest.main()
