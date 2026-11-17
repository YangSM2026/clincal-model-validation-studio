"""End-to-end command-line checks using the fictional example."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo.csv"
ARGS = [
    str(DEMO), "--outcome", "event_30d", "--event", "yes",
    "--numeric", "age", "baseline_score", "--categorical", "group",
]


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "cmvs", *args], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )


class CliTests(unittest.TestCase):
    def test_report_is_json_and_input_unchanged(self):
        before = hashlib.sha256(DEMO.read_bytes()).digest()
        result = run(*ARGS)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertGreater(len(report["predictions"]), 0)
        self.assertGreaterEqual(report["metrics"]["roc_auc"], 0)
        self.assertLessEqual(report["metrics"]["roc_auc"], 1)
        self.assertEqual(hashlib.sha256(DEMO.read_bytes()).digest(), before)

    def test_report_cannot_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "run.json"
            path.write_text("old report", encoding="utf-8")
            result = run(*ARGS, "--output", str(path))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "old report")

    def test_exclusive_report_creation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "run.json"
            result = run(*ARGS, "--output", str(path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("audit", json.loads(path.read_text(encoding="utf-8")))
            self.assertFalse(result.stdout)

    def test_fixed_probability_workflow(self):
        result = run(str(DEMO), "--outcome", "event_30d", "--event", "yes",
                     "--probability", "fixed_risk")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(len(report["predictions"]), 600)
        self.assertIsNone(report["comparator"])

    def test_reject_conflicting_model_modes(self):
        result = run(*ARGS, "--probability", "fixed_risk")
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined", result.stderr)


if __name__ == "__main__":
    unittest.main()
