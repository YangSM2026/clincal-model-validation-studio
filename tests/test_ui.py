"""Smoke tests for the two Streamlit workflows (optional UI dependencies)."""
from pathlib import Path
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "install requirements-ui.txt to test the interface")
class UiTests(unittest.TestCase):
    def test_workflow_switch_hides_stale_results(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Held-out performance", [item.value for item in app.subheader])
        self.assertEqual(len(app.metric), 4)
        app.radio[1].set_value("Evaluate fixed probabilities").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 4)
        self.assertIn("Fixed-probability performance", [item.value for item in app.subheader])
        self.assertNotIn("Constant-probability comparator", " ".join(item.value for item in app.markdown))

    def test_supplied_data_requires_file(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run()
        app.radio[0].set_value("Upload CSV").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertFalse(app.button)


if __name__ == "__main__":
    unittest.main()
