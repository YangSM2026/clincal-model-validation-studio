"""Smoke tests for the two Streamlit workflows (optional UI dependencies)."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
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
        self.assertIn("Training rows: 450 · Holdout rows: 150",
                      [item.value for item in app.caption])
        app.radio[1].set_value("Evaluate fixed probabilities").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 4)
        self.assertIn("Fixed-probability performance", [item.value for item in app.subheader])
        self.assertNotIn("Constant-probability comparator", " ".join(item.value for item in app.markdown))

    def test_missing_uploaded_outcomes_are_not_categories_in_either_workflow(self):
        rows = ["label,x,p"] + [f"{'001' if i % 2 else '000'},{i},{.7 if i % 2 else .2}"
                                 for i in range(20)]
        rows += [f"{missing},21,.4" for missing in ("", "NA", "n/a", "null", "nan", " ")]
        uploaded = SimpleNamespace(getvalue=lambda: "\n".join(rows).encode())
        for fixed in (False, True):
            with self.subTest(fixed=fixed), patch("streamlit.file_uploader", return_value=uploaded):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
                app.run()
                app.radio[0].set_value("Upload CSV").run()
                if fixed:
                    app.radio[1].set_value("Evaluate fixed probabilities").run()
                app.selectbox[0].set_value("label").run()
                self.assertEqual(app.selectbox[1].options, ["000", "001"])
                app.selectbox[1].set_value("001").run()
                if fixed:
                    app.selectbox[2].set_value("p").run()
                else:
                    app.multiselect[0].set_value(["x"]).run()
                self.assertFalse(app.button[0].disabled)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(len(app.metric), 4)
                self.assertEqual(app.session_state["validation"]["result"]["audit"]["n_missing_outcome"], 6)

    def test_supplied_data_requires_file(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run()
        app.radio[0].set_value("Upload CSV").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertFalse(app.button)


if __name__ == "__main__":
    unittest.main()
