"""Regressions for decision-curve thresholds on fictional inputs."""
import unittest
from math import nextafter

import pandas as pd

from cmvs.core import evaluate_predictions


class DecisionThresholdTests(unittest.TestCase):
    def test_exact_5_percent_grid_and_inclusive_ties(self):
        grid = [float(f"{step / 100:.2f}") for step in range(5, 100, 5)]
        for threshold in grid:
            with self.subTest(threshold=threshold):
                data = pd.DataFrame({"label": ["yes", "no"],
                                     "p": [threshold, nextafter(threshold, 0)]})
                result = evaluate_predictions(data, "label", "yes", "p")
                curve = result["decision_curve"]
                self.assertEqual([point["threshold"] for point in curve], grid)
                self.assertEqual(curve[grid.index(threshold)]["net_benefit"], .5)
