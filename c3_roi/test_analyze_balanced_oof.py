from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from c3_roi.analyze_balanced_oof import (
    balanced_mean,
    evaluate_balanced_oof,
    prepare_oof,
)


class BalancedOOFTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "image_id": ["a", "b", "c", "d", "e"],
            "fold": [1, 1, 1, 2, 2],
            "sex": ["F", "F", "M", "F", "M"],
            "target_months": [30.0, 31.0, 32.0, 90.0, 150.0],
            "base_clean": [31.0, 32.0, 33.0, 94.0, 155.0],
            "c_clean": [32.0, 33.0, 34.0, 92.0, 151.0],
            "base_artifact": [33.0, 34.0, 35.0, 96.0, 157.0],
            "c_artifact": [31.0, 32.0, 33.0, 93.0, 152.0],
        })

    def test_age_macro_gives_each_age_bin_equal_weight(self):
        frame = prepare_oof(self._frame())
        error = np.abs(frame["base_clean"] - frame["target_months"]).to_numpy()
        # Bin MAEs are 1, 4 and 5; the three-sample first bin still has 1/3 weight.
        self.assertEqual(balanced_mean(frame, error, ("age_bin",)), (1 + 4 + 5) / 3)

    def test_evaluation_reports_candidate_delta_and_interval(self):
        frame = prepare_oof(self._frame())
        metrics, age = evaluate_balanced_oof(frame, repetitions=200, seed=7)
        row = metrics[
            (metrics["view"] == "artifact")
            & (metrics["scheme"] == "age_macro")
            & (metrics["model"] == "c")
        ].iloc[0]
        self.assertLess(row["delta_vs_baseline"], 0.0)
        self.assertLessEqual(row["delta_ci_low"], row["delta_ci_high"])
        self.assertEqual(set(age["age_bin"]), {"0-59", "60-119", "120-179"})


if __name__ == "__main__":
    unittest.main()
