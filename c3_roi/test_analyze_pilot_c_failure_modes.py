from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from c3_roi.analyze_pilot_c_failure_modes import (
    build_group_summary,
    prepare_predictions,
)


class PilotCFailureModeTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "image_id": ["a", "b", "c", "d"],
            "sex": ["F", "M", "F", "M"],
            "target_months": [30.0, 90.0, 150.0, 210.0],
            "fold": [1, 1, 2, 2],
            "base_clean": [31.0, 92.0, 153.0, 214.0],
            "c_clean": [30.0, 91.0, 152.0, 213.0],
            "base_artifact": [34.0, 95.0, 156.0, 217.0],
            "c_artifact": [32.0, 93.0, 154.0, 215.0],
        })

    def test_fixed_blend_and_delta_are_exact(self):
        result = prepare_predictions(self._frame(), weight_c=0.75)
        expected = 0.25 * result["base_clean"] + 0.75 * result["c_clean"]
        np.testing.assert_allclose(result["blend_clean"], expected)
        np.testing.assert_allclose(
            result["c_clean_delta"],
            abs(result["c_clean"] - result["target_months"])
            - abs(result["base_clean"] - result["target_months"]),
        )

    def test_summary_includes_fold_age_and_paired_interval(self):
        result = prepare_predictions(self._frame(), weight_c=0.75)
        summary = build_group_summary(result, repetitions=200, seed=7)
        self.assertEqual(
            set(summary["group_type"]),
            {"overall", "fold", "age_bin", "fold_age", "sex"},
        )
        overall = summary[summary["group_type"] == "overall"].iloc[0]
        self.assertLess(overall["c_clean_delta"], 0.0)
        self.assertLessEqual(
            overall["c_clean_delta_ci_low"], overall["c_clean_delta_ci_high"],
        )

    def test_duplicate_ids_are_rejected(self):
        frame = self._frame()
        frame.loc[1, "image_id"] = "a"
        with self.assertRaises(ValueError):
            prepare_predictions(frame)


if __name__ == "__main__":
    unittest.main()
