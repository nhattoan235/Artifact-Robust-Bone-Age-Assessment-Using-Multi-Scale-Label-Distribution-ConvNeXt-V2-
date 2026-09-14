from __future__ import annotations

import unittest

import pandas as pd

from c3_roi.compare_pilot_b_c_fold import compare_frame


class ComparePilotBCFoldTests(unittest.TestCase):
    def test_comparison_attributes_shared_gain_to_consistency(self):
        frame = pd.DataFrame({
            "target_months": [30.0, 90.0, 150.0, 210.0],
            "age_bin": ["0-59", "60-119", "120-179", "180-228"],
            "base_clean": [31.0, 91.0, 151.0, 211.0],
            "b_clean": [31.0, 91.0, 151.0, 211.0],
            "c_clean": [31.0, 91.0, 151.0, 211.0],
            "base_artifact": [34.0, 94.0, 154.0, 214.0],
            "b_artifact": [33.0, 93.0, 153.0, 213.0],
            "c_artifact": [31.0, 91.0, 151.0, 211.0],
        })
        report = compare_frame(frame, repetitions=200, seed=3)
        self.assertEqual(report["overall"]["b_vs_baseline_artifact"]["gain"], 1.0)
        self.assertEqual(report["overall"]["c_vs_b_artifact"]["gain"], 2.0)
        self.assertTrue(
            report["consistency_diagnosis"]["c_adds_significant_artifact_gain_over_b"]
        )
        self.assertFalse(
            report["consistency_diagnosis"]["c_significantly_harms_clean_vs_b"]
        )


if __name__ == "__main__":
    unittest.main()
