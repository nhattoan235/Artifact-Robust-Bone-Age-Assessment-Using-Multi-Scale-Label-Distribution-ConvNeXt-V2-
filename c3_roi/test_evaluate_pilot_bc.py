from __future__ import annotations

import unittest

import numpy as np

from c3_roi.evaluate_pilot_bc import (
    _config_path,
    paired_bootstrap_ci,
    summarize_predictions,
)


class EvaluatePilotBCTests(unittest.TestCase):
    def test_config_path_supports_c_fold2_and_rejects_b_fold2(self):
        from pathlib import Path

        root = Path("/tmp/pilots")
        self.assertEqual(
            _config_path("C", root, 2).name,
            "C3_Z26_C3_ROI_V2_PILOT_C_FOLD_2_SEED_42.toml",
        )
        with self.assertRaises(ValueError):
            _config_path("B", root, 2)

    def test_paired_bootstrap_ci_detects_clear_candidate_gain(self):
        target = np.arange(40, dtype=np.float64)
        baseline = target + 2.0
        candidate = target + 1.0
        low, high = paired_bootstrap_ci(
            target, baseline, candidate, repetitions=1000, seed=42
        )
        self.assertLess(high, 0.0)
        self.assertAlmostEqual(low, -1.0)

    def test_summary_reports_clean_stress_disagreement_and_subgroups(self):
        records = [
            {"image_id": "1", "sex": "F", "target_months": 30.0,
             "clean_prediction_months": 31.0, "artifact_prediction_months": 33.0},
            {"image_id": "2", "sex": "M", "target_months": 150.0,
             "clean_prediction_months": 148.0, "artifact_prediction_months": 146.0},
        ]
        report = summarize_predictions(records, [0, 60, 120, 180, 229], repetitions=500)
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["fold"], 1)
        self.assertAlmostEqual(report["clean"]["mae"], 1.5)
        self.assertAlmostEqual(report["artifact"]["mae"], 3.5)
        self.assertAlmostEqual(report["disagreement_months"]["mean"], 2.0)
        self.assertEqual(set(report["subgroups_by_sex"]), {"F", "M"})
        self.assertIn("0-59", report["subgroups_by_age_bin"])
        self.assertIn("120-179", report["subgroups_by_age_bin"])


if __name__ == "__main__":
    unittest.main()
