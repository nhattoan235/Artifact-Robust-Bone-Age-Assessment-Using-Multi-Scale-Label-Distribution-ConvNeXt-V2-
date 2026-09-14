from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from c3_roi.evaluate_pilot_c_5fold import merge_oof, summarize_oof


def _frame(fold: int, *, baseline_artifact_offset: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = 2808 if fold == 1 else 2807
    image_ids = [f"f{fold}_{index}" for index in range(count)]
    target = (np.arange(count) % 228) + 1.0
    sex = np.where(np.arange(count) % 2, "M", "F")
    c_clean = target + 1.0
    c_artifact = target + 1.1
    base_clean = target + 1.0
    base_artifact = target + baseline_artifact_offset
    c = pd.DataFrame({
        "image_id": image_ids, "sex": sex, "target_months": target,
        "c_clean": c_clean, "c_artifact": c_artifact, "fold": fold,
    })
    baseline = pd.DataFrame({
        "image_id": image_ids, "sex": sex, "target_months": target,
        "base_clean": base_clean, "base_artifact": base_artifact,
    })
    return c, baseline


class EvaluatePilotC5FoldTests(unittest.TestCase):
    def test_merge_rejects_duplicate_image_ids_across_folds(self):
        c1, b1 = _frame(1)
        c2, b2 = _frame(2)
        c2.loc[0, "image_id"] = c1.loc[0, "image_id"]
        with self.assertRaises(ValueError):
            merge_oof([c1, c2], [b1, b2])

    def test_summary_reports_pooled_metrics_bootstrap_and_gates(self):
        c, b = _frame(2)
        frame = c.merge(b, on=["image_id", "sex", "target_months"])
        report = summarize_oof(frame, repetitions=200)
        self.assertEqual(report["count"], 2807)
        self.assertEqual(report["folds"], [2])
        self.assertGreater(report["artifact_gain_months"], 0.0)
        self.assertEqual(len(report["artifact_paired_bootstrap_95_ci_months"]), 2)
        self.assertIn("all_pass", report["gates"])
        self.assertIn("F", report["subgroups_by_sex"])
        self.assertIn("60-119", report["subgroups_by_age_bin"])


if __name__ == "__main__":
    unittest.main()
