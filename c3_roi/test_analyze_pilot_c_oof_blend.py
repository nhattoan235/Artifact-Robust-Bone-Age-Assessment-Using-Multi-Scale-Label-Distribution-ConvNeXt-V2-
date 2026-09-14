import unittest

import numpy as np
import pandas as pd

from c3_roi.analyze_pilot_c_oof_blend import (
    AGE_LABELS,
    analyze_weight,
    crossfit_weights,
    subgroup_bootstrap,
)


def _synthetic_oof() -> pd.DataFrame:
    rows = []
    ages = [30.0, 90.0, 150.0, 200.0]
    for fold in range(1, 6):
        for index, age in enumerate(ages):
            rows.append({
                "image_id": f"{fold}-{index}",
                "fold": fold,
                "sex": "F" if index % 2 == 0 else "M",
                "target_months": age,
                "base_clean": age,
                "c_clean": age + 0.05,
                "base_artifact": age + 2.0,
                "c_artifact": age + 0.8,
            })
    return pd.DataFrame(rows)


class PilotCOOFBlendTests(unittest.TestCase):
    def test_weight_endpoints_reproduce_baseline_and_pilot(self):
        frame = _synthetic_oof()
        baseline = analyze_weight(frame, 0.0)
        pilot = analyze_weight(frame, 1.0)

        self.assertAlmostEqual(baseline["clean_mae"], 0.0)
        self.assertAlmostEqual(baseline["artifact_mae"], 2.0)
        self.assertAlmostEqual(pilot["clean_mae"], 0.05)
        self.assertAlmostEqual(pilot["artifact_mae"], 0.8)
        self.assertEqual(set(pilot["age_bins"]), set(AGE_LABELS))

    def test_crossfit_selects_best_feasible_weight_without_heldout_fold(self):
        frame = _synthetic_oof()
        result = crossfit_weights(
            frame,
            weights=np.array([0.0, 0.5, 1.0]),
            min_disagreement_reduction=0.40,
        )

        self.assertTrue(result["all_training_splits_feasible"])
        self.assertEqual(result["selected_weights"], {
            "1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0,
        })
        self.assertAlmostEqual(result["summary"]["artifact_gain"], 1.2)
        self.assertEqual(len(result["predictions"]), len(frame))

    def test_subgroup_bootstrap_detects_consistent_artifact_gain(self):
        result = subgroup_bootstrap(
            _synthetic_oof(), repetitions=1000, seed=17
        )

        for label in AGE_LABELS:
            interval = result["age_bin"][label]["artifact_delta_ci_95"]
            self.assertLess(interval[1], 0.0)
        for label in ("F", "M"):
            interval = result["sex"][label]["artifact_delta_ci_95"]
            self.assertLess(interval[1], 0.0)


if __name__ == "__main__":
    unittest.main()
