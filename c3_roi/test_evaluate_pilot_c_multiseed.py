import unittest

import numpy as np
import pandas as pd

from c3_roi.evaluate_pilot_c_multiseed import (
    aggregate_multiseed,
    combine_seed_predictions,
    summarize_seed,
)


def _oof() -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": ["a", "b", "c", "d"],
        "fold": [1, 1, 2, 2],
        "sex": ["F", "M", "F", "M"],
        "target_months": [30.0, 90.0, 150.0, 200.0],
        "base_clean": [30.0, 90.0, 150.0, 200.0],
        "c_clean": [30.1, 90.1, 150.1, 200.1],
        "base_artifact": [32.0, 92.0, 152.0, 202.0],
        "c_artifact": [31.0, 91.0, 151.0, 201.0],
    })


class MultiSeedPilotCTests(unittest.TestCase):
    def test_combine_seed_predictions_aligns_by_image_id(self):
        oof = _oof()
        baseline = pd.DataFrame({
            "image_id": ["d", "c", "b", "a"],
            "artifact_prediction_months": [203.0, 153.0, 93.0, 33.0],
        })
        pilot = pd.DataFrame({
            "image_id": ["b", "a", "d", "c"],
            "artifact_prediction_months": [91.0, 31.0, 201.0, 151.0],
        })

        result = combine_seed_predictions(oof, baseline, pilot, seed=13, weight_c=0.75)

        self.assertEqual(result["image_id"].tolist(), ["a", "b", "c", "d"])
        self.assertEqual(result["artifact_seed"].unique().tolist(), [13])
        self.assertAlmostEqual(result.loc[0, "blend_artifact"], 31.5)

    def test_seed_summary_reports_clear_blend_gain(self):
        frame = combine_seed_predictions(
            _oof(),
            _oof()[["image_id", "base_artifact"]].rename(
                columns={"base_artifact": "artifact_prediction_months"}
            ),
            _oof()[["image_id", "c_artifact"]].rename(
                columns={"c_artifact": "artifact_prediction_months"}
            ),
            seed=12,
            weight_c=0.75,
        )

        report = summarize_seed(frame, repetitions=500, bootstrap_seed=7)

        self.assertAlmostEqual(report["baseline_artifact_mae"], 2.0)
        self.assertAlmostEqual(report["blend_artifact_mae"], 1.25)
        self.assertAlmostEqual(report["blend_artifact_gain"], 0.75)
        self.assertLess(report["blend_delta_ci_95"][1], 0.0)

    def test_multiseed_aggregation_uses_per_image_mean_and_worst_error(self):
        first = combine_seed_predictions(
            _oof(),
            pd.DataFrame({"image_id": ["a", "b", "c", "d"],
                          "artifact_prediction_months": [32, 92, 152, 202]}),
            pd.DataFrame({"image_id": ["a", "b", "c", "d"],
                          "artifact_prediction_months": [31, 91, 151, 201]}),
            seed=12, weight_c=1.0,
        )
        second = combine_seed_predictions(
            _oof(),
            pd.DataFrame({"image_id": ["a", "b", "c", "d"],
                          "artifact_prediction_months": [34, 94, 154, 204]}),
            pd.DataFrame({"image_id": ["a", "b", "c", "d"],
                          "artifact_prediction_months": [32, 92, 152, 202]}),
            seed=13, weight_c=1.0,
        )

        report = aggregate_multiseed(
            [first, second], repetitions=500, bootstrap_seed=9
        )

        self.assertEqual(report["image_count"], 4)
        self.assertEqual(report["artifact_seeds"], [12, 13])
        self.assertAlmostEqual(report["mean_seed_baseline_mae"], 3.0)
        self.assertAlmostEqual(report["mean_seed_blend_mae"], 1.5)
        self.assertAlmostEqual(report["worst_seed_baseline_mae"], 4.0)
        self.assertAlmostEqual(report["worst_seed_blend_mae"], 2.0)
        self.assertLess(report["mean_seed_delta_ci_95"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
