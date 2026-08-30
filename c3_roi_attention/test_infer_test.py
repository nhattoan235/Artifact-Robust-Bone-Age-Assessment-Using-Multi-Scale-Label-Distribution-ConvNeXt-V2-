import unittest

from c3_roi_attention.infer_test import average_fold_predictions, paired_delta_bootstrap


class TestAttentionTestInference(unittest.TestCase):
    def test_average_fold_predictions_matches_by_id_not_row_order(self):
        folds = [
            [
                {"image_id": "1", "sex": "M", "prediction_months": 10.0},
                {"image_id": "2", "sex": "F", "prediction_months": 20.0},
            ],
            [
                {"image_id": "2", "sex": "F", "prediction_months": 24.0},
                {"image_id": "1", "sex": "M", "prediction_months": 14.0},
            ],
        ]

        result = average_fold_predictions(folds)

        self.assertEqual(
            result,
            [
                {"image_id": "1", "sex": "M", "prediction_months": 12.0},
                {"image_id": "2", "sex": "F", "prediction_months": 22.0},
            ],
        )

    def test_average_fold_predictions_rejects_mismatched_ids(self):
        folds = [
            [{"image_id": "1", "sex": "M", "prediction_months": 10.0}],
            [{"image_id": "2", "sex": "M", "prediction_months": 12.0}],
        ]

        with self.assertRaisesRegex(ValueError, "ID sets differ"):
            average_fold_predictions(folds)

    def test_paired_delta_bootstrap_uses_attention_minus_baseline(self):
        rows = [
            {"target_months": 10.0, "attention_prediction_months": 11.0, "baseline_prediction_months": 13.0},
            {"target_months": 20.0, "attention_prediction_months": 18.0, "baseline_prediction_months": 24.0},
        ]

        result = paired_delta_bootstrap(rows, n_bootstrap=200, seed=42)

        self.assertEqual(result["estimate_months"], -2.0)
        self.assertLessEqual(result["lower_95"], result["estimate_months"])
        self.assertGreaterEqual(result["upper_95"], result["estimate_months"])


if __name__ == "__main__":
    unittest.main()
