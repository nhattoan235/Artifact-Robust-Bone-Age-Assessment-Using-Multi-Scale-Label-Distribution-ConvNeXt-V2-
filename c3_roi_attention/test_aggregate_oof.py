import unittest

from c3_roi_attention.aggregate_oof import metrics, paired_bootstrap_delta


class AggregateOOFTests(unittest.TestCase):
    def test_metrics_uses_absolute_error_in_months(self):
        rows = [
            {"target_months": 10.0, "prediction_months": 12.0},
            {"target_months": 20.0, "prediction_months": 16.0},
        ]
        self.assertEqual(metrics(rows)["mae_months"], 3.0)

    def test_paired_delta_is_attention_minus_c3(self):
        rows = [
            {"target_months": 10.0, "attention_prediction_months": 11.0, "c3_prediction_months": 13.0},
            {"target_months": 20.0, "attention_prediction_months": 18.0, "c3_prediction_months": 24.0},
        ]
        result = paired_bootstrap_delta(rows, n=100, seed=42)
        self.assertEqual(result["estimate_months"], -2.0)
        self.assertLessEqual(result["lower_95"], result["upper_95"])


if __name__ == "__main__":
    unittest.main()
