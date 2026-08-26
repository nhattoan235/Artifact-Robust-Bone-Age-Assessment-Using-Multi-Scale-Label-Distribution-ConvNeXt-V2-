import unittest

from artifact_eval.evaluate_models import build_model_registry, compute_metrics, fuse_model_output, parse_roi_bbox


class EvaluateModelsTests(unittest.TestCase):
    def test_registry_contains_all_required_model_families(self):
        registry = build_model_registry()
        self.assertEqual(
            {item["model_id"] for item in registry},
            {"E1_P7_5FOLD", "E0_IMAGE_ONLY", "E1_P10_VALIDATION", "E2_DUAL_OUTPUT", "D3_5FOLD", "C3_ROI_5FOLD"},
        )

    def test_roi_bbox_parser_returns_xywh(self):
        self.assertEqual(parse_roi_bbox("10,20,30,40"), (10, 20, 30, 40))

    def test_metrics_report_count_and_mae(self):
        result = compute_metrics([1.0, 3.0], [2.0, 2.0])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["mae_months"], 1.0)

    def test_label_distribution_fusion_uses_month_units_for_both_branches(self):
        fused = fuse_model_output(
            regression_normalized=[0.0],
            distribution_probabilities=[[1.0, 0.0]],
            target_mean=100.0,
            target_std=10.0,
            regression_weight=0.5,
        )
        self.assertEqual(float(fused[0]), 50.0)


if __name__ == "__main__":
    unittest.main()
