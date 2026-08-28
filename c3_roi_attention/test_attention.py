import unittest

import torch

from p1_baseline.model import build_model


class SpatialAttentionModelTests(unittest.TestCase):
    def test_attention_model_returns_one_normalized_age_per_sample(self):
        model = build_model("convnext_tiny_spatial_attention", False, 16, 256, 0.2)
        output = model(torch.randn(2, 3, 64, 64), torch.tensor([[0.0], [1.0]]))
        self.assertEqual(output.shape, (2,))
        self.assertTrue(torch.isfinite(output).all())

    def test_attention_gate_is_identity_at_initialization(self):
        model = build_model("convnext_tiny_spatial_attention", False, 16, 256, 0.2)
        image = torch.randn(1, 3, 64, 64)
        attention = model.spatial_attention(image)
        self.assertTrue(torch.allclose(attention, torch.ones_like(attention), atol=1e-6))

    def test_identity_gate_preserves_convnext_pooling_order(self):
        model = build_model("convnext_tiny_spatial_attention", False, 16, 256, 0.2)
        model.eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            feature_map = model.features(image)
            expected = model.feature_projection(model.avgpool(feature_map))
            actual = model.image_features(image)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_same_seed_preserves_shared_c3_initialization(self):
        torch.manual_seed(42)
        baseline = build_model("convnext_tiny", False, 16, 256, 0.2)
        torch.manual_seed(42)
        attention = build_model("convnext_tiny_spatial_attention", False, 16, 256, 0.2)
        baseline_shared = list(baseline.sex_embedding.parameters()) + list(baseline.regressor.parameters())
        attention_shared = list(attention.sex_embedding.parameters()) + list(attention.regressor.parameters())
        self.assertEqual(len(baseline_shared), len(attention_shared))
        for expected, actual in zip(baseline_shared, attention_shared):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
