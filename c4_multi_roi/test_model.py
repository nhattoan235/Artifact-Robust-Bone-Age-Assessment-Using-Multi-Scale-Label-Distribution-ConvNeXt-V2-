from __future__ import annotations

import unittest

import torch

from c4_multi_roi.model import SharedViewConvNeXt, build_c4_model
from p1_baseline.model import BoneAgeConvNeXt


class SharedViewModelTests(unittest.TestCase):
    def test_global_plus_six_forward_shape(self) -> None:
        model = build_c4_model(pretrained=False, view_mode="global_plus_six")
        prediction = model(torch.zeros(2, 7, 3, 64, 64), torch.zeros(2, 1))
        self.assertEqual(prediction.shape, (2,))

    def test_model_has_one_shared_backbone(self) -> None:
        model = build_c4_model(pretrained=False, view_mode="global_plus_six")
        self.assertIsInstance(model, SharedViewConvNeXt)
        self.assertTrue(hasattr(model, "backbone"))
        self.assertFalse(hasattr(model, "backbones"))

    def test_wrong_view_count_is_rejected(self) -> None:
        model = build_c4_model(pretrained=False, view_mode="global_plus_six")
        with self.assertRaisesRegex(ValueError, "7 views"):
            model(torch.zeros(1, 6, 3, 64, 64), torch.zeros(1, 1))

    def test_global_only_state_dict_is_e1_compatible(self) -> None:
        torch.manual_seed(42)
        e1 = BoneAgeConvNeXt(False, 16, 256, 0.2)
        c4 = build_c4_model(pretrained=False, view_mode="global_only")
        c4.load_state_dict(e1.state_dict(), strict=True)
        e1.eval()
        c4.eval()
        image = torch.randn(1, 3, 64, 64)
        sex = torch.ones(1, 1)
        with torch.inference_mode():
            self.assertTrue(torch.allclose(e1(image, sex), c4(image[:, None], sex), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
