from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_MASK_CODE = Path(r"D:\Learning\DoAn Tot nghiep\gpt-image-bone-age-synthesis\models\cau_hinh_C")
sys.path.insert(0, str(EXTERNAL_MASK_CODE))
from mask_generator import segment_hand  # noqa: E402
from c3_roi.segmentation import segment_hand_for_c3, segment_hand_with_fallback  # noqa: E402


class C3SegmentationTests(unittest.TestCase):
    def test_strict_policy_does_not_run_rescue_when_strict_succeeds(self) -> None:
        expected = (np.ones((2, 2), dtype=np.uint8), "hull", (0, 0, 2, 2))
        with patch("c3_roi.segmentation.segment_hand", return_value=expected) as segment:
            result = segment_hand_with_fallback(np.zeros((2, 2), dtype=np.uint8))

        self.assertEqual(result[3], "ok")
        self.assertEqual(segment.call_count, 1)
        self.assertEqual(segment.call_args.kwargs["max_borders"], 2)

    def test_border_rescue_is_explicit_when_strict_segmentation_fails(self) -> None:
        failed = (None, None, None)
        rescued = (np.ones((2, 2), dtype=np.uint8), "hull", (0, 0, 2, 2))
        with patch("c3_roi.segmentation.segment_hand", side_effect=[failed, rescued]) as segment:
            result = segment_hand_with_fallback(np.zeros((2, 2), dtype=np.uint8))

        self.assertEqual(result[3], "border_rescue")
        self.assertEqual(segment.call_count, 2)
        self.assertEqual(segment.call_args_list[0].kwargs["max_borders"], 2)
        self.assertEqual(segment.call_args_list[1].kwargs["max_borders"], 3)

    def test_failed_both_policies_returns_explicit_failure_reason(self) -> None:
        with patch("c3_roi.segmentation.segment_hand", return_value=(None, None, None)):
            result = segment_hand_with_fallback(np.zeros((2, 2), dtype=np.uint8))

        self.assertEqual(result, (None, None, None, "segment_hand_failed"))

    def test_c3_policy_recovers_hand_touching_three_image_borders(self) -> None:
        image_path = ROOT / "data/goc/boneage-training-dataset/boneage-training-dataset/1400.png"
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"))

        strict_region, _, strict_bbox = segment_hand(gray, max_borders=2)
        relaxed_region, _, relaxed_bbox = segment_hand_for_c3(gray)

        self.assertIsNone(strict_region)
        self.assertIsNone(strict_bbox)
        self.assertIsNotNone(relaxed_region)
        self.assertIsNotNone(relaxed_bbox)


if __name__ == "__main__":
    unittest.main()
