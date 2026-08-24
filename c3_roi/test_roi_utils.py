from __future__ import annotations

import unittest

from c3_roi.roi_utils import build_roi_record, parse_bbox


class RoiUtilsTests(unittest.TestCase):
    def test_parse_bbox_accepts_xywh_and_rejects_empty_values(self) -> None:
        self.assertEqual(parse_bbox("10,20,30,40"), (10, 20, 30, 40))
        self.assertIsNone(parse_bbox(""))


    def test_valid_mask_bbox_is_expanded_and_clamped(self) -> None:
        record = build_roi_record(
            image_id="7",
            width=100,
            height=80,
            bbox="10,20,30,20",
            fallback_reason="ok",
            margin=0.1,
        )
        self.assertEqual(record["roi_mode"], "mask_bbox")
        self.assertEqual(record["roi_bbox"], "7,18,36,24")


    def test_failed_mask_uses_full_image_without_dropping_sample(self) -> None:
        record = build_roi_record(
            image_id="8",
            width=100,
            height=80,
            bbox="",
            fallback_reason="segment_hand_failed",
        )
        self.assertEqual(record["roi_mode"], "global_fallback")
        self.assertEqual(record["roi_bbox"], "0,0,100,80")
        self.assertEqual(record["fallback_reason"], "segment_hand_failed")


if __name__ == "__main__":
    unittest.main()
