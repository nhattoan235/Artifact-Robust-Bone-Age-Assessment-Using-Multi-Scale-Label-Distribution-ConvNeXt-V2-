from __future__ import annotations

import inspect
import unittest

import cv2
import numpy as np

from c4_multi_roi.geometry import analyze_six_rois, propose_six_rois
from c4_multi_roi.schema import ROI_NAMES


def synthetic_hand() -> np.ndarray:
    mask = np.zeros((240, 160), dtype=np.uint8)
    cv2.rectangle(mask, (35, 75), (115, 215), 255, -1)
    for left, right, top in ((38, 50, 42), (53, 66, 25), (69, 84, 15), (88, 101, 30), (105, 116, 48)):
        cv2.rectangle(mask, (left, top), (right, 105), 255, -1)
    cv2.ellipse(mask, (130, 132), (28, 14), -35, 0, 360, 255, -1)
    return mask


class MultiROIGeometryTests(unittest.TestCase):
    def assert_boxes_in_bounds(self, boxes: dict, width: int, height: int) -> None:
        self.assertEqual(tuple(boxes), ROI_NAMES)
        for box in boxes.values():
            x, y, w, h = box
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, width)
            self.assertLessEqual(y + h, height)

    def test_upright_hand_produces_six_in_bounds_boxes(self) -> None:
        mask = synthetic_hand()
        boxes = propose_six_rois(mask, width=mask.shape[1], height=mask.shape[0])
        self.assert_boxes_in_bounds(boxes, mask.shape[1], mask.shape[0])

    def test_rotated_hand_is_deterministic_and_in_bounds(self) -> None:
        mask = synthetic_hand()
        matrix = cv2.getRotationMatrix2D((80, 120), 24, 1.0)
        rotated = cv2.warpAffine(mask, matrix, (160, 240), flags=cv2.INTER_NEAREST)
        first = propose_six_rois(rotated, width=160, height=240)
        second = propose_six_rois(rotated, width=160, height=240)
        self.assertEqual(first, second)
        self.assert_boxes_in_bounds(first, 160, 240)

    def test_mirror_moves_thumb_roi_to_other_side(self) -> None:
        mask = synthetic_hand()
        original = propose_six_rois(mask, width=160, height=240)
        mirrored = propose_six_rois(np.fliplr(mask), width=160, height=240)
        thumb_original_x = original["mcp_thumb"][0] + original["mcp_thumb"][2] / 2
        index_original_x = original["mcp_index"][0] + original["mcp_index"][2] / 2
        thumb_mirror_x = mirrored["mcp_thumb"][0] + mirrored["mcp_thumb"][2] / 2
        index_mirror_x = mirrored["mcp_index"][0] + mirrored["mcp_index"][2] / 2
        self.assertGreater(thumb_original_x, index_original_x)
        self.assertLess(thumb_mirror_x, index_mirror_x)

    def test_empty_mask_uses_explicit_fallback(self) -> None:
        result = analyze_six_rois(np.zeros((200, 100), dtype=np.uint8), width=100, height=200)
        self.assertEqual(result.quality_flag, "fallback_empty_mask")
        self.assert_boxes_in_bounds(result.boxes, 100, 200)

    def test_geometry_api_cannot_receive_labels_or_predictions(self) -> None:
        parameters = set(inspect.signature(analyze_six_rois).parameters)
        self.assertFalse(parameters & {"age", "target", "sex", "prediction"})


if __name__ == "__main__":
    unittest.main()
