from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXTERNAL_MASK_CODE = Path(r"D:\Learning\DoAn Tot nghiep\gpt-image-bone-age-synthesis\models\cau_hinh_C")
sys.path.insert(0, str(EXTERNAL_MASK_CODE))
from mask_generator import segment_hand  # noqa: E402


# Some valid radiographs include the forearm touching the image boundary. Keep
# the permissive rule as an explicit rescue instead of making it the default:
# this prevents border/cassette regions from entering the ROI when the strict
# policy already finds a valid hand.
C3_STRICT_MAX_BORDERS = 2
C3_RESCUE_MAX_BORDERS = 3


def _segmentation_is_valid(result: tuple[object, object, object]) -> bool:
    region, _hull, bbox = result
    return region is not None and bbox is not None


def segment_hand_with_fallback(gray: np.ndarray):
    """Segment with a strict policy, then an explicit border-contact rescue.

    Returns ``(region, hull, bbox, reason)`` where ``reason`` is ``ok`` for
    the strict pass, ``border_rescue`` for the relaxed pass, and
    ``segment_hand_failed`` when both passes fail.
    """
    strict_result = segment_hand(gray, max_borders=C3_STRICT_MAX_BORDERS)
    if _segmentation_is_valid(strict_result):
        return (*strict_result, "ok")

    rescue_result = segment_hand(gray, max_borders=C3_RESCUE_MAX_BORDERS)
    if _segmentation_is_valid(rescue_result):
        return (*rescue_result, "border_rescue")

    return None, None, None, "segment_hand_failed"


def segment_hand_for_c3(gray: np.ndarray):
    """Backward-compatible three-value wrapper for the C3 segmenter."""
    return segment_hand_with_fallback(gray)[:3]
