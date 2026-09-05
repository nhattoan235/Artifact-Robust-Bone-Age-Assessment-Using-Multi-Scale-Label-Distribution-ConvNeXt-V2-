from __future__ import annotations

from typing import Any


ROI_SUCCESS_REASONS = frozenset({"ok", "border_rescue"})


def parse_bbox(value: str | None) -> tuple[int, int, int, int] | None:
    if not value or not value.strip():
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected x,y,w,h bbox, got {value!r}")
    x, y, width, height = (int(float(part)) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError(f"BBox dimensions must be positive, got {value!r}")
    return x, y, width, height


def clamp_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = bbox
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(left, min(width, x + box_width))
    bottom = max(top, min(height, y + box_height))
    return left, top, right - left, bottom - top


def expand_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int, margin: float
) -> tuple[int, int, int, int]:
    if not 0 <= margin <= 0.5:
        raise ValueError("margin must be between 0 and 0.5")
    x, y, box_width, box_height = bbox
    dx = round(box_width * margin)
    dy = round(box_height * margin)
    return clamp_bbox(
        (x - dx, y - dy, box_width + 2 * dx, box_height + 2 * dy),
        width,
        height,
    )


def format_bbox(bbox: tuple[int, int, int, int]) -> str:
    return ",".join(str(value) for value in bbox)


def build_roi_record(
    *,
    image_id: str,
    width: int,
    height: int,
    bbox: str | None,
    fallback_reason: str,
    margin: float = 0.12,
) -> dict[str, Any]:
    parsed = None
    roi_fallback_reason = fallback_reason
    if fallback_reason in ROI_SUCCESS_REASONS:
        try:
            parsed = parse_bbox(bbox)
        except (TypeError, ValueError):
            roi_fallback_reason = "invalid_bbox"
        if parsed is None:
            roi_fallback_reason = "invalid_bbox"

    if parsed is not None and fallback_reason in ROI_SUCCESS_REASONS:
        roi_bbox = expand_bbox(parsed, width, height, margin)
        roi_mode = "mask_bbox"
    else:
        roi_bbox = (0, 0, width, height)
        roi_mode = "global_fallback"
    return {
        "image_id": image_id,
        "roi_mode": roi_mode,
        "roi_bbox": format_bbox(roi_bbox),
        "fallback_reason": roi_fallback_reason,
    }
