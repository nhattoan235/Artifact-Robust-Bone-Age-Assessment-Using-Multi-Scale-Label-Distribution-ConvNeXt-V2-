from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schema import ROI_NAMES


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class GeometryResult:
    boxes: dict[str, Box]
    quality_flag: str
    angle_degrees: float
    thumb_side: str
    hand_bbox: Box


_ROI_SPECS = {
    "carpal": (0.50, 0.76, 0.62, 0.25),
    "mcp_thumb": (0.84, 0.54, 0.25, 0.20),
    "mcp_index": (0.66, 0.43, 0.19, 0.18),
    "mcp_middle": (0.50, 0.40, 0.19, 0.18),
    "mcp_ring": (0.34, 0.43, 0.19, 0.18),
    "mcp_little": (0.17, 0.49, 0.22, 0.20),
}


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == index
    if int(component.sum()) < max(64, round(component.size * 0.02)):
        return None
    return component


def _clamp_box(points: np.ndarray, width: int, height: int) -> Box:
    left = max(0, min(width - 1, int(np.floor(points[:, 0].min()))))
    top = max(0, min(height - 1, int(np.floor(points[:, 1].min()))))
    right = max(left + 1, min(width, int(np.ceil(points[:, 0].max())) + 1))
    bottom = max(top + 1, min(height, int(np.ceil(points[:, 1].max())) + 1))
    return left, top, right - left, bottom - top


def _fallback_frame(width: int, height: int) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    return u, v, 0.12 * width, 0.88 * width, 0.15 * height, 0.96 * height


def analyze_six_rois(mask: np.ndarray, *, width: int, height: int) -> GeometryResult:
    if width <= 1 or height <= 1:
        raise ValueError("width and height must be greater than one")
    array = np.asarray(mask)
    if array.shape != (height, width):
        raise ValueError(f"mask shape {array.shape} does not match {(height, width)}")

    component = _largest_component(array)
    if component is None:
        u, v, q_min, q_max, t_min, t_max = _fallback_frame(width, height)
        quality_flag = "fallback_empty_mask"
        thumb_right = True
        hand_bbox = (
            round(q_min),
            round(t_min),
            max(1, round(q_max - q_min)),
            max(1, round(t_max - t_min)),
        )
    else:
        ys, xs = np.nonzero(component)
        points = np.column_stack((xs, ys)).astype(np.float64)
        centered = points - points.mean(axis=0, keepdims=True)
        covariance = np.cov(centered, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        v = vectors[:, int(np.argmax(values))]
        if v[1] < 0:
            v = -v
        u = np.array([v[1], -v[0]])
        q = points @ u
        t = points @ v
        q_min, q_max = np.percentile(q, (0.5, 99.5))
        t_min, t_max = np.percentile(t, (0.5, 99.5))
        q_span = max(float(q_max - q_min), 1.0)
        t_span = max(float(t_max - t_min), 1.0)
        q_normalized = (q - q_min) / q_span
        t_normalized = (t - t_min) / t_span
        palm_band = q_normalized[(t_normalized >= 0.62) & (t_normalized <= 0.82)]
        thumb_band = q_normalized[(t_normalized >= 0.42) & (t_normalized <= 0.68)]
        center = float(np.median(palm_band)) if palm_band.size else 0.5
        if thumb_band.size:
            left_extension = center - float(np.percentile(thumb_band, 1.0))
            right_extension = float(np.percentile(thumb_band, 99.0)) - center
            thumb_right = right_extension >= left_extension
        else:
            thumb_right = True
        x, y, box_width, box_height = cv2.boundingRect(component.astype(np.uint8))
        hand_bbox = (int(x), int(y), int(box_width), int(box_height))
        quality_flag = "ok"

    q_span = max(float(q_max - q_min), 1.0)
    t_span = max(float(t_max - t_min), 1.0)
    boxes: dict[str, Box] = {}
    for name in ROI_NAMES:
        cx, cy, relative_width, relative_height = _ROI_SPECS[name]
        if not thumb_right:
            cx = 1.0 - cx
        half_q = relative_width * q_span / 2.0
        half_t = relative_height * t_span / 2.0
        q_center = q_min + cx * q_span
        t_center = t_min + cy * t_span
        corners = np.array(
            [
                u * (q_center - half_q) + v * (t_center - half_t),
                u * (q_center + half_q) + v * (t_center - half_t),
                u * (q_center + half_q) + v * (t_center + half_t),
                u * (q_center - half_q) + v * (t_center + half_t),
            ]
        )
        boxes[name] = _clamp_box(corners, width, height)

    angle = float(np.degrees(np.arctan2(v[0], v[1])))
    return GeometryResult(
        boxes=boxes,
        quality_flag=quality_flag,
        angle_degrees=angle,
        thumb_side="right" if thumb_right else "left",
        hand_bbox=hand_bbox,
    )


def propose_six_rois(mask: np.ndarray, *, width: int, height: int) -> dict[str, Box]:
    return analyze_six_rois(mask, width=width, height=height).boxes

