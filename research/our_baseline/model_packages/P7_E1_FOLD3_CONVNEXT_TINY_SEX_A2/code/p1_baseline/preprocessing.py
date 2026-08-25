from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FullHandSettings:
    """Cau hinh khoa cho tien xu ly toan bo ban tay P3-B1."""

    version: str = "full_hand_v1"
    threshold_factors: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00)
    minimum_area_fraction: float = 0.055
    maximum_area_fraction: float = 0.65
    minimum_bbox_width_fraction: float = 0.24
    minimum_bbox_height_fraction: float = 0.38
    dilation_fraction: float = 0.012
    crop_margin_fraction: float = 0.065
    feather_fraction: float = 0.004
    maximum_alignment_degrees: float = 12.0


@dataclass
class FullHandQC:
    success: bool
    fallback: bool
    reason: str
    otsu_threshold: float
    threshold_factor: float
    component_area_fraction: float
    bbox_width_fraction: float
    bbox_height_fraction: float
    touches_top: bool
    touches_bottom: bool
    touches_left: bool
    touches_right: bool
    alignment_degrees: float
    applied_alignment_degrees: float
    output_width: int
    output_height: int

    def to_dict(self) -> dict:
        return asdict(self)


def _odd(value: float, minimum: int = 3) -> int:
    result = max(minimum, int(round(value)))
    return result if result % 2 else result + 1


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def _component_candidates(binary: np.ndarray) -> list[tuple[np.ndarray, dict]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    image_area = float(height * width)
    candidates: list[tuple[np.ndarray, dict]] = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        area_fraction = area / image_area
        width_fraction = w / width
        height_fraction = h / height
        cx, cy = centroids[label]
        center_distance = ((cx / width - 0.5) / 0.5) ** 2 + ((cy / height - 0.52) / 0.52) ** 2
        center_overlap = labels[height * 2 // 5:height * 4 // 5, width * 3 // 10:width * 7 // 10]
        overlaps_center = bool(np.any(center_overlap == label))
        border_count = int(x == 0) + int(y == 0) + int(x + w >= width) + int(y + h >= height)
        fill_fraction = area / float(w * h)
        # Dien tich va chieu cao duoc uu tien; marker thuong nho/le bien. Thanh phan
        # phu ca ba/bon bien bi phat nang vi thuong la nen bi threshold nham.
        score = (
            1.2 * min(area_fraction, 0.45)
            + 1.4 * min(height_fraction, 0.95)
            + 0.7 * min(width_fraction, 0.90)
            + (1.0 if overlaps_center else -1.5)
            - 0.45 * center_distance
            - 3.0 * abs(area_fraction - 0.30)
            - 4.0 * max(0.0, fill_fraction - 0.58)
            - (2.5 if border_count >= 3 else 0.0)
        )
        candidates.append((labels == label, {
            "x": x, "y": y, "w": w, "h": h, "area": area,
            "area_fraction": area_fraction, "width_fraction": width_fraction,
            "height_fraction": height_fraction, "overlaps_center": overlaps_center,
            "border_count": border_count, "fill_fraction": fill_fraction, "score": score,
        }))
    return candidates


def _select_hand_mask(gray: np.ndarray, settings: FullHandSettings) -> tuple[np.ndarray | None, dict]:
    original_height, original_width = gray.shape
    # Connected components tren anh X-quang 2K-4K rat ton CPU/RAM. Tim silhouette
    # tren ban thu nho, sau do dua mask nhi phan ve dung kich thuoc goc.
    scale = min(1.0, 768.0 / max(original_height, original_width))
    if scale < 1.0:
        width = max(1, int(round(original_width * scale)))
        height = max(1, int(round(original_height * scale)))
        working = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    else:
        working = gray
        height, width = gray.shape
    low, high = np.percentile(working, [0.5, 99.5])
    if high <= low + 1:
        return None, {"reason": "near_constant_image", "otsu": 0.0, "factor": 0.0}
    normalized = np.clip((working.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    blur_kernel = _odd(min(height, width) * 0.009)
    blurred = cv2.GaussianBlur(normalized, (blur_kernel, blur_kernel), 0)
    otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morphology_kernel = _odd(min(height, width) * 0.012)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel))

    best: tuple[np.ndarray, dict] | None = None
    for factor in settings.threshold_factors:
        threshold = int(round(float(otsu) * factor))
        binary = np.where(blurred >= threshold, 255, 0).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        for component, metadata in _component_candidates(binary):
            metadata = dict(metadata, factor=factor, threshold=threshold, otsu=float(otsu))
            plausible = (
                settings.minimum_area_fraction <= metadata["area_fraction"] <= settings.maximum_area_fraction
                and metadata["width_fraction"] >= settings.minimum_bbox_width_fraction
                and metadata["height_fraction"] >= settings.minimum_bbox_height_fraction
                and metadata["overlaps_center"]
                and metadata["border_count"] < 3
                and metadata["fill_fraction"] <= 0.70
            )
            if not plausible:
                continue
            if best is None or metadata["score"] > best[1]["score"]:
                best = (component.astype(np.uint8) * 255, metadata)
    if best is None:
        return None, {"reason": "no_plausible_component", "otsu": float(otsu), "factor": 0.0}
    mask, metadata = best
    mask = _fill_holes(mask)
    if scale < 1.0:
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return mask, metadata


def _principal_alignment(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return 0.0
    points = np.column_stack((xs, ys)).astype(np.float64)
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    vx, vy = vectors[:, int(np.argmax(values))]
    if vy < 0:
        vx, vy = -vx, -vy
    return float(np.degrees(np.arctan2(vx, vy)))


def _rotate_expand(array: np.ndarray, degrees: float, interpolation: int) -> np.ndarray:
    height, width = array.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(np.ceil(height * sine + width * cosine))
    new_height = int(np.ceil(height * cosine + width * sine))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(array, matrix, (new_width, new_height), flags=interpolation, borderValue=0)


def _fallback(gray: np.ndarray, reason: str, otsu: float = 0.0) -> tuple[Image.Image, FullHandQC, np.ndarray]:
    height, width = gray.shape
    qc = FullHandQC(
        success=False, fallback=True, reason=reason, otsu_threshold=otsu, threshold_factor=0.0,
        component_area_fraction=0.0, bbox_width_fraction=0.0, bbox_height_fraction=0.0,
        touches_top=False, touches_bottom=False, touches_left=False, touches_right=False,
        alignment_degrees=0.0, applied_alignment_degrees=0.0,
        output_width=width, output_height=height,
    )
    return Image.fromarray(gray, mode="L"), qc, np.zeros_like(gray, dtype=np.uint8)


def full_hand_preprocess(
    image: Image.Image, settings: FullHandSettings | None = None,
) -> tuple[Image.Image, FullHandQC, np.ndarray]:
    """Mask + can giua/can chinh bao thu; tra anh goc neu QC khong chac chan."""
    settings = settings or FullHandSettings()
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    mask, metadata = _select_hand_mask(gray, settings)
    if mask is None:
        return _fallback(gray, metadata["reason"], metadata.get("otsu", 0.0))

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return _fallback(gray, "empty_component", metadata.get("otsu", 0.0))
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    touches = (y0 == 0, y1 == height - 1, x0 == 0, x1 == width - 1)
    raw_angle = _principal_alignment(mask)
    applied_angle = -raw_angle if abs(raw_angle) <= settings.maximum_alignment_degrees else 0.0

    dilation = _odd(np.hypot(height, width) * settings.dilation_fraction)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    safe_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    feather = _odd(np.hypot(height, width) * settings.feather_fraction)
    alpha = cv2.GaussianBlur(safe_mask, (feather, feather), 0).astype(np.float32) / 255.0
    neutralized = np.rint(gray.astype(np.float32) * alpha).astype(np.uint8)

    if applied_angle:
        neutralized = _rotate_expand(neutralized, applied_angle, cv2.INTER_CUBIC)
        safe_mask = _rotate_expand(safe_mask, applied_angle, cv2.INTER_NEAREST)

    ys, xs = np.nonzero(safe_mask)
    if len(xs) == 0:
        return _fallback(gray, "empty_after_alignment", metadata.get("otsu", 0.0))
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    box_width, box_height = x1 - x0 + 1, y1 - y0 + 1
    margin = int(round(max(box_width, box_height) * settings.crop_margin_fraction))
    x0, x1 = max(0, x0 - margin), min(neutralized.shape[1] - 1, x1 + margin)
    y0, y1 = max(0, y0 - margin), min(neutralized.shape[0] - 1, y1 + margin)
    cropped = neutralized[y0:y1 + 1, x0:x1 + 1]
    cropped_mask = safe_mask[y0:y1 + 1, x0:x1 + 1]

    qc = FullHandQC(
        success=True, fallback=False, reason="ok",
        otsu_threshold=float(metadata["otsu"]), threshold_factor=float(metadata["factor"]),
        component_area_fraction=float(metadata["area_fraction"]),
        bbox_width_fraction=float(metadata["width_fraction"]),
        bbox_height_fraction=float(metadata["height_fraction"]),
        touches_top=touches[0], touches_bottom=touches[1],
        touches_left=touches[2], touches_right=touches[3],
        alignment_degrees=raw_angle, applied_alignment_degrees=applied_angle,
        output_width=int(cropped.shape[1]), output_height=int(cropped.shape[0]),
    )
    return Image.fromarray(cropped, mode="L"), qc, cropped_mask
