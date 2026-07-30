from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig


@dataclass(frozen=True)
class ArtifactDetection:
    mask: np.ndarray
    boxes: tuple[tuple[int, int, int, int], ...]
    rejected_boxes: tuple[tuple[int, int, int, int], ...]
    metrics: dict[str, float | int | bool | str]


def _odd(value: int | float) -> int:
    value = max(3, int(round(value)))
    return value if value % 2 else value + 1


def _normalize_u8(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, (1.0, 99.5))
    if hi <= lo + 1:
        return gray.copy()
    return np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _estimate_exposure_bbox(norm: np.ndarray) -> tuple[int, int, int, int]:
    """Estimate the illuminated film rectangle when it sits on black canvas."""

    h, w = norm.shape
    # A modest normalized threshold follows the true film edge instead of
    # including its low-intensity outer glow/shadow.
    foreground = (norm > 24).astype(np.uint8) * 255
    close_px = max(5, _odd(min(h, w) * 0.012))
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)),
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if n <= 1:
        return (0, 0, w, h)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = map(int, stats[idx])
    if area < 0.18 * h * w:
        return (0, 0, w, h)
    return (x, y, bw, bh)


def detect_artifact_candidates(
    gray: np.ndarray,
    protected_mask: np.ndarray,
    config: PipelineConfig,
    manual_mask: np.ndarray | None = None,
) -> ArtifactDetection:
    """Detect text/marker groups outside the protected anatomy.

    The detector groups high local-contrast strokes, then edits a padded box
    around the group so both printed characters and their marker plate are
    removed. Long cassette edges are rejected.
    """

    h, w = gray.shape
    short = min(h, w)
    norm = _normalize_u8(gray)
    exposure_x, exposure_y, exposure_w, exposure_h = _estimate_exposure_bbox(norm)

    local_k = _odd(short * config.contrast_kernel_ratio)
    local_mean = cv2.GaussianBlur(norm, (local_k, local_k), 0)
    # Markers in this cohort are predominantly radiopaque white strokes or
    # white text on a dark plate. Positive residuals avoid treating ordinary
    # black/bright film boundaries as text.
    contrast = cv2.subtract(norm, local_mean)
    outside = protected_mask == 0
    outside_values = contrast[outside]
    outside_intensity = norm[outside]

    if outside_values.size < 100:
        threshold = 255.0
    else:
        threshold = max(
            config.min_contrast,
            float(np.percentile(outside_values, config.contrast_percentile)),
        )

    bright_threshold = (
        max(105.0, float(np.percentile(outside_intensity, 90.0)))
        if outside_intensity.size
        else 255.0
    )
    # Work from individual high-confidence bright components first. Grouping
    # the whole residual image directly lets chains of soft-tissue edges join
    # a real marker and causes the complete group to be rejected.
    guard_px = max(2, int(round(short * config.protection_guard_ratio)))
    guard_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(guard_px * 2 + 1), _odd(guard_px * 2 + 1))
    )
    anatomy_guard = cv2.dilate(
        (protected_mask > 0).astype(np.uint8) * 255, guard_kernel
    )
    strict_contrast = max(threshold, config.strict_min_contrast)
    strict_intensity = max(bright_threshold, config.strict_min_intensity)
    raw_strokes = (
        (contrast >= strict_contrast)
        & (norm >= strict_intensity)
        & (anatomy_guard == 0)
    ).astype(np.uint8) * 255

    # Retain character-sized components. This removes isolated detector noise,
    # cassette borders, and broad anatomical residuals before character
    # grouping. Thin characters must not be opened/eroded.
    ns, stroke_labels, stroke_stats, _ = cv2.connectedComponentsWithStats(
        raw_strokes, 8
    )
    img_area = h * w
    min_stroke_area = max(
        5, int(round(img_area * config.min_stroke_area_pct / 100.0))
    )
    max_stroke_area = int(round(img_area * config.max_stroke_area_pct / 100.0))
    strokes = np.zeros_like(gray)
    rejected_strokes = 0
    for idx in range(1, ns):
        x, y, bw, bh, area = map(int, stroke_stats[idx])
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        close_to_edge = (
            x < 0.025 * w
            or y < 0.025 * h
            or x + bw > 0.975 * w
            or y + bh > 0.975 * h
        )
        tiny_edge_object = (
            close_to_edge
            and max(bw, bh) < 0.030 * short
            and area < 0.0004 * img_area
        )
        if (
            area < min_stroke_area
            or area > max_stroke_area
            or aspect > 8.0
            or tiny_edge_object
        ):
            rejected_strokes += 1
            continue
        strokes[stroke_labels == idx] = 255

    group_px = max(3, int(round(short * config.group_kernel_ratio)))
    group_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (_odd(group_px * 2.2), _odd(group_px * 1.4))
    )
    grouped = cv2.dilate(strokes, group_kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    min_area = img_area * config.min_group_area_pct / 100.0
    max_area = img_area * config.max_group_area_pct / 100.0
    pad = max(3, int(round(short * config.artifact_box_padding_ratio)))

    accepted: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    candidate_mask = np.zeros_like(gray)

    for idx in range(1, n):
        x, y, bw, bh, area = map(int, stats[idx])
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        box = (x, y, bw, bh)
        if area < min_area or area > max_area or aspect > config.max_group_aspect:
            rejected.append(box)
            continue

        # Cassette/exposure boundaries can fragment into character-sized
        # pieces. Reject groups centred outside the exposure or glued to a
        # boundary. Actual labels are normally inset from that boundary.
        cx, cy = x + bw / 2.0, y + bh / 2.0
        # Film-edge glow is often 50–80 px wide in the original RSNA PNGs.
        boundary_band = 0.060 * short
        inside_exposure = (
            exposure_x <= cx <= exposure_x + exposure_w
            and exposure_y <= cy <= exposure_y + exposure_h
        )
        near_vertical = min(
            abs(cx - exposure_x), abs(cx - (exposure_x + exposure_w))
        ) < boundary_band
        near_horizontal = min(
            abs(cy - exposure_y), abs(cy - (exposure_y + exposure_h))
        ) < boundary_band
        boundary_fragment = (
            not inside_exposure
            or (near_vertical and near_horizontal)
            or (
                (near_vertical or near_horizontal)
                and (aspect > 1.8 or max(bw, bh) < 0.060 * short)
            )
        )
        outer_x = min(cx, w - cx) < 0.10 * short
        outer_y = min(cy, h - cy) < 0.10 * short
        tiny_outer_object = (
            (outer_x or outer_y)
            and max(bw, bh) < 0.060 * short
        )
        horizontal_edge_fragment = (
            bw > 2.5 * bh
            and (cy < 0.15 * h or cy > 0.85 * h)
        )
        vertical_edge_fragment = (
            bh > 2.5 * bw
            and (cx < 0.15 * w or cx > 0.85 * w)
        )
        if (
            boundary_fragment
            or tiny_outer_object
            or horizontal_edge_fragment
            or vertical_edge_fragment
        ):
            rejected.append(box)
            continue

        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        if (x1 - x0) * (y1 - y0) > 0.06 * img_area:
            # A label group this large is almost always an exposure/cassette
            # transition. Large regions are never auto-edited.
            rejected.append(box)
            continue
        touches_left = x0 == 0
        touches_right = x1 == w
        touches_top = y0 == 0
        touches_bottom = y1 == h
        if (touches_left or touches_right) and (touches_top or touches_bottom):
            # Film corners generate strong local contrast but are not labels.
            rejected.append(box)
            continue
        box_mask = np.zeros_like(gray)
        box_mask[y0:y1, x0:x1] = 255

        overlap_fraction = float(
            np.count_nonzero((box_mask > 0) & (protected_mask > 0))
        ) / max(np.count_nonzero(box_mask), 1)
        # A small box overlap is clipped away safely. A large overlap usually
        # indicates a bone edge, not an external label.
        if overlap_fraction > 0.35:
            rejected.append(box)
            continue

        # Require actual contrast strokes in the expanded box.
        stroke_count = int(np.count_nonzero(strokes[y0:y1, x0:x1]))
        if stroke_count < max(4, int(short * 0.004)):
            rejected.append(box)
            continue

        accepted.append((x0, y0, x1 - x0, y1 - y0))
        candidate_mask[y0:y1, x0:x1] = 255
        candidate_mask[protected_mask > 0] = 0

    mode = "auto"
    if manual_mask is not None:
        if manual_mask.shape != gray.shape:
            manual_mask = cv2.resize(
                manual_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
        candidate_mask = (manual_mask > 0).astype(np.uint8) * 255
        candidate_mask[protected_mask > 0] = 0
        accepted = []
        mode = "manual"

    edit_pct = float(np.count_nonzero(candidate_mask)) * 100.0 / img_area
    central_margin_x = int(w * config.central_safe_margin_ratio)
    central_margin_y = int(h * config.central_safe_margin_ratio)
    central = np.zeros_like(gray)
    central[
        central_margin_y : h - central_margin_y,
        central_margin_x : w - central_margin_x,
    ] = 255
    central_edit_pct = (
        float(np.count_nonzero((candidate_mask > 0) & (central > 0)))
        * 100.0
        / max(np.count_nonzero(candidate_mask), 1)
    )

    review_reason: list[str] = []
    if edit_pct > config.max_auto_edit_pct:
        review_reason.append("edit_area_too_large")
    if len(accepted) > config.max_boxes_before_review:
        review_reason.append("too_many_boxes")
    if central_edit_pct > 35.0 and mode == "auto":
        review_reason.append("central_candidate")
    if len(rejected) > 0:
        review_reason.append("rejected_candidates")

    return ArtifactDetection(
        mask=candidate_mask,
        boxes=tuple(accepted),
        rejected_boxes=tuple(rejected),
        metrics={
            "detector_mode": mode,
            "contrast_threshold": threshold,
            "bright_threshold": bright_threshold,
            "strict_contrast_threshold": strict_contrast,
            "strict_intensity_threshold": strict_intensity,
            "retained_stroke_components": ns - 1 - rejected_strokes,
            "rejected_stroke_components": rejected_strokes,
            "exposure_x": exposure_x,
            "exposure_y": exposure_y,
            "exposure_width": exposure_w,
            "exposure_height": exposure_h,
            "artifact_area_pct": edit_pct,
            "accepted_boxes": len(accepted),
            "rejected_boxes": len(rejected),
            "central_edit_pct": central_edit_pct,
            "detector_review_required": bool(review_reason),
            "detector_review_reason": "|".join(review_reason),
        },
    )
