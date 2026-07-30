from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _normalize_u8(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, (1.0, 99.5))
    if hi <= lo + 1:
        return gray.copy()
    return np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _largest_central_components(binary: np.ndarray) -> np.ndarray:
    """Keep foreground components plausibly belonging to the hand.

    Components are selected using centrality, vertical span, and proximity to
    the largest central component. The result may over-segment; that is safe
    because it is only used as a no-edit region.
    """

    h, w = binary.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros_like(binary)

    img_area = h * w
    candidates: list[tuple[float, int]] = []
    for idx in range(1, n):
        x, y, cw, ch, area = stats[idx]
        cx, cy = centroids[idx]
        if area < img_area * 0.0002:
            continue
        centrality = 1.0 - min(abs(cx / w - 0.5) / 0.5, 1.0)
        height_score = min(ch / max(h * 0.65, 1), 1.0)
        lower_score = 1.0 if y + ch > h * 0.58 else 0.0
        area_score = min(area / max(img_area * 0.18, 1), 1.0)
        score = 2.5 * centrality + 2.0 * height_score + lower_score + area_score
        candidates.append((score, idx))

    if not candidates:
        return np.zeros_like(binary)

    candidates.sort(reverse=True)
    main_idx = candidates[0][1]
    x, y, cw, ch, _ = stats[main_idx]
    pad_x, pad_y = int(w * 0.22), int(h * 0.16)
    x0, x1 = max(0, x - pad_x), min(w, x + cw + pad_x)
    y0, y1 = max(0, y - pad_y), min(h, y + ch + pad_y)

    selected = np.zeros_like(binary)
    for _, idx in candidates:
        cx, cy = centroids[idx]
        _, _, _, comp_h, area = stats[idx]
        if (x0 <= cx <= x1 and y0 <= cy <= y1) or (
            area > img_area * 0.006 and comp_h > h * 0.12 and abs(cx / w - 0.5) < 0.38
        ):
            selected[labels == idx] = 255
    return selected


def _largest_component_and_fill_holes(binary: np.ndarray) -> np.ndarray:
    """Keep the hand component and fill only enclosed holes, not finger gaps."""

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros_like(binary)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hand = (labels == idx).astype(np.uint8) * 255

    padded = cv2.copyMakeBorder(hand, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(hand, holes)


def build_protected_anatomy_mask(
    gray: np.ndarray,
    config: PipelineConfig,
    seed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Build an intentionally oversized, solid no-edit mask.

    A supplied seed mask (for example, a manually approved or previous hand
    mask) is unioned with independent threshold candidates. Convex hull is
    deliberate here: inter-finger background is safer to protect than edit.
    """

    if gray.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image")

    h, w = gray.shape
    short = min(h, w)
    norm = _normalize_u8(gray)
    blur = cv2.GaussianBlur(norm, (_odd(short * 0.009),) * 2, 0)
    otsu_t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if seed_mask is not None and np.any(seed_mask):
        if seed_mask.shape != gray.shape:
            seed_mask = cv2.resize(
                seed_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
        selected = (seed_mask > 0).astype(np.uint8) * 255
        # Preserve the true silhouette. A convex hull would incorrectly lock
        # large inter-finger/background regions and can protect a nearby label.
        close_px = _odd(short * 0.008)
        selected = cv2.morphologyEx(
            selected,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)),
            iterations=2,
        )
        selected = _largest_component_and_fill_holes(selected)
        source = "approved_seed"
    else:
        # Absolute thresholds frequently select the complete film/cassette.
        # A positive local residual instead emphasizes bones and soft-tissue
        # boundaries. Restricting seed evidence to the broad central corridor
        # prevents edge labels and cassette borders from defining the hull.
        background_k = _odd(short * 0.12)
        background = cv2.GaussianBlur(norm, (background_k, background_k), 0)
        positive = cv2.subtract(norm, background)

        corridor = np.zeros_like(gray, dtype=bool)
        corridor[:, int(w * 0.10) : int(w * 0.90)] = True
        values = positive[corridor]
        residual_t = max(5.0, float(np.percentile(values, 83.0)))
        evidence = (
            (positive >= residual_t)
            & (norm >= max(18, int(otsu_t * 0.45)))
            & corridor
        ).astype(np.uint8) * 255

        # Join neighbouring bone/soft-tissue evidence before selecting the
        # central hand-like component.
        k_join = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(short * 0.018), _odd(short * 0.028))
        )
        evidence = cv2.dilate(evidence, k_join, iterations=2)
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(short * 0.014),) * 2
        )
        evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, k_close, iterations=2)
        selected = _largest_central_components(evidence)
        source = "local_residual"

    points = cv2.findNonZero(selected)
    if points is None or len(points) < 20:
        # Last-resort no-edit region. It prefers leaving a label unchanged over
        # modifying anatomy when automatic foreground localization fails.
        protected = np.zeros_like(gray)
        cv2.ellipse(
            protected,
            (w // 2, int(h * 0.52)),
            (int(w * 0.36), int(h * 0.49)),
            0,
            0,
            360,
            255,
            -1,
        )
        source = "central_fallback"
    elif source == "approved_seed":
        protected = selected.copy()
    else:
        hull = cv2.convexHull(points)
        protected = np.zeros_like(gray)
        cv2.fillConvexPoly(protected, hull, 255)

    dilate_px = max(5, int(round(short * config.protection_dilate_ratio)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(2 * dilate_px + 1),) * 2
    )
    protected = cv2.dilate(protected, kernel, iterations=1)

    area_pct = float(np.count_nonzero(protected)) * 100.0 / protected.size
    status = "ok"
    if area_pct < config.protection_min_area_pct:
        status = "too_small"
    elif area_pct > config.protection_max_area_pct:
        status = "too_large"

    return protected, {
        "protection_area_pct": area_pct,
        "protection_source": source,
        "protection_status": status,
        "otsu_threshold": float(otsu_t),
    }
