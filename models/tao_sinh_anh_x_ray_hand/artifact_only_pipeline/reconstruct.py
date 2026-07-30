from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig


def _design_matrix(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    if degree <= 0:
        return np.ones((len(x), 1), dtype=np.float64)
    if degree == 1:
        return np.column_stack((np.ones_like(x), x, y))
    return np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))


def _fit_local_surface(
    gray: np.ndarray,
    edit_mask: np.ndarray,
    blocked_mask: np.ndarray,
    config: PipelineConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = gray.shape
    short = min(h, w)
    ring_px = max(8, int(round(short * config.ring_width_ratio)))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1)
    )
    ring = cv2.dilate(edit_mask, ring_kernel) > 0
    ring &= edit_mask == 0
    # Protected pixels may still be sampled (read-only). Excluding the large
    # conservative hull can leave only the black side of a nearby marker and
    # incorrectly reconstruct a black rectangle. The dominant-mode filter
    # below rejects bones/soft tissue when film background is available.

    ys, xs = np.where(ring)
    if len(xs) < 30:
        # Telea is a safe fallback for small edge-adjacent regions.
        return cv2.inpaint(gray, edit_mask, inpaintRadius=max(3, ring_px // 3), flags=cv2.INPAINT_TELEA)

    # Prefer samples on the side of the artifact that faces the image centre.
    # Edge labels otherwise contribute a large amount of black canvas to the
    # ring and produce a dark rectangular patch on the illuminated film.
    mask_ys, mask_xs = np.where(edit_mask > 0)
    tangent_axis: tuple[float, float] | None = None
    mask_cx = w / 2.0
    mask_cy = h / 2.0
    if len(mask_xs):
        mask_cx = float(np.mean(mask_xs))
        mask_cy = float(np.mean(mask_ys))
        inward_x = w / 2.0 - mask_cx
        inward_y = h / 2.0 - mask_cy
        inward = (
            (xs - mask_cx) * inward_x + (ys - mask_cy) * inward_y
        ) >= 0
        if np.count_nonzero(inward) >= 60:
            xs, ys = xs[inward], ys[inward]
            norm = float(np.hypot(inward_x, inward_y))
            if norm > 0.08 * min(h, w):
                # Model variation along the exposure edge, not away from it.
                # For a left/right marker this is the vertical axis; for a
                # top/bottom marker it is horizontal. This avoids unstable
                # extrapolation into the masked outer side.
                tangent_axis = (-inward_y / norm, inward_x / norm)

    # When a label lies close to the exposure edge, the ring can contain both
    # film background and black canvas. Fit the dominant local intensity mode
    # so the black side does not pull the reconstructed patch into a dark box.
    ring_values = gray[ys, xs]
    hist = np.bincount(ring_values, minlength=256).astype(np.float32)
    hist = np.convolve(hist, np.ones(11, dtype=np.float32), mode="same")
    dominant = int(np.argmax(hist))
    # Marker plates are commonly darker than the surrounding film and can
    # leak a narrow rim outside a hand-drawn mask. Prefer the brighter half of
    # the local mode so that this rim does not define the replacement level.
    background_floor = float(np.percentile(ring_values, 55.0))
    same_surface = (
        (np.abs(ring_values.astype(np.int16) - dominant) <= 36)
        & (ring_values >= background_floor)
    )
    if np.count_nonzero(same_surface) >= 60:
        xs, ys = xs[same_surface], ys[same_surface]

    if len(xs) > config.max_fit_samples:
        keep = rng.choice(len(xs), config.max_fit_samples, replace=False)
        xs, ys = xs[keep], ys[keep]

    z = gray[ys, xs].astype(np.float64)
    if tangent_axis is not None and config.polynomial_degree > 0:
        tangent_x, tangent_y = tangent_axis
        t = (
            (xs - mask_cx) * tangent_x + (ys - mask_cy) * tangent_y
        ) / max(float(min(h, w)), 1.0)
        if config.polynomial_degree == 1:
            A = np.column_stack((np.ones_like(t), t))
        else:
            A = np.column_stack((np.ones_like(t), t, t * t))
    else:
        x_norm = (xs - w / 2.0) / max(w / 2.0, 1.0)
        y_norm = (ys - h / 2.0) / max(h / 2.0, 1.0)
        A = _design_matrix(x_norm, y_norm, config.polynomial_degree)

    # Two robust trimming rounds reduce the influence of unmasked text strokes.
    keep = np.ones(len(z), dtype=bool)
    coeff = np.linalg.lstsq(A, z, rcond=None)[0]
    for _ in range(2):
        residual = z - A @ coeff
        med = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - med)) + 1e-6
        keep = np.abs(residual - med) < 3.5 * 1.4826 * mad
        if np.count_nonzero(keep) < 20:
            break
        coeff = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]

    yy, xx = np.mgrid[0:h, 0:w]
    if tangent_axis is not None and config.polynomial_degree > 0:
        tangent_x, tangent_y = tangent_axis
        t_all = (
            (xx.ravel() - mask_cx) * tangent_x
            + (yy.ravel() - mask_cy) * tangent_y
        ) / max(float(min(h, w)), 1.0)
        if config.polynomial_degree == 1:
            prediction_matrix = np.column_stack((np.ones_like(t_all), t_all))
        else:
            prediction_matrix = np.column_stack(
                (np.ones_like(t_all), t_all, t_all * t_all)
            )
    else:
        xx_n = (xx.ravel() - w / 2.0) / max(w / 2.0, 1.0)
        yy_n = (yy.ravel() - h / 2.0) / max(h / 2.0, 1.0)
        prediction_matrix = _design_matrix(
            xx_n, yy_n, config.polynomial_degree
        )
    trend = (prediction_matrix @ coeff).reshape(h, w)
    # The edit region lies outside the inward sampling half-ring, so a fitted
    # plane is necessarily extrapolated there. Bound that extrapolation to
    # the robust local background range to prevent dark/bright rectangular
    # patches at exposure edges.
    fit_values = z[keep] if np.count_nonzero(keep) else z
    trend_lo, trend_hi = np.percentile(fit_values, (10.0, 90.0))
    trend = np.clip(trend, trend_lo, trend_hi)

    residual = z[keep] - A[keep] @ coeff if np.count_nonzero(keep) else z - A @ coeff
    noise_std = float(np.clip(np.std(residual), 0.4, 12.0))
    noise = rng.normal(0.0, noise_std, size=(h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    return np.clip(trend + noise, 0, 255).astype(np.uint8)


def reconstruct_artifacts(
    gray: np.ndarray,
    artifact_mask: np.ndarray,
    protected_mask: np.ndarray,
    config: PipelineConfig,
    seed_offset: int = 0,
) -> np.ndarray:
    """Replace only artifact-mask pixels and preserve every other pixel exactly."""

    edit = (artifact_mask > 0).astype(np.uint8) * 255
    edit[protected_mask > 0] = 0
    if not np.any(edit):
        return gray.copy()

    rng = np.random.default_rng(config.random_seed + int(seed_offset))
    synth = _fit_local_surface(gray, edit, protected_mask, config, rng)

    short = min(gray.shape)
    blend_px = max(2, int(round(short * config.blend_inside_ratio)))
    distance = cv2.distanceTransform(edit, cv2.DIST_L2, 5)
    # Match the synthesized patch to the immediate unchanged boundary. The
    # surface model follows the local trend, while this robust offset removes
    # a visible dark/bright silhouette of an irregular hand-drawn mask.
    seam_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blend_px * 2 + 1, blend_px * 2 + 1)
    )
    outer_ring = (cv2.dilate(edit, seam_kernel) > 0) & (edit == 0)
    inner_ring = (edit > 0) & (distance <= max(2, blend_px * 1.5))
    if np.count_nonzero(outer_ring) >= 20 and np.count_nonzero(inner_ring) >= 20:
        level_shift = float(np.median(gray[outer_ring])) - float(
            np.median(synth[inner_ring])
        )
        synth = np.clip(
            synth.astype(np.float32) + level_shift, 0, 255
        ).astype(np.uint8)

    alpha = np.clip(distance / max(blend_px, 1), 0.0, 1.0).astype(np.float32)

    result = gray.copy()
    blended = (
        gray.astype(np.float32) * (1.0 - alpha)
        + synth.astype(np.float32) * alpha
    )
    inside = edit > 0
    result[inside] = np.clip(blended[inside], 0, 255).astype(np.uint8)

    # Hard guarantees. These assignments make the safety invariant explicit.
    result[protected_mask > 0] = gray[protected_mask > 0]
    result[edit == 0] = gray[edit == 0]
    return result
