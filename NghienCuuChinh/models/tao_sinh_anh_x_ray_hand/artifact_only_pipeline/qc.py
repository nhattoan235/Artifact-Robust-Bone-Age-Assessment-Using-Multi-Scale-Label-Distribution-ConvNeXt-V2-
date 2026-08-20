from __future__ import annotations

import cv2
import numpy as np


def compute_pixel_preservation_qc(
    original: np.ndarray,
    cleaned: np.ndarray,
    artifact_mask: np.ndarray,
    protected_mask: np.ndarray,
) -> dict[str, float | int | bool]:
    if original.shape != cleaned.shape:
        raise ValueError("Original and cleaned images must have identical dimensions")

    diff = np.abs(original.astype(np.int16) - cleaned.astype(np.int16))
    changed = diff > 0
    protected = protected_mask > 0
    artifact = artifact_mask > 0
    outside_artifact = ~artifact

    changed_inside_protected = int(np.count_nonzero(changed & protected))
    changed_outside_artifact = int(np.count_nonzero(changed & outside_artifact))
    changed_total = int(np.count_nonzero(changed))

    return {
        "changed_pixels": changed_total,
        "changed_pct": changed_total * 100.0 / original.size,
        "changed_inside_protected": changed_inside_protected,
        "changed_outside_artifact": changed_outside_artifact,
        "max_diff_inside_protected": int(diff[protected].max()) if np.any(protected) else 0,
        "max_diff_outside_artifact": int(diff[outside_artifact].max())
        if np.any(outside_artifact)
        else 0,
        "pixel_preservation_pass": bool(
            changed_inside_protected == 0 and changed_outside_artifact == 0
        ),
    }


def make_review_overlay(
    original: np.ndarray,
    cleaned: np.ndarray,
    artifact_mask: np.ndarray,
    protected_mask: np.ndarray,
) -> np.ndarray:
    """Four-panel visual QC: original, masks, cleaned, absolute difference."""

    h, w = original.shape
    orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    masks = orig_bgr.copy()
    green = np.zeros_like(masks)
    green[:, :, 1] = 255
    red = np.zeros_like(masks)
    red[:, :, 2] = 255
    p = protected_mask > 0
    a = artifact_mask > 0
    masks[p] = (0.65 * masks[p] + 0.35 * green[p]).astype(np.uint8)
    masks[a] = (0.35 * masks[a] + 0.65 * red[a]).astype(np.uint8)

    diff = cv2.absdiff(original, cleaned)
    diff_vis = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_TURBO)

    panels = [orig_bgr, masks, cleaned_bgr, diff_vis]
    names = ["ORIGINAL", "GREEN=PROTECTED / RED=EDIT", "CLEANED", "ABS DIFF"]
    for panel, name in zip(panels, names):
        cv2.rectangle(panel, (0, 0), (min(w, 520), 42), (0, 0, 0), -1)
        cv2.putText(panel, name, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.hstack(panels)

