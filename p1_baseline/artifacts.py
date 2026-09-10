"""Deterministic, mild image artifacts used by the robustness pilot.

The transform changes acquisition appearance while keeping the hand anatomy and
the target label unchanged.  It deliberately acts on intensity, blur/noise and
an image-edge band; it does not erase or paste over the central hand.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def apply_mild_artifact(
    image: Image.Image,
    *,
    seed: int | None = None,
    severity: float = 1.0,
    rng: random.Random | None = None,
) -> Image.Image:
    """Return a deterministic mild acquisition-artifact view of a grayscale image."""
    if severity <= 0:
        return image.copy()
    if rng is None:
        rng = random.Random(seed)
    severity = min(float(severity), 1.0)
    result = image.convert("L").copy()

    # Scanner exposure/contrast drift.
    result = ImageEnhance.Brightness(result).enhance(
        1.0 + rng.uniform(-0.07, 0.07) * severity
    )
    result = ImageEnhance.Contrast(result).enhance(
        1.0 + rng.uniform(-0.14, 0.14) * severity
    )
    if rng.random() < 0.75:
        gamma = 1.0 + rng.uniform(-0.10, 0.10) * severity
        array = np.asarray(result, dtype=np.float32) / 255.0
        array = np.power(np.clip(array, 0.0, 1.0), gamma)
        result = Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L")

    # Slight focus loss and sensor noise, both kept small enough to preserve the
    # radiographic structures needed for age assessment.
    if rng.random() < 0.60:
        result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.75) * severity))
    array = np.asarray(result, dtype=np.float32)
    noise_std = rng.uniform(1.0, 5.0) * severity
    if noise_std > 0:
        # Seed NumPy from the local Python RNG so generation remains fully
        # reproducible without a slow Python loop over every pixel.
        noise_rng = np.random.default_rng(rng.getrandbits(64))
        noise = noise_rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        array = np.clip(array + noise, 0.0, 255.0).astype(np.uint8)
        result = Image.fromarray(array, mode="L")

    # A narrow edge band models clipping/marker/shadow at the scan boundary.
    # The band never covers the central crop where the hand anatomy is located.
    width, height = result.size
    band = max(1, int(round(min(width, height) * rng.uniform(0.008, 0.022) * severity)))
    draw = ImageDraw.Draw(result)
    value = int(rng.choice([0, 24, 220, 255]))
    side = rng.choice(("left", "right", "top", "bottom"))
    if side == "left":
        draw.rectangle((0, 0, band - 1, height - 1), fill=value)
    elif side == "right":
        draw.rectangle((width - band, 0, width - 1, height - 1), fill=value)
    elif side == "top":
        draw.rectangle((0, 0, width - 1, band - 1), fill=value)
    else:
        draw.rectangle((0, height - band, width - 1, height - 1), fill=value)
    return result
