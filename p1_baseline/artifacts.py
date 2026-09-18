"""Deterministic, mild image artifacts used by the robustness pilot.

The transform changes acquisition appearance while keeping the hand anatomy and
the target label unchanged.  It deliberately acts on intensity, blur/noise and
an image-edge band; it does not erase or paste over the central hand.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


ARTIFACT_PROFILES = (
    "clean",
    "exposure_dark",
    "exposure_bright",
    "contrast_low",
    "contrast_high",
    "gamma_dark",
    "gamma_bright",
    "blur",
    "noise",
    "edge_band",
    "composite_mild",
)


def apply_artifact_profile(
    image: Image.Image,
    profile: str,
    *,
    seed: int = 0,
    severity: float = 1.0,
) -> Image.Image:
    """Apply one deterministic diagnostic artifact without changing anatomy.

    ``severity=1`` is the strongest value used by ``mild_v1`` for that single
    component. Values in ``[0, 1]`` therefore stay inside the training range.
    """
    if profile not in ARTIFACT_PROFILES:
        raise ValueError(f"artifact profile khong hop le: {profile}")
    severity = float(severity)
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity phai nam trong [0, 1]")
    result = image.convert("L").copy()
    if profile == "clean" or severity == 0.0:
        return result
    if profile == "composite_mild":
        # Match the locked evaluator: consume its probability draw before
        # applying mild_v1 to the same local RNG stream.
        rng = random.Random(seed)
        rng.random()
        return apply_mild_artifact(result, rng=rng, severity=severity)
    if profile == "exposure_dark":
        return ImageEnhance.Brightness(result).enhance(1.0 - 0.07 * severity)
    if profile == "exposure_bright":
        return ImageEnhance.Brightness(result).enhance(1.0 + 0.07 * severity)
    if profile == "contrast_low":
        return ImageEnhance.Contrast(result).enhance(1.0 - 0.14 * severity)
    if profile == "contrast_high":
        return ImageEnhance.Contrast(result).enhance(1.0 + 0.14 * severity)
    if profile in {"gamma_dark", "gamma_bright"}:
        gamma = 1.0 + (0.10 if profile == "gamma_dark" else -0.10) * severity
        array = np.asarray(result, dtype=np.float32) / 255.0
        array = np.power(np.clip(array, 0.0, 1.0), gamma)
        return Image.fromarray(
            np.round(array * 255.0).astype(np.uint8), mode="L"
        )
    if profile == "blur":
        return result.filter(ImageFilter.GaussianBlur(radius=0.75 * severity))
    if profile == "noise":
        array = np.asarray(result, dtype=np.float32)
        noise_rng = np.random.default_rng(seed)
        noise = noise_rng.normal(
            0.0, 5.0 * severity, size=array.shape,
        ).astype(np.float32)
        return Image.fromarray(
            np.clip(array + noise, 0.0, 255.0).astype(np.uint8), mode="L"
        )
    # A worst-range edge band with deterministic side/value.
    rng = random.Random(seed)
    width, height = result.size
    band = max(1, int(round(min(width, height) * 0.022 * severity)))
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
