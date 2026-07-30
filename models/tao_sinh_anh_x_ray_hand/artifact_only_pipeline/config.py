from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Resolution-normalized parameters for artifact-only cleaning."""

    max_working_side: int = 1400

    # The anatomy mask is intentionally conservative. It is not used for
    # compositing; it only defines pixels that must never be edited.
    protection_dilate_ratio: float = 0.006
    protection_min_area_pct: float = 8.0
    protection_max_area_pct: float = 82.0

    # Local-contrast detector parameters.
    contrast_kernel_ratio: float = 0.035
    contrast_percentile: float = 98.6
    min_contrast: float = 10.0
    strict_min_contrast: float = 18.0
    strict_min_intensity: float = 160.0
    protection_guard_ratio: float = 0.010
    group_kernel_ratio: float = 0.012
    min_stroke_area_pct: float = 0.00008
    max_stroke_area_pct: float = 0.50
    min_group_area_pct: float = 0.002
    max_group_area_pct: float = 4.5
    max_group_aspect: float = 11.0
    # Text is frequently printed on a slightly darker marker plate. The box
    # therefore extends beyond the bright glyphs to remove the full plate.
    artifact_box_padding_ratio: float = 0.035

    # Candidate artifacts are expected away from the anatomy, usually near
    # the film boundary. Central candidates require manual confirmation.
    central_safe_margin_ratio: float = 0.12
    max_auto_edit_pct: float = 8.0
    max_boxes_before_review: int = 12

    # Local background reconstruction.
    ring_width_ratio: float = 0.025
    # Feather only the innermost 2–3 pixels of a reviewed mask. A wide
    # distance ramp merely fades radiopaque letters/plates and leaves a
    # recognizable ghost instead of removing the artifact.
    blend_inside_ratio: float = 0.004
    # With inward-biased ring sampling, a local plane follows the exposure
    # gradient without extrapolating the black canvas across the edit region.
    polynomial_degree: int = 1
    max_fit_samples: int = 30_000
    random_seed: int = 20260728
