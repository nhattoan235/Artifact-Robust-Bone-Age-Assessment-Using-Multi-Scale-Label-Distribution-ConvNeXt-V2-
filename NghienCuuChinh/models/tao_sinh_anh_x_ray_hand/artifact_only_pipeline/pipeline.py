from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .detector import ArtifactDetection, detect_artifact_candidates
from .protection import build_protected_anatomy_mask
from .qc import compute_pixel_preservation_qc, make_review_overlay
from .reconstruct import reconstruct_artifacts


@dataclass(frozen=True)
class ProcessResult:
    case_id: str
    cleaned: np.ndarray
    protected_mask: np.ndarray
    artifact_mask: np.ndarray
    review_overlay: np.ndarray
    metrics: dict[str, object]


class ArtifactOnlyPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    @staticmethod
    def _read_optional_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
        if path is None or not path.exists():
            return None
        if path.suffix.lower() == ".npy":
            mask = np.load(path)
        else:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255

    def process(
        self,
        image_path: Path,
        seed_protection_path: Path | None = None,
        manual_artifact_path: Path | None = None,
    ) -> ProcessResult:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read image: {image_path}")

        case_id = image_path.stem
        seed = self._read_optional_mask(seed_protection_path, gray.shape)
        manual = self._read_optional_mask(manual_artifact_path, gray.shape)

        protected, protection_metrics = build_protected_anatomy_mask(
            gray, self.config, seed_mask=seed
        )
        manual_override_fraction = 0.0
        if manual is not None and np.any(manual):
            # A reviewed manual artifact mask is the authoritative edit
            # boundary. Older conservative protection seeds sometimes include
            # an adjacent marker plate or even most of the exposure field.
            # Unlock only explicitly painted pixels; everything outside the
            # manual mask remains protected by the usual hard invariants.
            manual_pixels = manual > 0
            manual_override_fraction = float(
                np.count_nonzero(manual_pixels & (protected > 0))
            ) / max(np.count_nonzero(manual_pixels), 1)
            protected = protected.copy()
            protected[manual_pixels] = 0

        detection: ArtifactDetection = detect_artifact_candidates(
            gray, protected, self.config, manual_mask=manual
        )
        cleaned = reconstruct_artifacts(
            gray,
            detection.mask,
            protected,
            self.config,
            seed_offset=int(case_id) if case_id.isdigit() else 0,
        )
        qc = compute_pixel_preservation_qc(gray, cleaned, detection.mask, protected)
        overlay = make_review_overlay(gray, cleaned, detection.mask, protected)

        reasons: list[str] = []
        if protection_metrics["protection_status"] != "ok":
            reasons.append(f"protection_{protection_metrics['protection_status']}")
        if detection.metrics["detector_review_required"]:
            reasons.append(str(detection.metrics["detector_review_reason"]))
        if not qc["pixel_preservation_pass"]:
            reasons.append("pixel_preservation_failed")
        if detection.metrics["artifact_area_pct"] == 0:
            reasons.append("no_artifact_detected")

        if manual is not None and qc["pixel_preservation_pass"]:
            status = "MANUAL_MASK_CLEANED"
        elif reasons:
            status = "AUTO_CLEANED_REVIEW_REQUIRED"
        else:
            status = "AUTO_CLEANED"

        metrics: dict[str, object] = {
            "Case_ID": case_id,
            "Status": status,
            "Review_Reason": "|".join(filter(None, reasons)),
            **protection_metrics,
            "manual_protection_override_fraction": manual_override_fraction,
            **detection.metrics,
            **qc,
            "width": gray.shape[1],
            "height": gray.shape[0],
        }
        return ProcessResult(
            case_id=case_id,
            cleaned=cleaned,
            protected_mask=protected,
            artifact_mask=detection.mask,
            review_overlay=overlay,
            metrics=metrics,
        )
