from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact_only_pipeline.config import PipelineConfig
from artifact_only_pipeline.detector import detect_artifact_candidates
from artifact_only_pipeline.pipeline import ArtifactOnlyPipeline
from artifact_only_pipeline.qc import compute_pixel_preservation_qc
from artifact_only_pipeline.reconstruct import reconstruct_artifacts


class ArtifactOnlyTests(unittest.TestCase):
    def test_reconstruction_changes_only_artifact_mask(self) -> None:
        h, w = 180, 140
        yy, xx = np.mgrid[0:h, 0:w]
        original = np.clip(35 + 0.08 * xx + 0.04 * yy, 0, 255).astype(np.uint8)
        original[20:42, 8:34] = 245

        protected = np.zeros_like(original)
        cv2.ellipse(protected, (w // 2, h // 2), (35, 70), 0, 0, 360, 255, -1)
        artifact = np.zeros_like(original)
        artifact[15:48, 3:40] = 255

        cleaned = reconstruct_artifacts(
            original, artifact, protected, PipelineConfig(), seed_offset=1
        )
        qc = compute_pixel_preservation_qc(original, cleaned, artifact, protected)
        self.assertTrue(qc["pixel_preservation_pass"])
        self.assertEqual(qc["changed_inside_protected"], 0)
        self.assertEqual(qc["changed_outside_artifact"], 0)
        self.assertGreater(qc["changed_pixels"], 0)

    def test_detector_never_edits_protected_anatomy(self) -> None:
        image = np.full((240, 180), 40, np.uint8)
        protected = np.zeros_like(image)
        cv2.ellipse(protected, (90, 130), (45, 95), 0, 0, 360, 255, -1)
        image[35:55, 8:28] = 250
        image[90:112, 82:102] = 250

        detection = detect_artifact_candidates(image, protected, PipelineConfig())
        overlap = np.count_nonzero((detection.mask > 0) & (protected > 0))
        self.assertEqual(overlap, 0)

    def test_reviewed_manual_mask_overrides_only_its_protected_pixels(self) -> None:
        image = np.full((180, 140), 42, np.uint8)
        image[25:55, 8:42] = 235
        seed = np.full_like(image, 255)
        manual = np.zeros_like(image)
        manual[20:60, 4:46] = 255

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "100.png"
            seed_path = root / "100_seed.png"
            manual_path = root / "100_manual.png"
            cv2.imwrite(str(image_path), image)
            cv2.imwrite(str(seed_path), seed)
            cv2.imwrite(str(manual_path), manual)

            result = ArtifactOnlyPipeline().process(
                image_path,
                seed_protection_path=seed_path,
                manual_artifact_path=manual_path,
            )

        self.assertEqual(
            np.count_nonzero((result.artifact_mask > 0) & (manual == 0)), 0
        )
        self.assertEqual(
            np.count_nonzero((result.protected_mask > 0) & (manual > 0)), 0
        )
        self.assertEqual(
            np.count_nonzero((result.cleaned != image) & (manual == 0)), 0
        )
        self.assertGreater(np.count_nonzero(result.cleaned != image), 0)


if __name__ == "__main__":
    unittest.main()
