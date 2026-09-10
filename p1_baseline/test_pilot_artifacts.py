import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from p1_baseline.artifacts import apply_mild_artifact
from p1_baseline.data import BoneAgeDataset


class PilotArtifactTests(unittest.TestCase):
    def test_mild_artifact_is_deterministic_and_preserves_shape(self):
        source = Image.fromarray(np.full((32, 40), 128, dtype=np.uint8), mode="L")
        first = apply_mild_artifact(source, seed=123, severity=1.0)
        second = apply_mild_artifact(source, seed=123, severity=1.0)
        self.assertEqual(first.size, source.size)
        self.assertEqual(first.mode, "L")
        self.assertEqual(np.array_equal(np.asarray(first), np.asarray(second)), True)
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(source)))

    def test_artifact_zero_severity_is_identity(self):
        source = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8), mode="L")
        result = apply_mild_artifact(source, seed=123, severity=0.0)
        self.assertTrue(np.array_equal(np.asarray(result), np.asarray(source)))

    def test_dataset_returns_paired_artifact_view(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.fromarray(np.full((20, 16), 128, dtype=np.uint8), mode="L").save(path)
            rows = [{
                "split": "train", "image_id": "1", "bone_age_months": "120",
                "sex": "F", "image_path": path.name, "sha256": "x", "readable": "1",
            }]
            dataset = BoneAgeDataset(
                rows, image_size=32, target_mean=100.0, target_std=20.0,
                train=True, seed=42, augmentation="none", image_root=directory,
                artifact_augmentation="mild_v1", artifact_probability=1.0,
            )
            item = dataset[0]
            self.assertIn("artifact_image", item)
            self.assertTrue(item["artifact_applied"].item())
            self.assertEqual(tuple(item["image"].shape), (3, 32, 32))
            self.assertEqual(tuple(item["artifact_image"].shape), (3, 32, 32))
            self.assertFalse(np.array_equal(item["image"].numpy(), item["artifact_image"].numpy()))


if __name__ == "__main__":
    unittest.main()
