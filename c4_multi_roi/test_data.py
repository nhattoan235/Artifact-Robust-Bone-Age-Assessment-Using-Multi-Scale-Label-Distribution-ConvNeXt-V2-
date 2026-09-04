from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from c4_multi_roi.data import MultiViewBoneAgeDataset, split_records
from c4_multi_roi.schema import ROI_NAMES


class MultiViewDatasetTests(unittest.TestCase):
    def build_record(self, root: Path, image_id: str = "1", fold: str = "1") -> dict[str, str]:
        image = Image.new("L", (20, 30), color=100)
        global_path = root / f"global_{image_id}.png"
        image.save(global_path)
        row = {
            "image_id": image_id,
            "fold": fold,
            "bone_age_months": "120",
            "sex": "M",
            "global_path": str(global_path),
        }
        for name in ROI_NAMES:
            path = root / f"{name}_{image_id}.png"
            image.save(path)
            row[f"roi_{name}_path"] = str(path)
            row[f"roi_{name}_quality"] = "ok"
        return row

    def test_dataset_returns_seven_views_in_locked_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = MultiViewBoneAgeDataset(
                [self.build_record(Path(directory))], image_root=".", image_size=32,
                target_mean=100, target_std=20, train=False,
            )
            sample = dataset[0]
            self.assertEqual(sample["views"].shape, (7, 3, 32, 32))
            self.assertEqual(tuple(sample["view_names"]), ("global",) + ROI_NAMES)
            self.assertEqual(float(sample["target_norm"]), 1.0)

    def test_shared_augmentation_keeps_identical_views_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = MultiViewBoneAgeDataset(
                [self.build_record(Path(directory))], image_root=".", image_size=32,
                target_mean=100, target_std=20, train=True, augmentation="light", epoch=3,
            )
            views = dataset[0]["views"]
            for index in range(1, 7):
                self.assertTrue(torch.equal(views[0], views[index]))

    def test_split_records_keeps_source_ids_disjoint(self) -> None:
        rows = [
            {"image_id": "1", "fold": "1"},
            {"image_id": "2", "fold": "2"},
            {"image_id": "3", "fold": "1"},
        ]
        train, validation = split_records(rows, validation_fold=1)
        self.assertEqual({row["image_id"] for row in validation}, {"1", "3"})
        self.assertEqual({row["image_id"] for row in train}, {"2"})


if __name__ == "__main__":
    unittest.main()
