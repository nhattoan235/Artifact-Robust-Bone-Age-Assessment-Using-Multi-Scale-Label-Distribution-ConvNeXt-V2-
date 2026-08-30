from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from c4_multi_roi.config import C4Config
from c4_multi_roi.schema import ROI_NAMES
from c4_multi_roi.trainer import C4Trainer, verified_copy


class TinySharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Linear(2, 1)

    def forward(self, views: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        image_value = views.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return self.regressor(torch.cat([image_value, sex], dim=1)).squeeze(1)


class C4TrainingTests(unittest.TestCase):
    def test_verified_copy_replaces_destination_without_rename_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "mirror" / "destination.bin"
            source.write_bytes(b"new-checkpoint")
            verified_copy(source, destination)
            self.assertEqual(destination.read_bytes(), b"new-checkpoint")

    def create_manifest(self, root: Path) -> Path:
        rows = []
        for index in range(4):
            image_id = str(index + 1)
            image = Image.new("L", (16, 20), color=50 + index * 20)
            global_path = root / f"global_{image_id}.png"
            image.save(global_path)
            row = {
                "image_id": image_id,
                "fold": "1" if index < 2 else "2",
                "bone_age_months": str(80 + index * 10),
                "sex": "M" if index % 2 else "F",
                "global_path": str(global_path),
            }
            for name in ROI_NAMES:
                path = root / f"{name}_{image_id}.png"
                image.save(path)
                row[f"roi_{name}_path"] = str(path)
                row[f"roi_{name}_quality"] = "ok"
            rows.append(row)
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_interrupt_checkpoint_and_resume_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.create_manifest(root)
            config = C4Config(
                manifest=str(manifest), image_root=".", validation_fold=1, expected_rows=4,
                run_id="SMOKE", output_root=str(root / "runs"), checkpoint_mirror_root=str(root / "mirror"),
                pretrained=False, image_size=16, epochs=2, batch_size=1, grad_accum_steps=1,
                patience=10, amp=False, num_workers=0, log_every_steps=1,
                checkpoint_every_steps=1, checkpoint_every_minutes=999,
            )
            trainer = C4Trainer(config, model=TinySharedModel())
            self.assertEqual(trainer.fit(interrupt_after_global_step=1), "interrupted")
            checkpoint = config.run_dir / "last.ckpt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((config.mirror_dir / "last.ckpt").is_file())
            trainer.close()

            resumed = C4Trainer(config, model=TinySharedModel())
            status = resumed.fit(resume=checkpoint)
            self.assertEqual(status, "completed")
            self.assertTrue((config.run_dir / "val_predictions_best.csv").is_file())
            self.assertEqual(resumed.epoch, 2)
            resumed.close()


if __name__ == "__main__":
    unittest.main()
