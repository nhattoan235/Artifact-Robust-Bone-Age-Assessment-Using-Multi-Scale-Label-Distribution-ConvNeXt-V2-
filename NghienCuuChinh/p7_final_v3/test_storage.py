from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

import torch

from p1_baseline.trainer import CHECKPOINT_KEYS
from .storage import MountedDriveStore, SnapshotLimitReached, sha256_file


def payload(step: int) -> dict:
    value = {key: None for key in CHECKPOINT_KEYS}
    value.update({"global_step": step, "epoch": 1, "batch_in_epoch": 2, "best_mae": 6.1})
    return value


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "P7_STORAGE_V3.marker").write_text("P7 V3", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def store(self, maximum: int = 3) -> MountedDriveStore:
        return MountedDriveStore(self.root, "P7_FINAL_V3_FOLD_1", maximum)

    def checkpoint(self, step: int) -> Path:
        path = self.root / f"local_{step}.ckpt"
        torch.save(payload(step), path)
        return path

    def test_append_restore_and_fallback(self):
        store = self.store()
        first = store.save_snapshot(self.checkpoint(2))
        second = store.save_snapshot(self.checkpoint(4))
        self.assertEqual(len(store.snapshots()), 2)
        second.checkpoint.write_bytes(b"corrupt")
        restored = store.restore_latest(self.root / "restored.ckpt")
        self.assertEqual(torch.load(restored, weights_only=False)["global_step"], 2)
        self.assertEqual(first.sha256, sha256_file(first.checkpoint))

    def test_same_step_not_duplicated(self):
        store = self.store()
        source = self.checkpoint(2)
        self.assertIsNotNone(store.save_snapshot(source))
        self.assertIsNone(store.save_snapshot(source, force=True))
        self.assertEqual(len(store.snapshots()), 1)

    def test_limit_never_deletes(self):
        store = self.store(2)
        store.save_snapshot(self.checkpoint(2))
        store.save_snapshot(self.checkpoint(4))
        with self.assertRaises(SnapshotLimitReached):
            store.save_snapshot(self.checkpoint(6))
        self.assertEqual(len(store.snapshots()), 2)

    def test_best_model_survives_missing_local_best(self):
        store = self.store()
        source = self.checkpoint(8)
        value = torch.load(source, weights_only=False)
        value.update({"best_epoch": 2, "best_mae": 5.9, "model": {"weight": torch.tensor([1.0])},
                      "config_hash": "c", "code_version": "v", "train_manifest_hash": "t", "val_manifest_hash": "x"})
        torch.save(value, source)
        saved = store.save_best_model(source)
        source.unlink()
        restored = torch.load(store.verified_latest_best(), weights_only=False)
        self.assertEqual(restored["best_mae"], 5.9)
        self.assertEqual(saved.epoch, 2)

    def test_finalize_uses_local_best_when_best_store_is_full(self):
        store = MountedDriveStore(self.root, "P7_FINAL_V3_FOLD_1", 3, max_best_models=1)
        run_dir = self.root / "local_run"
        run_dir.mkdir()
        old = self.checkpoint(8)
        old_value = torch.load(old, weights_only=False)
        old_value.update({"best_epoch": 1, "best_mae": 6.2, "model": {"weight": torch.tensor([1.0])},
                          "config_hash": "c", "code_version": "v", "train_manifest_hash": "t", "val_manifest_hash": "x"})
        torch.save(old_value, old)
        store.save_best_model(old)
        new_value = dict(old_value)
        new_value.update({"best_epoch": 2, "best_mae": 5.8, "global_step": 12, "model": {"weight": torch.tensor([2.0])}})
        torch.save(new_value, run_dir / "best_mae.ckpt")
        (run_dir / "run_state.json").write_text(
            '{"status":"early_stopped","best_mae":5.8}', encoding="utf-8"
        )
        (run_dir / "val_predictions_best.csv").write_text(
            "image_id,target_months,prediction_months,absolute_error,sex\n1,10,11,1,M\n",
            encoding="utf-8",
        )
        result = store.finalize(run_dir)
        final_model = torch.load(store.result_root / "best_model.pt", weights_only=False)
        self.assertEqual(result["best_mae"], 5.8)
        self.assertEqual(final_model["best_mae"], 5.8)
        self.assertEqual(float(final_model["model"]["weight"][0]), 2.0)

    def test_finalize_recovers_prediction_from_persistent_status(self):
        store = self.store()
        run_dir = self.root / "resumed_run"
        run_dir.mkdir()
        value = payload(12)
        value.update({"best_epoch": 2, "best_mae": 5.8, "model": {"weight": torch.tensor([2.0])},
                      "config_hash": "c", "code_version": "v", "train_manifest_hash": "t", "val_manifest_hash": "x"})
        torch.save(value, run_dir / "best_mae.ckpt")
        (run_dir / "run_state.json").write_text(
            '{"status":"early_stopped","best_mae":5.8}', encoding="utf-8"
        )
        status = store.run_root / "status"
        status.mkdir(parents=True)
        prediction = "image_id,target_months,prediction_months,absolute_error,sex\n1,10,11,1,M\n"
        (status / "val_predictions_best.csv").write_text(prediction, encoding="utf-8")
        store.finalize(run_dir)
        self.assertEqual((store.result_root / "val_predictions_best.csv").read_text(encoding="utf-8"), prediction)

    def test_preflight_requires_colab_mount(self):
        with self.assertRaises(RuntimeError):
            self.store().preflight()
        store = self.store()
        with patch.object(Path, "resolve", return_value=Path("/content/drive/MyDrive/p7")):
            # Chỉ kiểm tra guard đường dẫn; I/O probe vẫn ở temporary thực.
            store.root = self.root
        self.assertTrue((self.root / "P7_STORAGE_V3.marker").is_file())


if __name__ == "__main__":
    unittest.main()
