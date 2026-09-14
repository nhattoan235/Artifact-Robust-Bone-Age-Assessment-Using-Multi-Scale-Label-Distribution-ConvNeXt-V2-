from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from c3_roi.pilot_bc_runner import (
    pilot_config_path,
    select_resume_checkpoint,
    terminal_run_status,
)
from p1_baseline.config import Config, scientific_config_hash


class PilotBCRunnerTests(unittest.TestCase):
    def test_pilot_config_path_accepts_fold_and_keeps_b_locked_to_folds1_and5(self):
        root = Path("/tmp/pilots")
        self.assertEqual(
            pilot_config_path("b", root).name,
            "C3_Z26_C3_ROI_V2_PILOT_B_FOLD_1_SEED_42.toml",
        )
        self.assertEqual(
            pilot_config_path("C", root, 3).name,
            "C3_Z26_C3_ROI_V2_PILOT_C_FOLD_3_SEED_42.toml",
        )
        self.assertEqual(
            pilot_config_path("B", root, 5).name,
            "C3_Z26_C3_ROI_V2_PILOT_B_FOLD_5_SEED_42.toml",
        )
        with self.assertRaises(ValueError):
            pilot_config_path("B", root, 2)
        with self.assertRaises(ValueError):
            pilot_config_path("C", root, 6)
        with self.assertRaises(ValueError):
            pilot_config_path("A", root)

    def test_terminal_status_is_read_from_persistent_run_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run_state.json").write_text(
                json.dumps({"status": "early_stopped"}), encoding="utf-8"
            )
            self.assertEqual(terminal_run_status(run_dir), "early_stopped")

    def test_instability_status_never_silently_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run_state.json").write_text(
                json.dumps({"status": "stopped_instability"}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                terminal_run_status(run_dir)

    def test_resume_ignores_corrupt_last_and_uses_latest_compatible_periodic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = replace(
                Config(), run_id="pilot", output_root=str(root / "local"),
                checkpoint_mirror_root=str(root / "mirror"),
            )
            run_dir = root / "mirror" / "pilot"
            periodic = run_dir / "periodic"
            periodic.mkdir(parents=True)
            (run_dir / "last.ckpt").write_bytes(b"not a checkpoint")

            common = {
                "config_hash": scientific_config_hash(cfg),
                "train_manifest_hash": "train",
                "val_manifest_hash": "val",
                "code_version": "code",
            }
            older = periodic / "epoch_002.ckpt"
            newer = periodic / "epoch_004.ckpt"
            torch.save({**common, "epoch": 2, "samples_seen_in_epoch": 0, "global_step": 20}, older)
            torch.save({**common, "epoch": 4, "samples_seen_in_epoch": 3, "global_step": 40}, newer)

            selected = select_resume_checkpoint(
                cfg, train_hash="train", val_hash="val", code_version="code"
            )
            self.assertEqual(selected, newer)


if __name__ == "__main__":
    unittest.main()
