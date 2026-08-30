from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from c4_multi_roi.colab_runner import checkpoint_compatible, select_resume_checkpoint
from c4_multi_roi.config import C4Config, scientific_config_hash


class ColabRunnerTests(unittest.TestCase):
    def test_selects_matching_mirror_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = C4Config(
                manifest=str(root / "manifest.csv"), run_id="RUN",
                output_root=str(root / "local"), checkpoint_mirror_root=str(root / "mirror"),
            )
            path = config.mirror_dir / "last.ckpt"
            path.parent.mkdir(parents=True)
            torch.save({
                "config_hash": scientific_config_hash(config),
                "manifest_hash": "m",
                "code_hash": "c",
            }, path)
            self.assertTrue(checkpoint_compatible(path, config, manifest_hash="m", code_hash="c"))
            self.assertEqual(select_resume_checkpoint(config, manifest_hash="m", code_hash="c"), path)

    def test_rejects_mismatched_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = C4Config(
                manifest=str(root / "manifest.csv"), run_id="RUN",
                output_root=str(root / "local"), checkpoint_mirror_root=str(root / "mirror"),
            )
            path = config.mirror_dir / "last.ckpt"
            path.parent.mkdir(parents=True)
            torch.save({"config_hash": "bad", "manifest_hash": "m", "code_hash": "c"}, path)
            self.assertIsNone(select_resume_checkpoint(config, manifest_hash="m", code_hash="c"))


if __name__ == "__main__":
    unittest.main()
