from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from c4_multi_roi.config import load_config
from c4_multi_roi.prepare_configs import write_colab_configs


class PrepareConfigsTests(unittest.TestCase):
    def test_writes_five_locked_colab_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = write_colab_configs(output, manifest_sha256="a" * 64)
            self.assertEqual(len(paths), 5)
            for fold, path in enumerate(paths, start=1):
                config = load_config(path)
                self.assertEqual(config.validation_fold, fold)
                self.assertEqual(config.run_id, f"C4_MULTI_ROI_V1_FOLD_{fold}")
                self.assertEqual(config.expected_manifest_sha256, "a" * 64)
                self.assertEqual(config.checkpoint_mirror_root, "/content/drive/MyDrive/data/c4_multi_roi_runs")


if __name__ == "__main__":
    unittest.main()
