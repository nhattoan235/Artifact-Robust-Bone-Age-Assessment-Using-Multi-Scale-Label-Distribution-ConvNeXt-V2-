from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from c3_roi import build_pilot_c_5fold_bundle as builder


class PilotC5FoldBundleTests(unittest.TestCase):
    def test_bundle_contains_all_fold_configs_runner_evaluator_and_aggregator(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot_c_5fold.zip"
            report = builder.build_bundle(output)
            self.assertTrue(Path(report["notebook"]).is_file())
            self.assertGreaterEqual(report["files"], 20)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                for fold in range(1, 6):
                    self.assertIn(
                        f"C3_Z26_C3_ROI_V2_PILOTS/configs/C3_Z26_C3_ROI_V2_PILOT_C_FOLD_{fold}_SEED_42.toml",
                        names,
                    )
                self.assertIn(
                    "C3_Z26_C3_ROI_V2_PILOTS/evaluate_pilot_c_5fold.py", names
                )
                notebook = json.loads(
                    archive.read(
                        "C3_Z26_C3_ROI_V2_PILOTS/C3_Z26_C3_ROI_V2_PILOT_C_5FOLD.ipynb"
                    )
                )
                source = "".join(
                    line for cell in notebook["cells"] for line in cell.get("source", [])
                )
                self.assertIn("FOLD = 2", source)
                self.assertIn("--fold", source)
                self.assertIn("evaluate_pilot_c_5fold.py", source)
                self.assertIn("C3_Z26_COMBO_V2_FINAL.zip", source)


if __name__ == "__main__":
    unittest.main()
