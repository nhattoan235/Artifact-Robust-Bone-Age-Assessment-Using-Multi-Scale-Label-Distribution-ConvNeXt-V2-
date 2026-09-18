from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from c3_roi.build_pilot_bc_followup_bundle import build_bundle


class BuildPilotBCFollowupBundleTests(unittest.TestCase):
    def test_bundle_contains_sequential_diagnosis_and_control(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "followup.zip"
            result = build_bundle(output)
            self.assertGreater(result["files"], 10)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertIn(
                    "C3_Z26_C3_ROI_V2_PILOTS/evaluate_artifact_matrix.py", names,
                )
                self.assertIn(
                    "C3_Z26_C3_ROI_V2_PILOTS/compare_pilot_b_c_fold.py", names,
                )
                self.assertIn(
                    "C3_Z26_C3_ROI_V2_PILOTS/configs/"
                    "C3_Z26_C3_ROI_V2_PILOT_B_FOLD_5_SEED_42.toml", names,
                )
                notebook = json.loads(archive.read(
                    "C3_Z26_C3_ROI_V2_PILOTS/"
                    "C3_Z26_C3_ROI_V2_FOLLOWUP_FOLD5.ipynb"
                ))
                source = "".join(
                    line for cell in notebook["cells"] for line in cell.get("source", [])
                )
                self.assertLess(source.index("evaluate_artifact_matrix.py"),
                                source.index("--pilot', 'B'"))
                self.assertLess(source.index("--pilot', 'B'"),
                                source.index("compare_pilot_b_c_fold.py"))


if __name__ == "__main__":
    unittest.main()
