from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

from c3_roi import build_pilot_bc_bundle as builder


ROOT = Path(__file__).resolve().parents[1]


class PilotBCBundleTests(unittest.TestCase):
    def test_pilot_configs_are_direct_regression_and_only_c_has_consistency(self):
        configs = {}
        for pilot in ("B", "C"):
            path = ROOT / "c3_roi" / "pilot_configs" / f"C3_Z26_C3_ROI_V2_PILOT_{pilot}_FOLD_1_SEED_42.toml"
            configs[pilot] = tomllib.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(configs[pilot]["model"]["architecture"], "convnext_tiny")
            self.assertEqual(configs[pilot]["run"]["seed"], 42)
            self.assertEqual(configs[pilot]["model"]["artifact_warmup_epochs"], 3)
            self.assertEqual(configs[pilot]["model"]["artifact_ramp_epochs"], 5)
        self.assertEqual(configs["B"]["model"]["consistency_weight"], 0.0)
        self.assertEqual(configs["C"]["model"]["consistency_weight"], 0.30)

    def test_c_fold2_to_fold5_configs_match_locked_splits_and_scientific_settings(self):
        expected = {
            2: (
                11229, 2807,
                "f4270f3983badf5bd63bfa52f8e6918e4a3886dba51c6b0550d8c38837d4212e",
                "26124f6279aa93aa291ca297eb2156d72f0328e566db3df228ca3a1d354fef74",
            ),
            3: (
                11229, 2807,
                "896d4fd4efb43a0726d57a3084ba597364d9aedb73551774f34b58fec213be9e",
                "c894202bb22f453ffef0eff9dcbd5a3e2f72c5b090073951388b7ed19a6ab82b",
            ),
            4: (
                11229, 2807,
                "f82483c670fc62a8e478466e5df8d49473843b757716245b9577a085421add74",
                "60440af4a7bb01c4d4b3a7cba35916e6f1c9d674b2933ef279636107b107fe9f",
            ),
            5: (
                11229, 2807,
                "413153e8bf865c1c866397e24520b0914d7aba2da9e8397fcde9d207ae07adbf",
                "23724152ada9b697a16143ce265a7a27276a03879f15b83aba7050b79226ea41",
            ),
        }
        fold1 = __import__("tomllib").loads(
            (ROOT / "c3_roi" / "pilot_configs" /
             "C3_Z26_C3_ROI_V2_PILOT_C_FOLD_1_SEED_42.toml")
            .read_text(encoding="utf-8")
        )
        for fold, (train_count, val_count, train_hash, val_hash) in expected.items():
            path = ROOT / "c3_roi" / "pilot_configs" / (
                f"C3_Z26_C3_ROI_V2_PILOT_C_FOLD_{fold}_SEED_42.toml"
            )
            self.assertTrue(path.is_file(), path)
            config = __import__("tomllib").loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["data"]["expected_train_count"], train_count)
            self.assertEqual(config["data"]["expected_val_count"], val_count)
            self.assertEqual(config["data"]["expected_train_hash"], train_hash)
            self.assertEqual(config["data"]["expected_val_hash"], val_hash)
            self.assertEqual(config["run"]["run_id"],
                             f"C3_Z26_C3_ROI_V2_PILOT_C_V2_FOLD_{fold}_SEED_42")
            self.assertEqual(config["model"]["consistency_weight"], 0.30)
            self.assertEqual(config["model"]["artifact_augmentation"], "mild_v1")
            self.assertEqual(config["model"]["artifact_warmup_epochs"], 3)
            self.assertEqual(config["model"]["artifact_ramp_epochs"], 5)
            self.assertEqual(config["model"]["architecture"], fold1["model"]["architecture"])
            self.assertEqual(config["training"], fold1["training"])

    def test_v2_bundle_bootstraps_colab_and_contains_resume_and_evaluation(self):
        self.assertEqual(builder.DEFAULT_OUTPUT.name, "C3_Z26_C3_ROI_V2_PILOT_BC_CODE_V2.zip")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / builder.DEFAULT_OUTPUT.name
            report = builder.build_bundle(output)
            self.assertEqual(report["files"], 17)
            self.assertTrue(Path(report["notebook"]).is_file())
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertIn("C3_Z26_C3_ROI_V2_PILOTS/pilot_bc_runner.py", names)
                self.assertIn("C3_Z26_C3_ROI_V2_PILOTS/evaluate_pilot_bc.py", names)
                notebook = json.loads(archive.read("C3_Z26_C3_ROI_V2_PILOTS/C3_Z26_C3_ROI_V2_PILOT_BC.ipynb"))
                source = "".join(
                    line for cell in notebook["cells"] for line in cell.get("source", [])
                )
                self.assertIn("C3_Z26_C3_ROI_T4_CODE_V2.zip", source)
                self.assertIn("C3_Z26_COMBO_V2_FINAL.zip", source)
                self.assertIn("pip", source)
                self.assertIn("pilot_bc_runner.py", source)
                self.assertIn("evaluate_pilot_bc.py", source)
                self.assertNotIn("rmtree('/content/C3_Z26_C3_ROI_V2_PILOTS'", source)


if __name__ == "__main__":
    unittest.main()
