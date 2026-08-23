from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path

from p1_baseline.config import load_config
from d3_oof.aggregate import aggregate_oof
from d3_oof.registry import build_registry, verify_reused_splits


ROOT = Path(__file__).resolve().parents[1]


class D3OOFRegistryTests(unittest.TestCase):
    def test_reused_manifests_match_all_p7_validation_ids(self) -> None:
        audit = verify_reused_splits(ROOT)

        self.assertEqual(audit["pooled_unique_validation_ids"], 14036)
        self.assertEqual(
            [audit["folds"][str(fold)]["validation_count"] for fold in range(1, 6)],
            [2808, 2807, 2807, 2807, 2807],
        )
        self.assertTrue(all(item["same_validation_ids"] for item in audit["folds"].values()))

    def test_registry_builds_five_locked_d3_configs_without_test_reference(self) -> None:
        expected_stats = {
            1: (127.23833273957962, 41.248974358112605),
            2: (127.41472971769525, 41.24421735515304),
            3: (127.35292546086028, 41.29279781131869),
            4: (127.25817080772998, 41.18746900058828),
            5: (127.25621159497729, 41.20649499429804),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "registry"
            report = build_registry(ROOT, output_dir)

            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["test_used"])
            self.assertEqual(len(report["folds"]), 5)
            saved = json.loads((output_dir / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)

            for fold in range(1, 6):
                config_path = output_dir / f"fold_{fold}.toml"
                text = config_path.read_text(encoding="utf-8")
                self.assertNotIn("test", text.lower())
                cfg = load_config(config_path)
                self.assertEqual(cfg.run_id, f"D3_OOF_V1_FOLD_{fold}")
                self.assertEqual(cfg.architecture, "convnext_tiny_ldl")
                self.assertEqual(cfg.seed, 42)
                self.assertEqual(cfg.label_distribution_sigma, 2.0)
                self.assertEqual(cfg.label_distribution_weight, 0.2)
                self.assertEqual(cfg.regression_inference_weight, 0.5)
                self.assertEqual(cfg.batch_size * cfg.grad_accum_steps, 36)
                self.assertEqual(
                    (cfg.target_mean, cfg.target_std), expected_stats[fold]
                )
                fold_report = report["folds"][str(fold)]
                self.assertEqual(cfg.expected_train_hash, fold_report["train_hash"])
                self.assertEqual(cfg.expected_val_hash, fold_report["validation_hash"])


class D3OOFAggregationTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _fixture(self, root: Path) -> tuple[Path, Path]:
        run_root = root / "d3_runs"
        for fold in range(1, 6):
            e1_rows = []
            d3_rows = []
            for offset in range(2):
                image_id = (fold - 1) * 2 + offset + 1
                target = float(20 + image_id * 10)
                sex = "F" if image_id % 2 else "M"
                e1_prediction = target + (4.0 if image_id % 2 else -6.0)
                regression = target + (-2.0 if image_id % 2 else 3.0)
                distribution = target + (2.0 if image_id % 2 else -1.0)
                fused = 0.5 * regression + 0.5 * distribution
                e1_rows.append({
                    "image_id": image_id,
                    "target_months": target,
                    "prediction_months": e1_prediction,
                    "absolute_error": abs(e1_prediction - target),
                    "sex": sex,
                })
                d3_rows.append({
                    "image_id": image_id,
                    "target_months": target,
                    "prediction_months": fused,
                    "absolute_error": abs(fused - target),
                    "sex": sex,
                    "regression_months": regression,
                    "distribution_months": distribution,
                })
            self._write_csv(
                root / f"p7_results/results/P7_FINAL_V3_FOLD_{fold}/val_predictions_best.csv",
                e1_rows,
            )
            self._write_csv(
                run_root / f"D3_OOF_V1_FOLD_{fold}/val_predictions_best.csv", d3_rows
            )
        return root, run_root

    def test_aggregate_writes_integrity_checked_oof_and_fixed_ensembles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace, run_root = self._fixture(Path(temp_dir))
            output_dir = workspace / "outputs"
            report = aggregate_oof(workspace, run_root, output_dir, expected_count=10)

            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["test_used"])
            self.assertEqual(report["count"], 10)
            self.assertEqual(report["ensemble_weights"], {"E1": 0.5, "D3": 0.5})
            self.assertIn("E1_D3_FUSED_50_50", report["mae"])
            self.assertIn("paired_bootstrap_95_ci", report["comparisons"]["E1_D3_FUSED_50_50_vs_E1"])
            self.assertIn("prediction_correlation", report["comparisons"]["D3_FUSED_vs_E1"])
            self.assertIn("residual_correlation", report["comparisons"]["D3_FUSED_vs_E1"])
            self.assertTrue((output_dir / "D3_OOF_predictions.csv").is_file())
            self.assertTrue((output_dir / "E1_D3_ensemble_predictions.csv").is_file())
            self.assertTrue((output_dir / "D3_OOF_report.json").is_file())

    def test_aggregate_rejects_duplicate_ids_across_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace, run_root = self._fixture(Path(temp_dir))
            for base in (
                workspace / "p7_results/results/P7_FINAL_V3_FOLD_2",
                run_root / "D3_OOF_V1_FOLD_2",
            ):
                path = base / "val_predictions_best.csv"
                rows = self._read_csv(path)
                rows[0]["image_id"] = "1"
                self._write_csv(path, rows)
            with self.assertRaisesRegex(RuntimeError, "Duplicate OOF ID"):
                aggregate_oof(workspace, run_root, workspace / "outputs", expected_count=10)

    def test_aggregate_rejects_target_or_sex_mismatch(self) -> None:
        for field, bad_value in (("target_months", "999"), ("sex", "X")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                workspace, run_root = self._fixture(Path(temp_dir))
                path = run_root / "D3_OOF_V1_FOLD_3/val_predictions_best.csv"
                rows = self._read_csv(path)
                rows[0][field] = bad_value
                self._write_csv(path, rows)
                with self.assertRaisesRegex(RuntimeError, "Target/sex mismatch"):
                    aggregate_oof(workspace, run_root, workspace / "outputs", expected_count=10)

    def test_aggregate_requires_d3_branch_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace, run_root = self._fixture(Path(temp_dir))
            path = run_root / "D3_OOF_V1_FOLD_4/val_predictions_best.csv"
            rows = self._read_csv(path)
            for row in rows:
                row.pop("regression_months")
            self._write_csv(path, rows)
            with self.assertRaisesRegex(RuntimeError, "regression_months"):
                aggregate_oof(workspace, run_root, workspace / "outputs", expected_count=10)


if __name__ == "__main__":
    unittest.main()
