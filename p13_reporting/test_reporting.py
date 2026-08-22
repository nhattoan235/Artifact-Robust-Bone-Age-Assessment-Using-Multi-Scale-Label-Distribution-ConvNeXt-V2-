from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .build_thesis_assets import (
    bootstrap_mean_ci,
    collect_hashable_outputs,
    model_metrics,
    paired_forest_data,
    validate_alignment,
)


def frame(predictions: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [1, 2, 3, 4],
            "target_months": [10.0, 20.0, 30.0, 40.0],
            "prediction_months": predictions,
            "sex": ["F", "F", "M", "M"],
        }
    )


class ReportingTests(unittest.TestCase):
    def test_manifest_is_excluded_from_output_hashes(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "figure.png").write_bytes(b"figure")
            (output_dir / "report_manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                [path.name for path in collect_hashable_outputs(output_dir)],
                ["figure.png"],
            )
    def test_model_metrics(self) -> None:
        metrics = model_metrics(frame([9.0, 22.0, 27.0, 44.0]))
        self.assertAlmostEqual(metrics["mae_months"], 2.5)
        self.assertAlmostEqual(metrics["female_mae_months"], 1.5)
        self.assertAlmostEqual(metrics["male_mae_months"], 3.5)

    def test_validate_alignment_rejects_target_drift(self) -> None:
        frames = {
            "E1 sex embedding": frame([10, 20, 30, 40]),
            "E0 image-only": frame([10, 20, 30, 40]),
            "E2 dual-output": frame([10, 20, 30, 40]),
        }
        frames["E2 dual-output"].loc[0, "target_months"] = 11
        with self.assertRaisesRegex(ValueError, "Target không khớp"):
            validate_alignment(frames)

    def test_bootstrap_mean_ci_is_deterministic(self) -> None:
        values = np.arange(20, dtype=float)
        first = bootstrap_mean_ci(values, 500, 2026)
        second = bootstrap_mean_ci(values, 500, 2026)
        self.assertEqual(first, second)
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())

    def test_paired_forest_has_locked_six_rows(self) -> None:
        frames = {
            "E1 sex embedding": frame([10, 20, 30, 40]),
            "E0 image-only": frame([11, 22, 33, 44]),
            "E2 dual-output": frame([9, 19, 29, 39]),
        }
        result = paired_forest_data(frames, 200, 2026)
        self.assertEqual(len(result), 6)
        self.assertEqual(set(result["group"]), {"Overall", "Nữ", "Nam"})
        e0 = result[(result["candidate"] == "E0 image-only") & (result["group"] == "Overall")]
        self.assertGreater(float(e0.iloc[0]["delta_mae_months"]), 0)


if __name__ == "__main__":
    unittest.main()
