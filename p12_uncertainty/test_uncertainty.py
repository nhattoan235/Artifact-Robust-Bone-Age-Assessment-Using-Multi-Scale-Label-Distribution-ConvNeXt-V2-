from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from .analyze_oof_uncertainty import (
    VIEW_COLUMNS,
    benjamini_hochberg,
    bootstrap_rank_correlation_ci,
    risk_coverage_metrics,
    validate_input,
)


def synthetic_frame(count: int = 40) -> pd.DataFrame:
    rows = []
    offsets = np.asarray([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    for index in range(count):
        scale = 0.1 + index / count
        target = float((index * 6) % 229)
        base = target + index * 0.1
        views = base + offsets * scale
        row = {
            "image_id": index + 1,
            "fold": index % 5 + 1,
            "target_months": target,
            "sex": "F" if index % 2 == 0 else "M",
            "raw_prediction_months": base + 0.2,
            "tta_prediction_months": float(views.mean()),
            "tta_std_months": float(views.std(ddof=0)),
        }
        row.update({column: float(value) for column, value in zip(VIEW_COLUMNS, views)})
        rows.append(row)
    return pd.DataFrame(rows)


class UncertaintyAnalysisTests(unittest.TestCase):
    def test_validate_input_accepts_recomputed_tta(self) -> None:
        report = validate_input(synthetic_frame(), expected_rows=40)
        self.assertEqual(report["row_count"], 40)
        self.assertLessEqual(report["max_tta_mean_recompute_diff"], 1e-8)

    def test_validate_input_rejects_duplicate_id(self) -> None:
        frame = synthetic_frame()
        frame.loc[1, "image_id"] = frame.loc[0, "image_id"]
        with self.assertRaisesRegex(ValueError, "image_id không duy nhất"):
            validate_input(frame, expected_rows=40)

    def test_validate_input_rejects_tta_mean_drift(self) -> None:
        frame = synthetic_frame()
        frame.loc[0, "tta_prediction_months"] += 0.01
        with self.assertRaisesRegex(ValueError, "TTA mean không khớp"):
            validate_input(frame, expected_rows=40)

    def test_bootstrap_rank_correlation_is_deterministic_and_positive(self) -> None:
        x = np.arange(1, 101, dtype=float)
        y = x + np.sin(x) * 0.01
        first = bootstrap_rank_correlation_ci(x, y, 500, 2026)
        second = bootstrap_rank_correlation_ci(x, y, 500, 2026)
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0.99)

    def test_benjamini_hochberg_is_monotone_by_rank(self) -> None:
        pvalues = [0.01, 0.04, 0.03, 0.20]
        adjusted = benjamini_hochberg(pvalues)
        order = np.argsort(pvalues)
        ordered_adjusted = np.asarray(adjusted)[order]
        self.assertTrue(np.all(np.diff(ordered_adjusted) >= -1e-12))
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))

    def test_risk_coverage_counts_decrease(self) -> None:
        rows = risk_coverage_metrics(synthetic_frame())
        counts = [row["retained_count"] for row in rows]
        self.assertEqual(counts, [40, 36, 32, 28, 20])
        self.assertTrue(all(a >= b for a, b in zip(counts, counts[1:])))


if __name__ == "__main__":
    unittest.main()
