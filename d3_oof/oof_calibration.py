from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "d3_oof/outputs/D3_TTA_OOF_V1/D3_TTA_OOF_predictions.csv"
OUTPUT = ROOT / "d3_oof/outputs/D3_TTA_OOF_V1/D3_TTA_calibration_report.json"


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction - target).mean())


def main() -> int:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target = np.array([float(row["target_months"]) for row in rows])
    e1 = np.array([float(row["e1_tta_prediction_months"]) for row in rows])
    d3 = np.array([float(row["d3_tta_prediction_months"]) for row in rows])
    ensemble = 0.5 * e1 + 0.5 * d3
    fold = np.array([int(row["fold"]) for row in rows])

    calibrated = np.empty_like(ensemble)
    fold_params = {}
    for held_out in range(1, 6):
        train_mask = fold != held_out
        val_mask = fold == held_out
        slope, intercept = fit_linear(ensemble[train_mask], target[train_mask])
        calibrated[val_mask] = slope * ensemble[val_mask] + intercept
        fold_params[str(held_out)] = {
            "slope": slope,
            "intercept": intercept,
            "fit_count": int(train_mask.sum()),
            "held_out_count": int(val_mask.sum()),
        }

    raw_mae = mae(ensemble, target)
    calibrated_mae = mae(calibrated, target)
    report = {
        "status": "PASS",
        "test_used": False,
        "count": len(rows),
        "protocol": "global linear calibration; parameters fit on 4 folds and applied to held-out fold; no test access",
        "baseline": {"ensemble_50_50_mae": raw_mae},
        "cross_fitted_calibration": {
            "mae": calibrated_mae,
            "delta_mae_calibrated_minus_baseline": calibrated_mae - raw_mae,
            "fold_parameters": fold_params,
        },
        "decision": "REJECT" if calibrated_mae >= raw_mae else "CANDIDATE",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
