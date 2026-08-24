from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "d3_oof" / "outputs" / "D3_TTA_OOF_V1" / "D3_TTA_OOF_predictions.csv"
OUTPUT = ROOT / "d3_oof" / "outputs" / "D3_TTA_OOF_V1" / "D3_TTA_subgroup_report.json"


def mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def metrics(mask: np.ndarray, target: np.ndarray, e1: np.ndarray, d3: np.ndarray) -> dict:
    candidate = 0.5 * e1 + 0.5 * d3
    base_error = np.abs(e1 - target)
    candidate_error = np.abs(candidate - target)
    delta = candidate_error - base_error
    return {
        "count": int(mask.sum()),
        "e1_tta_mae": mae(e1[mask] - target[mask]),
        "d3_tta_mae": mae(d3[mask] - target[mask]),
        "ensemble_50_50_mae": mae(candidate[mask] - target[mask]),
        "delta_vs_e1_tta": float(delta[mask].mean()),
        "ensemble_better_count": int((delta[mask] < 0).sum()),
        "ensemble_worse_count": int((delta[mask] > 0).sum()),
    }


def main() -> int:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"target_months", "sex", "fold", "d3_tta_std_months", "e1_tta_prediction_months", "d3_tta_prediction_months"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    target = np.array([float(r["target_months"]) for r in rows])
    e1 = np.array([float(r["e1_tta_prediction_months"]) for r in rows])
    d3 = np.array([float(r["d3_tta_prediction_months"]) for r in rows])
    age = target
    std = np.array([float(r["d3_tta_std_months"]) for r in rows])
    sex = np.array([r["sex"] for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])

    age_bins = [(0, 60, "0-59"), (60, 120, "60-119"), (120, 180, "120-179"), (180, 229, "180-228")]
    q1, q2 = np.quantile(std, [1 / 3, 2 / 3])
    groups = {
        "overall": np.ones(len(rows), dtype=bool),
        "sex": {s: sex == s for s in ("F", "M")},
        "age_bin": {name: (age >= low) & (age < high) for low, high, name in age_bins},
        "fold": {str(f): fold == f for f in range(1, 6)},
        "tta_uncertainty": {
            "low": std <= q1,
            "medium": (std > q1) & (std <= q2),
            "high": std > q2,
        },
    }
    report = {
        "status": "PASS",
        "test_used": False,
        "count": len(rows),
        "protocol": "fixed 50/50 E1-TTA + D3-TTA; subgroup analysis only, no test tuning",
        "tta_std_tercile_thresholds_months": {"q33": float(q1), "q67": float(q2)},
        "groups": {},
    }
    for category, category_groups in groups.items():
        if isinstance(category_groups, np.ndarray):
            category_groups = {"overall": category_groups}
        report["groups"][category] = {
            name: metrics(mask, target, e1, d3)
            for name, mask in category_groups.items()
            if int(mask.sum()) > 0
        }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
