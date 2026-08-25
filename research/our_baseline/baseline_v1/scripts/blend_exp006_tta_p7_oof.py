"""Audit a fixed 50/50 blend between P7 friend OOF and EXP006 P7-TTA OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "n": int(len(target)),
        "mae_months": float(absolute.mean()),
        "rmse_months": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_months": float(np.median(absolute)),
        "accuracy_pm6": float(np.mean(absolute <= 6)),
        "accuracy_pm12": float(np.mean(absolute <= 12)),
        "accuracy_pm18": float(np.mean(absolute <= 18)),
        "signed_bias_months": float(error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p7-oof", type=Path, required=True)
    parser.add_argument("--exp006-tta-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p7 = pd.read_csv(args.p7_oof)
    tta = pd.read_csv(args.exp006_tta_oof)
    p7["image_id"] = p7["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    tta["image_id"] = tta["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    required_p7 = {"image_id", "target_months", "prediction_months"}
    required_tta = {"image_id", "target_months", "tta_prediction_months"}
    if not required_p7.issubset(p7.columns):
        raise ValueError(f"P7 OOF missing columns: {sorted(required_p7 - set(p7.columns))}")
    if not required_tta.issubset(tta.columns):
        raise ValueError(f"TTA OOF missing columns: {sorted(required_tta - set(tta.columns))}")
    if p7["image_id"].duplicated().any() or tta["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in OOF input")

    merged = p7[["image_id", "target_months", "prediction_months"]].merge(
        tta[["image_id", "target_months", "tta_prediction_months"]],
        on="image_id", how="inner", suffixes=("_p7", "_exp006"), validate="one_to_one",
    )
    if len(merged) != 14036:
        raise ValueError(f"Expected 14,036 overlapping OOF rows, got {len(merged)}")
    if not np.allclose(merged["target_months_p7"], merged["target_months_exp006"], atol=1e-6):
        raise ValueError("P7 and EXP006 target labels do not match")

    target = merged["target_months_p7"].to_numpy(float)
    p7_pred = merged["prediction_months"].to_numpy(float)
    tta_pred = merged["tta_prediction_months"].to_numpy(float)
    blend_pred = 0.5 * p7_pred + 0.5 * tta_pred
    result = merged[["image_id", "target_months_p7"]].rename(columns={"target_months_p7": "target_months"})
    result["p7_prediction_months"] = p7_pred
    result["exp006_tta_prediction_months"] = tta_pred
    result["blend_50_50_prediction_months"] = blend_pred
    result["blend_minus_target_months"] = blend_pred - target
    prediction_path = args.output_dir / "blend_oof_predictions.csv"
    result.to_csv(prediction_path, index=False)

    report = {
        "experiment": "EXP008_FIXED_BLEND_P7_AND_EXP006_TTA_OOF",
        "protocol": "Fixed 50/50 blend of two row-aligned OOF prediction sources",
        "test_labels_used": False,
        "selection_warning": "50/50 was fixed before this audit; do not tune weights on the labeled 200-image test set.",
        "count": int(len(result)),
        "target_match": True,
        "methods": {
            "p7_friend_oof": metrics(target, p7_pred),
            "exp006_tta_oof": metrics(target, tta_pred),
            "blend_50_50": metrics(target, blend_pred),
        },
        "delta_blend_minus_exp006_tta": {
            "mae_months": float(np.abs(blend_pred - target).mean() - np.abs(tta_pred - target).mean()),
            "rmse_months": float(np.sqrt(np.mean((blend_pred - target) ** 2)) - np.sqrt(np.mean((tta_pred - target) ** 2))),
        },
        "sources": {
            "p7_oof": str(args.p7_oof),
            "exp006_tta_oof": str(args.exp006_tta_oof),
        },
        "prediction_file": str(prediction_path),
    }
    report_path = args.output_dir / "blend_oof_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = f"""# EXP008 — Fixed 50/50 P7 + EXP006 TTA OOF blend

## Protocol

- Rows: {len(result)}
- Test labels used: **No**
- Weight: P7 friend OOF `0.50` + EXP006 TTA OOF `0.50`
- Target labels matched exactly across both sources.

## Metrics

| Method | MAE | RMSE | Median AE | Accuracy ±6 | Accuracy ±12 |
|---|---:|---:|---:|---:|---:|
| P7 friend OOF | {metrics(target, p7_pred)['mae_months']:.4f} | {metrics(target, p7_pred)['rmse_months']:.4f} | {metrics(target, p7_pred)['median_absolute_error_months']:.4f} | {metrics(target, p7_pred)['accuracy_pm6']:.4%} | {metrics(target, p7_pred)['accuracy_pm12']:.4%} |
| EXP006 TTA OOF | {metrics(target, tta_pred)['mae_months']:.4f} | {metrics(target, tta_pred)['rmse_months']:.4f} | {metrics(target, tta_pred)['median_absolute_error_months']:.4f} | {metrics(target, tta_pred)['accuracy_pm6']:.4%} | {metrics(target, tta_pred)['accuracy_pm12']:.4%} |
| Fixed 50/50 blend | **{metrics(target, blend_pred)['mae_months']:.4f}** | **{metrics(target, blend_pred)['rmse_months']:.4f}** | **{metrics(target, blend_pred)['median_absolute_error_months']:.4f}** | **{metrics(target, blend_pred)['accuracy_pm6']:.4%}** | **{metrics(target, blend_pred)['accuracy_pm12']:.4%}** |

## Decision

Keep the fixed blend as the current OOF candidate. The next model, LDL, must produce row-aligned OOF predictions on the same 14,036 images and beat this `MAE={metrics(target, blend_pred)['mae_months']:.4f}` before being added to a three-model blend.

The labeled 200-image test set remains exploratory and is not used for selecting the blend.
"""
    (args.output_dir / "EXP008_FIXED_BLEND_OOF_AUDIT.md").write_text(audit, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
