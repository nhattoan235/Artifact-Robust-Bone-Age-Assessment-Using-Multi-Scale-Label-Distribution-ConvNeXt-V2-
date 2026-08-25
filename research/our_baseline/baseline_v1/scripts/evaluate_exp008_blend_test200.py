"""Exploratory test-200 audit for the fixed EXP008 50/50 blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
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
    parser.add_argument("--friend-test", type=Path, required=True)
    parser.add_argument("--exp006-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    friend = pd.read_csv(args.friend_test)
    exp006 = pd.read_csv(args.exp006_test)
    friend["image_id"] = friend["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    exp006["image_id"] = exp006["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    merged = friend[["image_id", "target_months", "prediction_friend_p7_ensemble"]].merge(
        exp006[["image_id", "target_months", "prediction_tta_5fold"]],
        on="image_id", how="inner", suffixes=("_friend", "_exp006"), validate="one_to_one",
    )
    if len(merged) != 200:
        raise ValueError(f"Expected 200 overlapping rows, got {len(merged)}")
    if not np.allclose(merged["target_months_friend"], merged["target_months_exp006"], atol=1e-6):
        raise ValueError("Test targets do not match")

    target = merged["target_months_friend"].to_numpy(float)
    friend_pred = merged["prediction_friend_p7_ensemble"].to_numpy(float)
    exp006_pred = merged["prediction_tta_5fold"].to_numpy(float)
    blend = 0.5 * friend_pred + 0.5 * exp006_pred
    output = merged[["image_id", "target_months_friend"]].rename(columns={"target_months_friend": "target_months"})
    output["friend_p7_prediction_months"] = friend_pred
    output["exp006_tta_prediction_months"] = exp006_pred
    output["blend_50_50_prediction_months"] = blend
    output.to_csv(args.output_dir / "exp008_blend_test200_predictions.csv", index=False)

    report = {
        "experiment": "EXP008_FIXED_BLEND_P7_AND_EXP006_TTA_TEST200",
        "evaluation": "exploratory_test_200",
        "test_labels_used": True,
        "selection_warning": "Do not use this test report to select weights or hyperparameters.",
        "protocol": "Fixed 50/50 blend; weights were fixed from OOF before reading this test report.",
        "count": 200,
        "methods": {
            "friend_p7_ensemble": metric(target, friend_pred),
            "exp006_tta_5fold": metric(target, exp006_pred),
            "blend_50_50": metric(target, blend),
        },
        "prediction_file": str(args.output_dir / "exp008_blend_test200_predictions.csv"),
        "sources": {"friend_test": str(args.friend_test), "exp006_test": str(args.exp006_test)},
    }
    (args.output_dir / "exp008_blend_test200_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
