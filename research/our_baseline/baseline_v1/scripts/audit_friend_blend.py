"""Audit a fixed-weight blend between the local baseline and friend's P7 OOF.

This script is intentionally an audit tool: it never reads the 200-image test
labels and it records the exact source files, overlap, target agreement, and
metrics for every requested blend weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_id(series: pd.Series) -> pd.Series:
    # CSV writers can turn integer IDs into strings such as "1386.0".
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def metric_row(name: str, target: np.ndarray, pred: np.ndarray, weight: float) -> dict:
    error = pred - target
    absolute = np.abs(error)
    return {
        "model": name,
        "friend_weight": weight,
        "n": int(target.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(absolute)),
        "within_6_months": float(np.mean(absolute <= 6.0)),
        "within_12_months": float(np.mean(absolute <= 12.0)),
        "within_18_months": float(np.mean(absolute <= 18.0)),
    }


def load_own(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "id" if "id" in frame.columns else "image_id"
    target_col = "boneage" if "boneage" in frame.columns else "target_months"
    pred_col = "pred" if "pred" in frame.columns else "prediction_months"
    required = {id_col, target_col, pred_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Own prediction file is missing columns: {sorted(missing)}")
    result = frame[[id_col, target_col, pred_col]].rename(
        columns={id_col: "image_id", target_col: "target_months", pred_col: "own_prediction_months"}
    )
    result["image_id"] = normalize_id(result["image_id"])
    result["target_months"] = pd.to_numeric(result["target_months"], errors="raise")
    result["own_prediction_months"] = pd.to_numeric(result["own_prediction_months"], errors="raise")
    if result["image_id"].duplicated().any():
        raise ValueError("Own validation predictions contain duplicate image IDs")
    return result


def load_friend(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(root.glob("P7_FINAL_V3_FOLD_*/val_predictions_best.csv"))
    if not files:
        raise FileNotFoundError(f"No P7 fold prediction files found under {root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        required = {"image_id", "target_months", "prediction_months"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        part = frame[["image_id", "target_months", "prediction_months"]].copy()
        part["source_file"] = str(path)
        frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    result["image_id"] = normalize_id(result["image_id"])
    result["target_months"] = pd.to_numeric(result["target_months"], errors="raise")
    result["prediction_months"] = pd.to_numeric(result["prediction_months"], errors="raise")
    duplicates = result[result["image_id"].duplicated(keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Friend OOF predictions contain duplicate IDs, examples: {duplicates.image_id.head().tolist()}")
    return result, files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--own-predictions", type=Path, required=True)
    parser.add_argument("--friend-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    own = load_own(args.own_predictions)
    friend, friend_files = load_friend(args.friend_root)
    merged = own.merge(friend, on="image_id", how="inner", suffixes=("_own", "_friend"), validate="one_to_one")
    if merged.empty:
        raise ValueError("No image IDs overlap between own validation and friend OOF predictions")

    target_delta = np.abs(merged["target_months_own"] - merged["target_months_friend"])
    target_agreement = bool(np.all(target_delta <= 1e-6))
    target = merged["target_months_own"].to_numpy(dtype=np.float64)
    own_pred = merged["own_prediction_months"].to_numpy(dtype=np.float64)
    friend_pred = merged["prediction_months"].to_numpy(dtype=np.float64)

    rows = [metric_row("own_baseline", target, own_pred, 0.0)]
    for weight in WEIGHTS[1:]:
        blended = (1.0 - weight) * own_pred + weight * friend_pred
        rows.append(metric_row(f"blend_{weight:.2f}", target, blended, weight))
    # The friend-only row is already represented by weight=1.0; keep ordering explicit.
    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values("friend_weight").reset_index(drop=True)

    merged["target_months"] = merged["target_months_own"]
    merged["friend_weight_0_25_prediction"] = 0.75 * own_pred + 0.25 * friend_pred
    merged["friend_weight_0_50_prediction"] = 0.50 * own_pred + 0.50 * friend_pred
    merged["friend_weight_0_75_prediction"] = 0.25 * own_pred + 0.75 * friend_pred
    merged = merged[
        [
            "image_id",
            "target_months",
            "own_prediction_months",
            "prediction_months",
            "friend_weight_0_25_prediction",
            "friend_weight_0_50_prediction",
            "friend_weight_0_75_prediction",
            "source_file",
        ]
    ].rename(columns={"prediction_months": "friend_prediction_months"})
    merged.to_csv(args.output_dir / "blend_merged_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "blend_metrics.csv", index=False)

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "fixed-weight blend on overlap between own official validation predictions and friend's P7 OOF predictions",
        "test_labels_used": False,
        "own_prediction_file": str(args.own_predictions),
        "own_prediction_sha256": sha256(args.own_predictions),
        "friend_root": str(args.friend_root),
        "friend_prediction_files": [str(path) for path in friend_files],
        "friend_prediction_sha256": {str(path): sha256(path) for path in friend_files},
        "own_validation_rows": int(len(own)),
        "friend_oof_rows": int(len(friend)),
        "overlap_rows": int(len(merged)),
        "target_values_agree": target_agreement,
        "max_target_difference_months": float(target_delta.max()),
        "weights_tested": list(WEIGHTS),
        "selection_warning": "The official validation split is development data. Any best blend weight must be confirmed on a separate OOF/holdout protocol before being reported as final.",
        "metrics_file": str(args.output_dir / "blend_metrics.csv"),
        "merged_predictions_file": str(args.output_dir / "blend_merged_predictions.csv"),
    }
    (args.output_dir / "blend_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"overlap={len(merged)} target_values_agree={target_agreement} max_target_delta={target_delta.max():.6f}")


if __name__ == "__main__":
    main()
