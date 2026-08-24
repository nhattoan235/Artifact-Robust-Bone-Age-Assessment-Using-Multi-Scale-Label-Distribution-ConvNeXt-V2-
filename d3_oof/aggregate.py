from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


AGE_BINS = ((0, 60, "0-59"), (60, 120, "60-119"), (120, 180, "120-179"), (180, 229, "180-228"))


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing prediction file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty prediction file: {path}")
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing prediction columns {sorted(missing)}: {path}")
    return rows


def _bootstrap_ci(delta: np.ndarray, seed: int, samples: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start : start + size] = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _group_mae(error: np.ndarray, target: np.ndarray, sex: np.ndarray) -> tuple[dict, dict]:
    by_sex = {group: float(error[sex == group].mean()) for group in ("F", "M")}
    by_age = {
        name: float(error[(target >= low) & (target < high)].mean())
        for low, high, name in AGE_BINS
        if np.any((target >= low) & (target < high))
    }
    return by_sex, by_age


def _comparison(
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
    seed: int,
) -> dict:
    candidate_error = np.abs(candidate_prediction - target)
    baseline_error = np.abs(baseline_prediction - target)
    delta = candidate_error - baseline_error
    return {
        "delta_mae_candidate_minus_baseline": float(delta.mean()),
        "paired_bootstrap_95_ci": _bootstrap_ci(delta, seed),
        "candidate_better_images": int((delta < 0).sum()),
        "candidate_worse_images": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "prediction_correlation": float(np.corrcoef(candidate_prediction, baseline_prediction)[0, 1]),
        "residual_correlation": float(
            np.corrcoef(candidate_prediction - target, baseline_prediction - target)[0, 1]
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_oof(
    workspace: Path,
    d3_run_root: Path,
    output_dir: Path,
    expected_count: int = 14036,
) -> dict:
    workspace = workspace.resolve()
    d3_run_root = d3_run_root.resolve()
    required_e1 = {"image_id", "target_months", "prediction_months", "sex"}
    required_d3 = required_e1 | {"regression_months", "distribution_months"}
    pooled: list[dict] = []
    seen_ids: set[str] = set()
    for fold in range(1, 6):
        e1_path = workspace / f"p7_results/results/P7_FINAL_V3_FOLD_{fold}/val_predictions_best.csv"
        d3_path = d3_run_root / f"D3_OOF_V1_FOLD_{fold}/val_predictions_best.csv"
        e1_rows = _read_rows(e1_path, required_e1)
        d3_rows = _read_rows(d3_path, required_d3)
        e1 = {str(int(float(row["image_id"]))): row for row in e1_rows}
        d3 = {str(int(float(row["image_id"]))): row for row in d3_rows}
        if len(e1) != len(e1_rows) or len(d3) != len(d3_rows):
            raise RuntimeError(f"Duplicate prediction ID within fold {fold}")
        if set(e1) != set(d3):
            raise RuntimeError(f"Missing or extra IDs between E1 and D3 fold {fold}")
        for image_id in sorted(e1, key=int):
            if image_id in seen_ids:
                raise RuntimeError(f"Duplicate OOF ID across folds: {image_id}")
            seen_ids.add(image_id)
            e1_row, d3_row = e1[image_id], d3[image_id]
            e1_target = float(e1_row["target_months"])
            d3_target = float(d3_row["target_months"])
            if e1_target != d3_target or e1_row["sex"] != d3_row["sex"]:
                raise RuntimeError(f"Target/sex mismatch at ID={image_id}")
            e1_prediction = float(e1_row["prediction_months"])
            d3_fused = float(d3_row["prediction_months"])
            d3_regression = float(d3_row["regression_months"])
            pooled.append({
                "image_id": image_id,
                "fold": fold,
                "target_months": e1_target,
                "sex": e1_row["sex"],
                "e1_prediction_months": e1_prediction,
                "d3_fused_prediction_months": d3_fused,
                "d3_regression_prediction_months": d3_regression,
                "d3_distribution_prediction_months": float(d3_row["distribution_months"]),
                "ensemble_fused_50_50_months": 0.5 * e1_prediction + 0.5 * d3_fused,
                "ensemble_regression_50_50_months": 0.5 * e1_prediction + 0.5 * d3_regression,
            })
    if len(pooled) != expected_count or len(seen_ids) != expected_count:
        raise RuntimeError(f"Expected {expected_count} unique OOF IDs, got {len(seen_ids)}")

    pooled.sort(key=lambda row: int(row["image_id"]))
    target = np.array([row["target_months"] for row in pooled], dtype=np.float64)
    sex = np.array([row["sex"] for row in pooled])
    predictions = {
        "E1": np.array([row["e1_prediction_months"] for row in pooled]),
        "D3_FUSED": np.array([row["d3_fused_prediction_months"] for row in pooled]),
        "D3_REGRESSION_ONLY": np.array([row["d3_regression_prediction_months"] for row in pooled]),
        "E1_D3_FUSED_50_50": np.array([row["ensemble_fused_50_50_months"] for row in pooled]),
        "E1_D3_REGRESSION_50_50": np.array([row["ensemble_regression_50_50_months"] for row in pooled]),
    }
    errors = {name: np.abs(prediction - target) for name, prediction in predictions.items()}
    mae_by_sex, mae_by_age = {}, {}
    for name, error in errors.items():
        mae_by_sex[name], mae_by_age[name] = _group_mae(error, target, sex)
    comparisons = {}
    for index, (candidate, baseline) in enumerate((
        ("D3_FUSED", "E1"),
        ("D3_REGRESSION_ONLY", "E1"),
        ("E1_D3_FUSED_50_50", "E1"),
        ("E1_D3_REGRESSION_50_50", "E1"),
    )):
        comparisons[f"{candidate}_vs_{baseline}"] = _comparison(
            predictions[candidate], predictions[baseline], target, 2026 + index
        )
    report = {
        "status": "PASS",
        "test_used": False,
        "count": len(pooled),
        "ensemble_weights": {"E1": 0.5, "D3": 0.5},
        "mae": {name: float(error.mean()) for name, error in errors.items()},
        "mae_by_sex": mae_by_sex,
        "mae_by_age_bin": mae_by_age,
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    d3_rows = [{
        "image_id": row["image_id"],
        "fold": row["fold"],
        "target_months": row["target_months"],
        "prediction_months": row["d3_fused_prediction_months"],
        "regression_months": row["d3_regression_prediction_months"],
        "distribution_months": row["d3_distribution_prediction_months"],
        "absolute_error": abs(row["d3_fused_prediction_months"] - row["target_months"]),
        "sex": row["sex"],
    } for row in pooled]
    _write_csv(output_dir / "D3_OOF_predictions.csv", d3_rows)
    _write_csv(output_dir / "E1_D3_ensemble_predictions.csv", pooled)
    (output_dir / "D3_OOF_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate_oof(args.workspace, args.run_root, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
