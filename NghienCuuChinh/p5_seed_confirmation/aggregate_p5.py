from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

from p1_baseline.metrics import compute_metrics


ROOT = Path("p1_baseline/runs")
SEEDS = (17, 42, 123)
PATHS = {
    17: (
        ROOT / "P5_D0_CONVNEXT_TINY_SEED17/val_predictions_best.csv",
        ROOT / "P5_D3_CONVNEXT_TINY_LDL_SEED17/val_predictions_best.csv",
    ),
    42: (
        ROOT / "P2_A2_LIGHT_FLIP_CONVNEXT_TINY_SEED42/val_predictions_best.csv",
        ROOT / "P4_D3_CONVNEXT_TINY_LDL_SIGMA2_LAMBDA02_SEED42/val_predictions_best.csv",
    ),
    123: (
        ROOT / "P5_D0_CONVNEXT_TINY_SEED123/val_predictions_best.csv",
        ROOT / "P5_D3_CONVNEXT_TINY_LDL_SEED123/val_predictions_best.csv",
    ),
}
AGE_BINS = [0, 60, 120, 180, 229]


def load(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu prediction artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["image_id"]: row for row in rows}


def records(rows: dict[str, dict[str, str]], prediction_column: str) -> list[dict]:
    return [
        {
            "target_months": float(row["target_months"]),
            "prediction_months": float(row[prediction_column]),
            "sex": row["sex"],
        }
        for _, row in sorted(rows.items(), key=lambda item: int(float(item[0])))
    ]


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_ci(delta_matrix: np.ndarray, seed: int = 42, samples: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    image_count = delta_matrix.shape[1]
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.integers(0, image_count, size=image_count)
        draws[index] = delta_matrix[:, sampled].mean()
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main() -> None:
    seed_reports: dict[str, dict] = {}
    fused_deltas = []
    regression_deltas = []
    d0_maes = []
    fused_maes = []
    regression_maes = []

    reference_ids: list[str] | None = None
    for seed in SEEDS:
        d0_rows, d3_rows = (load(path) for path in PATHS[seed])
        ids = sorted(d0_rows, key=lambda value: int(float(value)))
        if ids != sorted(d3_rows, key=lambda value: int(float(value))) or len(ids) != 1425:
            raise RuntimeError(f"ID không khớp hoặc không đủ 1.425 ảnh ở seed {seed}")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"Thứ tự/tập ID khác giữa các seed tại seed {seed}")

        for image_id in ids:
            a, b = d0_rows[image_id], d3_rows[image_id]
            if a["target_months"] != b["target_months"] or a["sex"] != b["sex"]:
                raise RuntimeError(f"Ground truth/sex không khớp tại seed={seed}, ID={image_id}")
            if "regression_months" not in b:
                raise RuntimeError(f"D3 thiếu regression_months tại seed={seed}")

        d0_metrics = compute_metrics(records(d0_rows, "prediction_months"), AGE_BINS)
        fused_metrics = compute_metrics(records(d3_rows, "prediction_months"), AGE_BINS)
        regression_metrics = compute_metrics(records(d3_rows, "regression_months"), AGE_BINS)
        d0_maes.append(d0_metrics["mae"])
        fused_maes.append(fused_metrics["mae"])
        regression_maes.append(regression_metrics["mae"])

        target = np.array([float(d0_rows[item]["target_months"]) for item in ids])
        d0_prediction = np.array([float(d0_rows[item]["prediction_months"]) for item in ids])
        fused_prediction = np.array([float(d3_rows[item]["prediction_months"]) for item in ids])
        regression_prediction = np.array([float(d3_rows[item]["regression_months"]) for item in ids])
        fused_delta = np.abs(fused_prediction - target) - np.abs(d0_prediction - target)
        regression_delta = np.abs(regression_prediction - target) - np.abs(d0_prediction - target)
        fused_deltas.append(fused_delta)
        regression_deltas.append(regression_delta)
        seed_reports[str(seed)] = {
            "d0": d0_metrics,
            "d3_fused": fused_metrics,
            "d3_regression_only_adaptive": regression_metrics,
            "delta_fused_minus_d0": float(fused_delta.mean()),
            "delta_regression_minus_d0": float(regression_delta.mean()),
        }

    fused_matrix = np.stack(fused_deltas)
    regression_matrix = np.stack(regression_deltas)
    report = {
        "status": "PASS",
        "seeds": list(SEEDS),
        "count_per_seed": 1425,
        "seed_reports": seed_reports,
        "across_seed": {
            "d0_mae": summarize(d0_maes),
            "d3_fused_mae": summarize(fused_maes),
            "d3_regression_only_adaptive_mae": summarize(regression_maes),
            "delta_fused_minus_d0": summarize([float(row.mean()) for row in fused_matrix]),
            "delta_regression_minus_d0": summarize([float(row.mean()) for row in regression_matrix]),
            "cluster_bootstrap_95_ci_fused": bootstrap_ci(fused_matrix),
            "cluster_bootstrap_95_ci_regression_only_adaptive": bootstrap_ci(regression_matrix),
            "seeds_favoring_fused": int(sum(row.mean() < 0 for row in fused_matrix)),
            "seeds_favoring_regression_only_adaptive": int(sum(row.mean() < 0 for row in regression_matrix)),
        },
        "decision_rule": {
            "minimum_mean_improvement_months": 0.10,
            "minimum_seeds_favoring_candidate": 2,
            "primary": "D3 fused",
            "secondary_adaptive": "D3 regression-only",
        },
    }
    output = Path("p5_seed_confirmation/P5_AGGREGATE.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["across_seed"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
