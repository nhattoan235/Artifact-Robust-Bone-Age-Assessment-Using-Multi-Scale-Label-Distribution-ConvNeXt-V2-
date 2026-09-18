"""Compute balanced MAE variants from locked validation OOF predictions.

The script only reweights existing per-image errors. It does not train,
re-infer, select checkpoints, or access the 200-image RSNA test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "results/pilot_bc_20260914/oof/"
    "C3_Z26_C3_ROI_V2_PILOT_C_OOF_predictions.csv"
)
DEFAULT_OUTPUT = Path("results/pilot_bc_20260914/balanced_oof")
AGE_EDGES = (0, 60, 120, 180, 229)
AGE_LABELS = ("0-59", "60-119", "120-179", "180-228")
REQUIRED_COLUMNS = {
    "image_id", "fold", "sex", "target_months",
    "base_clean", "c_clean", "base_artifact", "c_artifact",
}
SCHEMES: dict[str, tuple[str, ...]] = {
    "ordinary_micro": (),
    "age_macro": ("age_bin",),
    "sex_macro": ("sex",),
    "age_sex_macro": ("age_bin", "sex"),
    "fold_age_macro": ("fold", "age_bin"),
}


def prepare_oof(frame: pd.DataFrame, weight_c: float = 0.73) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"OOF thieu cot: {sorted(missing)}")
    if frame.empty or frame["image_id"].astype(str).duplicated().any():
        raise ValueError("OOF rong hoac co image_id trung")
    if not 0.0 <= weight_c <= 1.0:
        raise ValueError("weight_c phai nam trong [0, 1]")
    work = frame.copy()
    work["age_bin"] = pd.cut(
        work["target_months"], AGE_EDGES, right=False, labels=AGE_LABELS,
    )
    if work["age_bin"].isna().any():
        raise ValueError("target_months nam ngoai cac age bin da khoa")
    work["blend_clean"] = (
        (1.0 - weight_c) * work["base_clean"] + weight_c * work["c_clean"]
    )
    work["blend_artifact"] = (
        (1.0 - weight_c) * work["base_artifact"]
        + weight_c * work["c_artifact"]
    )
    return work


def _arrays_by_group(
    frame: pd.DataFrame, values: np.ndarray, group_columns: tuple[str, ...],
) -> list[np.ndarray]:
    if not group_columns:
        return [values]
    series = pd.Series(values, index=frame.index)
    grouper: str | list[str]
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    return [
        series.loc[index].to_numpy(dtype=np.float64)
        for _, index in frame.groupby(grouper, observed=True, sort=True).groups.items()
    ]


def balanced_mean(
    frame: pd.DataFrame, values: np.ndarray,
    group_columns: tuple[str, ...],
) -> float:
    groups = _arrays_by_group(frame, values, group_columns)
    return float(np.mean([group.mean() for group in groups]))


def stratified_bootstrap_ci(
    frame: pd.DataFrame, delta: np.ndarray,
    group_columns: tuple[str, ...], *, repetitions: int, seed: int,
) -> list[float]:
    if repetitions <= 0:
        raise ValueError("repetitions phai > 0")
    groups = _arrays_by_group(frame, delta, group_columns)
    rng = np.random.default_rng(seed)
    sampled = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 128):
        size = min(128, repetitions - start)
        group_means = np.empty((size, len(groups)), dtype=np.float64)
        for group_index, group in enumerate(groups):
            indices = rng.integers(0, len(group), size=(size, len(group)))
            group_means[:, group_index] = group[indices].mean(axis=1)
        sampled[start:start + size] = group_means.mean(axis=1)
    return [float(value) for value in np.quantile(sampled, (0.025, 0.975))]


def evaluate_balanced_oof(
    frame: pd.DataFrame, *, repetitions: int = 20000,
    seed: int = 20260914,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for view_offset, view in enumerate(("clean", "artifact")):
        target = frame["target_months"].to_numpy(dtype=np.float64)
        errors = {
            model: np.abs(frame[f"{model}_{view}"].to_numpy(dtype=np.float64) - target)
            for model in ("base", "c", "blend")
        }
        for scheme_offset, (scheme, group_columns) in enumerate(SCHEMES.items()):
            baseline_value = balanced_mean(frame, errors["base"], group_columns)
            for model_offset, model in enumerate(("base", "c", "blend")):
                value = balanced_mean(frame, errors[model], group_columns)
                row: dict[str, object] = {
                    "view": view,
                    "scheme": scheme,
                    "groups": 1 if not group_columns else int(
                        frame.groupby(
                            group_columns[0] if len(group_columns) == 1
                            else list(group_columns),
                            observed=True,
                        ).ngroups
                    ),
                    "model": model,
                    "mae": value,
                    "delta_vs_baseline": value - baseline_value,
                }
                if model == "base":
                    row["delta_ci_low"] = 0.0
                    row["delta_ci_high"] = 0.0
                else:
                    interval = stratified_bootstrap_ci(
                        frame, errors[model] - errors["base"], group_columns,
                        repetitions=repetitions,
                        seed=seed + view_offset * 1000 + scheme_offset * 10 + model_offset,
                    )
                    row["delta_ci_low"], row["delta_ci_high"] = interval
                rows.append(row)

        for group_name, group in frame.groupby("age_bin", observed=True, sort=True):
            target_group = group["target_months"].to_numpy(dtype=np.float64)
            for model in ("base", "c", "blend"):
                prediction = group[f"{model}_{view}"].to_numpy(dtype=np.float64)
                group_rows.append({
                    "view": view,
                    "age_bin": str(group_name),
                    "count": int(len(group)),
                    "model": model,
                    "mae": float(np.abs(prediction - target_group).mean()),
                })
    return pd.DataFrame(rows), pd.DataFrame(group_rows)


def run_analysis(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT,
    *, weight_c: float = 0.73, repetitions: int = 20000,
) -> dict[str, object]:
    frame = prepare_oof(pd.read_csv(input_path), weight_c=weight_c)
    metrics, age_groups = evaluate_balanced_oof(
        frame, repetitions=repetitions,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "balanced_oof_metrics.csv"
    age_path = output_dir / "balanced_oof_age_groups.csv"
    metrics.to_csv(metrics_path, index=False)
    age_groups.to_csv(age_path, index=False)

    selected = metrics[metrics["scheme"].isin(("ordinary_micro", "age_macro"))]
    report = {
        "protocol": (
            "Locked 14,036-image validation OOF; existing predictions only; "
            "fixed blend=0.27 baseline + 0.73 Pilot C; no test access"
        ),
        "test_accessed": False,
        "count": int(len(frame)),
        "weight_c": float(weight_c),
        "bootstrap_repetitions": int(repetitions),
        "definitions": {
            "ordinary_micro": "Every image has equal weight.",
            "age_macro": "Each of four locked age bins has 25% weight.",
            "sex_macro": "Female and male groups each have 50% weight.",
            "age_sex_macro": "Each observed age-by-sex cell has equal weight.",
            "fold_age_macro": "Each of 20 fold-by-age cells has equal weight.",
        },
        "headline": selected.to_dict(orient="records"),
        "files": {
            "metrics": str(metrics_path),
            "age_groups": str(age_path),
        },
    }
    report_path = output_dir / "balanced_oof_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(selected.to_string(index=False))
    print(f"SAVED: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Balanced metrics from locked OOF")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weight-c", type=float, default=0.73)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()
    run_analysis(
        args.input, args.output_dir, weight_c=args.weight_c,
        repetitions=args.bootstrap_repetitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
