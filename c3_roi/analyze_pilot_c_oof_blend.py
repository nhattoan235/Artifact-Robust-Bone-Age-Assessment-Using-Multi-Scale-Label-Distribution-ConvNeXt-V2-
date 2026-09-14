"""Cross-fitted baseline/Pilot-C blending and subgroup bootstrap analysis.

The script consumes validation OOF predictions only.  It never reads the
200-image RSNA test set.  Blend weights are selected on four folds and applied
to the held-out fold, so the reported cross-fitted result does not evaluate a
weight on the same fold that selected it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/OOF/"
    "C3_Z26_C3_ROI_V2_PILOT_C_OOF_predictions.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "blend_analysis"
AGE_EDGES = (0, 60, 120, 180, 229)
AGE_LABELS = ("0-59", "60-119", "120-179", "180-228")
REQUIRED_COLUMNS = {
    "image_id", "fold", "sex", "target_months",
    "base_clean", "c_clean", "base_artifact", "c_artifact",
}


def _validate(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"OOF thiếu cột: {sorted(missing)}")
    if frame.empty:
        raise ValueError("OOF rỗng")
    if frame["image_id"].astype(str).duplicated().any():
        raise ValueError("OOF có image_id trùng")
    if frame["fold"].isna().any():
        raise ValueError("OOF có fold rỗng")
    for column in REQUIRED_COLUMNS - {"image_id", "sex"}:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"OOF cột {column} có NaN/Inf")


def _age_bins(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame["target_months"], bins=AGE_EDGES, right=False,
        labels=AGE_LABELS,
    )


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.abs(prediction - target).mean())


def analyze_weight(frame: pd.DataFrame, weight_c: float) -> dict:
    """Evaluate one global Pilot-C weight against the locked baseline."""
    _validate(frame)
    weight_c = float(weight_c)
    if not 0.0 <= weight_c <= 1.0:
        raise ValueError("weight_c phải nằm trong [0, 1]")

    y = frame["target_months"].to_numpy(dtype=np.float64)
    base_clean = frame["base_clean"].to_numpy(dtype=np.float64)
    base_artifact = frame["base_artifact"].to_numpy(dtype=np.float64)
    c_clean = frame["c_clean"].to_numpy(dtype=np.float64)
    c_artifact = frame["c_artifact"].to_numpy(dtype=np.float64)
    clean = (1.0 - weight_c) * base_clean + weight_c * c_clean
    artifact = (1.0 - weight_c) * base_artifact + weight_c * c_artifact

    baseline_clean_mae = _mae(y, base_clean)
    baseline_artifact_mae = _mae(y, base_artifact)
    clean_mae = _mae(y, clean)
    artifact_mae = _mae(y, artifact)
    baseline_disagreement = float(np.abs(base_artifact - base_clean).mean())
    disagreement = float(np.abs(artifact - clean).mean())

    age_values: dict[str, dict[str, float | int]] = {}
    age_bins = _age_bins(frame)
    for label in AGE_LABELS:
        mask = (age_bins == label).to_numpy()
        if not mask.any():
            continue
        age_values[label] = {
            "count": int(mask.sum()),
            "clean_delta": _mae(y[mask], clean[mask]) - _mae(y[mask], base_clean[mask]),
            "artifact_delta": (
                _mae(y[mask], artifact[mask])
                - _mae(y[mask], base_artifact[mask])
            ),
        }

    return {
        "weight_c": weight_c,
        "count": int(len(frame)),
        "clean_mae": clean_mae,
        "baseline_clean_mae": baseline_clean_mae,
        "clean_delta": clean_mae - baseline_clean_mae,
        "artifact_mae": artifact_mae,
        "baseline_artifact_mae": baseline_artifact_mae,
        "artifact_delta": artifact_mae - baseline_artifact_mae,
        "artifact_gain": baseline_artifact_mae - artifact_mae,
        "disagreement_mean": disagreement,
        "baseline_disagreement_mean": baseline_disagreement,
        "disagreement_reduction": (
            1.0 - disagreement / max(baseline_disagreement, 1e-12)
        ),
        "max_age_clean_delta": max(v["clean_delta"] for v in age_values.values()),
        "max_age_artifact_delta": max(
            v["artifact_delta"] for v in age_values.values()
        ),
        "age_bins": age_values,
    }


def _is_feasible(
    summary: dict, *, clean_margin: float, age_clean_margin: float,
    age_artifact_margin: float, min_disagreement_reduction: float,
) -> bool:
    eps = 1e-12
    return bool(
        summary["clean_delta"] <= clean_margin + eps
        and summary["max_age_clean_delta"] <= age_clean_margin + eps
        and summary["max_age_artifact_delta"] <= age_artifact_margin + eps
        and summary["disagreement_reduction"] >= min_disagreement_reduction - eps
    )


def _search(
    frame: pd.DataFrame, weights: Iterable[float], *, clean_margin: float,
    age_clean_margin: float, age_artifact_margin: float,
    min_disagreement_reduction: float,
) -> tuple[list[dict], dict | None]:
    grid: list[dict] = []
    for weight in weights:
        summary = analyze_weight(frame, float(weight))
        summary["feasible"] = _is_feasible(
            summary, clean_margin=clean_margin,
            age_clean_margin=age_clean_margin,
            age_artifact_margin=age_artifact_margin,
            min_disagreement_reduction=min_disagreement_reduction,
        )
        grid.append(summary)
    feasible = [row for row in grid if row["feasible"]]
    if not feasible:
        return grid, None
    selected = max(
        feasible,
        key=lambda row: (
            row["artifact_gain"], -row["clean_delta"],
            row["disagreement_reduction"], -row["weight_c"],
        ),
    )
    return grid, selected


def crossfit_weights(
    frame: pd.DataFrame, *, weights: Iterable[float] | None = None,
    clean_margin: float = 0.05, age_clean_margin: float = 0.20,
    age_artifact_margin: float = 0.0,
    min_disagreement_reduction: float = 0.40,
) -> dict:
    """Select on four folds, apply to the fifth, and pool held-out rows."""
    _validate(frame)
    if weights is None:
        weights = np.linspace(0.0, 1.0, 101)
    weights = tuple(float(value) for value in weights)
    if not weights:
        raise ValueError("Cần ít nhất một weight")

    selected_weights: dict[str, float] = {}
    selection_details: dict[str, dict] = {}
    prediction_parts: list[pd.DataFrame] = []
    all_feasible = True
    folds = sorted(int(value) for value in frame["fold"].unique())
    for held_out in folds:
        train = frame[frame["fold"] != held_out]
        held = frame[frame["fold"] == held_out].copy()
        _, selected = _search(
            train, weights,
            clean_margin=clean_margin,
            age_clean_margin=age_clean_margin,
            age_artifact_margin=age_artifact_margin,
            min_disagreement_reduction=min_disagreement_reduction,
        )
        if selected is None:
            all_feasible = False
            selected_weight = 0.0
            selection_details[str(held_out)] = {
                "training_split_feasible": False,
                "selected_weight_c": selected_weight,
                "fallback": "locked_baseline",
            }
        else:
            selected_weight = float(selected["weight_c"])
            selection_details[str(held_out)] = {
                "training_split_feasible": True,
                "selected_weight_c": selected_weight,
                "training_metrics": {
                    key: value for key, value in selected.items()
                    if key not in {"age_bins"}
                },
            }
        selected_weights[str(held_out)] = selected_weight
        held["blend_weight_c"] = selected_weight
        held["blend_clean"] = (
            (1.0 - selected_weight) * held["base_clean"]
            + selected_weight * held["c_clean"]
        )
        held["blend_artifact"] = (
            (1.0 - selected_weight) * held["base_artifact"]
            + selected_weight * held["c_artifact"]
        )
        prediction_parts.append(held)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    candidate = predictions.copy()
    candidate["c_clean"] = candidate["blend_clean"]
    candidate["c_artifact"] = candidate["blend_artifact"]
    summary = analyze_weight(candidate, 1.0)
    summary["constraints_pass_on_pooled_heldout"] = _is_feasible(
        summary, clean_margin=clean_margin,
        age_clean_margin=age_clean_margin,
        age_artifact_margin=age_artifact_margin,
        min_disagreement_reduction=min_disagreement_reduction,
    )
    return {
        "all_training_splits_feasible": all_feasible,
        "selected_weights": selected_weights,
        "selection_details": selection_details,
        "summary": summary,
        "predictions": predictions,
    }


def _paired_ci(delta: np.ndarray, repetitions: int, seed: int) -> list[float]:
    if len(delta) == 0:
        raise ValueError("Không thể bootstrap nhóm rỗng")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        values[start:start + size] = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def subgroup_bootstrap(
    frame: pd.DataFrame, *, repetitions: int = 20000, seed: int = 20260914,
) -> dict:
    """Paired candidate-minus-baseline MAE intervals by age and sex."""
    _validate(frame)
    if repetitions <= 0:
        raise ValueError("repetitions phải > 0")
    work = frame.copy()
    work["age_bin"] = _age_bins(work)
    output: dict[str, dict] = {"age_bin": {}, "sex": {}}
    offset = 0
    for group_type, labels in (("age_bin", AGE_LABELS), ("sex", sorted(work["sex"].astype(str).unique()))):
        values = work[group_type].astype(str)
        for label in labels:
            group = work[values == str(label)]
            if group.empty:
                continue
            y = group["target_months"].to_numpy(dtype=np.float64)
            clean_delta = (
                np.abs(group["c_clean"].to_numpy(dtype=np.float64) - y)
                - np.abs(group["base_clean"].to_numpy(dtype=np.float64) - y)
            )
            artifact_delta = (
                np.abs(group["c_artifact"].to_numpy(dtype=np.float64) - y)
                - np.abs(group["base_artifact"].to_numpy(dtype=np.float64) - y)
            )
            output[group_type][str(label)] = {
                "count": int(len(group)),
                "clean_delta": float(clean_delta.mean()),
                "clean_delta_ci_95": _paired_ci(
                    clean_delta, repetitions, seed + offset
                ),
                "artifact_delta": float(artifact_delta.mean()),
                "artifact_delta_ci_95": _paired_ci(
                    artifact_delta, repetitions, seed + 100 + offset
                ),
            }
            offset += 1
    return output


def run_analysis(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT,
    *, repetitions: int = 20000,
) -> dict:
    frame = pd.read_csv(input_path)
    _validate(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = np.linspace(0.0, 1.0, 101)
    constraints = {
        "clean_margin": 0.05,
        "age_clean_margin": 0.20,
        "age_artifact_margin": 0.0,
        "min_disagreement_reduction": 0.40,
    }
    full_grid, full_selected = _search(frame, weights, **constraints)
    grid_rows = [
        {key: value for key, value in row.items() if key != "age_bins"}
        for row in full_grid
    ]
    grid_path = output_dir / "pilot_c_blend_grid_exploratory.csv"
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)

    crossfit = crossfit_weights(frame, weights=weights, **constraints)
    predictions = crossfit.pop("predictions")
    prediction_path = output_dir / "pilot_c_blend_crossfit_predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    pilot_bootstrap = subgroup_bootstrap(
        frame, repetitions=repetitions, seed=20260914
    )
    blend_frame = predictions.copy()
    blend_frame["c_clean"] = blend_frame["blend_clean"]
    blend_frame["c_artifact"] = blend_frame["blend_artifact"]
    blend_bootstrap = subgroup_bootstrap(
        blend_frame, repetitions=repetitions, seed=20261014
    )
    report = {
        "protocol": (
            "Five-fold OOF only; global baseline/Pilot-C blend; weight for each "
            "held-out fold selected on the other four folds"
        ),
        "test_accessed": False,
        "count": int(len(frame)),
        "constraints": constraints,
        "full_oof_grid_is_exploratory": True,
        "full_oof_best_feasible": None if full_selected is None else {
            key: value for key, value in full_selected.items() if key != "age_bins"
        },
        "crossfit": crossfit,
        "pilot_c_subgroup_bootstrap": pilot_bootstrap,
        "crossfit_blend_subgroup_bootstrap": blend_bootstrap,
        "files": {
            "grid_csv": str(grid_path),
            "crossfit_predictions_csv": str(prediction_path),
        },
    }
    report_path = output_dir / "pilot_c_blend_crossfit_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SAVED: {grid_path}")
    print(f"SAVED: {prediction_path}")
    print(f"SAVED: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-fitted Pilot C blend and subgroup bootstrap"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()
    run_analysis(
        args.input, args.output_dir,
        repetitions=args.bootstrap_repetitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
