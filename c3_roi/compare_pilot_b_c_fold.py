"""Paired comparison of Pilot B, Pilot C and baseline on one locked fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs"
)
DEFAULT_OOF = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/OOF/"
    "C3_Z26_C3_ROI_V2_PILOT_C_OOF_predictions.csv"
)
AGE_EDGES = (0, 60, 120, 180, 229)
AGE_LABELS = ("0-59", "60-119", "120-179", "180-228")


def _paired_ci(delta: np.ndarray, repetitions: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        values[start:start + size] = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def _comparison(
    target: np.ndarray, reference: np.ndarray, candidate: np.ndarray,
    *, repetitions: int, seed: int,
) -> dict[str, Any]:
    reference_error = np.abs(reference - target)
    candidate_error = np.abs(candidate - target)
    delta = candidate_error - reference_error
    return {
        "count": int(len(target)),
        "reference_mae": float(reference_error.mean()),
        "candidate_mae": float(candidate_error.mean()),
        "delta_candidate_minus_reference": float(delta.mean()),
        "gain": float(-delta.mean()),
        "delta_ci_95": _paired_ci(delta, repetitions, seed),
    }


def compare_frame(
    frame: pd.DataFrame, *, repetitions: int = 20000, seed: int = 20260914,
) -> dict[str, Any]:
    required = {
        "target_months", "base_clean", "base_artifact",
        "b_clean", "b_artifact", "c_clean", "c_artifact", "age_bin",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Du lieu thieu cot: {sorted(missing)}")
    output: dict[str, Any] = {"overall": {}, "age_bin": {}}
    comparisons = (
        ("b_vs_baseline", "base", "b"),
        ("c_vs_baseline", "base", "c"),
        ("c_vs_b", "b", "c"),
    )
    for view_offset, view in enumerate(("clean", "artifact")):
        y = frame["target_months"].to_numpy(dtype=np.float64)
        for comp_offset, (name, reference, candidate) in enumerate(comparisons):
            output["overall"][f"{name}_{view}"] = _comparison(
                y,
                frame[f"{reference}_{view}"].to_numpy(dtype=np.float64),
                frame[f"{candidate}_{view}"].to_numpy(dtype=np.float64),
                repetitions=repetitions,
                seed=seed + view_offset * 100 + comp_offset,
            )
    for age_offset, label in enumerate(AGE_LABELS):
        group = frame[frame["age_bin"] == label]
        if group.empty:
            continue
        output["age_bin"][label] = {}
        y = group["target_months"].to_numpy(dtype=np.float64)
        for view_offset, view in enumerate(("clean", "artifact")):
            for comp_offset, (name, reference, candidate) in enumerate(comparisons):
                output["age_bin"][label][f"{name}_{view}"] = _comparison(
                    y,
                    group[f"{reference}_{view}"].to_numpy(dtype=np.float64),
                    group[f"{candidate}_{view}"].to_numpy(dtype=np.float64),
                    repetitions=repetitions,
                    seed=seed + 1000 + age_offset * 100 + view_offset * 10 + comp_offset,
                )
    c_vs_b_clean = output["overall"]["c_vs_b_clean"]
    c_vs_b_artifact = output["overall"]["c_vs_b_artifact"]
    output["consistency_diagnosis"] = {
        "c_adds_significant_artifact_gain_over_b": bool(
            c_vs_b_artifact["gain"] >= 0.10
            and c_vs_b_artifact["delta_ci_95"][1] < 0.0
        ),
        "c_significantly_harms_clean_vs_b": bool(
            c_vs_b_clean["delta_ci_95"][0] > 0.0
        ),
        "interpretation_rule": (
            "B and C share artifact augmentation; their only scientific "
            "difference is consistency_weight 0.00 versus 0.30."
        ),
    }
    return output


def run_comparison(
    *, fold: int = 5, run_root: Path = DEFAULT_RUN_ROOT,
    oof_path: Path = DEFAULT_OOF, repetitions: int = 20000,
) -> dict[str, Any]:
    b_run = run_root / f"C3_Z26_C3_ROI_V2_PILOT_B_V2_FOLD_{fold}_SEED_42"
    c_run = run_root / f"C3_Z26_C3_ROI_V2_PILOT_C_V2_FOLD_{fold}_SEED_42"
    b_path = b_run / "artifact_robustness_predictions.csv"
    c_path = c_run / "artifact_robustness_predictions.csv"
    for path in (b_path, c_path, oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Thieu file: {path}")
    rename = {
        "clean_prediction_months": "clean",
        "artifact_prediction_months": "artifact",
    }
    b = pd.read_csv(b_path).rename(columns={key: f"b_{value}" for key, value in rename.items()})
    c = pd.read_csv(c_path).rename(columns={key: f"c_eval_{value}" for key, value in rename.items()})
    oof = pd.read_csv(oof_path)
    oof = oof[oof["fold"] == fold].copy()
    for part in (b, c, oof):
        part["image_id"] = part["image_id"].astype(str)
    frame = oof.merge(
        b[["image_id", "b_clean", "b_artifact"]],
        on="image_id", validate="one_to_one",
    ).merge(
        c[["image_id", "c_eval_clean", "c_eval_artifact"]],
        on="image_id", validate="one_to_one",
    )
    max_c_mismatch = max(
        float(np.abs(frame["c_clean"] - frame["c_eval_clean"]).max()),
        float(np.abs(frame["c_artifact"] - frame["c_eval_artifact"]).max()),
    )
    if max_c_mismatch > 0.02:
        raise RuntimeError(f"Pilot C integrity FAIL: mismatch={max_c_mismatch:.6f}")
    frame["age_bin"] = pd.cut(
        frame["target_months"], AGE_EDGES, right=False, labels=AGE_LABELS,
    ).astype(str)
    result = compare_frame(frame, repetitions=repetitions)
    report = {
        "protocol": (
            f"Fold {fold} paired validation only; baseline vs Pilot B vs Pilot C; "
            "no 200-image test access"
        ),
        "test_accessed": False,
        "fold": fold,
        "count": int(len(frame)),
        "pilot_c_integrity_max_mismatch_months": max_c_mismatch,
        **result,
    }
    output_dir = b_run / "comparison_vs_c_and_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"pilot_b_c_baseline_fold_{fold}_predictions.csv"
    report_path = output_dir / f"pilot_b_c_baseline_fold_{fold}_report.json"
    frame.to_csv(prediction_path, index=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SAVED: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pilot B and C on one fold")
    parser.add_argument("--fold", type=int, default=5, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()
    run_comparison(
        fold=args.fold, run_root=args.run_root, oof_path=args.oof,
        repetitions=args.bootstrap_repetitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
