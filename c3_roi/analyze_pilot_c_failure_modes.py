"""Diagnose where Pilot C helps or hurts using locked five-fold OOF predictions.

This analysis is validation-only.  It does not read the 200-image RSNA test set
and it does not tune a new blend weight.  The fixed blend is the previously
selected 27% baseline + 73% Pilot C candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "results/pilot_bc_20260914/oof/"
    "C3_Z26_C3_ROI_V2_PILOT_C_OOF_predictions.csv"
)
DEFAULT_OUTPUT = Path("results/pilot_bc_20260914/failure_analysis")
AGE_EDGES = (0, 60, 120, 180, 229)
AGE_LABELS = ("0-59", "60-119", "120-179", "180-228")
REQUIRED_COLUMNS = {
    "image_id", "sex", "target_months", "fold",
    "base_clean", "c_clean", "base_artifact", "c_artifact",
}


def _validate(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"OOF thieu cot: {sorted(missing)}")
    if frame.empty:
        raise ValueError("OOF rong")
    if frame["image_id"].astype(str).duplicated().any():
        raise ValueError("OOF co image_id trung")
    for column in REQUIRED_COLUMNS - {"image_id", "sex"}:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"OOF cot {column} co NaN/Inf")


def _paired_ci(delta: np.ndarray, repetitions: int, seed: int) -> list[float]:
    if len(delta) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sampled = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        sampled[start:start + size] = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(sampled, (0.025, 0.975))]


def prepare_predictions(frame: pd.DataFrame, weight_c: float = 0.73) -> pd.DataFrame:
    """Add fixed-blend predictions and per-image error contributions."""
    _validate(frame)
    if not 0.0 <= weight_c <= 1.0:
        raise ValueError("weight_c phai nam trong [0, 1]")
    work = frame.copy()
    work["image_id"] = work["image_id"].astype(str)
    work["age_bin"] = pd.cut(
        work["target_months"], AGE_EDGES, right=False, labels=AGE_LABELS,
    ).astype(str)
    work["blend_clean"] = (
        (1.0 - weight_c) * work["base_clean"] + weight_c * work["c_clean"]
    )
    work["blend_artifact"] = (
        (1.0 - weight_c) * work["base_artifact"]
        + weight_c * work["c_artifact"]
    )
    for model in ("base", "c", "blend"):
        for view in ("clean", "artifact"):
            prediction = f"{model}_{view}"
            work[f"{prediction}_signed_error"] = (
                work[prediction] - work["target_months"]
            )
            work[f"{prediction}_absolute_error"] = work[
                f"{prediction}_signed_error"
            ].abs()
        work[f"{model}_disagreement"] = (
            work[f"{model}_artifact"] - work[f"{model}_clean"]
        ).abs()
    for model in ("c", "blend"):
        for view in ("clean", "artifact"):
            work[f"{model}_{view}_delta"] = (
                work[f"{model}_{view}_absolute_error"]
                - work[f"base_{view}_absolute_error"]
            )
    return work


def _group_row(
    group: pd.DataFrame, *, group_type: str, group_value: str,
    repetitions: int, seed: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "group_type": group_type,
        "group_value": str(group_value),
        "count": int(len(group)),
        "target_mean": float(group["target_months"].mean()),
        "female_fraction": float((group["sex"].astype(str) == "F").mean()),
    }
    for model in ("base", "c", "blend"):
        for view in ("clean", "artifact"):
            row[f"{model}_{view}_mae"] = float(
                group[f"{model}_{view}_absolute_error"].mean()
            )
            row[f"{model}_{view}_bias"] = float(
                group[f"{model}_{view}_signed_error"].mean()
            )
        row[f"{model}_disagreement"] = float(
            group[f"{model}_disagreement"].mean()
        )
    for model_offset, model in enumerate(("c", "blend")):
        for view_offset, view in enumerate(("clean", "artifact")):
            delta = group[f"{model}_{view}_delta"].to_numpy(dtype=np.float64)
            row[f"{model}_{view}_delta"] = float(delta.mean())
            interval = _paired_ci(
                delta, repetitions, seed + model_offset * 10 + view_offset,
            )
            row[f"{model}_{view}_delta_ci_low"] = interval[0]
            row[f"{model}_{view}_delta_ci_high"] = interval[1]
    return row


def build_group_summary(
    frame: pd.DataFrame, *, repetitions: int = 20000, seed: int = 20260914,
) -> pd.DataFrame:
    """Summarize overall, fold, age, fold-age and sex failure modes."""
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
    groups.extend(
        ("fold", str(key), group) for key, group in frame.groupby("fold", observed=True)
    )
    groups.extend(
        ("age_bin", str(key), group)
        for key, group in frame.groupby("age_bin", observed=True)
    )
    groups.extend(
        ("fold_age", f"fold_{fold}|{age}", group)
        for (fold, age), group in frame.groupby(["fold", "age_bin"], observed=True)
    )
    groups.extend(
        ("sex", str(key), group) for key, group in frame.groupby("sex", observed=True)
    )
    for offset, (group_type, value, group) in enumerate(groups):
        rows.append(_group_row(
            group, group_type=group_type, group_value=value,
            repetitions=repetitions, seed=seed + offset * 100,
        ))
    return pd.DataFrame(rows)


def _transition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in (6.0, 12.0):
        for model in ("c", "blend"):
            for view in ("clean", "artifact"):
                base_ok = frame[f"base_{view}_absolute_error"] <= threshold
                candidate_ok = frame[f"{model}_{view}_absolute_error"] <= threshold
                rows.append({
                    "threshold_months": threshold,
                    "model": model,
                    "view": view,
                    "baseline_ok_candidate_bad": int((base_ok & ~candidate_ok).sum()),
                    "baseline_bad_candidate_ok": int((~base_ok & candidate_ok).sum()),
                    "both_ok": int((base_ok & candidate_ok).sum()),
                    "both_bad": int((~base_ok & ~candidate_ok).sum()),
                })
    return pd.DataFrame(rows)


def _fold5_concentration(frame: pd.DataFrame) -> dict[str, object]:
    fold = frame[frame["fold"] == 5].copy()
    output: dict[str, object] = {"count": int(len(fold))}
    for model in ("c", "blend"):
        delta = fold[f"{model}_clean_delta"].to_numpy(dtype=np.float64)
        sorted_positive = np.sort(delta[delta > 0])[::-1]
        positive_sum = float(sorted_positive.sum())
        item: dict[str, object] = {
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "worsened_fraction": float((delta > 0).mean()),
            "improved_fraction": float((delta < 0).mean()),
            "positive_error_sum": positive_sum,
            "negative_error_sum": float(delta[delta < 0].sum()),
        }
        for share in (0.50, 0.80):
            if positive_sum <= 0:
                count = 0
            else:
                count = int(np.searchsorted(
                    np.cumsum(sorted_positive), share * positive_sum,
                ) + 1)
            item[f"images_for_{int(share * 100)}pct_positive_error"] = count
            item[f"fraction_for_{int(share * 100)}pct_positive_error"] = (
                float(count / len(fold))
            )
        for trim_fraction in (0.01, 0.05):
            cutoff = int(np.floor(len(delta) * trim_fraction))
            keep = np.argsort(np.abs(delta))[:len(delta) - cutoff]
            item[f"trimmed_{int(trim_fraction * 100)}pct_mean_delta"] = float(
                delta[keep].mean()
            )
        output[model] = item
    return output


def run_analysis(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT,
    *, weight_c: float = 0.73, repetitions: int = 20000,
) -> dict[str, object]:
    raw = pd.read_csv(input_path)
    frame = prepare_predictions(raw, weight_c)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_group_summary(frame, repetitions=repetitions)
    summary_path = output_dir / "fold_age_failure_summary.csv"
    summary.to_csv(summary_path, index=False)

    transition = _transition_summary(frame)
    transition_path = output_dir / "error_threshold_transitions.csv"
    transition.to_csv(transition_path, index=False)

    worst_columns = [
        "image_id", "fold", "sex", "age_bin", "target_months",
        "base_clean", "c_clean", "blend_clean",
        "base_clean_absolute_error", "c_clean_absolute_error",
        "blend_clean_absolute_error", "c_clean_delta", "blend_clean_delta",
        "base_artifact", "c_artifact", "blend_artifact",
        "c_artifact_delta", "blend_artifact_delta",
    ]
    worst = frame.sort_values("blend_clean_delta", ascending=False).head(200)
    worst_path = output_dir / "worst_200_clean_regressions.csv"
    worst[worst_columns].to_csv(worst_path, index=False)

    fold_rows = summary[summary["group_type"] == "fold"].copy()
    worst_fold_row = fold_rows.loc[fold_rows["blend_clean_delta"].idxmax()]
    fold_age_rows = summary[summary["group_type"] == "fold_age"].copy()
    worst_fold_age_row = fold_age_rows.loc[
        fold_age_rows["blend_clean_delta"].idxmax()
    ]
    report: dict[str, object] = {
        "protocol": (
            "Locked five-fold validation OOF only; no RSNA 200-image test access; "
            f"fixed blend={(1.0 - weight_c):.2f} baseline + {weight_c:.2f} Pilot C"
        ),
        "test_accessed": False,
        "count": int(len(frame)),
        "weight_c": float(weight_c),
        "worst_clean_fold": {
            "fold": int(worst_fold_row["group_value"]),
            "count": int(worst_fold_row["count"]),
            "pilot_c_delta": float(worst_fold_row["c_clean_delta"]),
            "blend_delta": float(worst_fold_row["blend_clean_delta"]),
            "blend_delta_ci_95": [
                float(worst_fold_row["blend_clean_delta_ci_low"]),
                float(worst_fold_row["blend_clean_delta_ci_high"]),
            ],
        },
        "worst_fold_age_group": {
            "group": str(worst_fold_age_row["group_value"]),
            "count": int(worst_fold_age_row["count"]),
            "pilot_c_delta": float(worst_fold_age_row["c_clean_delta"]),
            "blend_delta": float(worst_fold_age_row["blend_clean_delta"]),
            "blend_delta_ci_95": [
                float(worst_fold_age_row["blend_clean_delta_ci_low"]),
                float(worst_fold_age_row["blend_clean_delta_ci_high"]),
            ],
        },
        "fold5_concentration": _fold5_concentration(frame),
        "next_control_experiment": {
            "pilot": "B",
            "fold": 5,
            "reason": (
                "Fold 5 is the largest fixed-blend clean regression. Training "
                "Pilot B on the same locked fold separates artifact augmentation "
                "from Pilot C's consistency penalty."
            ),
        },
        "files": {
            "group_summary": str(summary_path),
            "threshold_transitions": str(transition_path),
            "worst_regressions": str(worst_path),
        },
    }
    report_path = output_dir / "failure_analysis_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Pilot C OOF failure modes")
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
