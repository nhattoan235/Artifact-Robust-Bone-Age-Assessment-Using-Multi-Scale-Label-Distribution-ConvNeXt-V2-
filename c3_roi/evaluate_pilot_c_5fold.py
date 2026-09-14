"""Aggregate Pilot C and locked C3-V2 baseline over five validation folds.

This module never reads the RSNA test set.  It consumes the paired clean and
synthetic mild-artifact predictions produced per fold, re-infers the locked
baseline artifact view when a cached baseline prediction is absent, and writes
one auditable pooled OOF report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # package import for local tests
    from .evaluate_pilot_bc import _predict, paired_bootstrap_ci
except ImportError:  # script import from the Colab bundle
    from evaluate_pilot_bc import _predict, paired_bootstrap_ci

from p1_baseline.config import load_config  # noqa: E402


DEFAULT_C_ROOT = Path("/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs")
DEFAULT_BASELINE_ROOT = Path("/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2/runs")
DEFAULT_BASELINE_CODE_ROOT = Path("/content/C3_Z26_C3_ROI_V2")
DEFAULT_OUTPUT = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/OOF"
)
FOLDS = (1, 2, 3, 4, 5)
TOTAL_OOF_COUNT = 14036
REQUIRED_C_COLUMNS = {
    "image_id", "sex", "target_months", "clean_prediction_months",
    "artifact_prediction_months",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(fold: int) -> str:
    return f"C3_Z26_C3_ROI_V2_PILOT_C_V2_FOLD_{fold}_SEED_42"


def _expected_count(fold: int) -> int:
    return 2808 if fold == 1 else 2807


def _validate_prediction_frame(
    frame: pd.DataFrame, *, fold: int, source: str, expected_columns: Iterable[str],
) -> None:
    missing = set(expected_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: thiếu cột {sorted(missing)}")
    if len(frame) != _expected_count(fold):
        raise ValueError(
            f"{source}: Fold {fold} cần {_expected_count(fold)} dòng, có {len(frame)}"
        )
    ids = frame["image_id"].astype(str)
    if ids.duplicated().any():
        duplicate = ids[ids.duplicated()].iloc[0]
        raise ValueError(f"{source}: trùng image_id={duplicate}")
    numeric = [column for column in expected_columns if column not in {"image_id", "sex"}]
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: cột {column} có NaN/Inf")


def read_c_predictions(path: Path, fold: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _validate_prediction_frame(
        frame, fold=fold, source=str(path), expected_columns=REQUIRED_C_COLUMNS
    )
    output = frame[
        [
            "image_id", "sex", "target_months",
            "clean_prediction_months", "artifact_prediction_months",
        ]
    ].copy()
    output["image_id"] = output["image_id"].astype(str)
    output["fold"] = fold
    return output.rename(
        columns={
            "clean_prediction_months": "c_clean",
            "artifact_prediction_months": "c_artifact",
        }
    )


def read_baseline_predictions(path: Path, fold: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "image_id", "sex", "target_months",
        "clean_prediction_months", "artifact_prediction_months",
    }
    _validate_prediction_frame(
        frame, fold=fold, source=str(path), expected_columns=required
    )
    output = frame[
        [
            "image_id", "sex", "target_months",
            "clean_prediction_months", "artifact_prediction_months",
        ]
    ].copy()
    output["image_id"] = output["image_id"].astype(str)
    return output.rename(
        columns={
            "clean_prediction_months": "base_clean",
            "artifact_prediction_months": "base_artifact",
        }
    )


def _merge_fold(c: pd.DataFrame, baseline: pd.DataFrame, fold: int) -> pd.DataFrame:
    if set(c["image_id"]) != set(baseline["image_id"]):
        raise ValueError(f"Fold {fold}: baseline/C image_id mismatch")
    merged = c.merge(
        baseline,
        on="image_id",
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    if not np.allclose(merged["target_months"], merged["target_months_baseline"]):
        raise ValueError(f"Fold {fold}: target mismatch giữa baseline và C")
    if merged["sex"].astype(str).tolist() != merged["sex_baseline"].astype(str).tolist():
        raise ValueError(f"Fold {fold}: sex mismatch giữa baseline và C")
    return merged[
        [
            "image_id", "sex", "target_months", "fold",
            "c_clean", "c_artifact", "base_clean", "base_artifact",
        ]
    ]


def merge_oof(c_frames: list[pd.DataFrame], baseline_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(c_frames) != len(baseline_frames):
        raise ValueError("Số frame Pilot C và baseline không khớp")
    if not c_frames:
        raise ValueError("Không có fold để ghép")
    folds = [int(frame["fold"].iloc[0]) for frame in c_frames]
    if len(set(folds)) != len(folds):
        raise ValueError(f"Trùng fold: {folds}")
    merged = pd.concat(
        [_merge_fold(c, b, fold) for c, b, fold in zip(c_frames, baseline_frames, folds)],
        ignore_index=True,
    )
    if merged["image_id"].astype(str).duplicated().any():
        raise ValueError("OOF có image_id xuất hiện ở nhiều fold")
    return merged


def _metrics(values: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    error = np.abs(values - target)
    return {
        "count": int(len(error)),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((values - target) ** 2))),
        "median_ae": float(np.median(error)),
    }


def _group_metrics(frame: pd.DataFrame, key: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value, group in frame.groupby(key, sort=True):
        y = group["target_months"].to_numpy(dtype=np.float64)
        c_clean = group["c_clean"].to_numpy(dtype=np.float64)
        c_art = group["c_artifact"].to_numpy(dtype=np.float64)
        b_clean = group["base_clean"].to_numpy(dtype=np.float64)
        b_art = group["base_artifact"].to_numpy(dtype=np.float64)
        output[str(value)] = {
            "count": int(len(group)),
            "c_clean_mae": float(np.abs(c_clean - y).mean()),
            "baseline_clean_mae": float(np.abs(b_clean - y).mean()),
            "c_artifact_mae": float(np.abs(c_art - y).mean()),
            "baseline_artifact_mae": float(np.abs(b_art - y).mean()),
            "clean_delta_c_minus_baseline": float(
                np.abs(c_clean - y).mean() - np.abs(b_clean - y).mean()
            ),
            "artifact_delta_c_minus_baseline": float(
                np.abs(c_art - y).mean() - np.abs(b_art - y).mean()
            ),
            "c_disagreement_mean": float(np.abs(c_art - c_clean).mean()),
            "baseline_disagreement_mean": float(np.abs(b_art - b_clean).mean()),
        }
    return output


def summarize_oof(
    frame: pd.DataFrame, *, repetitions: int = 20000, seed: int = 20260913,
) -> dict[str, Any]:
    y = frame["target_months"].to_numpy(dtype=np.float64)
    base_clean = frame["base_clean"].to_numpy(dtype=np.float64)
    base_artifact = frame["base_artifact"].to_numpy(dtype=np.float64)
    c_clean = frame["c_clean"].to_numpy(dtype=np.float64)
    c_artifact = frame["c_artifact"].to_numpy(dtype=np.float64)
    base_disagreement = np.abs(base_artifact - base_clean)
    c_disagreement = np.abs(c_artifact - c_clean)
    clean_ci = paired_bootstrap_ci(
        y, base_clean, c_clean, repetitions=repetitions, seed=seed
    )
    artifact_ci = paired_bootstrap_ci(
        y, base_artifact, c_artifact, repetitions=repetitions, seed=seed + 1
    )
    disagreement_delta = c_disagreement - base_disagreement
    rng = np.random.default_rng(seed + 2)
    bootstrap = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        indices = rng.integers(0, len(frame), size=(size, len(frame)))
        bootstrap[start:start + size] = disagreement_delta[indices].mean(axis=1)
    disagreement_ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()

    clean_delta = float(np.abs(c_clean - y).mean() - np.abs(base_clean - y).mean())
    artifact_delta = float(np.abs(c_artifact - y).mean() - np.abs(base_artifact - y).mean())
    disagreement_reduction = float(
        1.0 - c_disagreement.mean() / max(base_disagreement.mean(), 1e-12)
    )
    subgroup_sex = _group_metrics(frame.assign(sex=frame["sex"].astype(str)), "sex")
    age_labels = pd.cut(
        frame["target_months"], bins=[0, 60, 120, 180, 229],
        right=False, labels=["0-59", "60-119", "120-179", "180-228"],
    )
    subgroup_age = _group_metrics(frame.assign(age_bin=age_labels), "age_bin")
    clean_age_noninferior = all(
        row["clean_delta_c_minus_baseline"] <= 0.20
        for row in subgroup_age.values()
    )
    sex_artifact_improves = all(
        row["artifact_delta_c_minus_baseline"] < 0
        for row in subgroup_sex.values()
    )
    gates = {
        "clean_noninferiority_pass": (
            clean_delta <= 0.10 and float(clean_ci[1]) <= 0.10
        ),
        "artifact_superiority_pass": (
            -artifact_delta >= 0.30 and float(artifact_ci[1]) < 0.0
        ),
        "stability_pass": disagreement_reduction >= 0.40,
        "subgroup_pass": clean_age_noninferior and sex_artifact_improves,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "protocol": (
            "Pooled 5-fold validation only; Pilot C versus locked C3-V2; "
            "fixed clean and synthetic mild-artifact paired views"
        ),
        "test_accessed": False,
        "count": int(len(frame)),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
        "baseline_clean": _metrics(base_clean, y),
        "pilot_c_clean": _metrics(c_clean, y),
        "baseline_artifact": _metrics(base_artifact, y),
        "pilot_c_artifact": _metrics(c_artifact, y),
        "clean_delta_c_minus_baseline_months": clean_delta,
        "clean_paired_bootstrap_95_ci_months": [float(v) for v in clean_ci],
        "artifact_delta_c_minus_baseline_months": artifact_delta,
        "artifact_gain_months": -artifact_delta,
        "artifact_paired_bootstrap_95_ci_months": [float(v) for v in artifact_ci],
        "baseline_disagreement_mean_months": float(base_disagreement.mean()),
        "pilot_c_disagreement_mean_months": float(c_disagreement.mean()),
        "disagreement_reduction_fraction": disagreement_reduction,
        "disagreement_delta_paired_bootstrap_95_ci_months": [
            float(v) for v in disagreement_ci
        ],
        "subgroups_by_sex": subgroup_sex,
        "subgroups_by_age_bin": subgroup_age,
        "gates": gates,
    }


def _baseline_cache_path(output_dir: Path, fold: int) -> Path:
    return output_dir / f"baseline_fold_{fold}_artifact_predictions.csv"


def load_or_infer_baseline(
    fold: int, *, baseline_root: Path, baseline_code_root: Path, output_dir: Path,
) -> pd.DataFrame:
    cached = _baseline_cache_path(output_dir, fold)
    if cached.is_file():
        return read_baseline_predictions(cached, fold)
    config_path = baseline_code_root / "configs_t4_b36" / f"fold_{fold}.toml"
    checkpoint = baseline_root / f"C3_Z26_C3_ROI_V2_FOLD_{fold}" / "best_mae.ckpt"
    if not config_path.is_file():
        raise FileNotFoundError(f"Thiếu baseline config Fold {fold}: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Thiếu baseline checkpoint Fold {fold}: {checkpoint}")
    cfg = load_config(config_path)
    records = _predict(cfg, checkpoint)
    frame = pd.DataFrame(records)
    frame.to_csv(cached, index=False)
    return read_baseline_predictions(cached, fold)


def run_aggregation(
    *, c_root: Path = DEFAULT_C_ROOT, baseline_root: Path = DEFAULT_BASELINE_ROOT,
    baseline_code_root: Path = DEFAULT_BASELINE_CODE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT, folds: Iterable[int] = FOLDS,
    repetitions: int = 20000,
) -> dict[str, Any]:
    selected_folds = tuple(int(fold) for fold in folds)
    if not selected_folds or any(fold not in FOLDS for fold in selected_folds):
        raise ValueError("folds phải là một tập con không rỗng của 1..5")
    output_dir.mkdir(parents=True, exist_ok=True)
    c_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for fold in selected_folds:
        run_dir = c_root / _run_id(fold)
        c_path = run_dir / "artifact_robustness_predictions.csv"
        if not c_path.is_file():
            raise FileNotFoundError(f"Thiếu Pilot C prediction Fold {fold}: {c_path}")
        c_frame = read_c_predictions(c_path, fold)
        baseline_frame = load_or_infer_baseline(
            fold, baseline_root=baseline_root,
            baseline_code_root=baseline_code_root, output_dir=output_dir,
        )
        c_frames.append(c_frame)
        baseline_frames.append(baseline_frame)
        report = run_dir / "artifact_robustness_report.json"
        provenance.append({
            "fold": fold,
            "pilot_c_predictions": str(c_path),
            "pilot_c_predictions_sha256": _sha256(c_path),
            "pilot_c_report": str(report),
            "pilot_c_checkpoint": str(run_dir / "best_mae.ckpt"),
            "baseline_checkpoint": str(
                baseline_root / f"C3_Z26_C3_ROI_V2_FOLD_{fold}" / "best_mae.ckpt"
            ),
        })
    merged = merge_oof(c_frames, baseline_frames)
    expected = sum(_expected_count(fold) for fold in selected_folds)
    if len(merged) != expected:
        raise ValueError(f"OOF count sai: expected {expected}, got {len(merged)}")
    report = summarize_oof(merged, repetitions=repetitions)
    report["expected_count"] = expected
    report["provenance"] = provenance
    prediction_path = output_dir / "C3_Z26_C3_ROI_V2_PILOT_C_OOF_predictions.csv"
    report_path = output_dir / "C3_Z26_C3_ROI_V2_PILOT_C_OOF_report.json"
    subgroup_path = output_dir / "C3_Z26_C3_ROI_V2_PILOT_C_OOF_subgroups.csv"
    merged.to_csv(prediction_path, index=False)
    subgroup_rows = []
    for group_type, groups in (
        ("sex", report["subgroups_by_sex"]),
        ("age_bin", report["subgroups_by_age_bin"]),
    ):
        for group, values in groups.items():
            subgroup_rows.append({"group_type": group_type, "group": group, **values})
    pd.DataFrame(subgroup_rows).to_csv(subgroup_path, index=False)
    report["predictions_csv"] = str(prediction_path)
    report["subgroups_csv"] = str(subgroup_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SAVED: {prediction_path}")
    print(f"SAVED: {report_path}")
    print(f"SAVED: {subgroup_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Pilot C full-fold OOF")
    parser.add_argument("--c-root", type=Path, default=DEFAULT_C_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--baseline-code-root", type=Path, default=DEFAULT_BASELINE_CODE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()
    run_aggregation(
        c_root=args.c_root, baseline_root=args.baseline_root,
        baseline_code_root=args.baseline_code_root, output_dir=args.output_dir,
        folds=args.folds, repetitions=args.bootstrap_repetitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
