from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


EXPECTED_ROWS = 14_036
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 2026
AGE_BINS = ((0, 59), (60, 119), (120, 179), (180, 228))
AGE_LABELS = tuple(f"{low}-{high}" for low, high in AGE_BINS)
VIEW_COLUMNS = tuple(
    f"prediction_rot_{rotation}_{suffix}_months"
    for rotation in (-10, -5, 0, 5, 10)
    for suffix in ("no_flip", "flip")
)
REQUIRED_COLUMNS = {
    "image_id",
    "fold",
    "target_months",
    "sex",
    "raw_prediction_months",
    "tta_prediction_months",
    "tta_std_months",
    *VIEW_COLUMNS,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def age_bin_label(age: float) -> str:
    for low, high in AGE_BINS:
        if low <= age <= high:
            return f"{low}-{high}"
    raise ValueError(f"Tuổi ngoài miền khóa 0-228: {age}")


def validate_input(frame: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> dict:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(f"Số dòng={len(frame)}, kỳ vọng={expected_rows}")
    if frame["image_id"].nunique() != len(frame):
        raise ValueError("image_id không duy nhất")
    if set(frame["fold"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("fold phải phủ đúng 1-5")
    if set(frame["sex"].astype(str)) != {"F", "M"}:
        raise ValueError("sex phải chỉ gồm F/M")

    numeric_columns = [
        "target_months",
        "raw_prediction_months",
        "tta_prediction_months",
        "tta_std_months",
        *VIEW_COLUMNS,
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Prediction/target/disagreement chứa NaN hoặc Inf")
    targets = frame["target_months"].to_numpy(dtype=np.float64)
    if np.any((targets < 0) | (targets > 228)):
        raise ValueError("target_months ngoài miền 0-228")
    if np.any(frame["tta_std_months"].to_numpy(dtype=np.float64) < 0):
        raise ValueError("tta_std_months âm")

    views = frame[list(VIEW_COLUMNS)].to_numpy(dtype=np.float64)
    stored_mean = frame["tta_prediction_months"].to_numpy(dtype=np.float64)
    stored_std = frame["tta_std_months"].to_numpy(dtype=np.float64)
    mean_diff = np.abs(views.mean(axis=1) - stored_mean)
    std_diff = np.abs(views.std(axis=1, ddof=0) - stored_std)
    max_mean_diff = float(mean_diff.max())
    max_std_diff = float(std_diff.max())
    if max_mean_diff > 1e-8:
        raise ValueError(f"TTA mean không khớp 10 views: max diff={max_mean_diff}")
    if max_std_diff > 1e-8:
        raise ValueError(f"TTA std không khớp 10 views: max diff={max_std_diff}")

    return {
        "row_count": int(len(frame)),
        "unique_image_ids": int(frame["image_id"].nunique()),
        "fold_counts": {
            str(int(key)): int(value)
            for key, value in frame["fold"].value_counts().sort_index().items()
        },
        "sex_counts": {
            str(key): int(value)
            for key, value in frame["sex"].value_counts().sort_index().items()
        },
        "max_tta_mean_recompute_diff": max_mean_diff,
        "max_tta_std_recompute_diff": max_std_diff,
    }


def _percentile_ci(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def bootstrap_mean_ci(
    values: np.ndarray, samples: int, seed: int, chunk_size: int = 128
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        output[start:stop] = values[index].mean(axis=1)
    return _percentile_ci(output)


def bootstrap_rank_correlation_ci(
    x: np.ndarray, y: np.ndarray, samples: int, seed: int, chunk_size: int = 64
) -> list[float]:
    """Paired bootstrap of Pearson correlation on full-sample rank scores."""
    x_rank = rankdata(np.asarray(x, dtype=np.float64), method="average")
    y_rank = rankdata(np.asarray(y, dtype=np.float64), method="average")
    if len(x_rank) < 3 or np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        index = rng.integers(0, len(x_rank), size=(stop - start, len(x_rank)))
        xs = x_rank[index]
        ys = y_rank[index]
        xs = xs - xs.mean(axis=1, keepdims=True)
        ys = ys - ys.mean(axis=1, keepdims=True)
        denominator = np.sqrt((xs * xs).sum(axis=1) * (ys * ys).sum(axis=1))
        output[start:stop] = (xs * ys).sum(axis=1) / denominator
    return _percentile_ci(output)


def benjamini_hochberg(pvalues: Iterable[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        running = min(running, float(values[index] * count / rank))
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def correlation_strength(rho: float) -> str:
    value = abs(rho)
    if value < 0.10:
        return "very_weak"
    if value < 0.30:
        return "weak"
    if value < 0.50:
        return "moderate"
    return "strong"


def group_metrics(
    frame: pd.DataFrame,
    group_type: str,
    label: str,
    sex: str,
    age_bin: str,
    bootstrap: int,
    seed: int,
) -> dict:
    raw_error = frame["raw_prediction_months"].to_numpy() - frame["target_months"].to_numpy()
    tta_error = frame["tta_prediction_months"].to_numpy() - frame["target_months"].to_numpy()
    raw_absolute = np.abs(raw_error)
    tta_absolute = np.abs(tta_error)
    disagreement = frame["tta_std_months"].to_numpy()
    gain = raw_absolute - tta_absolute
    rho_result = spearmanr(disagreement, tta_absolute)
    rho = float(rho_result.statistic)
    pvalue = float(rho_result.pvalue)
    return {
        "group_type": group_type,
        "group": label,
        "sex": sex,
        "age_bin": age_bin,
        "count": int(len(frame)),
        "raw_mae": float(raw_absolute.mean()),
        "tta_mae": float(tta_absolute.mean()),
        "tta_gain_raw_mae_minus_tta_mae": float(gain.mean()),
        "tta_gain_bootstrap_95_ci": bootstrap_mean_ci(gain, bootstrap, seed),
        "tta_signed_bias": float(tta_error.mean()),
        "tta_error_rate_gt12": float((tta_absolute > 12).mean()),
        "tta_error_rate_gt18": float((tta_absolute > 18).mean()),
        "disagreement_mean": float(disagreement.mean()),
        "disagreement_median": float(np.median(disagreement)),
        "disagreement_p95": float(np.quantile(disagreement, 0.95)),
        "spearman_rho": rho,
        "spearman_strength": correlation_strength(rho),
        "spearman_pvalue": pvalue,
        "spearman_bootstrap_95_ci": bootstrap_rank_correlation_ci(
            disagreement, tta_absolute, bootstrap, seed + 1
        ),
        "auc_error_gt12": safe_auc(tta_absolute > 12, disagreement),
        "auc_error_gt18": safe_auc(tta_absolute > 18, disagreement),
    }


def build_groups(frame: pd.DataFrame, bootstrap: int, seed: int) -> list[dict]:
    groups = [group_metrics(frame, "overall", "overall", "ALL", "ALL", bootstrap, seed)]
    next_seed = seed + 100
    for sex in ("F", "M"):
        subset = frame[frame["sex"] == sex]
        groups.append(group_metrics(subset, "sex", sex, sex, "ALL", bootstrap, next_seed))
        next_seed += 100
    for sex in ("F", "M"):
        for age_bin in AGE_LABELS:
            subset = frame[(frame["sex"] == sex) & (frame["age_bin"] == age_bin)]
            groups.append(
                group_metrics(
                    subset,
                    "sex_age",
                    f"{sex}:{age_bin}",
                    sex,
                    age_bin,
                    bootstrap,
                    next_seed,
                )
            )
            next_seed += 100
    exploratory = [group for group in groups if group["group_type"] == "sex_age"]
    qvalues = benjamini_hochberg(group["spearman_pvalue"] for group in exploratory)
    for group, qvalue in zip(exploratory, qvalues):
        group["spearman_bh_qvalue"] = qvalue
    return groups


def quartile_metrics(frame: pd.DataFrame) -> list[dict]:
    ranked = frame["tta_std_months"].rank(method="first")
    quartile = pd.qcut(ranked, 4, labels=("Q1_low", "Q2", "Q3", "Q4_high"))
    rows = []
    for label in quartile.cat.categories:
        subset = frame[quartile == label]
        absolute = np.abs(subset["tta_prediction_months"] - subset["target_months"])
        rows.append(
            {
                "quartile": str(label),
                "count": int(len(subset)),
                "disagreement_min": float(subset["tta_std_months"].min()),
                "disagreement_max": float(subset["tta_std_months"].max()),
                "tta_mae": float(absolute.mean()),
                "error_rate_gt12": float((absolute > 12).mean()),
                "error_rate_gt18": float((absolute > 18).mean()),
            }
        )
    return rows


def risk_coverage_metrics(frame: pd.DataFrame) -> list[dict]:
    ordered = frame.sort_values(["tta_std_months", "image_id"], kind="stable")
    rows = []
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        keep = max(1, int(math.floor(len(ordered) * coverage)))
        subset = ordered.iloc[:keep]
        absolute = np.abs(subset["tta_prediction_months"] - subset["target_months"])
        rows.append(
            {
                "coverage": coverage,
                "retained_count": int(keep),
                "max_retained_disagreement": float(subset["tta_std_months"].max()),
                "tta_mae": float(absolute.mean()),
                "error_rate_gt12": float((absolute > 12).mean()),
                "error_rate_gt18": float((absolute > 18).mean()),
            }
        )
    return rows


def age_standardized_metrics(frame: pd.DataFrame) -> dict:
    weights = frame["age_bin"].value_counts(normalize=True).reindex(AGE_LABELS)
    report = {"overall_age_bin_weights": {key: float(value) for key, value in weights.items()}}
    for sex in ("F", "M"):
        subset = frame[frame["sex"] == sex]
        raw_values = []
        tta_values = []
        for label in AGE_LABELS:
            part = subset[subset["age_bin"] == label]
            raw_values.append(float(np.abs(part["raw_prediction_months"] - part["target_months"]).mean()))
            tta_values.append(float(np.abs(part["tta_prediction_months"] - part["target_months"]).mean()))
        report[sex] = {
            "raw_mae": float(np.dot(weights.to_numpy(), raw_values)),
            "tta_mae": float(np.dot(weights.to_numpy(), tta_values)),
        }
    return report


def analyze(
    input_path: Path,
    output_dir: Path,
    bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    expected_rows: int = EXPECTED_ROWS,
) -> dict:
    frame = pd.read_csv(input_path)
    integrity = validate_input(frame, expected_rows)
    frame = frame.copy()
    frame["age_bin"] = frame["target_months"].map(age_bin_label)
    groups = build_groups(frame, bootstrap, seed)
    quartiles = quartile_metrics(frame)
    risk_coverage = risk_coverage_metrics(frame)
    sex_age_groups = [group for group in groups if group["group_type"] == "sex_age"]
    primary = groups[0]
    primary_supported = primary["spearman_bootstrap_95_ci"][0] > 0
    report = {
        "status": "PASS",
        "test_accessed": False,
        "analysis_type": "post-hoc analysis of locked out-of-fold predictions",
        "protocol_path": "AI_Context/09_P12_UNCERTAINTY_PROTOCOL.md",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "bootstrap_samples": int(bootstrap),
        "bootstrap_seed": int(seed),
        "integrity": integrity,
        "primary": {
            "endpoint": "Spearman(tta_std_months, abs(tta_prediction-target))",
            "rho": primary["spearman_rho"],
            "strength": primary["spearman_strength"],
            "pvalue": primary["spearman_pvalue"],
            "bootstrap_95_ci": primary["spearman_bootstrap_95_ci"],
            "h4_association": "SUPPORTED" if primary_supported else "NOT_SUPPORTED",
        },
        "overall": primary,
        "groups": groups,
        "worst_sex_age_group_tta_mae": max(sex_age_groups, key=lambda item: item["tta_mae"]),
        "age_standardized_by_sex": age_standardized_metrics(frame),
        "disagreement_quartiles": quartiles,
        "risk_coverage": risk_coverage,
        "collapse_checks": {
            "raw_prediction_min": float(frame["raw_prediction_months"].min()),
            "raw_prediction_max": float(frame["raw_prediction_months"].max()),
            "raw_prediction_std": float(frame["raw_prediction_months"].std(ddof=0)),
            "tta_prediction_min": float(frame["tta_prediction_months"].min()),
            "tta_prediction_max": float(frame["tta_prediction_months"].max()),
            "tta_prediction_std": float(frame["tta_prediction_months"].std(ddof=0)),
        },
        "limitations": [
            "Disagreement is not calibrated clinical uncertainty.",
            "Thresholds and risk-coverage are descriptive on the same OOF set.",
            "Subgroup analyses are exploratory even with BH correction.",
            "External validation is required before clinical use.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(groups).to_csv(output_dir / "group_metrics.csv", index=False)
    pd.DataFrame(risk_coverage).to_csv(output_dir / "risk_coverage.csv", index=False)
    pd.DataFrame(quartiles).to_csv(output_dir / "disagreement_quartiles.csv", index=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "p9_inference/outputs/P9_I_TTA_BIAS_OOF/"
            "P9_I_TTA_BIAS_OOF_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("p12_uncertainty/outputs/P12_OOF_UNCERTAINTY"),
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()
    report = analyze(
        args.input, args.output_dir, args.bootstrap, args.seed, args.expected_rows
    )
    print(json.dumps({
        "status": report["status"],
        "primary": report["primary"],
        "overall_tta_mae": report["overall"]["tta_mae"],
        "overall_auc_gt12": report["overall"]["auc_error_gt12"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
