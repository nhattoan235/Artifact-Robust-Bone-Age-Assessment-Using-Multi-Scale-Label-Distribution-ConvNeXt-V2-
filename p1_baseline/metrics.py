from __future__ import annotations

import math
import statistics
from collections import defaultdict


def compute_metrics(records: list[dict], age_bins: list[int]) -> dict:
    targets = [float(r["target_months"]) for r in records]
    preds = [float(r["prediction_months"]) for r in records]
    errors = [abs(p - t) for p, t in zip(preds, targets)]
    squared = [(p - t) ** 2 for p, t in zip(preds, targets)]
    result = {
        "count": len(records),
        "mae": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean(squared)),
        "median_ae": statistics.median(errors),
        "accuracy_6m": statistics.fmean(e <= 6 for e in errors),
        "accuracy_12m": statistics.fmean(e <= 12 for e in errors),
        "accuracy_18m": statistics.fmean(e <= 18 for e in errors),
        "prediction_mean": statistics.fmean(preds),
        "prediction_std": statistics.stdev(preds) if len(preds) > 1 else 0.0,
        "prediction_min": min(preds),
        "prediction_max": max(preds),
    }
    sex_groups = defaultdict(list)
    prediction_groups = defaultdict(list)
    for record in records:
        sex_groups[record["sex"]].append(abs(record["prediction_months"] - record["target_months"]))
        prediction_groups[record["sex"]].append(float(record["prediction_months"]))
    result["mae_by_sex"] = {sex: statistics.fmean(values) for sex, values in sorted(sex_groups.items())}
    result["prediction_std_by_sex"] = {
        sex: statistics.stdev(values) if len(values) > 1 else 0.0
        for sex, values in sorted(prediction_groups.items())
    }
    by_bin: dict[str, list[float]] = defaultdict(list)
    for record in records:
        age = record["target_months"]
        for left, right in zip(age_bins[:-1], age_bins[1:]):
            if left <= age < right:
                by_bin[f"{left}-{right - 1}"] .append(abs(record["prediction_months"] - age))
                break
    result["mae_by_age_bin"] = {key: statistics.fmean(values) for key, values in by_bin.items()}
    return result
