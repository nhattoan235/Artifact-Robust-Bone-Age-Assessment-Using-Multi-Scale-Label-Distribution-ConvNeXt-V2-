"""Aggregate C3-ROI and E1 fold predictions without touching test labels."""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
C3_ROOT = ROOT / "c3_roi" / "runs" / "C3_ROI_V1"
E1_ROOT = ROOT / "p7_results" / "results"
OUT = ROOT / "c3_roi" / "outputs" / "C3_ROI_V1_OOF"


def load(root: Path, prefix: str):
    rows = []
    by_id = {}
    for fold in range(1, 6):
        path = root / f"{prefix}{fold}" / "val_predictions_best.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["fold"] = fold
                row["target_months"] = float(row["target_months"])
                row["prediction_months"] = float(row["prediction_months"])
                by_id[row["image_id"]] = row
                rows.append(row)
    return rows, by_id


def metrics(rows, prediction_key="prediction_months"):
    errors = [abs(r["target_months"] - r[prediction_key]) for r in rows]
    sq = [(r["target_months"] - r[prediction_key]) ** 2 for r in rows]
    signed = [r[prediction_key] - r["target_months"] for r in rows]
    return {
        "count": len(rows),
        "mae_months": sum(errors) / len(errors),
        "rmse_months": math.sqrt(sum(sq) / len(sq)),
        "median_absolute_error_months": sorted(errors)[len(errors) // 2],
        "accuracy_within_6_months": sum(e <= 6 for e in errors) / len(errors),
        "accuracy_within_12_months": sum(e <= 12 for e in errors) / len(errors),
        "accuracy_within_18_months": sum(e <= 18 for e in errors) / len(errors),
        "signed_bias_months": sum(signed) / len(signed),
    }


def grouped(rows, key):
    out = {}
    for row in rows:
        out.setdefault(key(row), []).append(row)
    return {str(name): metrics(group) for name, group in sorted(out.items(), key=lambda item: str(item[0]))}


def bootstrap_mae(rows, prediction_key, seed=42, n=2000):
    rng = random.Random(seed)
    errors = [abs(r["target_months"] - r[prediction_key]) for r in rows]
    values = []
    for _ in range(n):
        values.append(sum(rng.choice(errors) for _ in errors) / len(errors))
    values.sort()
    return {"lower_95": values[int(0.025 * n)], "upper_95": values[int(0.975 * n) - 1]}


def bootstrap_delta(rows, left_key, right_key, seed=42, n=2000):
    rng = random.Random(seed)
    deltas = [abs(r["target_months"] - r[left_key]) - abs(r["target_months"] - r[right_key]) for r in rows]
    values = []
    for _ in range(n):
        values.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    values.sort()
    return {"estimate": sum(deltas) / len(deltas), "lower_95": values[int(0.025 * n)], "upper_95": values[int(0.975 * n) - 1]}


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den


def main():
    c3_rows, c3 = load(C3_ROOT, "C3_ROI_V1_FOLD_")
    e1_rows, e1 = load(E1_ROOT, "P7_FINAL_V3_FOLD_")
    if set(c3) != set(e1):
        raise SystemExit("OOF ID mismatch between C3 and E1")
    if len(c3) != 14036 or len(c3) != len(set(c3)):
        raise SystemExit("C3 OOF count/uniqueness audit failed")

    rows = []
    for image_id in sorted(c3, key=lambda value: int(value)):
        a, b = c3[image_id], e1[image_id]
        if a["target_months"] != b["target_months"] or a["sex"] != b["sex"]:
            raise SystemExit(f"target/sex mismatch for {image_id}")
        row = dict(a)
        row["e1_prediction_months"] = b["prediction_months"]
        row["ensemble_50_50_prediction_months"] = (a["prediction_months"] + b["prediction_months"]) / 2
        row["ensemble_50_50_absolute_error"] = abs(row["target_months"] - row["ensemble_50_50_prediction_months"])
        rows.append(row)

    def as_ensemble(row):
        copy = dict(row)
        copy["prediction_months"] = row["ensemble_50_50_prediction_months"]
        return copy

    report = {
        "protocol": "development OOF only; RSNA test not used",
        "c3_run": "C3_ROI_V1",
        "baseline": "E1-full / P7_FINAL_V3",
        "audit": {"c3_rows": len(c3_rows), "e1_rows": len(e1_rows), "unique_ids": len(c3), "id_sets_equal": True, "target_and_sex_match": True},
        "c3": {"overall": metrics(c3_rows), "by_fold": {str(f): metrics([r for r in c3_rows if r["fold"] == f]) for f in range(1, 6)}, "by_sex": grouped(c3_rows, lambda r: r["sex"])},
        "e1": {"overall": metrics(e1_rows), "by_fold": {str(f): metrics([r for r in e1_rows if r["fold"] == f]) for f in range(1, 6)}, "by_sex": grouped(e1_rows, lambda r: r["sex"])},
        "ensemble_50_50": {
            "overall": metrics([as_ensemble(r) for r in rows]),
            "by_fold": {str(f): metrics([as_ensemble(r) for r in rows if r["fold"] == f]) for f in range(1, 6)},
            "by_sex": grouped([as_ensemble(r) for r in rows], lambda r: r["sex"]),
            "by_age_bin": grouped([as_ensemble(r) for r in rows], lambda r: "0-59" if r["target_months"] < 60 else "60-119" if r["target_months"] < 120 else "120-179" if r["target_months"] < 180 else "180-228"),
        },
        "complementarity": {
            "prediction_pearson_c3_vs_e1": pearson([r["prediction_months"] for r in rows], [r["e1_prediction_months"] for r in rows]),
            "paired_mae_delta_c3_minus_e1_months": metrics(c3_rows)["mae_months"] - metrics(e1_rows)["mae_months"],
            "paired_mae_delta_ensemble_minus_e1_months": metrics([as_ensemble(r) for r in rows])["mae_months"] - metrics(e1_rows)["mae_months"],
        },
        "bootstrap_95_ci_mae": {
            "c3": bootstrap_mae(c3_rows, "prediction_months"),
            "e1": bootstrap_mae(e1_rows, "prediction_months"),
            "ensemble_50_50": bootstrap_mae([as_ensemble(r) for r in rows], "prediction_months"),
        },
        "paired_bootstrap_95_ci_delta_months": {
            "c3_minus_e1": bootstrap_delta(rows, "prediction_months", "e1_prediction_months"),
            "ensemble_50_50_minus_e1": bootstrap_delta(rows, "ensemble_50_50_prediction_months", "e1_prediction_months"),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "C3_ROI_V1_OOF_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with (OUT / "C3_ROI_V1_E1_ensemble_OOF_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["image_id", "target_months", "sex", "fold", "prediction_months", "e1_prediction_months", "ensemble_50_50_prediction_months", "ensemble_50_50_absolute_error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    print(json.dumps({"c3_mae": report["c3"]["overall"]["mae_months"], "e1_mae": report["e1"]["overall"]["mae_months"], "ensemble_mae": report["ensemble_50_50"]["overall"]["mae_months"], "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
