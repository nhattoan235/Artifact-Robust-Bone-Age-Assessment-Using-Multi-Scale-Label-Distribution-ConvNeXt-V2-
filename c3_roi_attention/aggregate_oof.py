"""Locked OOF comparison of C3-ROI + attention against C3-ROI."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def metrics(rows: list[dict[str, float]], prediction_key: str = "prediction_months") -> dict[str, float | int]:
    errors = [abs(float(row["target_months"]) - float(row[prediction_key])) for row in rows]
    squared = [(float(row["target_months"]) - float(row[prediction_key])) ** 2 for row in rows]
    return {
        "count": len(rows),
        "mae_months": sum(errors) / len(errors),
        "rmse_months": math.sqrt(sum(squared) / len(squared)),
        "median_absolute_error_months": statistics.median(errors),
        "accuracy_within_6_months": sum(error <= 6 for error in errors) / len(errors),
        "accuracy_within_12_months": sum(error <= 12 for error in errors) / len(errors),
    }


def paired_bootstrap_delta(
    rows: list[dict[str, float]], n: int = 5000, seed: int = 42,
) -> dict[str, float]:
    """Return attention MAE minus C3 MAE; negative values favor attention."""
    deltas = [
        abs(float(row["target_months"]) - float(row["attention_prediction_months"]))
        - abs(float(row["target_months"]) - float(row["c3_prediction_months"]))
        for row in rows
    ]
    rng = random.Random(seed)
    estimates = [sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(n)]
    estimates.sort()
    return {
        "estimate_months": sum(deltas) / len(deltas),
        "lower_95": estimates[int(0.025 * n)],
        "upper_95": estimates[int(0.975 * n) - 1],
    }


def load_oof(root: Path, prefix: str) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    for fold in range(1, 6):
        path = root / f"{prefix}{fold}" / "val_predictions_best.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Fold {fold} predictions: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, object] = dict(raw)
                row["fold"] = fold
                row["target_months"] = float(raw["target_months"])
                row["prediction_months"] = float(raw["prediction_months"])
                image_id = str(raw["image_id"])
                if image_id in by_id:
                    raise ValueError(f"Duplicate OOF image_id: {image_id}")
                by_id[image_id] = row
                rows.append(row)
    return rows, by_id


def grouped_metrics(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: metrics(group) for name, group in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-root",
        type=Path,
        default=ROOT / "c3_roi_attention" / "runs" / "C3_ROI_ATTN_V1",
    )
    parser.add_argument(
        "--c3-root", type=Path, default=ROOT / "c3_roi" / "runs" / "C3_ROI_V1",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "c3_roi_attention" / "outputs" / "C3_ROI_ATTN_V1_OOF",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    attention_rows, attention = load_oof(args.attention_root, "C3_ROI_ATTN_V1_FOLD_")
    c3_rows, c3 = load_oof(args.c3_root, "C3_ROI_V1_FOLD_")
    if set(attention) != set(c3):
        raise SystemExit("OOF ID mismatch between C3-ROI + attention and C3-ROI")
    if len(attention) != 14036:
        raise SystemExit(f"Expected 14,036 unique OOF IDs, found {len(attention)}")

    paired: list[dict[str, object]] = []
    for image_id in sorted(attention, key=lambda value: int(value)):
        attn_row = attention[image_id]
        c3_row = c3[image_id]
        if attn_row["target_months"] != c3_row["target_months"] or attn_row["sex"] != c3_row["sex"]:
            raise SystemExit(f"Target/sex mismatch for image_id={image_id}")
        paired.append({
            "image_id": image_id,
            "target_months": attn_row["target_months"],
            "sex": attn_row["sex"],
            "fold": attn_row["fold"],
            "attention_prediction_months": attn_row["prediction_months"],
            "c3_prediction_months": c3_row["prediction_months"],
        })

    attention_metric_rows = [
        {**row, "prediction_months": row["attention_prediction_months"]} for row in paired
    ]
    c3_metric_rows = [{**row, "prediction_months": row["c3_prediction_months"]} for row in paired]
    report = {
        "protocol": "locked 5-fold development OOF; test 200 not used",
        "comparison": "C3_ROI_ATTN_V1 vs C3_ROI_V1",
        "audit": {"unique_ids": len(paired), "id_sets_equal": True, "target_and_sex_match": True},
        "attention": {
            "overall": metrics(attention_metric_rows),
            "by_fold": grouped_metrics(attention_metric_rows, "fold"),
            "by_sex": grouped_metrics(attention_metric_rows, "sex"),
        },
        "c3_roi": {
            "overall": metrics(c3_metric_rows),
            "by_fold": grouped_metrics(c3_metric_rows, "fold"),
            "by_sex": grouped_metrics(c3_metric_rows, "sex"),
        },
        "paired_bootstrap_attention_minus_c3": paired_bootstrap_delta(
            paired, n=args.bootstrap, seed=42
        ),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "C3_ROI_ATTN_V1_vs_C3_ROI_V1_OOF.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_path = args.output_root / "C3_ROI_ATTN_V1_vs_C3_ROI_V1_predictions.csv"
    fields = [
        "image_id", "target_months", "sex", "fold",
        "attention_prediction_months", "c3_prediction_months",
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in paired)
    print(json.dumps({"report": str(report_path), **report["paired_bootstrap_attention_minus_c3"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
