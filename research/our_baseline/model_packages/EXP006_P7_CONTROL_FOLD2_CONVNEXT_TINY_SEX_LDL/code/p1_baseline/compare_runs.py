from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

from .metrics import compute_metrics


def load(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["image_id"]: {"target": float(row["target_months"]), "prediction": float(row["prediction_months"]), "sex": row["sex"]} for row in rows}


def percentile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return sorted_values[left]
    return sorted_values[left] * (right - position) + sorted_values[right] * (position - left)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    if set(baseline) != set(candidate) or len(baseline) != 1425:
        raise SystemExit("Hai prediction file không có cùng 1.425 ID")
    ids = sorted(baseline, key=int)
    deltas = []
    candidate_records = []
    for image_id in ids:
        a, b = baseline[image_id], candidate[image_id]
        if a["target"] != b["target"] or a["sex"] != b["sex"]:
            raise SystemExit(f"Ground truth/sex không khớp tại ID {image_id}")
        deltas.append(abs(b["prediction"] - b["target"]) - abs(a["prediction"] - a["target"]))
        candidate_records.append({"target_months": b["target"], "prediction_months": b["prediction"], "sex": b["sex"]})
    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.bootstrap):
        boot.append(statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas))
    boot.sort()
    candidate_metrics = compute_metrics(candidate_records, [0, 60, 120, 180, 229])
    report = {
        "status": "PASS",
        "count": len(ids),
        "delta_definition": "candidate_absolute_error - baseline_absolute_error; negative favors candidate",
        "baseline_mae": statistics.fmean(abs(baseline[i]["prediction"] - baseline[i]["target"]) for i in ids),
        "candidate_mae": candidate_metrics["mae"],
        "delta_mae": statistics.fmean(deltas),
        "paired_bootstrap_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "candidate_metrics": candidate_metrics,
        "candidate_better_pairs": sum(value < 0 for value in deltas),
        "candidate_worse_pairs": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.seed,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
