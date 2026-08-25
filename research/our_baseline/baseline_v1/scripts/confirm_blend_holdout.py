"""Confirm the pre-registered 50/50 blend on a deterministic internal holdout.

This uses already exported out-of-sample predictions. The split is made by
image ID, with 70% for development/weight inspection and 30% for confirmation.
It is not claimed to be a pristine holdout because the earlier EXP-001 audit
inspected the complete official validation file; that limitation is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_group(image_id: str) -> str:
    # Stable across machines and Python versions.
    value = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "development_70" if value < 0.70 else "holdout_30"


def metrics(name: str, frame: pd.DataFrame, weight: float) -> dict:
    target = frame.target_months.to_numpy(dtype=np.float64)
    pred = (1.0 - weight) * frame.own_prediction_months + weight * frame.friend_prediction_months
    error = pred.to_numpy() - target
    absolute = np.abs(error)
    return {
        "split": name,
        "friend_weight": weight,
        "n": int(len(frame)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(absolute)),
        "within_6_months": float(np.mean(absolute <= 6)),
        "within_12_months": float(np.mean(absolute <= 12)),
        "within_18_months": float(np.mean(absolute <= 18)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.merged_predictions)
    required = {"image_id", "target_months", "own_prediction_months", "friend_prediction_months"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["split"] = frame["image_id"].astype(str).map(split_group)
    rows = []
    for split in ("development_70", "holdout_30"):
        part = frame[frame.split == split].copy()
        for weight in WEIGHTS:
            rows.append(metrics(split, part, weight))
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "holdout_blend_metrics.csv", index=False)
    frame.to_csv(args.output_dir / "holdout_assignment.csv", index=False)
    print(result.to_string(index=False))
    holdout = result[result.split == "holdout_30"].sort_values("mae")
    best = holdout.iloc[0]
    fixed = holdout[np.isclose(holdout.friend_weight, 0.5)].iloc[0]
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_predictions": str(args.merged_predictions),
        "source_sha256": sha256(args.merged_predictions),
        "split_rule": "sha256(image_id) normalized to [0,1): <0.70 development, otherwise holdout",
        "development_rows": int((frame.split == "development_70").sum()),
        "holdout_rows": int((frame.split == "holdout_30").sum()),
        "weights_tested": list(WEIGHTS),
        "fixed_weight_confirmed": 0.5,
        "holdout_fixed_50_50_mae": float(fixed.mae),
        "holdout_best_weight": float(best.friend_weight),
        "holdout_best_mae": float(best.mae),
        "strict_independence_warning": "The complete official validation file was inspected in EXP-001 before this split was created. This is an internal split-confirmation, not a pristine unseen holdout. A strict confirmation requires a new training split or an untouched external holdout.",
    }
    (args.output_dir / "holdout_blend_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        f"holdout_best_weight={best.friend_weight:.2f} holdout_best_mae={best.mae:.6f} "
        f"fixed_50_50_mae={fixed.mae:.6f} n_holdout={int(fixed.n)}"
    )


if __name__ == "__main__":
    main()
