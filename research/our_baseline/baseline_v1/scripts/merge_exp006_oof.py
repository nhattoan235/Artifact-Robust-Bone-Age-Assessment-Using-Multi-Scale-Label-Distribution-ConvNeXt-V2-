"""Merge one five-fold friend's-trainer experiment into leakage-safe OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame) -> dict:
    error = frame["prediction_months"].to_numpy(float) - frame["target_months"].to_numpy(float)
    absolute = np.abs(error)
    return {
        "n": int(len(frame)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "median_ae": float(np.median(absolute)),
        "acc_pm6": float(np.mean(absolute <= 6)),
        "acc_pm12": float(np.mean(absolute <= 12)),
        "acc_pm18": float(np.mean(absolute <= 18)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    source_files = []
    for fold in range(1, args.folds + 1):
        run_dir = args.runs_root / f"{args.run_prefix}{fold}"
        path = run_dir / "val_predictions_best.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold prediction: {path}")
        frame = pd.read_csv(path)
        required = {"image_id", "target_months", "prediction_months", "sex"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        part = frame[["image_id", "target_months", "prediction_months", "sex"]].copy()
        part["image_id"] = part["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        part["target_months"] = pd.to_numeric(part["target_months"], errors="raise")
        part["prediction_months"] = pd.to_numeric(part["prediction_months"], errors="raise")
        part["fold"] = fold
        part["source_file"] = str(path.resolve())
        if part["image_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs inside {path}")
        frames.append(part)
        source_files.append(path)

    oof = pd.concat(frames, ignore_index=True)
    if oof["image_id"].duplicated().any():
        duplicates = oof.loc[oof["image_id"].duplicated(keep=False), "image_id"].head().tolist()
        raise ValueError(f"Duplicate IDs across folds: {duplicates}")
    oof = oof.sort_values("image_id", key=lambda s: s.map(lambda x: int(float(x)))).reset_index(drop=True)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_labels_used": False,
        "run_prefix": args.run_prefix,
        "runs_root": str(args.runs_root.resolve()),
        "count": int(len(oof)),
        "fold_counts": {str(fold): int((oof["fold"] == fold).sum()) for fold in range(1, args.folds + 1)},
        "pooled": metrics(oof),
        "folds": {str(fold): metrics(oof[oof["fold"] == fold]) for fold in range(1, args.folds + 1)},
        "sources": {str(path): sha256_file(path) for path in source_files},
        "notes": [
            "OOF rows are predictions from the fold that did not train on the row.",
            "This report is the selection gate for TTA/LDL/blend candidates.",
            "The 200-image test set is not read.",
        ],
    }
    (args.output_dir / "oof_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
