"""Leakage-safe C3-ROI TTA OOF inference and E1-TTA ensemble.

This script evaluates the pre-registered candidate
0.5 * E1-TTA + 0.5 * C3-ROI-TTA on development OOF predictions only.
The RSNA test set is never read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p1_baseline.model import build_model
from p9_inference.tta_bias_oof import predict_transform, load_resolved_config


ROTATIONS = [-10.0, -5.0, 0.0, 5.0, 10.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_c3_oof(c3_root: Path) -> list[dict]:
    rows: list[dict] = []
    for fold in range(1, 6):
        run_dir = c3_root / f"C3_ROI_V1_FOLD_{fold}"
        val_path = run_dir / "val_predictions_best.csv"
        config = load_resolved_config(run_dir / "config_resolved.yaml")
        manifest_path = ROOT / str(config["val_manifest"])
        manifest = {str(row["image_id"]): row for row in read_csv(manifest_path)}
        for item in read_csv(val_path):
            image_id = str(item["image_id"])
            source = manifest.get(image_id)
            if source is None:
                raise RuntimeError(f"Missing C3 validation manifest row: {image_id}")
            image_path = ROOT / source["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if source["sex"] != item["sex"]:
                raise RuntimeError(f"Sex mismatch for {image_id}")
            rows.append(
                {
                    "image_id": image_id,
                    "fold": fold,
                    "target_months": float(item["target_months"]),
                    "sex": item["sex"],
                    "image_path": str(image_path),
                    "image_sha256": source["sha256"],
                    "c3_raw_training_prediction_months": float(item["prediction_months"]),
                    "image_size": int(config["image_size"]),
                    "target_mean": float(config["target_mean"]),
                    "target_std": float(config["target_std"]),
                }
            )
    if len(rows) != 14036 or len({row["image_id"] for row in rows}) != 14036:
        raise RuntimeError("C3 OOF must contain exactly 14,036 unique IDs")
    return rows


def load_c3_model(run_dir: Path, device: torch.device):
    config = load_resolved_config(run_dir / "config_resolved.yaml")
    model = build_model(
        str(config["architecture"]),
        False,
        int(config["sex_embedding_dim"]),
        int(config["head_hidden_dim"]),
        float(config["dropout"]),
        int(config["age_class_count"]),
        sex_mode=str(config.get("sex_mode", "embedding")),
    ).to(device)
    checkpoint_path = run_dir / "best_mae.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model._p9_target_config = {
        "target_mean": float(config["target_mean"]),
        "target_std": float(config["target_std"]),
    }
    return model


def metric_arrays(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mae_months": float(np.mean(absolute)),
        "rmse_months": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_months": float(np.median(absolute)),
        "accuracy_within_6_months": float(np.mean(absolute <= 6)),
        "accuracy_within_12_months": float(np.mean(absolute <= 12)),
        "accuracy_within_18_months": float(np.mean(absolute <= 18)),
        "signed_bias_months": float(np.mean(error)),
    }


def paired_bootstrap_delta(target: np.ndarray, candidate: np.ndarray, reference: np.ndarray, seed: int, samples: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = np.abs(candidate - target) - np.abs(reference - target)
    values = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 128):
        stop = min(start + 128, samples)
        indices = rng.integers(0, len(deltas), size=(stop - start, len(deltas)))
        values[start:stop] = deltas[indices].mean(axis=1)
    return {
        "estimate_months": float(np.mean(deltas)),
        "lower_95_months": float(np.quantile(values, 0.025)),
        "upper_95_months": float(np.quantile(values, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3-root", type=Path, default=Path("c3_roi/runs/C3_ROI_V1"))
    parser.add_argument("--e1-tta-csv", type=Path, default=Path("p9_inference/outputs/P9_I_TTA_BIAS_OOF/P9_I_TTA_BIAS_OOF_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("c3_roi/outputs/C3_ROI_TTA_OOF"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-per-fold", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = read_c3_oof(args.c3_root)
    e1_rows = {str(row["image_id"]): row for row in read_csv(args.e1_tta_csv)}
    if set(e1_rows) != {row["image_id"] for row in rows}:
        raise RuntimeError("E1-TTA and C3 OOF ID sets do not match")
    if args.limit_per_fold is not None:
        rows = [row for row in rows if row["fold"] <= 5 and sum(1 for other in rows if other["fold"] == row["fold"] and int(other["image_id"]) <= int(row["image_id"])) <= args.limit_per_fold]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, float] = {}
    raw_recomputed: dict[str, float] = {}
    fold_reports = {}

    for fold in range(1, 6):
        fold_rows = [row for row in rows if row["fold"] == fold]
        run_dir = args.c3_root / f"C3_ROI_V1_FOLD_{fold}"
        model = load_c3_model(run_dir, device)
        transforms = []
        print(f"fold={fold} start rows={len(fold_rows)}", flush=True)
        for rotation in ROTATIONS:
            for flip in (False, True):
                name = f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}"
                print(f"fold={fold} view={name} start", flush=True)
                values = predict_transform(
                    model, fold_rows, fold_rows[0]["image_size"], rotation, flip,
                    device, args.batch_size, args.num_workers, args.amp,
                )
                transforms.append(values)
                print(f"fold={fold} view={name} done", flush=True)
        names_count = len(transforms)
        for row in fold_rows:
            image_id = row["image_id"]
            values = [transform[image_id] for transform in transforms]
            raw_recomputed[image_id] = values[2]  # rot_0_no_flip
            predictions[image_id] = float(np.mean(values))
        fold_reports[str(fold)] = {"count": len(fold_rows), "tta_views": names_count}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"fold={fold} rows={len(fold_rows)}", flush=True)

    ordered = sorted(rows, key=lambda row: int(row["image_id"]))
    target = np.asarray([row["target_months"] for row in ordered], dtype=np.float64)
    e1_tta = np.asarray([float(e1_rows[row["image_id"]]["tta_prediction_months"]) for row in ordered], dtype=np.float64)
    c3_tta = np.asarray([predictions[row["image_id"]] for row in ordered], dtype=np.float64)
    c3_raw = np.asarray([float(row["c3_raw_training_prediction_months"]) for row in ordered], dtype=np.float64)
    candidate = 0.5 * e1_tta + 0.5 * c3_tta
    raw_ensemble = 0.5 * e1_tta + 0.5 * c3_raw

    output_rows = []
    for index, row in enumerate(ordered):
        output_rows.append({
            "image_id": row["image_id"], "fold": row["fold"], "sex": row["sex"],
            "target_months": target[index], "e1_tta_months": e1_tta[index],
            "c3_raw_months": c3_raw[index], "c3_tta_months": c3_tta[index],
            "ensemble_e1tta_c3raw_months": raw_ensemble[index],
            "ensemble_e1tta_c3tta_months": candidate[index],
        })
    with (args.output_dir / "C3_ROI_TTA_OOF_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "status": "PASS",
        "protocol": "development OOF only; 5-fold C3-ROI; rotations -10,-5,0,5,10 x flip/no-flip; fixed 50/50 E1-TTA + C3-ROI-TTA",
        "test_accessed": False,
        "input_rows": len(ordered),
        "tta": {"rotations": ROTATIONS, "flip_values": [False, True], "amp": args.amp},
        "fold_inference": fold_reports,
        "metrics": {
            "e1_tta": metric_arrays(target, e1_tta),
            "c3_raw": metric_arrays(target, c3_raw),
            "c3_tta": metric_arrays(target, c3_tta),
            "e1_tta_plus_c3_raw_50_50": metric_arrays(target, raw_ensemble),
            "e1_tta_plus_c3_tta_50_50": metric_arrays(target, candidate),
        },
        "paired_bootstrap_delta_vs_e1_tta": paired_bootstrap_delta(target, candidate, e1_tta, 20260903),
        "paired_bootstrap_delta_vs_e1_tta_plus_c3_raw": paired_bootstrap_delta(target, candidate, raw_ensemble, 20260904),
        "source_sha256": {"e1_tta_csv": sha256_file(args.e1_tta_csv)},
        "notes": [
            "No RSNA test image or test label was read.",
            "The 50/50 ensemble weight was fixed before evaluating this OOF result.",
            "The candidate is retained only if it improves OOF with the locked statistical gates.",
        ],
    }
    (args.output_dir / "C3_ROI_TTA_OOF_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
