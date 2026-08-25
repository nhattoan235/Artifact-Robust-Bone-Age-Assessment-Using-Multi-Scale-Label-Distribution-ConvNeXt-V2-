"""Evaluate leakage-safe TTA on EXP-006/007 fold checkpoints.

The script consumes only OOF validation predictions and local development
images. It supports direct P7 checkpoints and LDL checkpoints; for LDL it
evaluates the regression output, leaving fused distribution inference for the
separate LDL audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "image_path", "sha256", "sex"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest thiếu cột: {sorted(required)}")
    result = {}
    for row in rows:
        image_id = str(row["image_id"]).strip()
        if image_id.endswith(".0"):
            image_id = image_id[:-2]
        if image_id in result:
            raise ValueError(f"Duplicate image ID in manifest: {image_id}")
        result[image_id] = row
    return result


def load_config(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, raw = line.split(":", 1)
        result[key.strip()] = raw.strip().strip('"')
    return result


def load_model(run_dir: Path, device: torch.device):
    repo_root = run_dir.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from p1_baseline.model import build_model

    config = load_config(run_dir / "config_resolved.yaml")
    model = build_model(
        config["architecture"], False, int(config["sex_embedding_dim"]),
        int(config["head_hidden_dim"]), float(config["dropout"]),
        int(config["age_class_count"]), sex_mode=config.get("sex_mode", "embedding"),
    ).to(device)
    checkpoint = run_dir / "best_mae.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    model._p9_target_config = {
        "target_mean": float(config["target_mean"]),
        "target_std": float(config["target_std"]),
    }
    return model, config


def metric(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "n": int(len(target)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "median_ae": float(np.median(absolute)),
        "acc_pm6": float(np.mean(absolute <= 6)),
        "acc_pm12": float(np.mean(absolute <= 12)),
        "acc_pm18": float(np.mean(absolute <= 18)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--friend-repo", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rotations", nargs="+", type=float, default=[-10, -5, 0, 5, 10])
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()
    print(f"[TTA] Starting. friend_repo={args.friend_repo}", flush=True)
    print(f"[TTA] manifest={args.development_manifest}", flush=True)
    print(f"[TTA] oof={args.oof_csv}", flush=True)
    if 0 not in args.rotations:
        raise ValueError("rotations must include 0")
    if str(args.friend_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(args.friend_repo.resolve()))
    from p9_inference.tta_bias_oof import predict_transform

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(f"[TTA] device={device} amp={args.amp} batch_size={args.batch_size} workers={args.num_workers}", flush=True)

    manifest = load_manifest(args.development_manifest)
    oof = pd.read_csv(args.oof_csv)
    required = {"image_id", "target_months", "prediction_months", "sex", "fold"}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF thiếu cột: {sorted(missing)}")
    oof["image_id"] = oof["image_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if oof["image_id"].duplicated().any():
        raise ValueError("OOF chứa ID trùng")
    if len(oof) != len(manifest):
        raise ValueError(f"OOF/manifest count mismatch: {len(oof)} vs {len(manifest)}")

    output_rows = []
    rotations = [float(x) for x in args.rotations]
    flips = [False] if args.no_flip else [False, True]
    print(f"[TTA] rows={len(oof)} folds={sorted(oof['fold'].astype(int).unique())}", flush=True)
    print(f"[TTA] rotations={rotations} flips={flips}", flush=True)
    for fold in sorted(oof["fold"].astype(int).unique()):
        print(f"[TTA] Fold {fold}: preparing rows", flush=True)
        fold_frame = oof[oof["fold"].astype(int) == fold].copy()
        rows = []
        for _, item in fold_frame.iterrows():
            image_id = str(item["image_id"])
            if image_id not in manifest:
                raise ValueError(f"OOF ID missing from manifest: {image_id}")
            source = manifest[image_id]
            if str(source["sex"]) != str(item["sex"]):
                raise ValueError(f"Sex mismatch for {image_id}")
            image_path = Path(source["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            rows.append({
                "image_id": image_id,
                "sex": str(item["sex"]),
                "image_path": str(image_path),
                "target_months": float(item["target_months"]),
                "fold": fold,
                "image_sha256": source["sha256"],
                "oof_prediction_months": float(item["prediction_months"]),
            })
        run_dir = args.runs_root / f"{args.run_prefix}{fold}"
        print(f"[TTA] Fold {fold}: loading checkpoint from {run_dir}", flush=True)
        model, config = load_model(run_dir, device)
        print(f"[TTA] Fold {fold}: checkpoint loaded, {len(rows)} images", flush=True)
        transforms = {}
        for rotation in rotations:
            for flip in flips:
                print(f"[TTA] Fold {fold}: rotation={rotation:g}, flip={flip} START", flush=True)
                name = f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}"
                transforms[name] = predict_transform(
                    model, rows, int(config["image_size"]), rotation, flip,
                    device, args.batch_size, args.num_workers, args.amp,
                )
                print(f"[TTA] Fold {fold}: rotation={rotation:g}, flip={flip} DONE", flush=True)
        names = [f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}" for rotation in rotations for flip in flips]
        for row in rows:
            values = np.asarray([transforms[name][row["image_id"]] for name in names], dtype=np.float64)
            row["raw_prediction_months"] = float(transforms["rot_0_no_flip"][row["image_id"]])
            row["tta_prediction_months"] = float(values.mean())
            row["tta_std_months"] = float(values.std())
            for name in names:
                row[f"prediction_{name}_months"] = float(transforms[name][row["image_id"]])
            output_rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[TTA] Fold {fold}: complete", flush=True)

    result = pd.DataFrame(output_rows).sort_values("image_id", key=lambda s: s.map(lambda x: int(float(x))))
    result.to_csv(args.output_dir / "tta_oof_predictions.csv", index=False)
    target = result["target_months"].to_numpy(float)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_labels_used": False,
        "protocol": "P7-compatible pad-square TTA on fold-held-out development rows",
        "device": str(device),
        "folds": int(result["fold"].nunique()),
        "count": int(len(result)),
        "rotations": rotations,
        "flips": flips,
        "metrics": {
            "p7_oof_reference": metric(target, result["oof_prediction_months"].to_numpy(float)),
            "raw_recomputed": metric(target, result["raw_prediction_months"].to_numpy(float)),
            "tta": metric(target, result["tta_prediction_months"].to_numpy(float)),
        },
        "source_sha256": {
            "development_manifest": sha256_file(args.development_manifest),
            "oof_csv": sha256_file(args.oof_csv),
        },
        "notes": [
            "No test CSV or test label was read.",
            "Raw and TTA are both retained; TTA is accepted only if OOF improves consistently.",
            "For LDL checkpoints this script evaluates the regression output only.",
        ],
    }
    (args.output_dir / "tta_oof_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TTA] Complete. Results written to {args.output_dir}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
