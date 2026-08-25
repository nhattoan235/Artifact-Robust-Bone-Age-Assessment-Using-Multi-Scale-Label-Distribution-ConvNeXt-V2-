"""Exploratory raw/TTA evaluation for the 200-image labeled test set.

The labels are read only because the user explicitly requested this diagnostic.
They must not be used for model, weight, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_config(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        value = raw.strip().strip('"')
        if value.lower() in {"true", "false"}:
            result[key.strip()] = value.lower() == "true"
        else:
            try:
                result[key.strip()] = int(value)
            except ValueError:
                try:
                    result[key.strip()] = float(value)
                except ValueError:
                    result[key.strip()] = value
    return result


def metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    error = prediction - target
    absolute = np.abs(error)
    result: dict[str, object] = {
        "n": int(len(target)),
        "mae_months": float(absolute.mean()),
        "rmse_months": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_months": float(np.median(absolute)),
        "accuracy_pm6": float(np.mean(absolute <= 6)),
        "accuracy_pm12": float(np.mean(absolute <= 12)),
        "accuracy_pm18": float(np.mean(absolute <= 18)),
        "signed_bias_months": float(error.mean()),
    }
    result["age_bins"] = {}
    for low, high in [(0, 60), (60, 120), (120, 180), (180, 229)]:
        mask = (target >= low) & (target < high)
        if np.any(mask):
            result["age_bins"][f"{low}-{high - 1}"] = {
                "n": int(mask.sum()),
                "mae_months": float(absolute[mask].mean()),
                "signed_bias_months": float(error[mask].mean()),
            }
    return result


def load_model(friend_repo: Path, run_dir: Path, device: torch.device):
    if str(friend_repo) not in sys.path:
        sys.path.insert(0, str(friend_repo))
    from p1_baseline.model import build_model

    config = load_config(run_dir / "config_resolved.yaml")
    model = build_model(
        str(config["architecture"]),
        False,
        int(config["sex_embedding_dim"]),
        int(config["head_hidden_dim"]),
        float(config["dropout"]),
        int(config["age_class_count"]),
        sex_mode=str(config.get("sex_mode", "embedding")),
    ).to(device)
    checkpoint = run_dir / "best_mae.ckpt"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    model._p9_target_config = {
        "target_mean": float(config["target_mean"]),
        "target_std": float(config["target_std"]),
    }
    return model, config


def make_rows(data_root: Path, test_csv: Path) -> list[dict[str, object]]:
    source = pd.read_csv(test_csv)
    if len(source) != 200:
        raise RuntimeError(f"Expected exactly 200 test rows, got {len(source)}")
    id_col = "id" if "id" in source.columns else "Image ID"
    target_col = "boneage" if "boneage" in source.columns else "Bone Age (months)"
    sex_col = "male" if "male" in source.columns else "sex"
    path_map: dict[str, Path] = {}
    for path in data_root.rglob("*.png"):
        if path.stem in path_map:
            raise RuntimeError(f"Duplicate test image ID: {path.stem}")
        path_map[path.stem] = path.resolve()
    rows: list[dict[str, object]] = []
    for _, item in source.iterrows():
        image_id = str(item[id_col]).strip()
        if image_id.endswith(".0"):
            image_id = image_id[:-2]
        if image_id not in path_map:
            raise FileNotFoundError(f"Missing test image: {image_id}")
        raw_sex = str(item[sex_col]).strip().lower()
        sex = "M" if raw_sex in {"true", "1", "male", "m"} else "F"
        rows.append({
            "image_id": image_id,
            "sex": sex,
            "image_path": str(path_map[image_id]),
            "target_months": float(item[target_col]),
        })
    if len({str(row["image_id"]) for row in rows}) != 200:
        raise RuntimeError("Test image IDs are not unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--friend-repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rotations", nargs="+", type=float, default=[-10, -5, 0, 5, 10])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if 0 not in args.rotations:
        raise ValueError("rotations must include 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.friend_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(args.friend_repo.resolve()))
    from p9_inference.tta_bias_oof import predict_transform

    rows = make_rows(args.data_root, args.test_csv)
    print(f"[TEST200-TTA] device={device} rows={len(rows)}", flush=True)
    print(f"[TEST200-TTA] rotations={args.rotations} flips=[False, True]", flush=True)

    output_rows = [{"image_id": row["image_id"], "sex": row["sex"], "target_months": row["target_months"]} for row in rows]
    fold_raw: list[np.ndarray] = []
    fold_tta: list[np.ndarray] = []
    names = [f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}" for rotation in args.rotations for flip in [False, True]]
    transform_values: dict[str, list[np.ndarray]] = {name: [] for name in names}

    for fold in range(1, 6):
        run_dir = args.runs_root / f"{args.run_prefix}{fold}"
        checkpoint = run_dir / "best_mae.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"[TEST200-TTA] fold={fold} loading {checkpoint}", flush=True)
        model, config = load_model(args.friend_repo, run_dir, device)
        per_transform: dict[str, np.ndarray] = {}
        for rotation in args.rotations:
            for flip in [False, True]:
                name = f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}"
                print(f"[TEST200-TTA] fold={fold} {name} START", flush=True)
                pred_map = predict_transform(
                    model, rows, int(config["image_size"]), rotation, flip,
                    device, args.batch_size, args.num_workers, args.amp,
                )
                per_transform[name] = np.asarray([pred_map[str(row["image_id"])] for row in rows], dtype=np.float64)
                transform_values[name].append(per_transform[name])
                print(f"[TEST200-TTA] fold={fold} {name} DONE", flush=True)
        raw = per_transform["rot_0_no_flip"]
        tta = np.mean(np.stack(list(per_transform.values())), axis=0)
        fold_raw.append(raw)
        fold_tta.append(tta)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[TEST200-TTA] fold={fold} complete", flush=True)

    raw_ensemble = np.mean(np.stack(fold_raw), axis=0)
    tta_ensemble = np.mean(np.stack(fold_tta), axis=0)
    for index, row in enumerate(output_rows):
        row["prediction_raw_5fold"] = float(raw_ensemble[index])
        row["prediction_tta_5fold"] = float(tta_ensemble[index])
        row["tta_delta_months"] = float(tta_ensemble[index] - raw_ensemble[index])
        for name, values in transform_values.items():
            row[f"prediction_{name}_5fold"] = float(np.mean(np.stack(values), axis=0)[index])

    result = pd.DataFrame(output_rows)
    prediction_path = args.output_dir / "test200_p7_tta_predictions.csv"
    result.to_csv(prediction_path, index=False)
    target = result["target_months"].to_numpy(float)
    raw_metrics = metric(target, raw_ensemble)
    tta_metrics = metric(target, tta_ensemble)
    report = {
        "evaluation": "exploratory_test_200",
        "test_labels_used": True,
        "selection_warning": "Do not use this test report to select weights or hyperparameters.",
        "protocol": "EXP006 P7 five-fold ensemble; raw is rot_0_no_flip, TTA averages 5 rotations x 2 flip states per fold.",
        "device": str(device),
        "folds": 5,
        "count": 200,
        "rotations": args.rotations,
        "flips": [False, True],
        "methods": {"raw_5fold": raw_metrics, "tta_5fold": tta_metrics},
        "delta_tta_minus_raw": {
            "mae_months": tta_metrics["mae_months"] - raw_metrics["mae_months"],
            "rmse_months": tta_metrics["rmse_months"] - raw_metrics["rmse_months"],
        },
        "checkpoints": [str(args.runs_root / f"{args.run_prefix}{fold}" / "best_mae.ckpt") for fold in range(1, 6)],
        "prediction_file": str(prediction_path),
    }
    report_path = args.output_dir / "test200_p7_tta_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
