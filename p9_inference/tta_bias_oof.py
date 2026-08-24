from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from p1_baseline.model import build_model


def parse_scalar(value: str):
    value = value.strip()
    if value.lower() in {"null", "none"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("["):
        return json.loads(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_resolved_config(path: Path) -> dict:
    config = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        config[key.strip()] = parse_scalar(raw)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


class TTADataset(Dataset):
    """P7-compatible inference loader with Deeplasia-style rotate/flip TTA."""

    def __init__(self, rows: list[dict], image_size: int, rotation: float, flip: bool):
        self.rows = rows
        self.image_size = image_size
        self.rotation = float(rotation)
        self.flip = bool(flip)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = pad_square(source.convert("L"))
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            if self.flip:
                image = TF.hflip(image)
            image = TF.rotate(
                image,
                angle=self.rotation,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0,
            )
            tensor = TF.pil_to_tensor(image).float().div_(255.0).repeat(3, 1, 1)
            tensor = TF.normalize(
                tensor,
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            )
        return {
            "image": tensor,
            "sex": torch.tensor([1.0 if row["sex"] == "M" else 0.0], dtype=torch.float32),
            "image_id": row["image_id"],
        }


def read_inputs(manifest_path: Path, oof_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {str(row["image_id"]): row for row in csv.DictReader(handle)}
    with oof_path.open("r", encoding="utf-8-sig", newline="") as handle:
        oof_rows = list(csv.DictReader(handle))

    if len(manifest) != 14036 or len(oof_rows) != 14036:
        raise RuntimeError(
            f"Expected 14,036 development/OOF rows, got manifest={len(manifest)}, oof={len(oof_rows)}"
        )
    if len({str(row["image_id"]) for row in oof_rows}) != len(oof_rows):
        raise RuntimeError("OOF contains duplicate image IDs")

    rows = []
    for item in oof_rows:
        image_id = str(item["image_id"])
        if image_id not in manifest:
            raise RuntimeError(f"OOF ID missing from development manifest: {image_id}")
        source = manifest[image_id]
        if str(source["image_id"]) != image_id or source["sex"] != item["sex"]:
            raise RuntimeError(f"Manifest/OOF identity mismatch for ID={image_id}")
        if abs(float(source["bone_age_months"]) - float(item["target_months"])) > 1e-6:
            raise RuntimeError(f"Manifest/OOF target mismatch for ID={image_id}")
        image_path = Path(source["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing development image for ID={image_id}: {image_path}")
        rows.append(
            {
                "image_id": image_id,
                "fold": int(item["fold"]),
                "target_months": float(item["target_months"]),
                "oof_prediction_months": float(item["prediction_months"]),
                "sex": item["sex"],
                "image_path": str(image_path),
                "image_sha256": source["sha256"],
            }
        )
    folds = {row["fold"] for row in rows}
    if folds != {1, 2, 3, 4, 5}:
        raise RuntimeError(f"Unexpected OOF folds: {sorted(folds)}")
    return rows


def predict_transform(
    model: torch.nn.Module,
    rows: list[dict],
    image_size: int,
    rotation: float,
    flip: bool,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> dict[str, float]:
    dataset = TTADataset(rows, image_size, rotation, flip)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions: dict[str, float] = {}
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            sex = batch["sex"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                output = model(images, sex)
            if isinstance(output, dict):
                output = output["regression"]
            values = output.float().cpu().numpy()
            config = getattr(model, "_p9_target_config", None)
            if config is None:
                raise RuntimeError("Model target normalization metadata is missing")
            values = values * config["target_std"] + config["target_mean"]
            for image_id, value in zip(batch["image_id"], values.tolist()):
                predictions[str(image_id)] = float(value)
    if len(predictions) != len(rows):
        raise RuntimeError("Prediction count does not match input rows")
    return predictions


def load_fold_model(fold_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    config = load_resolved_config(fold_dir / "config_resolved.yaml")
    model = build_model(
        str(config["architecture"]),
        False,
        int(config["sex_embedding_dim"]),
        int(config["head_hidden_dim"]),
        float(config["dropout"]),
        int(config["age_class_count"]),
        sex_mode=str(config.get("sex_mode", "embedding")),
    ).to(device)
    checkpoint = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model._p9_target_config = {
        "target_mean": float(config["target_mean"]),
        "target_std": float(config["target_std"]),
    }
    return model, config


def fit_bias(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Fit Deeplasia's signed-error correction: e = slope*yhat + intercept."""
    x = np.column_stack([prediction, np.ones_like(prediction)])
    slope, intercept = np.linalg.lstsq(x, prediction - y, rcond=None)[0]
    return {"slope": float(slope), "intercept": float(intercept)}


def apply_bias(prediction: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return prediction - (prediction * params["slope"] + params["intercept"])


def bootstrap_ci(errors: np.ndarray, samples: int, seed: int) -> list[float]:
    if samples <= 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    # Batched vectorization avoids tens of thousands of Python-level loops on
    # the 14,036-row OOF set. Keep the batch bounded so the temporary index
    # matrix does not consume hundreds of MB.
    values = np.empty(samples, dtype=np.float64)
    batch_size = 256
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, len(errors), size=(stop - start, len(errors)))
        values[start:stop] = errors[indices].mean(axis=1)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def metrics(rows: list[dict], prediction_key: str, bootstrap_samples: int, bootstrap_seed: int) -> dict:
    y = np.asarray([row["target_months"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    errors = np.abs(prediction - y)
    result = {
        "count": int(len(rows)),
        "mae_months": float(np.mean(errors)),
        "rmse_months": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "median_absolute_error_months": float(np.median(errors)),
        "accuracy_within_6_months": float(np.mean(errors <= 6)),
        "accuracy_within_12_months": float(np.mean(errors <= 12)),
        "accuracy_within_18_months": float(np.mean(errors <= 18)),
        "signed_bias_prediction_minus_target_months": float(np.mean(prediction - y)),
        "prediction_mean_months": float(np.mean(prediction)),
        "prediction_std_months": float(np.std(prediction)),
        "bootstrap_95_ci_mae_months": bootstrap_ci(errors, bootstrap_samples, bootstrap_seed),
    }
    age_bins = [(0, 60), (60, 120), (120, 180), (180, 229)]
    result["age_bins"] = {}
    for low, high in age_bins:
        mask = (y >= low) & (y < high)
        if not np.any(mask):
            continue
        result["age_bins"][f"{low}-{high - 1}"] = {
            "count": int(np.sum(mask)),
            "mae_months": float(np.mean(errors[mask])),
            "signed_bias_months": float(np.mean(prediction[mask] - y[mask])),
        }
    result["sex"] = {}
    for sex in ["F", "M"]:
        mask = np.asarray([row["sex"] == sex for row in rows])
        if np.any(mask):
            result["sex"][sex] = {
                "count": int(np.sum(mask)),
                "mae_months": float(np.mean(errors[mask])),
                "signed_bias_months": float(np.mean(prediction[mask] - y[mask])),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="P9-I: leakage-safe TTA and bias correction on P7 OOF")
    parser.add_argument("--development-manifest", type=Path, default=Path("p0_audit/outputs/development_manifest_14036.csv"))
    parser.add_argument("--oof-csv", type=Path, default=Path("p7_oof_final/oof_final/P7_OOF_predictions.csv"))
    parser.add_argument("--results-root", type=Path, default=Path("p7_results/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("p9_inference/outputs/P9_I_TTA_BIAS_OOF"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast; off by default for raw reproduction fidelity")
    parser.add_argument("--rotations", nargs="+", type=float, default=[-10, -5, 0, 5, 10])
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--limit-per-fold", type=int, default=None, help="Smoke-test limit; never use for final results")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if 0 not in args.rotations:
        raise ValueError("rotations must include 0 for raw prediction integrity check")
    if args.limit_per_fold is not None and args.limit_per_fold < 4:
        raise ValueError("limit-per-fold must be at least 4")

    all_rows = read_inputs(args.development_manifest, args.oof_csv)
    rows_by_fold = {fold: [row for row in all_rows if row["fold"] == fold] for fold in range(1, 6)}
    if args.limit_per_fold is not None:
        rows_by_fold = {fold: rows[: args.limit_per_fold] for fold, rows in rows_by_fold.items()}
        rows = [row for fold in range(1, 6) for row in rows_by_fold[fold]]
    else:
        rows = all_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rotation_values = [float(value) for value in args.rotations]
    flip_values = [False] if args.no_flip else [False, True]
    per_row = {row["image_id"]: dict(row) for row in rows}
    fold_reports = {}

    for fold in range(1, 6):
        fold_rows = rows_by_fold[fold]
        fold_dir = args.results_root / f"P7_FINAL_V3_FOLD_{fold}"
        model, config = load_fold_model(fold_dir, device)
        transform_predictions: dict[str, dict[str, float]] = {}
        for rotation in rotation_values:
            for flip in flip_values:
                name = f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}"
                transform_predictions[name] = predict_transform(
                    model, fold_rows, int(config["image_size"]), rotation, flip,
                    device, args.batch_size, args.num_workers, args.amp,
                )
                for image_id, value in transform_predictions[name].items():
                    per_row[image_id][f"prediction_{name}_months"] = value

        raw_name = "rot_0_no_flip"
        tta_names = [
            f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}"
            for rotation in rotation_values for flip in flip_values
        ]
        for row in fold_rows:
            image_id = row["image_id"]
            values = np.asarray([transform_predictions[name][image_id] for name in tta_names], dtype=np.float64)
            per_row[image_id]["raw_prediction_months"] = float(transform_predictions[raw_name][image_id])
            per_row[image_id]["tta_prediction_months"] = float(np.mean(values))
            per_row[image_id]["tta_std_months"] = float(np.std(values))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fold_reports[str(fold)] = {
            "count": len(fold_rows),
            "rotations": rotation_values,
            "flips": flip_values,
            "model_dir": str(fold_dir),
        }

    ordered_rows = [per_row[row["image_id"]] for row in rows]
    raw_prediction = np.asarray([row["raw_prediction_months"] for row in ordered_rows], dtype=np.float64)
    tta_prediction = np.asarray([row["tta_prediction_months"] for row in ordered_rows], dtype=np.float64)
    targets = np.asarray([row["target_months"] for row in ordered_rows], dtype=np.float64)
    expected_raw = np.asarray([row["oof_prediction_months"] for row in ordered_rows], dtype=np.float64)
    raw_difference = raw_prediction - expected_raw

    # Leave-one-fold-out correction: parameters for fold f are fit only on the
    # other four OOF folds, so no sample's own label enters its correction.
    correction_parameters = {}
    for fold in range(1, 6):
        train_rows = [row for row in ordered_rows if row["fold"] != fold]
        val_rows = [row for row in ordered_rows if row["fold"] == fold]
        train_y = np.asarray([row["target_months"] for row in train_rows], dtype=np.float64)
        raw_train = np.asarray([row["raw_prediction_months"] for row in train_rows], dtype=np.float64)
        tta_train = np.asarray([row["tta_prediction_months"] for row in train_rows], dtype=np.float64)
        raw_params = fit_bias(train_y, raw_train)
        tta_params = fit_bias(train_y, tta_train)
        correction_parameters[str(fold)] = {
            "fit_sample_count": len(train_rows),
            "raw": raw_params,
            "tta": tta_params,
        }
        for row in val_rows:
            row["raw_bias_corrected_months"] = float(apply_bias(np.asarray([row["raw_prediction_months"]]), raw_params)[0])
            row["tta_bias_corrected_months"] = float(apply_bias(np.asarray([row["tta_prediction_months"]]), tta_params)[0])

    # Persist predictions before the comparatively expensive bootstrap step so
    # a statistics interruption never forces model inference to be repeated.
    fieldnames = [
        "image_id", "fold", "target_months", "sex", "image_sha256",
        "oof_prediction_months", "raw_prediction_months", "tta_prediction_months",
        "tta_std_months", "raw_bias_corrected_months", "tta_bias_corrected_months",
    ]
    for rotation in rotation_values:
        for flip in flip_values:
            fieldnames.append(f"prediction_rot_{rotation:g}_{'flip' if flip else 'no_flip'}_months")
    output_csv = args.output_dir / "P9_I_TTA_BIAS_OOF_predictions.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered_rows)

    metric_keys = {
        "p7_oof_reference_raw": "oof_prediction_months",
        "raw_recomputed": "raw_prediction_months",
        "tta": "tta_prediction_months",
        "raw_bias_corrected_crossfit": "raw_bias_corrected_months",
        "tta_bias_corrected_crossfit": "tta_bias_corrected_months",
    }
    metric_report = {
        name: metrics(ordered_rows, key, args.bootstrap, args.bootstrap_seed + index)
        for index, (name, key) in enumerate(metric_keys.items())
    }
    metric_report["tta_disagreement"] = {
        "mean_std_months": float(np.mean([row["tta_std_months"] for row in ordered_rows])),
        "median_std_months": float(np.median([row["tta_std_months"] for row in ordered_rows])),
        "p95_std_months": float(np.quantile([row["tta_std_months"] for row in ordered_rows], 0.95)),
    }

    report = {
        "status": "PASS",
        "protocol": "P7 ConvNeXt-Tiny checkpoint re-inference; Deeplasia-style rotations/flips; leave-one-fold-out bias correction",
        "test_accessed": False,
        "subset": args.limit_per_fold is not None,
        "limit_per_fold": args.limit_per_fold,
        "input_rows": len(ordered_rows),
        "fold_counts": {str(fold): len(rows_by_fold[fold]) for fold in range(1, 6)},
        "source_sha256": {
            "development_manifest": sha256_file(args.development_manifest),
            "oof_csv": sha256_file(args.oof_csv),
        },
        "tta": {"rotations": rotation_values, "flips": flip_values, "amp": args.amp},
        "raw_reproduction_vs_p7_oof": {
            "mean_abs_diff_months": float(np.mean(np.abs(raw_difference))),
            "median_abs_diff_months": float(np.median(np.abs(raw_difference))),
            "max_abs_diff_months": float(np.max(np.abs(raw_difference))),
            "count_abs_diff_le_0_01": int(np.sum(np.abs(raw_difference) <= 0.01)),
            "count_abs_diff_le_0_1": int(np.sum(np.abs(raw_difference) <= 0.1)),
        },
        "fold_inference": fold_reports,
        "crossfit_bias_parameters": correction_parameters,
        "metrics": metric_report,
        "notes": [
            "Primary selection endpoint is cross-fitted OOF, never RSNA test.",
            "TTA uses the same P7 raw preprocessing/model input normalization; this is an inference ablation, not yet a full Deeplasia preprocessing reproduction.",
            "The P7 OOF reference is retained for an integrity check; raw_recomputed is the model output used by the correction ablations.",
        ],
    }
    (args.output_dir / "P9_I_TTA_BIAS_OOF_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
