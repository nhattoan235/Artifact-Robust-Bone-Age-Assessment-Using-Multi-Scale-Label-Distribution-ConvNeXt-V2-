"""Locked 10-view C3-Z26 V2 test-time augmentation inference.

The transform set was selected on development OOF predictions. This runner does
not expose transform or ensemble-weight arguments, so the test set cannot be
used to retune the TTA protocol.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p1_baseline.model import build_model


LOCKED_VIEWS = tuple(
    (rotation, flip)
    for rotation in (-10.0, -5.0, 0.0, 5.0, 10.0)
    for flip in (False, True)
)


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
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        result[key.strip()] = parse_scalar(raw)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_test_rows(metadata_path: Path, image_root: Path, expected_count: int = 200) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    required = {"image_id", "sex"}
    if not metadata_rows or not required.issubset(metadata_rows[0]):
        raise RuntimeError(f"Metadata must contain columns: {sorted(required)}")
    metadata = {str(row["image_id"]): row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise RuntimeError("Metadata contains duplicate image IDs")

    images = {path.stem: path for path in image_root.glob("*.png")}
    if len(metadata) != expected_count or len(images) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} test rows/images, got metadata={len(metadata)} images={len(images)}"
        )
    if set(metadata) != set(images):
        missing_images = sorted(set(metadata) - set(images))[:5]
        missing_metadata = sorted(set(images) - set(metadata))[:5]
        raise RuntimeError(
            "ID set mismatch between metadata and images: "
            f"missing_images={missing_images} missing_metadata={missing_metadata}"
        )

    rows = []
    for image_id in sorted(metadata, key=int):
        source = metadata[image_id]
        sex = source["sex"].strip().upper()
        if sex not in {"F", "M"}:
            raise RuntimeError(f"Invalid sex for image_id={image_id}: {sex!r}")
        row = {"image_id": image_id, "sex": sex, "image_path": str(images[image_id])}
        target = source.get("target_months", "").strip()
        if target:
            row["target_months"] = float(target)
        rows.append(row)
    target_presence = {"target_months" in row for row in rows}
    if len(target_presence) != 1:
        raise RuntimeError("target_months must be present for all rows or omitted for all rows")
    return rows


def pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


class TTADataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, rotation: float, flip: bool):
        self.rows = rows
        self.image_size = image_size
        self.rotation = rotation
        self.flip = flip

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
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
            tensor = TF.normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return {
            "image": tensor,
            "sex": torch.tensor([1.0 if row["sex"] == "M" else 0.0], dtype=torch.float32),
            "image_id": row["image_id"],
        }


def load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    config_path = run_dir / "config_resolved.yaml"
    checkpoint_path = run_dir / "best_mae.ckpt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing config/checkpoint in {run_dir}")
    config = load_resolved_config(config_path)
    model = build_model(
        str(config["architecture"]),
        False,
        int(config["sex_embedding_dim"]),
        int(config["head_hidden_dim"]),
        float(config["dropout"]),
        int(config["age_class_count"]),
        sex_mode=str(config.get("sex_mode", "embedding")),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


def predict_view(
    model: torch.nn.Module,
    rows: list[dict],
    config: dict,
    rotation: float,
    flip: bool,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> dict[str, float]:
    loader = DataLoader(
        TTADataset(rows, int(config["image_size"]), rotation, flip),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    result = {}
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
            normalized = output["regression"] if isinstance(output, dict) else output
            values = (
                normalized.float().cpu().numpy() * float(config["target_std"])
                + float(config["target_mean"])
            )
            for image_id, value in zip(batch["image_id"], values.tolist()):
                result[str(image_id)] = float(value)
    if len(result) != len(rows):
        raise RuntimeError(f"Prediction count mismatch: {len(result)} != {len(rows)}")
    return result


def aggregate_predictions(
    prediction_sets: list[dict[str, float]], expected_ids: set[str]
) -> dict[str, float]:
    if not prediction_sets:
        raise RuntimeError("No prediction sets to aggregate")
    for index, values in enumerate(prediction_sets):
        if set(values) != expected_ids:
            raise RuntimeError(f"Prediction ID mismatch at set {index}")
    return {
        image_id: float(np.mean([values[image_id] for values in prediction_sets]))
        for image_id in sorted(expected_ids, key=int)
    }


def metric_arrays(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mae_months": float(np.mean(absolute)),
        "rmse_months": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_months": float(np.median(absolute)),
        "accuracy_within_6_months": float(np.mean(absolute <= 6)),
        "accuracy_within_12_months": float(np.mean(absolute <= 12)),
        "signed_bias_months": float(np.mean(error)),
    }


def bootstrap_mae_ci(target: np.ndarray, prediction: np.ndarray, samples: int = 10000) -> list[float]:
    errors = np.abs(prediction - target)
    rng = np.random.default_rng(20260908)
    values = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, len(errors), size=(stop - start, len(errors)))
        values[start:stop] = errors[indices].mean(axis=1)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_bootstrap_delta(
    target: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    samples: int = 10000,
) -> dict[str, float | list[float]]:
    paired_delta = np.abs(candidate - target) - np.abs(reference - target)
    rng = np.random.default_rng(20260909)
    values = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, len(paired_delta), size=(stop - start, len(paired_delta)))
        values[start:stop] = paired_delta[indices].mean(axis=1)
    return {
        "estimate_months": float(np.mean(paired_delta)),
        "bootstrap_95_ci": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2/runs"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/content/C3_Z26_COMBO_V2/images/test"),
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_TTA_TEST"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("Locked C3 TTA test inference requires a CUDA runtime")

    rows = read_test_rows(args.metadata, args.image_root)
    expected_ids = {row["image_id"] for row in rows}
    all_predictions = []
    fold_predictions = {}
    raw_fold_predictions = {}
    checkpoint_hashes = {}

    for fold in range(1, 6):
        run_dir = args.run_root / f"C3_Z26_C3_ROI_V2_FOLD_{fold}"
        model, config = load_model(run_dir, device)
        view_predictions = []
        for view_index, (rotation, flip) in enumerate(LOCKED_VIEWS, start=1):
            print(
                f"fold={fold}/5 view={view_index}/10 rotation={rotation:g} flip={flip} start",
                flush=True,
            )
            values = predict_view(
                model, rows, config, rotation, flip, device,
                args.batch_size, args.num_workers, amp=True,
            )
            view_predictions.append(values)
            all_predictions.append(values)
            print(f"fold={fold}/5 view={view_index}/10 done", flush=True)
        fold_predictions[fold] = aggregate_predictions(view_predictions, expected_ids)
        raw_fold_predictions[fold] = view_predictions[4]  # locked rotation=0, flip=False
        checkpoint_hashes[str(fold)] = sha256_file(run_dir / "best_mae.ckpt")
        del model
        torch.cuda.empty_cache()

    final_predictions = aggregate_predictions(all_predictions, expected_ids)
    raw_predictions = aggregate_predictions(list(raw_fold_predictions.values()), expected_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in rows:
        image_id = row["image_id"]
        item = {
            "image_id": image_id,
            "sex": row["sex"],
            **{f"fold_{fold}_tta_months": fold_predictions[fold][image_id] for fold in range(1, 6)},
            "raw_5fold_prediction_months": raw_predictions[image_id],
            "prediction_months": final_predictions[image_id],
        }
        if "target_months" in row:
            item["target_months"] = row["target_months"]
            item["absolute_error"] = abs(final_predictions[image_id] - row["target_months"])
        output_rows.append(item)

    prediction_path = args.output_dir / "C3_Z26_C3_ROI_V2_TTA_TEST_predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "status": "PASS",
        "protocol": "locked OOF-selected C3-Z26 V2; five folds; rotations -10,-5,0,5,10 x flip/no-flip; uniform mean of 50 predictions",
        "protocol_locked_from_oof": True,
        "test_used_for_model_selection": False,
        "count": len(rows),
        "views_per_fold": len(LOCKED_VIEWS),
        "fold_count": 5,
        "metadata_sha256": sha256_file(args.metadata),
        "checkpoint_sha256": checkpoint_hashes,
    }
    if "target_months" in rows[0]:
        target = np.asarray([row["target_months"] for row in rows], dtype=np.float64)
        prediction = np.asarray([final_predictions[row["image_id"]] for row in rows])
        raw_prediction = np.asarray([raw_predictions[row["image_id"]] for row in rows])
        report["metrics"] = {
            "raw_5fold": metric_arrays(target, raw_prediction),
            "tta_5fold_10view": metric_arrays(target, prediction),
        }
        report["tta_mae_bootstrap_95_ci"] = bootstrap_mae_ci(target, prediction)
        report["paired_tta_minus_raw"] = paired_bootstrap_delta(
            target, prediction, raw_prediction
        )
    report_path = args.output_dir / "C3_Z26_C3_ROI_V2_TTA_TEST_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"predictions={prediction_path}", flush=True)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
