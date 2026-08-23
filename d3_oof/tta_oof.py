from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from p1_baseline.model import build_model


ROTATIONS = (-10.0, -5.0, 0.0, 5.0, 10.0)


def scalar(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def resolved_config(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            result[key.strip()] = scalar(value)
    return result


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


class TTADataset(Dataset):
    def __init__(self, rows, image_size: int, rotation: float, flip: bool):
        self.rows = rows
        self.image_size = image_size
        self.rotation = rotation
        self.flip = flip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = pad_square(source.convert("L"))
            image = TF.resize(
                image, [self.image_size, self.image_size],
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            )
            if self.flip:
                image = TF.hflip(image)
            image = TF.rotate(
                image, self.rotation, interpolation=InterpolationMode.BILINEAR,
                expand=False, fill=0,
            )
            tensor = TF.pil_to_tensor(image).float().div_(255.0).repeat(3, 1, 1)
            tensor = TF.normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return {
            "image": tensor,
            "sex": torch.tensor([1.0 if row["sex"] == "M" else 0.0], dtype=torch.float32),
            "image_id": row["image_id"],
        }


def load_fold_model(fold: int, root: Path, device: torch.device):
    run_dir = root / f"d3_oof/runs/D3_OOF_V1/D3_OOF_V1_FOLD_{fold}"
    config = resolved_config(run_dir / "config_resolved.yaml")
    model = build_model(
        str(config["architecture"]), False, int(config["sex_embedding_dim"]),
        int(config["head_hidden_dim"]), float(config["dropout"]),
        int(config["age_class_count"]),
    ).to(device)
    checkpoint = torch.load(run_dir / "best_mae.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


def predict(model, rows, config, rotation, flip, device, batch_size, workers, amp):
    loader = DataLoader(
        TTADataset(rows, int(config["image_size"]), rotation, flip),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    result = {}
    age_axis = torch.arange(int(config["age_class_count"]), device=device, dtype=torch.float32)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            sex = batch["sex"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                output = model(images, sex)
            regression = output["regression"].float() * float(config["target_std"]) + float(config["target_mean"])
            probabilities = torch.softmax(output["distribution_logits"].float(), dim=1)
            distribution = probabilities @ age_axis
            fused = float(config["regression_inference_weight"]) * regression + (
                1.0 - float(config["regression_inference_weight"])
            ) * distribution
            for image_id, value in zip(batch["image_id"], fused.detach().cpu().tolist()):
                result[str(image_id)] = float(value)
    return result


def bootstrap_ci(delta: np.ndarray, seed: int, samples: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(samples)
    for start in range(0, samples, 250):
        stop = min(start + 250, samples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        values[start:stop] = delta[indices].mean(axis=1)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("d3_oof/outputs/D3_TTA_OOF_V1"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-per-fold", type=int)
    args = parser.parse_args()
    root = args.workspace.resolve()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest_rows = {}
    for fold in range(1, 6):
        manifest = root / f"c1_curated/outputs/C1_AUDIT_OOF_V1/fold_{fold}_validation.csv"
        for row in read_csv(manifest):
            image_id = str(int(float(row["image_id"])))
            row = dict(row)
            row["image_id"] = image_id
            row["image_path"] = str(root / Path(row["image_path"]))
            manifest_rows[image_id] = row

    d3_rows = read_csv(root / "d3_oof/outputs/D3_OOF_V1/D3_OOF_predictions.csv")
    p9_rows = read_csv(root / "p9_inference/outputs/P9_I_TTA_BIAS_OOF/P9_I_TTA_BIAS_OOF_predictions.csv")
    p9 = {str(int(float(row["image_id"]))): row for row in p9_rows}
    rows = []
    for row in d3_rows:
        image_id = str(int(float(row["image_id"])))
        source = manifest_rows[image_id]
        if not Path(source["image_path"]).is_file():
            raise FileNotFoundError(source["image_path"])
        if image_id not in p9:
            raise RuntimeError(f"Missing E1 TTA prediction for ID={image_id}")
        rows.append({
            "image_id": image_id,
            "fold": int(row["fold"]),
            "target_months": float(row["target_months"]),
            "sex": row["sex"],
            "image_path": source["image_path"],
            "d3_oof_prediction_months": float(row["prediction_months"]),
            "e1_tta_prediction_months": float(p9[image_id]["tta_prediction_months"]),
        })
    rows.sort(key=lambda row: int(row["image_id"]))
    if args.limit_per_fold is not None:
        rows = [
            row
            for fold in range(1, 6)
            for row in [item for item in rows if item["fold"] == fold][:args.limit_per_fold]
        ]
    by_fold = {fold: [row for row in rows if row["fold"] == fold] for fold in range(1, 6)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 6):
        model, config = load_fold_model(fold, root, device)
        transforms = {}
        for rotation in ROTATIONS:
            for flip in (False, True):
                key = f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}"
                transforms[key] = predict(
                    model, by_fold[fold], config, rotation, flip, device,
                    args.batch_size, args.num_workers, args.amp,
                )
        for row in by_fold[fold]:
            values = np.asarray([
                transforms[f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}"][row["image_id"]]
                for rotation in ROTATIONS for flip in (False, True)
            ])
            row["d3_tta_prediction_months"] = float(values.mean())
            row["d3_tta_std_months"] = float(values.std())
            row["d3_raw_recomputed_months"] = transforms["rot_0_no_flip"][row["image_id"]]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fieldnames = [
        "image_id", "fold", "target_months", "sex",
        "d3_oof_prediction_months", "d3_raw_recomputed_months",
        "d3_tta_prediction_months", "d3_tta_std_months",
        "e1_tta_prediction_months",
    ]
    with (args.output_dir / "D3_TTA_OOF_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    target = np.asarray([row["target_months"] for row in rows])
    d3_raw = np.asarray([row["d3_oof_prediction_months"] for row in rows])
    d3_recomputed = np.asarray([row["d3_raw_recomputed_months"] for row in rows])
    d3_tta = np.asarray([row["d3_tta_prediction_months"] for row in rows])
    e1_tta = np.asarray([row["e1_tta_prediction_months"] for row in rows])
    ensemble = 0.5 * e1_tta + 0.5 * d3_tta
    raw_diff = d3_recomputed - d3_raw
    delta = np.abs(ensemble - target) - np.abs(e1_tta - target)
    report = {
        "status": "PASS",
        "test_used": False,
        "count": len(rows),
        "subset": args.limit_per_fold is not None,
        "tta": {"rotations": list(ROTATIONS), "flips": [False, True], "amp": args.amp},
        "raw_reproduction_check": {
            "mean_abs_diff_months": float(np.abs(raw_diff).mean()),
            "max_abs_diff_months": float(np.abs(raw_diff).max()),
        },
        "mae": {
            "E1_TTA": float(np.abs(e1_tta - target).mean()),
            "D3_TTA": float(np.abs(d3_tta - target).mean()),
            "E1_TTA_D3_TTA_50_50": float(np.abs(ensemble - target).mean()),
        },
        "ensemble_vs_E1_TTA": {
            "delta_mae_candidate_minus_baseline": float(delta.mean()),
            "paired_bootstrap_95_ci": bootstrap_ci(delta, 20260823),
            "candidate_better_images": int((delta < 0).sum()),
            "candidate_worse_images": int((delta > 0).sum()),
            "prediction_correlation": float(np.corrcoef(ensemble, e1_tta)[0, 1]),
            "residual_correlation": float(np.corrcoef(ensemble - target, e1_tta - target)[0, 1]),
        },
        "tta_disagreement": {
            "mean_std_months": float(np.mean([row["d3_tta_std_months"] for row in rows])),
            "p95_std_months": float(np.quantile([row["d3_tta_std_months"] for row in rows], 0.95)),
        },
    }
    (args.output_dir / "D3_TTA_OOF_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
