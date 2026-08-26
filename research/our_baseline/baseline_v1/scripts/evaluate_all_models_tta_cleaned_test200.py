"""Evaluate every packaged model on the cleaned 200-image test set.

For each checkpoint this script reports:
  * raw: the model's native deterministic preprocessing, no TTA;
  * tta: mean of five rotations (-10, -5, 0, 5, 10 degrees) with and without
    horizontal flip.

The test labels are read only for this exploratory audit. They must not be
used to select weights or hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from evaluate_all_models_cleaned_test200 import (
    BASELINE_ROOT,
    DEFAULT_IMAGES,
    DEFAULT_LABELS,
    DEFAULT_OUTPUT,
    MEAN,
    PACKAGES,
    STD,
    OldBaselineModel,
    checkpoint_state,
    load_resolved_yaml,
    load_test_frame,
    load_toml_flat,
    metrics,
    model_specs,
    normalize_id,
)


# Locked TTA protocol used by the earlier EXP006 benchmark.
ROTATIONS = (-10.0, -5.0, 0.0, 5.0, 10.0)
FLIPS = (False, True)


def pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


class TTADataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        kind: str,
        image_size: int,
        rotation: float,
        flip: bool,
        image_normalization: str = "imagenet",
    ):
        self.frame = frame.reset_index(drop=True)
        self.kind = kind
        self.image_size = int(image_size)
        self.rotation = float(rotation)
        self.flip = bool(flip)
        self.image_normalization = image_normalization

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("L")
            if self.kind == "p1":
                image = pad_square(image)
                image = TF.resize(
                    image,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
            else:
                # Match boneage_baseline.py: direct square resize for the old
                # official checkpoints, before applying inference transforms.
                image = image.resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BILINEAR,
                )
            if self.flip:
                image = TF.hflip(image)
            if self.rotation != 0:
                image = TF.rotate(
                    image,
                    angle=self.rotation,
                    interpolation=InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0,
                )
            tensor = TF.pil_to_tensor(image).float().div_(255.0).repeat(3, 1, 1)
            if self.image_normalization == "imagenet":
                tensor = TF.normalize(tensor, MEAN, STD)
            elif self.image_normalization == "per_image_zscore":
                tensor = (tensor - tensor.mean()) / tensor.std().clamp_min(1e-6)
        sex = torch.tensor([1.0 if row.sex == "M" else 0.0], dtype=torch.float32)
        return tensor, sex, str(row.image_id)


def read_config(path: Path, code_root: Path):
    if str(code_root.resolve()) not in sys.path:
        sys.path.insert(0, str(code_root.resolve()))
    # Import only config/model. p1_baseline.data imports cv2, which is not
    # needed for inference because this script owns its TTA dataset.
    from p1_baseline.config import Config, load_config

    if path.suffix.lower() in {".yaml", ".yml"}:
        values = load_resolved_yaml(path)
        return Config(**values)
    return load_config(path)


def build_p1_model(config, code_root: Path, checkpoint: Path, device: torch.device):
    if str(code_root.resolve()) not in sys.path:
        sys.path.insert(0, str(code_root.resolve()))
    from p1_baseline.model import build_model

    model = build_model(
        config.architecture,
        False,
        config.sex_embedding_dim,
        config.head_hidden_dim,
        config.dropout,
        config.age_class_count,
        sex_mode=config.sex_mode,
    ).to(device)
    state = checkpoint_state(checkpoint, device)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    model.eval()
    return model


def predict_transform(
    model: nn.Module,
    frame: pd.DataFrame,
    spec: dict,
    config,
    rotation: float,
    flip: bool,
    device: torch.device,
    batch_size: int,
    amp: bool,
):
    image_size = int(getattr(config, "image_size", 512)) if config is not None else 512
    normalization = getattr(config, "image_normalization", "imagenet") if config is not None else "imagenet"
    dataset = TTADataset(
        frame,
        spec["kind"],
        image_size,
        rotation,
        flip,
        normalization,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values = []
    with torch.inference_mode():
        for images, sex, _ in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                output = model(images.to(device), sex.to(device))
            if isinstance(output, dict):
                output = output["regression"]
            output = output.float().cpu().numpy()
            if config is not None:
                output = output * float(config.target_std) + float(config.target_mean)
            values.append(output)
    return np.concatenate(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Dùng FP16 khi chạy CUDA")
    parser.add_argument("--cpu-threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--spec-index", type=int, default=0, help="1-based model index; 0 evaluates all models")
    args = parser.parse_args()

    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads phải >= 1")
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.cpu_threads, os.cpu_count() or 1)))
    except RuntimeError:
        # Safe when the script is imported after another torch operation.
        pass

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    frame = load_test_frame(args.labels, args.image_root)
    if len(frame) != 200:
        raise ValueError(f"Kỳ vọng 200 ảnh, nhận được {len(frame)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = frame.target_months.to_numpy(dtype=np.float64)
    predictions = frame[["image_id", "target_months", "sex"]].copy()
    report = {
        "evaluation": "exploratory_cleaned_test_200_all_models_raw_vs_tta",
        "test_labels_used": True,
        "selection_warning": "Không dùng kết quả này để chọn hyperparameter/weight; đây là đánh giá thăm dò trên test 200.",
        "image_root": str(args.image_root.resolve()),
        "labels": str(args.labels.resolve()),
        "image_count": int(len(frame)),
        "device": str(device),
        "tta_protocol": {
            "rotations_degrees": list(ROTATIONS),
            "horizontal_flip": list(FLIPS),
            "views_per_model": len(ROTATIONS) * len(FLIPS),
            "aggregation": "equal mean of 10 views; rot_0_no_flip is raw",
        },
        "models": {},
        "groups": {},
    }

    model_predictions: dict[str, dict[str, np.ndarray]] = {}
    group_predictions: dict[str, dict[str, list[np.ndarray]]] = {}
    specs = model_specs()
    if args.spec_index:
        if not 1 <= args.spec_index <= len(specs):
            raise ValueError(f"--spec-index phải nằm trong 1..{len(specs)}")
        specs = [specs[args.spec_index - 1]]
        report["selected_spec_index"] = args.spec_index
    for index, spec in enumerate(specs, start=1):
        for key in ["checkpoint", "config", "code"]:
            if key in spec and not spec[key].exists():
                raise FileNotFoundError(f"Thiếu {key} của {spec['name']}: {spec[key]}")
        print(f"[{index}/{len(specs)}] {spec['name']} LOAD device={device}", flush=True)
        config = None
        if spec["kind"] == "old":
            model = OldBaselineModel().to(device)
            state = checkpoint_state(spec["checkpoint"], device)
            model.load_state_dict(state["model"] if "model" in state else state, strict=True)
            model.eval()
        else:
            config = read_config(spec["config"], spec["code"])
            model = build_p1_model(config, spec["code"], spec["checkpoint"], device)

        per_view = {}
        for rotation in ROTATIONS:
            for flip in FLIPS:
                view_name = f"rot_{rotation:g}_{'flip' if flip else 'no_flip'}"
                print(f"[{spec['name']}] {view_name} START", flush=True)
                per_view[view_name] = predict_transform(
                    model,
                    frame,
                    spec,
                    config,
                    rotation,
                    flip,
                    device,
                    args.batch_size,
                    args.amp,
                )
                print(f"[{spec['name']}] {view_name} DONE", flush=True)
        raw = per_view["rot_0_no_flip"]
        tta = np.mean(np.stack(list(per_view.values())), axis=0)
        model_predictions[spec["name"]] = {"raw": raw, "tta": tta}
        predictions[f"pred_{spec['name']}_raw"] = raw
        predictions[f"pred_{spec['name']}_tta"] = tta
        report["models"][spec["name"]] = {
            "group": spec["group"],
            "checkpoint": str(spec["checkpoint"].resolve()),
            "raw": metrics(target, raw),
            "tta": metrics(target, tta),
            "delta_tta_minus_raw": {
                "mae_months": float(metrics(target, tta)["mae_months"] - metrics(target, raw)["mae_months"]),
                "rmse_months": float(metrics(target, tta)["rmse_months"] - metrics(target, raw)["rmse_months"]),
            },
        }
        group_predictions.setdefault(spec["group"], {"raw": [], "tta": []})["raw"].append(raw)
        group_predictions[spec["group"]]["tta"].append(tta)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for group, values in group_predictions.items():
        raw_group = np.mean(np.stack(values["raw"]), axis=0)
        tta_group = np.mean(np.stack(values["tta"]), axis=0)
        predictions[f"pred_group_{group}_raw"] = raw_group
        predictions[f"pred_group_{group}_tta"] = tta_group
        report["groups"][group] = {
            "model_count": len(values["raw"]),
            "raw": metrics(target, raw_group),
            "tta": metrics(target, tta_group),
            "delta_tta_minus_raw": {
                "mae_months": float(metrics(target, tta_group)["mae_months"] - metrics(target, raw_group)["mae_months"]),
                "rmse_months": float(metrics(target, tta_group)["rmse_months"] - metrics(target, raw_group)["rmse_months"]),
            },
        }

    all_raw = np.mean(np.stack([model_predictions[s["name"]]["raw"] for s in specs]), axis=0)
    all_tta = np.mean(np.stack([model_predictions[s["name"]]["tta"] for s in specs]), axis=0)
    all_label = "all_15_models" if len(specs) == 15 else "selected_models"
    predictions[f"pred_{all_label}_raw"] = all_raw
    predictions[f"pred_{all_label}_tta"] = all_tta
    report["groups"][all_label] = {
        "model_count": len(specs),
        "raw": metrics(target, all_raw),
        "tta": metrics(target, all_tta),
        "delta_tta_minus_raw": {
            "mae_months": float(metrics(target, all_tta)["mae_months"] - metrics(target, all_raw)["mae_months"]),
            "rmse_months": float(metrics(target, all_tta)["rmse_months"] - metrics(target, all_raw)["rmse_months"]),
        },
        "note": "Equal-weight mean across the evaluated checkpoints; not used for model selection.",
    }

    suffix = "all_models" if len(specs) == 15 else f"model_{args.spec_index:02d}"
    prediction_path = args.output_dir / f"cleaned_test200_{suffix}_raw_vs_tta_predictions.csv"
    report_path = args.output_dir / f"cleaned_test200_{suffix}_raw_vs_tta_report.json"
    predictions.to_csv(prediction_path, index=False)
    report["prediction_file"] = str(prediction_path.resolve())
    qc_path = args.image_root.parent / "artifact_only_qc.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        report["qc_summary"] = {
            "rows": int(len(qc)),
            "status_counts": {str(k): int(v) for k, v in qc["Status"].value_counts().items()},
            "pixel_preservation_pass_count": int(qc["pixel_preservation_pass"].sum()),
            "changed_pct_mean": float(qc["changed_pct"].mean()),
        }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
