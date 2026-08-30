"""Evaluate all recovered model versions on the artifact-only cleaned set.

Primary input is the 200-image ``cleaned`` directory.  E1/E0/E2/D3 use whole
hand images. C3 uses the locked C3 ROI coordinates applied to those same cleaned
images. No model is retrained and no prediction is selected using the result.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data/model_eval_inventory"
ARTIFACT_ROOT = INVENTORY / "artifact_only_200_manual_v3/artifact_only_200_manual_v3"
OUTPUT = ROOT / "artifact_eval/outputs/artifact_only_200_manual_v3"


def parse_roi_bbox(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(float(x.strip())) for x in value.split(","))
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError(f"Invalid ROI bbox: {value}")
    return parts


def compute_metrics(targets: list[float] | np.ndarray, predictions: list[float] | np.ndarray) -> dict:
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    errors = np.abs(prediction - target)
    return {
        "count": int(errors.size),
        "mae_months": float(errors.mean()),
        "rmse_months": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "median_absolute_error_months": float(np.median(errors)),
        "accuracy_within_6_months": float(np.mean(errors <= 6)),
        "accuracy_within_12_months": float(np.mean(errors <= 12)),
        "accuracy_within_18_months": float(np.mean(errors <= 18)),
        "bias_months": float(np.mean(prediction - target)),
        "prediction_mean_months": float(prediction.mean()),
        "prediction_std_months": float(prediction.std()),
    }


def fuse_model_output(
    regression_normalized: list[float] | np.ndarray,
    distribution_probabilities: list[list[float]] | np.ndarray,
    target_mean: float,
    target_std: float,
    regression_weight: float,
) -> np.ndarray:
    """Fuse regression and LDL branches after converting both to months."""
    regression_months = np.asarray(regression_normalized, dtype=np.float64) * target_std + target_mean
    probabilities = np.asarray(distribution_probabilities, dtype=np.float64)
    age_axis = np.arange(probabilities.shape[1], dtype=np.float64)
    distribution_months = probabilities @ age_axis
    return regression_weight * regression_months + (1.0 - regression_weight) * distribution_months


def bootstrap_ci(targets: np.ndarray, predictions: np.ndarray, seed: int = 42, n: int = 10000) -> list[float]:
    errors = np.abs(predictions - targets)
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        means[i] = rng.choice(errors, size=errors.size, replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def build_model_registry() -> list[dict]:
    return [
        {"model_id": "E1_P7_5FOLD", "kind": "folds", "source": "P7_E1_FOLD*_CONVNEXT_TINY_SEX_A2"},
        {"model_id": "E0_IMAGE_ONLY", "kind": "single", "source": "P11_E0_IMAGE_ONLY_CONVNEXT_TINY_A2"},
        {"model_id": "E1_P10_VALIDATION", "kind": "single", "source": "P10_E1_VALIDATION_CONVNEXT_TINY_SEX_A2"},
        {"model_id": "E2_DUAL_OUTPUT", "kind": "single", "source": "P11_E2_DUAL_OUTPUT_CONVNEXT_TINY_A2"},
        {"model_id": "D3_5FOLD", "kind": "folds", "source": "d3_oof/runs/D3_OOF_V1"},
        {"model_id": "C3_ROI_5FOLD", "kind": "folds", "source": "c3_roi/runs/C3_ROI_V1"},
    ]


def _load_model_module():
    path = INVENTORY / "P11_E2_DUAL_OUTPUT_CONVNEXT_TINY_A2/code/p1_baseline/model.py"
    spec = importlib.util.spec_from_file_location("artifact_eval_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_resolved_yaml(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        else:
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                value = value.strip('"\'')
        result[key.strip()] = value
    return result


def _rows() -> list[dict]:
    gt_path = ROOT / "data/goc/rsna_test_ground_truth.csv"
    with gt_path.open(encoding="utf-8-sig", newline="") as handle:
        gt = {Path(row["image_ID"]).stem: row for row in csv.DictReader(handle)}
    image_dir = ARTIFACT_ROOT / "cleaned"
    images = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))
    if len(images) != 200:
        raise RuntimeError(f"Expected 200 cleaned images, found {len(images)}")
    if {p.stem for p in images} != set(gt):
        raise RuntimeError("Cleaned image IDs do not match ground truth IDs")
    rows = []
    for image in images:
        item = gt[image.stem]
        rows.append({"image_id": image.stem, "sex": item["sex"].strip().upper(), "target": float(item["bone_age"]), "path": image})
    return rows


def _roi_rows(rows: list[dict]) -> list[dict]:
    manifest = ROOT / "c3_roi/cache/C3_ROI_V1_TEST/test_roi_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        by_id = {row["image_id"]: row for row in csv.DictReader(handle)}
    out = []
    for row in rows:
        if row["image_id"] not in by_id:
            raise RuntimeError(f"Missing C3 ROI coordinate for {row['image_id']}")
        out.append({**row, "roi_bbox": by_id[row["image_id"]]["roi_bbox"], "roi_mode": by_id[row["image_id"]]["roi_mode"]})
    return out


class EvaluationDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, roi: bool = False):
        self.rows, self.image_size, self.roi = rows, image_size, roi

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as source:
            image = source.convert("L")
            if self.roi:
                x, y, width, height = parse_roi_bbox(row["roi_bbox"])
                image = image.crop((x, y, x + width, y + height))
            side = max(image.size)
            canvas = Image.new("L", (side, side), color=0)
            canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
            image = TF.resize(canvas, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
            tensor = TF.pil_to_tensor(image).float().div_(255).repeat(3, 1, 1)
            tensor = TF.normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return {"image": tensor, "sex": torch.tensor([1.0 if row["sex"] == "M" else 0.0]), "image_id": row["image_id"]}


def _model_and_config(model_module, checkpoint: Path, config_path: Path, sex_mode: str | None = None):
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config = {k: v for section in raw.values() if isinstance(section, dict) for k, v in section.items()}
    mode = sex_mode or str(config.get("sex_mode", "embedding"))
    model = model_module.build_model(str(config["architecture"]), False, int(config.get("sex_embedding_dim", 16)), int(config.get("head_hidden_dim", 256)), float(config.get("dropout", 0.2)), int(config.get("age_class_count", 229)), sex_mode=mode)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, config, mode


def _resolved_pair(run_dir: Path) -> tuple[Path, dict]:
    config = _parse_resolved_yaml(run_dir / "config_resolved.yaml")
    return run_dir / "best_mae.ckpt", config


def _predict(model, config: dict, rows: list[dict], device: torch.device, roi: bool) -> np.ndarray:
    dataset = EvaluationDataset(rows, int(config["image_size"]), roi=roi)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    values = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            sex = batch["sex"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(image, sex)
            if isinstance(output, dict):
                regression = output["regression"].float()
                distribution = torch.softmax(output["distribution_logits"].float(), dim=1)
                normalized = torch.as_tensor(
                    fuse_model_output(
                        regression.detach().cpu().numpy(),
                        distribution.detach().cpu().numpy(),
                        float(config["target_mean"]),
                        float(config["target_std"]),
                        float(config.get("regression_inference_weight", 0.5)),
                    ), device=device, dtype=torch.float32,
                )
                values.extend(normalized.cpu().numpy().tolist())
                continue
            else:
                normalized = output.float()
            values.extend((normalized.cpu().numpy() * float(config["target_std"]) + float(config["target_mean"])).tolist())
    return np.asarray(values, dtype=np.float64)


def _load_specs():
    specs = []
    for fold in range(1, 6):
        base = INVENTORY / f"P7_E1_FOLD{fold}_CONVNEXT_TINY_SEX_A2"
        specs.append(("E1_P7_5FOLD", fold, base / "model/best_model.pt", base / f"config/fold_{fold}.toml", False, "embedding"))
    base = INVENTORY / "P11_E0_IMAGE_ONLY_CONVNEXT_TINY_A2"
    specs.append(("E0_IMAGE_ONLY", 1, base / "model/best_mae.ckpt", base / "config/p11_e0_image_only_seed42.toml", False, "none"))
    base = INVENTORY / "P10_E1_VALIDATION_CONVNEXT_TINY_SEX_A2"
    specs.append(("E1_P10_VALIDATION", 1, base / "model/best_mae.ckpt", base / "config/p10_b0_p2_control.toml", False, "embedding"))
    base = INVENTORY / "P11_E2_DUAL_OUTPUT_CONVNEXT_TINY_A2"
    specs.append(("E2_DUAL_OUTPUT", 1, base / "model/best_mae.ckpt", base / "config/p11_e2_shared_dual_output_seed42.toml", False, "dual_output"))
    for fold in range(1, 6):
        run = ROOT / f"d3_oof/runs/D3_OOF_V1/D3_OOF_V1_FOLD_{fold}"
        checkpoint, _ = _resolved_pair(run)
        specs.append(("D3_5FOLD", fold, checkpoint, run / "config_resolved.yaml", False, "embedding"))
        run = ROOT / f"c3_roi/runs/C3_ROI_V1/C3_ROI_V1_FOLD_{fold}"
        checkpoint, _ = _resolved_pair(run)
        specs.append(("C3_ROI_5FOLD", fold, checkpoint, run / "config_resolved.yaml", True, "embedding"))
    return specs


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _rows()
    roi_rows = _roi_rows(rows)
    model_module = _load_model_module()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = {}
    fold_predictions = {}
    for model_id, fold, checkpoint, config_path, roi, sex_mode in _load_specs():
        print(f"[{model_id}] fold={fold} device={device} input={'ROI' if roi else 'cleaned_full'}", flush=True)
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Missing artifact for {model_id} fold {fold}: {checkpoint} / {config_path}")
        config = _parse_resolved_yaml(config_path) if config_path.suffix in {".yaml", ".yml"} else {k: v for section in tomllib.loads(config_path.read_text(encoding="utf-8")).values() if isinstance(section, dict) for k, v in section.items()}
        model = model_module.build_model(str(config["architecture"]), False, int(config.get("sex_embedding_dim", 16)), int(config.get("head_hidden_dim", 256)), float(config.get("dropout", 0.2)), int(config.get("age_class_count", 229)), sex_mode=sex_mode)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        pred = _predict(model, config, roi_rows if roi else rows, device, roi)
        fold_predictions.setdefault(model_id, []).append(pred)
        del state, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for model_id, values in fold_predictions.items():
        predictions[model_id] = np.mean(np.stack(values), axis=0)

    target = np.asarray([r["target"] for r in rows], dtype=np.float64)
    all_results = {key: compute_metrics(target, value) | {"bootstrap_95_ci_mae": bootstrap_ci(target, value)} for key, value in predictions.items()}
    predictions["E1_P7_plus_D3_50_50"] = 0.5 * predictions["E1_P7_5FOLD"] + 0.5 * predictions["D3_5FOLD"]
    predictions["E1_P7_plus_C3_50_50"] = 0.5 * predictions["E1_P7_5FOLD"] + 0.5 * predictions["C3_ROI_5FOLD"]
    predictions["ALL_6_MODEL_FAMILIES_EQUAL"] = np.mean(np.stack([predictions[k] for k in ("E1_P7_5FOLD", "E0_IMAGE_ONLY", "E1_P10_VALIDATION", "E2_DUAL_OUTPUT", "D3_5FOLD", "C3_ROI_5FOLD")]), axis=0)
    for key in list(predictions)[len(all_results):]:
        all_results[key] = compute_metrics(target, predictions[key]) | {"bootstrap_95_ci_mae": bootstrap_ci(target, predictions[key])}

    sex = np.asarray([r["sex"] for r in rows])
    for key, pred in predictions.items():
        all_results[key]["by_sex"] = {s: compute_metrics(target[sex == s], pred[sex == s]) for s in ("F", "M")}
    with (OUTPUT / "artifact_only_200_all_model_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["image_id", "sex", "target_months"] + list(predictions)
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"image_id": row["image_id"], "sex": row["sex"], "target_months": row["target"], **{key: float(value[i]) for key, value in predictions.items()}})
    report = {
        "status": "PASS",
        "dataset": "artifact_only_200_manual_v3/cleaned",
        "dataset_zip": "data/artifact_only_200_manual_v3.zip",
        "count": len(rows),
        "ground_truth_source": "data/goc/rsna_test_ground_truth.csv",
        "ground_truth_used_for_metric_only": True,
        "inference_mode": "no TTA; cleaned full-hand input for E1/E0/E2/D3; locked C3 ROI coordinates applied to cleaned input",
        "model_families": build_model_registry(),
        "results": all_results,
        "artifacts": {"predictions_csv": str((OUTPUT / "artifact_only_200_all_model_predictions.csv").relative_to(ROOT))},
    }
    (OUTPUT / "artifact_only_200_all_model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["mae_months"] for k, v in all_results.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
