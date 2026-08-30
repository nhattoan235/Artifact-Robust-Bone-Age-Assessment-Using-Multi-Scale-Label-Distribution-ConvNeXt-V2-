"""Run the same six model families on original test images and compare to cleaned images."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from artifact_eval.evaluate_models import (
    OUTPUT as CLEAN_OUTPUT,
    ROOT,
    _load_model_module,
    _load_specs,
    _parse_resolved_yaml,
    _predict,
    _roi_rows,
    compute_metrics,
    bootstrap_ci,
)

ORIGINAL_DIR = ROOT / "data/goc/boneage-test-dataset/boneage-test-dataset"
OUTPUT = ROOT / "artifact_eval/outputs/artifact_only_200_manual_v3"


def original_rows() -> list[dict]:
    gt_path = ROOT / "data/goc/rsna_test_ground_truth.csv"
    with gt_path.open(encoding="utf-8-sig", newline="") as handle:
        gt = {Path(row["image_ID"]).stem: row for row in csv.DictReader(handle)}
    images = sorted(ORIGINAL_DIR.glob("*.png"), key=lambda p: int(p.stem))
    if len(images) != 200 or {p.stem for p in images} != set(gt):
        raise RuntimeError("Original test set must contain exactly the 200 ground-truth IDs")
    return [{"image_id": p.stem, "sex": gt[p.stem]["sex"].strip().upper(), "target": float(gt[p.stem]["bone_age"]), "path": p} for p in images]


def read_clean_predictions() -> tuple[list[dict], dict[str, np.ndarray]]:
    path = CLEAN_OUTPUT / "artifact_only_200_all_model_predictions.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [k for k in rows[0] if k not in {"image_id", "sex", "target_months"}]
    values = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64) for key in keys}
    return rows, values


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = original_rows()
    roi_rows = _roi_rows(rows)
    model_module = _load_model_module()
    predictions: dict[str, np.ndarray] = {}
    fold_predictions: dict[str, list[np.ndarray]] = {}
    for model_id, fold, checkpoint, config_path, roi, sex_mode in _load_specs():
        print(f"[{model_id}] fold={fold} original input={'ROI' if roi else 'full'}", flush=True)
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Missing artifact: {checkpoint} / {config_path}")
        config = _parse_resolved_yaml(config_path) if config_path.suffix in {".yaml", ".yml"} else {
            k: v for section in __import__("tomllib").loads(config_path.read_text(encoding="utf-8")).values() if isinstance(section, dict) for k, v in section.items()
        }
        model = model_module.build_model(str(config["architecture"]), False, int(config.get("sex_embedding_dim", 16)), int(config.get("head_hidden_dim", 256)), float(config.get("dropout", 0.2)), int(config.get("age_class_count", 229)), sex_mode=sex_mode)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        fold_predictions.setdefault(model_id, []).append(_predict(model, config, roi_rows if roi else rows, device, roi))
        del state, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for model_id, values in fold_predictions.items():
        predictions[model_id] = np.mean(np.stack(values), axis=0)
    predictions["E1_P7_plus_D3_50_50"] = 0.5 * predictions["E1_P7_5FOLD"] + 0.5 * predictions["D3_5FOLD"]
    predictions["E1_P7_plus_C3_50_50"] = 0.5 * predictions["E1_P7_5FOLD"] + 0.5 * predictions["C3_ROI_5FOLD"]
    predictions["ALL_6_MODEL_FAMILIES_EQUAL"] = np.mean(np.stack([predictions[k] for k in ("E1_P7_5FOLD", "E0_IMAGE_ONLY", "E1_P10_VALIDATION", "E2_DUAL_OUTPUT", "D3_5FOLD", "C3_ROI_5FOLD")]), axis=0)

    target = np.asarray([r["target"] for r in rows], dtype=np.float64)
    clean_rows, clean_predictions = read_clean_predictions()
    if [r["image_id"] for r in clean_rows] != [r["image_id"] for r in rows]:
        raise RuntimeError("Original and cleaned rows are not aligned")
    comparison = {}
    for key, original in predictions.items():
        clean = clean_predictions[key]
        orig_metrics = compute_metrics(target, original) | {"bootstrap_95_ci_mae": bootstrap_ci(target, original)}
        clean_metrics = compute_metrics(target, clean) | {"bootstrap_95_ci_mae": bootstrap_ci(target, clean)}
        delta = original - clean
        comparison[key] = {
            "original": orig_metrics,
            "cleaned": clean_metrics,
            "cleaned_minus_original_mae": clean_metrics["mae_months"] - orig_metrics["mae_months"],
            "absolute_prediction_shift_mean_months": float(np.abs(delta).mean()),
            "prediction_correlation_original_cleaned": float(np.corrcoef(original, clean)[0, 1]),
        }
    with (OUTPUT / "original_vs_cleaned_all_model_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["image_id", "sex", "target_months"] + [f"{key}_original" for key in predictions] + [f"{key}_cleaned" for key in predictions]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"image_id": row["image_id"], "sex": row["sex"], "target_months": row["target"], **{f"{key}_original": float(value[i]) for key, value in predictions.items()}, **{f"{key}_cleaned": float(clean_predictions[key][i]) for key in predictions}})
    report = {"status": "PASS", "count": len(rows), "original_dataset": "data/goc/boneage-test-dataset/boneage-test-dataset", "cleaned_dataset": "artifact_only_200_manual_v3/cleaned", "same_ground_truth": True, "protocol": "same six model families, same checkpoints, no TTA, fixed 50/50 ensembles", "results": comparison}
    (OUTPUT / "ORIGINAL_VS_CLEANED_MODEL_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {"original_mae": value["original"]["mae_months"], "cleaned_mae": value["cleaned"]["mae_months"], "delta_cleaned_minus_original": value["cleaned_minus_original_mae"]} for key, value in comparison.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
