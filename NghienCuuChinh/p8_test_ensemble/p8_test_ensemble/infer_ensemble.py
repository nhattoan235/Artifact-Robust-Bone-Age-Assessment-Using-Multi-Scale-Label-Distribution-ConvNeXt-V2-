from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from p1_baseline.data import BoneAgeDataset
from p1_baseline.model import build_model


def parse_scalar(value: str):
    value = value.strip()
    if value in {"null", "None"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("["):
        return json.loads(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_resolved_config(value: bytes) -> dict:
    output = {}
    for line in value.decode("utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        output[key.strip()] = parse_scalar(raw)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_test_rows(images: Path, sex_csv: Path, ground_truth: Path) -> tuple[list[dict], dict[str, dict]]:
    with sex_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        sex_rows = list(csv.DictReader(handle))
    with ground_truth.open("r", encoding="utf-8-sig", newline="") as handle:
        gt_rows = list(csv.DictReader(handle))
    sex_map = {str(int(row["Case ID"])): row["Sex"].strip().upper() for row in sex_rows}
    gt_map = {str(int(row["patient_ID"])): row for row in gt_rows}
    image_map = {path.stem: path.resolve() for path in images.glob("*.png")}
    ids = sorted(image_map, key=int)
    if len(ids) != 200 or set(ids) != set(sex_map) or set(ids) != set(gt_map):
        raise RuntimeError(f"Test không khớp: images={len(ids)}, sex={len(sex_map)}, gt={len(gt_map)}")
    if any(sex_map[item] != gt_map[item]["sex"].strip().upper() for item in ids):
        raise RuntimeError("Sex test không khớp ground truth")
    rows = [{
        "split": "test", "image_id": item, "bone_age_months": "0",
        "sex": sex_map[item], "image_path": str(image_map[item]), "sha256": "", "readable": "True",
    } for item in ids]
    return rows, gt_map


def metrics(predictions: list[dict]) -> dict:
    errors = np.asarray([float(item["absolute_error"]) for item in predictions], dtype=np.float64)
    return {
        "count": int(errors.size), "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))), "median_ae": float(np.median(errors)),
        "accuracy_6m": float(np.mean(errors <= 6)),
        "accuracy_12m": float(np.mean(errors <= 12)),
        "accuracy_18m": float(np.mean(errors <= 18)),
        "prediction_mean": float(np.mean([float(item["prediction_months"]) for item in predictions])),
        "prediction_std": float(np.std([float(item["prediction_months"]) for item in predictions])),
    }


def bootstrap_ci(predictions: list[dict], samples: int, seed: int) -> list[float]:
    errors = [float(item["absolute_error"]) for item in predictions]
    rng = random.Random(seed)
    values = [statistics.fmean(errors[rng.randrange(len(errors))] for _ in errors) for _ in range(samples)]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser(description="P8: 5-fold ensemble trên RSNA test 200")
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--test-sex", type=Path, required=True)
    parser.add_argument("--test-ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("p8_test_ensemble/outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu CUDA nhưng không thấy GPU")
    rows, gt_map = read_test_rows(args.test_images, args.test_sex, args.test_ground_truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_fold: list[list[dict]] = []
    with zipfile.ZipFile(args.results_zip) as archive:
        if archive.testzip():
            raise RuntimeError("Results ZIP CRC lỗi")
        for fold in range(1, 6):
            prefix = f"results/P7_FINAL_V3_FOLD_{fold}/"
            config = load_resolved_config(archive.read(prefix + "config_resolved.yaml"))
            model = build_model(
                str(config["architecture"]), False, int(config["sex_embedding_dim"]),
                int(config["head_hidden_dim"]), float(config["dropout"]), int(config["age_class_count"]),
            ).to(device)
            state = torch.load(io.BytesIO(archive.read(prefix + "best_model.pt")), map_location=device, weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            dataset = BoneAgeDataset(
                rows, int(config["image_size"]), float(config["target_mean"]), float(config["target_std"]),
                train=False, preprocessing="none", image_root="",
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            fold_predictions = []
            use_amp = device.type == "cuda"
            with torch.inference_mode():
                for batch in loader:
                    images = batch["image"].to(device, non_blocking=True)
                    sex = batch["sex"].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        output = model(images, sex)
                    if isinstance(output, dict):
                        prediction_norm = output["regression"]
                    else:
                        prediction_norm = output
                    prediction = prediction_norm.float().cpu().numpy() * float(config["target_std"]) + float(config["target_mean"])
                    for image_id, value in zip(batch["image_id"], prediction.tolist()):
                        target = float(gt_map[str(image_id)]["bone_age"])
                        fold_predictions.append({
                            "image_id": str(image_id), "sex": gt_map[str(image_id)]["sex"],
                            "target_months": target, "prediction_months": float(value),
                            "absolute_error": abs(float(value) - target), "fold": fold,
                        })
            if len(fold_predictions) != 200:
                raise RuntimeError(f"Fold {fold} không đủ 200 prediction")
            with (args.output_dir / f"P8_fold_{fold}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(fold_predictions[0]))
                writer.writeheader(); writer.writerows(fold_predictions)
            fold_metrics = metrics(fold_predictions)
            print(json.dumps({"fold": fold, **fold_metrics}, ensure_ascii=False), flush=True)
            per_fold.append(fold_predictions)
            del model, loader, dataset, state
            if device.type == "cuda":
                torch.cuda.empty_cache()

    by_id = {item["image_id"]: [] for item in per_fold[0]}
    for fold_predictions in per_fold:
        for item in fold_predictions:
            by_id[item["image_id"]].append(float(item["prediction_months"]))
    ensemble = []
    for row in rows:
        image_id = row["image_id"]
        target = float(gt_map[image_id]["bone_age"])
        prediction = statistics.fmean(by_id[image_id])
        ensemble.append({
            "image_id": image_id, "sex": gt_map[image_id]["sex"],
            "target_months": target, "prediction_months": prediction,
            "absolute_error": abs(prediction - target), "model_count": len(by_id[image_id]),
        })
    output_csv = args.output_dir / "P8_ensemble_predictions.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ensemble[0]))
        writer.writeheader(); writer.writerows(ensemble)
    report = {
        "status": "PASS", "protocol": "fixed 5-fold equal-weight ensemble; no test tuning",
        "test_used_for_model_selection": False, "test_images": 200,
        "ground_truth_sha256": sha256_file(args.test_ground_truth),
        "fold_metrics": [metrics(item) for item in per_fold],
        "ensemble_metrics": metrics(ensemble),
        "ensemble_bootstrap_95_ci_mae": bootstrap_ci(ensemble, args.bootstrap, args.bootstrap_seed),
        "bootstrap_samples": args.bootstrap, "bootstrap_seed": args.bootstrap_seed,
        "benchmarks": {
            "deeplasia_rsna_test_mae_months": 3.87,
            "bram_2025_rsna_test_mae_months": 3.68,
            "note": "Benchmarks are reported literature values on the same nominal 200-image RSNA test; compare protocol details and uncertainty.",
        },
    }
    (args.output_dir / "P8_test_ensemble_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
