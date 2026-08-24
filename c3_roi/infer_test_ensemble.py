"""Run fixed C3-ROI 5-fold test inference and combine with locked E1 predictions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p1_baseline.data import BoneAgeDataset
from p1_baseline.model import build_model


def parse_value(value: str):
    value = value.strip().strip('"').strip("'")
    if value.lower() == "true": return True
    if value.lower() == "false": return False
    if value.lower() in {"null", "none"}: return None
    try: return int(value)
    except ValueError:
        try: return float(value)
        except ValueError: return value


def load_config(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            result[key.strip()] = parse_value(value)
    return result


def metrics(rows):
    errors = [abs(float(r["target_months"]) - float(r["prediction_months"])) for r in rows]
    return {
        "count": len(rows),
        "mae_months": sum(errors) / len(errors),
        "rmse_months": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "median_absolute_error_months": statistics.median(errors),
    }


def read_test_rows(test_sex: Path, test_images: Path) -> list[dict]:
    with test_sex.open(encoding="utf-8-sig", newline="") as f:
        sex = {str(int(r["Case ID"])): r["Sex"].strip().upper() for r in csv.DictReader(f)}
    images = sorted(test_images.glob("*.png"), key=lambda p: int(p.stem))
    if len(images) != 200:
        raise RuntimeError(f"Expected 200 test images, found {len(images)}")
    return [{"split": "test", "image_id": p.stem, "bone_age_months": "0", "sex": sex[p.stem], "image_path": str(p), "sha256": "", "readable": "True"} for p in images]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("c3_roi/runs/C3_ROI_V1"))
    ap.add_argument("--roi-root", type=Path, default=Path("c3_roi/cache/C3_ROI_V1_TEST/roi"))
    ap.add_argument("--test-sex", type=Path, default=Path("data/goc/boneage-test-dataset.csv"))
    ap.add_argument("--e1-predictions", type=Path, default=Path("p8_test_ensemble/outputs/P8_ensemble_predictions.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("c3_roi/outputs/C3_ROI_V1_TEST"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("C3 test inference should run on CUDA/Colab; local CPU run is intentionally blocked")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_test_rows(args.test_sex, args.roi_root)
    per_fold = []
    for fold in range(1, 6):
        cfg = load_config(args.run_root / f"C3_ROI_V1_FOLD_{fold}" / "config_resolved.yaml")
        model = build_model(str(cfg["architecture"]), False, int(cfg["sex_embedding_dim"]), int(cfg["head_hidden_dim"]), float(cfg["dropout"]), int(cfg["age_class_count"])).to(device)
        checkpoint = torch.load(args.run_root / f"C3_ROI_V1_FOLD_{fold}" / "best_mae.ckpt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        dataset = BoneAgeDataset(rows, int(cfg["image_size"]), float(cfg["target_mean"]), float(cfg["target_std"]), train=False, preprocessing="none")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        predictions = []
        with torch.inference_mode():
            for batch in loader:
                output = model(batch["image"].to(device, non_blocking=True), batch["sex"].to(device, non_blocking=True))
                normalized = output["regression"] if isinstance(output, dict) else output
                values = normalized.float().cpu() * float(cfg["target_std"]) + float(cfg["target_mean"])
                predictions.extend({"image_id": str(i), "sex": s, "prediction_months": float(p)} for i, s, p in zip(batch["image_id"], batch["sex_text"], values.tolist()))
        with (args.output_dir / f"C3_fold_{fold}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
        per_fold.append(predictions)
        del model, checkpoint
        torch.cuda.empty_cache()
        print(f"fold={fold} predictions={len(predictions)}", flush=True)
    by_id = {r["image_id"]: [] for r in rows}
    for fold in per_fold:
        for r in fold: by_id[r["image_id"]].append(r["prediction_months"])
    c3 = [{"image_id": r["image_id"], "sex": r["sex"], "prediction_months": sum(by_id[r["image_id"]]) / 5} for r in rows]
    e1 = {}
    with args.e1_predictions.open(encoding="utf-8-sig", newline="") as f:
        e1 = {r["image_id"]: r for r in csv.DictReader(f)}
    ensemble = []
    for r in c3:
        b = e1[r["image_id"]]
        p = (r["prediction_months"] + float(b["prediction_months"])) / 2
        ensemble.append({**r, "prediction_months": p, "e1_prediction_months": float(b["prediction_months"]), "ensemble_50_50_prediction_months": p, "target_months": float(b["target_months"]), "absolute_error": abs(p - float(b["target_months"]))})
    with (args.output_dir / "C3_E1_50_50_test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ensemble[0])); writer.writeheader(); writer.writerows(ensemble)
    report = {"protocol": "fixed C3 5-fold + locked E1 5-fold, equal weight 50/50; no test tuning", "test_used_for_model_selection": False, "c3": metrics([{**r, "target_months": e1[r["image_id"]]["target_months"]} for r in c3]), "e1": metrics([{**r, "prediction_months": float(e1[r["image_id"]]["prediction_months"]), "target_months": float(e1[r["image_id"]]["target_months"])} for r in c3]), "ensemble_50_50": metrics(ensemble)}
    (args.output_dir / "C3_E1_50_50_test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
