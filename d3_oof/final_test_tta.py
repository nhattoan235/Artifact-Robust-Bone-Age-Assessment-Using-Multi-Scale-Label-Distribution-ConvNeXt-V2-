from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from p1_baseline.model import build_model

from d3_oof.tta_oof import ROTATIONS, TTADataset, resolved_config


TEST_CSV = ROOT / "data/goc/rsna_test_ground_truth.csv"
TEST_ROOT = ROOT / "data/goc/boneage-test-dataset/boneage-test-dataset"
OUTPUT = ROOT / "d3_oof/outputs/FINAL_TEST_TTA_V1"


def load_rows() -> list[dict]:
    with TEST_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for item in source:
        image_id = Path(item["image_ID"]).stem
        image_path = TEST_ROOT / item["image_ID"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append({
            "image_id": image_id,
            "sex": item["sex"],
            "target_months": float(item["bone_age"]),
            "image_path": str(image_path),
        })
    return rows


def load_model(branch: str, fold: int, device: torch.device):
    if branch == "E1":
        run_dir = ROOT / f"p7_results/results/P7_FINAL_V3_FOLD_{fold}"
        config = resolved_config(run_dir / "config_resolved.yaml")
        checkpoint_path = run_dir / "best_model.pt"
    else:
        run_dir = ROOT / f"d3_oof/runs/D3_OOF_V1/D3_OOF_V1_FOLD_{fold}"
        config = resolved_config(run_dir / "config_resolved.yaml")
        checkpoint_path = run_dir / "best_mae.ckpt"
    model = build_model(
        str(config["architecture"]), False, int(config["sex_embedding_dim"]),
        int(config["head_hidden_dim"]), float(config["dropout"]),
        int(config["age_class_count"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


def predict_branch(model, rows, config, branch: str, rotation: float, flip: bool, device: torch.device) -> dict[str, float]:
    loader = DataLoader(
        TTADataset(rows, int(config["image_size"]), rotation, flip),
        batch_size=16, shuffle=False, num_workers=2, pin_memory=device.type == "cuda",
    )
    values_by_id = {}
    age_axis = torch.arange(int(config["age_class_count"]), device=device, dtype=torch.float32)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            sex = batch["sex"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(images, sex)
            if branch == "E1":
                prediction = output.float() * float(config["target_std"]) + float(config["target_mean"])
            else:
                regression = output["regression"].float() * float(config["target_std"]) + float(config["target_mean"])
                distribution = torch.softmax(output["distribution_logits"].float(), dim=1) @ age_axis
                weight = float(config["regression_inference_weight"])
                prediction = weight * regression + (1.0 - weight) * distribution
            for image_id, value in zip(batch["image_id"], prediction.detach().cpu().tolist()):
                values_by_id[str(image_id)] = float(value)
    return values_by_id


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    branch_predictions = {}
    for branch in ("E1", "D3"):
        fold_predictions = []
        for fold in range(1, 6):
            print(f"[{branch}] fold {fold}/5", flush=True)
            model, config = load_model(branch, fold, device)
            transforms = []
            for rotation in ROTATIONS:
                for flip in (False, True):
                    transforms.append(predict_branch(model, rows, config, branch, rotation, flip, device))
            fold_pred = np.array([
                np.mean([result[row["image_id"]] for result in transforms]) for row in rows
            ], dtype=np.float64)
            fold_predictions.append(fold_pred)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        branch_predictions[branch] = np.mean(np.stack(fold_predictions), axis=0)

    target = np.array([row["target_months"] for row in rows], dtype=np.float64)
    e1 = branch_predictions["E1"]
    d3 = branch_predictions["D3"]
    ensemble = 0.5 * e1 + 0.5 * d3

    def group_metrics(mask: np.ndarray) -> dict:
        return {
            "count": int(mask.sum()),
            "E1_TTA_MAE": float(np.abs(e1[mask] - target[mask]).mean()),
            "D3_TTA_MAE": float(np.abs(d3[mask] - target[mask]).mean()),
            "ensemble_50_50_MAE": float(np.abs(ensemble[mask] - target[mask]).mean()),
        }

    sex = np.array([row["sex"] for row in rows])
    report = {
        "status": "PASS",
        "test_used": True,
        "count": len(rows),
        "protocol": "5-fold model average; each fold uses rotations -10,-5,0,5,10 with flip/no-flip; fixed 50/50 E1-TTA + D3-TTA",
        "mae": group_metrics(np.ones(len(rows), dtype=bool)),
        "by_sex": {s: group_metrics(sex == s) for s in ("F", "M")},
        "by_age_bin": {
            name: group_metrics((target >= low) & (target < high))
            for low, high, name in ((0, 60, "0-59"), (60, 120, "60-119"), (120, 180, "120-179"), (180, 229, "180-228"))
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "FINAL_TEST_TTA_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "sex", "target_months", "e1_tta", "d3_tta", "ensemble_50_50", "absolute_error"])
        writer.writeheader()
        for row, p1, p3, pe in zip(rows, e1, d3, ensemble):
            writer.writerow({"image_id": row["image_id"], "sex": row["sex"], "target_months": row["target_months"], "e1_tta": p1, "d3_tta": p3, "ensemble_50_50": pe, "absolute_error": abs(pe - row["target_months"])})
    (OUTPUT / "FINAL_TEST_TTA_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
