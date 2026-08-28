"""Locked 200-case test inference for C3-ROI + Spatial Attention."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
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
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            result[key.strip()] = parse_value(value)
    return result


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average_fold_predictions(folds: list[list[dict]]) -> list[dict]:
    if not folds:
        raise ValueError("At least one fold is required")
    indexed = [{str(row["image_id"]): row for row in fold} for fold in folds]
    expected_ids = set(indexed[0])
    if any(set(fold) != expected_ids for fold in indexed[1:]):
        raise ValueError("Fold prediction ID sets differ")
    result = []
    for image_id in sorted(expected_ids, key=lambda value: int(value)):
        rows = [fold[image_id] for fold in indexed]
        sexes = {str(row["sex"]) for row in rows}
        if len(sexes) != 1:
            raise ValueError(f"Sex differs across folds for image_id={image_id}")
        result.append({
            "image_id": image_id,
            "sex": rows[0]["sex"],
            "prediction_months": statistics.fmean(float(row["prediction_months"]) for row in rows),
        })
    return result


def metrics(rows: list[dict], prediction_key: str = "prediction_months") -> dict:
    errors = [abs(float(row["target_months"]) - float(row[prediction_key])) for row in rows]
    return {
        "count": len(rows),
        "mae_months": statistics.fmean(errors),
        "rmse_months": math.sqrt(statistics.fmean(error * error for error in errors)),
        "median_absolute_error_months": statistics.median(errors),
    }


def paired_delta_bootstrap(rows: list[dict], n_bootstrap: int = 5000, seed: int = 42) -> dict:
    deltas = [
        abs(float(row["attention_prediction_months"]) - float(row["target_months"]))
        - abs(float(row["baseline_prediction_months"]) - float(row["target_months"]))
        for row in rows
    ]
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(n_bootstrap)
    )
    return {
        "estimate_months": statistics.fmean(deltas),
        "lower_95": samples[int(0.025 * n_bootstrap)],
        "upper_95": samples[min(n_bootstrap - 1, int(0.975 * n_bootstrap))],
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def read_test_rows(test_sex: Path, roi_root: Path) -> list[dict]:
    with test_sex.open(encoding="utf-8-sig", newline="") as handle:
        sex = {str(int(row["Case ID"])): row["Sex"].strip().upper() for row in csv.DictReader(handle)}
    images = sorted(roi_root.glob("*.png"), key=lambda path: int(path.stem))
    if len(images) != 200:
        raise RuntimeError(f"Expected 200 test images, found {len(images)}")
    if {path.stem for path in images} != set(sex):
        raise RuntimeError("Test ROI image IDs do not match the locked sex manifest")
    return [
        {
            "split": "test",
            "image_id": path.stem,
            "bone_age_months": "0",
            "sex": sex[path.stem],
            "image_path": str(path),
            "sha256": "",
            "readable": "True",
        }
        for path in images
    ]


def infer_fold(run_dir: Path, rows: list[dict], device: torch.device, batch_size: int, num_workers: int) -> list[dict]:
    cfg = load_resolved_config(run_dir / "config_resolved.yaml")
    model = build_model(
        str(cfg["architecture"]), False, int(cfg["sex_embedding_dim"]),
        int(cfg["head_hidden_dim"]), float(cfg["dropout"]), int(cfg["age_class_count"]),
    ).to(device)
    checkpoint = torch.load(run_dir / "best_mae.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = BoneAgeDataset(
        rows, int(cfg["image_size"]), float(cfg["target_mean"]), float(cfg["target_std"]),
        train=False, preprocessing="none",
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["image"].to(device, non_blocking=True),
                batch["sex"].to(device, non_blocking=True),
            )
            normalized = output["regression"] if isinstance(output, dict) else output
            values = normalized.float().cpu() * float(cfg["target_std"]) + float(cfg["target_mean"])
            predictions.extend(
                {"image_id": str(image_id), "sex": sex, "prediction_months": float(prediction)}
                for image_id, sex, prediction in zip(batch["image_id"], batch["sex_text"], values.tolist())
            )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("c3_roi_attention/runs/C3_ROI_ATTN_V1"))
    parser.add_argument("--roi-root", type=Path, default=Path("c3_roi/cache/C3_ROI_V1_TEST/roi"))
    parser.add_argument("--test-sex", type=Path, default=Path("data/goc/boneage-test-dataset.csv"))
    parser.add_argument("--reference", type=Path, default=Path("p8_test_ensemble/outputs/P8_ensemble_predictions.csv"))
    parser.add_argument("--c3-root", type=Path, default=Path("c3_roi/outputs/C3_ROI_V1_TEST"))
    parser.add_argument("--output-dir", type=Path, default=Path("c3_roi_attention/outputs/C3_ROI_ATTN_V1_TEST"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("Locked test inference requires CUDA")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = read_test_rows(args.test_sex, args.roi_root)
    per_fold = []
    for fold in range(1, 6):
        predictions = infer_fold(
            args.run_root / f"C3_ROI_ATTN_V1_FOLD_{fold}", test_rows,
            device, args.batch_size, args.num_workers,
        )
        write_csv(args.output_dir / f"ATTN_fold_{fold}_predictions.csv", predictions)
        per_fold.append(predictions)
        torch.cuda.empty_cache()
        print(f"fold={fold} predictions={len(predictions)}", flush=True)

    attention = average_fold_predictions(per_fold)
    c3_folds = [read_csv(args.c3_root / f"C3_fold_{fold}_predictions.csv") for fold in range(1, 6)]
    c3 = {row["image_id"]: row for row in average_fold_predictions(c3_folds)}
    reference = {row["image_id"]: row for row in read_csv(args.reference)}
    if set(reference) != {row["image_id"] for row in attention} or set(reference) != set(c3):
        raise RuntimeError("Attention, C3 and locked test reference ID sets differ")

    paired = []
    for row in attention:
        image_id = row["image_id"]
        target = float(reference[image_id]["target_months"])
        paired.append({
            "image_id": image_id,
            "sex": row["sex"],
            "target_months": target,
            "attention_prediction_months": row["prediction_months"],
            "baseline_prediction_months": float(c3[image_id]["prediction_months"]),
            "e1_prediction_months": float(reference[image_id]["prediction_months"]),
            "attention_absolute_error": abs(row["prediction_months"] - target),
            "baseline_absolute_error": abs(float(c3[image_id]["prediction_months"]) - target),
        })
    write_csv(args.output_dir / "C3_ROI_ATTN_V1_TEST_predictions.csv", paired)

    as_attention = [{**row, "prediction_months": row["attention_prediction_months"]} for row in paired]
    as_c3 = [{**row, "prediction_months": row["baseline_prediction_months"]} for row in paired]
    as_e1 = [{**row, "prediction_months": row["e1_prediction_months"]} for row in paired]
    by_sex = {}
    for sex in sorted({row["sex"] for row in paired}):
        group = [row for row in paired if row["sex"] == sex]
        by_sex[sex] = {
            "attention": metrics([{**row, "prediction_months": row["attention_prediction_months"]} for row in group]),
            "c3_roi": metrics([{**row, "prediction_months": row["baseline_prediction_months"]} for row in group]),
        }
    report = {
        "protocol": "locked exploratory re-evaluation on the original 200-case test; no test tuning; five-fold mean; no TTA",
        "test_used_for_model_selection": False,
        "audit": {"count": len(paired), "unique_ids": len({row["image_id"] for row in paired}), "id_sets_equal": True},
        "attention": metrics(as_attention),
        "c3_roi": metrics(as_c3),
        "e1_reference": metrics(as_e1),
        "paired_bootstrap_attention_minus_c3": paired_delta_bootstrap(paired),
        "by_sex": by_sex,
    }
    report_path = args.output_dir / "C3_ROI_ATTN_V1_TEST_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
