"""Evaluate clean and fixed synthetic-artifact Fold-1 views for pilot B/C."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1_baseline.config import load_config  # noqa: E402
from p1_baseline.data import BoneAgeDataset, load_manifest  # noqa: E402
from p1_baseline.metrics import compute_metrics  # noqa: E402
from p1_baseline.model import build_model  # noqa: E402


DEFAULT_PILOT_ROOT = Path("/content/C3_Z26_C3_ROI_V2_PILOTS")
BASELINE_FOLD1_MAE = 6.253462484419516


def paired_bootstrap_ci(
    target: np.ndarray, baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray, *, repetitions: int = 10000,
    seed: int = 20260912,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=np.float64)
    baseline_prediction = np.asarray(baseline_prediction, dtype=np.float64)
    candidate_prediction = np.asarray(candidate_prediction, dtype=np.float64)
    if target.ndim != 1 or not (
        len(target) == len(baseline_prediction) == len(candidate_prediction)
    ):
        raise ValueError("Ba vector paired phải một chiều và cùng độ dài")
    if len(target) == 0 or repetitions <= 0:
        raise ValueError("Cần dữ liệu và repetitions > 0")
    paired_delta = (
        np.abs(candidate_prediction - target) - np.abs(baseline_prediction - target)
    )
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    # Chunking keeps memory bounded on Colab for 2,808 x 10,000 resamples.
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        indices = rng.integers(0, len(target), size=(size, len(target)))
        values[start:start + size] = paired_delta[indices].mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _basic_group(records: list[dict[str, Any]]) -> dict[str, float | int]:
    clean = [abs(float(row["clean_prediction_months"]) - float(row["target_months"])) for row in records]
    artifact = [abs(float(row["artifact_prediction_months"]) - float(row["target_months"])) for row in records]
    disagreement = [abs(float(row["artifact_prediction_months"]) - float(row["clean_prediction_months"])) for row in records]
    return {
        "count": len(records),
        "clean_mae": statistics.fmean(clean),
        "artifact_mae": statistics.fmean(artifact),
        "artifact_minus_clean_mae": statistics.fmean(artifact) - statistics.fmean(clean),
        "mean_disagreement_months": statistics.fmean(disagreement),
    }


def summarize_predictions(
    records: list[dict[str, Any]], age_bins: list[int], *, fold: int = 1,
    repetitions: int = 10000,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Không có prediction để đánh giá")
    clean_records = [
        {"image_id": row["image_id"], "sex": row["sex"],
         "target_months": float(row["target_months"]),
         "prediction_months": float(row["clean_prediction_months"])}
        for row in records
    ]
    artifact_records = [
        {"image_id": row["image_id"], "sex": row["sex"],
         "target_months": float(row["target_months"]),
         "prediction_months": float(row["artifact_prediction_months"])}
        for row in records
    ]
    target = np.asarray([row["target_months"] for row in records], dtype=np.float64)
    clean_prediction = np.asarray(
        [row["clean_prediction_months"] for row in records], dtype=np.float64
    )
    artifact_prediction = np.asarray(
        [row["artifact_prediction_months"] for row in records], dtype=np.float64
    )
    disagreement = np.abs(artifact_prediction - clean_prediction)
    sex_groups = {
        sex: _basic_group([row for row in records if row["sex"] == sex])
        for sex in sorted({str(row["sex"]) for row in records})
    }
    age_groups: dict[str, Any] = {}
    for left, right in zip(age_bins[:-1], age_bins[1:]):
        group = [row for row in records if left <= float(row["target_months"]) < right]
        if group:
            age_groups[f"{left}-{right - 1}"] = _basic_group(group)
    return {
        "protocol": (
            f"Fold {fold} validation only; fixed clean and synthetic "
            "mild-artifact paired views"
        ),
        "fold": fold,
        "test_accessed": False,
        "count": len(records),
        "clean": compute_metrics(clean_records, age_bins),
        "artifact": compute_metrics(artifact_records, age_bins),
        "artifact_minus_clean_mae_months": float(
            np.abs(artifact_prediction - target).mean()
            - np.abs(clean_prediction - target).mean()
        ),
        "artifact_minus_clean_paired_bootstrap_95_ci_months": list(
            paired_bootstrap_ci(
                target, clean_prediction, artifact_prediction,
                repetitions=repetitions,
            )
        ),
        "disagreement_months": {
            "mean": float(disagreement.mean()),
            "median": float(np.median(disagreement)),
            "p95": float(np.quantile(disagreement, 0.95)),
        },
        "subgroups_by_sex": sex_groups,
        "subgroups_by_age_bin": age_groups,
        "limitations": [
            "Artifact view is synthetic mild_v1, not an external hospital dataset.",
            "Pilot selection is restricted to Fold 1; confirmation requires all folds.",
        ],
    }


def _config_path(pilot: str, pilot_root: Path, fold: int = 1) -> Path:
    name = pilot.upper()
    if name not in {"B", "C"}:
        raise ValueError("pilot phải là B hoặc C")
    if fold not in {1, 2, 3, 4, 5}:
        raise ValueError("fold phải nằm trong 1..5")
    if name == "B" and fold != 1:
        raise ValueError("Pilot B chỉ được khóa ở Fold 1")
    return pilot_root / "configs" / f"C3_Z26_C3_ROI_V2_PILOT_{name}_FOLD_{fold}_SEED_42.toml"


def _checkpoint_path(cfg) -> Path:
    candidates = [cfg.run_dir / "best_mae.ckpt"]
    if cfg.checkpoint_mirror_root:
        candidates.insert(0, Path(cfg.checkpoint_mirror_root) / cfg.run_id / "best_mae.ckpt")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Không tìm thấy best_mae.ckpt ở local hoặc Drive mirror")


@torch.inference_mode()
def _predict(cfg, checkpoint: Path) -> list[dict[str, Any]]:
    rows = load_manifest(cfg.val_manifest, "validation_official")
    dataset = BoneAgeDataset(
        rows, cfg.image_size, cfg.target_mean, cfg.target_std,
        train=True, epoch=0, seed=20260912, augmentation="none",
        preprocessing=cfg.preprocessing, preprocessed_root=cfg.preprocessed_root,
        image_root=cfg.image_root, image_normalization=cfg.image_normalization,
        artifact_augmentation="mild_v1", artifact_probability=1.0,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        cfg.architecture, False, cfg.sex_embedding_dim, cfg.head_hidden_dim,
        cfg.dropout, cfg.age_class_count, sex_mode=cfg.sex_mode,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        clean = batch["image"].to(device, non_blocking=True)
        artifact = batch["artifact_image"].to(device, non_blocking=True)
        sex = batch["sex"].to(device, non_blocking=True)
        output = model(torch.cat([clean, artifact]), torch.cat([sex, sex]))
        if isinstance(output, dict):
            output = output["regression"]
        clean_norm, artifact_norm = output.chunk(2)
        clean_months = (clean_norm * cfg.target_std + cfg.target_mean).cpu().tolist()
        artifact_months = (artifact_norm * cfg.target_std + cfg.target_mean).cpu().tolist()
        for image_id, sex_text, target, clean_value, artifact_value in zip(
            batch["image_id"], batch["sex_text"], batch["target_months"].tolist(),
            clean_months, artifact_months,
        ):
            records.append({
                "image_id": image_id, "sex": sex_text,
                "target_months": float(target),
                "clean_prediction_months": float(clean_value),
                "artifact_prediction_months": float(artifact_value),
                "clean_absolute_error": abs(float(clean_value) - float(target)),
                "artifact_absolute_error": abs(float(artifact_value) - float(target)),
                "disagreement_months": abs(float(artifact_value) - float(clean_value)),
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pilot B/C robustness without test data")
    parser.add_argument("--pilot", required=True, choices=("B", "C", "b", "c"))
    parser.add_argument("--fold", type=int, default=1, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    config_path = _config_path(args.pilot, args.pilot_root, args.fold)
    cfg = load_config(config_path)
    checkpoint = _checkpoint_path(cfg)
    records = _predict(cfg, checkpoint)
    report = summarize_predictions(
        records, cfg.age_bins, fold=args.fold, repetitions=args.bootstrap_repetitions
    )
    report.update({"run_id": cfg.run_id, "checkpoint": str(checkpoint)})
    output_dir = Path(cfg.checkpoint_mirror_root) / cfg.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "artifact_robustness_predictions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report_path = output_dir / "artifact_robustness_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SAVED: {csv_path}")
    print(f"SAVED: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
