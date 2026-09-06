from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import tomllib
from torch.utils.data import DataLoader

from .cache_utils import sha256_file
from .data import MultiViewBoneAgeDataset, load_manifest
from .model import build_c4_model
from .config import C4Config, load_config, scientific_config_hash


LOCKED_MANIFEST_RELATIVE = Path("c4_multi_roi/cache/C4_MULTI_ROI_V2/manifest.csv")
LOCKED_MANIFEST_SHA256 = "d56634f084163bd360a3b43847958869fdc5fa3dcf19d1aae1836f441e612b92"
LOCKED_INVENTORY_RELATIVE = Path("data/model_eval_inventory/C4_MULTI_ROI_V2_5FOLD_SELECTED_20260903")
EXPECTED_ROWS = 14024
EXPECTED_FOLD_COUNTS = {1: 2805, 2: 2805, 3: 2804, 4: 2805, 5: 2805}


def load_locked_manifest(workspace: Path) -> list[dict[str, str]]:
    path = workspace.resolve() / LOCKED_MANIFEST_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"Locked C4 V2 manifest not found: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != LOCKED_MANIFEST_SHA256:
        raise RuntimeError(f"Manifest hash mismatch: {actual_hash} != {LOCKED_MANIFEST_SHA256}")
    rows = load_manifest(path)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} manifest rows, got {len(rows)}")
    required = {"image_id", "source_split", "fold", "bone_age_months", "sex", "global_path"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {sorted(missing)}")
    if any(str(row["source_split"]).lower() == "test" for row in rows):
        raise RuntimeError("Test records are forbidden in the OOF manifest")
    ids = [str(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate image_id in the OOF manifest")
    return rows


def expected_oof_rows(rows: list[dict[str, str]]) -> int:
    ids = {str(row["image_id"]) for row in rows}
    if len(ids) != len(rows):
        raise RuntimeError("Duplicate image_id in OOF rows")
    counts = {fold: len(select_validation_rows(rows, fold)) for fold in range(1, 6)}
    if counts != EXPECTED_FOLD_COUNTS:
        raise RuntimeError(f"Unexpected fold counts: {counts}")
    return len(ids)


def select_validation_rows(rows: list[dict[str, str]], fold: int) -> list[dict[str, str]]:
    if fold not in range(1, 6):
        raise ValueError(f"fold must be 1..5, got {fold}")
    selected = [row for row in rows if int(row["fold"]) == fold]
    if not selected:
        raise RuntimeError(f"No rows found for fold {fold}")
    if any(str(row["source_split"]).lower() == "test" for row in selected):
        raise RuntimeError(f"Test record selected for OOF fold {fold}")
    return selected


def resolve_fold_artifacts(inventory: Path, fold: int) -> tuple[Path, Path]:
    checkpoint = inventory / f"FOLD_{fold}" / "best_mae.ckpt"
    config = inventory / "configs" / f"fold_{fold}_colab.toml"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing selected checkpoint for fold {fold}: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Missing selected config for fold {fold}: {config}")
    return checkpoint, config


def _device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(checkpoint_path: Path, config: C4Config, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError(f"Invalid C4 checkpoint: {checkpoint_path}")
    saved_hash = checkpoint.get("config_hash")
    expected_hash = scientific_config_hash(config)
    if saved_hash and saved_hash != expected_hash:
        raise RuntimeError(
            f"Checkpoint/config hash mismatch for {checkpoint_path.name}: {saved_hash} != {expected_hash}"
        )
    model = build_c4_model(
        pretrained=False,
        view_mode=config.view_mode,
        sex_embedding_dim=config.sex_embedding_dim,
        hidden_dim=config.head_hidden_dim,
        dropout=config.dropout,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty prediction file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _infer_fold(
    workspace: Path,
    rows: list[dict[str, str]],
    fold: int,
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    progress_every: int,
    max_samples: int | None,
) -> list[dict]:
    config = load_config(config_path)
    if config.validation_fold != fold or config.view_mode != "global_plus_six":
        raise RuntimeError(f"Fold {fold} config is not global_plus_six for validation fold {fold}")
    selected = select_validation_rows(rows, fold)
    if max_samples is not None:
        selected = selected[:max_samples]
    dataset = MultiViewBoneAgeDataset(
        selected,
        image_root=workspace,
        image_size=config.image_size,
        target_mean=config.target_mean,
        target_std=config.target_std,
        view_mode=config.view_mode,
        train=False,
        seed=config.seed,
        augmentation="none",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = _load_model(checkpoint_path, config, device)
    amp_enabled = device.type == "cuda"
    started = time.perf_counter()
    predictions: list[dict] = []
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            views = batch["views"].to(device, non_blocking=True)
            sex = batch["sex"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predicted_norm = model(views, sex)
            predicted = predicted_norm.float().cpu() * config.target_std + config.target_mean
            for image_id, sex_text, target, value in zip(
                batch["image_id"], batch["sex_text"], batch["target_months"], predicted
            ):
                target_value = float(target)
                prediction_value = float(value)
                predictions.append({
                    "image_id": str(image_id),
                    "fold": fold,
                    "sex": str(sex_text),
                    "target_months": target_value,
                    "prediction_months": prediction_value,
                    "absolute_error": abs(prediction_value - target_value),
                })
                seen += 1
            if seen == len(selected) or seen % progress_every < len(batch["image_id"]):
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = seen / elapsed
                eta = (len(selected) - seen) / rate if rate else float("inf")
                print(f"OOF fold {fold}: {seen}/{len(selected)} samples, {rate:.2f} sample/s, ETA {eta / 60:.1f} min", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _write_rows(output_dir / f"fold_{fold}_predictions.csv", predictions)
    return predictions


def aggregate_predictions(prediction_rows: list[dict], output_dir: Path, *, complete: bool) -> dict:
    by_id: dict[str, dict] = {}
    for row in prediction_rows:
        image_id = str(row["image_id"])
        if image_id in by_id:
            raise RuntimeError(f"Duplicate OOF prediction ID: {image_id}")
        by_id[image_id] = row
    ordered = [by_id[key] for key in sorted(by_id, key=lambda value: int(float(value)))]
    if complete and len(ordered) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} OOF predictions, got {len(ordered)}")
    errors = np.asarray([float(row["prediction_months"]) - float(row["target_months"]) for row in ordered], dtype=np.float64)
    report = {
        "status": "PASS" if complete else "PARTIAL",
        "protocol": "C4 V2 global plus six ROI, locked five-fold OOF",
        "test_used": False,
        "count": len(ordered),
        "mae": float(np.abs(errors).mean()) if len(errors) else None,
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
    }
    if ordered:
        _write_rows(output_dir / "C4_MULTI_ROI_V2_OOF_predictions.csv", ordered)
    (output_dir / "C4_MULTI_ROI_V2_OOF_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def run_oof(
    workspace: Path,
    output_dir: Path,
    *,
    inventory: Path | None = None,
    device: str = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    progress_every: int = 25,
    folds: tuple[int, ...] = (1, 2, 3, 4, 5),
    max_samples: int | None = None,
) -> dict:
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    inventory = (workspace / LOCKED_INVENTORY_RELATIVE if inventory is None else inventory).resolve()
    if batch_size <= 0 or num_workers < 0 or progress_every <= 0:
        raise ValueError("batch_size and progress_every must be positive; num_workers cannot be negative")
    rows = load_locked_manifest(workspace)
    expected_oof_rows(rows)
    selected_device = _device(device)
    prediction_rows: list[dict] = []
    for fold in folds:
        checkpoint, config_path = resolve_fold_artifacts(inventory, fold)
        prediction_rows.extend(_infer_fold(
            workspace, rows, fold, checkpoint, config_path, output_dir,
            selected_device, batch_size, num_workers, progress_every, max_samples,
        ))
    complete = max_samples is None and set(folds) == set(range(1, 6))
    report = aggregate_predictions(prediction_rows, output_dir, complete=complete)
    report.update({
        "manifest_sha256": LOCKED_MANIFEST_SHA256,
        "device": str(selected_device),
        "folds_run": list(folds),
    })
    (output_dir / "C4_MULTI_ROI_V2_OOF_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run leakage-safe C4 V2 five-fold OOF inference")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("c4_multi_roi/outputs/C4_MULTI_ROI_V2_OOF_LOCAL"))
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--fold", type=int, action="append", choices=range(1, 6))
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    folds = tuple(args.fold) if args.fold else (1, 2, 3, 4, 5)
    report = run_oof(
        args.workspace, args.output_dir, inventory=args.inventory, device=args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        progress_every=args.progress_every, folds=folds, max_samples=args.max_samples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
