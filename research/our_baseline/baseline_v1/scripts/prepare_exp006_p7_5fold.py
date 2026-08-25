"""Prepare the EXP-006 five-fold P7 control and follow-up LDL configs.

This preparation step only uses the 14,036 development rows from the friend's
locked manifest. It rewrites image paths to the local data root, creates one
train/validation manifest per fold, and emits reproducible TOML configs for:

* P7 direct regression control;
* D3 LDL regression-only candidate;
* D3 LDL fused candidate.

No test CSV or test label is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_hash(path: Path) -> str:
    """Match p1_baseline.data.manifest_hash byte-for-byte."""
    digest = hashlib.sha256()
    keys = ["split", "image_id", "bone_age_months", "sex", "sha256"]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # The friend's locked implementation intentionally sorts lexical numeric
    # strings (e.g. "10000" before "1377"); preserve that exact behavior.
    for row in sorted(rows, key=lambda item: str(int(float(item["image_id"])) )):
        digest.update(("\t".join(row.get(key, "") for key in keys) + "\n").encode("utf-8"))
    return digest.hexdigest()


def make_strata(frame: pd.DataFrame) -> np.ndarray:
    age = pd.to_numeric(frame["bone_age_months"], errors="raise")
    bins = pd.cut(age, bins=[0, 60, 120, 180, 229], labels=False, include_lowest=True)
    sex = (frame["sex"].astype(str).str.upper() == "M").astype(int)
    return (sex.to_numpy() * 10 + bins.fillna(0).astype(int).to_numpy())


def local_image_map(data_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in data_root.rglob("*.png"):
        image_id = normalize_id(path.stem)
        if image_id in result:
            raise ValueError(f"Duplicate image ID {image_id}: {result[image_id]} and {path}")
        result[image_id] = path.resolve()
    return result


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def make_config(
    *,
    config_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    output_root: Path,
    run_id: str,
    train_hash: str,
    val_hash: str,
    train_count: int,
    val_count: int,
    target_mean: float,
    target_std: float,
    architecture: str,
    regression_inference_weight: float,
    checkpoint_mirror_root: Path | None,
) -> None:
    mirror_line = (
        f"checkpoint_mirror_root = {toml_string(str(checkpoint_mirror_root.resolve()))}"
        if checkpoint_mirror_root else ""
    )
    lines = [
        "[data]",
        f"train_manifest = {toml_string(str(train_manifest.resolve()))}",
        f"val_manifest = {toml_string(str(val_manifest.resolve()))}",
        f"expected_train_hash = {toml_string(train_hash)}",
        f"expected_val_hash = {toml_string(val_hash)}",
        f"expected_train_count = {train_count}",
        f"expected_val_count = {val_count}",
        "",
        "[run]",
        f"run_id = {toml_string(run_id)}",
        f"output_root = {toml_string(str(output_root.resolve()))}",
        'device = "auto"',
        "seed = 42",
        "deterministic = true",
        "allow_tf32 = true",
        mirror_line,
        "",
        "[model]",
        f"architecture = {toml_string(architecture)}",
        'sex_mode = "embedding"',
        "pretrained = true",
        "image_size = 512",
        "sex_embedding_dim = 16",
        "head_hidden_dim = 256",
        "dropout = 0.2",
        f"target_mean = {target_mean:.12f}",
        f"target_std = {target_std:.12f}",
        "age_class_count = 229",
        "label_distribution_sigma = 2.0",
        "label_distribution_weight = 0.2",
        f"regression_inference_weight = {regression_inference_weight}",
        'augmentation = "light"',
        "horizontal_flip_probability = 0.5",
        "rotation_degrees = 7.0",
        "translation_fraction = 0.03",
        "scale_min = 0.95",
        "scale_max = 1.05",
        "brightness_delta = 0.10",
        "contrast_delta = 0.10",
        "gamma_min = 0.90",
        "gamma_max = 1.10",
        'preprocessing = "none"',
        'image_normalization = "imagenet"',
        "",
        "[training]",
        "epochs = 35",
        "batch_size = 12",
        "grad_accum_steps = 3",
        "learning_rate = 0.0002",
        "min_learning_rate = 0.000001",
        "weight_decay = 0.05",
        "smooth_l1_beta_months = 3.0",
        'regression_loss = "smooth_l1"',
        "gradient_clip_norm = 5.0",
        "patience = 8",
        "min_delta_mae = 0.01",
        "amp = true",
        'amp_dtype = "float16"',
        "amp_init_scale = 4096.0",
        "",
        "[operations]",
        "num_workers = 2",
        "log_every_steps = 25",
        "checkpoint_every_steps = 500",
        "checkpoint_every_minutes = 20.0",
        # Drive quota is limited: keep only best + last checkpoints. The
        # trainer still writes a temporary periodic checkpoint locally, but
        # removes periodic mirrors instead of accumulating duplicates.
        "periodic_keep = 0",
        "best_keep = 1",
        "",
        "[warnings]",
        "warning_gradient_norm = 50.0",
        "warning_gradient_consecutive = 3",
        "warning_prediction_soft_min = 0.0",
        "warning_prediction_soft_max = 228.0",
        "warning_prediction_hard_min = -60.0",
        "warning_prediction_hard_max = 300.0",
        "warning_prediction_std_min = 1.0",
        "warning_sex_mae_gap = 2.5",
        "warning_overfit_epochs = 3",
        "age_bins = [0, 60, 120, 180, 229]",
    ]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(line for line in lines if line != "") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-output-root", type=Path, default=None)
    parser.add_argument("--checkpoint-mirror-root", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least 2")

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from sklearn.model_selection import StratifiedKFold

    source_path = repo_root / "p0_audit" / "outputs" / "development_manifest_14036.csv"
    source = pd.read_csv(source_path)
    required = {"split", "image_id", "bone_age_months", "sex", "image_path", "sha256", "readable"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Source manifest missing columns: {sorted(missing)}")
    source["image_id"] = source["image_id"].map(normalize_id)
    if len(source) != 14036 or source["image_id"].duplicated().any():
        raise RuntimeError("Development manifest must contain 14,036 unique rows")

    image_map = local_image_map(args.data_root.resolve())
    missing_images = sorted(set(source["image_id"]) - set(image_map), key=lambda x: int(x))
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} local images: {missing_images[:10]}")
    source["image_path"] = source["image_id"].map(lambda value: str(image_map[value]))
    source["readable"] = "True"
    source["error"] = ""
    source = source[["split", "image_id", "bone_age_months", "sex", "image_path", "sha256", "readable", "error"]]
    source = source.sort_values("image_id", key=lambda s: s.map(lambda x: int(x))).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output_dir / "development_manifest_local.csv", index=False)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    strata = make_strata(source)
    run_output_root = (args.run_output_root or (args.output_dir / "runs")).resolve()
    fold_records = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(source)), strata), start=1):
        fold_dir = args.output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train = source.iloc[train_idx].copy()
        val = source.iloc[val_idx].copy()
        train["split"] = "train"
        val["split"] = "validation_official"
        train = train.sort_values("image_id", key=lambda s: s.map(lambda x: int(x))).reset_index(drop=True)
        val = val.sort_values("image_id", key=lambda s: s.map(lambda x: int(x))).reset_index(drop=True)
        train_path = fold_dir / "train_manifest.csv"
        val_path = fold_dir / "validation_manifest.csv"
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        target_mean = float(pd.to_numeric(train["bone_age_months"]).mean())
        target_std = float(pd.to_numeric(train["bone_age_months"]).std(ddof=1))
        train_hash = manifest_hash(train_path)
        val_hash = manifest_hash(val_path)
        run_p7 = f"EXP006_P7_CONTROL_FOLD_{fold_index}"
        run_d3_reg = f"EXP007_D3_LDL_REGONLY_FOLD_{fold_index}"
        run_d3_fused = f"EXP008_D3_LDL_FUSED_FOLD_{fold_index}"
        make_config(
            config_path=fold_dir / "p7_control.toml", train_manifest=train_path, val_manifest=val_path,
            output_root=run_output_root, run_id=run_p7, train_hash=train_hash, val_hash=val_hash,
            train_count=len(train), val_count=len(val), target_mean=target_mean, target_std=target_std,
            architecture="convnext_tiny", regression_inference_weight=0.5,
            checkpoint_mirror_root=args.checkpoint_mirror_root,
        )
        make_config(
            config_path=fold_dir / "d3_ldl_regonly.toml", train_manifest=train_path, val_manifest=val_path,
            output_root=run_output_root, run_id=run_d3_reg, train_hash=train_hash, val_hash=val_hash,
            train_count=len(train), val_count=len(val), target_mean=target_mean, target_std=target_std,
            architecture="convnext_tiny_ldl", regression_inference_weight=1.0,
            checkpoint_mirror_root=args.checkpoint_mirror_root,
        )
        make_config(
            config_path=fold_dir / "d3_ldl_fused.toml", train_manifest=train_path, val_manifest=val_path,
            output_root=run_output_root, run_id=run_d3_fused, train_hash=train_hash, val_hash=val_hash,
            train_count=len(train), val_count=len(val), target_mean=target_mean, target_std=target_std,
            architecture="convnext_tiny_ldl", regression_inference_weight=0.5,
            checkpoint_mirror_root=args.checkpoint_mirror_root,
        )
        fold_records.append({
            "fold": fold_index,
            "train_count": len(train),
            "val_count": len(val),
            "train_manifest": str(train_path.resolve()),
            "val_manifest": str(val_path.resolve()),
            "train_hash": train_hash,
            "val_hash": val_hash,
            "target_mean": target_mean,
            "target_std": target_std,
            "p7_config": str((fold_dir / "p7_control.toml").resolve()),
            "d3_regonly_config": str((fold_dir / "d3_ldl_regonly.toml").resolve()),
            "d3_fused_config": str((fold_dir / "d3_ldl_fused.toml").resolve()),
        })

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "EXP-006/007/008 five-fold roadmap preparation",
        "test_labels_used": False,
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": sha256_file(source_path),
        "local_data_root": str(args.data_root.resolve()),
        "development_count": len(source),
        "folds": args.folds,
        "seed": args.seed,
        "stratification": "sex + age bins [0,60,120,180,229]",
        "run_output_root": str(run_output_root),
        "folds_detail": fold_records,
        "next_step": "Run P7 control folds; merge val_predictions_best.csv; only then run TTA and LDL candidates.",
    }
    (args.output_dir / "roadmap_preparation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
