"""Prepare a fresh train/holdout split for the friend's P7 trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def split_score(image_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def normalize_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-output-root", type=Path, default=None)
    parser.add_argument("--checkpoint-mirror-root", type=Path, default=None)
    args = parser.parse_args()
    if not 0.01 <= args.holdout_fraction <= 0.40:
        raise ValueError("holdout-fraction must be between 0.01 and 0.40")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.repo_root) not in sys.path:
        sys.path.insert(0, str(args.repo_root))
    from p1_baseline.data import manifest_hash

    source = pd.read_csv(args.repo_root / "p0_audit/outputs/development_manifest_14036.csv")
    required = {"split", "image_id", "bone_age_months", "sex", "image_path", "sha256", "readable"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Source manifest missing columns: {sorted(missing)}")
    source["image_id"] = source["image_id"].map(normalize_id)
    source["score"] = source["image_id"].map(lambda value: split_score(value, args.seed))

    image_map: dict[str, Path] = {}
    for path in args.data_root.rglob("*.png"):
        if path.stem in image_map:
            raise ValueError(f"Duplicate local image ID {path.stem}: {image_map[path.stem]} and {path}")
        image_map[path.stem] = path.resolve()
    missing_images = sorted(set(source.image_id) - set(image_map))
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} local images: {missing_images[:10]}")

    original_train = source[source.split == "train"].copy()
    original_val = source[source.split == "validation_official"].copy()
    holdout = original_train[original_train.score < args.holdout_fraction].copy()
    new_train = pd.concat(
        [original_train[original_train.score >= args.holdout_fraction], original_val],
        ignore_index=True,
    )
    holdout["split"] = "validation_official"
    new_train["split"] = "train"
    for frame in (new_train, holdout):
        frame["image_path"] = frame.image_id.map(lambda value: str(image_map[value]))
        frame["readable"] = "True"
        frame["error"] = ""
    columns = ["split", "image_id", "bone_age_months", "sex", "image_path", "sha256", "readable", "error"]
    new_train = new_train.sort_values("image_id").reset_index(drop=True)[columns]
    holdout = holdout.sort_values("image_id").reset_index(drop=True)[columns]
    if set(new_train.image_id).intersection(holdout.image_id):
        raise RuntimeError("Train/holdout ID leakage detected")

    target_mean = float(new_train.bone_age_months.astype(float).mean())
    target_std = float(new_train.bone_age_months.astype(float).std(ddof=1))
    new_train.to_csv(args.output_dir / "train_manifest.csv", index=False)
    holdout.to_csv(args.output_dir / "holdout_manifest.csv", index=False)
    train_hash = manifest_hash(new_train.astype(str).to_dict("records"))
    val_hash = manifest_hash(holdout.astype(str).to_dict("records"))

    run_id = "EXP004_FRIEND_P7_FRESH_HOLDOUT_SEED42"
    output_root = (args.run_output_root or (args.output_dir / "runs")).resolve()
    config_path = args.output_dir / "friend_p7_fresh_holdout.toml"
    config_lines = [
        "[data]",
        f'train_manifest = "{(args.output_dir / "train_manifest.csv").resolve().as_posix()}"',
        f'val_manifest = "{(args.output_dir / "holdout_manifest.csv").resolve().as_posix()}"',
        f'expected_train_hash = "{train_hash}"',
        f'expected_val_hash = "{val_hash}"',
        f"expected_train_count = {len(new_train)}",
        f"expected_val_count = {len(holdout)}",
        "", "[run]", f'run_id = "{run_id}"', f'output_root = "{output_root.as_posix()}"',
        "seed = 42", 'device = "auto"', "deterministic = true", "allow_tf32 = true",
        *( [f'checkpoint_mirror_root = "{args.checkpoint_mirror_root.resolve().as_posix()}"'] if args.checkpoint_mirror_root else [] ),
        "", "[model]", 'architecture = "convnext_tiny"', 'sex_mode = "embedding"',
        "pretrained = true", "image_size = 512", "sex_embedding_dim = 16", "head_hidden_dim = 256", "dropout = 0.2",
        f"target_mean = {target_mean}", f"target_std = {target_std}", "age_class_count = 229",
        "label_distribution_sigma = 2.0", "label_distribution_weight = 0.2", "regression_inference_weight = 0.5",
        'augmentation = "light"', "horizontal_flip_probability = 0.5", "rotation_degrees = 7.0",
        "translation_fraction = 0.03", "scale_min = 0.95", "scale_max = 1.05", "brightness_delta = 0.10",
        "contrast_delta = 0.10", "gamma_min = 0.90", "gamma_max = 1.10", 'preprocessing = "none"',
        'image_normalization = "imagenet"', "", "[training]", "epochs = 35", "batch_size = 12", "grad_accum_steps = 3",
        "learning_rate = 0.0002", "min_learning_rate = 0.000001", "weight_decay = 0.05",
        "smooth_l1_beta_months = 3.0", 'regression_loss = "smooth_l1"', "gradient_clip_norm = 5.0",
        "patience = 8", "min_delta_mae = 0.01", "amp = true", 'amp_dtype = "float16"', "amp_init_scale = 4096.0",
        "", "[operations]", "num_workers = 2", "log_every_steps = 25", "checkpoint_every_steps = 500",
        "checkpoint_every_minutes = 20.0", "periodic_keep = 2", "best_keep = 3", "", "[warnings]",
        "warning_gradient_norm = 50.0", "warning_gradient_consecutive = 3", "warning_prediction_soft_min = 0.0",
        "warning_prediction_soft_max = 228.0", "warning_prediction_hard_min = -60.0", "warning_prediction_hard_max = 300.0",
        "warning_prediction_std_min = 1.0", "warning_sex_mae_gap = 2.5", "warning_overfit_epochs = 3", "age_bins = [0, 60, 120, 180, 229]",
    ]
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "EXP-004 fresh holdout with friend's P7 trainer",
        "source_manifest": str(args.repo_root / "p0_audit/outputs/development_manifest_14036.csv"),
        "data_root": str(args.data_root), "original_train_count": int(len(original_train)),
        "original_official_validation_count": int(len(original_val)), "new_train_count": int(len(new_train)),
        "new_holdout_count": int(len(holdout)), "holdout_fraction": args.holdout_fraction, "seed": args.seed,
        "split_rule": "sha256(seed:image_id) / 0xffffffff < holdout_fraction, applied only to original train",
        "target_mean_new_train": target_mean, "target_std_new_train": target_std,
        "train_manifest_hash": train_hash, "holdout_manifest_hash": val_hash, "test_labels_loaded": False,
        "config": str(config_path), "trainer_source": str(args.repo_root / "p1_baseline"),
    }
    (args.output_dir / "fresh_holdout_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
