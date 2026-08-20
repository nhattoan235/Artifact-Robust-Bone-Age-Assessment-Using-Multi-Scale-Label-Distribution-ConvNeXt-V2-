from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from .config import load_config, scientific_config_hash
from .data import BoneAgeDataset, load_manifest, manifest_hash
from .model import build_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="p1_baseline/configs/p1_a0.toml")
    parser.add_argument("--no-pretrained", action="store_true", help="Không tải weights; chỉ kiểm tra pipeline")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train = load_manifest(cfg.train_manifest, "train")
    val = load_manifest(cfg.val_manifest, "validation_official")
    checks = {
        "train_count": len(train) == cfg.expected_train_count,
        "val_count": len(val) == cfg.expected_val_count,
        "train_hash": manifest_hash(train) == cfg.expected_train_hash,
        "val_hash": manifest_hash(val) == cfg.expected_val_hash,
        "test_path_absent": all(
            "test" not in str(value).lower()
            for key, value in asdict(cfg).items()
            if "path" in key.lower() or "manifest" in key.lower() or "root" in key.lower()
        ),
    }
    dataset = BoneAgeDataset(
        train[:1], cfg.image_size, cfg.target_mean, cfg.target_std,
        preprocessing=cfg.preprocessing, preprocessed_root=cfg.preprocessed_root,
        image_root=cfg.image_root, image_normalization=cfg.image_normalization,
    )
    sample = dataset[0]
    model = build_model(
        cfg.architecture, cfg.pretrained and not args.no_pretrained,
        cfg.sex_embedding_dim, cfg.head_hidden_dim, cfg.dropout, cfg.age_class_count,
    )
    with torch.inference_mode():
        output = model(sample["image"].unsqueeze(0), sample["sex"].unsqueeze(0))
    checks["sample_shape"] = tuple(sample["image"].shape) == (3, cfg.image_size, cfg.image_size)
    if isinstance(output, dict):
        checks["forward_shape"] = (
            tuple(output["regression"].shape) == (1,)
            and tuple(output["distribution_logits"].shape) == (1, cfg.age_class_count)
        )
    else:
        checks["forward_shape"] = tuple(output.shape) == (1,)
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "config_hash": scientific_config_hash(cfg), "sample_id": sample["image_id"], "sample_target_months": float(sample["target_months"])}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
