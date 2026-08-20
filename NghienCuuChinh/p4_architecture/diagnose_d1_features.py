from __future__ import annotations

import json
from pathlib import Path

import torch

from p1_baseline.config import load_config
from p1_baseline.data import BoneAgeDataset, load_manifest
from p1_baseline.model import build_model


CONFIG = Path("p1_baseline/configs/p4_d1_convnextv2_tiny_b6a6.toml")
CHECKPOINT = Path("p1_baseline/runs/P4_D1_CONVNEXTV2_TINY_FCMAE_IN1K_B6A6_SEED42/best_mae.ckpt")


def summarize(features: torch.Tensor) -> dict[str, float]:
    features = features.float()
    return {
        "mean": features.mean().item(),
        "std_all": features.std().item(),
        "mean_feature_std_across_images": features.std(dim=0).mean().item(),
        "mean_pair_distance": torch.pdist(features).mean().item(),
        "max_abs": features.abs().max().item(),
    }


def main() -> None:
    cfg = load_config(CONFIG)
    rows = load_manifest(cfg.val_manifest, "validation_official")[:8]
    dataset = BoneAgeDataset(rows, cfg.image_size, cfg.target_mean, cfg.target_std)
    images = torch.stack([dataset[i]["image"] for i in range(len(dataset))])
    sex = torch.stack([dataset[i]["sex"] for i in range(len(dataset))])

    fresh = build_model(cfg.architecture, True, cfg.sex_embedding_dim, cfg.head_hidden_dim, cfg.dropout).eval()
    trained = build_model(cfg.architecture, False, cfg.sex_embedding_dim, cfg.head_hidden_dim, cfg.dropout).eval()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    trained.load_state_dict(state["model"])

    result: dict[str, object] = {}
    with torch.no_grad():
        fresh_features = fresh.backbone(images)
        trained_features = trained.backbone(images)
        result["fresh_features"] = summarize(fresh_features)
        result["trained_features"] = summarize(trained_features)
        result["fresh_predictions_months"] = (
            fresh(images, sex) * cfg.target_std + cfg.target_mean
        ).tolist()
        result["trained_predictions_months"] = (
            trained(images, sex) * cfg.target_std + cfg.target_mean
        ).tolist()

    first = trained.regressor[0].weight.detach().float()
    result["trained_head_image_weight_norm"] = first[:, : trained.backbone.num_features].norm().item()
    result["trained_head_sex_weight_norm"] = first[:, trained.backbone.num_features :].norm().item()
    result["unique_trained_predictions_rounded"] = len(
        set(round(value, 5) for value in result["trained_predictions_months"])
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
