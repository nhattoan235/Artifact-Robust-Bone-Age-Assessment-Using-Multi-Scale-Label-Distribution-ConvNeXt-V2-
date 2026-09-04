from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


VIEW_COUNTS = {
    "global_only": 1,
    "six_roi_only": 6,
    "global_plus_six": 7,
}


class SharedViewConvNeXt(nn.Module):
    def __init__(
        self,
        *,
        pretrained: bool,
        view_mode: str,
        sex_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if view_mode not in VIEW_COUNTS:
            raise ValueError(f"unsupported view_mode: {view_mode}")
        self.view_mode = view_mode
        self.view_count = VIEW_COUNTS[view_mode]
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(backbone.classifier[0], backbone.classifier[1])
        self.backbone = backbone
        self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.GELU())
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * self.view_count + sex_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, views: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        if views.ndim != 5:
            raise ValueError(f"views must have shape [B,V,C,H,W], got {tuple(views.shape)}")
        batch, view_count, channels, height, width = views.shape
        if view_count != self.view_count:
            raise ValueError(f"{self.view_mode} requires {self.view_count} views, got {view_count}")
        features = self.backbone(views.reshape(batch * view_count, channels, height, width))
        features = features.reshape(batch, view_count * features.shape[-1])
        conditioned = torch.cat([features, self.sex_embedding(sex)], dim=1)
        return self.regressor(conditioned).squeeze(1)


def build_c4_model(
    *,
    pretrained: bool,
    view_mode: str,
    sex_embedding_dim: int = 16,
    hidden_dim: int = 256,
    dropout: float = 0.2,
) -> SharedViewConvNeXt:
    return SharedViewConvNeXt(
        pretrained=pretrained,
        view_mode=view_mode,
        sex_embedding_dim=sex_embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

