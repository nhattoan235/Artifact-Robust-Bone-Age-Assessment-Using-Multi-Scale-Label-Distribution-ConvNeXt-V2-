from __future__ import annotations

import torch
import timm
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class BoneAgeConvNeXt(nn.Module):
    def __init__(self, pretrained: bool, sex_embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(backbone.classifier[0], backbone.classifier[1])
        self.backbone = backbone
        self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.GELU())
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim + sex_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        conditioned = torch.cat([features, self.sex_embedding(sex)], dim=1)
        return self.regressor(conditioned).squeeze(1)


class BoneAgeConvNeXtV2(nn.Module):
    """ConvNeXt V2-Tiny FCMAE fine-tuned tren ImageNet-1K, khong dung ImageNet-22K."""

    TIMM_MODEL_NAME = "convnextv2_tiny.fcmae_ft_in1k"

    def __init__(self, pretrained: bool, sex_embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.backbone = timm.create_model(
            self.TIMM_MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features
        self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.GELU())
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim + sex_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        conditioned = torch.cat([features, self.sex_embedding(sex)], dim=1)
        return self.regressor(conditioned).squeeze(1)


class BoneAgeConvNeXtMultiScale(nn.Module):
    """ConvNeXt-Tiny với fusion đặc trưng từ bốn stage.

    Lớp fusion được khởi tạo để sao chép nguyên vẹn stage cuối. Vì vậy tại
    thời điểm bắt đầu, nhánh ảnh tương đương D0 và các stage sớm chỉ đóng góp
    khi quá trình tối ưu học được trọng số hữu ích.
    """

    stage_indices = (1, 3, 5, 7)
    stage_channels = (96, 192, 384, 768)
    fused_dim = 768

    def __init__(self, pretrained: bool, sex_embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.features = backbone.features
        self.stage_norms = nn.ModuleList(
            nn.LayerNorm(channels, eps=1e-6) for channels in self.stage_channels
        )
        with torch.no_grad():
            self.stage_norms[-1].weight.copy_(backbone.classifier[0].weight)
            self.stage_norms[-1].bias.copy_(backbone.classifier[0].bias)
        self.fusion = nn.Linear(sum(self.stage_channels), self.fused_dim, bias=False)
        with torch.no_grad():
            self.fusion.weight.zero_()
            final_offset = sum(self.stage_channels[:-1])
            self.fusion.weight[:, final_offset:].copy_(torch.eye(self.fused_dim))

        self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.GELU())
        self.regressor = nn.Sequential(
            nn.Linear(self.fused_dim + sex_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_features(self, image: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        stage_vectors = []
        x = self.features[0](image)
        norm_index = 0
        for index in range(1, len(self.features)):
            x = self.features[index](x)
            if index in self.stage_indices:
                pooled = x.mean(dim=(-2, -1))
                stage_vectors.append(self.stage_norms[norm_index](pooled))
                norm_index += 1
        fused = self.fusion(torch.cat(stage_vectors, dim=1))
        return stage_vectors, fused

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        _, fused = self.forward_features(image)
        conditioned = torch.cat([fused, self.sex_embedding(sex)], dim=1)
        return self.regressor(conditioned).squeeze(1)


class BoneAgeConvNeXtLabelDistribution(nn.Module):
    """D0 ConvNeXt-Tiny với một head phụ học phân phối tuổi theo tháng."""

    def __init__(
        self, pretrained: bool, sex_embedding_dim: int, hidden_dim: int,
        dropout: float, age_class_count: int,
    ):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(backbone.classifier[0], backbone.classifier[1])
        self.backbone = backbone
        self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.GELU())
        conditioned_dim = feature_dim + sex_embedding_dim
        self.regressor = nn.Sequential(
            nn.Linear(conditioned_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.distribution_head = nn.Sequential(
            nn.Linear(conditioned_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, age_class_count),
        )

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        conditioned = torch.cat([features, self.sex_embedding(sex)], dim=1)
        return {
            "regression": self.regressor(conditioned).squeeze(1),
            "distribution_logits": self.distribution_head(conditioned),
        }


class SmokeNet(nn.Module):
    """Backbone nhỏ chỉ dùng để kiểm thử trainer/checkpoint trên CPU."""
    def __init__(self, sex_embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.GELU())
        self.regressor = nn.Sequential(nn.Linear(16 + sex_embedding_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        return self.regressor(torch.cat([self.features(image), self.sex_embedding(sex)], dim=1)).squeeze(1)


def build_model(
    architecture: str, pretrained: bool, sex_embedding_dim: int,
    hidden_dim: int, dropout: float, age_class_count: int = 229,
) -> nn.Module:
    if architecture == "convnext_tiny":
        return BoneAgeConvNeXt(pretrained, sex_embedding_dim, hidden_dim, dropout)
    if architecture == "convnextv2_tiny":
        return BoneAgeConvNeXtV2(pretrained, sex_embedding_dim, hidden_dim, dropout)
    if architecture == "convnext_tiny_multiscale":
        return BoneAgeConvNeXtMultiScale(pretrained, sex_embedding_dim, hidden_dim, dropout)
    if architecture == "convnext_tiny_ldl":
        return BoneAgeConvNeXtLabelDistribution(
            pretrained, sex_embedding_dim, hidden_dim, dropout, age_class_count
        )
    if architecture == "smoke_cnn":
        return SmokeNet(sex_embedding_dim, hidden_dim, dropout)
    raise ValueError(f"Kiến trúc không hỗ trợ: {architecture}")
