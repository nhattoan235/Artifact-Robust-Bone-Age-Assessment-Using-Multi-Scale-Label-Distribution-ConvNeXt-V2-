from __future__ import annotations

import torch
import timm
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)


class BoneAgeConvNeXt(nn.Module):
    SUPPORTED_SEX_MODES = {"none", "embedding", "dual_output"}

    def __init__(
        self, pretrained: bool, sex_embedding_dim: int, hidden_dim: int,
        dropout: float, sex_mode: str = "embedding",
    ):
        super().__init__()
        if sex_mode not in self.SUPPORTED_SEX_MODES:
            raise ValueError(f"sex_mode không hỗ trợ: {sex_mode}")
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(backbone.classifier[0], backbone.classifier[1])
        self.backbone = backbone
        self.sex_mode = sex_mode
        if sex_mode == "embedding":
            # Giữ nguyên tên module và thứ tự khởi tạo của E1 để checkpoint cũ
            # tiếp tục load strict và forward không thay đổi.
            self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.GELU())
            self.regressor = nn.Sequential(
                nn.Linear(feature_dim + sex_embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        elif sex_mode == "none":
            self.regressor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.shared_regressor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.female_head = nn.Linear(hidden_dim, 1)
            self.male_head = nn.Linear(hidden_dim, 1)

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        if self.sex_mode == "embedding":
            conditioned = torch.cat([features, self.sex_embedding(sex)], dim=1)
            return self.regressor(conditioned).squeeze(1)
        if self.sex_mode == "none":
            return self.regressor(features).squeeze(1)
        if tuple(sex.shape) != (features.shape[0], 1):
            raise ValueError("sex phải có shape [batch, 1] cho dual_output")
        shared = self.shared_regressor(features)
        female_prediction = self.female_head(shared).squeeze(1)
        male_prediction = self.male_head(shared).squeeze(1)
        return torch.where(sex.squeeze(1) >= 0.5, male_prediction, female_prediction)


class BoneAgeEfficientNetB0(nn.Module):
    """EfficientNet-B0 with sex input and a Deeplasia-style regression head."""

    def __init__(self, pretrained: bool, sex_embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        first_conv = backbone.features[0][0]
        one_channel_conv = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=1,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )
        with torch.no_grad():
            if pretrained:
                one_channel_conv.weight.copy_(
                    first_conv.weight.sum(dim=1, keepdim=True)
                )
            else:
                one_channel_conv.weight.copy_(
                    first_conv.weight.mean(dim=1, keepdim=True)
                )
            if first_conv.bias is not None:
                one_channel_conv.bias.copy_(first_conv.bias)
        backbone.features[0][0] = one_channel_conv
        feature_dim = backbone.classifier[1].in_features
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.ReLU())
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim + sex_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        # Deeplasia huấn luyện EfficientNet với ảnh grayscale một kênh.
        features = self.pool(self.features(image[:, :1])).flatten(1)
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
    def __init__(
        self, sex_embedding_dim: int, hidden_dim: int, dropout: float,
        sex_mode: str = "embedding",
    ):
        super().__init__()
        if sex_mode not in BoneAgeConvNeXt.SUPPORTED_SEX_MODES:
            raise ValueError(f"sex_mode không hỗ trợ: {sex_mode}")
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.sex_mode = sex_mode
        if sex_mode == "embedding":
            self.sex_embedding = nn.Sequential(nn.Linear(1, sex_embedding_dim), nn.GELU())
            self.regressor = nn.Sequential(
                nn.Linear(16 + sex_embedding_dim, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
        elif sex_mode == "none":
            self.regressor = nn.Sequential(
                nn.Linear(16, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.shared_regressor = nn.Sequential(
                nn.Linear(16, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            )
            self.female_head = nn.Linear(hidden_dim, 1)
            self.male_head = nn.Linear(hidden_dim, 1)

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        if self.sex_mode == "embedding":
            return self.regressor(
                torch.cat([features, self.sex_embedding(sex)], dim=1)
            ).squeeze(1)
        if self.sex_mode == "none":
            return self.regressor(features).squeeze(1)
        if tuple(sex.shape) != (features.shape[0], 1):
            raise ValueError("sex phải có shape [batch, 1] cho dual_output")
        shared = self.shared_regressor(features)
        female_prediction = self.female_head(shared).squeeze(1)
        male_prediction = self.male_head(shared).squeeze(1)
        return torch.where(sex.squeeze(1) >= 0.5, male_prediction, female_prediction)


def build_model(
    architecture: str, pretrained: bool, sex_embedding_dim: int,
    hidden_dim: int, dropout: float, age_class_count: int = 229,
    sex_mode: str = "embedding",
) -> nn.Module:
    if sex_mode != "embedding" and architecture not in {"convnext_tiny", "smoke_cnn"}:
        raise ValueError(
            "sex_mode none/dual_output hiện chỉ hỗ trợ convnext_tiny hoặc smoke_cnn"
        )
    if architecture == "convnext_tiny":
        return BoneAgeConvNeXt(
            pretrained, sex_embedding_dim, hidden_dim, dropout, sex_mode
        )
    if architecture == "efficientnet_b0":
        return BoneAgeEfficientNetB0(pretrained, sex_embedding_dim, hidden_dim, dropout)
    if architecture == "convnextv2_tiny":
        return BoneAgeConvNeXtV2(pretrained, sex_embedding_dim, hidden_dim, dropout)
    if architecture == "convnext_tiny_multiscale":
        return BoneAgeConvNeXtMultiScale(pretrained, sex_embedding_dim, hidden_dim, dropout)
    if architecture == "convnext_tiny_ldl":
        return BoneAgeConvNeXtLabelDistribution(
            pretrained, sex_embedding_dim, hidden_dim, dropout, age_class_count
        )
    if architecture == "smoke_cnn":
        return SmokeNet(sex_embedding_dim, hidden_dim, dropout, sex_mode)
    raise ValueError(f"Kiến trúc không hỗ trợ: {architecture}")
