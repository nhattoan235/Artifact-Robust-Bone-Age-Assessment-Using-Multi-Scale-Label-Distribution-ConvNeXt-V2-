"""
Backbone cho Giai đoạn 1.

Nguồn tham chiếu:
- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019.
- RadImageNet: Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research
  Dataset for Effective Transfer Learning", Radiology: AI, 2022.
  Trọng số PyTorch chính thức: https://github.com/BMEII-AI/RadImageNet
  LƯU Ý QUAN TRỌNG (đã xác minh từ repo gốc): RadImageNet chỉ phát hành trọng số cho
  ResNet50, DenseNet121, InceptionResNetV2, InceptionV3 — KHÔNG có EfficientNet-B4.
  => EfficientNet-B4 trong ensemble này dùng ImageNet pretrained (torchvision), không
     phải một sai sót mà là giới hạn khách quan của nguồn trọng số RadImageNet công khai.
     Cần nêu rõ điểm này trong phần Methods/Limitations của khóa luận.

LƯU Ý VỀ ĐỊNH DẠNG CHECKPOINT RADIMAGENET (đã xác minh bằng debug thực tế):
Bản .pt PyTorch chính thức của RadImageNet lưu state_dict với key dạng
'backbone.<idx>.<rest>', trong đó <idx> là CHỈ SỐ THỨ TỰ trong
nn.Sequential(*model.children()) lúc họ convert/export mô hình — KHÔNG phải tên
module gốc của torchvision (vd 'conv1', 'layer1', 'bn1'...). Do đó không thể
load_state_dict() trực tiếp — cần remap lại tên key theo đúng thứ tự
named_children() của model đích trước khi nạp (xem _remap_backbone_prefixed_keys).
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class RegressionHead(nn.Module):
    """Head hồi quy tuổi xương, có thể nhận thêm gender embedding (auxiliary linear layer)."""

    def __init__(self, in_features: int, use_gender: bool = True, gender_dim: int = 16):
        super().__init__()
        self.use_gender = use_gender
        if use_gender:
            self.gender_fc = nn.Linear(1, gender_dim)
            in_features = in_features + gender_dim
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, feat, gender=None):
        if self.use_gender:
            g = torch.relu(self.gender_fc(gender))
            feat = torch.cat([feat, g], dim=1)
        return self.head(feat).squeeze(1)


class BoneAgeNet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, use_gender: bool = True):
        super().__init__()
        self.backbone = backbone
        self.head = RegressionHead(feat_dim, use_gender=use_gender)

    def forward(self, image, gender=None):
        feat = self.backbone(image)
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        return self.head(feat, gender)


def _remap_backbone_prefixed_keys(state: dict, model: nn.Module) -> dict:
    """RadImageNet .pt (bản convert PyTorch chính thức) lưu key dạng
    'backbone.<idx>.<rest>', với <idx> là chỉ số thứ tự trong
    nn.Sequential(*model.children()) lúc convert — KHÔNG phải tên module gốc
    (vd 'conv1', 'layer1'...). Remap lại theo đúng tên children thật của model
    hiện tại (dựa trên thứ tự named_children(), giữ nguyên đúng thứ tự khai báo
    trong torchvision) để load_state_dict khớp được.

    Ví dụ với torchvision.resnet50(): children theo thứ tự là
    conv1(0), bn1(1), relu(2), maxpool(3), layer1(4), layer2(5), layer3(6),
    layer4(7), avgpool(8), fc(9) — khớp đúng với 'backbone.0.weight' shape
    [64,3,7,7] (=conv1.weight) và 'backbone.4.0.conv1.weight' (=layer1[0].conv1.weight)
    đã xác minh thực tế từ checkpoint.
    """
    idx_to_name = {str(i): name for i, (name, _) in enumerate(model.named_children())}
    remapped = {}
    skipped_no_prefix_match = 0
    for k, v in state.items():
        parts = k.split(".")
        if len(parts) < 3 or parts[0] != "backbone":
            skipped_no_prefix_match += 1
            continue
        name = idx_to_name.get(parts[1])
        if name is None:
            skipped_no_prefix_match += 1
            continue
        remapped[".".join([name] + parts[2:])] = v
    if skipped_no_prefix_match:
        print(f"[remap] Bỏ qua {skipped_no_prefix_match} key không khớp prefix 'backbone.<idx>.' "
              f"khi remap (có thể là key phụ như 'num_batches_tracked' lệch định dạng, hoặc "
              f"phần fc/classifier cuối vốn không cần nạp).")
    return remapped


def _load_radimagenet_weights(model: nn.Module, weights_path: str, arch: str):
    """Nạp state_dict RadImageNet (.pt) đã tải thủ công. Tự động remap key nếu
    checkpoint dùng prefix 'backbone.<idx>.' (định dạng export chính thức của
    RadImageNet PyTorch) thay vì tên module gốc. Bỏ qua các key không khớp
    (ví dụ fc/classifier cuối) vì ta thay bằng RegressionHead riêng."""
    state = torch.load(weights_path, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state

    if any(k.startswith("backbone.") for k in state.keys()):
        state = _remap_backbone_prefixed_keys(state, model)

    model_state = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in model_state and model_state[k].shape == v.shape}
    missing = len(model_state) - len(matched)
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"[RadImageNet:{arch}] nạp {len(matched)}/{len(model_state)} tensor "
          f"(thiếu/khác shape: {missing}) từ {weights_path}")
    return model


def build_resnet50(cfg):
    net = tvm.resnet50(weights=None if cfg.radimagenet_resnet50_path else
                        tvm.ResNet50_Weights.IMAGENET1K_V2)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    if cfg.radimagenet_resnet50_path:
        net = _load_radimagenet_weights(net, cfg.radimagenet_resnet50_path, "resnet50")
    return BoneAgeNet(net, feat_dim, use_gender=cfg.use_gender_embedding)


def build_densenet121(cfg):
    net = tvm.densenet121(weights=None if cfg.radimagenet_densenet121_path else
                           tvm.DenseNet121_Weights.IMAGENET1K_V1)
    feat_dim = net.classifier.in_features
    net.classifier = nn.Identity()
    if cfg.radimagenet_densenet121_path:
        net = _load_radimagenet_weights(net, cfg.radimagenet_densenet121_path, "densenet121")
    # torchvision DenseNet trả feature map (N,C,H,W) qua .features; bọc thêm pooling
    net = nn.Sequential(net.features, nn.ReLU(inplace=True),
                         nn.AdaptiveAvgPool2d(1))
    return BoneAgeNet(net, feat_dim, use_gender=cfg.use_gender_embedding)


def build_efficientnet_b4(cfg):
    # Không có trọng số RadImageNet cho EfficientNet -> dùng ImageNet (xem docstring trên).
    net = tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.IMAGENET1K_V1)
    feat_dim = net.classifier[-1].in_features
    net.classifier = nn.Identity()
    return BoneAgeNet(net, feat_dim, use_gender=cfg.use_gender_embedding)


BUILDERS = {
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    "efficientnet_b4": build_efficientnet_b4,
}


def build_model(arch: str, cfg):
    if arch not in BUILDERS:
        raise ValueError(f"Kiến trúc không hỗ trợ: {arch}. Chọn trong {list(BUILDERS)}")
    return BUILDERS[arch](cfg)