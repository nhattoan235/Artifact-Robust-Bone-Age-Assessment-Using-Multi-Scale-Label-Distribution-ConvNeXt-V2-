"""
Config Giai đoạn 1 — Ensemble đa kiến trúc + RadImageNet pretrained.
Baseline tham chiếu (Matsuoka et al., arXiv:2511.23066): MAE 6.26 tháng.

CHỈ SỬA CÁC ĐƯỜNG DẪN Ở PHẦN "== ĐƯỜNG DẪN CẦN CHỈNH ==" — phần còn lại
được giữ khớp với mô tả Methods của baseline để so sánh công bằng
(apples-to-apples), trừ các điểm khác biệt kiến trúc chủ đích của Giai đoạn 1.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # == ĐƯỜNG DẪN CẦN CHỈNH ==
    # Drive: gpt-image-bone-age-synthesis/rsna_original/{boneage-training-dataset(.csv), boneage-test-dataset(.csv)}
    data_root: str = "/content/drive/MyDrive/gpt-image-bone-age-synthesis/rsna_original"
    train_csv: str = "boneage-training-dataset.csv"    # id,boneage,male
    train_img_dir: str = "boneage-training-dataset"    # ảnh .png, tên = {id}.png
    test_csv: str = "boneage-test-dataset.csv"
    test_img_dir: str = "boneage-test-dataset"
    output_dir: str = "/content/drive/MyDrive/gpt-image-bone-age-synthesis/phase1_outputs"

    # Trọng số RadImageNet (PyTorch .pt, tải thủ công từ
    # https://github.com/BMEII-AI/RadImageNet do yêu cầu đăng ký/thoả thuận sử dụng).
    # Repo chính thức CHỈ cung cấp: ResNet50, DenseNet121, InceptionResNetV2, InceptionV3.
    # -> KHÔNG có EfficientNet-B4: nhánh này bắt buộc dùng ImageNet.
    radimagenet_resnet50_path: Optional[str] = "/content/drive/MyDrive/gpt-image-bone-age-synthesis/RadImageNet_pytorch/ResNet50.pt"
    radimagenet_densenet121_path: Optional[str] = "/content/drive/MyDrive/gpt-image-bone-age-synthesis/RadImageNet_pytorch/DenseNet121.pt"

    # == GIỮ KHỚP BASELINE ==
    image_size: int = 320          # baseline dùng 320x320 cho toàn bộ ảnh input
    in_channels: int = 3           # RGB hoá qua channel duplication (baseline)
    n_folds: int = 5
    sex_specific: bool = True      # 2 ensemble riêng: nam / nữ (theo baseline)
    seed: int = 42

    batch_size: int = 16
    # Override batch size theo kiến trúc — cần thiết với GPU VRAM nhỏ (vd RTX 4050 6GB):
    # EfficientNet-B4 tốn VRAM hơn ResNet50/DenseNet121 đáng kể ở cùng batch size.
    # effective batch size = batch_size * grad_accum_steps -> vẫn giữ khớp baseline
    # dù batch_size vật lý nhỏ hơn, chỉ cần tăng grad_accum_steps tương ứng.
    per_arch_batch_size: dict = field(default_factory=lambda: {
        "resnet50": 16,
        "densenet121": 12,
        "efficientnet_b4": 6,
    })
    grad_accum_steps: int = 10     # baseline: gradient accumulation 10 bước (áp cho batch_size=16)
    use_amp: bool = True           # mixed precision — bắt buộc với VRAM <= 8GB
    epochs: int = 50

    # == Checkpoint / resume cho Colab (session giới hạn ~6h, có thể bị ngắt) ==
    resume: bool = True
    save_every_n_epochs: int = 1   # lưu "last" checkpoint sau mỗi N epoch để resume
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"   # cosine annealing LR (baseline)
    loss: str = "smooth_l1"        # Huber loss (baseline)

    # Augmentation (baseline): affine (rotation ±15°, translate, scale),
    # contrast/brightness, sharpening
    aug_rotate_deg: float = 15.0
    aug_translate_pct: float = 0.05
    aug_scale_range: tuple = (0.9, 1.1)
    aug_brightness_contrast: float = 0.2
    aug_sharpen_prob: float = 0.3

    use_gender_embedding: bool = True   # auxiliary linear layer (baseline, optional)

    # == THAY ĐỔI CỦA GIAI ĐOẠN 1 (so với baseline: 5-fold cùng ResNet50) ==
    architectures: tuple = ("resnet50", "densenet121", "efficientnet_b4")
    ensemble_mode: str = "stacking"     # "average" hoặc "stacking" (meta-learner tuyến tính)

    num_workers: int = 4
    device: str = "cuda"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


cfg = Config()


def effective_batch_size(cfg) -> int:
    return cfg.batch_size * cfg.grad_accum_steps  # baseline: 16*10 = 160


def grad_accum_for_arch(cfg, arch: str) -> int:
    """Tính lại số bước accumulation để effective batch size ~ khớp baseline (160)
    dù batch_size vật lý mỗi kiến trúc khác nhau do giới hạn VRAM."""
    phys_bs = cfg.per_arch_batch_size.get(arch, cfg.batch_size)
    target = effective_batch_size(cfg)
    return max(1, round(target / phys_bs))
