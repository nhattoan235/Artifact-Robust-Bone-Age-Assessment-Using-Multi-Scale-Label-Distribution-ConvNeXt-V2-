from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    run_id: str = "P1_A0_CONVNEXT_TINY_SEED42"
    train_manifest: str = "p0_audit/outputs/train_manifest.csv"
    val_manifest: str = "p0_audit/outputs/validation_manifest.csv"
    output_root: str = "p1_baseline/runs"
    expected_train_hash: str = "7328667e6822ab074d442155e33eba89606861bb13bbf804be8a1138c78f5285"
    expected_val_hash: str = "f650a20432b557405035d12c52716a6c40fdf20e1343ac3ad0cfb7cd8db21631"
    expected_train_count: int = 12611
    expected_val_count: int = 1425
    image_root: str = ""
    checkpoint_mirror_root: str = ""

    architecture: str = "convnext_tiny"
    pretrained: bool = True
    image_size: int = 512
    sex_embedding_dim: int = 16
    head_hidden_dim: int = 256
    dropout: float = 0.2
    target_mean: float = 127.32098961224328
    target_std: float = 41.18202139939604
    age_class_count: int = 229
    label_distribution_sigma: float = 2.0
    label_distribution_weight: float = 0.2
    regression_inference_weight: float = 0.5
    augmentation: str = "none"
    horizontal_flip_probability: float = 0.0
    rotation_degrees: float = 0.0
    translation_fraction: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    brightness_delta: float = 0.0
    contrast_delta: float = 0.0
    gamma_min: float = 1.0
    gamma_max: float = 1.0
    shear_degrees: float = 0.0
    clahe_probability: float = 0.0
    sharpen_probability: float = 0.0
    preprocessing: str = "none"
    preprocessed_root: str = "p3_preprocessing/outputs/official_mask_v1/cache"
    image_normalization: str = "imagenet"

    seed: int = 42
    epochs: int = 35
    batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    scheduler_factor: float = 0.2
    scheduler_patience: int = 10
    scheduler_cooldown: int = 0
    scheduler_min_lr: float = 1e-4
    smooth_l1_beta_months: float = 3.0
    regression_loss: str = "smooth_l1"
    sampling_strategy: str = "permutation"
    gradient_clip_norm: float = 5.0
    patience: int = 8
    min_delta_mae: float = 0.01
    amp: bool = True
    amp_dtype: str = "auto"
    amp_init_scale: float = 4096.0

    num_workers: int = 2
    log_every_steps: int = 25
    checkpoint_every_steps: int = 500
    checkpoint_every_minutes: float = 20.0
    periodic_keep: int = 2
    best_keep: int = 3
    device: str = "auto"
    deterministic: bool = True
    allow_tf32: bool = True

    warning_gradient_norm: float = 50.0
    warning_gradient_consecutive: int = 3
    warning_prediction_soft_min: float = 0.0
    warning_prediction_soft_max: float = 228.0
    warning_prediction_hard_min: float = -60.0
    warning_prediction_hard_max: float = 300.0
    warning_prediction_std_min: float = 1.0
    warning_sex_mae_gap: float = 2.5
    warning_overfit_epochs: int = 3
    age_bins: list[int] = field(default_factory=lambda: [0, 60, 120, 180, 229])

    max_train_samples: int | None = None
    max_val_samples: int | None = None

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_id


OPERATIONAL_KEYS = {
    "num_workers", "log_every_steps", "checkpoint_every_steps", "periodic_keep",
    "best_keep", "checkpoint_every_minutes", "device", "output_root", "run_id",
    "image_root", "checkpoint_mirror_root",
}


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    flat: dict[str, Any] = {}
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)
    unknown = set(flat) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Khóa cấu hình không được hỗ trợ: {sorted(unknown)}")
    cfg = Config(**flat)
    if cfg.augmentation not in {"none", "flip", "light", "deeplasia_fancy"}:
        raise ValueError("augmentation phải là none, flip, light hoặc deeplasia_fancy")
    if cfg.preprocessing not in {"none", "official_mask_v1", "deeplasia_mask_v1", "deeplasia_mask_histogram_v1"}:
        raise ValueError("preprocessing không được hỗ trợ")
    if cfg.image_normalization not in {"imagenet", "per_image_zscore", "zero_one"}:
        raise ValueError("image_normalization không được hỗ trợ")
    if cfg.regression_loss not in {"smooth_l1", "mae", "mse"}:
        raise ValueError("regression_loss phải là smooth_l1, mae hoặc mse")
    if cfg.sampling_strategy not in {"permutation", "manifest_weighted"}:
        raise ValueError("sampling_strategy phải là permutation hoặc manifest_weighted")
    if cfg.optimizer_name not in {"adamw", "adam"}:
        raise ValueError("optimizer_name phải là adamw hoặc adam")
    if cfg.scheduler_name not in {"cosine", "reduce_on_plateau"}:
        raise ValueError("scheduler_name phải là cosine hoặc reduce_on_plateau")
    if not 0.0 < cfg.scheduler_factor < 1.0:
        raise ValueError("scheduler_factor phải trong (0, 1)")
    if cfg.scheduler_patience < 0 or cfg.scheduler_cooldown < 0:
        raise ValueError("scheduler_patience/scheduler_cooldown phải >= 0")
    if cfg.scheduler_min_lr < 0:
        raise ValueError("scheduler_min_lr phải >= 0")
    if not 0.0 <= cfg.horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability phải trong [0, 1]")
    if not (0 < cfg.scale_min <= cfg.scale_max):
        raise ValueError("scale_min/scale_max không hợp lệ")
    if cfg.shear_degrees < 0 or not 0.0 <= cfg.clahe_probability <= 1.0 or not 0.0 <= cfg.sharpen_probability <= 1.0:
        raise ValueError("Các tham số deeplasia_fancy không hợp lệ")
    if not (cfg.warning_prediction_hard_min < cfg.warning_prediction_soft_min < cfg.warning_prediction_soft_max < cfg.warning_prediction_hard_max):
        raise ValueError("Các ngưỡng prediction soft/hard không hợp lệ")
    if cfg.age_class_count < 2:
        raise ValueError("age_class_count phải >= 2")
    if cfg.expected_train_count <= 0 or cfg.expected_val_count <= 0:
        raise ValueError("expected_train_count/expected_val_count phải > 0")
    if cfg.label_distribution_sigma <= 0:
        raise ValueError("label_distribution_sigma phải > 0")
    if cfg.label_distribution_weight < 0:
        raise ValueError("label_distribution_weight phải >= 0")
    if not 0.0 <= cfg.regression_inference_weight <= 1.0:
        raise ValueError("regression_inference_weight phải trong [0, 1]")
    return cfg


def scientific_config_hash(cfg: Config) -> str:
    values = {k: v for k, v in asdict(cfg).items() if k not in OPERATIONAL_KEYS}
    blob = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_resolved_yaml(cfg: Config, path: Path) -> None:
    """Ghi YAML đơn giản, không cần dependency PyYAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in asdict(cfg).items():
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, (list, dict)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value)
        lines.append(f"{key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
