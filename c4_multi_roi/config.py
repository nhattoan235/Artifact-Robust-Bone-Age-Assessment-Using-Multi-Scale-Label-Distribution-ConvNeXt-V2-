from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class C4Config:
    manifest: str
    image_root: str = "."
    validation_fold: int = 1
    expected_rows: int = 14036
    expected_manifest_sha256: str = ""
    run_id: str = "C4_MULTI_ROI_V1_FOLD_1"
    output_root: str = "c4_multi_roi/runs/C4_MULTI_ROI_V1"
    checkpoint_mirror_root: str = ""
    seed: int = 42
    device: str = "auto"
    deterministic: bool = True
    allow_tf32: bool = True
    view_mode: str = "global_plus_six"
    pretrained: bool = True
    image_size: int = 512
    sex_embedding_dim: int = 16
    head_hidden_dim: int = 256
    dropout: float = 0.2
    target_mean: float = 127.23833273957962
    target_std: float = 41.248974358112605
    augmentation: str = "light"
    epochs: int = 35
    batch_size: int = 1
    grad_accum_steps: int = 36
    learning_rate: float = 0.0002
    min_learning_rate: float = 0.000001
    weight_decay: float = 0.05
    smooth_l1_beta_months: float = 3.0
    gradient_clip_norm: float = 5.0
    patience: int = 8
    min_delta_mae: float = 0.01
    amp: bool = True
    amp_dtype: str = "auto"
    amp_init_scale: float = 4096.0
    num_workers: int = 2
    log_every_steps: int = 25
    checkpoint_every_steps: int = 250
    checkpoint_every_minutes: float = 15.0

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_id

    @property
    def mirror_dir(self) -> Path | None:
        return Path(self.checkpoint_mirror_root) / self.run_id if self.checkpoint_mirror_root else None


OPERATIONAL_KEYS = {
    "output_root",
    "checkpoint_mirror_root",
    "device",
    "num_workers",
    "log_every_steps",
    "checkpoint_every_steps",
    "checkpoint_every_minutes",
}


def scientific_config_hash(config: C4Config) -> str:
    value = {key: item for key, item in asdict(config).items() if key not in OPERATIONAL_KEYS}
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> C4Config:
    with Path(path).open("rb") as handle:
        document = tomllib.load(handle)
    values: dict = {}
    allowed = {field.name for field in fields(C4Config)}
    for section, section_values in document.items():
        if not isinstance(section_values, dict):
            raise ValueError(f"TOML top-level value must be a section: {section}")
        for key, value in section_values.items():
            if key not in allowed:
                raise ValueError(f"unknown config key: {section}.{key}")
            if key in values:
                raise ValueError(f"duplicate config key: {key}")
            values[key] = value
    config = C4Config(**values)
    if config.view_mode not in {"global_only", "six_roi_only", "global_plus_six"}:
        raise ValueError(f"invalid view_mode: {config.view_mode}")
    if not 1 <= config.validation_fold <= 5:
        raise ValueError("validation_fold must be 1..5")
    if config.batch_size <= 0 or config.grad_accum_steps <= 0 or config.epochs <= 0:
        raise ValueError("batch_size, grad_accum_steps and epochs must be positive")
    if config.amp_dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("amp_dtype must be auto, float16 or bfloat16")
    return config

