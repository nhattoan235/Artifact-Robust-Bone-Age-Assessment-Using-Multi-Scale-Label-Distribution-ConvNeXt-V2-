from __future__ import annotations

import argparse
from pathlib import Path

from .cache_utils import sha256_file


TEMPLATE = '''[data]
manifest = "c4_multi_roi/cache/C4_MULTI_ROI_V1/manifest.csv"
image_root = "."
validation_fold = {fold}
expected_rows = 14036
expected_manifest_sha256 = "{manifest_sha256}"

[run]
run_id = "C4_MULTI_ROI_V1_FOLD_{fold}"
output_root = "/content/c4_local_runs/C4_MULTI_ROI_V1"
checkpoint_mirror_root = "/content/drive/MyDrive/data/c4_multi_roi_runs"
seed = 42
device = "auto"
deterministic = true
allow_tf32 = true

[model]
view_mode = "global_plus_six"
pretrained = true
image_size = 512
sex_embedding_dim = 16
head_hidden_dim = 256
dropout = 0.2
target_mean = 127.23833273957962
target_std = 41.248974358112605
augmentation = "light"

[training]
epochs = 35
batch_size = 1
grad_accum_steps = 36
learning_rate = 0.0002
min_learning_rate = 0.000001
weight_decay = 0.05
smooth_l1_beta_months = 3.0
gradient_clip_norm = 5.0
patience = 8
min_delta_mae = 0.01
amp = true
amp_dtype = "auto"
amp_init_scale = 4096.0

[operations]
num_workers = 2
log_every_steps = 25
checkpoint_every_steps = 250
checkpoint_every_minutes = 15.0
'''


def write_colab_configs(output: Path, *, manifest_sha256: str) -> list[Path]:
    if len(manifest_sha256) != 64:
        raise ValueError("manifest_sha256 must have 64 characters")
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for fold in range(1, 6):
        path = output / f"fold_{fold}_colab.toml"
        path.write_text(TEMPLATE.format(fold=fold, manifest_sha256=manifest_sha256), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("c4_multi_roi/cache/C4_MULTI_ROI_V1/manifest.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("c4_multi_roi/configs"))
    args = parser.parse_args()
    paths = write_colab_configs(args.output, manifest_sha256=sha256_file(args.manifest))
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
