from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1_baseline.data import manifest_hash  # noqa: E402

FOLDS = range(1, 6)
TARGETS = {
    1: (127.23833273957962, 41.248974358112605),
    2: (127.23833273957962, 41.248974358112605),
    3: (127.23833273957962, 41.248974358112605),
    4: (127.23833273957962, 41.248974358112605),
    5: (127.23833273957962, 41.248974358112605),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_hash(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    keys = ["split", "image_id", "bone_age_months", "sex", "sha256"]
    keys.extend(
        key for key in ["age_bin", "sex_age_stratum", "sample_weight", "audit_status", "audit_reason"]
        if any(key in row for row in rows)
    )
    for row in sorted(rows, key=lambda item: str(int(float(item["image_id"])) )):
        digest.update(("\t".join(row.get(key, "") for key in keys) + "\n").encode())
    return digest.hexdigest()


def transform_rows(rows: list[dict[str, str]], split: str, source_splits: dict[str, str]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        image_id = str(int(float(row["image_id"])))
        transformed = dict(row)
        transformed["split"] = split
        source_split = source_splits[image_id]
        transformed["image_path"] = f"c3_roi/cache/C3_ROI_V1/roi/{source_split}/{image_id}.jpg"
        transformed["roi_status"] = "blocked_until_roi_cache_pass"
        output.append(transformed)
    return output


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_config(fold: int, train: list[dict[str, str]], val: list[dict[str, str]], target_mean: float, target_std: float) -> str:
    return f'''[data]
train_manifest = "c3_roi/manifests/fold_{fold}_train.csv"
val_manifest = "c3_roi/manifests/fold_{fold}_validation.csv"
expected_train_hash = "{manifest_hash(train)}"
expected_val_hash = "{manifest_hash(val)}"
expected_train_count = {len(train)}
expected_val_count = {len(val)}
image_root = "."

[run]
run_id = "C3_ROI_V1_FOLD_{fold}"
output_root = "c3_roi/runs/C3_ROI_V1"
seed = 42
device = "auto"
deterministic = true
allow_tf32 = true

[model]
architecture = "convnext_tiny"
pretrained = true
image_size = 512
sex_embedding_dim = 16
head_hidden_dim = 256
dropout = 0.2
target_mean = {target_mean!r}
target_std = {target_std!r}
augmentation = "light"
horizontal_flip_probability = 0.5
rotation_degrees = 7.0
translation_fraction = 0.03
scale_min = 0.95
scale_max = 1.05
brightness_delta = 0.10
contrast_delta = 0.10
gamma_min = 0.90
gamma_max = 1.10
preprocessing = "none"
image_normalization = "imagenet"

[training]
epochs = 35
batch_size = 6
grad_accum_steps = 6
learning_rate = 0.0002
min_learning_rate = 0.000001
weight_decay = 0.05
smooth_l1_beta_months = 3.0
regression_loss = "smooth_l1"
sampling_strategy = "permutation"
gradient_clip_norm = 5.0
patience = 8
min_delta_mae = 0.01
amp = true
amp_dtype = "auto"
amp_init_scale = 4096.0

[operations]
num_workers = 2
log_every_steps = 25
checkpoint_every_steps = 500
checkpoint_every_minutes = 20.0
periodic_keep = 2
best_keep = 3

[warnings]
age_bins = [0, 60, 120, 180, 229]
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "c3_roi")
    args = parser.parse_args()
    manifest_root = args.output_root / "manifests"
    config_root = args.output_root / "configs"
    registry = {
        "status": "PREPARED_BLOCKED",
        "protocol": "C3 independent ROI ConvNeXt-Tiny on locked D3/P7 folds",
        "seed": 42,
        "test_used": False,
        "assignment": {"colab": [1, 2, 3], "local": [4, 5]},
        "roi_cache_required": "c3_roi/cache/C3_ROI_V1/roi/{split}/{image_id}.png",
        "fallback_policy": "keep_all_rows_and_record_global_fallback",
        "folds": {},
    }
    mask_rows = read_rows(ROOT / "c3_roi/cache/C3_MASK_REUSE_CANDIDATE_V1/mask_manifest.csv")
    source_splits = {str(int(float(row["image_id"]))): row["split"] for row in mask_rows}
    if len(source_splits) != len(mask_rows):
        raise RuntimeError("Duplicate image IDs in mask manifest")
    for fold in FOLDS:
        source = ROOT / f"c1_curated/outputs/C1_AUDIT_OOF_V1"
        train = transform_rows(read_rows(source / f"fold_{fold}_train.csv"), "train", source_splits)
        val = transform_rows(read_rows(source / f"fold_{fold}_validation.csv"), "validation_official", source_splits)
        train_path = manifest_root / f"fold_{fold}_train.csv"
        val_path = manifest_root / f"fold_{fold}_validation.csv"
        write_rows(train_path, train)
        write_rows(val_path, val)
        mean, std = TARGETS[fold]
        config_path = config_root / f"fold_{fold}.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(render_config(fold, train, val, mean, std), encoding="utf-8")
        registry["folds"][str(fold)] = {
            "execution": "colab" if fold <= 3 else "local",
            "train_count": len(train),
            "validation_count": len(val),
            "train_hash": manifest_hash(train),
            "validation_hash": manifest_hash(val),
            "config": str(config_path.resolve()),
            "roi_status": "blocked_until_roi_cache_pass",
        }
    output = args.output_root / "registry.json"
    output.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(registry, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
