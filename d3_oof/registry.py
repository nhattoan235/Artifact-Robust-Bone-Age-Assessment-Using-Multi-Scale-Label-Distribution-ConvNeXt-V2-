from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from p1_baseline.data import load_manifest, manifest_hash


FOLD_COUNT = 5
EXPECTED_VALIDATION_COUNTS = {1: 2808, 2: 2807, 3: 2807, 4: 2807, 5: 2807}


def _prediction_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {str(int(float(row["image_id"]))) for row in rows}
    if len(ids) != len(rows):
        raise RuntimeError(f"Duplicate prediction IDs: {path}")
    return ids


def _yaml_number(path: Path, key: str) -> float:
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Missing {key} in {path}")


def verify_reused_splits(workspace: Path) -> dict:
    workspace = workspace.resolve()
    folds: dict[str, dict] = {}
    pooled: set[str] = set()
    for fold in range(1, FOLD_COUNT + 1):
        train_path = workspace / f"c1_curated/outputs/C1_AUDIT_OOF_V1/fold_{fold}_train.csv"
        validation_path = workspace / f"c1_curated/outputs/C1_AUDIT_OOF_V1/fold_{fold}_validation.csv"
        prediction_path = workspace / f"p7_results/results/P7_FINAL_V3_FOLD_{fold}/val_predictions_best.csv"
        train_rows = load_manifest(train_path, "train")
        validation_rows = load_manifest(validation_path, "validation_official")
        manifest_ids = {str(int(float(row["image_id"]))) for row in validation_rows}
        p7_ids = _prediction_ids(prediction_path)
        expected_count = EXPECTED_VALIDATION_COUNTS[fold]
        if len(validation_rows) != expected_count or len(manifest_ids) != expected_count:
            raise RuntimeError(f"Unexpected validation count for fold {fold}")
        if pooled.intersection(manifest_ids):
            raise RuntimeError(f"Validation overlap detected at fold {fold}")
        pooled.update(manifest_ids)
        folds[str(fold)] = {
            "train_manifest": str(train_path),
            "validation_manifest": str(validation_path),
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "train_hash": manifest_hash(train_rows),
            "validation_hash": manifest_hash(validation_rows),
            "same_validation_ids": manifest_ids == p7_ids,
        }
    if len(pooled) != 14036:
        raise RuntimeError(f"Expected 14,036 pooled validation IDs, got {len(pooled)}")
    if not all(item["same_validation_ids"] for item in folds.values()):
        raise RuntimeError("Reused manifests do not match P7 validation IDs")
    return {"folds": folds, "pooled_unique_validation_ids": len(pooled)}


def _render_config(
    fold: int, fold_report: dict, target_mean: float, target_std: float,
) -> str:
    return f'''[data]
train_manifest = "{Path(fold_report["train_manifest"]).as_posix()}"
val_manifest = "{Path(fold_report["validation_manifest"]).as_posix()}"
expected_train_hash = "{fold_report["train_hash"]}"
expected_val_hash = "{fold_report["validation_hash"]}"
expected_train_count = {fold_report["train_count"]}
expected_val_count = {fold_report["validation_count"]}

[run]
run_id = "D3_OOF_V1_FOLD_{fold}"
output_root = "d3_oof/runs/D3_OOF_V1"
seed = 42
device = "auto"
deterministic = true
allow_tf32 = true

[model]
architecture = "convnext_tiny_ldl"
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

[label_distribution]
age_class_count = 229
label_distribution_sigma = 2.0
label_distribution_weight = 0.2
regression_inference_weight = 0.5

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


def build_registry(workspace: Path, output_dir: Path) -> dict:
    workspace = workspace.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_audit = verify_reused_splits(workspace)
    report = {
        "status": "PASS",
        "protocol": "D3 label-distribution OOF on locked P7 fold assignments",
        "test_used": False,
        "pooled_unique_validation_ids": split_audit["pooled_unique_validation_ids"],
        "folds": {},
    }
    for fold in range(1, FOLD_COUNT + 1):
        p7_config = workspace / f"p7_results/results/P7_FINAL_V3_FOLD_{fold}/config_resolved.yaml"
        target_mean = _yaml_number(p7_config, "target_mean")
        target_std = _yaml_number(p7_config, "target_std")
        item = dict(split_audit["folds"][str(fold)])
        config_path = output_dir / f"fold_{fold}.toml"
        config_path.write_text(
            _render_config(fold, item, target_mean, target_std), encoding="utf-8"
        )
        item.update({
            "target_mean": target_mean,
            "target_std": target_std,
            "config": str(config_path.resolve()),
        })
        report["folds"][str(fold)] = item
    (output_dir / "registry.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_registry(args.workspace, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
