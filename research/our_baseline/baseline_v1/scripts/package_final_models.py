"""Build reproducible per-model packages following the group's model layout."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch


ROOT = Path("D:/do_an_tot_nghiep")
PROJECT = ROOT / "project"
OUT = PROJECT / "final_model_packages"
FRIEND = PROJECT / "friend_repo"
BASELINE = PROJECT / "baseline_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_friend_code(package: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted((FRIEND / "p1_baseline").glob("*.py")):
        destination = package / "code" / "p1_baseline" / source.name
        if copy_file(source, destination):
            copied.append(str(destination.relative_to(package)))
    requirements = FRIEND / "p1_baseline" / "requirements.txt"
    if copy_file(requirements, package / "code" / "p1_baseline" / "requirements.txt"):
        copied.append("code/p1_baseline/requirements.txt")
    inference_files = {
        FRIEND / "p8_test_ensemble" / "infer_ensemble.py": "infer_ensemble.py",
        FRIEND / "p9_inference" / "tta_bias_oof.py": "tta_bias_oof.py",
    }
    for source, name in inference_files.items():
        destination = package / "code" / "inference" / name
        if copy_file(source, destination):
            copied.append(str(destination.relative_to(package)))
    return copied


def copy_baseline_code(package: Path) -> list[str]:
    copied: list[str] = []
    for name in ("boneage_baseline.py", "requirements.txt"):
        source = BASELINE / name
        destination = package / "code" / name
        if copy_file(source, destination):
            copied.append(str(destination.relative_to(package)))
    return copied


def inspect_checkpoint(path: Path) -> dict:
    result = {
        "torch_load": "NOT_ATTEMPTED",
        "python_type": None,
        "top_level_keys": [],
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256(path),
        "state_dict_location": None,
        "tensor_count": None,
        "parameter_count": None,
    }
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        result["torch_load"] = "PASS"
        result["python_type"] = type(checkpoint).__name__
        if isinstance(checkpoint, dict):
            result["top_level_keys"] = sorted(str(key) for key in checkpoint.keys())
            state = None
            if isinstance(checkpoint.get("model"), dict):
                result["state_dict_location"] = "model"
                state = checkpoint["model"]
            elif all(hasattr(value, "numel") for value in checkpoint.values()):
                result["state_dict_location"] = "top_level"
                state = checkpoint
            if isinstance(state, dict):
                tensors = [value for value in state.values() if hasattr(value, "numel")]
                result["tensor_count"] = len(tensors)
                result["parameter_count"] = int(sum(int(value.numel()) for value in tensors))
    except Exception as exc:  # Keep package creation alive and record the failure.
        result["torch_load"] = "FAIL"
        result["load_error"] = f"{type(exc).__name__}: {exc}"
    return result


def copy_evidence(package: Path, sources: list[tuple[Path, str]]) -> list[str]:
    copied: list[str] = []
    for source, name in sources:
        destination = package / "evidence" / name
        if copy_file(source, destination):
            copied.append(str(destination.relative_to(package)))
    return copied


def create_package(
    *,
    package_id: str,
    technique: str,
    evaluation: str,
    status: str,
    mae: float | None,
    model_source: Path,
    config_sources: list[tuple[Path, str]],
    code_family: str,
    evidence_sources: list[tuple[Path, str]],
    interpretation: str,
) -> None:
    package = OUT / package_id
    package.mkdir(parents=True, exist_ok=True)
    model_name = "best_model.pt" if model_source.suffix.lower() == ".pt" else "best_mae.ckpt"
    model_destination = package / "model" / model_name
    copy_file(model_source, model_destination)

    included: list[str] = [str(model_destination.relative_to(package))]
    for source, name in config_sources:
        destination = package / "config" / name
        if copy_file(source, destination):
            included.append(str(destination.relative_to(package)))
    if code_family == "friend":
        included.extend(copy_friend_code(package))
    else:
        included.extend(copy_baseline_code(package))
    included.extend(copy_evidence(package, evidence_sources))

    inspection = inspect_checkpoint(model_source)
    manifest = {
        "package_id": package_id,
        "created_utc": "2026-08-25",
        "technique": technique,
        "evaluation": evaluation,
        "mae_months": mae,
        "status": status,
        "result_note": interpretation,
        "source_checkpoint": str(model_source),
        "checkpoint_inspection": inspection,
        "included_files": sorted(set(included)),
    }
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    included.append("manifest.json")

    readme = f"""# {package_id}

## Model

- Technique: {technique}
- Evaluation: {evaluation}
- MAE: **{mae if mae is not None else 'not recorded'} months**
- Status: `{status}`
- Original checkpoint: `{model_source}`
- Checkpoint SHA-256: `{inspection['checkpoint_sha256']}`
- Checkpoint bytes: {inspection['checkpoint_bytes']}
- PyTorch load audit: `{inspection['torch_load']}`
- State-dict location: `{inspection.get('state_dict_location')}`
- Tensor count: `{inspection.get('tensor_count')}`
- Parameter count: `{inspection.get('parameter_count')}`

## Interpretation

{interpretation}

## Package layout

- `model/`: unchanged final checkpoint.
- `config/`: configuration or data manifest used by the run.
- `code/`: matching model, data, training and inference implementation.
- `evidence/`: metrics, reports, manifests and audit files.
- `manifest.json`: package metadata and checkpoint inspection.

## Loading note

Load the checkpoint with the matching code and configuration. Start on CPU:

```python
checkpoint = torch.load("model/{model_name}", map_location="cpu", weights_only=False)
```

Do not assume `.pt` and `.ckpt` packages have the same top-level layout.

## Scientific-use note

- Keep preprocessing, image normalization, target normalization, sex handling and architecture identical to `config/`.
- Do not use the labeled 200-image test set to select checkpoints, hyperparameters or blend weights.
- Re-evaluate any modification on validation or OOF data first.
"""
    (package / "README_MODEL.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p7_report = FRIEND / "p7_final_v3" / "P7_OOF_report.json"
    exp006_oof = BASELINE / "outputs" / "results" / "exp006-p7-tta-results" / "oof" / "oof_report.json"
    exp006_tta = BASELINE / "outputs" / "exp006_p7_tta" / "exp006_p7_tta" / "results" / "tta_oof_report.json"
    p7_audit = FRIEND / "P7_5FOLD_AUDIT.txt"

    p7_root = PROJECT / "p7_results" / "results"
    for fold in range(1, 6):
        run = p7_root / f"P7_FINAL_V3_FOLD_{fold}"
        result_manifest = json.loads((run / "result_manifest.json").read_text(encoding="utf-8"))
        package_id = f"P7_E1_FOLD{fold}_CONVNEXT_TINY_SEX_A2"
        create_package(
            package_id=package_id,
            technique="ConvNeXt-Tiny 512 + pad_square + light augmentation + sex embedding + LDL/regression configuration",
            evaluation=f"P7 fold {fold} OOF validation",
            mae=float(result_manifest["best_mae"]),
            status="OFFICIAL_P7_FOLD_MODEL",
            model_source=run / "best_model.pt",
            config_sources=[(run / "config_resolved.yaml", "config_resolved.yaml")],
            code_family="friend",
            evidence_sources=[
                (run / "result_manifest.json", "result_manifest.json"),
                (run / "run_state.json", "run_state.json"),
                (run / "metrics.jsonl", "metrics.jsonl"),
                (run / "train.log", "train.log"),
                (run / "warnings.log", "warnings.log"),
                (run / "val_predictions_best.csv", "val_predictions_best.csv"),
                (p7_report, "P7_OOF_report.json"),
                (p7_audit, "P7_5FOLD_AUDIT.txt"),
            ],
            interpretation="One fold of the five-fold P7 ensemble. Use all five packages for equal-weight ensemble or TTA ensemble.",
        )

    exp006_root = BASELINE / "outputs" / "exp006_p7_tta" / "exp006_p7_tta" / "runs"
    exp006_report = json.loads(exp006_oof.read_text(encoding="utf-8"))
    for fold in range(1, 6):
        run = exp006_root / f"EXP006_P7_CONTROL_FOLD_{fold}"
        package_id = f"EXP006_P7_CONTROL_FOLD{fold}_CONVNEXT_TINY_SEX_LDL"
        create_package(
            package_id=package_id,
            technique="ConvNeXt-Tiny 512 + pad_square + light augmentation + sex embedding + LDL/regression control",
            evaluation=f"EXP006 pooled OOF, fold {fold} contribution",
            mae=float(exp006_report["folds"][str(fold)]["mae"]),
            status="EXP006_P7_CONTROL_FOLD_MODEL",
            model_source=run / "best_mae.ckpt",
            config_sources=[(run / "config_resolved.yaml", "config_resolved.yaml")],
            code_family="friend",
            evidence_sources=[
                (exp006_oof, "EXP006_P7_CONTROL_OOF_report.json"),
                (exp006_tta, "EXP006_P7_TTA_OOF_report.json"),
                (BASELINE / "EXP006_P7_CONTROL_OOF_AUDIT.md", "EXP006_P7_CONTROL_OOF_AUDIT.md"),
                (BASELINE / "EXP006_P7_TTA_DATASET_AUDIT.md", "EXP006_P7_TTA_DATASET_AUDIT.md"),
            ],
            interpretation="One of five fold checkpoints used for EXP006 pooled OOF MAE 6.323629 and P7-compatible TTA MAE 6.296203.",
        )

    exp004_run = BASELINE / "outputs" / "exp004_friend_holdout" / "checkpoint_mirror" / "checkpoint_mirror" / "EXP004_FRIEND_P7_FRESH_HOLDOUT_SEED42"
    exp004_state = json.loads((exp004_run / "run_state.json").read_text(encoding="utf-8"))
    create_package(
        package_id="EXP004_FRIEND_P7_FRESH_HOLDOUT_CONVNEXT_TINY_SEX",
        technique="ConvNeXt-Tiny 512 + pad_square + light augmentation + sex embedding + Smooth L1/LDL configuration",
        evaluation="Fresh 10% holdout from original training data",
        mae=float(exp004_state["best_mae"]),
        status="FRIEND_P7_FRESH_HOLDOUT_MODEL",
        model_source=exp004_run / "best_mae.ckpt",
        config_sources=[(BASELINE / "outputs" / "exp004_friend_holdout" / "friend_p7_fresh_holdout.toml", "friend_p7_fresh_holdout.toml")],
        code_family="friend",
        evidence_sources=[
            (BASELINE / "outputs" / "exp004_friend_holdout" / "fresh_holdout_audit.json", "fresh_holdout_audit.json"),
            (exp004_run / "run_state.json", "run_state.json"),
            (exp004_run / "metrics.jsonl", "metrics.jsonl"),
            (exp004_run / "train.log", "train.log"),
            (exp004_run / "warnings.log", "warnings.log"),
        ],
        interpretation="Best checkpoint from the fresh holdout reproduction run. It is not directly comparable to five-fold pooled OOF without matching protocol.",
    )

    official_root = BASELINE / "outputs"
    official_specs = [
        (
            "OFFICIAL_A2_LIGHT_FLIP_CONVNEXT_TINY_SEX",
            official_root / "official_a2_light_flip_seed42" / "official_a2_light_flip_seed42" / "official" / "best.pt",
            official_root / "official_a2_light_flip_seed42" / "official_a2_light_flip_seed42",
            "a2_light_flip recipe: ConvNeXt-Tiny 512 + sex feature + light flip augmentation",
        ),
        (
            "OFFICIAL_P7_REFERENCE_CONVNEXT_TINY_SEX",
            official_root / "official_gpu" / "official" / "best.pt",
            official_root / "official_gpu",
            "p7_reference recipe: ConvNeXt-Tiny 512 + sex feature + P7 reference preprocessing",
        ),
    ]
    for package_id, model_source, run_root, technique in official_specs:
        report = json.loads(next(run_root.rglob("report.json")).read_text(encoding="utf-8"))
        create_package(
            package_id=package_id,
            technique=technique,
            evaluation="Official train/validation split (1,425 validation images)",
            mae=float(report["best_mae"]),
            status="OFFICIAL_SINGLE_SPLIT_MODEL",
            model_source=model_source,
            config_sources=[(run_root / "data_manifest.json", "data_manifest.json")],
            code_family="baseline",
            evidence_sources=[
                (run_root / "official_report.json", "official_report.json"),
                (next(run_root.rglob("report.json")), "report.json"),
                (next(run_root.rglob("training_log.csv")), "training_log.csv"),
                (next(run_root.rglob("val_predictions.csv")), "val_predictions.csv"),
            ],
            interpretation="Official single-split baseline checkpoint. Use the recorded recipe and validation report when comparing against later experiments.",
        )

    print(f"Created packages in {OUT}")
    print(f"Package count: {len([p for p in OUT.iterdir() if p.is_dir()])}")


if __name__ == "__main__":
    main()
