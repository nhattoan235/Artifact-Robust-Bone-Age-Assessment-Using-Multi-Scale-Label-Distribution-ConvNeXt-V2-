"""Prepare the EXP009 five-fold LDL training package from the locked EXP006 split."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("D:/do_an_tot_nghiep")
BASELINE = ROOT / "project" / "baseline_v1"
SOURCE = BASELINE / "outputs" / "exp006_roadmap"
OUTPUT = BASELINE / "outputs" / "exp009_ldl_roadmap"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected text not found: {old}")
    return text.replace(old, new, 1)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for fold in range(1, 6):
        source_fold = SOURCE / f"fold_{fold}"
        output_fold = OUTPUT / f"fold_{fold}"
        output_fold.mkdir(parents=True, exist_ok=True)
        for name in ("train_manifest.csv", "validation_manifest.csv"):
            shutil.copy2(source_fold / name, output_fold / name)

        source_config = source_fold / "d3_ldl_fused.toml"
        text = source_config.read_text(encoding="utf-8")
        text = replace_once(text, f'run_id = "EXP008_D3_LDL_FUSED_FOLD_{fold}"', f'run_id = "EXP009_LDL_FUSED_FOLD_{fold}"')
        lines = text.splitlines()
        output_lines = [index for index, line in enumerate(lines) if line.startswith("output_root =")]
        if len(output_lines) != 1:
            raise ValueError("Expected exactly one output_root line")
        lines[output_lines[0]] = f'output_root = "{(OUTPUT / "runs").as_posix()}"'
        text = "\n".join(lines) + "\n"
        text = text.replace("periodic_keep = 2", "periodic_keep = 0")
        text = text.replace("best_keep = 3", "best_keep = 1")
        config_path = output_fold / "exp009_ldl_fused.toml"
        config_path.write_text(text, encoding="utf-8")
        train = output_fold / "train_manifest.csv"
        validation = output_fold / "validation_manifest.csv"
        records.append({
            "fold": fold,
            "run_id": f"EXP009_LDL_FUSED_FOLD_{fold}",
            "train_count": sum(1 for _ in train.open(encoding="utf-8")) - 1,
            "val_count": sum(1 for _ in validation.open(encoding="utf-8")) - 1,
            "train_manifest_sha256": sha256(train),
            "val_manifest_sha256": sha256(validation),
            "config": str(config_path),
        })

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "EXP009_LDL_FUSED_FIVE_FOLD",
        "test_labels_used": False,
        "source_split": str(SOURCE),
        "architecture": "convnext_tiny_ldl",
        "label_distribution_sigma": 2.0,
        "label_distribution_weight": 0.2,
        "regression_inference_weight": 0.5,
        "same_split_as": "EXP006_P7_CONTROL_FOLD_1..5",
        "selection_gate": "Pooled OOF MAE must beat EXP008 fixed 50/50 blend MAE 6.120448 before any test-200 comparison.",
        "folds": records,
    }
    (OUTPUT / "EXP009_LDL_ROADMAP_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUTPUT / "README_EXP009_LDL.md").write_text(
        """# EXP009 — LDL fused five-fold roadmap\n\n"
        "This package reuses the locked EXP006 five-fold split and changes only the architecture to `convnext_tiny_ldl`.\n\n"
        "The candidate must be evaluated with leakage-safe OOF predictions. The current gate is the fixed EXP008 P7 + EXP006-TTA blend at MAE 6.120448.\n\n"
        "No test CSV or test label was used during preparation.\n""",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
