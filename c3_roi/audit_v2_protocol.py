"""Read-only audit of the frozen C3-Z26 C3-ROI V2 fold package.

This checks protocol metadata; it does not load images or claim to exclude
patient-level or visually near-duplicate leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import tomllib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "c3_roi/outputs/C3_Z26_C3_ROI_T4_CODE_V2/C3_Z26_C3_ROI_V2"
DEFAULT_DATASET = ROOT / "c3_roi/outputs/C3_Z26_COMBO_V2_FINAL_BUILD/C3_Z26_COMBO_V2"
DEFAULT_RESCUE = ROOT / "c3_roi/outputs/C3_Z26_RESCUE_V2_SAFE_CANDIDATES/rescue_audit.csv"
DEFAULT_TEST = ROOT / "p0_audit/outputs/test_manifest_LOCKED_NO_AGE.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_hash(rows: list[dict[str, str]]) -> str:
    """Match the fingerprint used by p1_baseline.data without importing torch."""
    keys = ["split", "image_id", "bone_age_months", "sex", "sha256"]
    keys += [
        key for key in ("age_bin", "sex_age_stratum", "sample_weight", "audit_status", "audit_reason")
        if any(key in row for row in rows)
    ]
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(int(float(item["image_id"])))):
        digest.update(("\t".join(row.get(key, "") for key in keys) + "\n").encode())
    return digest.hexdigest()


def audit(package: Path, dataset: Path, rescue_path: Path, test_path: Path) -> dict:
    folds = {}
    validation_rows: list[dict[str, str]] = []
    all_validation_ids: set[str] = set()
    target_normalization_warnings = []
    for fold in range(1, 6):
        train = read_csv(package / "manifests" / f"fold_{fold}_train.csv")
        val = read_csv(package / "manifests" / f"fold_{fold}_validation.csv")
        with (package / "configs_t4_b36" / f"fold_{fold}.toml").open("rb") as handle:
            cfg = tomllib.load(handle)
        train_ids = {row["image_id"] for row in train}
        val_ids = {row["image_id"] for row in val}
        train_hashes = {row["sha256"] for row in train}
        val_hashes = {row["sha256"] for row in val}
        targets = [float(row["bone_age_months"]) for row in train]
        actual_mean = statistics.mean(targets)
        actual_std = statistics.stdev(targets)
        cfg_mean = float(cfg["model"]["target_mean"])
        cfg_std = float(cfg["model"]["target_std"])
        stats_match = abs(cfg_mean - actual_mean) < 1e-8 and abs(cfg_std - actual_std) < 1e-8
        if not stats_match:
            target_normalization_warnings.append(fold)
        folds[str(fold)] = {
            "train_count": len(train),
            "validation_count": len(val),
            "within_fold_id_overlap": len(train_ids & val_ids),
            "within_fold_sha256_overlap": len(train_hashes & val_hashes),
            "validation_id_seen_in_other_fold": len(val_ids & all_validation_ids),
            "expected_counts_match": len(train) == cfg["data"]["expected_train_count"]
            and len(val) == cfg["data"]["expected_val_count"],
            "manifest_hashes_match": manifest_hash(train) == cfg["data"]["expected_train_hash"]
            and manifest_hash(val) == cfg["data"]["expected_val_hash"],
            "train_target_mean": actual_mean,
            "train_target_std_sample": actual_std,
            "configured_target_mean": cfg_mean,
            "configured_target_std": cfg_std,
            "target_normalization_matches_own_train": stats_match,
        }
        all_validation_ids.update(val_ids)
        validation_rows.extend(val)

    validation_by_id = {row["image_id"]: row for row in validation_rows}
    handoff = read_csv(dataset / "handoff_manifest.csv")
    handoff_by_id = {row["image_id"]: row for row in handoff if row["split"] != "test"}
    rescue = read_csv(rescue_path)
    rescued_ids = {row["image_id"] for row in rescue if row["rescue_status"] == "candidate_rescued"}
    test_rows = read_csv(test_path)
    test_ids = {row["image_id"] for row in test_rows}
    age_counts = Counter(
        "0-59" if float(row["bone_age_months"]) < 60 else
        "60-119" if float(row["bone_age_months"]) < 120 else
        "120-179" if float(row["bone_age_months"]) < 180 else "180-228"
        for row in validation_rows
    )
    recorded_fallback = {
        row["image_id"] for row in validation_rows if row["roi_mode"] == "global_fallback"
    }
    return {
        "scope": "local manifest/config/handoff metadata only; no image decoding or patient identity check",
        "folds": folds,
        "oof_count": len(validation_rows),
        "oof_unique_ids": len(validation_by_id),
        "oof_unique_sha256": len({row["sha256"] for row in validation_rows}),
        "age_bin_counts": dict(sorted(age_counts.items())),
        "sex_counts": dict(Counter(row["sex"] for row in validation_rows)),
        "test_id_overlap_with_oof": len(test_ids & set(validation_by_id)),
        "test_manifest_has_age_labels": any("bone_age_months" in row or "target_months" in row for row in test_rows),
        "oof_handoff_sha256_match_count": sum(
            row["sha256"] == handoff_by_id.get(image_id, {}).get("output_sha256")
            for image_id, row in validation_by_id.items()
        ),
        "recorded_global_fallback_count": len(recorded_fallback),
        "rescued_but_still_recorded_global_fallback_count": len(recorded_fallback & rescued_ids),
        "effective_global_fallback_count_after_rescue": len(recorded_fallback - rescued_ids),
        "target_normalization_fold_mismatch": target_normalization_warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rescue", type=Path, default=DEFAULT_RESCUE)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.package, args.dataset, args.rescue, args.test_manifest)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
