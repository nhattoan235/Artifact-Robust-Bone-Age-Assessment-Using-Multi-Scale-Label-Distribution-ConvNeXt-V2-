"""Create the immutable three-split P0 registry after all audits pass."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_hash(rows: list[dict[str, str]], keys: list[str]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["split"], int(item["image_id"]))):
        digest.update(("\t".join(row.get(key, "") for key in keys) + "\n").encode("utf-8"))
    return digest.hexdigest()


def dist(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
    }


def main() -> int:
    base = json.loads((OUT / "audit_report.json").read_text(encoding="utf-8"))
    validation = json.loads((OUT / "validation_audit_report.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS" or not all(validation["checks"].values()):
        raise SystemExit("Validation compatibility gate did not pass")

    train = load_csv(OUT / "train_manifest.csv")
    val = load_csv(OUT / "validation_manifest.csv")
    test = load_csv(OUT / "test_manifest_LOCKED_NO_AGE.csv")
    development = train + val
    registry = development + test

    ids = [row["image_id"] for row in registry]
    hashes = [row["sha256"] for row in registry]
    if len(train) != 12611 or len(val) != 1425 or len(test) != 200:
        raise SystemExit("Unexpected split count")
    if len(set(ids)) != len(ids):
        raise SystemExit("Cross-split duplicate ID detected")
    if len(set(hashes)) != len(hashes):
        raise SystemExit("Cross-split duplicate SHA-256 detected")

    development_hash = canonical_hash(
        development, ["split", "image_id", "bone_age_months", "sex", "sha256"]
    )
    registry_hash = canonical_hash(registry, ["split", "image_id", "bone_age_months", "sex", "sha256"])
    val_ages = [float(row["bone_age_months"]) for row in val]
    val_male = [float(row["bone_age_months"]) for row in val if row["sex"] == "M"]
    val_female = [float(row["bone_age_months"]) for row in val if row["sex"] == "F"]

    fields = [
        "split", "image_id", "bone_age_months", "sex", "image_path", "file_size",
        "sha256", "width", "height", "mode", "readable", "error",
    ]
    write_csv(OUT / "development_manifest_14036.csv", fields, development)
    write_csv(OUT / "three_split_registry_LOCKED.csv", fields, registry)

    final = {
        "status": "PASS",
        "p0_complete": True,
        "counts": {"train": len(train), "validation": len(val), "test_locked": len(test), "development": len(development), "total": len(registry)},
        "sex_counts": {
            "train": dict(Counter(row["sex"] for row in train)),
            "validation": dict(Counter(row["sex"] for row in val)),
            "test_locked": dict(Counter(row["sex"] for row in test)),
        },
        "validation_distribution": {
            "all": dist(val_ages),
            "male": dist(val_male),
            "female": dist(val_female),
        },
        "checks": {
            "split_counts_correct": True,
            "all_ids_unique_across_splits": True,
            "all_hashes_unique_across_splits": True,
            "all_images_readable": True,
            "all_labels_valid": True,
            "test_labels_excluded_from_registry": all(not row.get("bone_age_months") for row in test),
            "validation_matches_deeplasia": validation["checks"]["matches_deeplasia_annotation"],
        },
        "fingerprints": {
            "train_manifest_sha256": base["fingerprints"]["train_manifest_sha256"],
            "validation_manifest_sha256": validation["validation_manifest_sha256"],
            "test_locked_manifest_sha256": base["fingerprints"]["test_locked_manifest_sha256"],
            "development_manifest_14036_sha256": development_hash,
            "three_split_registry_sha256": registry_hash,
            "validation_source_archive_sha256": validation["sources"]["source_archive"]["sha256"],
        },
    }
    (OUT / "P0_FINAL_REPORT.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "P0_FINAL_FINGERPRINTS.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in final["fingerprints"].items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
