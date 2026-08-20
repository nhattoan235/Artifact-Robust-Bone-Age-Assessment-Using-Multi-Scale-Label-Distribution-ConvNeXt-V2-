"""Compatibility gate for the official RSNA validation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audit_rsna import audit_image, canonical_manifest_hash, numeric_id, normalize_bool_sex, sha256_file, write_csv


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--official-csv", type=Path, required=True)
    parser.add_argument("--deeplasia-annotation", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    official = load_csv(args.official_csv)
    deeplasia = [row for row in load_csv(args.deeplasia_annotation) if row.get("dir") == "bone_age"]
    train = load_csv(args.train_manifest)
    test = load_csv(args.test_manifest)

    image_paths = sorted(args.validation_root.rglob("*.png"), key=lambda path: int(path.stem))
    print(f"Auditing {len(image_paths)} validation images...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        images = list(executor.map(audit_image, image_paths))

    image_by_id = {item.image_id: item for item in images}
    official_by_id = {numeric_id(row["Image ID"]): row for row in official}
    deeplasia_by_id = {numeric_id(row["image_ID"]): row for row in deeplasia}
    train_ids = {numeric_id(row["image_id"]) for row in train}
    test_ids = {numeric_id(row["image_id"]) for row in test}
    val_ids = set(official_by_id)
    image_ids = set(image_by_id)

    official_duplicate_ids = sorted(
        key for key, count in Counter(numeric_id(row["Image ID"]) for row in official).items() if count > 1
    )
    image_duplicate_ids = sorted(
        key for key, count in Counter(item.image_id for item in images).items() if count > 1
    )

    invalid_sex: list[str] = []
    invalid_age: list[str] = []
    deeplasia_missing: list[str] = []
    deeplasia_sex_mismatch: list[str] = []
    deeplasia_age_mismatch: list[str] = []
    manifest: list[dict] = []
    for image_id in sorted(val_ids, key=int):
        row = official_by_id[image_id]
        sex = normalize_bool_sex(row.get("male", ""))
        try:
            age = float(row.get("Bone Age (months)", ""))
        except ValueError:
            age = float("nan")
        if sex is None:
            invalid_sex.append(image_id)
        if not (age == age and 0 <= age <= 228):
            invalid_age.append(image_id)

        reference = deeplasia_by_id.get(image_id)
        if reference is None:
            deeplasia_missing.append(image_id)
        else:
            if sex != normalize_bool_sex(reference.get("sex", "")):
                deeplasia_sex_mismatch.append(image_id)
            try:
                reference_age = float(reference.get("bone_age", ""))
                if abs(age - reference_age) > 1e-6:
                    deeplasia_age_mismatch.append(image_id)
            except ValueError:
                deeplasia_age_mismatch.append(image_id)

        image = image_by_id.get(image_id)
        manifest.append(
            {
                "split": "validation_official",
                "image_id": image_id,
                "bone_age_months": "" if age != age else age,
                "sex": sex or "INVALID",
                "image_path": image.path if image else "",
                "file_size": image.file_size if image else "",
                "sha256": image.sha256 if image else "",
                "width": image.width if image else "",
                "height": image.height if image else "",
                "mode": image.mode if image else "",
                "readable": image.readable if image else False,
                "error": image.error if image else "MISSING_IMAGE",
            }
        )

    train_hashes = {row["sha256"] for row in train if row.get("sha256")}
    test_hashes = {row["sha256"] for row in test if row.get("sha256")}
    val_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for image in images:
        val_hash_to_ids[image.sha256].append(image.image_id)
    duplicate_hash_groups = {
        file_hash: ids for file_hash, ids in val_hash_to_ids.items() if file_hash and len(ids) > 1
    }
    cross_train_hashes = sorted(set(val_hash_to_ids) & train_hashes - {""})
    cross_test_hashes = sorted(set(val_hash_to_ids) & test_hashes - {""})

    manifest_hash = canonical_manifest_hash(
        manifest, ["split", "image_id", "bone_age_months", "sex", "sha256"]
    )
    checks = {
        "official_csv_rows_1425": len(official) == 1425,
        "png_count_1425": len(images) == 1425,
        "official_unique_ids_1425": len(val_ids) == 1425 and not official_duplicate_ids,
        "image_unique_ids_1425": len(image_ids) == 1425 and not image_duplicate_ids,
        "csv_ids_equal_image_ids": val_ids == image_ids,
        "no_train_id_overlap": not (val_ids & train_ids),
        "no_test_id_overlap": not (val_ids & test_ids),
        "all_images_readable": all(image.readable for image in images),
        "all_grayscale_png": all(image.mode == "L" and image.format == "PNG" for image in images),
        "sex_counts_match_expected": Counter(row["sex"] for row in manifest) == {"M": 773, "F": 652},
        "labels_valid": not invalid_sex and not invalid_age,
        "matches_deeplasia_annotation": not deeplasia_missing and not deeplasia_sex_mismatch and not deeplasia_age_mismatch,
        "no_duplicate_hash_in_validation": not duplicate_hash_groups,
        "no_hash_overlap_with_train": not cross_train_hashes,
        "no_hash_overlap_with_test": not cross_test_hashes,
    }

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "official_csv": len(official),
            "png": len(images),
            "male": sum(row["sex"] == "M" for row in manifest),
            "female": sum(row["sex"] == "F" for row in manifest),
        },
        "differences": {
            "csv_without_image": sorted(val_ids - image_ids, key=int),
            "image_without_csv": sorted(image_ids - val_ids, key=int),
            "train_id_overlap": sorted(val_ids & train_ids, key=int),
            "test_id_overlap": sorted(val_ids & test_ids, key=int),
            "unreadable_images": [image.image_id for image in images if not image.readable],
            "invalid_sex": invalid_sex,
            "invalid_age": invalid_age,
            "deeplasia_missing": deeplasia_missing,
            "deeplasia_sex_mismatch": deeplasia_sex_mismatch,
            "deeplasia_age_mismatch": deeplasia_age_mismatch,
            "duplicate_validation_hash_groups": duplicate_hash_groups,
            "hash_overlap_train": cross_train_hashes,
            "hash_overlap_test": cross_test_hashes,
        },
        "sources": {
            "source_archive": {"path": str(args.source_archive.resolve()), "sha256": sha256_file(args.source_archive)},
            "official_csv": {"path": str(args.official_csv.resolve()), "sha256": sha256_file(args.official_csv)},
            "deeplasia_annotation": {"path": str(args.deeplasia_annotation.resolve()), "sha256": sha256_file(args.deeplasia_annotation)},
        },
        "validation_manifest_sha256": manifest_hash,
    }

    write_csv(
        out / "validation_manifest.csv",
        [
            "split", "image_id", "bone_age_months", "sex", "image_path", "file_size",
            "sha256", "width", "height", "mode", "readable", "error",
        ],
        manifest,
    )
    (out / "validation_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "validation_fingerprint.txt").write_text(
        f"validation_manifest_sha256={manifest_hash}\n"
        f"source_archive_sha256={report['sources']['source_archive']['sha256']}\n"
        f"official_validation_csv_sha256={report['sources']['official_csv']['sha256']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks, "counts": report["counts"], "validation_manifest_sha256": manifest_hash}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
