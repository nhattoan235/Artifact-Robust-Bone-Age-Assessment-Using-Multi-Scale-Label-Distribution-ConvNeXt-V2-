"""Read-only P0 audit for the local RSNA Pediatric Bone Age dataset.

This script never edits source data and never emits test bone-age labels. It
creates reproducible manifests and integrity summaries inside the workspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ImageAudit:
    image_id: str
    path: str
    file_size: int
    sha256: str
    width: int | None
    height: int | None
    mode: str | None
    format: str | None
    readable: bool
    error: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_image(path: Path) -> ImageAudit:
    image_id = path.stem
    try:
        file_hash = sha256_file(path)
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
            image.verify()
        return ImageAudit(
            image_id=image_id,
            path=str(path.resolve()),
            file_size=path.stat().st_size,
            sha256=file_hash,
            width=width,
            height=height,
            mode=mode,
            format=image_format,
            readable=True,
            error="",
        )
    except Exception as exc:  # audit must record every failure, not hide it
        return ImageAudit(
            image_id=image_id,
            path=str(path.resolve()),
            file_size=path.stat().st_size if path.exists() else -1,
            sha256="",
            width=None,
            height=None,
            mode=None,
            format=None,
            readable=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_bool_sex(value: str) -> str | None:
    value = str(value).strip().lower()
    if value in {"true", "1", "m", "male"}:
        return "M"
    if value in {"false", "0", "f", "female"}:
        return "F"
    return None


def numeric_id(value: str) -> str:
    value = str(value).strip()
    if value.lower().endswith(".png"):
        value = value[:-4]
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def duplicate_groups(records: list[ImageAudit]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.sha256:
            groups[record.sha256].append(record.image_id)
    return {key: sorted(values) for key, values in groups.items() if len(values) > 1}


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_manifest_hash(rows: list[dict[str, Any]], keys: list[str]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: numeric_id(str(item["image_id"]))):
        line = "\t".join(str(row.get(key, "")) for key in keys) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--locked-test-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = root / "boneage-training-dataset.csv"
    test_meta_csv = root / "boneage-test-dataset.csv"
    train_dir = root / "boneage-training-dataset" / "boneage-training-dataset"
    test_dir = root / "boneage-test-dataset" / "boneage-test-dataset"

    required = [train_csv, test_meta_csv, train_dir, test_dir, args.locked_test_labels]
    missing_sources = [str(path) for path in required if not path.exists()]
    if missing_sources:
        print(json.dumps({"fatal": "missing sources", "paths": missing_sources}, indent=2))
        return 2

    train_rows = load_csv(train_csv)
    test_meta_rows = load_csv(test_meta_csv)
    locked_test_rows = load_csv(args.locked_test_labels)

    train_paths = sorted(train_dir.glob("*.png"), key=lambda path: int(path.stem))
    test_paths = sorted(test_dir.glob("*.png"), key=lambda path: int(path.stem))

    print(f"Auditing {len(train_paths)} train and {len(test_paths)} test PNG files...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        train_images = list(executor.map(audit_image, train_paths))
        test_images = list(executor.map(audit_image, test_paths))

    train_image_by_id = {record.image_id: record for record in train_images}
    test_image_by_id = {record.image_id: record for record in test_images}

    train_label_ids = [numeric_id(row.get("id", "")) for row in train_rows]
    train_image_ids = [record.image_id for record in train_images]
    test_meta_ids = [numeric_id(row.get("Case ID", "")) for row in test_meta_rows]
    test_image_ids = [record.image_id for record in test_images]
    locked_test_ids = [numeric_id(row.get("image_ID", "")) for row in locked_test_rows]

    train_manifest: list[dict[str, Any]] = []
    train_ages: list[float] = []
    train_age_by_sex: dict[str, list[float]] = {"M": [], "F": []}
    invalid_train_sex: list[str] = []
    invalid_train_age: list[str] = []
    for row in train_rows:
        image_id = numeric_id(row.get("id", ""))
        sex = normalize_bool_sex(row.get("male", ""))
        try:
            age = float(row.get("boneage", ""))
        except ValueError:
            age = float("nan")
        if sex is None:
            invalid_train_sex.append(image_id)
        if not (age == age and 0 <= age <= 228):
            invalid_train_age.append(image_id)
        else:
            train_ages.append(age)
            if sex in train_age_by_sex:
                train_age_by_sex[sex].append(age)
        image = train_image_by_id.get(image_id)
        train_manifest.append(
            {
                "split": "train",
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

    test_meta_by_id = {numeric_id(row.get("Case ID", "")): row for row in test_meta_rows}
    test_manifest: list[dict[str, Any]] = []
    invalid_test_sex: list[str] = []
    for image_id in sorted(test_meta_by_id, key=int):
        row = test_meta_by_id[image_id]
        sex = normalize_bool_sex(row.get("Sex", ""))
        if sex is None:
            invalid_test_sex.append(image_id)
        image = test_image_by_id.get(image_id)
        test_manifest.append(
            {
                "split": "test_locked",
                "image_id": image_id,
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

    locked_by_id = {numeric_id(row.get("image_ID", "")): row for row in locked_test_rows}
    locked_sex_mismatches: list[str] = []
    locked_invalid_ages: list[str] = []
    for image_id, metadata in test_meta_by_id.items():
        locked = locked_by_id.get(image_id)
        if locked is None:
            continue
        if normalize_bool_sex(metadata.get("Sex", "")) != normalize_bool_sex(locked.get("sex", "")):
            locked_sex_mismatches.append(image_id)
        try:
            age = float(locked.get("bone_age", ""))
            if not 0 <= age <= 228:
                locked_invalid_ages.append(image_id)
        except ValueError:
            locked_invalid_ages.append(image_id)

    train_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for record in train_images:
        train_hash_to_ids[record.sha256].append(record.image_id)
    test_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for record in test_images:
        test_hash_to_ids[record.sha256].append(record.image_id)
    cross_split_hashes = sorted(set(train_hash_to_ids) & set(test_hash_to_ids) - {""})

    train_manifest_hash = canonical_manifest_hash(
        train_manifest, ["split", "image_id", "bone_age_months", "sex", "sha256"]
    )
    test_manifest_hash = canonical_manifest_hash(test_manifest, ["split", "image_id", "sex", "sha256"])

    report: dict[str, Any] = {
        "audit_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "pillow": Image.__version__,
            "workers": args.workers,
        },
        "sources": {
            "train_csv": {"path": str(train_csv), "sha256": sha256_file(train_csv)},
            "test_metadata_csv": {"path": str(test_meta_csv), "sha256": sha256_file(test_meta_csv)},
            "test_labels_locked": {
                "path": str(args.locked_test_labels.resolve()),
                "sha256": sha256_file(args.locked_test_labels),
                "labels_emitted": False,
            },
            "validation": {"present": False, "expected_count": 1425},
        },
        "counts": {
            "train_csv_rows": len(train_rows),
            "train_png": len(train_images),
            "test_metadata_rows": len(test_meta_rows),
            "test_png": len(test_images),
            "locked_test_label_rows": len(locked_test_rows),
        },
        "id_integrity": {
            "duplicate_train_csv_ids": sorted(
                key for key, count in Counter(train_label_ids).items() if count > 1
            ),
            "duplicate_train_image_ids": sorted(
                key for key, count in Counter(train_image_ids).items() if count > 1
            ),
            "train_csv_without_image": sorted(set(train_label_ids) - set(train_image_ids), key=numeric_id),
            "train_image_without_csv": sorted(set(train_image_ids) - set(train_label_ids), key=numeric_id),
            "duplicate_test_metadata_ids": sorted(
                key for key, count in Counter(test_meta_ids).items() if count > 1
            ),
            "test_metadata_without_image": sorted(set(test_meta_ids) - set(test_image_ids), key=numeric_id),
            "test_image_without_metadata": sorted(set(test_image_ids) - set(test_meta_ids), key=numeric_id),
            "train_test_id_overlap": sorted(set(train_image_ids) & set(test_image_ids), key=numeric_id),
        },
        "label_integrity": {
            "invalid_train_sex_ids": invalid_train_sex,
            "invalid_train_age_ids": invalid_train_age,
            "invalid_test_metadata_sex_ids": invalid_test_sex,
            "locked_test_ids_match_metadata": set(locked_test_ids) == set(test_meta_ids),
            "locked_test_missing_ids": sorted(set(test_meta_ids) - set(locked_test_ids), key=numeric_id),
            "locked_test_extra_ids": sorted(set(locked_test_ids) - set(test_meta_ids), key=numeric_id),
            "locked_test_sex_mismatch_ids": locked_sex_mismatches,
            "locked_test_invalid_age_ids": locked_invalid_ages,
            "test_age_distribution_inspected_or_emitted": False,
        },
        "train_distribution": {
            "sex_counts": Counter(row["sex"] for row in train_manifest),
            "age_all": distribution(train_ages),
            "age_male": distribution(train_age_by_sex["M"]),
            "age_female": distribution(train_age_by_sex["F"]),
        },
        "image_integrity": {
            "unreadable_train": [asdict(record) for record in train_images if not record.readable],
            "unreadable_test": [asdict(record) for record in test_images if not record.readable],
            "train_modes": Counter(record.mode for record in train_images),
            "test_modes": Counter(record.mode for record in test_images),
            "train_formats": Counter(record.format for record in train_images),
            "test_formats": Counter(record.format for record in test_images),
            "train_width": distribution([float(record.width) for record in train_images if record.width]),
            "train_height": distribution([float(record.height) for record in train_images if record.height]),
            "test_width": distribution([float(record.width) for record in test_images if record.width]),
            "test_height": distribution([float(record.height) for record in test_images if record.height]),
            "duplicate_hash_groups_train": duplicate_groups(train_images),
            "duplicate_hash_groups_test": duplicate_groups(test_images),
            "cross_split_duplicate_hashes": [
                {
                    "sha256": file_hash,
                    "train_ids": train_hash_to_ids[file_hash],
                    "test_ids": test_hash_to_ids[file_hash],
                }
                for file_hash in cross_split_hashes
            ],
        },
        "fingerprints": {
            "train_manifest_sha256": train_manifest_hash,
            "test_locked_manifest_sha256": test_manifest_hash,
        },
    }

    write_csv(
        output_dir / "train_manifest.csv",
        [
            "split", "image_id", "bone_age_months", "sex", "image_path", "file_size",
            "sha256", "width", "height", "mode", "readable", "error",
        ],
        train_manifest,
    )
    write_csv(
        output_dir / "test_manifest_LOCKED_NO_AGE.csv",
        [
            "split", "image_id", "sex", "image_path", "file_size", "sha256",
            "width", "height", "mode", "readable", "error",
        ],
        test_manifest,
    )
    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=dict)
    with (output_dir / "fingerprints.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"train_manifest_sha256={train_manifest_hash}\n")
        handle.write(f"test_locked_manifest_sha256={test_manifest_hash}\n")
        handle.write(f"train_csv_sha256={report['sources']['train_csv']['sha256']}\n")
        handle.write(f"test_metadata_csv_sha256={report['sources']['test_metadata_csv']['sha256']}\n")
        handle.write(f"test_labels_locked_sha256={report['sources']['test_labels_locked']['sha256']}\n")

    concise = {
        "counts": report["counts"],
        "missing_validation": not report["sources"]["validation"]["present"],
        "unreadable_train": len(report["image_integrity"]["unreadable_train"]),
        "unreadable_test": len(report["image_integrity"]["unreadable_test"]),
        "train_duplicate_hash_groups": len(report["image_integrity"]["duplicate_hash_groups_train"]),
        "test_duplicate_hash_groups": len(report["image_integrity"]["duplicate_hash_groups_test"]),
        "cross_split_duplicate_hashes": len(report["image_integrity"]["cross_split_duplicate_hashes"]),
        "fingerprints": report["fingerprints"],
    }
    print(json.dumps(concise, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
