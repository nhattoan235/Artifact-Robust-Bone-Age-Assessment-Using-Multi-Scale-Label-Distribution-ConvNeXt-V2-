from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from .cache_utils import sha256_file
from .schema import ROI_NAMES


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def audit_rows(rows: list[dict[str, str]], *, check_files: bool) -> dict:
    counts = Counter(str(row.get("image_id", "")) for row in rows)
    duplicates = sorted(image_id for image_id, count in counts.items() if image_id and count > 1)
    missing_fields: list[str] = []
    missing_files: list[str] = []
    unreadable_files: list[str] = []
    hash_mismatch: list[str] = []
    paths: list[str] = []
    for row in rows:
        image_id = str(row.get("image_id", ""))
        for name in ROI_NAMES:
            path_key = f"roi_{name}_path"
            sha_key = f"roi_{name}_sha256"
            bbox_key = f"roi_{name}_bbox"
            if any(not row.get(key) for key in (path_key, sha_key, bbox_key)):
                missing_fields.append(f"{image_id}:{name}")
                continue
            paths.append(row[path_key])
            if check_files:
                path = _resolve(row[path_key])
                if not path.is_file():
                    missing_files.append(str(path))
                    continue
                try:
                    with Image.open(path) as handle:
                        handle.verify()
                except Exception:
                    unreadable_files.append(str(path))
                    continue
                if sha256_file(path) != row[sha_key]:
                    hash_mismatch.append(str(path))
    duplicate_paths = sorted(path for path, count in Counter(paths).items() if count > 1)
    passed = not any((duplicates, missing_fields, missing_files, unreadable_files, hash_mismatch, duplicate_paths))
    return {
        "pass": passed,
        "rows": len(rows),
        "unique_ids": len(counts),
        "roi_records": len(rows) * len(ROI_NAMES),
        "duplicate_ids": duplicates,
        "duplicate_roi_paths": duplicate_paths,
        "missing_fields": missing_fields,
        "missing_files": missing_files,
        "unreadable_files": unreadable_files,
        "hash_mismatch": hash_mismatch,
        "geometry_quality_counts": dict(Counter(row.get("geometry_quality", "") for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "c4_multi_roi/cache/C4_MULTI_ROI_V1/manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "c4_multi_roi/outputs/cache_audit_v1.json")
    parser.add_argument("--expected-rows", type=int, default=14036)
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = audit_rows(rows, check_files=not args.skip_file_check)
    report["expected_rows"] = args.expected_rows
    if len(rows) != args.expected_rows:
        report["pass"] = False
        report["row_count_error"] = f"{len(rows)} != {args.expected_rows}"
    report["manifest_sha256"] = sha256_file(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
