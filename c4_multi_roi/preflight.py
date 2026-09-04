from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from .cache_utils import sha256_file
from .config import C4Config, load_config, scientific_config_hash
from .data import load_manifest
from .schema import ROI_NAMES


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def preflight_report(config: C4Config, *, check_all_files: bool) -> dict:
    manifest = Path(config.manifest)
    errors: list[str] = []
    if not manifest.is_file():
        return {"status": "FAIL", "errors": [f"missing manifest: {manifest}"], "checks": {}}
    rows = load_manifest(manifest)
    if len(rows) != config.expected_rows:
        errors.append(f"manifest rows {len(rows)} != {config.expected_rows}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate source image_id")
    fold_counts = Counter(row["fold"] for row in rows)
    if config.expected_rows >= 5 and set(fold_counts) != {"1", "2", "3", "4", "5"}:
        errors.append(f"fold coverage invalid: {dict(fold_counts)}")
    manifest_hash = sha256_file(manifest)
    if config.expected_manifest_sha256 and manifest_hash != config.expected_manifest_sha256:
        errors.append("manifest SHA-256 mismatch")
    root = Path(config.image_root)
    sample = rows if check_all_files else rows[: min(200, len(rows))]
    global_files = 0
    roi_files = 0
    unreadable: list[str] = []
    for row in sample:
        paths: list[Path] = []
        if config.view_mode in {"global_only", "global_plus_six"}:
            paths.append(_resolve(root, row["global_path"]))
            global_files += 1
        if config.view_mode in {"six_roi_only", "global_plus_six"}:
            for name in ROI_NAMES:
                key = f"roi_{name}_path"
                if not row.get(key):
                    errors.append(f"ID={row['image_id']} missing {key}")
                    continue
                paths.append(_resolve(root, row[key]))
                roi_files += 1
        for path in paths:
            if not path.is_file():
                unreadable.append(str(path))
                continue
            try:
                with Image.open(path) as handle:
                    handle.verify()
            except Exception:
                unreadable.append(str(path))
    if unreadable:
        errors.append(f"missing/unreadable view files: {len(unreadable)}; first={unreadable[:3]}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": {
            "manifest_rows": len(rows),
            "unique_ids": len(set(ids)),
            "fold_counts": dict(sorted(fold_counts.items())),
            "checked_rows": len(sample),
            "global_files": global_files,
            "roi_files": roi_files,
            "manifest_sha256": manifest_hash,
            "config_hash": scientific_config_hash(config),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    report = preflight_report(config, check_all_files=not args.quick)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
