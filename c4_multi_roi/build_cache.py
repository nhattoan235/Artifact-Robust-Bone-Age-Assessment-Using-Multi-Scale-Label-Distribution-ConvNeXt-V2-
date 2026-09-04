from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .cache_utils import crop_pad_resize, sha256_file
from .geometry import analyze_six_rois
from .schema import ROI_NAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK_MANIFEST = ROOT / "c3_roi/cache/C3_MASK_REUSE_CANDIDATE_V1/mask_manifest.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "c4_multi_roi/cache/C4_MULTI_ROI_V1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_relative_path(image_id: str, split: str) -> Path:
    normalized = str(split).strip().lower()
    if normalized == "train":
        return Path("data/goc/boneage-training-dataset/boneage-training-dataset") / f"{image_id}.png"
    if normalized == "validation_official":
        return Path("data/rsna_official_validation/images") / f"{image_id}.png"
    raise ValueError(f"Unsupported development split: {split}")


def _rank(row: dict[str, str], seed: int) -> str:
    return hashlib.sha256(f"C4-PILOT-{seed}-{row['image_id']}".encode()).hexdigest()


def select_pilot_rows(rows: list[dict[str, str]], *, count: int, seed: int) -> list[dict[str, str]]:
    if count <= 0 or count > len(rows):
        raise ValueError("pilot count must be within available rows")
    fallback_target = min(round(count * 0.20), sum(row.get("fallback_reason") != "ok" for row in rows))
    fallback = sorted((row for row in rows if row.get("fallback_reason") != "ok"), key=lambda row: _rank(row, seed))
    normal = sorted((row for row in rows if row.get("fallback_reason") == "ok"), key=lambda row: _rank(row, seed))
    selected = fallback[:fallback_target] + normal[: count - fallback_target]
    return sorted(selected, key=lambda row: int(float(row["image_id"])))


def load_fold_metadata(manifest_root: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for fold in range(1, 6):
        for row in read_csv(manifest_root / f"fold_{fold}_validation.csv"):
            image_id = str(int(float(row["image_id"])))
            if image_id in metadata:
                raise RuntimeError(f"image_id appears in multiple validation folds: {image_id}")
            metadata[image_id] = {
                "fold": str(fold),
                "bone_age_months": row["bone_age_months"],
                "sex": row["sex"],
                "age_bin": row.get("age_bin", ""),
            }
    return metadata


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _write_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, optimize=True, progressive=True)


def build_rows(
    source_rows: list[dict[str, str]],
    *,
    output_root: Path,
    image_size: int,
    jpeg_quality: int,
    metadata: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for index, source_row in enumerate(source_rows, start=1):
        image_id = str(int(float(source_row["image_id"])))
        if image_id not in metadata:
            raise RuntimeError(f"image_id missing from locked fold metadata: {image_id}")
        source_path = _resolve(source_row["source_image"])
        mask_path = _resolve(source_row["mask_path"])
        if not source_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(source_path if not source_path.is_file() else mask_path)
        with Image.open(source_path) as handle:
            source_image = handle.convert("L")
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L"))
        geometry = analyze_six_rois(mask, width=source_image.width, height=source_image.height)
        row = {
            "image_id": image_id,
            "source_split": source_row["split"],
            "fold": metadata[image_id]["fold"],
            "bone_age_months": metadata[image_id]["bone_age_months"],
            "sex": metadata[image_id]["sex"],
            "age_bin": metadata[image_id]["age_bin"],
            "global_path": source_relative_path(image_id, source_row["split"]).as_posix(),
            "source_sha256": source_row["source_image_sha256"],
            "mask_sha256": source_row["mask_sha256"],
            "geometry_quality": geometry.quality_flag,
            "angle_degrees": f"{geometry.angle_degrees:.6f}",
            "thumb_side": geometry.thumb_side,
            "hand_bbox": ",".join(map(str, geometry.hand_bbox)),
        }
        for name in ROI_NAMES:
            relative = Path("c4_multi_roi/cache/C4_MULTI_ROI_V1/roi") / name / source_row["split"] / f"{image_id}.jpg"
            destination = ROOT / relative if output_root == DEFAULT_OUTPUT_ROOT else output_root / "roi" / name / source_row["split"] / f"{image_id}.jpg"
            roi_image = crop_pad_resize(source_image, geometry.boxes[name], size=image_size)
            _write_jpeg(roi_image, destination, jpeg_quality)
            row[f"roi_{name}_path"] = relative.as_posix() if output_root == DEFAULT_OUTPUT_ROOT else destination.as_posix()
            row[f"roi_{name}_bbox"] = ",".join(map(str, geometry.boxes[name]))
            row[f"roi_{name}_sha256"] = sha256_file(destination)
            row[f"roi_{name}_quality"] = geometry.quality_flag
        output.append(row)
        if index % 100 == 0 or index == len(source_rows):
            print(f"processed={index}/{len(source_rows)}", flush=True)
    return output


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-manifest", type=Path, default=DEFAULT_MASK_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifest.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    rows = read_csv(args.mask_manifest)
    if args.pilot is not None:
        rows = select_pilot_rows(rows, count=args.pilot, seed=args.seed)
    elif args.limit is not None:
        rows = rows[: args.limit]
    metadata = load_fold_metadata(ROOT / "c3_roi/manifests")
    built = build_rows(
        rows,
        output_root=args.output_root,
        image_size=args.image_size,
        jpeg_quality=args.jpeg_quality,
        metadata=metadata,
    )
    write_rows(args.manifest, built)
    summary = {
        "rows": len(built),
        "roi_files": len(built) * len(ROI_NAMES),
        "fallback_rows": sum(row["geometry_quality"] != "ok" for row in built),
        "manifest": str(args.manifest),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
