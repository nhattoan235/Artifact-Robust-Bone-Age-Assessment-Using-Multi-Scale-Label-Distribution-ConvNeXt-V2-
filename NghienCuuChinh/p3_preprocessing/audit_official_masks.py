from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from p1_baseline.data import load_manifest, pad_square


EXPECTED_ZIP_BYTES = 306_332_153
EXPECTED_ZIP_MD5 = "a692de2799d99fd4bfec8cd535cd7979"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise RuntimeError(f"Khong doc duoc mask: {path}")
    return (value > (int(value.max()) // 2)).astype(np.uint8)


def align_manual(manual: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    target_h, target_w = target_shape
    h, w = manual.shape
    dh, dw = h - target_h, w - target_w
    if dh >= 0 and dw >= 0 and dh % 2 == 0 and dw % 2 == 0:
        top, left = dh // 2, dw // 2
        return manual[top:top + target_h, left:left + target_w], f"center_crop_{dh}x{dw}"
    return cv2.resize(manual, (target_w, target_h), interpolation=cv2.INTER_NEAREST), "nearest_resize"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(a, b).sum()) / denominator


def compare_sources(mask_root: Path) -> dict:
    source_dirs = {"eff_unet": mask_root / "eff_unet", "Tensormask": mask_root / "Tensormask"}
    values: dict[str, list[float]] = {name: [] for name in source_dirs}
    alignment_modes: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    for manual_path in sorted((mask_root / "manual").glob("*.png")):
        manual = binary_mask(manual_path)
        for name, source_dir in source_dirs.items():
            auto_path = source_dir / manual_path.name
            if not auto_path.exists():
                missing[name] += 1
                continue
            auto = binary_mask(auto_path)
            aligned, mode = align_manual(manual, auto.shape)
            alignment_modes[mode] += 1
            # Dice o 512 la du cho viec xep hang hai nguon va nhanh hon nhieu
            # so voi connected-pixel tren anh goc 2K-4K.
            aligned = cv2.resize(aligned, (512, 512), interpolation=cv2.INTER_NEAREST)
            auto = cv2.resize(auto, (512, 512), interpolation=cv2.INTER_NEAREST)
            values[name].append(dice(aligned, auto))
    result = {}
    for name, scores in values.items():
        array = np.asarray(scores, dtype=np.float64)
        result[name] = {
            "n": len(scores), "missing_manual_pairs": int(missing[name]),
            "mean_dice": float(array.mean()), "median_dice": float(np.median(array)),
            "p01_dice": float(np.percentile(array, 1)), "p05_dice": float(np.percentile(array, 5)),
            "minimum_dice": float(array.min()),
        }
    ranking = sorted(result, key=lambda name: result[name]["mean_dice"], reverse=True)
    return {"sources": result, "ranking": ranking, "manual_alignment_modes": dict(alignment_modes)}


def panel(original: Image.Image, mask: np.ndarray, processed: Image.Image, label: str, side: int = 320) -> Image.Image:
    def fit(image: Image.Image) -> Image.Image:
        image = image.convert("L")
        image.thumbnail((side, side - 30), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(image, ((side - image.width) // 2, 28 + (side - 28 - image.height) // 2))
        return canvas

    images = [fit(original), fit(Image.fromarray(mask * 255)), fit(processed)]
    output = Image.new("RGB", (side * 3, side), "black")
    for index, image in enumerate(images):
        output.paste(image.convert("RGB"), (index * side, 0))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.text((6, 6), label, fill="yellow", font=font)
    draw.text((side + 6, 6), "OFFICIAL MASK", fill="yellow", font=font)
    draw.text((side * 2 + 6, 6), "MASKED / NO CROP / NO ROTATE", fill="yellow", font=font)
    return output


def audit(manifests: list[tuple[str, str]], mask_root: Path, zip_path: Path, output_dir: Path, limit: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_sources(mask_root)
    primary, secondary = comparison["ranking"][:2]
    source_dirs = {"eff_unet": mask_root / "eff_unet", "Tensormask": mask_root / "Tensormask"}
    cache_root = output_dir / "cache"
    records: list[dict] = []
    candidate_panels: list[tuple[float, Image.Image]] = []

    for manifest_path, split in manifests:
        rows = load_manifest(manifest_path, split)
        if limit:
            rows = rows[:limit]
        split_dir = cache_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows, 1):
            source_path = Path(row["image_path"])
            with Image.open(source_path) as source:
                original = source.convert("L")
            gray = np.asarray(original, dtype=np.uint8)
            chosen = "raw_fallback"
            mask_path: Path | None = None
            for name in (primary, secondary):
                candidate = source_dirs[name] / f"{row['image_id']}.png"
                if candidate.exists():
                    chosen, mask_path = name, candidate
                    break
            shape_match = True
            empty_mask = False
            if mask_path is None:
                mask = np.ones_like(gray, dtype=np.uint8)
            else:
                mask = binary_mask(mask_path)
                shape_match = mask.shape == gray.shape
                if not shape_match:
                    mask = np.zeros_like(gray, dtype=np.uint8)
                empty_mask = not bool(mask.any())
            valid = shape_match and not empty_mask
            processed_array = gray * mask if valid else gray
            processed_original = Image.fromarray(processed_array, mode="L")
            processed = TF.resize(
                pad_square(processed_original), [512, 512],
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            )
            cache_path = split_dir / f"{row['image_id']}.png"
            processed.save(cache_path, compress_level=1)

            qc_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            count, _, stats, _ = cv2.connectedComponentsWithStats(qc_mask, connectivity=8)
            areas_cc = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)), reverse=True)
            ys, xs = np.nonzero(qc_mask)
            area_fraction = float(qc_mask.mean())
            if len(xs):
                x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                bbox_w, bbox_h = (x1 - x0 + 1) / qc_mask.shape[1], (y1 - y0 + 1) / qc_mask.shape[0]
                touches = [y0 == 0, y1 == qc_mask.shape[0] - 1, x0 == 0, x1 == qc_mask.shape[1] - 1]
            else:
                bbox_w = bbox_h = 0.0
                touches = [False] * 4
            largest_component_fraction = areas_cc[0] / max(1, int(qc_mask.sum())) if areas_cc else 0.0
            record = {
                "split": split, "image_id": row["image_id"], "bone_age_months": row["bone_age_months"],
                "sex": row["sex"], "source_path": str(source_path), "mask_source": chosen,
                "mask_path": str(mask_path) if mask_path else "", "cache_path": str(cache_path.resolve()),
                "shape_match": shape_match, "empty_mask": empty_mask, "valid": valid,
                "area_fraction": area_fraction, "bbox_width_fraction": bbox_w, "bbox_height_fraction": bbox_h,
                "component_count": max(0, count - 1), "largest_component_fraction": largest_component_fraction,
                "touches_top": touches[0], "touches_bottom": touches[1],
                "touches_left": touches[2], "touches_right": touches[3],
            }
            records.append(record)
            unusual = 100.0 * int(not valid or chosen == "raw_fallback") + 30.0 * int(largest_component_fraction < 0.98) + 25.0 * abs(area_fraction - 0.30) + 5.0 * sum(touches)
            if index <= 2 or unusual >= 14:
                candidate_panels.append((unusual, panel(original, qc_mask, processed, f"{split} ID={row['image_id']} age={row['bone_age_months']} sex={row['sex']} src={chosen}")))
            if index % 500 == 0:
                print(f"OFFICIAL_MASK_AUDIT split={split} {index}/{len(rows)}", flush=True)

    csv_path = output_dir / "p3_official_mask_qc.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    invalid = [row for row in records if not row["valid"]]
    raw_fallback = [row for row in records if row["mask_source"] == "raw_fallback"]
    areas = np.asarray([row["area_fraction"] for row in records if row["valid"]], dtype=np.float64)
    zip_size, zip_md5 = zip_path.stat().st_size, file_md5(zip_path)
    summary = {
        "protocol": "official_mask_v1_zero_background_no_crop_no_rotation",
        "source_comparison_against_528_manual_masks": comparison,
        "primary_source": primary, "secondary_source_for_missing_only": secondary,
        "source_counts": dict(Counter(row["mask_source"] for row in records)), "total": len(records),
        "invalid_count": len(invalid), "raw_fallback_count": len(raw_fallback),
        "raw_fallback_ids": [row["image_id"] for row in raw_fallback],
        "shape_mismatch_ids": [row["image_id"] for row in records if not row["shape_match"]],
        "empty_mask_ids": [row["image_id"] for row in records if row["empty_mask"]],
        "area_fraction": {"p001": float(np.percentile(areas, 0.1)), "p01": float(np.percentile(areas, 1)), "p50": float(np.median(areas)), "p99": float(np.percentile(areas, 99)), "p999": float(np.percentile(areas, 99.9))},
        "zip_provenance": {"path": str(zip_path.resolve()), "bytes": zip_size, "md5": zip_md5, "expected_bytes": EXPECTED_ZIP_BYTES, "expected_md5": EXPECTED_ZIP_MD5, "verified": zip_size == EXPECTED_ZIP_BYTES and zip_md5 == EXPECTED_ZIP_MD5, "doi": "10.5281/zenodo.7611677"},
        "gate": {
            "zip_verified": zip_size == EXPECTED_ZIP_BYTES and zip_md5 == EXPECTED_ZIP_MD5,
            "invalid_count_zero": len(invalid) == 0, "raw_fallback_at_most_one": len(raw_fallback) <= 1,
            "manual_mean_dice_at_least_0_98": comparison["sources"][primary]["mean_dice"] >= 0.98,
            "requires_visual_review": True,
        },
        "qc_csv": str(csv_path.resolve()), "cache_root": str(cache_root.resolve()),
    }
    summary["gate"]["automatic_pass"] = all(value for key, value in summary["gate"].items() if key != "requires_visual_review")
    temporary = output_dir / "p3_official_mask_summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "p3_official_mask_summary.json")
    selected = sorted(candidate_panels, key=lambda item: item[0], reverse=True)[:24]
    if selected:
        w, h = selected[0][1].size
        montage = Image.new("RGB", (w, h * len(selected)), "black")
        for index, (_, image) in enumerate(selected):
            montage.paste(image, (0, index * h))
        montage.save(output_dir / "p3_official_mask_visual_top24.jpg", quality=90)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", default="external")
    parser.add_argument("--zip-path", default="external/RSNA_bone_age_masks.clean.zip")
    parser.add_argument("--output-dir", default="p3_preprocessing/outputs/official_mask_v1")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    audit([("p0_audit/outputs/train_manifest.csv", "train"), ("p0_audit/outputs/validation_manifest.csv", "validation_official")], Path(args.mask_root), Path(args.zip_path), Path(args.output_dir), args.limit)


if __name__ == "__main__":
    main()
