from __future__ import annotations

"""Tạo cache P9 theo các bước intensity của Deeplasia, có provenance và QC.

Script chỉ dùng train và validation chính thức. ``mask_v1`` là mask + trừ
percentile 1 của foreground; ``mask_histogram_v1`` thêm histogram equalization.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from p1_baseline.data import load_manifest

MODES = {"mask_v1", "mask_histogram_v1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise RuntimeError(f"Không đọc được mask: {path}")
    maximum = int(value.max())
    if maximum <= 0:
        raise RuntimeError(f"Mask rỗng: {path}")
    return (value > maximum // 2).astype(np.uint8)


def foreground_histogram_equalization(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CDF equalization xác định, chỉ tính pixel foreground như Deeplasia."""
    # Deeplasia lập histogram trên ``image[image > 0]`` sau khi đã trừ
    # percentile. Pixel foreground bị clip về 0 không được tính vào CDF.
    foreground = image[image > 0]
    if foreground.size == 0:
        raise RuntimeError("Không thể equalize foreground rỗng")
    histogram = np.bincount(foreground, minlength=256).astype(np.float64)
    cdf = np.cumsum(histogram) / foreground.size
    result = np.rint(cdf[image] * 255.0).astype(np.uint8)
    result[mask == 0] = 0
    return result


def process(gray: np.ndarray, mask: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    if gray.shape != mask.shape:
        raise RuntimeError(f"Shape image/mask không khớp: {gray.shape} vs {mask.shape}")
    masked = (gray * mask).astype(np.uint8)
    foreground = masked[masked > 0]
    if foreground.size == 0:
        raise RuntimeError("Foreground rỗng sau masking")
    percentile_01 = float(np.percentile(foreground, 1))
    shifted = cv2.subtract(masked, int(round(percentile_01)))
    shifted[mask == 0] = 0
    return (
        foreground_histogram_equalization(shifted, mask)
        if mode == "mask_histogram_v1" else shifted,
        percentile_01,
    )


def select_mask(mask_root: Path, image_id: str) -> tuple[np.ndarray | None, str, Path | None]:
    for source in ("eff_unet", "Tensormask"):
        path = mask_root / source / f"{image_id}.png"
        if path.is_file():
            return binary_mask(path), source, path
    return None, "raw_fallback", None


def atomic_png(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    Image.fromarray(array, mode="L").save(temporary, format="PNG", compress_level=1)
    os.replace(temporary, destination)


def visual_panel(raw: np.ndarray, mask: np.ndarray, processed: np.ndarray, label: str) -> Image.Image:
    side = 280
    def fit(value: np.ndarray) -> Image.Image:
        image = Image.fromarray(value, mode="L")
        image.thumbnail((side, side - 28), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(image, ((side - image.width) // 2, 28 + (side - 28 - image.height) // 2))
        return canvas
    result = Image.new("RGB", (side * 3, side), "black")
    for index, value in enumerate((raw, mask * 255, processed)):
        result.paste(fit(value).convert("RGB"), (index * side, 0))
    draw, font = ImageDraw.Draw(result), ImageFont.load_default()
    draw.text((5, 5), label, fill="yellow", font=font)
    draw.text((side + 5, 5), "MASK", fill="yellow", font=font)
    draw.text((2 * side + 5, 5), "P9 OUTPUT", fill="yellow", font=font)
    return result


def run(manifests: list[tuple[Path, str]], mask_root: Path, output_dir: Path, mode: str, limit: int | None) -> dict:
    if mode not in MODES:
        raise ValueError(f"Mode không hợp lệ: {mode}")
    records: list[dict] = []
    candidates: list[tuple[float, Image.Image]] = []
    for manifest, split in manifests:
        rows = load_manifest(manifest, split)
        if limit is not None:
            rows = rows[:limit]
        for index, row in enumerate(rows, 1):
            with Image.open(row["image_path"]) as source:
                raw = np.asarray(source.convert("L"), dtype=np.uint8)
            mask, source_name, mask_path = select_mask(mask_root, row["image_id"])
            raw_fallback = mask is None
            if raw_fallback:
                mask = np.ones_like(raw, dtype=np.uint8)
            if mask.shape != raw.shape or not bool(mask.any()):
                raise RuntimeError(f"Mask không hợp lệ cho ID={row['image_id']}")
            processed, percentile_01 = process(raw, mask, mode)
            output_path = output_dir / "cache" / split / f"{row['image_id']}.png"
            atomic_png(processed, output_path)
            area_fraction = float(mask.mean())
            record = {
                "split": split, "image_id": row["image_id"], "sex": row["sex"],
                "mask_source": source_name, "mask_path": str(mask_path) if mask_path else "",
                "raw_fallback": raw_fallback, "foreground_area_fraction": area_fraction,
                "foreground_percentile_01": percentile_01, "input_sha256": row["sha256"],
                "output_path": str(output_path.resolve()), "output_sha256": sha256(output_path),
            }
            records.append(record)
            unusual = 100.0 * int(raw_fallback) + 35.0 * abs(area_fraction - 0.325) + percentile_01 / 255.0
            if index <= 3 or unusual >= 10.0:
                candidates.append((unusual, visual_panel(raw, mask, processed, f"{split} ID={row['image_id']} src={source_name}")))
            if index % 500 == 0:
                print(f"P9_PREPROCESS {mode} {split} {index}/{len(rows)}", flush=True)
    if not records:
        raise RuntimeError("Không có record để audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = output_dir / "qc.csv"
    with qc_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:24]
    visual_path = output_dir / "visual_top24.jpg"
    if selected:
        width, height = selected[0][1].size
        montage = Image.new("RGB", (width, height * len(selected)), "black")
        for index, (_, image) in enumerate(selected):
            montage.paste(image, (0, index * height))
        montage.save(visual_path, quality=90)
    areas = np.asarray([record["foreground_area_fraction"] for record in records])
    summary = {
        "status": "PASS", "protocol": f"deeplasia_{mode}",
        "source_precedence": ["eff_unet", "Tensormask", "raw_fallback"],
        "total": len(records), "source_counts": dict(Counter(record["mask_source"] for record in records)),
        "raw_fallback_ids": [record["image_id"] for record in records if record["raw_fallback"]],
        "foreground_area_fraction": {"p01": float(np.percentile(areas, 1)), "median": float(np.median(areas)), "p99": float(np.percentile(areas, 99))},
        "manifest_sha256": {split: sha256(path) for path, split in manifests},
        "qc_csv": str(qc_path.resolve()), "visual_review": str(visual_path.resolve()),
        "cache_root": str((output_dir / "cache").resolve()), "test_used": False,
    }
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="P9 Deeplasia-inspired preprocessing; development-only")
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--mask-root", type=Path, default=Path("external"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(
        [(Path("p0_audit/outputs/train_manifest.csv"), "train"), (Path("p0_audit/outputs/validation_manifest.csv"), "validation_official")],
        args.mask_root, args.output_dir, args.mode, args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
