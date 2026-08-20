from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from p1_baseline.data import load_manifest
from p1_baseline.preprocessing import FullHandSettings, full_hand_preprocess


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def make_panel(original: Image.Image, processed: Image.Image, mask: np.ndarray, text: str, side: int = 360) -> Image.Image:
    def fit(image: Image.Image) -> Image.Image:
        image = image.convert("L")
        image.thumbnail((side, side - 34), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(image, ((side - image.width) // 2, 30 + (side - 30 - image.height) // 2))
        return canvas

    mask_image = Image.fromarray(mask, mode="L") if mask.size else Image.new("L", original.size, 0)
    panels = [fit(original), fit(mask_image), fit(processed)]
    result = Image.new("RGB", (side * 3, side), "black")
    for index, panel in enumerate(panels):
        result.paste(panel.convert("RGB"), (index * side, 0))
    draw = ImageDraw.Draw(result)
    draw.text((8, 8), text, fill="yellow", font=ImageFont.load_default())
    draw.text((side + 8, 8), "MASK", fill="yellow", font=ImageFont.load_default())
    draw.text((side * 2 + 8, 8), "PROCESSED", fill="yellow", font=ImageFont.load_default())
    return result


def audit(manifests: list[tuple[str, str]], output_dir: Path, limit: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "cache"
    rows_out: list[dict] = []
    visual_candidates: list[tuple[float, Image.Image]] = []
    settings = FullHandSettings()

    for manifest_path, split in manifests:
        rows = load_manifest(manifest_path, split)
        if limit:
            rows = rows[:limit]
        split_dir = cache_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows, 1):
            with Image.open(row["image_path"]) as source:
                original = source.convert("L")
            processed, qc, mask = full_hand_preprocess(original, settings)
            processed.save(split_dir / f"{row['image_id']}.png", compress_level=3)
            record = {
                "split": split, "image_id": row["image_id"], "bone_age_months": row["bone_age_months"],
                "sex": row["sex"], "source_path": row["image_path"],
                "cache_path": str((split_dir / f"{row['image_id']}.png").resolve()), **qc.to_dict(),
            }
            rows_out.append(record)
            unusual = (
                (100.0 if qc.fallback else 0.0)
                + abs(qc.alignment_degrees)
                + 20.0 * int(qc.touches_top or qc.touches_left or qc.touches_right)
                + 10.0 * abs(qc.component_area_fraction - 0.30)
            )
            if index <= 2 or qc.fallback or unusual > 22:
                panel = make_panel(
                    original, processed, mask,
                    f"{split} ID={row['image_id']} age={row['bone_age_months']} sex={row['sex']} "
                    f"fallback={qc.fallback} angle={qc.applied_alignment_degrees:.1f}",
                )
                visual_candidates.append((unusual, panel))
            if index % 500 == 0:
                print(f"AUDIT_PROGRESS split={split} {index}/{len(rows)}", flush=True)

    csv_path = output_dir / "p3_preprocessing_qc.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    failures = Counter(row["reason"] for row in rows_out if row["fallback"])
    successful = [row for row in rows_out if row["success"]]
    fallback_rate = sum(bool(row["fallback"]) for row in rows_out) / len(rows_out)
    side_touch_rate = sum(bool(row["touches_top"] or row["touches_left"] or row["touches_right"]) for row in successful) / max(1, len(successful))
    summary = {
        "settings": settings.__dict__, "total": len(rows_out), "success": len(successful),
        "fallback_count": len(rows_out) - len(successful), "fallback_rate": fallback_rate,
        "fallback_reasons": dict(failures), "non_wrist_border_touch_rate": side_touch_rate,
        "area_fraction": {
            "p01": percentile([r["component_area_fraction"] for r in successful], 1),
            "p50": percentile([r["component_area_fraction"] for r in successful], 50),
            "p99": percentile([r["component_area_fraction"] for r in successful], 99),
        },
        "applied_alignment_abs_degrees": {
            "p50": percentile([abs(r["applied_alignment_degrees"]) for r in successful], 50),
            "p95": percentile([abs(r["applied_alignment_degrees"]) for r in successful], 95),
            "max": max((abs(r["applied_alignment_degrees"]) for r in successful), default=0.0),
        },
        "gate": {
            "fallback_rate_max": 0.02, "fallback_rate_pass": fallback_rate <= 0.02,
            "requires_visual_review": True,
            "pass": fallback_rate <= 0.02,
        },
        "qc_csv": str(csv_path.resolve()), "cache_root": str(cache_root.resolve()),
    }
    atomic_json(output_dir / "p3_preprocessing_summary.json", summary)

    selected = sorted(visual_candidates, key=lambda item: item[0], reverse=True)[:24]
    if selected:
        width, height = selected[0][1].size
        montage = Image.new("RGB", (width, height * len(selected)), "black")
        for index, (_, panel) in enumerate(selected):
            montage.paste(panel, (0, index * height))
        montage.save(output_dir / "p3_visual_audit_top24.jpg", quality=90)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="p3_preprocessing/outputs/full_hand_v1")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    audit([
        ("p0_audit/outputs/train_manifest.csv", "train"),
        ("p0_audit/outputs/validation_manifest.csv", "validation_official"),
    ], Path(args.output_dir), args.limit)


if __name__ == "__main__":
    main()
