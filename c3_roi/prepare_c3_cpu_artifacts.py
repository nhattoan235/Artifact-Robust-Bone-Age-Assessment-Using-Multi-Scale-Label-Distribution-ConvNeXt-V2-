from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from roi_utils import build_roi_record


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(rows: list[dict[str, str]], path: Path, limit: int) -> None:
    chosen = rows[:limit]
    tile_width, tile_height = 360, 300
    sheet = Image.new("RGB", (tile_width * 2, tile_height * ((len(chosen) + 1) // 2)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(chosen):
        image_path = resolve_path(row["source_image"])
        mask_path = resolve_path(row["mask_path"])
        with Image.open(image_path) as image:
            image = image.convert("L")
            image.thumbnail((170, 245))
            image_rgb = Image.merge("RGB", (image, image, image))
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask.thumbnail((170, 245))
            mask_rgb = Image.merge("RGB", (mask, mask, mask))
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(image_rgb, (5, 28))
        tile.paste(mask_rgb, (185, 28))
        draw_tile = ImageDraw.Draw(tile)
        draw_tile.text((5, 5), f"id={row['image_id']} mode={row.get('roi_mode', row.get('fallback_reason', ''))}", fill="black")
        draw_tile.text((5, 270), "original", fill="black")
        draw_tile.text((185, 270), "mask", fill="black")
        x = (index % 2) * tile_width
        y = (index // 2) * tile_height
        sheet.paste(tile, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "c3_roi/cache/C3_MASK_REUSE_CANDIDATE_V1",
    )
    parser.add_argument("--qc-limit", type=int, default=24)
    args = parser.parse_args()

    manifest_path = args.cache_root / "mask_manifest.csv"
    rows = read_rows(manifest_path)
    if not rows:
        raise ValueError("Mask manifest is empty")

    roi_rows: list[dict[str, str]] = []
    for row in rows:
        roi = build_roi_record(
            image_id=row["image_id"],
            width=int(row["width"]),
            height=int(row["height"]),
            bbox=row.get("bbox", ""),
            fallback_reason=row.get("fallback_reason", ""),
        )
        roi_rows.append({**row, **roi})

    failures = [row for row in roi_rows if row["roi_mode"] == "global_fallback"]
    valid = [row for row in roi_rows if row["roi_mode"] == "mask_bbox"]
    split_counts = Counter(row["split"] for row in roi_rows)
    failure_by_split = Counter(row["split"] for row in failures)
    summary = {
        "cache_root": str(args.cache_root),
        "rows": len(roi_rows),
        "mask_bbox_rows": len(valid),
        "global_fallback_rows": len(failures),
        "fallback_rate": len(failures) / len(roi_rows),
        "rows_by_split": dict(split_counts),
        "fallback_by_split": dict(failure_by_split),
        "fallback_reasons": dict(Counter(row["fallback_reason"] for row in failures)),
        "test_rows_in_manifest": sum(row["split"].lower().startswith("test") for row in roi_rows),
        "policy": {
            "keep_all_development_rows": True,
            "test_masks_used": False,
            "fallback_is_explicit": True,
            "roi_margin": 0.12,
        },
    }
    output_root = args.cache_root / "cpu_artifacts"
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "roi_manifest.csv", roi_rows)
    (output_root / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_contact_sheet(failures, output_root / "qc_failures.jpg", args.qc_limit)
    make_contact_sheet(valid, output_root / "qc_valid.jpg", args.qc_limit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
