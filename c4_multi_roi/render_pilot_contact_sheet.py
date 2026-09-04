from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from .cache_utils import sha256_file
from .pilot_visual import render_overlay
from .schema import ROI_NAMES


ROOT = Path(__file__).resolve().parents[1]


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError(f"invalid box: {value}")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "c4_multi_roi/outputs/pilot_v1/pilot_manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "c4_multi_roi/outputs/pilot_v1/contact_sheet.jpg")
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tile_width, tile_height = 280, 380
    row_count = (len(rows) + args.columns - 1) // args.columns
    sheet = Image.new("RGB", (args.columns * tile_width, row_count * tile_height), color=(24, 24, 24))
    for index, row in enumerate(rows):
        source_path = ROOT / row["global_path"]
        with Image.open(source_path) as handle:
            source = handle.convert("L")
        boxes = {name: parse_box(row[f"roi_{name}_bbox"]) for name in ROI_NAMES}
        overlay = render_overlay(
            source,
            boxes,
            title=f"ID {row['image_id']} | {row['geometry_quality']} | thumb={row['thumb_side']}",
        )
        overlay.thumbnail((tile_width - 8, tile_height - 8), Image.Resampling.LANCZOS)
        x = (index % args.columns) * tile_width + (tile_width - overlay.width) // 2
        y = (index // args.columns) * tile_height + (tile_height - overlay.height) // 2
        sheet.paste(overlay, (x, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, format="JPEG", quality=92, optimize=True, progressive=True)
    report = {
        "pilot_rows": len(rows),
        "roi_records": len(rows) * len(ROI_NAMES),
        "geometry_ok_rows": sum(row["geometry_quality"] == "ok" for row in rows),
        "fallback_rows": sum(row["geometry_quality"] != "ok" for row in rows),
        "automatic_complete_rate": 1.0,
        "manual_anatomical_acceptance": "PENDING_VISUAL_REVIEW",
        "manifest_sha256": sha256_file(args.manifest),
        "contact_sheet_sha256": sha256_file(args.output),
    }
    report_path = args.output.with_name("pilot_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
