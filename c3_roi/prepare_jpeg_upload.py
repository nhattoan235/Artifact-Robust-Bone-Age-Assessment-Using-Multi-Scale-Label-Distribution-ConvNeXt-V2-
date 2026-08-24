from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=ROOT / "c3_roi/cache/C3_ROI_V1")
    args = parser.parse_args()
    roi_root = args.cache_root / "roi"
    pngs = list(roi_root.rglob("*.png"))
    for index, source in enumerate(pngs, start=1):
        target = source.with_suffix(".jpg")
        with Image.open(source) as image:
            image = image.convert("L")
            side = max(image.size)
            canvas = Image.new("L", (side, side), color=0)
            canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
            canvas = canvas.resize((512, 512), Image.Resampling.BICUBIC)
            canvas.save(target, format="JPEG", quality=95, optimize=True, progressive=True)
        if index % 500 == 0 or index == len(pngs):
            print(f"converted={index}/{len(pngs)}", flush=True)

    for manifest_path in (ROOT / "c3_roi/manifests").glob("fold_*.csv"):
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for row in rows:
            row["image_path"] = row["image_path"].replace(".png", ".jpg")
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"png_count={len(pngs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
