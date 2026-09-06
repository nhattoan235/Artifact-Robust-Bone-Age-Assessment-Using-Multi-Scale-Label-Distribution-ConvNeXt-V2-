from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from roi_utils import build_roi_record, parse_bbox


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=ROOT / "c3_roi/cache/C3_MASK_REUSE_CANDIDATE_V1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "c3_roi/cache/C3_ROI_V1",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_rows(args.mask_root / "mask_manifest.csv")
    if args.limit is not None:
        rows = rows[: args.limit]
    roi_dir = args.output_root / "roi"
    output_rows = []
    for index, row in enumerate(rows, start=1):
        source_path = resolve_path(row["source_image"])
        with Image.open(source_path) as source:
            image = source.convert("L")
            roi_record = build_roi_record(
                image_id=row["image_id"],
                width=int(row["width"]),
                height=int(row["height"]),
                bbox=row.get("bbox", ""),
                fallback_reason=row.get("fallback_reason", ""),
            )
            x, y, width, height = (int(value) for value in roi_record["roi_bbox"].split(","))
            crop = image.crop((x, y, x + width, y + height))
            split = row["split"]
            output_path = roi_dir / split / f"{row['image_id']}.png"
            relative_path = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Không dùng PNG optimize ở bước chuẩn bị: với 14.036 ảnh X-quang,
            # tối ưu nén làm CPU trở thành nút thắt lớn mà không thay đổi pixel.
            crop.save(output_path, format="PNG", optimize=False)
        output_rows.append({
            "image_id": row["image_id"],
            "split": split,
            "source_image": row["source_image"],
            "source_image_sha256": row["source_image_sha256"],
            "mask_sha256": row["mask_sha256"],
            "roi_path": str(relative_path),
            "roi_mode": roi_record["roi_mode"],
            "roi_bbox": roi_record["roi_bbox"],
            "fallback_reason": roi_record["fallback_reason"],
            "readable": "true",
        })
        if index % 500 == 0 or index == len(rows):
            print(f"processed={index}/{len(rows)}", flush=True)

    if not output_rows:
        raise RuntimeError("No rows to process")
    manifest_path = args.output_root / "roi_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote={manifest_path}")
    print(f"rows={len(output_rows)} fallback={sum(r['roi_mode'] == 'global_fallback' for r in output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
