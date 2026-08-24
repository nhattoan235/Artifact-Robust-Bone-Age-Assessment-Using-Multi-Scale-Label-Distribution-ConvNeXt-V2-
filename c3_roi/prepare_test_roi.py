"""Prepare the locked C3 ROI preprocessing for the 200-image test set."""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASK_CODE = Path(r"D:\Learning\DoAn Tot nghiep\gpt-image-bone-age-synthesis\models\cau_hinh_C")
sys.path.insert(0, str(MASK_CODE))
from mask_generator import segment_hand  # noqa: E402

from roi_utils import build_roi_record


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    image_root = ROOT / "data/goc/boneage-test-dataset/boneage-test-dataset"
    out_root = ROOT / "c3_roi/cache/C3_ROI_V1_TEST"
    out_root.mkdir(parents=True, exist_ok=True)
    roi_dir = out_root / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    images = sorted(image_root.glob("*.png"), key=lambda p: int(p.stem))
    if len(images) != 200:
        raise RuntimeError(f"Expected 200 test images, found {len(images)}")
    for index, source in enumerate(images, 1):
        image_id = source.stem
        with Image.open(source) as image:
            gray = np.asarray(image.convert("L"))
        _region, _hull, bbox = segment_hand(gray)
        fallback = "ok" if bbox is not None else "segment_hand_failed"
        record = build_roi_record(
            image_id=image_id, width=gray.shape[1], height=gray.shape[0],
            bbox="" if bbox is None else ",".join(str(int(v)) for v in bbox),
            fallback_reason=fallback,
        )
        x, y, w, h = (int(v) for v in record["roi_bbox"].split(","))
        crop = Image.fromarray(gray).crop((x, y, x + w, y + h))
        destination = roi_dir / f"{image_id}.png"
        crop.save(destination, format="PNG", optimize=True)
        rows.append({
            "split": "test", "image_id": image_id, "source_image": str(source),
            "roi_image": str(destination), "source_sha256": sha256(source),
            "roi_sha256": sha256(destination), "roi_mode": record["roi_mode"],
            "roi_bbox": record["roi_bbox"], "fallback_reason": fallback,
            "width": str(gray.shape[1]), "height": str(gray.shape[0]),
        })
        if index % 25 == 0 or index == len(images):
            print(f"processed={index}/200", flush=True)
    with (out_root / "test_roi_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote={out_root}")
    print(f"mask_bbox={sum(r['roi_mode'] == 'mask_bbox' for r in rows)} fallback={sum(r['roi_mode'] != 'mask_bbox' for r in rows)}")


if __name__ == "__main__":
    main()
