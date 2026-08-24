from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_MASK_CODE = Path(r"D:\Learning\DoAn Tot nghiep\gpt-image-bone-age-synthesis\models\cau_hinh_C")
sys.path.insert(0, str(EXTERNAL_MASK_CODE))
from mask_generator import segment_hand  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_image(row: dict[str, str]) -> Path:
    image_id = str(int(float(row["image_id"])))
    if row.get("split") == "validation_official":
        return ROOT / "data/rsna_official_validation/images" / f"{image_id}.png"
    return ROOT / "data/goc/boneage-training-dataset/boneage-training-dataset" / f"{image_id}.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "c3_roi/cache/C3_MASK_REUSE_CANDIDATE_V1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_manifest(ROOT / "p0_audit/outputs/train_manifest.csv") + read_manifest(ROOT / "p0_audit/outputs/validation_manifest.csv")
    rows.sort(key=lambda row: int(float(row["image_id"])))
    if args.limit is not None:
        rows = rows[: args.limit]
    mask_dir = args.output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, row in enumerate(rows, start=1):
        image_id = str(int(float(row["image_id"])))
        image_path = resolve_image(row)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"))
        hand_region, _hull, bbox = segment_hand(gray)
        reason = "ok" if hand_region is not None and bbox is not None else "segment_hand_failed"
        mask_path = mask_dir / f"{image_id}.png"
        if hand_region is None:
            hand_region = np.zeros_like(gray, dtype=np.uint8)
        Image.fromarray(hand_region.astype(np.uint8), mode="L").save(mask_path, format="PNG", optimize=True)
        output_rows.append({
            "image_id": image_id,
            "split": row["split"],
            "source_image": str(image_path),
            "source_image_sha256": sha256(image_path),
            "mask_path": str(mask_path),
            "mask_sha256": sha256(mask_path),
            "height": str(gray.shape[0]),
            "width": str(gray.shape[1]),
            "mask_pixels": str(int(np.count_nonzero(hand_region))),
            "bbox": "" if bbox is None else ",".join(str(int(x)) for x in bbox),
            "fallback_reason": reason,
        })
        if index % 100 == 0 or index == len(rows):
            print(f"processed={index}/{len(rows)}", flush=True)
    manifest_path = args.output_root / "mask_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote={manifest_path}")
    print(f"rows={len(output_rows)} failures={sum(r['fallback_reason'] != 'ok' for r in output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
