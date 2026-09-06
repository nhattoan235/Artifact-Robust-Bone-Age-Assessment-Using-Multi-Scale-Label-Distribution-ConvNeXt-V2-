from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXTERNAL_MASK_CODE = Path(r"D:\Learning\DoAn Tot nghiep\gpt-image-bone-age-synthesis\models\cau_hinh_C")
sys.path.insert(0, str(EXTERNAL_MASK_CODE))
from c3_roi.segmentation import segment_hand_with_fallback  # noqa: E402


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


def process_row(row: dict[str, str], mask_dir: Path) -> dict[str, str]:
    """Build one mask row in a worker process; each image has a unique output."""
    cv2.setNumThreads(1)
    image_id = str(int(float(row["image_id"])))
    image_path = resolve_image(row)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"))
    hand_region, _hull, bbox, reason = segment_hand_with_fallback(gray)
    mask_path = mask_dir / f"{image_id}.png"
    if hand_region is None:
        hand_region = np.zeros_like(gray, dtype=np.uint8)
    Image.fromarray(hand_region.astype(np.uint8), mode="L").save(
        mask_path, format="PNG", optimize=True
    )
    return {
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "c3_roi/cache/C3_MASK_REUSE_CANDIDATE_V1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1) // 2)),
        help="CPU worker processes; segmentation itself is NumPy/OpenCV, not CUDA.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    rows = read_manifest(ROOT / "p0_audit/outputs/train_manifest.csv") + read_manifest(ROOT / "p0_audit/outputs/validation_manifest.csv")
    rows.sort(key=lambda row: int(float(row["image_id"])))
    if args.limit is not None:
        rows = rows[: args.limit]
    mask_dir = args.output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = executor.map(process_row, rows, [mask_dir] * len(rows), chunksize=1)
        for index, output_row in enumerate(futures, start=1):
            output_rows.append(output_row)
            if index % 100 == 0 or index == len(rows):
                print(f"processed={index}/{len(rows)}", flush=True)
    manifest_path = args.output_root / "mask_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote={manifest_path}")
    print(
        f"rows={len(output_rows)} "
        f"strict_ok={sum(r['fallback_reason'] == 'ok' for r in output_rows)} "
        f"border_rescue={sum(r['fallback_reason'] == 'border_rescue' for r in output_rows)} "
        f"fallback={sum(r['fallback_reason'] == 'segment_hand_failed' for r in output_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
