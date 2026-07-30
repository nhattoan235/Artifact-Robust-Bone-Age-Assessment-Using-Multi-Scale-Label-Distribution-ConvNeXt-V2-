from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    if n <= 1:
        return np.zeros_like(mask, np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8) * 255


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create anatomy-preserving hand ROI crops from original pixels."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--protection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-ratio", type=float, default=0.018)
    args = parser.parse_args()

    crop_dir = args.output_dir / "crops"
    overlay_dir = args.output_dir / "review"
    crop_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    paths = sorted(args.input_dir.glob("*.png"), key=lambda p: p.stem)
    for index, path in enumerate(paths, 1):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        protection_path = args.protection_dir / f"{path.stem}.png"
        mask = cv2.imread(str(protection_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            raise ValueError(f"Missing image or protection mask for {path.stem}")
        if mask.shape != gray.shape:
            mask = cv2.resize(
                mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        hand = largest_component(mask)
        ys, xs = np.where(hand > 0)
        h, w = gray.shape
        area_pct = np.count_nonzero(hand) * 100.0 / hand.size
        fallback = False
        if len(xs) < 20 or area_pct > 76.0:
            # Conservative central crop for unreliable, near-full-frame masks.
            x0, x1 = int(w * 0.08), int(w * 0.92)
            y0, y1 = 0, h
            fallback = True
        else:
            margin = max(8, int(round(min(h, w) * args.margin_ratio)))
            x0 = max(0, int(xs.min()) - margin)
            x1 = min(w, int(xs.max()) + margin + 1)
            y0 = max(0, int(ys.min()) - margin)
            y1 = min(h, int(ys.max()) + margin + 1)

        crop = gray[y0:y1, x0:x1].copy()
        cv2.imwrite(str(crop_dir / f"{path.stem}.png"), crop)

        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), max(2, min(h, w) // 450))
        preview = cv2.resize(
            preview,
            (max(1, int(w * min(900 / w, 1100 / h))), max(1, int(h * min(900 / w, 1100 / h)))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(overlay_dir / f"{path.stem}.jpg"), preview)

        rows.append(
            {
                "Case_ID": path.stem,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "crop_width": x1 - x0,
                "crop_height": y1 - y0,
                "protection_area_pct": area_pct,
                "fallback_central_crop": fallback,
                "original_pixels_only": True,
            }
        )
        print(f"[{index:03d}/{len(paths):03d}] {path.stem}: {x1-x0}x{y1-y0} fallback={fallback}")

    with (args.output_dir / "hand_roi_qc.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed {len(rows)}/{len(paths)} ROI crops")


if __name__ == "__main__":
    main()
