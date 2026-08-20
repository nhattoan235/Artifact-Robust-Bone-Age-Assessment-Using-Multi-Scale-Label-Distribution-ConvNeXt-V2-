from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def robust_background(gray: np.ndarray, hand: np.ndarray) -> int:
    values = gray[hand == 0]
    if values.size < 100:
        return int(np.median(gray))
    hist = np.bincount(values, minlength=256).astype(np.float32)
    hist = np.convolve(hist, np.ones(13, np.float32), mode="same")
    return int(np.argmax(hist))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preserve hand pixels exactly and replace only the exterior."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--protection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "hand_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    paths = sorted(args.input_dir.glob("*.png"), key=lambda p: p.stem)

    for index, path in enumerate(paths, 1):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        hand = cv2.imread(
            str(args.protection_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE
        )
        if gray is None or hand is None:
            raise ValueError(f"Missing image or mask for {path.stem}")
        if hand.shape != gray.shape:
            hand = cv2.resize(
                hand, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        hand = (hand > 0).astype(np.uint8) * 255
        area_pct = np.count_nonzero(hand) * 100.0 / hand.size
        reliable = 7.0 <= area_pct <= 76.0
        background = robust_background(gray, hand)

        if reliable:
            result = np.full_like(gray, background)
            result[hand > 0] = gray[hand > 0]
            status = "MASKED_HAND_READY"
        else:
            # Never risk deleting anatomy when the protection mask is invalid.
            result = gray.copy()
            status = "UNCHANGED_REVIEW_MASK"

        diff = cv2.absdiff(gray, result)
        changed_inside = int(np.count_nonzero((diff > 0) & (hand > 0)))
        cv2.imwrite(str(image_dir / f"{path.stem}.png"), result)
        cv2.imwrite(str(mask_dir / f"{path.stem}.png"), hand)
        rows.append(
            {
                "Case_ID": path.stem,
                "Status": status,
                "hand_area_pct": area_pct,
                "background_level": background,
                "changed_inside_hand": changed_inside,
                "anatomy_pixel_preservation_pass": changed_inside == 0,
            }
        )
        print(
            f"[{index:03d}/{len(paths):03d}] {path.stem}: "
            f"{status} hand={area_pct:.1f}% background={background}"
        )

    with (args.output_dir / "masked_hand_qc.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed {len(rows)}/{len(paths)} masked-hand images")


if __name__ == "__main__":
    main()
