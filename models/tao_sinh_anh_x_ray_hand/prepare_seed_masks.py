from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def is_hand_like(mask: np.ndarray) -> bool:
    binary = (mask > 0).astype(np.uint8) * 255
    area_pct = np.count_nonzero(binary) * 100.0 / binary.size
    if not 5.0 <= area_pct <= 75.0:
        return False
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    _, (rw, rh), _ = cv2.minAreaRect(contour)
    solidity = area / max(hull_area, 1.0)
    extent = area / max(rw * rh, 1.0)
    return not (area_pct > 15.0 and solidity > 0.90 and extent > 0.82)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Union legacy hand masks into conservative protection seeds."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("mask_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_case: dict[str, list[Path]] = {}
    for directory in args.mask_dirs:
        for path in directory.glob("*_mask.npy"):
            by_case.setdefault(path.name.replace("_mask.npy", ""), []).append(path)

    for case_id, paths in sorted(by_case.items()):
        masks = [np.load(path) for path in paths]
        masks = [mask > 0 for mask in masks if is_hand_like(mask)]
        if not masks:
            continue
        shape = masks[0].shape
        if any(mask.shape != shape for mask in masks):
            raise ValueError(f"Mask shape mismatch for case {case_id}")
        union = np.logical_or.reduce(masks).astype(np.uint8) * 255
        cv2.imwrite(str(args.output_dir / f"{case_id}.png"), union)

    print(f"Wrote {len(by_case)} protection seeds to {args.output_dir}")


if __name__ == "__main__":
    main()
