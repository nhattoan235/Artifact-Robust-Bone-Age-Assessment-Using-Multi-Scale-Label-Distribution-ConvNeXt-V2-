from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def fit_thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-sheet", type=int, default=25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        [*args.input_dir.glob("*.png"), *args.input_dir.glob("*.jpg")],
        key=lambda path: path.stem,
    )
    cols = 5
    cell_w, cell_h = 260, 330
    for page, start in enumerate(range(0, len(paths), args.per_sheet), 1):
        batch = paths[start : start + args.per_sheet]
        rows = math.ceil(len(batch) / cols)
        sheet = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
        for index, path in enumerate(batch):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            cell = np.zeros((cell_h, cell_w, 3), np.uint8)
            cell[: cell_h - 28] = fit_thumbnail(image, cell_w, cell_h - 28)
            cv2.putText(
                cell,
                path.stem,
                (8, cell_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )
            row, col = divmod(index, cols)
            sheet[
                row * cell_h : (row + 1) * cell_h,
                col * cell_w : (col + 1) * cell_w,
            ] = cell
        cv2.imwrite(str(args.output_dir / f"sheet_{page:02d}.jpg"), sheet)
    print(f"Wrote {math.ceil(len(paths) / args.per_sheet)} sheets for {len(paths)} images")


if __name__ == "__main__":
    main()
