from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_pad_resize(image: Image.Image, box: tuple[int, int, int, int], *, size: int) -> Image.Image:
    if size <= 0:
        raise ValueError("size must be positive")
    x, y, width, height = (int(value) for value in box)
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        raise ValueError(f"invalid crop box: {box}")
    if x + width > image.width or y + height > image.height:
        raise ValueError(f"crop box {box} exceeds image size {image.size}")
    crop = image.convert("L").crop((x, y, x + width, y + height))
    side = max(crop.width, crop.height)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas.resize((size, size), Image.Resampling.BICUBIC)

