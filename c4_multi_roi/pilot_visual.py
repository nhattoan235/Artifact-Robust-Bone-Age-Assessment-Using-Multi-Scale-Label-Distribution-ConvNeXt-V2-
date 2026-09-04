from __future__ import annotations

from collections.abc import Mapping

from PIL import Image, ImageDraw


COLORS = {
    "carpal": (255, 0, 0),
    "mcp_thumb": (255, 165, 0),
    "mcp_index": (255, 255, 0),
    "mcp_middle": (0, 255, 0),
    "mcp_ring": (0, 180, 255),
    "mcp_little": (200, 0, 255),
}


def render_overlay(
    source: Image.Image,
    boxes: Mapping[str, tuple[int, int, int, int]],
    *,
    title: str,
) -> Image.Image:
    output = source.convert("RGB")
    draw = ImageDraw.Draw(output)
    line_width = max(2, round(max(output.size) / 350))
    for name, (x, y, width, height) in boxes.items():
        color = COLORS.get(name, (255, 255, 255))
        draw.rectangle((x, y, x + width, y + height), outline=color, width=line_width)
        draw.text((x + line_width, y + line_width), name, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    draw.rectangle((0, 0, min(output.width, 420), 28), fill=(0, 0, 0))
    draw.text((6, 6), title, fill=(255, 255, 255))
    return output

