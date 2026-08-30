from __future__ import annotations

import unittest

from PIL import Image

from c4_multi_roi.pilot_visual import render_overlay


class PilotVisualTests(unittest.TestCase):
    def test_overlay_keeps_source_size_and_returns_rgb(self) -> None:
        source = Image.new("L", (100, 200), color=80)
        boxes = {"carpal": (10, 20, 30, 40)}
        result = render_overlay(source, boxes, title="1")
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, source.size)


if __name__ == "__main__":
    unittest.main()
