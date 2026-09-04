from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from c4_multi_roi.cache_utils import crop_pad_resize, sha256_file


class CacheUtilsTests(unittest.TestCase):
    def test_crop_pad_resize_is_square_and_deterministic(self) -> None:
        image = Image.new("L", (100, 200), color=120)
        first = crop_pad_resize(image, (10, 20, 50, 100), size=64)
        second = crop_pad_resize(image, (10, 20, 50, 100), size=64)
        self.assertEqual(first.mode, "L")
        self.assertEqual(first.size, (64, 64))
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_crop_pad_resize_rejects_out_of_bounds_box(self) -> None:
        image = Image.new("L", (100, 200), color=120)
        with self.assertRaises(ValueError):
            crop_pad_resize(image, (90, 190, 20, 20), size=64)

    def test_sha256_file_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.bin"
            path.write_bytes(b"c4")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"c4").hexdigest())


if __name__ == "__main__":
    unittest.main()
