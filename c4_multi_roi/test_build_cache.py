from __future__ import annotations

import unittest

from c4_multi_roi.build_cache import select_pilot_rows, source_relative_path


class BuildCacheTests(unittest.TestCase):
    def test_source_relative_path_uses_locked_dataset_layout(self) -> None:
        self.assertEqual(
            source_relative_path("123", "train").as_posix(),
            "data/goc/boneage-training-dataset/boneage-training-dataset/123.png",
        )
        self.assertEqual(
            source_relative_path("456", "validation_official").as_posix(),
            "data/rsna_official_validation/images/456.png",
        )

    def test_source_relative_path_rejects_test(self) -> None:
        with self.assertRaises(ValueError):
            source_relative_path("1", "test")

    def test_pilot_selection_is_deterministic_and_includes_fallbacks(self) -> None:
        rows = [
            {"image_id": str(index), "fallback_reason": "segment_hand_failed" if index % 5 == 0 else "ok"}
            for index in range(500)
        ]
        first = select_pilot_rows(rows, count=100, seed=42)
        second = select_pilot_rows(rows, count=100, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(sum(row["fallback_reason"] != "ok" for row in first), 20)


if __name__ == "__main__":
    unittest.main()
