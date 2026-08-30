from __future__ import annotations

import unittest

from c4_multi_roi.schema import ROI_NAMES, VIEW_NAMES, validate_record


class MultiROIRecordTests(unittest.TestCase):
    def valid_record(self) -> dict:
        roi_paths = {name: f"roi/{name}/1.jpg" for name in ROI_NAMES}
        roi_boxes = {name: "1,2,30,40" for name in ROI_NAMES}
        roi_hashes = {name: format(index + 1, "x") * 64 for index, name in enumerate(ROI_NAMES)}
        return {
            "image_id": "1",
            "split": "train",
            "fold": "1",
            "global_path": "global/1.jpg",
            "source_sha256": "a" * 64,
            "sex": "M",
            "bone_age_months": "120",
            "roi_paths": roi_paths,
            "roi_boxes": roi_boxes,
            "roi_sha256": roi_hashes,
            "quality_flags": {name: "ok" for name in ROI_NAMES},
        }

    def test_view_order_is_locked(self) -> None:
        self.assertEqual(
            ROI_NAMES,
            (
                "carpal",
                "mcp_thumb",
                "mcp_index",
                "mcp_middle",
                "mcp_ring",
                "mcp_little",
            ),
        )
        self.assertEqual(VIEW_NAMES, ("global",) + ROI_NAMES)

    def test_valid_record_is_accepted(self) -> None:
        validate_record(self.valid_record())

    def test_record_rejects_missing_roi(self) -> None:
        record = self.valid_record()
        del record["roi_paths"]["mcp_little"]
        with self.assertRaisesRegex(ValueError, "roi_paths"):
            validate_record(record)

    def test_record_rejects_test_split(self) -> None:
        record = self.valid_record()
        record["split"] = "test"
        with self.assertRaisesRegex(ValueError, "test"):
            validate_record(record, development_only=True)

    def test_record_rejects_invalid_hash(self) -> None:
        record = self.valid_record()
        record["source_sha256"] = "short"
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            validate_record(record)


if __name__ == "__main__":
    unittest.main()
