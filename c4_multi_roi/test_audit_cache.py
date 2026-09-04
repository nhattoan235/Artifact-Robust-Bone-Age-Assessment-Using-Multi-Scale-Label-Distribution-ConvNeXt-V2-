from __future__ import annotations

import unittest

from c4_multi_roi.audit_cache import audit_rows
from c4_multi_roi.schema import ROI_NAMES


def valid_row(image_id: str) -> dict[str, str]:
    row = {"image_id": image_id, "global_path": f"data/{image_id}.png", "fold": "1"}
    for name in ROI_NAMES:
        row[f"roi_{name}_path"] = f"roi/{name}/{image_id}.jpg"
        row[f"roi_{name}_sha256"] = "a" * 64
        row[f"roi_{name}_bbox"] = "1,2,30,40"
    return row


class CacheAuditTests(unittest.TestCase):
    def test_valid_rows_have_unique_ids_and_six_rois(self) -> None:
        report = audit_rows([valid_row("1"), valid_row("2")], check_files=False)
        self.assertTrue(report["pass"])
        self.assertEqual(report["unique_ids"], 2)
        self.assertEqual(report["roi_records"], 12)

    def test_duplicate_id_fails(self) -> None:
        report = audit_rows([valid_row("1"), valid_row("1")], check_files=False)
        self.assertFalse(report["pass"])
        self.assertEqual(report["duplicate_ids"], ["1"])


if __name__ == "__main__":
    unittest.main()
