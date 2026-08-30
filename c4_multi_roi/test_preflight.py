from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from c4_multi_roi.config import C4Config
from c4_multi_roi.preflight import preflight_report
from c4_multi_roi.schema import ROI_NAMES


class PreflightTests(unittest.TestCase):
    def test_preflight_passes_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("L", (10, 10), color=80).save(image)
            row = {
                "image_id": "1", "fold": "1", "bone_age_months": "100", "sex": "F",
                "global_path": str(image),
            }
            for name in ROI_NAMES:
                row[f"roi_{name}_path"] = str(image)
                row[f"roi_{name}_quality"] = "ok"
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            config = C4Config(manifest=str(manifest), expected_rows=1, image_root=".")
            report = preflight_report(config, check_all_files=True)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["checks"]["roi_files"], 6)


if __name__ == "__main__":
    unittest.main()
