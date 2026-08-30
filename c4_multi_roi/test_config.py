from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from c4_multi_roi.config import C4Config, load_config, scientific_config_hash


class C4ConfigTests(unittest.TestCase):
    def test_output_root_does_not_change_scientific_hash(self) -> None:
        config = C4Config(manifest="manifest.csv")
        changed = replace(config, output_root="another/output")
        self.assertEqual(scientific_config_hash(config), scientific_config_hash(changed))

    def test_view_mode_changes_scientific_hash(self) -> None:
        config = C4Config(manifest="manifest.csv")
        changed = replace(config, view_mode="six_roi_only")
        self.assertNotEqual(scientific_config_hash(config), scientific_config_hash(changed))

    def test_load_config_rejects_unknown_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text('[data]\nmanifest="x.csv"\nunknown=1\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
