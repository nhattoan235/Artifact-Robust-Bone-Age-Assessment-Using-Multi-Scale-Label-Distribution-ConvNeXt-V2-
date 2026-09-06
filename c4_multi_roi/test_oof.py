from __future__ import annotations

import unittest
from pathlib import Path

from c4_multi_roi.oof import (
    LOCKED_MANIFEST_SHA256,
    expected_oof_rows,
    load_locked_manifest,
    resolve_fold_artifacts,
    select_validation_rows,
)


WORKSPACE = Path(__file__).resolve().parents[1]


class C4OOFInputTests(unittest.TestCase):
    def test_locked_manifest_has_exact_five_fold_partition(self) -> None:
        rows = load_locked_manifest(WORKSPACE)
        self.assertEqual(len(rows), 14024)
        self.assertEqual(expected_oof_rows(rows), 14024)
        self.assertEqual(
            {fold: len(select_validation_rows(rows, fold)) for fold in range(1, 6)},
            {1: 2805, 2: 2805, 3: 2804, 4: 2805, 5: 2805},
        )
        self.assertEqual(LOCKED_MANIFEST_SHA256, "d56634f084163bd360a3b43847958869fdc5fa3dcf19d1aae1836f441e612b92")

    def test_each_fold_resolves_to_selected_best_checkpoint(self) -> None:
        inventory = WORKSPACE / "data/model_eval_inventory/C4_MULTI_ROI_V2_5FOLD_SELECTED_20260903"
        for fold in range(1, 6):
            checkpoint, config = resolve_fold_artifacts(inventory, fold)
            self.assertEqual(checkpoint.name, "best_mae.ckpt")
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(config.name, "fold_%d_colab.toml" % fold)
            self.assertTrue(config.is_file())

    def test_validation_selection_does_not_touch_test_records(self) -> None:
        rows = load_locked_manifest(WORKSPACE)
        for fold in range(1, 6):
            selected = select_validation_rows(rows, fold)
            self.assertTrue(all(row["source_split"] != "test" for row in selected))
            self.assertTrue(all(int(row["fold"]) == fold for row in selected))


if __name__ == "__main__":
    unittest.main()
