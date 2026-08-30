from __future__ import annotations

import unittest

from c4_multi_roi.build_colab_notebooks import build_notebook


class ColabNotebookTests(unittest.TestCase):
    def test_notebook_hardcodes_requested_fold_and_archives(self) -> None:
        notebook = build_notebook(4)
        source = "".join(notebook["cells"][1]["source"])
        self.assertIn("FOLD = 4", source)
        self.assertIn("C4_MULTI_ROI_V1_DATA.zip", source)
        self.assertIn("C4_MULTI_ROI_COLAB_CODE.zip", source)
        self.assertIn("data_dev_v1.zip", source)
        self.assertIn("c4_multi_roi.colab_runner", source)
        compile(source, "colab_cell.py", "exec")

    def test_notebook_rejects_invalid_fold(self) -> None:
        with self.assertRaises(ValueError):
            build_notebook(0)


if __name__ == "__main__":
    unittest.main()
