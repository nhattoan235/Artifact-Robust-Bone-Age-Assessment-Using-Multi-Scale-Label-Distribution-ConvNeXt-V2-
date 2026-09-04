from __future__ import annotations

import unittest

from c4_multi_roi.build_handoff import code_member_paths


class BuildHandoffTests(unittest.TestCase):
    def test_code_members_include_runner_and_exclude_tests_and_cache(self) -> None:
        members = {path.as_posix() for path in code_member_paths()}
        self.assertIn("c4_multi_roi/colab_runner.py", members)
        self.assertIn("c4_multi_roi/config.py", members)
        self.assertFalse(any("test_" in member for member in members))
        self.assertFalse(any("/cache/" in member for member in members))


if __name__ == "__main__":
    unittest.main()
