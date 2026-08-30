import unittest
from pathlib import Path

from c3_roi_attention.colab_runner import build_train_command


class ColabRunnerTests(unittest.TestCase):
    def test_fresh_run_does_not_request_missing_checkpoint(self):
        command = build_train_command("python", Path("fold_2_colab.toml"), None)
        self.assertEqual(command, ["python", "-m", "p1_baseline.train", "--config", "fold_2_colab.toml"])

    def test_resume_is_added_only_for_existing_checkpoint(self):
        command = build_train_command("python", Path("fold_2_colab.toml"), Path("last.ckpt"))
        self.assertEqual(command[-2:], ["--resume", "last.ckpt"])


if __name__ == "__main__":
    unittest.main()
