import unittest

from c3_roi_attention.prepare_configs import make_attention_config


class AttentionConfigTests(unittest.TestCase):
    def test_only_experiment_identity_and_architecture_are_changed(self):
        source = """[run]
run_id = \"C3_ROI_V1_FOLD_3\"
output_root = \"c3_roi/runs/C3_ROI_V1\"
seed = 42

[model]
architecture = \"convnext_tiny\"
image_size = 512
"""
        actual = make_attention_config(source, fold=3)
        expected = """[run]
run_id = \"C3_ROI_ATTN_V1_FOLD_3\"
output_root = \"c3_roi_attention/runs/C3_ROI_ATTN_V1\"
seed = 42

[model]
architecture = \"convnext_tiny_spatial_attention\"
image_size = 512
"""
        self.assertEqual(actual, expected)

    def test_rejects_source_from_the_wrong_fold(self):
        source = 'run_id = "C3_ROI_V1_FOLD_1"\n'
        with self.assertRaises(ValueError):
            make_attention_config(source, fold=2)


if __name__ == "__main__":
    unittest.main()
