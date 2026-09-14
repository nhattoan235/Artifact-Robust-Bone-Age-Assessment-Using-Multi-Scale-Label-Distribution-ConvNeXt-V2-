from __future__ import annotations

import unittest

import pandas as pd

from c3_roi.evaluate_artifact_matrix import build_cases, summarize_matrix


class ArtifactMatrixTests(unittest.TestCase):
    def test_cases_have_one_clean_reference(self):
        cases = build_cases(["noise", "blur"], [0.5, 1.0])
        self.assertEqual(cases[0], ("clean", 0.0, "clean"))
        self.assertEqual(len(cases), 5)
        with self.assertRaises(ValueError):
            build_cases(["noise"], [1.5])

    def test_summary_uses_paired_candidate_minus_baseline_delta(self):
        frame = pd.DataFrame({
            "image_id": ["a", "b", "a", "b"],
            "sex": ["F", "M", "F", "M"],
            "target_months": [30.0, 90.0, 30.0, 90.0],
            "profile": ["clean", "clean", "noise", "noise"],
            "severity": [0.0, 0.0, 1.0, 1.0],
            "case": ["clean", "clean", "noise_s1.00", "noise_s1.00"],
            "base_prediction": [32.0, 92.0, 34.0, 94.0],
            "c_prediction": [31.0, 91.0, 32.0, 92.0],
        })
        summary, subgroups = summarize_matrix(
            frame, weight_c=0.75, repetitions=200, seed=7,
        )
        noise = summary[summary["case"] == "noise_s1.00"].iloc[0]
        self.assertEqual(noise["base_mae"], 4.0)
        self.assertEqual(noise["c_mae"], 2.0)
        self.assertEqual(noise["c_delta"], -2.0)
        self.assertEqual(noise["blend_mae"], 2.5)
        self.assertEqual(set(subgroups["age_bin"]), {"0-59", "60-119"})


if __name__ == "__main__":
    unittest.main()
