# Artifact-Robust Bone-Age Assessment

This repository contains the shared bone-age assessment research workspace.

## Contents

- `p1_baseline/`–`p13_reporting/`: the original shared research pipeline and reports.
- `c3_roi/`: ROI/local-global experiment code and recorded artifacts.
- `d3_oof/`: OOF, TTA, calibration, and final-test evaluation utilities.
- `research/our_baseline/`: consolidated code, configurations, evidence, and Markdown documentation produced by the other project member.

The consolidated bundle is the main entry point for reviewing the full experimental history:

- [Research bundle](research/our_baseline/README.md)
- [Results summary](research/our_baseline/RESULTS_SUMMARY.md)
- [Experiment registry](research/our_baseline/EXPERIMENT_REGISTRY.md)
- [IEEE references](research/our_baseline/REFERENCES_IEEE.md)

## Data and checkpoints

Image datasets, test labels, model checkpoints, pretrained weights, archives, and
runtime caches are intentionally not committed. They remain in the local/Kaggle/Drive
artifacts referenced by the audit files. The repository stores the code, manifests,
configuration, metrics, predictions, and provenance needed to review or reproduce the
experiments when the data and checkpoints are supplied separately.
