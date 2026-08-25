# Consolidated research bundle — bone-age baseline

This directory consolidates the work carried out for the bone-age assessment
baseline and its improvement experiments. It is intended to make the work
reviewable and reproducible inside the shared GitHub repository.

## Scope

The project predicts bone age in months from RSNA pediatric hand radiographs.
The near-term benchmark was the group model's exploratory MAE of about 4.73 months
on the labeled 200-image RSNA reference set. The final research target is MAE below
3.5 months.

The 200-image set was used only for exploratory measurement after methods were
developed. It was not used to select training hyperparameters, checkpoints, TTA,
or blend weights. The selection gate for the later LDL work is the 14,036-row OOF
MAE of 6.120448.

## Layout

```text
research/our_baseline/
├── baseline_v1/       # baseline code, scripts, configs, logs, reports, predictions
├── model_packages/    # reproducibility metadata/evidence/code; model weights omitted
├── context_docs/      # project handoff and planning documents
├── RESULTS_SUMMARY.md # consolidated methods and verified metrics
├── EXPERIMENT_REGISTRY.md
├── REFERENCES_IEEE.md
└── MANIFEST_POLICY.md
```

## Main methods

1. ConvNeXt-Tiny pretrained on ImageNet with sex embedding and direct regression.
2. P7-style preprocessing: square padding, bicubic resize to 512×512, grayscale
   replicated to RGB, and ImageNet normalization.
3. Five-fold stratified OOF evaluation over 14,036 development images.
4. Test-time augmentation using rotations of −10°, −5°, 0°, 5°, 10° with and
   without horizontal flip.
5. Fixed 50/50 prediction blending selected from OOF evidence.
6. Label Distribution Learning (LDL) roadmap using a ConvNeXt-Tiny LDL head;
   its five-fold training was prepared but must beat the OOF gate before being
   considered a valid improvement.

## Reproduction entry points

The code is under `baseline_v1/`:

- `boneage_baseline.py`: original standalone baseline.
- `scripts/prepare_exp006_p7_5fold.py`: prepare the five-fold P7 protocol.
- `scripts/merge_exp006_oof.py`: merge held-out fold predictions.
- `scripts/evaluate_exp006_tta_oof.py`: evaluate TTA on OOF predictions.
- `scripts/blend_exp006_tta_p7_oof.py`: calculate the fixed OOF blend.
- `scripts/prepare_exp009_ldl_roadmap.py`: prepare the LDL experiment.

Most configuration files and manifests are under `baseline_v1/outputs/`.
They may contain historical absolute paths; replace those paths for the target
machine or Kaggle/Colab runtime before execution.

## Excluded artifacts

Weights (`.ckpt`, `.pt`, `.pth`), image datasets, archives, runtime caches, and
secrets are deliberately excluded. See [MANIFEST_POLICY.md](MANIFEST_POLICY.md).
