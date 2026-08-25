# P7_E1_FOLD5_CONVNEXT_TINY_SEX_A2

## Model

- Technique: ConvNeXt-Tiny 512 + pad_square + light augmentation + sex embedding + LDL/regression configuration
- Evaluation: P7 fold 5 OOF validation
- MAE: **6.327675379422537 months**
- Status: `OFFICIAL_P7_FOLD_MODEL`
- Original checkpoint: `D:\do_an_tot_nghiep\project\p7_results\results\P7_FINAL_V3_FOLD_5\best_model.pt`
- Checkpoint SHA-256: `d83f3fb6aba843cdc4ff04438d7b51cb0112a665a4253127b1f38b8afdea190b`
- Checkpoint bytes: 112155595
- PyTorch load audit: `PASS`
- State-dict location: `model`
- Tensor count: `186`
- Parameter count: `28021377`

## Interpretation

One fold of the five-fold P7 ensemble. Use all five packages for equal-weight ensemble or TTA ensemble.

## Package layout

- `model/`: unchanged final checkpoint.
- `config/`: configuration or data manifest used by the run.
- `code/`: matching model, data, training and inference implementation.
- `evidence/`: metrics, reports, manifests and audit files.
- `manifest.json`: package metadata and checkpoint inspection.

## Loading note

Load the checkpoint with the matching code and configuration. Start on CPU:

```python
checkpoint = torch.load("model/best_model.pt", map_location="cpu", weights_only=False)
```

Do not assume `.pt` and `.ckpt` packages have the same top-level layout.

## Scientific-use note

- Keep preprocessing, image normalization, target normalization, sex handling and architecture identical to `config/`.
- Do not use the labeled 200-image test set to select checkpoints, hyperparameters or blend weights.
- Re-evaluate any modification on validation or OOF data first.
