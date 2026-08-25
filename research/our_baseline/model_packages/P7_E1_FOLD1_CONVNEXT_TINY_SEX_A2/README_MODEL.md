# P7_E1_FOLD1_CONVNEXT_TINY_SEX_A2

## Model

- Technique: ConvNeXt-Tiny 512 + pad_square + light augmentation + sex embedding + LDL/regression configuration
- Evaluation: P7 fold 1 OOF validation
- MAE: **6.298709880252849 months**
- Status: `OFFICIAL_P7_FOLD_MODEL`
- Original checkpoint: `D:\do_an_tot_nghiep\project\p7_results\results\P7_FINAL_V3_FOLD_1\best_model.pt`
- Checkpoint SHA-256: `fab81f6dcda6ba0adc3cbd861cc89ebd5f7f91e8fd18e1cc2fb6c4a9db1a93ab`
- Checkpoint bytes: 112162635
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
