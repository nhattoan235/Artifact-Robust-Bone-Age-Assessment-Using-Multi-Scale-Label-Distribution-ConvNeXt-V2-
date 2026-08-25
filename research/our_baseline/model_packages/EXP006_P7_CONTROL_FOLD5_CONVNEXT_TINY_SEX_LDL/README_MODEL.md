# EXP006_P7_CONTROL_FOLD5_CONVNEXT_TINY_SEX_LDL

## Model

- Technique: ConvNeXt-Tiny 512 + pad_square + light augmentation + sex embedding + LDL/regression control
- Evaluation: EXP006 pooled OOF, fold 5 contribution
- MAE: **6.310028500178126 months**
- Status: `EXP006_P7_CONTROL_FOLD_MODEL`
- Original checkpoint: `D:\do_an_tot_nghiep\project\baseline_v1\outputs\exp006_p7_tta\exp006_p7_tta\runs\EXP006_P7_CONTROL_FOLD_5\best_mae.ckpt`
- Checkpoint SHA-256: `4ffaf663f26096d6a059652bcffe9db4867d798edb006c38ad19df30c2b0b08d`
- Checkpoint bytes: 336504463
- PyTorch load audit: `PASS`
- State-dict location: `model`
- Tensor count: `186`
- Parameter count: `28021377`

## Interpretation

One of five fold checkpoints used for EXP006 pooled OOF MAE 6.323629 and P7-compatible TTA MAE 6.296203.

## Package layout

- `model/`: unchanged final checkpoint.
- `config/`: configuration or data manifest used by the run.
- `code/`: matching model, data, training and inference implementation.
- `evidence/`: metrics, reports, manifests and audit files.
- `manifest.json`: package metadata and checkpoint inspection.

## Loading note

Load the checkpoint with the matching code and configuration. Start on CPU:

```python
checkpoint = torch.load("model/best_mae.ckpt", map_location="cpu", weights_only=False)
```

Do not assume `.pt` and `.ckpt` packages have the same top-level layout.

## Scientific-use note

- Keep preprocessing, image normalization, target normalization, sex handling and architecture identical to `config/`.
- Do not use the labeled 200-image test set to select checkpoints, hyperparameters or blend weights.
- Re-evaluate any modification on validation or OOF data first.
