# EXP004_FRIEND_P7_FRESH_HOLDOUT_CONVNEXT_TINY_SEX

## Model

- Technique: ConvNeXt-Tiny 512 + pad_square + light augmentation + sex embedding + Smooth L1/LDL configuration
- Evaluation: Fresh 10% holdout from original training data
- MAE: **6.213795882936508 months**
- Status: `FRIEND_P7_FRESH_HOLDOUT_MODEL`
- Original checkpoint: `D:\do_an_tot_nghiep\project\baseline_v1\outputs\exp004_friend_holdout\checkpoint_mirror\checkpoint_mirror\EXP004_FRIEND_P7_FRESH_HOLDOUT_SEED42\best_mae.ckpt`
- Checkpoint SHA-256: `399e9dae02cf0ee39b33beae0613c5d2c04605d0539950e5adebe6bf392f8def`
- Checkpoint bytes: 336504203
- PyTorch load audit: `PASS`
- State-dict location: `model`
- Tensor count: `186`
- Parameter count: `28021377`

## Interpretation

Best checkpoint from the fresh holdout reproduction run. It is not directly comparable to five-fold pooled OOF without matching protocol.

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
