# OFFICIAL_A2_LIGHT_FLIP_CONVNEXT_TINY_SEX

## Model

- Technique: a2_light_flip recipe: ConvNeXt-Tiny 512 + sex feature + light flip augmentation
- Evaluation: Official train/validation split (1,425 validation images)
- MAE: **6.706370830535889 months**
- Status: `OFFICIAL_SINGLE_SPLIT_MODEL`
- Original checkpoint: `D:\do_an_tot_nghiep\project\baseline_v1\outputs\official_a2_light_flip_seed42\official_a2_light_flip_seed42\official\best.pt`
- Checkpoint SHA-256: `0240d29d47b3064030ce4d9504a8cbf20b0bef54895f61b9c9b8271335b78c27`
- Checkpoint bytes: 112180679
- PyTorch load audit: `PASS`
- State-dict location: `model`
- Tensor count: `188`
- Parameter count: `28027105`

## Interpretation

Official single-split baseline checkpoint. Use the recorded recipe and validation report when comparing against later experiments.

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
