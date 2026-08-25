# OFFICIAL_P7_REFERENCE_CONVNEXT_TINY_SEX

## Model

- Technique: p7_reference recipe: ConvNeXt-Tiny 512 + sex feature + P7 reference preprocessing
- Evaluation: Official train/validation split (1,425 validation images)
- MAE: **6.4711432456970215 months**
- Status: `OFFICIAL_SINGLE_SPLIT_MODEL`
- Original checkpoint: `D:\do_an_tot_nghiep\project\baseline_v1\outputs\official_gpu\official\best.pt`
- Checkpoint SHA-256: `a5f03c8b0a52cf252f5ce0d658262e2857d98e663b1e98fba19cecf38089b77c`
- Checkpoint bytes: 112180615
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
