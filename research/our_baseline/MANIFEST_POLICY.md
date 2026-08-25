# Bundle manifest policy

## Included

- Python source, notebooks, requirements, TOML/YAML configuration.
- Markdown, text, JSON, JSONL, CSV, and log evidence.
- Split manifests and prediction files needed to audit OOF calculations.
- Model-package code, configuration, README, manifest, and evidence files.
- Project handoff and planning Markdown documents.

## Excluded

- `.ckpt`, `.pt`, `.pth`, `.bin`, and other model/pretrained weight files.
- Image datasets, test images, masks, and generated image collections.
- ZIP/RAR/7z archives and temporary downloads.
- Python caches and generated runtime files.
- Word/PDF artifacts and screenshots.
- Credentials such as `kaggle.json`, API keys, tokens, or `.env` files.

The excluded artifacts remain available in the local project folders and can be
attached separately as private Kaggle/Drive datasets when reproducing a run.
