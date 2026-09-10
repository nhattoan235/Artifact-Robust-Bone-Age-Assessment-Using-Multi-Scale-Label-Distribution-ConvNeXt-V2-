"""Build the small Colab code/config bundle for the B/C artifact pilots."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "C3_Z26_C3_ROI_V2_PILOT_BC_CODE_V1.zip"
P1_FILES = {
    "__init__.py", "config.py", "data.py", "metrics.py", "model.py",
    "preflight.py", "preprocessing.py", "train.py", "trainer.py", "artifacts.py",
}
CONFIG_FILES = {
    "C3_Z26_C3_ROI_V2_PILOT_B_FOLD_1_SEED_42.toml",
    "C3_Z26_C3_ROI_V2_PILOT_C_FOLD_1_SEED_42.toml",
}


def notebook() -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown", "metadata": {}, "outputs": [],
                "source": [
                    "# C3-Z26 C3-ROI — artifact pilots B/C\n",
                    "Chạy từng pilot một trên Fold 1, seed 42. B là augmentation-only; C thêm consistency loss.\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [
                    "PILOT = 'B'  # đổi thành 'C' sau khi B kết thúc\n",
                    "assert PILOT in {'B', 'C'}\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n",
                    "from pathlib import Path\n",
                    "import zipfile, shutil, subprocess, sys\n",
                    "DATA_DIR = Path('/content/drive/MyDrive/RSNA_DATA')\n",
                    "CODE_ZIP = DATA_DIR / 'C3_Z26_C3_ROI_V2_PILOT_BC_CODE_V1.zip'\n",
                    "assert CODE_ZIP.is_file(), f'Thiếu {CODE_ZIP}'\n",
                    "assert Path('/content/C3_Z26_C3_ROI_V2/manifests/fold_1_train.csv').is_file(), 'Thiếu C3-Z26 manifest trên /content'\n",
                    "assert Path('/content/C3_Z26_COMBO_V2').is_dir(), 'Thiếu ảnh C3_Z26_COMBO_V2 trên /content'\n",
                    "shutil.rmtree('/content/p1_baseline', ignore_errors=True)\n",
                    "shutil.rmtree('/content/C3_Z26_C3_ROI_V2_PILOTS', ignore_errors=True)\n",
                    "with zipfile.ZipFile(CODE_ZIP) as zf: zf.extractall('/content')\n",
                    "sys.path.insert(0, '/content')\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [
                    "cfg = Path(f'/content/C3_Z26_C3_ROI_V2_PILOTS/configs/C3_Z26_C3_ROI_V2_PILOT_{PILOT}_FOLD_1_SEED_42.toml')\n",
                    "subprocess.run([sys.executable, '-m', 'p1_baseline.preflight', '--config', str(cfg), '--no-pretrained'], check=True)\n",
                    "print(f'Preflight PASS: {cfg}')\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [
                    "# Chạy một pilot cho đến early-stop; output được tách riêng B/C và mirror sang Drive.\n",
                    "subprocess.run([sys.executable, '-u', '-m', 'p1_baseline.train', '--config', str(cfg)], check=True)\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [
                    "run_dir = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/runs') / f'C3_Z26_C3_ROI_V2_PILOT_{PILOT}_FOLD_1_SEED_42'\n",
                    "print((run_dir / 'train.log').read_text(encoding='utf-8')[-4000:])\n",
                    "print('Best checkpoint:', run_dir / 'best_mae.ckpt')\n",
                ],
            },
        ],
        "metadata": {
            "colab": {"name": "C3_Z26_C3_ROI_V2_PILOT_BC.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def readme() -> str:
    return """# C3-Z26 C3-ROI artifact pilots B/C

## Protocol

Both pilots use the exact existing C3-Z26 Fold 1 split, seed 42, ConvNeXt-Tiny
with LDL head, and clean validation MAE for early stopping. They do not read the
RSNA test set and do not run TTA during selection.

* **B**: paired mild artifact augmentation, supervised loss on clean and artifact views.
* **C**: B plus `consistency_weight = 0.30`, penalizing prediction disagreement
  between the two views.

The training batch is 18 with gradient accumulation 2, so the effective clean
sample count remains 36 while the clean/artifact pair is processed together.
Each run has a unique ID and writes to `C3_Z26_C3_ROI_V2_PILOTS`; it cannot
overwrite the existing baseline or the LDL/seed pilots.

## Colab

1. Upload this ZIP to `MyDrive/RSNA_DATA/`.
2. Open `C3_Z26_C3_ROI_V2_PILOT_BC.ipynb` from the ZIP after extracting it, or
   copy its cells into a fresh Colab runtime.
3. Set `PILOT = 'B'`, run all cells, and wait for `RUN STATUS`.
4. Start a fresh runtime, change `PILOT = 'C'`, and repeat.

The input paths must already exist in the runtime:
`/content/C3_Z26_C3_ROI_V2/manifests/` and `/content/C3_Z26_COMBO_V2/`.
Checkpoints and logs are mirrored to `MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs/`.

## Decision rule

Compare B and C against the locked Fold 1 baseline MAE 6.253462. Keep a pilot
only if clean validation MAE improves and paired bootstrap evaluation on a
separate artifact validation view does not show a degradation. Do not use the
200-image RSNA test to choose between B and C.
"""


def build_bundle(output_zip: Path = DEFAULT_OUTPUT) -> dict:
    with tempfile.TemporaryDirectory(prefix="c3_pilot_bc_") as temporary:
        stage = Path(temporary)
        p1_stage = stage / "p1_baseline"
        p1_stage.mkdir(parents=True)
        for name in sorted(P1_FILES):
            shutil.copy2(ROOT / "p1_baseline" / name, p1_stage / name)
        config_stage = stage / "C3_Z26_C3_ROI_V2_PILOTS" / "configs"
        config_stage.mkdir(parents=True)
        for name in sorted(CONFIG_FILES):
            shutil.copy2(ROOT / "c3_roi" / "pilot_configs" / name, config_stage / name)
        pilot_root = config_stage.parent
        (pilot_root / "C3_Z26_C3_ROI_V2_PILOT_BC.ipynb").write_text(
            json.dumps(notebook(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pilot_root / "README.md").write_text(readme(), encoding="utf-8")
        (stage / "requirements_colab.txt").write_text(
            "timm>=1.0\nnumpy>=1.24\nPillow>=10.0\n", encoding="utf-8"
        )
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        if output_zip.exists():
            output_zip.unlink()
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    with zipfile.ZipFile(output_zip) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Bundle ZIP hỏng tại {bad}")
        names = archive.namelist()
    return {"output_zip": str(output_zip), "bytes": output_zip.stat().st_size, "files": len(names), "sha256": _sha256(output_zip)}


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
