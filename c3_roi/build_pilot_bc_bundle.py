"""Build the small Colab code/config bundle for the B/C artifact pilots."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "C3_Z26_C3_ROI_V2_PILOT_BC_CODE_V2.zip"
P1_FILES = {
    "__init__.py", "config.py", "data.py", "metrics.py", "model.py",
    "preflight.py", "preprocessing.py", "train.py", "trainer.py", "artifacts.py",
}
CONFIG_FILES = {
    "C3_Z26_C3_ROI_V2_PILOT_B_FOLD_1_SEED_42.toml",
    "C3_Z26_C3_ROI_V2_PILOT_C_FOLD_1_SEED_42.toml",
}
PILOT_SCRIPTS = {"pilot_bc_runner.py", "evaluate_pilot_bc.py"}


def notebook() -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown", "metadata": {}, "outputs": [],
                "source": [
                    "# C3-Z26 C3-ROI V2 — artifact pilots B/C\n",
                    "Fold 1, seed 42, direct regression. B là artifact augmentation; C thêm consistency loss.\n",
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
                    "MAIN_CODE_ZIP = DATA_DIR / 'C3_Z26_C3_ROI_T4_CODE_V2.zip'\n",
                    "DATA_ZIP = DATA_DIR / 'C3_Z26_COMBO_V2_FINAL.zip'\n",
                    "PILOT_ZIP = DATA_DIR / 'C3_Z26_C3_ROI_V2_PILOT_BC_CODE_V2.zip'\n",
                    "for path in (MAIN_CODE_ZIP, DATA_ZIP, PILOT_ZIP):\n",
                    "    assert path.is_file(), f'Thiếu {path}'\n",
                    "if not Path('/content/C3_Z26_C3_ROI_V2/manifests/fold_1_train.csv').is_file():\n",
                    "    print('Extract main C3-ROI V2 code...')\n",
                    "    with zipfile.ZipFile(MAIN_CODE_ZIP) as zf: zf.extractall('/content')\n",
                    "if not Path('/content/C3_Z26_COMBO_V2').is_dir():\n",
                    "    print('Extract C3-Z26 image data...')\n",
                    "    with zipfile.ZipFile(DATA_ZIP) as zf: zf.extractall('/content')\n",
                    "shutil.rmtree('/content/p1_baseline', ignore_errors=True)\n",
                    "with zipfile.ZipFile(PILOT_ZIP) as zf: zf.extractall('/content')\n",
                    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', '/content/requirements_colab.txt'], check=True)\n",
                    "sys.path.insert(0, '/content')\n",
                    "assert Path('/content/C3_Z26_C3_ROI_V2/manifests/fold_1_train.csv').is_file()\n",
                    "assert Path('/content/C3_Z26_COMBO_V2').is_dir()\n",
                    "print('SETUP PASS')\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [
                    "# Runner tự preflight, chọn checkpoint tương thích và resume từ Drive nếu Colab ngắt.\n",
                    "runner = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/pilot_bc_runner.py')\n",
                    "subprocess.run([sys.executable, '-u', str(runner), '--pilot', PILOT], check=True)\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [
                    "# Đánh giá paired clean/artifact trên Fold 1; không đọc tập test 200 ảnh.\n",
                    "evaluator = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/evaluate_pilot_bc.py')\n",
                    "subprocess.run([sys.executable, '-u', str(evaluator), '--pilot', PILOT], check=True)\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [
                    "run_id = f'C3_Z26_C3_ROI_V2_PILOT_{PILOT}_V2_FOLD_1_SEED_42'\n",
                    "run_dir = DATA_DIR / 'C3_Z26_C3_ROI_V2_PILOTS' / 'runs' / run_id\n",
                    "print((run_dir / 'train.log').read_text(encoding='utf-8')[-4000:])\n",
                    "print((run_dir / 'artifact_robustness_report.json').read_text(encoding='utf-8'))\n",
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
direct regression, and clean validation MAE for early stopping. They do not read the
RSNA test set and do not run TTA during selection.

* **B**: epochs 1–3 clean, then a 5-epoch ramp into paired mild artifact
  augmentation with supervised loss on clean and artifact views.
* **C**: the same schedule plus `consistency_weight = 0.30`, penalizing prediction disagreement
  between the two views.

The training batch is 18 with gradient accumulation 2, so the effective clean
sample count remains 36 while the clean/artifact pair is processed together.
Each run has a unique ID and writes to `C3_Z26_C3_ROI_V2_PILOTS`; it cannot
overwrite the existing baseline or the LDL/seed pilots.

## Colab

1. Keep these three ZIPs in `MyDrive/RSNA_DATA/`:
   `C3_Z26_C3_ROI_T4_CODE_V2.zip`, `C3_Z26_COMBO_V2_FINAL.zip`, and
   `C3_Z26_C3_ROI_V2_PILOT_BC_CODE_V2.zip`.
2. Open `C3_Z26_C3_ROI_V2_PILOT_BC.ipynb` from the ZIP after extracting it, or
   copy its cells into a fresh Colab runtime.
3. Set `PILOT = 'B'`, run all cells, and wait for `RUN STATUS`.
4. Start a fresh runtime, change `PILOT = 'C'`, and repeat.

The notebook extracts the main code and image ZIPs when their runtime folders
do not exist, then installs dependencies and overlays the V2 pilot code.
Checkpoints and logs are mirrored to `MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs/`.
Rerunning the same pilot resumes only a checkpoint whose config, manifests, and
code hash all match. An incompatible checkpoint stops the run instead of being overwritten.

## Decision rule

Compare B and C against the locked Fold 1 baseline MAE 6.253462. Keep a pilot
only if clean validation MAE improves and paired bootstrap evaluation on a
separate artifact validation view does not show a degradation. Do not use the
200-image RSNA test to choose between B and C.
"""


def build_bundle(output_zip: Path = DEFAULT_OUTPUT) -> dict:
    standalone_notebook = output_zip.with_name("C3_Z26_C3_ROI_V2_PILOT_BC.ipynb")
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
        for name in sorted(PILOT_SCRIPTS):
            shutil.copy2(ROOT / "c3_roi" / name, pilot_root / name)
        (pilot_root / "C3_Z26_C3_ROI_V2_PILOT_BC.ipynb").write_text(
            json.dumps(notebook(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pilot_root / "README.md").write_text(readme(), encoding="utf-8")
        (stage / "requirements_colab.txt").write_text(
            "timm>=1.0\nnumpy>=1.24\nPillow>=10.0\n", encoding="utf-8"
        )
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            pilot_root / "C3_Z26_C3_ROI_V2_PILOT_BC.ipynb", standalone_notebook
        )
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
    return {
        "output_zip": str(output_zip),
        "notebook": str(standalone_notebook),
        "bytes": output_zip.stat().st_size,
        "files": len(names),
        "sha256": _sha256(output_zip),
    }


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
