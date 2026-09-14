"""Build the Colab bundle for locked Pilot C Fold 2--5 training and OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_CODE_V1.zip"
P1_FILES = {
    "__init__.py", "config.py", "data.py", "metrics.py", "model.py",
    "preflight.py", "preprocessing.py", "train.py", "trainer.py", "artifacts.py",
}
CONFIG_FILES = [
    f"C3_Z26_C3_ROI_V2_PILOT_C_FOLD_{fold}_SEED_42.toml"
    for fold in range(1, 6)
]
PILOT_FILES = {
    "pilot_bc_runner.py", "evaluate_pilot_bc.py", "evaluate_pilot_c_5fold.py",
}


def notebook() -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown", "metadata": {}, "outputs": [],
                "source": [
                    "# C3-Z26 C3-ROI V2 — Pilot C full 5-fold\n",
                    "Mỗi runtime chạy một fold. Fold 1 đã hoàn tất; dùng FOLD=2,3,4,5 lần lượt.\n",
                    "Pilot C = mild artifact augmentation + consistency loss 0.30.\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                "source": [
                    "FOLD = 2  # chỉ đổi thành 3, 4, 5 ở runtime kế tiếp\n",
                    "assert FOLD in {2, 3, 4, 5}\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                "source": [
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n",
                    "from pathlib import Path\n",
                    "import zipfile, shutil, subprocess, sys\n",
                    "DATA_DIR = Path('/content/drive/MyDrive/RSNA_DATA')\n",
                    "MAIN_CODE_ZIP = DATA_DIR / 'C3_Z26_C3_ROI_T4_CODE_V2.zip'\n",
                    "DATA_ZIP = DATA_DIR / 'C3_Z26_COMBO_V2_FINAL.zip'\n",
                    "PILOT_ZIP = DATA_DIR / 'C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_CODE_V1.zip'\n",
                    "for path in (MAIN_CODE_ZIP, DATA_ZIP, PILOT_ZIP):\n",
                    "    assert path.is_file(), f'Thiếu {path}'\n",
                    "if not Path('/content/C3_Z26_C3_ROI_V2/manifests/fold_2_train.csv').is_file():\n",
                    "    with zipfile.ZipFile(MAIN_CODE_ZIP) as zf: zf.extractall('/content')\n",
                    "if not Path('/content/C3_Z26_COMBO_V2').is_dir():\n",
                    "    with zipfile.ZipFile(DATA_ZIP) as zf: zf.extractall('/content')\n",
                    "shutil.rmtree('/content/p1_baseline', ignore_errors=True)\n",
                    "with zipfile.ZipFile(PILOT_ZIP) as zf: zf.extractall('/content')\n",
                    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', '/content/requirements_colab.txt'], check=True)\n",
                    "assert Path(f'/content/C3_Z26_C3_ROI_V2/manifests/fold_{FOLD}_train.csv').is_file()\n",
                    "assert Path('/content/C3_Z26_COMBO_V2').is_dir()\n",
                    "print('SETUP PASS — FOLD', FOLD)\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                "source": [
                    "# Có thể chạy lại cell này nếu Colab ngắt; runner chỉ resume checkpoint cùng fold.\n",
                    "runner = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/pilot_bc_runner.py')\n",
                    "subprocess.run([sys.executable, '-u', str(runner), '--pilot', 'C', '--fold', str(FOLD)], check=True)\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                "source": [
                    "evaluator = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/evaluate_pilot_bc.py')\n",
                    "subprocess.run([sys.executable, '-u', str(evaluator), '--pilot', 'C', '--fold', str(FOLD)], check=True)\n",
                    "run_id = f'C3_Z26_C3_ROI_V2_PILOT_C_V2_FOLD_{FOLD}_SEED_42'\n",
                    "run_dir = DATA_DIR / 'C3_Z26_C3_ROI_V2_PILOTS' / 'runs' / run_id\n",
                    "for name in ('best_mae.ckpt', 'run_state.json', 'train.log', 'artifact_robustness_predictions.csv', 'artifact_robustness_report.json'):\n",
                    "    assert (run_dir / name).is_file(), run_dir / name\n",
                    "print('FOLD SAVED:', run_dir)\n",
                ],
            },
            {
                "cell_type": "markdown", "metadata": {}, "outputs": [],
                "source": [
                    "## Sau khi Fold 2–5 đều có report\n",
                    "Chạy cell cuối trong một runtime T4 để ghép Fold 1–5 và inference baseline artifact nếu chưa cache.\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                "source": [
                    "aggregator = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/evaluate_pilot_c_5fold.py')\n",
                    "subprocess.run([sys.executable, '-u', str(aggregator), '--bootstrap-repetitions', '20000'], check=True)\n",
                ],
            },
        ],
        "metadata": {
            "colab": {"name": "C3_Z26_C3_ROI_V2_PILOT_C_5FOLD.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def readme() -> str:
    return """# C3-Z26 C3-ROI V2 — Pilot C full 5-fold

Fold 1 đã train và đánh giá. Bundle này chạy mới Fold 2–5, mỗi fold độc lập,
seed 42, cùng toàn bộ hyperparameter Pilot C Fold 1:

- ConvNeXt-Tiny 512, sex embedding, direct regression;
- light geometric/photometric augmentation;
- mild_v1 artifact augmentation, probability 1.0;
- consistency loss 0.30, warmup 3 epoch, ramp 5 epoch;
- clean validation MAE để early stop, patience 8.

## Colab

Giữ ba ZIP trong `MyDrive/RSNA_DATA/`:

```text
C3_Z26_C3_ROI_T4_CODE_V2.zip
C3_Z26_COMBO_V2_FINAL.zip
C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_CODE_V1.zip
```

Mở notebook, đặt `FOLD = 2`, bật T4 GPU rồi chạy các cell. Mỗi runtime chỉ chạy
một fold. Khi Colab ngắt, chạy lại cùng `FOLD`; runner tự resume checkpoint cùng
fold và từ chối checkpoint sai config/split/code. Sau khi evaluator in đủ các dòng
`SAVED`, có thể disconnect và chuyển sang fold kế tiếp.

## OOF aggregation

Sau khi có report của Fold 1–5, chạy cell aggregator. Nó ghép đúng 14.036 ID,
inference baseline clean/artifact trên cùng view, chạy paired bootstrap và lưu
prediction/report/subgroup vào `C3_Z26_C3_ROI_V2_PILOTS/OOF/`. Tập test 200 ảnh
không được đọc trong train, selection hoặc OOF aggregation.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(output_zip: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    standalone_notebook = output_zip.with_name("C3_Z26_C3_ROI_V2_PILOT_C_5FOLD.ipynb")
    with tempfile.TemporaryDirectory(prefix="c3_pilot_c_5fold_") as temporary:
        stage = Path(temporary)
        p1_stage = stage / "p1_baseline"
        p1_stage.mkdir(parents=True)
        for name in sorted(P1_FILES):
            shutil.copy2(ROOT / "p1_baseline" / name, p1_stage / name)
        pilot_stage = stage / "C3_Z26_C3_ROI_V2_PILOTS"
        config_stage = pilot_stage / "configs"
        config_stage.mkdir(parents=True)
        for name in CONFIG_FILES:
            shutil.copy2(ROOT / "c3_roi" / "pilot_configs" / name, config_stage / name)
        for name in sorted(PILOT_FILES):
            shutil.copy2(ROOT / "c3_roi" / name, pilot_stage / name)
        notebook_path = pilot_stage / "C3_Z26_C3_ROI_V2_PILOT_C_5FOLD.ipynb"
        notebook_path.write_text(
            json.dumps(notebook(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pilot_stage / "README.md").write_text(readme(), encoding="utf-8")
        (stage / "requirements_colab.txt").write_text(
            "timm>=1.0\nnumpy>=1.24\nPillow>=10.0\n", encoding="utf-8"
        )
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notebook_path, standalone_notebook)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_bundle(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    checksum = Path(str(args.output) + ".sha256.txt")
    checksum.write_text(f"{result['sha256']}  {args.output.name}\n", encoding="utf-8")
    print(f"checksum: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
