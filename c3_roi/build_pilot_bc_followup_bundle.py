"""Build the Fold-5 artifact diagnosis and Pilot-B control Colab bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "c3_roi" / "outputs" / "C3_Z26_C3_ROI_V2_FOLLOWUP_FOLD5_V1" /
    "C3_Z26_C3_ROI_V2_FOLLOWUP_FOLD5_V1.zip"
)
P1_FILES = {
    "__init__.py", "config.py", "data.py", "metrics.py", "model.py",
    "preflight.py", "preprocessing.py", "train.py", "trainer.py", "artifacts.py",
}
PILOT_FILES = {
    "pilot_bc_runner.py", "evaluate_pilot_bc.py", "evaluate_artifact_matrix.py",
    "compare_pilot_b_c_fold.py",
}
CONFIG_FILES = {
    "C3_Z26_C3_ROI_V2_PILOT_B_FOLD_5_SEED_42.toml",
    "C3_Z26_C3_ROI_V2_PILOT_C_FOLD_5_SEED_42.toml",
}


def _notebook() -> dict[str, Any]:
    cells = [
        {
            "cell_type": "markdown", "metadata": {}, "outputs": [],
            "source": [
                "# C3-ROI V2 — tuần tự Fold 5: artifact matrix rồi Pilot B\n",
                "Không đọc test 200 ảnh. Chạy từng cell theo thứ tự; chỉ train ở bước 3.\n",
            ],
        },
        {
            "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "from pathlib import Path\n",
                "import json, shutil, subprocess, sys, zipfile\n",
                "DATA_DIR = Path('/content/drive/MyDrive/RSNA_DATA')\n",
                "MAIN_ZIP = DATA_DIR / 'C3_Z26_C3_ROI_T4_CODE_V2.zip'\n",
                "DATA_ZIP = DATA_DIR / 'C3_Z26_COMBO_V2_FINAL.zip'\n",
                "FOLLOWUP_ZIP = DATA_DIR / 'C3_Z26_C3_ROI_V2_FOLLOWUP_FOLD5_V1.zip'\n",
                "for p in (MAIN_ZIP, DATA_ZIP, FOLLOWUP_ZIP): assert p.is_file(), f'Thiếu {p}'\n",
                "if not Path('/content/C3_Z26_C3_ROI_V2/manifests/fold_5_train.csv').is_file():\n",
                "    with zipfile.ZipFile(MAIN_ZIP) as zf: zf.extractall('/content')\n",
                "if not Path('/content/C3_Z26_COMBO_V2').is_dir():\n",
                "    with zipfile.ZipFile(DATA_ZIP) as zf: zf.extractall('/content')\n",
                "shutil.rmtree('/content/p1_baseline', ignore_errors=True)\n",
                "shutil.rmtree('/content/C3_Z26_C3_ROI_V2_PILOTS', ignore_errors=True)\n",
                "with zipfile.ZipFile(FOLLOWUP_ZIP) as zf: zf.extractall('/content')\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', '/content/requirements_colab.txt'], check=True)\n",
                "assert Path('/content/C3_Z26_C3_ROI_V2/manifests/fold_5_validation.csv').is_file()\n",
                "assert Path('/content/C3_Z26_COMBO_V2').is_dir()\n",
                "print('SETUP PASS')\n",
            ],
        },
        {
            "cell_type": "markdown", "metadata": {}, "outputs": [],
            "source": [
                "## Bước 2 — tách artifact theo loại trên Fold 5\n",
                "Chỉ inference baseline và Pilot C. Thường khoảng 45–70 phút trên T4.\n",
            ],
        },
        {
            "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [
                "matrix = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/evaluate_artifact_matrix.py')\n",
                "subprocess.run([sys.executable, '-u', str(matrix), '--fold', '5', '--image-batch-size', '1', '--bootstrap-repetitions', '20000'], check=True)\n",
            ],
        },
        {
            "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [
                "import pandas as pd\n",
                "MATRIX_DIR = DATA_DIR / 'C3_Z26_C3_ROI_V2_PILOTS/OOF/artifact_matrix_fold5'\n",
                "matrix_summary = pd.read_csv(MATRIX_DIR / 'artifact_matrix_summary.csv')\n",
                "display(matrix_summary.sort_values('blend_delta'))\n",
                "assert (MATRIX_DIR / 'artifact_matrix_report.json').is_file()\n",
                "print('ARTIFACT MATRIX SAVED')\n",
            ],
        },
        {
            "cell_type": "markdown", "metadata": {}, "outputs": [],
            "source": [
                "## Bước 3 — train đối chứng Pilot B Fold 5\n",
                "B và C giống hệt nhau trừ consistency weight: B=0.00, C=0.30.\n",
            ],
        },
        {
            "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [
                "runner = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/pilot_bc_runner.py')\n",
                "subprocess.run([sys.executable, '-u', str(runner), '--pilot', 'B', '--fold', '5'], check=True)\n",
            ],
        },
        {
            "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [
                "evaluator = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/evaluate_pilot_bc.py')\n",
                "subprocess.run([sys.executable, '-u', str(evaluator), '--pilot', 'B', '--fold', '5', '--bootstrap-repetitions', '20000'], check=True)\n",
            ],
        },
        {
            "cell_type": "markdown", "metadata": {}, "outputs": [],
            "source": ["## Bước 4 — paired bootstrap B/C/baseline và quyết định\n"],
        },
        {
            "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [
                "compare = Path('/content/C3_Z26_C3_ROI_V2_PILOTS/compare_pilot_b_c_fold.py')\n",
                "subprocess.run([sys.executable, '-u', str(compare), '--fold', '5', '--bootstrap-repetitions', '20000'], check=True)\n",
            ],
        },
        {
            "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [
                "RUN = DATA_DIR / 'C3_Z26_C3_ROI_V2_PILOTS/runs/C3_Z26_C3_ROI_V2_PILOT_B_V2_FOLD_5_SEED_42'\n",
                "required = [RUN/'best_mae.ckpt', RUN/'artifact_robustness_predictions.csv', RUN/'artifact_robustness_report.json', RUN/'comparison_vs_c_and_baseline/pilot_b_c_baseline_fold_5_report.json']\n",
                "for p in required: print(p.name, p.is_file())\n",
                "assert all(p.is_file() for p in required)\n",
                "state = json.loads((RUN/'run_state.json').read_text())\n",
                "assert state['status'] in {'early_stopped', 'completed'}\n",
                "print('FOLLOW-UP COMPLETE — SAFE TO DISCONNECT')\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "C3_Z26_C3_ROI_V2_FOLLOWUP_FOLD5.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    notebook_output = output.with_name("C3_Z26_C3_ROI_V2_FOLLOWUP_FOLD5.ipynb")
    with tempfile.TemporaryDirectory(prefix="c3_followup_f5_") as temporary:
        stage = Path(temporary)
        p1_stage = stage / "p1_baseline"
        p1_stage.mkdir(parents=True)
        for name in sorted(P1_FILES):
            shutil.copy2(ROOT / "p1_baseline" / name, p1_stage / name)
        pilot_stage = stage / "C3_Z26_C3_ROI_V2_PILOTS"
        config_stage = pilot_stage / "configs"
        config_stage.mkdir(parents=True)
        for name in sorted(CONFIG_FILES):
            shutil.copy2(ROOT / "c3_roi" / "pilot_configs" / name, config_stage / name)
        for name in sorted(PILOT_FILES):
            shutil.copy2(ROOT / "c3_roi" / name, pilot_stage / name)
        notebook_path = pilot_stage / "C3_Z26_C3_ROI_V2_FOLLOWUP_FOLD5.ipynb"
        notebook_path.write_text(
            json.dumps(_notebook(), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (stage / "requirements_colab.txt").write_text(
            "numpy>=1.24\npandas>=2.0\nPillow>=10.0\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notebook_path, notebook_output)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP hong tai {bad}")
        names = archive.namelist()
    result = {
        "output": str(output),
        "notebook": str(notebook_output),
        "files": len(names),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }
    output.with_suffix(output.suffix + ".sha256.txt").write_text(
        f"{result['sha256']}  {output.name}\n", encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
