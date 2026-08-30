from __future__ import annotations

import argparse
import json
from pathlib import Path


CELL_TEMPLATE = r'''from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

FOLD = {fold}
DRIVE_DATA = Path('/content/drive/MyDrive/data')
WORKSPACE = Path('/content/c4_multi_roi_workspace')
GLOBAL_ZIP = DRIVE_DATA / 'data_dev_v1.zip'
ROI_ZIP = DRIVE_DATA / 'C4_MULTI_ROI_V1_DATA.zip'
CODE_ZIP = DRIVE_DATA / 'C4_MULTI_ROI_COLAB_CODE.zip'

for required in (GLOBAL_ZIP, ROI_ZIP, CODE_ZIP):
    assert required.is_file(), f'Không tìm thấy file trên Drive: {{required}}'
assert FOLD in range(1, 6)

if WORKSPACE.exists():
    assert str(WORKSPACE) == '/content/c4_multi_roi_workspace'
    shutil.rmtree(WORKSPACE)
WORKSPACE.mkdir(parents=True)

def extract_zip(path: Path, destination: Path) -> None:
    print(f'Đang giải nén {{path.name}} ...', flush=True)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for index, member in enumerate(members, start=1):
            archive.extract(member, destination)
            if index % 5000 == 0 or index == len(members):
                print(f'  {{index}}/{{len(members)}} files', flush=True)

extract_zip(CODE_ZIP, WORKSPACE)
extract_zip(ROI_ZIP, WORKSPACE)
extract_zip(GLOBAL_ZIP, WORKSPACE)

def ensure_expected_directory(expected: Path, marker: str) -> None:
    if expected.is_dir():
        return
    candidates = [p.parent for p in WORKSPACE.rglob(marker) if p.is_file()]
    assert candidates, f'Không tìm thấy thư mục chứa {{marker}} sau giải nén'
    expected.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(candidates[0], expected, target_is_directory=True)
    print(f'Linked {{expected}} -> {{candidates[0]}}')

ensure_expected_directory(
    WORKSPACE / 'data/goc/boneage-training-dataset/boneage-training-dataset',
    '1377.png',
)

validation_expected = WORKSPACE / 'data/rsna_official_validation/images'
if not validation_expected.is_dir():
    validation_candidates = [
        path for path in WORKSPACE.rglob('images')
        if path.is_dir() and (path / '10018.png').is_file()
    ]
    assert validation_candidates, 'Không tìm thấy ảnh official validation sau giải nén'
    validation_expected.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(validation_candidates[0], validation_expected, target_is_directory=True)

os.chdir(WORKSPACE)
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '-r', 'c4_multi_roi/requirements_colab.txt'],
    check=True,
)

roi_root = WORKSPACE / 'c4_multi_roi/cache/C4_MULTI_ROI_V1/roi'
roi_count = sum(1 for path in roi_root.rglob('*.jpg'))
train_count = sum(1 for _ in (WORKSPACE / 'data/goc/boneage-training-dataset/boneage-training-dataset').glob('*.png'))
validation_count = sum(1 for _ in validation_expected.glob('*.png'))
print({{'fold': FOLD, 'roi_images': roi_count, 'global_images': train_count + validation_count}})
assert roi_count == 84216, f'Cần 84216 ROI, hiện có {{roi_count}}'
assert train_count + validation_count == 14036, (
    f'Cần 14036 ảnh global, hiện có {{train_count + validation_count}}'
)

subprocess.run(
    [sys.executable, '-u', '-m', 'c4_multi_roi.colab_runner', '--fold', str(FOLD)],
    cwd=WORKSPACE,
    check=True,
)
'''


def build_notebook(fold: int) -> dict:
    if fold not in range(1, 6):
        raise ValueError("fold must be 1..5")
    source = CELL_TEMPLATE.format(fold=fold)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": f"C4_MULTI_ROI_FOLD_{fold}.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# C4 Global + Six ROI — Fold {fold}\n",
                    "Bật T4 GPU, sau đó chọn **Runtime → Run all**. Không sửa cell.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("c4_multi_roi/notebooks"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        path = args.output / f"C4_MULTI_ROI_COLAB_FOLD_{fold}.ipynb"
        path.write_text(json.dumps(build_notebook(fold), ensure_ascii=False, indent=1), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
