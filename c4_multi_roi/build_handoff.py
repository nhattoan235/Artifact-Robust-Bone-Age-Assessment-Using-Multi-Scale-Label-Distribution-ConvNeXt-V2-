from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from .cache_utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "handoff_packages/C4_MULTI_ROI_COLAB_READY"


PRODUCTION_FILES = (
    "__init__.py",
    "schema.py",
    "cache_utils.py",
    "config.py",
    "data.py",
    "model.py",
    "trainer.py",
    "train.py",
    "preflight.py",
    "colab_runner.py",
    "requirements_colab.txt",
    "protocol_v1.json",
)


def code_member_paths() -> list[Path]:
    paths = [Path("c4_multi_roi") / name for name in PRODUCTION_FILES]
    paths.extend(
        Path("c4_multi_roi/configs") / path.name
        for path in sorted((ROOT / "c4_multi_roi/configs").glob("fold_*_colab.toml"))
    )
    return paths


def _build_code_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in code_member_paths():
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, relative.as_posix())


def _build_data_zip(path: Path) -> None:
    cache_root = ROOT / "c4_multi_roi/cache/C4_MULTI_ROI_V1"
    manifest = cache_root / "manifest.csv"
    roi_files = sorted((cache_root / "roi").rglob("*.jpg"))
    if len(roi_files) != 84216:
        raise RuntimeError(f"C4 ROI cache incomplete: {len(roi_files)} != 84216")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(manifest, manifest.relative_to(ROOT).as_posix())
        for index, source in enumerate(roi_files, start=1):
            archive.write(source, source.relative_to(ROOT).as_posix())
            if index % 5000 == 0 or index == len(roi_files):
                print(f"packed ROI={index}/{len(roi_files)}", flush=True)


def _validate_zip(path: Path, *, expected_jpegs: int | None = None) -> dict:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    if bad is not None:
        raise RuntimeError(f"corrupt member in {path}: {bad}")
    jpeg_count = sum(name.lower().endswith(".jpg") for name in names)
    if expected_jpegs is not None and jpeg_count != expected_jpegs:
        raise RuntimeError(f"{path} JPEG count {jpeg_count} != {expected_jpegs}")
    return {"members": len(names), "jpeg_count": jpeg_count, "sha256": sha256_file(path)}


def _readme() -> str:
    return """# C4 Global + Six ROI — Colab ready

## Ba file ZIP phải có trong `MyDrive/data`

1. `data_dev_v1.zip` — dữ liệu global cũ đã có trên Drive.
2. `C4_MULTI_ROI_V1_DATA.zip` — 84.216 ảnh ROI và manifest.
3. `C4_MULTI_ROI_COLAB_CODE.zip` — code/config đã khóa.

## Chạy không sửa cell

1. Mở đúng notebook `C4_MULTI_ROI_COLAB_FOLD_X.ipynb`.
2. Chọn `Runtime > Change runtime type > T4 GPU`.
3. Chọn `Runtime > Run all`.

Notebook tự mount Drive, kiểm tra tên ZIP, giải nén, cài thư viện, kiểm tra đủ
14.036 global images và 84.216 ROI images, chạy preflight, tự resume checkpoint
phù hợp và train đúng fold. Không sửa `FOLD`, đường dẫn hoặc config.

Checkpoint được lưu tại:

`MyDrive/data/c4_multi_roi_runs/C4_MULTI_ROI_V1_FOLD_X/`

Nếu Colab ngắt, chạy lại đúng notebook của fold đó; runner sẽ tự resume. Không
chạy cùng một fold đồng thời trên hai tài khoản.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    code_zip = args.output / "C4_MULTI_ROI_COLAB_CODE.zip"
    data_zip = args.output / "C4_MULTI_ROI_V1_DATA.zip"
    _build_code_zip(code_zip)
    _build_data_zip(data_zip)
    notebook_output = args.output / "notebooks"
    notebook_output.mkdir(exist_ok=True)
    for source in sorted((ROOT / "c4_multi_roi/notebooks").glob("*.ipynb")):
        shutil.copy2(source, notebook_output / source.name)
    code_report = _validate_zip(code_zip)
    data_report = _validate_zip(data_zip, expected_jpegs=84216)
    (args.output / "README.md").write_text(_readme(), encoding="utf-8")
    (args.output / "PACKAGE_SHA256.txt").write_text(
        f"{code_report['sha256']}  {code_zip.name}\n{data_report['sha256']}  {data_zip.name}\n",
        encoding="utf-8",
    )
    print({"code": code_report, "data": data_report, "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
