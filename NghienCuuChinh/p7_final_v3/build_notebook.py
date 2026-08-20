from __future__ import annotations

import json
from pathlib import Path


def md(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True)}


def code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": value.splitlines(keepends=True)}


cells = [
    md("""# P7 V3 — chạy một fold, lưu an toàn trên Drive mount

Không dùng Drive API, email hay Folder ID. Mỗi session chỉ mount **một Drive lưu trữ**. Checkpoint local mỗi 5 phút; bản persistent mới mỗi 60 phút, tối đa 10 bản/fold.
"""),
    md("## 1. Mount Drive lưu trữ\n\nỞ cửa sổ chọn tài khoản, chọn Drive chứa thư mục `bone_age_p7_v3`. Tài khoản mở Colab có thể là A, B hoặc C."),
    code("""from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print('Drive mount xong. Không cần auth.authenticate_user().')
"""),
    md("## 2. Cấu hình duy nhất\n\nCấu trúc Drive phải là `bone_age_p7_v3/P7_COLAB_BUNDLE_V3.zip` và `bone_age_p7_v3/sources/{train_source.zip,validation_source.zip}`. Hai source có thể là shortcut."),
    code("""from pathlib import Path
import json, os, shutil, subprocess, sys

FOLD = 1                         # đổi 1..5
REMOTE_MINUTES = 60.0            # không giảm khi train thật
MAX_SNAPSHOTS = 10               # khoảng tối đa 3.3 GiB/fold

STORAGE_ROOT = Path('/content/drive/MyDrive/bone_age_p7_v3')
BUNDLE = STORAGE_ROOT / 'P7_COLAB_BUNDLE_V3.zip'
TRAIN_SOURCE = STORAGE_ROOT / 'sources/train_source.zip'
VALIDATION_SOURCE = STORAGE_ROOT / 'sources/validation_source.zip'

LOCAL_ROOT = Path('/content/p7_v3')
LOCAL_PROJECT = LOCAL_ROOT / 'project'
LOCAL_DATA = LOCAL_ROOT / 'data'
LOCAL_RUNS = LOCAL_ROOT / 'runs'
RUNTIME_CONFIG = LOCAL_ROOT / f'fold_{FOLD}.toml'

assert 1 <= FOLD <= 5, 'FOLD phải từ 1 đến 5'
assert STORAGE_ROOT.is_dir(), f'Không thấy folder: {STORAGE_ROOT}. Bạn đã mount đúng Drive chưa?'
assert BUNDLE.is_file(), f'Thiếu: {BUNDLE}'
assert TRAIN_SOURCE.is_file(), f'Thiếu ZIP/shortcut: {TRAIN_SOURCE}'
assert VALIDATION_SOURCE.is_file(), f'Thiếu ZIP/shortcut: {VALIDATION_SOURCE}'
(STORAGE_ROOT / 'P7_STORAGE_V3.marker').write_text('P7 V3 STORAGE — DO NOT DELETE', encoding='utf-8')
print('CẤU HÌNH PASS:', {'fold': FOLD, 'storage': str(STORAGE_ROOT)})
"""),
    md("## 3. Cài code vào ổ local Colab"),
    code("""if LOCAL_PROJECT.exists():
    shutil.rmtree(LOCAL_PROJECT)
LOCAL_PROJECT.mkdir(parents=True)
subprocess.run(['unzip', '-q', '-o', str(BUNDLE), '-d', str(LOCAL_PROJECT)], check=True)
os.chdir(LOCAL_PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'p7_final_v3/requirements_colab.txt'], check=True)
os.environ['TORCH_HOME'] = str(LOCAL_ROOT / 'torch_cache')
subprocess.run([
    sys.executable, '-u', '-m', 'p7_final_v3.prepare_runtime', '--fold', str(FOLD),
    '--output', str(RUNTIME_CONFIG), '--local-data', str(LOCAL_DATA), '--local-runs', str(LOCAL_RUNS),
], check=True)
source_manifest = json.loads(Path('p7_final_v3/source_manifest.json').read_text(encoding='utf-8'))
assert TRAIN_SOURCE.stat().st_size == source_manifest['files']['train']['bytes'], 'Sai kích thước train_source.zip'
assert VALIDATION_SOURCE.stat().st_size == source_manifest['files']['validation']['bytes'], 'Sai kích thước validation_source.zip'
print('CODE + SOURCE FINGERPRINT: PASS')
"""),
    md("## 4. Kiểm tra lưu trữ trước khi đụng tới GPU"),
    code("""subprocess.run([
    sys.executable, '-u', '-m', 'p7_final_v3.preflight',
    '--storage-root', str(STORAGE_ROOT), '--fold', str(FOLD),
    '--max-snapshots', str(MAX_SNAPSHOTS), '--local-root', '/content',
], check=True)
"""),
    md("## 5. Giải nén và kiểm tra đủ 14.036 ảnh\n\nMỗi session Colab mới phải chạy lại bước này vì `/content` là ổ tạm."),
    code("""subprocess.run([
    sys.executable, '-u', '-m', 'p7_final.stage_images',
    '--train-source', str(TRAIN_SOURCE), '--validation-source', str(VALIDATION_SOURCE),
    '--destination-root', str(LOCAL_DATA),
], check=True)
subprocess.run([
    sys.executable, '-u', '-m', 'p7_final_v2.validate_setup',
    '--data-root', str(LOCAL_DATA), '--check-images',
], check=True)
"""),
    md("## 6. GPU, preflight mô hình và benchmark"),
    code("""import torch
assert torch.cuda.is_available(), 'Chưa bật GPU: Runtime > Change runtime type > GPU'
print('GPU:', torch.cuda.get_device_name(0), '| Torch/CUDA:', torch.__version__, torch.version.cuda)
subprocess.run([sys.executable, '-u', '-m', 'p1_baseline.preflight', '--config', str(RUNTIME_CONFIG)], check=True)

RUN_BENCHMARK = True             # sau lần PASS đầu có thể đổi False
if RUN_BENCHMARK:
    subprocess.run([
        sys.executable, '-u', '-m', 'p1_baseline.benchmark_batch',
        '--config', str(RUNTIME_CONFIG), '--batches', '12', '--steps', '2',
    ], check=True)
"""),
    md("""## 7. Kiểm thử ngắt–resume bắt buộc

- Lượt đầu để `SMOKE_TARGET = 2`.
- Đổi sang tài khoản Colab khác, mount cùng Drive, chạy lại bước 1–6, đặt `SMOKE_TARGET = 4`.
- Chỉ khi log có `RESUME CHECK` và bước tiếp tục từ 2 thì mới train thật.
"""),
    code("""SMOKE_TARGET = 2  # lượt sau đổi thành 4
command = [
    sys.executable, '-u', '-m', 'p7_final_v3.run_fold', '--fold', str(FOLD),
    '--config', str(RUNTIME_CONFIG), '--storage-root', str(STORAGE_ROOT),
    '--remote-minutes', str(REMOTE_MINUTES), '--max-snapshots', str(MAX_SNAPSHOTS),
    '--interrupt-after-global-step', str(SMOKE_TARGET),
]
result = subprocess.run(command)
assert result.returncode == 0, f'Smoke test lỗi, return code={result.returncode}'
"""),
    md("## 8. Train thật / tự resume\n\nChạy cell này sau khi hai lượt smoke PASS. Nếu Colab reset, chạy lại từ bước 1 rồi chạy cell này; không cần chỉ đường checkpoint."),
    code("""command = [
    sys.executable, '-u', '-m', 'p7_final_v3.run_fold', '--fold', str(FOLD),
    '--config', str(RUNTIME_CONFIG), '--storage-root', str(STORAGE_ROOT),
    '--remote-minutes', str(REMOTE_MINUTES), '--max-snapshots', str(MAX_SNAPSHOTS),
]
result = subprocess.run(command)
if result.returncode == 75:
    print('DỪNG AN TOÀN: đã đủ 10 checkpoint. Xem bước 10, giữ 2 bản mới nhất rồi dọn bản cũ.')
elif result.returncode != 0:
    raise RuntimeError(f'Train lỗi, return code={result.returncode}')
else:
    print('SESSION/FOLD KẾT THÚC HỢP LỆ')
"""),
    md("## 9. Xem trạng thái cả 5 fold"),
    code("""subprocess.run([
    sys.executable, '-u', '-m', 'p7_final_v3.status', '--storage-root', str(STORAGE_ROOT)
], check=True)
"""),
    md("""## 10. Dọn dung lượng có kiểm soát

Chỉ làm khi fold đã có `result_verified: true`, hoặc khi bước 8 báo đã đủ giới hạn. Trong `snapshots` giữ **2 cặp `.ckpt` + `.json` mới nhất**; trong `best_models` giữ **1 cặp `.pt` + `.json` mới nhất**. Xóa các cặp cũ rồi vào Thùng rác Drive chọn **Xóa vĩnh viễn**. Không xóa thư mục `results`.
"""),
    code("""from pathlib import Path
snapshot_dir = STORAGE_ROOT / 'checkpoints' / f'P7_FINAL_V3_FOLD_{FOLD}' / 'snapshots'
metadata = sorted(snapshot_dir.glob('resume_*.json'), reverse=True)
print('THƯ MỤC:', snapshot_dir)
print('GIỮ LẠI 2 CẶP MỚI NHẤT:')
for item in metadata[:2]: print(' ', item.stem + '.ckpt', '+', item.name)
print('CÓ THỂ XÓA CÁC CẶP CŨ:', max(0, len(metadata) - 2))
best_dir = STORAGE_ROOT / 'checkpoints' / f'P7_FINAL_V3_FOLD_{FOLD}' / 'best_models'
best_metadata = sorted(best_dir.glob('best_*.json'), reverse=True)
print('BEST MODEL GIỮ LẠI:', best_metadata[0].stem + '.pt' if best_metadata else 'chưa có')
print('BEST MODEL CÓ THỂ XÓA:', max(0, len(best_metadata) - 1))
print('Notebook KHÔNG tự xóa để tránh xóa nhầm và tránh file nằm âm thầm trong Trash.')
"""),
    md("## 11. Tổng hợp OOF sau khi đủ 5 fold"),
    code("""RUN_OOF = False
if RUN_OOF:
    subprocess.run([
        sys.executable, '-u', '-m', 'p7_final_v3.aggregate',
        '--storage-root', str(STORAGE_ROOT), '--bootstrap', '10000',
    ], check=True)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
output = Path("p7_final_v3/P7_COLAB_V3.ipynb")
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Created {output}")
