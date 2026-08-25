# EXP-006 → EXP-008: P7 5-fold control, TTA và LDL

Tài liệu này là quy trình chạy tiếp theo sau khi đã tạo bộ công cụ trong:

- `scripts/prepare_exp006_p7_5fold.py`
- `scripts/merge_exp006_oof.py`
- `scripts/evaluate_exp006_tta_oof.py`

Tập test 200 ảnh không được đọc trong các bước này.

## 1. Chuẩn bị Colab

Đưa lên Drive:

- toàn bộ `friend_repo`;
- thư mục `project/baseline_v1/scripts` mới nhất;
- dữ liệu `data_goc` hoặc shortcut tới thư mục dữ liệu.

Sau đó chạy:

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount('/content/drive')

# Chỉnh ba đường dẫn này theo Drive của bạn.
DRIVE_ROOT = Path('/content/drive/MyDrive/boneage_colab')
FRIEND_REPO = Path('/content/friend_repo')
BASELINE = Path('/content/project/baseline_v1')
DATA = Path('/content/data_goc')
ROADMAP = Path('/content/exp006_roadmap')
MIRROR = DRIVE_ROOT / 'outputs' / 'exp006_roadmap'

import os
os.environ['FRIEND_REPO'] = str(FRIEND_REPO)
os.environ['BASELINE'] = str(BASELINE)
os.environ['DATA'] = str(DATA)
os.environ['ROADMAP'] = str(ROADMAP)
os.environ['MIRROR'] = str(MIRROR)
```

Nếu dữ liệu vẫn nằm trên Drive, có thể dùng trực tiếp đường dẫn Drive cho
`DATA`, nhưng đọc ảnh từ `/content` thường nhanh và ổn định hơn. Không cần
copy nhãn test.

## 2. Cài dependency và kiểm tra GPU

```python
%cd /content/friend_repo
!pip install -q -r p1_baseline/requirements.txt
!nvidia-smi
!python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Nếu `CUDA: False`, dừng trước khi train; không chạy 5 fold bằng CPU.

## 3. Copy code và tạo manifest/config

```python
!mkdir -p /content/project/baseline_v1/scripts
!cp -f "$DRIVE_ROOT/project/baseline_v1/scripts/prepare_exp006_p7_5fold.py" /content/project/baseline_v1/scripts/
!cp -f "$DRIVE_ROOT/project/baseline_v1/scripts/merge_exp006_oof.py" /content/project/baseline_v1/scripts/
!cp -f "$DRIVE_ROOT/project/baseline_v1/scripts/evaluate_exp006_tta_oof.py" /content/project/baseline_v1/scripts/
```

Đảm bảo `FRIEND_REPO`, `DATA` và các script đã tồn tại:

```python
assert FRIEND_REPO.exists()
assert (FRIEND_REPO / 'p0_audit/outputs/development_manifest_14036.csv').exists()
assert DATA.exists()
assert (BASELINE / 'scripts/prepare_exp006_p7_5fold.py').exists()
```

Tạo 5 fold. `MIRROR` dùng để lưu checkpoint nhỏ lên Drive; run chính nằm
trên `/content` để giảm I/O:

```python
!python /content/project/baseline_v1/scripts/prepare_exp006_p7_5fold.py \
  --repo-root {FRIEND_REPO} \
  --data-root {DATA} \
  --output-dir {ROADMAP} \
  --run-output-root {ROADMAP}/runs \
  --checkpoint-mirror-root {MIRROR}
```

## 4. Preflight bắt buộc

```python
!cd {FRIEND_REPO} && python -m p1_baseline.preflight \
  --config {ROADMAP}/fold_1/p7_control.toml --no-pretrained
!cd {FRIEND_REPO} && python -m p1_baseline.preflight \
  --config {ROADMAP}/fold_1/d3_ldl_regonly.toml --no-pretrained
```

Kết quả phải là `status: PASS`, trong đó `train_hash`, `val_hash`,
`test_path_absent`, `sample_shape` và `forward_shape` đều đúng.

## 5. EXP-006 — train P7 control 5 fold

Chỉ chạy P7 trước. Mỗi fold là một run riêng:

```python
import subprocess
for fold in range(1, 6):
    cfg = ROADMAP / f'fold_{fold}' / 'p7_control.toml'
    print('START FOLD', fold)
    subprocess.run(['python', '-m', 'p1_baseline.train', '--config', str(cfg)], cwd=FRIEND_REPO, check=True)
```

Sau mỗi fold cần có:

```text
/content/exp006_roadmap/runs/EXP006_P7_CONTROL_FOLD_1/best_mae.ckpt
/content/exp006_roadmap/runs/EXP006_P7_CONTROL_FOLD_1/val_predictions_best.csv
```

Nếu Colab hết phiên, chạy lại đúng config với checkpoint gần nhất:

```python
!cd {FRIEND_REPO} && python -m p1_baseline.train \
  --config {ROADMAP}/fold_1/p7_control.toml \
  --resume {ROADMAP}/runs/EXP006_P7_CONTROL_FOLD_1/last.ckpt
```

Không đổi config khoa học giữa lúc resume.

## 6. Gộp OOF và ra quyết định TTA

```python
!python /content/project/baseline_v1/scripts/merge_exp006_oof.py \
  --runs-root {ROADMAP}/runs \
  --run-prefix EXP006_P7_CONTROL_FOLD_ \
  --output-dir {ROADMAP}/p7_control_oof
```

Sau khi OOF đủ 14.036 dòng, chạy TTA:

```python
!python /content/project/baseline_v1/scripts/evaluate_exp006_tta_oof.py \
  --friend-repo {FRIEND_REPO} \
  --development-manifest {ROADMAP}/development_manifest_local.csv \
  --oof-csv {ROADMAP}/p7_control_oof/oof_predictions.csv \
  --runs-root {ROADMAP}/runs \
  --run-prefix EXP006_P7_CONTROL_FOLD_ \
  --output-dir {ROADMAP}/p7_tta_oof \
  --device cuda --amp --batch-size 16 --num-workers 2
```

Chỉ giữ TTA nếu `metrics.tta.mae` cải thiện so với
`metrics.p7_oof_reference.mae` một cách nhất quán. Không dùng test 200 để
chọn TTA.

## 7. EXP-007/008 — chỉ chạy sau khi P7 control và TTA đã có kết quả

Nếu P7 control đã PASS và TTA được ghi nhận, mới chạy lần lượt:

```python
for fold in range(1, 6):
    cfg = ROADMAP / f'fold_{fold}' / 'd3_ldl_regonly.toml'
    print('START D3 REGONLY FOLD', fold)
    subprocess.run(['python', '-m', 'p1_baseline.train', '--config', str(cfg)], cwd=FRIEND_REPO, check=True)
```

```python
!python /content/project/baseline_v1/scripts/merge_exp006_oof.py \
  --runs-root {ROADMAP}/runs \
  --run-prefix EXP007_D3_LDL_REGONLY_FOLD_ \
  --output-dir {ROADMAP}/d3_regonly_oof
```

Sau đó mới chạy D3 fused với config `d3_ldl_fused.toml`, dùng prefix
`EXP008_D3_LDL_FUSED_FOLD_`. Mỗi biến thể phải được ghi riêng; không gộp
fused và regression-only thành một kết quả.

## 8. Artifact cần tải về Drive

Ít nhất phải lưu:

- `roadmap_preparation_audit.json`;
- toàn bộ manifest từng fold;
- các file `.toml`;
- `oof_predictions.csv` và `oof_report.json`;
- `tta_oof_predictions.csv` và `tta_oof_report.json`;
- `best_mae.ckpt`, `last.ckpt`, `metrics.jsonl`, `run_state.json` của từng fold;
- log Colab và mã commit/code version.

Kết quả test 200 chỉ chạy ở bước tham khảo sau cùng, sau khi mọi lựa chọn đã
được khóa bằng OOF/holdout.
