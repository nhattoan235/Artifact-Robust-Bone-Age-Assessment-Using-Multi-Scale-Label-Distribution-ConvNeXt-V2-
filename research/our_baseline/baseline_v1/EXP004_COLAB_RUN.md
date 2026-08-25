# Chạy EXP-004 trên Colab bằng code P7 của bạn cùng nhóm

Mục tiêu: train lại ConvNeXt-Tiny P7 trên split holdout mới, lưu checkpoint và prediction để xác nhận blend 50/50.

## 1. Mount Drive và chuẩn bị dữ liệu

Đặt dữ liệu ở một thư mục Drive, ví dụ:

```text
MyDrive/boneage_colab/data_goc/
MyDrive/boneage_colab/exp004_friend_holdout/
```

Trong Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone --depth 1 https://github.com/nhattoan235/Artifact-Robust-Bone-Age-Assessment-Using-Multi-Scale-Label-Distribution-ConvNeXt-V2-.git /content/friend_repo
!cp -r /content/friend_repo/p1_baseline /content/friend_repo/p1_baseline_backup
!pip install -q -r /content/friend_repo/p1_baseline/requirements.txt
```

Nếu `data_goc` là shortcut từ một tài khoản Drive khác, không dùng thẳng
`cp /content/drive/MyDrive/.../data_goc` ngay. Cell dưới đây tự kiểm tra:

1. thư mục trực tiếp trong `MyDrive`;
2. các thư mục shortcut đã được DriveFS giải quyết dưới `.shortcut-targets-by-id`;
3. cấu trúc thật qua hai file bắt buộc `boneage-training-dataset.csv` và `boneage-validation-dataset/Validation Dataset.csv`.

Sau đó mới copy sang local runtime để đọc ảnh nhanh hơn:

```python
from pathlib import Path
import shutil

drive_root = Path('/content/drive/MyDrive/boneage_colab')
direct = drive_root / 'data_goc'
shortcut_root = Path('/content/drive/.shortcut-targets-by-id')

def is_rsna_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / 'boneage-training-dataset.csv').is_file()
        and (path / 'boneage-validation-dataset' / 'Validation Dataset.csv').is_file()
        and (path / 'boneage-training-dataset').is_dir()
    )

candidates = [direct]
if shortcut_root.is_dir():
    candidates.extend(p for p in shortcut_root.iterdir() if p.is_dir())

source = next((p for p in candidates if is_rsna_root(p)), None)
if source is None:
    print('Không tìm thấy data_goc trực tiếp hoặc shortcut đã resolve.')
    print('Các mục trong shortcut root:', list(shortcut_root.iterdir()) if shortcut_root.exists() else 'không có')
    raise FileNotFoundError(
        'Hãy chia sẻ thư mục đích cho tài khoản Colab hoặc đặt đúng shortcut data_goc.'
    )

print('Nguồn dữ liệu được chọn:', source)
local_root = Path('/content/data_goc')
if local_root.exists():
    shutil.rmtree(local_root)
shutil.copytree(source, local_root)
print('Đã copy tới:', local_root)
```

Nếu đoạn trên không tìm được shortcut, hãy lấy ID của thư mục đích trong URL Drive rồi dùng đường dẫn trực tiếp:

```python
TARGET_FOLDER_ID = 'DAN_ID_THU_MUC_DICH_VAO_DAY'
source = Path('/content/drive/.shortcut-targets-by-id') / TARGET_FOLDER_ID
assert is_rsna_root(source), f'Cấu trúc data không đúng: {source}'
```

Sau đó chạy lại phần copy:

```python
local_root = Path('/content/data_goc')
if local_root.exists():
    shutil.rmtree(local_root)
shutil.copytree(source, local_root)
print('Đã copy tới:', local_root)
```

Không copy file nhãn test vào thư mục dùng cho train.

## 2. Lấy script chuẩn bị split

Upload hoặc copy các file sau từ workspace vào `/content/project/baseline_v1/scripts/`:

- `prepare_friend_holdout.py`
- `verify_friend_p7_inference.py` nếu cần kiểm tra checkpoint cũ

Hoặc clone workspace/repo chứa hai script này. Sau đó chạy:

```python
!python /content/project/baseline_v1/scripts/prepare_friend_holdout.py \
  --repo-root /content/friend_repo \
  --data-root /content/data_goc \
  --output-dir /content/drive/MyDrive/boneage_colab/exp004_friend_holdout \
  --run-output-root /content/exp004_friend_runs \
  --checkpoint-mirror-root /content/drive/MyDrive/boneage_colab/exp004_friend_holdout/checkpoint_mirror \
  --holdout-fraction 0.10 \
  --seed 42
```

Lệnh này tạo manifest và config trên Drive, nhưng checkpoint/đọc ảnh chạy local. Checkpoint nhỏ và log được mirror sang Drive.

## 3. Preflight

```python
!cd /content/friend_repo && python -m p1_baseline.preflight \
  --config /content/drive/MyDrive/boneage_colab/exp004_friend_holdout/friend_p7_fresh_holdout.toml \
  --no-pretrained
```

Kết quả phải là `PREFLIGHT: PASS` với 12.776 train và 1.260 holdout.

## 4. Smoke GPU

Trước khi train dài, chạy config smoke đã tạo hoặc tạo lại config smoke tương ứng. Smoke phải xác nhận:

- `device=cuda`;
- loss hữu hạn;
- gradient hữu hạn;
- không OOM;
- checkpoint và `run_state.json` được mirror lên Drive.

## 5. Train đầy đủ

```python
!cd /content/friend_repo && python -m p1_baseline.train \
  --config /content/drive/MyDrive/boneage_colab/exp004_friend_holdout/friend_p7_fresh_holdout.toml
```

Cấu hình full đã khóa:

- ConvNeXt-Tiny pretrained ImageNet-1K;
- 512×512, grayscale pad vuông rồi lặp 3 kênh;
- sex embedding 16 chiều;
- batch 12, gradient accumulation 3;
- 35 epoch, patience 8;
- AdamW, learning rate `2e-4`, weight decay `0.05`;
- Smooth L1 beta 3 tháng;
- light augmentation và AMP FP16.

## 6. Kiểm tra sau train

Các file cần giữ:

```text
/content/exp004_friend_runs/EXP004_FRIEND_P7_FRESH_HOLDOUT_SEED42/
├── best_mae.ckpt
├── val_predictions_best.csv
├── metrics.jsonl
├── train.log
├── warnings.log
├── config_resolved.yaml
├── environment.txt
└── run_state.json
```

Bản mirror nằm trong:

```text
MyDrive/boneage_colab/exp004_friend_holdout/checkpoint_mirror/
```

Sau khi tải `val_predictions_best.csv` về workspace, chạy bước blend holdout riêng. Không chọn lại trọng số dựa trên test 200 ảnh.

## 7. Nhật ký

Mọi lần chạy, kể cả lỗi Colab, OOM, hết thời gian hoặc early stopping, phải bổ sung vào:

`project/baseline_v1/EXPERIMENT_LOG.md`
