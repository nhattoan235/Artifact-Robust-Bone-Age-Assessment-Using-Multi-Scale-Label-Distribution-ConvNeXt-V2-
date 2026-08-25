# EXP-006 trên Kaggle

Quy trình này dùng Kaggle thay cho Colab. Kaggle không truy cập trực tiếp được
shortcut Google Drive, vì vậy dữ liệu/code/checkpoint phải được đưa vào Kaggle
Dataset hoặc Notebook Input trước khi chạy.

## 1. Tạo Kaggle Dataset input

Tạo một Dataset, ví dụ:

```text
boneage-exp006-assets/
├── data_goc/
│   ├── boneage-training-dataset/
│   ├── boneage-validation-dataset/
│   ├── boneage-training-dataset.csv
│   └── boneage-validation-dataset/Validation Dataset.csv
├── friend_repo/
└── baseline_scripts/
```

Không cần đưa test 200 và file `boneage-dataset.zip` vào Dataset train.

Để resume EXP006 Fold 1 hiện tại, tạo thêm một Dataset nhỏ hoặc thêm vào Dataset
input một file:

```text
checkpoints/
└── EXP006_P7_CONTROL_FOLD_1/
    └── last.ckpt
```

Checkpoint hiện tại có thể lấy từ bản local đã tải về hoặc từ thư mục mirror:

```text
D:\do_an_tot_nghiep\project\baseline_v1\outputs\outputs\outputs\exp006_roadmap\EXP006_P7_CONTROL_FOLD_1\last.ckpt
```

Nếu không muốn upload checkpoint, có thể train Fold 1 lại từ đầu; nhưng resume sẽ
tiết kiệm khoảng 9 epoch đã chạy.

## 2. Tạo Notebook Kaggle và bật GPU

Trong Notebook:

1. Chọn Accelerator: GPU.
2. Chọn T4 nếu được cấp.
3. Add Input → Dataset `boneage-exp006-assets`.
4. Add Input → Dataset checkpoint nếu resume.

Kaggle input là read-only. Mọi checkpoint và output phải ghi vào
`/kaggle/working`.

## 3. Cell kiểm tra input

Thay `<dataset-slug>` bằng slug thực tế của Dataset trong phần Input.

```python
from pathlib import Path
import torch

ASSET_ROOT = Path('/kaggle/input/<dataset-slug>')
DATA_SRC = ASSET_ROOT / 'data_goc'
FRIEND_SRC = ASSET_ROOT / 'friend_repo'
SCRIPTS_SRC = ASSET_ROOT / 'baseline_scripts'

print('DATA:', DATA_SRC.exists(), DATA_SRC)
print('FRIEND:', FRIEND_SRC.exists(), FRIEND_SRC)
print('SCRIPTS:', SCRIPTS_SRC.exists(), SCRIPTS_SRC)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')

assert DATA_SRC.exists()
assert FRIEND_SRC.exists()
assert SCRIPTS_SRC.exists()
assert torch.cuda.is_available(), 'Kaggle chưa cấp GPU'
```

## 4. Copy code vào working

Code nhỏ nên copy vào `/kaggle/working`; dữ liệu vẫn đọc từ input:

```python
import shutil
from pathlib import Path

FRIEND_REPO = Path('/kaggle/working/friend_repo')
BASELINE = Path('/kaggle/working/project/baseline_v1')

shutil.copytree(FRIEND_SRC, FRIEND_REPO, dirs_exist_ok=True)
shutil.copytree(SCRIPTS_SRC, BASELINE / 'scripts', dirs_exist_ok=True)

print('Code ready:', FRIEND_REPO, BASELINE / 'scripts')
```

## 5. Cài dependency và preflight

```python
%cd /kaggle/working/friend_repo
!pip install -q -r p1_baseline/requirements.txt
!nvidia-smi
```

Tạo config/split vào working. Không dùng đường dẫn Drive hoặc Windows:

```python
ROADMAP = Path('/kaggle/working/exp006_roadmap')
RUNS = ROADMAP / 'runs'
MIRROR = Path('/kaggle/working/outputs/exp006_roadmap')
```

```python
!python /kaggle/working/project/baseline_v1/scripts/prepare_exp006_p7_5fold.py \
  --repo-root /kaggle/working/friend_repo \
  --data-root {DATA_SRC} \
  --output-dir {ROADMAP} \
  --run-output-root {RUNS} \
  --checkpoint-mirror-root {MIRROR}
```

```python
%cd /kaggle/working/friend_repo
!python -m p1_baseline.preflight \
  --config /kaggle/working/exp006_roadmap/fold_1/p7_control.toml \
  --no-pretrained
```

Kết quả phải là `status: PASS`.

## 6. Resume Fold 1 từ epoch 9

Nếu đã thêm Dataset checkpoint vào Input, khai báo đường dẫn thực tế:

```python
CHECKPOINT = Path('/kaggle/input/<checkpoint-dataset-slug>/checkpoints/EXP006_P7_CONTROL_FOLD_1/last.ckpt')
print(CHECKPOINT.exists(), CHECKPOINT)
```

Resume:

```python
%cd /kaggle/working/friend_repo
!python -m p1_baseline.train \
  --config /kaggle/working/exp006_roadmap/fold_1/p7_control.toml \
  --resume {CHECKPOINT}
```

Checkpoint/metrics được ghi trong:

```text
/kaggle/working/exp006_roadmap/runs/EXP006_P7_CONTROL_FOLD_1/
/kaggle/working/outputs/exp006_roadmap/EXP006_P7_CONTROL_FOLD_1/
```

## 7. Train các fold còn lại

Không chạy cả 5 fold trong một lần nếu tổng thời gian có thể vượt giới hạn
runtime. Chạy từng fold hoặc từng nhóm nhỏ:

```python
import subprocess

for fold in range(2, 6):
    config = ROADMAP / f'fold_{fold}' / 'p7_control.toml'
    print('START FOLD', fold)
    subprocess.run(
        ['python', '-m', 'p1_baseline.train', '--config', str(config)],
        cwd=FRIEND_REPO,
        check=True,
    )
```

Nếu gần hết thời gian, dừng ở cuối epoch và kiểm tra:

```python
!find /kaggle/working/exp006_roadmap/runs -name "last.ckpt" -o -name "best_mae.ckpt"
```

## 8. Gộp OOF sau khi đủ 5 fold

```python
!python /kaggle/working/project/baseline_v1/scripts/merge_exp006_oof.py \
  --runs-root /kaggle/working/exp006_roadmap/runs \
  --run-prefix EXP006_P7_CONTROL_FOLD_ \
  --output-dir /kaggle/working/exp006_roadmap/p7_control_oof
```

OOF phải đủ 14.036 dòng trước khi chạy TTA hoặc D3.

## 9. Chạy TTA trên OOF

```python
!python /kaggle/working/project/baseline_v1/scripts/evaluate_exp006_tta_oof.py \
  --friend-repo /kaggle/working/friend_repo \
  --development-manifest /kaggle/working/exp006_roadmap/development_manifest_local.csv \
  --oof-csv /kaggle/working/exp006_roadmap/p7_control_oof/oof_predictions.csv \
  --runs-root /kaggle/working/exp006_roadmap/runs \
  --run-prefix EXP006_P7_CONTROL_FOLD_ \
  --output-dir /kaggle/working/exp006_roadmap/p7_tta_oof \
  --device cuda --amp --batch-size 16 --num-workers 2
```

Chỉ giữ TTA nếu OOF MAE cải thiện; không dùng test 200 để chọn.

## 10. Lưu output trước khi runtime kết thúc

Kaggle không tự động biến mọi file trong `/kaggle/working` thành bản lưu lâu dài
nếu Notebook chưa được Save Version. Trước khi hết thời gian:

1. Dừng train ở cuối epoch.
2. Kiểm tra có `last.ckpt` và `best_mae.ckpt`.
3. Chọn **Save Version** hoặc **Save & Run All** một lần, không bấm liên tục.
4. Bảo đảm output của Notebook chứa thư mục `outputs/exp006_roadmap`.

Nếu cần resume ở phiên Kaggle mới, dùng một trong hai cách:

- Add Output/Notebook version trước đó làm Input nếu giao diện Kaggle hỗ trợ.
- Tải thư mục output/checkpoint về máy rồi tạo Kaggle Dataset checkpoint mới.

Tối thiểu cần mang sang phiên sau:

```text
last.ckpt
best_mae.ckpt
config_resolved.yaml
metrics.jsonl
run_state.json
```

## 11. Các nguyên tắc không đổi

- Không train vào `/kaggle/input` vì đây là read-only.
- Không dùng đường dẫn Windows hoặc Drive shortcut trong config Kaggle.
- Không đổi tham số khoa học khi resume checkpoint.
- Nếu đổi learning rate, batch, augmentation, loss hoặc model, tạo run ID mới.
- Chỉ chạy D3 sau khi P7 5-fold OOF hoàn tất.
- Test 200 chỉ đánh giá tham khảo sau khi khóa phương án bằng OOF/holdout.

## 12. Resume checkpoint EXP006 từ Colab sang Kaggle

Checkpoint Fold 1 được tạo ở Colab với hai đường dẫn manifest bắt đầu bằng
`/content/exp006_roadmap/...`. Phiên bản code hiện tại đưa manifest sang
`/kaggle/working/...`; do `train_manifest` và `val_manifest` vẫn được đưa vào
config hash ở phiên bản trainer cũ, resume có thể báo `Config hash khớp: KHÔNG`
dù toàn bộ tham số khoa học thực sự giống nhau.

Với checkpoint hiện tại, dùng cell tương thích sau trước khi resume. Cell chỉ
tạo bản sao manifest nhỏ ở đường dẫn cũ để giữ nguyên hash; ảnh vẫn nằm ở
`/kaggle/working` vì đường dẫn ảnh bên trong manifest đã được prepare đúng:

```python
from pathlib import Path
import shutil

roadmap = Path('/kaggle/working/exp006_roadmap')
old_root = Path('/content/exp006_roadmap')
old_fold = old_root / 'fold_1'
new_fold = roadmap / 'fold_1'
old_fold.mkdir(parents=True, exist_ok=True)

for name in ('train_manifest.csv', 'validation_manifest.csv'):
    shutil.copy2(new_fold / name, old_fold / name)

src_cfg = new_fold / 'p7_control.toml'
compat_cfg = new_fold / 'p7_control_resume_legacy.toml'
text = src_cfg.read_text()
text = text.replace(
    '/kaggle/working/exp006_roadmap/fold_1/',
    '/content/exp006_roadmap/fold_1/',
)
compat_cfg.write_text(text)
print('compat config:', compat_cfg)
```

Kiểm tra trước khi train:

```python
%cd /kaggle/working/friend_repo
!python -m p1_baseline.preflight \
  --config /kaggle/working/exp006_roadmap/fold_1/p7_control_resume_legacy.toml \
  --no-pretrained
```

Sau đó resume bằng `last.ckpt`:

```python
!python -m p1_baseline.train \
  --config /kaggle/working/exp006_roadmap/fold_1/p7_control_resume_legacy.toml \
  --resume /kaggle/input/exp006-fold1-checkpoint/exp006-fold1-checkpoint/checkpoints/EXP006_P7_CONTROL_FOLD_1/last.ckpt
```

Log đúng phải hiển thị:

```text
Split hash khớp: CÓ
Config hash khớp: CÓ
Code version khớp: CÓ
Optimizer/scheduler/scaler đã phục hồi: CÓ
```

Không dùng `--no-pretrained` khi chạy train; tùy chọn này ở preflight chỉ để
không tải lại weights trong bước kiểm tra. Không cần train lại từ đầu.
