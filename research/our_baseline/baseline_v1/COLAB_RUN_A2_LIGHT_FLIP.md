# Chạy baseline trên Google Colab

Tài liệu này dùng cho recipe `a2_light_flip` và checkpoint persistent trên Google Drive.

## 1. Chuẩn bị Google Drive

Tạo cấu trúc:

```text
MyDrive/boneage_colab/
├── boneage_baseline.py
├── data_goc/
└── outputs/
```

Upload file `project/baseline_v1/boneage_baseline.py` mới nhất vào thư mục
`boneage_colab`. Thư mục `data_goc` của bạn đã được giải nén nên không cần ZIP.

## 2. Cell 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Cell 2 — Chuẩn bị code và dữ liệu

```bash
!mkdir -p /content/project
!cp /content/drive/MyDrive/boneage_colab/boneage_baseline.py \
    /content/project/boneage_baseline.py
!cp -r /content/drive/MyDrive/boneage_colab/data_goc /content/
```

Sau khi giải nén, kiểm tra thư mục dữ liệu:

```python
from pathlib import Path
print(list(Path('/content/data_goc').iterdir())[:10])
```

Nếu ZIP giải nén ra thư mục tên khác, sửa `--data-root` ở cell train cho đúng.

## 4. Cell 3 — Smoke test

```bash
!python /content/project/boneage_baseline.py \
  --mode smoke \
  --device cuda \
  --data-root /content/data_goc \
  --recipe a2_light_flip \
  --output-root /content/drive/MyDrive/boneage_colab/outputs \
  --run-name smoke_a2_light_flip \
  --no-pretrained
```

## 5. Cell 4 — Train official

```bash
!python /content/project/boneage_baseline.py \
  --mode official \
  --device cuda \
  --data-root /content/data_goc \
  --recipe a2_light_flip \
  --output-root /content/drive/MyDrive/boneage_colab/outputs \
  --run-name official_a2_light_flip_seed42 \
  --pretrained \
  --amp fp16 \
  --img-size 512 \
  --batch-size 2 \
  --epochs 100 \
  --patience 15 \
  --workers 2
```

## 6. Resume sau khi Colab ngắt

Chạy lại đúng Cell 4 với cùng `--run-name`. Script tự tìm:

```text
/content/drive/MyDrive/boneage_colab/outputs/
└── official_a2_light_flip_seed42/
    └── official/
        └── resume.pt
```

Không đổi recipe, seed, batch hoặc các tham số khoa học giữa chừng. Nếu đổi cấu hình,
dùng `--run-name` mới để không trộn checkpoint giữa hai thí nghiệm.

## 7. Theo dõi kết quả

```python
import pandas as pd
from pathlib import Path

run = Path('/content/drive/MyDrive/boneage_colab/outputs/'
           'official_a2_light_flip_seed42/official')
log = pd.read_csv(run / 'training_log.csv')
display(log.tail())
print('best validation MAE:', log.val_mae.min())
```

Kết quả cuối nằm trong:

```text
report.json
best.pt
resume.pt
training_log.csv
val_predictions.csv
```
