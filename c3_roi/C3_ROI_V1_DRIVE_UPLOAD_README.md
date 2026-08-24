# C3 ROI V1 — gói upload hoàn chỉnh

Gói này đã bao gồm toàn bộ phần cần cho C, ngoại trừ ảnh gốc đã có sẵn trên
Google Drive.

## Cấu trúc sau khi giải nén

```text
C3_ROI_V1/
├── masks/                         # 14.036 mask PNG
├── mask_manifest.csv
├── cpu_artifacts/                 # audit, ROI manifest, QC
├── manifests/                     # train/validation của 5 fold
├── configs/                       # config Fold 1–5
├── registry.json
├── c3_code/                       # script chuẩn bị và utility C
├── p1_baseline/                   # code trainer/model/data/preflight
└── README.md
```

## Đặt trên Drive

Đặt thư mục đã giải nén cạnh data cũ:

```text
MyDrive/data/
├── data_dev_v1.zip
└── C3_ROI_V1/
```

Không cần upload lại hoặc giải nén lại `data_dev_v1.zip`.

## Phân bổ fold

- Colab: Fold 1, 2, 3.
- Local: Fold 4, 5.

## Trạng thái bắt buộc

Các config đang ở trạng thái `PREPARED_BLOCKED`. Gói hiện có mask candidate để
audit, nhưng chưa có ROI crop chính thức. Không chạy train C trước khi re-audit
mask test-blind và tạo đủ:

```text
C3_ROI_V1/roi/train/{image_id}.png
C3_ROI_V1/roi/validation_official/{image_id}.png
```

Mask candidate hiện có 2.595 fallback (18,5%) và không được coi là kết quả C cuối.

