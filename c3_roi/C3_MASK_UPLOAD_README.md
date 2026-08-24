# C3 mask upload package

Gói này bổ sung mask cho bộ dữ liệu cũ đã có trên Google Drive. Không cần upload
lại `data_dev_v1.zip`.

## Nội dung

```text
masks/{image_id}.png          # 14.036 file mask, tên theo image_id
mask_manifest.csv             # source SHA, mask SHA, bbox, split, trạng thái
cpu_artifacts/audit_summary.json
cpu_artifacts/roi_manifest.csv
cpu_artifacts/qc_failures.jpg
cpu_artifacts/qc_valid.jpg
```

## Cách đặt trên Drive/Colab

Giải nén vào một thư mục riêng, ví dụ:

```text
MyDrive/data/C3_MASK_CANDIDATE_V1/
├── masks/
├── mask_manifest.csv
└── cpu_artifacts/
```

Không đặt mask đè vào thư mục ảnh gốc và không cần giải nén lại data cũ.

## Kiểm tra sau khi giải nén

- Phải có 14.036 mask PNG.
- `mask_manifest.csv` phải có 14.036 dòng dữ liệu.
- Có 12.611 dòng `train` và 1.425 dòng `validation_official`.
- Không có test row.

## Cảnh báo nghiên cứu

Đây là `C3_MASK_REUSE_CANDIDATE_V1`, dùng để audit/QC. Có 2.595 mask thất bại
(18,5%) và một số ngưỡng của bộ phân đoạn bên ngoài cần re-audit test-blind.
Không dùng gói này để train C chính thức trước khi cache pass gate.

