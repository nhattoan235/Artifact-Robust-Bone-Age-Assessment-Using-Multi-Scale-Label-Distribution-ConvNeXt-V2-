# C — Kết quả chuẩn bị các phần không cần GPU

Ngày cập nhật: 2026-08-24

## Trạng thái

Đã chuẩn bị audit, QC và manifest ROI cho cache ứng viên:

```text
c3_roi/cache/C3_MASK_REUSE_CANDIDATE_V1/
```

Kết quả:

| Hạng mục | Số lượng |
|---|---:|
| Development rows | 14.036 |
| Official train | 12.611 |
| Official validation | 1.425 |
| Mask có bbox | 11.441 |
| Mask thất bại | 2.595 |
| Tỷ lệ fallback | 18,488% |
| Test rows trong manifest | 0 |

## Artifact đã tạo

```text
c3_roi/cache/C3_MASK_REUSE_CANDIDATE_V1/cpu_artifacts/
├── audit_summary.json
├── roi_manifest.csv
├── qc_failures.jpg
└── qc_valid.jpg
```

`roi_manifest.csv` giữ nguyên toàn bộ 14.036 ảnh. Ảnh có mask dùng `mask_bbox`; ảnh
không có mask dùng `global_fallback`. Fallback được ghi công khai trong từng dòng,
không loại ảnh âm thầm.

## Kiểm tra an toàn

- Không có test row trong manifest.
- Không dùng mask từ thư mục test 200.
- Không chọn tham số dựa trên nhãn test.
- Unit test cho parsing bbox, clamp/margin và fallback đã pass 3/3.

## Giới hạn bắt buộc

Cache này **chưa được dùng để train C chính thức** vì:

1. Tỷ lệ segmentation thất bại 18,488%, cao hơn ngưỡng kiểm soát 1%.
2. Một số ngưỡng hình học trong implementation phân đoạn bên ngoài có chú thích
   được đo từ 200 ảnh test. Vì vậy cache này chỉ là candidate để audit, không phải
   preprocessing test-blind đã khóa.

## Bước tiếp theo

1. Re-derive ngưỡng segmentation chỉ trên development train/validation.
2. Chạy pilot trên mẫu cố định, xem contact sheet và đo tỷ lệ fail.
3. Chỉ khi pass QC mới sinh ROI cache chính thức và chuẩn bị train 5-fold.

