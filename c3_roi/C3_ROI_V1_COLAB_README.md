# C3 ROI V1 — phân bổ chạy

Trạng thái hiện tại: **PREPARED_BLOCKED**.

## Phân bổ

| Nơi chạy | Fold |
|---|---|
| Colab | 1, 2, 3 |
| Local | 4, 5 |

Các fold dùng đúng split đã khóa của D3/P7, seed 42 và pooled validation 14.036
ID. Config đã được kiểm tra load thành công; hash train/validation khớp registry
D3.

## Điều kiện trước khi chạy

Không chạy train khi `roi_status = blocked_until_roi_cache_pass`. Cần có đủ file:

```text
c3_roi/cache/C3_ROI_V1/roi/train/{image_id}.png
c3_roi/cache/C3_ROI_V1/roi/validation_official/{image_id}.png
```

ROI cache phải được tạo từ mask test-blind, fallback được ghi trong manifest, và
fallback rate phải đạt gate đã đăng ký. Cache candidate hiện tại không được dùng
trực tiếp vì fallback 18,5% và ngưỡng segmentation có nguồn gốc cần re-audit.

## Sau khi cache pass

1. Upload `c3_roi/` cùng code `p1_baseline/` lên Drive/Colab.
2. Chạy preflight đúng config của fold.
3. Chạy pilot interrupt/resume trước Fold 1 đầy đủ.
4. Chạy Fold 1–3 trên Colab, Fold 4–5 local.
5. Không đổi config khoa học giữa các lần resume.

