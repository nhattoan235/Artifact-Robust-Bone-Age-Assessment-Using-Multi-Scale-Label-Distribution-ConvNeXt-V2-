# P3 – Tiền xử lý toàn bàn tay

## Trạng thái cuối cùng

- B0 không train lại: tái sử dụng A2, validation MAE `6,1847917`, RMSE `8,4864714`.
- Audit B1 đã PASS; huấn luyện B1 early stop hợp lệ sau 22 epoch.
- Best checkpoint ở epoch 14: validation MAE `6,23894`, RMSE `8,44963`, median AE `5,0`.
- Kết luận: không chọn B1 làm preprocessing chính; P4 dùng `preprocessing=none`.
- Chưa xem kết quả test và không dùng test để chọn giao thức.

## Giao thức B1 đã khóa trước khi xem MAE

- Tên: `official_mask_v1_zero_background_no_crop_no_rotation`.
- Mask: Rassmann et al., DOI dữ liệu `10.5281/zenodo.7611677`.
- ZIP: `306332153` byte; MD5 `a692de2799d99fd4bfec8cd535cd7979`.
- `eff_unet` là nguồn chính; `Tensormask` chỉ bù ID thiếu.
- Chỉ đặt nền ngoài mask bằng 0; không crop, không xoay, không hard ROI.
- Sau masking: pad vuông và resize 512 giống pipeline B0; augmentation giữ nguyên A2.

Lý do không crop/rotate: mã Deeplasia dùng `mask_crop_size=-1` trên RSNA, còn kết quả hard ROI trước đó đã làm MAE xấu hơn. Do đó B1 chỉ thay đổi một biến khoa học là full-hand background masking.

## Bằng chứng chọn nguồn mask

So với mask thủ công ở kích thước audit 512:

| Nguồn | Số cặp | Mean Dice | Median Dice | P05 Dice |
|---|---:|---:|---:|---:|
| Efficient-UNet | 526 | 0,98878 | 0,99095 | 0,97834 |
| TensorMask | 527 | 0,98810 | 0,99101 | 0,97488 |

Efficient-UNet được khóa vì mean Dice và lower tail tốt hơn một chút.

## Audit toàn bộ development set

- Tổng: 14.036 ảnh = 12.611 train + 1.425 validation.
- Efficient-UNet: 14.013 ảnh.
- TensorMask fallback: 22 ảnh.
- Raw fallback: một ảnh, ID `3100`.
- Mask rỗng: 0.
- Lệch kích thước mask/ảnh: 0.
- Cache độc lập: đúng 12.611 file train và 1.425 file validation.
- Area fraction: p0,1=0,07771; p1=0,14374; median=0,32510; p99=0,45047; p99,9=0,49074.

Artifacts:

- `p3_preprocessing/outputs/official_mask_v1/p3_official_mask_summary.json`
- `p3_preprocessing/outputs/official_mask_v1/p3_official_mask_qc.csv`
- `p3_preprocessing/outputs/official_mask_v1/p3_official_mask_visual_top24.jpg`
- `p3_preprocessing/outputs/official_mask_v1/cache`

Prototype threshold cổ điển đã bị loại trước khi train vì fallback 37,5% sau khi ngăn mask chọn nhầm nền sáng.

## Huấn luyện B1

- Config: `p1_baseline/configs/p3_b1_official_mask.toml`.
- Run: `p1_baseline/runs/P3_B1_OFFICIAL_MASK_CONVNEXT_TINY_SEED42`.
- Scientific config hash: `75fe74d828b80daf98b4bf543e860a5de4fc528e70a5d93c3c0e7257f24b1c51`.
- Preflight: PASS.
- Unit tests: 10/10 PASS.
- Smoke + resume: PASS qua hai optimizer step; split/config/code hash đều khớp.
- `warnings.log` rỗng trước lượt train dài.
- Sau epoch 1, `warnings.log` vẫn rỗng; đề xuất tự động của trainer: `TIẾP TỤC`.
- Checkpoint: mỗi 500 optimizer step hoặc tối đa 20 phút; có `last.ckpt` và `best_mae.ckpt`.

Log đang dùng:

- `p1_baseline/P3_B1_launcher.stdout.log`
- `p1_baseline/P3_B1_launcher.stderr.log`
- `p1_baseline/runs/P3_B1_OFFICIAL_MASK_CONVNEXT_TINY_SEED42/train.log`
- `p1_baseline/runs/P3_B1_OFFICIAL_MASK_CONVNEXT_TINY_SEED42/warnings.log`
- `p1_baseline/runs/P3_B1_OFFICIAL_MASK_CONVNEXT_TINY_SEED42/run_state.json`

## So sánh paired B1–B0 và quyết định

- B0 MAE: `6,18479`; B1 MAE: `6,23894`.
- Delta `MAE_B1 - MAE_B0 = +0,05414` tháng; số dương nghiêng về B0.
- Paired bootstrap 95% CI: `[-0,10167; +0,21180]` tháng.
- B1 tốt hơn trên 594 ảnh, xấu hơn trên 626 ảnh, hòa 205 ảnh.
- B1 RMSE tốt hơn nhẹ `0,03684` tháng, accuracy ±12/±18 tốt hơn nhẹ, nhưng accuracy ±6 giảm `1,12` điểm phần trăm.
- CI chứa 0: không có bằng chứng B1 gây hại có ý nghĩa thống kê, nhưng cũng không có bằng chứng cải thiện primary endpoint MAE.
- Theo quy tắc khóa trước: **loại B1 khỏi pipeline chính; P4 tiếp tục với B0/A2 và `preprocessing=none`**.
- Giữ checkpoint B1 cho Artifact Invariance Test/phân tích phụ, không dùng làm model chính.
