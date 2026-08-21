# P9-I — TTA và bias correction trên OOF

## Mục tiêu

Đo riêng đóng góp của inference Deeplasia trên mô hình P7 đã khóa, trước khi
huấn luyện backbone mới. Thí nghiệm này không phải là tái lập đầy đủ Deeplasia:
ảnh vẫn dùng preprocessing/normalization của P7; chỉ TTA và bias correction được
thay đổi.

## Thiết kế

- Nguồn model: năm `best_model.pt` của P7 final.
- Dữ liệu: 14.036 mẫu development, phân theo fold P7.
- TTA: xoay `[-10, -5, 0, 5, 10]` độ, có và không flip ngang.
- Raw: `rotation=0`, không flip.
- Bias correction: hồi quy signed error `prediction - target` theo prediction.
- Cross-fitting: correction cho fold `f` chỉ fit trên OOF của bốn fold còn lại.
- Không truy cập đường dẫn hoặc nhãn RSNA test.

## Các endpoint

1. P7 OOF reference.
2. Raw prediction tái suy luận từ checkpoint.
3. TTA trung bình.
4. Raw + bias correction cross-fitted.
5. TTA + bias correction cross-fitted.

## Tiêu chí quyết định

- TTA/correction chỉ được xem là ứng viên nếu cải thiện OOF một cách nhất quán,
  confidence interval của paired delta được kiểm tra và không làm sụt subgroup.
- Không dùng test để chọn giữa raw, TTA, correction hoặc tổ hợp của chúng.
- Nếu raw tái suy luận không khớp P7 OOF, phải điều tra sai khác trước khi tin
  kết quả ablation.

## Chạy smoke

```powershell
& 'C:\Users\ASUS\Documents\tao_sinh_anh_x_ray_hand\.boneage_env\Scripts\python.exe' -m p9_inference.tta_bias_oof --limit-per-fold 16 --batch-size 8 --device cuda --output-dir p9_inference/outputs/P9_I_TTA_BIAS_OOF_SMOKE
```

## Chạy đầy đủ

```powershell
& 'C:\Users\ASUS\Documents\tao_sinh_anh_x_ray_hand\.boneage_env\Scripts\python.exe' -m p9_inference.tta_bias_oof --batch-size 16 --device cuda
```

Output chính:

- `P9_I_TTA_BIAS_OOF_predictions.csv`
- `P9_I_TTA_BIAS_OOF_report.json`
