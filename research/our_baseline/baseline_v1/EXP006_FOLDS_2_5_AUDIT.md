# EXP006 — Audit Fold 2–5

Ngày kiểm tra: 2026-08-24  
Thư mục kiểm tra:

```text
D:\do_an_tot_nghiep\project\baseline_v1\outputs\results\exp006_roadmap\runs
```

## 1. Kết luận nhanh

Fold 2–5 đã chạy xong và dừng bằng early stopping hợp lệ. Mỗi fold có đủ
`best_mae.ckpt`, `last.ckpt`, `metrics.jsonl`, `run_state.json` và
`val_predictions_best.csv`.

Các bản sao trong:

```text
outputs\results\outputs\exp006_roadmap\EXP006_P7_CONTROL_FOLD_n
```

đã được so sánh SHA-256 với bản trong `runs`; checkpoint và prediction của
cả bốn fold đều trùng nhau.

## 2. Kết quả từng fold

| Fold | Trạng thái | Epoch cuối | Best epoch | Best MAE | Validation rows | Global step |
|---:|---|---:|---:|---:|---:|---:|
| 2 | early_stopped | 29 | 20 | 6.257337 | 2.807 | 9.048 |
| 3 | early_stopped | 21 | 12 | 6.344306 | 2.807 | 6.552 |
| 4 | early_stopped | 15 | 6 | 6.466544 | 2.807 | 4.680 |
| 5 | early_stopped | 21 | 12 | 6.310029 | 2.807 | 6.552 |

Trung bình Best MAE của Fold 2–5: **6.3441 tháng**. Đây là trung bình các
MAE từng fold, chưa phải OOF MAE chính thức.

## 3. Kiểm tra tính toàn vẹn

- Tổng prediction Fold 2–5: **11.228 dòng**.
- Số image ID duy nhất: **11.228**.
- Không có ID trùng giữa bốn fold.
- Mỗi fold có 11.229 mẫu train và 2.807 mẫu validation.
- Các fold dùng cùng recipe khoa học: ConvNeXt-Tiny, ảnh 512, augmentation
  `light`, batch 12, gradient accumulation 3, AdamW, cosine scheduler,
  AMP FP16 và patience 8.
- Không thấy lỗi NaN/Inf hoặc lỗi CUDA trong các log đã kiểm tra.
- Peak VRAM ghi nhận khoảng **4.358 MiB**.

## 4. Cảnh báo cần lưu ý

- Fold 2, 3, 4 và 5 đều kết thúc do validation không cải thiện đủ 8 epoch.
- Train loss tiếp tục giảm trong khi validation MAE dao động/xấu đi ở các
  epoch cuối; đây là dấu hiệu overfit nhẹ.
- Fold 3 và Fold 5 có một số prediction vượt nhẹ miền tuổi 0–228.
- Fold 4 yếu nhất với Best MAE 6.466544; cần giữ lại khi tính OOF, không được
  loại fold chỉ vì MAE cao.

## 5. Trạng thái Fold 1

Trong thư mục local `runs`, Fold 1 hiện chỉ có các file khởi tạo và chưa có
checkpoint/prediction hoàn chỉnh. Checkpoint Fold 1 hợp lệ đang nằm ở Dataset
checkpoint riêng và cần resume trên Kaggle bằng config tương thích đường dẫn
legacy. Vì vậy hiện chưa được phép merge OOF đủ 14.036 dòng.

## 6. Bước tiếp theo

1. Resume Fold 1 từ `last.ckpt`.
2. Tải đầy đủ thư mục Fold 1 về cùng cấu trúc `runs/EXP006_P7_CONTROL_FOLD_1`.
3. Kiểm tra đủ 5 fold và tổng cộng 14.036 prediction không trùng ID.
4. Chạy `merge_exp006_oof.py`.
5. Chỉ sau khi có OOF đủ 14.036 dòng mới đánh giá TTA hoặc các biến thể LDL.

Các số liệu trên chỉ dùng development OOF; chưa sử dụng nhãn test 200 ảnh.
