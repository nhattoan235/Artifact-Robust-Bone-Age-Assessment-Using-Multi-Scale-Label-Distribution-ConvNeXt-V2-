# EXP006 — Audit Fold 1 sau khi resume

Ngày kiểm tra: 2026-08-24  
Thư mục mirror:

```text
D:\do_an_tot_nghiep\project\baseline_v1\outputs\results\outputs\exp006_roadmap\EXP006_P7_CONTROL_FOLD_1
```

## Kết quả

- Run: `EXP006_P7_CONTROL_FOLD_1`
- Trạng thái: `early_stopped`
- Epoch cuối: 18
- Global step: 5.616
- Best epoch: 10 theo log hiển thị một-based
- Best MAE: **6.239962 tháng**
- Validation rows: **2.808**
- Peak VRAM: **4.426 MiB**
- Best checkpoint và last checkpoint đều tồn tại.
- `val_predictions_best.csv` có 2.808 dòng và 2.808 image ID duy nhất.

## Cảnh báo

Fold 1 có cảnh báo validation MAE xấu đi trong khi train loss tiếp tục giảm,
đồng thời một số prediction vượt nhẹ miền tuổi 0–228, khoảng cuối cùng được
ghi nhận là `[-2.76, 236.50]`. Đây là cảnh báo ổn định/overfit, không phải lỗi
runtime; early stopping đã dừng run sau 8 epoch không cải thiện.

## Kiểm tra toàn bộ EXP006 P7

Ghép tạm thời `val_predictions_best.csv` của Fold 1–5 để kiểm tra readiness:

- Tổng dòng: **14.036**
- Image ID duy nhất: **14.036**
- ID trùng: **0**
- MAE OOF tạm tính: **6.323629 tháng**
- RMSE OOF tạm tính: **8.529248 tháng**
- Median absolute error xấp xỉ: **4.75 tháng**
- MAE nữ: **6.555828** (n=6.430)
- MAE nam: **6.127332** (n=7.606)

MAE 6.323629 là kết quả OOF P7 control trước TTA/bias correction/LDL. Đây
chưa phải kết quả test 200 ảnh và không dùng nhãn test 200 ảnh.

## Lưu ý về vị trí file

Fold 1 hoàn chỉnh hiện ở thư mục `outputs\exp006_roadmap\...`, còn Fold 2–5
đang có bản chạy trong `exp006_roadmap\runs\...` và bản mirror tương ứng.
Trước khi chạy script merge chính thức, cần đưa Fold 1 về cùng một
`runs\EXP006_P7_CONTROL_FOLD_1` hoặc dùng một thư mục staging thống nhất.
