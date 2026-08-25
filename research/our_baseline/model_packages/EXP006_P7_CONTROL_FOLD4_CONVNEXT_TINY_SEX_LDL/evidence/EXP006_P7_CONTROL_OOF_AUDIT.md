# EXP006 — P7 Control OOF chính thức

Ngày merge: 2026-08-24  
Run prefix: `EXP006_P7_CONTROL_FOLD_`

## Kết quả merge

- `test_labels_used`: **false**
- Tổng OOF rows: **14.036**
- Fold counts: 2.808 / 2.807 / 2.807 / 2.807 / 2.807
- OOF được tạo từ prediction của fold không train trên chính row đó.
- Tập test 200 ảnh không được đọc.

## Pooled OOF metrics

| Metric | Giá trị |
|---|---:|
| MAE | **6.323629** |
| RMSE | **8.529248** |
| Median AE | **4.75** |
| Accuracy ±6 tháng | 0.591408 |
| Accuracy ±12 tháng | 0.864491 |
| Accuracy ±18 tháng | 0.956683 |

## Theo fold

| Fold | n | MAE | RMSE | Median AE |
|---:|---:|---:|---:|---:|
| 1 | 2.808 | 6.239962 | 8.500631 | 4.75 |
| 2 | 2.807 | 6.257337 | 8.543694 | 4.625 |
| 3 | 2.807 | 6.344306 | 8.545695 | 4.75 |
| 4 | 2.807 | 6.466544 | 8.522673 | 5.00 |
| 5 | 2.807 | 6.310029 | 8.533479 | 4.75 |

## Đánh giá

Đây là baseline P7 Control OOF chính thức, dùng làm selection gate cho TTA,
LDL và blend. MAE 6.323629 còn cao hơn mục tiêu vượt kết quả nhóm bạn khoảng
4.73 tháng, nhưng đã tạo được nền đánh giá leakage-safe để thử các cải tiến.

## Bước tiếp theo

Chạy TTA trên OOF, giữ lại TTA chỉ khi MAE giảm nhất quán so với 6.323629.
Không dùng nhãn test 200 ảnh để chọn TTA, bias correction, LDL hoặc blend.
