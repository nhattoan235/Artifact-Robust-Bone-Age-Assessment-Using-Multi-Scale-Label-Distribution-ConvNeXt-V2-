# Đánh giá C3 + TTA trên OOF

## Kết luận chính

C3-TTA OOF hợp lệ: 14.036/14.036 ảnh, đủ 5 fold, mỗi ảnh được dự đoán bởi checkpoint của fold không dùng ảnh đó để train. Test-200 không được đọc.

TTA cải thiện C3 độc lập nhưng mức cải thiện nhỏ:

| Phương pháp | MAE | RMSE | Median AE | ±6 | ±12 | ±18 |
|---|---:|---:|---:|---:|---:|---:|
| C3 raw | 6.6168 | 8.9459 | 5.0518 | 57.00% | 85.00% | 95.14% |
| C3 + TTA 10 biến thể | **6.5593** | **8.8725** | 5.0635 | **57.25%** | **85.30%** | **95.15%** |

TTA giảm MAE `0.0576` tháng, tương đương khoảng `0.87%`. Paired bootstrap cho delta TTA − raw:

```text
Delta MAE: -0.0576 tháng
95% CI: [-0.0857, -0.0288]
```

Khoảng tin cậy không chứa 0, do đó cải thiện tuy nhỏ nhưng nhất quán trên OOF.

## Theo fold

| Fold | Raw MAE | TTA MAE | Cải thiện |
|---:|---:|---:|---:|
| 1 | 6.6885 | 6.6715 | 0.0170 |
| 2 | 6.5063 | 6.4252 | 0.0811 |
| 3 | 6.6222 | 6.6050 | 0.0172 |
| 4 | 6.8566 | 6.7311 | 0.1255 |
| 5 | 6.4104 | 6.3630 | 0.0474 |

TTA cải thiện MAE ở cả 5 fold. Fold 4 hưởng lợi nhiều nhất; Fold 1 hưởng lợi ít nhất.

## Theo subgroup

TTA giảm MAE ở tất cả subgroup chính:

- Nữ: `6.8437 → 6.7651` tháng.
- Nam: `6.4251 → 6.3853` tháng.
- 0–59 tháng: `7.2483 → 7.1327` tháng.
- 60–119 tháng: `7.7697 → 7.6714` tháng.
- 120–179 tháng: `6.1008 → 6.0809` tháng.
- 180–228 tháng: `5.8883 → 5.7550` tháng.

Nhóm khó nhất vẫn là 60–119 tháng; TTA chưa giải quyết được sai số cao ở nhóm này.

## So sánh với P7 và blend OOF

Kết quả đã đối chiếu trên cùng 14.036 ID:

| Mô hình | OOF MAE |
|---|---:|
| P7 raw | 6.3239 |
| P7 + TTA | 6.2962 |
| C3 raw | 6.6168 |
| C3 + TTA | 6.5593 |
| P7-TTA + C3 raw, trọng số 66/34 | **6.1758** |
| P7-TTA + C3-TTA, trọng số tối ưu gần 65/35 | 6.1784 |

C3-TTA tốt hơn C3 raw khi đứng riêng, nhưng chưa cải thiện blend tốt nhất hiện có. C3 raw vẫn tạo blend tốt hơn một chút do tính bổ sung sai số với P7-TTA.

## Quyết định

1. Giữ C3-TTA như một kết quả inference hợp lệ.
2. Không thay thế P7-TTA bằng C3-TTA khi blend hiện tại.
3. Giữ candidate ensemble chính là khoảng `66% P7-TTA + 34% C3 raw`.
4. Chỉ đánh giá test-200 sau khi khóa trọng số bằng OOF; không tối ưu thêm trọng số trên test-200.

Nguồn dữ liệu chi tiết: `c3_tta_oof_report.json`, `c3_tta_oof_predictions.csv` và `C3_TTA_OOF_EVALUATION.md` trong thư mục kết quả.

