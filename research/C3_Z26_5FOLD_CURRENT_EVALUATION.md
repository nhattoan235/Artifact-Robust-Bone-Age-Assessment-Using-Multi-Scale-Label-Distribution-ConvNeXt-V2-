# Đánh giá C3-Z26 5-fold hiện tại

## Trạng thái dữ liệu

- Đủ 5 fold.
- Mỗi fold có `last.ckpt`, `best_mae.ckpt`, `config.json`, `train.log` và validation predictions.
- Có `FOLD_COMPLETED.marker` ở cả 5 fold.
- Tổng OOF: `14,036` dòng, `14,036` ID duy nhất.
- Không có fold nào dừng do thiếu thời gian; các fold đều kết thúc bằng early stopping.

## Cấu hình xác nhận

- ConvNeXtV2-Base + bilinear pooling + uncertainty branch + sex embedding.
- 512 × 512, batch 16, gradient accumulation 2.
- AdamW, learning rate `2e-5`, weight decay `0.05`.
- Seed `42`, tối đa 40 epoch, patience 8.

## Validation metrics của bộ checkpoint hiện tại

| Fold | N | Best MAE | RMSE | Median AE | ±6 | ±12 | ±18 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2825 | 6.8149 | 9.5381 | 5.2447 | 56.14% | 83.68% | 95.04% |
| 2 | 2817 | 6.6275 | 8.7332 | 5.1529 | 55.73% | 84.98% | 95.39% |
| 3 | 2804 | 6.6618 | 8.8697 | 5.0855 | 56.67% | 84.49% | 95.22% |
| 4 | 2797 | 6.7971 | 8.9312 | 5.3659 | 54.95% | 84.05% | 94.64% |
| 5 | 2793 | 6.4405 | 8.7258 | 4.8970 | 58.61% | 85.93% | 95.63% |
| **Pooled OOF** | **14036** | **6.6686** | **8.9655** | **5.1396** | **56.42%** | **84.63%** | **95.18%** |

Fold 5 tốt nhất; Fold 1 kém nhất. Chênh lệch MAE giữa hai fold là khoảng `0.3744` tháng, cho thấy vẫn có khác biệt theo phân vùng dữ liệu nhưng không có fold nào bất thường nghiêm trọng.

## Early stopping

| Fold | Epoch dừng | Best epoch gần nhất | Best MAE |
|---:|---:|---:|---:|
| 1 | 19 | 11 | 6.8149 |
| 2 | 15 | 7 | 6.6275 |
| 3 | 16 | 8 | 6.6618 |
| 4 | 17 | 9 | 6.7971 |
| 5 | 17 | 9 | 6.4405 |

Loss train tiếp tục giảm nhưng validation MAE không còn cải thiện, nên early stopping là hợp lý. Các `last.ckpt` có metric kém hơn `best_mae.ckpt`; khi inference phải dùng `best_mae.ckpt`.

## So sánh với C3 OOF/TTA trước đó

Bộ checkpoint hiện tại có pooled raw MAE `6.6686`, trong khi báo cáo C3 raw trước đó là `6.6168`. Hai bộ không dùng cùng checkpoint: hash checkpoint khác nhau. Chênh lệch là `+0.0518` tháng, khoảng `0.78%` kém hơn.

Do đó, báo cáo C3-TTA trước đó không thể gắn trực tiếp cho bộ checkpoint hiện tại. Nếu muốn báo cáo TTA cho đúng bộ mới, phải chạy lại C3-TTA OOF bằng 5 `best_mae.ckpt` trong thư mục này.

## Kết luận

Bộ kết quả hiện tại hoàn chỉnh và có thể dùng làm một run C3 5-fold độc lập. Tuy nhiên, trước khi chốt ensemble hoặc đánh giá test-200, cần:

1. Dùng chính 5 checkpoint hiện tại để chạy lại C3-TTA OOF.
2. Tạo lại blend với P7/P7-TTA trên cùng ID OOF.
3. Chỉ sau đó mới đánh giá test-200.

