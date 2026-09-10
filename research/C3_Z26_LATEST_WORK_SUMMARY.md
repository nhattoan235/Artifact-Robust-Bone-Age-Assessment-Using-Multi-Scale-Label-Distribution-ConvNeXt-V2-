# C3-Z26 — Tổng hợp công việc mới nhất

## 1. Mục tiêu

Triển khai và đánh giá một mô hình dự đoán tuổi xương độc lập dựa trên ConvNeXtV2-Base, sau đó kiểm tra khả năng cải thiện bằng Test-Time Augmentation (TTA) và ensemble với P7.

Phần thực nghiệm C3 được tách khỏi hướng C3-ROI của thành viên khác trong nhóm. C3 hiện tại sử dụng ảnh bàn tay toàn cảnh, không dùng crop ROI và không dùng mask trong mô hình.

## 2. Kiến trúc C3

```text
Ảnh X-quang 512×512
        ↓
ConvNeXtV2-Base pretrained
        ↓
Feature map
        ↓
Conv 1×1: 1024 → 512 → GELU → Conv 1×1: 512 → 128
        ↓
Bilinear pooling: 128×128 = 16.384 chiều
        ↓
Signed square-root + L2 normalization
        ↓
Ghép với sex embedding 32 chiều
        ↓
FC 16.416 → 256 → GELU → Dropout 0,2
        ↓
Mean prediction + uncertainty/log-variance branch
```

### Thành phần chính

- Backbone: ConvNeXtV2-Base pretrained.
- Input: ảnh 512 × 512, chuyển ảnh xám thành 3 kênh và chuẩn hóa theo ImageNet.
- Bilinear pooling: học tương tác giữa các đặc trưng hình thái xương.
- Sex embedding: mã hóa giới tính thành vector 32 chiều.
- Mean head: dự đoán tuổi xương.
- Uncertainty branch: ước lượng độ bất định của dự đoán.
- Loss: Smooth L1 kết hợp trọng số uncertainty.

## 3. Dữ liệu và protocol 5-fold

Dataset recipe: `C3_Z26_COMBO_V1`.

- Tổng development images: 14.036.
- Official train images: 12.611.
- Official validation images: 1.425.
- 200 ảnh test không được dùng để chọn checkpoint.
- Mỗi ảnh validation đúng một lần trong OOF.
- Các fold có thể trỏ tới cả `images/train` và `images/validation_official`, vì 5-fold được tạo trên toàn bộ development set chứ không theo official split.

### Kích thước fold

| Fold | Train | Validation |
|---:|---:|---:|
| 1 | 11.211 | 2.825 |
| 2 | 11.219 | 2.817 |
| 3 | 11.232 | 2.804 |
| 4 | 11.239 | 2.797 |
| 5 | 11.243 | 2.793 |

## 4. Cấu hình huấn luyện

- Epoch tối đa: 40.
- Early stopping patience: 8.
- Batch size vật lý: 16.
- Gradient accumulation: 2, effective batch size 32.
- Optimizer: AdamW.
- Learning rate: `2e-5`.
- Weight decay: `0.05`.
- AMP FP16.
- Gradient clipping: `5.0`.
- Seed: `42`.
- Workers: `2`.
- Thời gian mỗi phiên Colab: khoảng 10 giờ 30 phút.
- Checkpoint lưu trực tiếp trên Google Drive để resume.

## 5. Cách tổ chức train

Để giảm thời gian, 5 tài khoản Colab được dùng song song:

```text
Acc 1 → Fold 1
Acc 2 → Fold 2
Acc 3 → Fold 3
Acc 4 → Fold 4
Acc 5 → Fold 5
```

Mỗi fold có thư mục riêng:

```text
c3_z26_parallel/runs/fold_1/
c3_z26_parallel/runs/fold_2/
c3_z26_parallel/runs/fold_3/
c3_z26_parallel/runs/fold_4/
c3_z26_parallel/runs/fold_5/
```

Mỗi thư mục gồm:

- `best_mae.ckpt`: checkpoint validation MAE tốt nhất.
- `last.ckpt`: checkpoint gần nhất để resume.
- `config.json`: cấu hình và target statistics.
- `train.log`: log huấn luyện.
- `val_predictions_best.csv`: dự đoán validation của checkpoint tốt nhất.
- `val_predictions_last.csv`: dự đoán validation cuối cùng.
- `FOLD_COMPLETED.marker`: đánh dấu fold hoàn tất.

## 6. Kết quả bộ 5-fold mới nhất

Kết quả lấy từ `C3_Z26_5FOLD_RESULTS`:

| Fold | MAE | RMSE | Median AE | ±6 tháng | ±12 tháng | ±18 tháng |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,8149 | 9,5381 | 5,2447 | 56,14% | 83,68% | 95,04% |
| 2 | 6,6275 | 8,7332 | 5,1529 | 55,73% | 84,98% | 95,39% |
| 3 | 6,6618 | 8,8697 | 5,0855 | 56,67% | 84,49% | 95,22% |
| 4 | 6,7971 | 8,9312 | 5,3659 | 54,95% | 84,05% | 94,64% |
| 5 | **6,4405** | 8,7258 | **4,8970** | **58,61%** | **85,93%** | **95,63%** |
| **Pooled OOF** | **6,6686** | **8,9655** | **5,1396** | **56,42%** | **84,63%** | **95,18%** |

Fold 5 là fold tốt nhất. Fold 1 là fold có MAE cao nhất. Chênh lệch giữa các fold chưa cho thấy fold lỗi bất thường.

## 7. Early stopping

| Fold | Epoch dừng | Best MAE |
|---:|---:|---:|
| 1 | 19 | 6,8149 |
| 2 | 15 | 6,6275 |
| 3 | 16 | 6,6618 |
| 4 | 17 | 6,7971 |
| 5 | 17 | 6,4405 |

Train loss tiếp tục giảm nhưng validation MAE không còn cải thiện, vì vậy early stopping hoạt động đúng. Khi inference phải dùng `best_mae.ckpt`, không dùng `last.ckpt`.

## 8. C3-TTA OOF đã thực hiện

TTA gồm 10 biến thể cho mỗi ảnh:

- Góc xoay: `-10°, -5°, 0°, +5°, +10°`.
- Hai trạng thái: không lật và lật ngang.

Kết quả của bộ checkpoint C3 trước đó:

| Phương pháp | MAE | RMSE |
|---|---:|---:|
| C3 raw | 6,6168 | 8,9459 |
| C3 + TTA | **6,5593** | **8,8725** |

TTA giảm MAE `0,0576` tháng. Paired bootstrap cho delta TTA − raw là:

```text
Delta MAE: -0,0576 tháng
95% CI: [-0,0857, -0,0288]
```

TTA cải thiện C3 ở cả 5 fold và các subgroup chính, nhưng mức cải thiện nhỏ.

## 9. Ensemble với P7

Các kết quả OOF trước đó:

| Mô hình | OOF MAE |
|---|---:|
| P7 raw | 6,3236 |
| P7 + TTA | 6,2962 |
| C3 raw | 6,6168 |
| C3 + TTA | 6,5593 |
| P7-TTA + C3 raw, khoảng 66/34 | **6,1758** |

C3 không tốt hơn P7 khi đứng riêng, nhưng có sai số bổ sung nên giúp ensemble. Phương án blend tốt nhất hiện được ghi nhận là khoảng:

```text
66% P7-TTA + 34% C3 raw
```

## 10. Lưu ý về hai bộ checkpoint

Bộ `C3_Z26_5FOLD_RESULTS` mới nhất có pooled raw MAE `6,6686`, trong khi báo cáo C3-TTA trước đó dùng một bộ checkpoint khác có raw MAE `6,6168`.

Hash checkpoint của hai bộ khác nhau, vì vậy:

- Không gán kết quả TTA cũ trực tiếp cho checkpoint mới.
- Không dùng blend cũ làm kết luận cuối cùng cho checkpoint mới.
- Cần chạy lại C3-TTA OOF và blend bằng đúng 5 checkpoint trong `C3_Z26_5FOLD_RESULTS`.

## 11. File và thư mục quan trọng

- Đánh giá checkpoint mới: [`C3_Z26_5FOLD_CURRENT_EVALUATION.md`](C3_Z26_5FOLD_CURRENT_EVALUATION.md)
- Đánh giá C3-TTA OOF: [`C3_TTA_OOF_ASSESSMENT.md`](C3_TTA_OOF_ASSESSMENT.md)
- Mã nguồn và bundle huấn luyện được lưu trong thư mục thực nghiệm C3-Z26 ngoài repository do kích thước dữ liệu/checkpoint lớn.

## 12. Việc cần làm tiếp theo

1. Chạy lại C3-TTA OOF bằng 5 checkpoint mới nhất.
2. Tạo lại bảng blend P7/P7-TTA với C3 raw và C3-TTA mới.
3. Khóa phương án bằng OOF.
4. Đánh giá một lần trên test-200 sạch và test-200 chưa làm sạch.
5. Ghi rõ trong báo cáo rằng TTA và ensemble chỉ được lựa chọn sau khi kiểm tra OOF.
