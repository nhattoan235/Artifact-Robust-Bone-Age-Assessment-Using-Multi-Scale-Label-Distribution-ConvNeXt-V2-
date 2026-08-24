# Hướng A — D3 Label Distribution Learning kết hợp TTA

## Thông tin thí nghiệm

| Trường | Giá trị |
|---|---|
| Tên hướng | Plan A — D3 OOF + TTA |
| Bài toán | Dự đoán tuổi xương từ ảnh X-quang bàn tay |
| Backbone | ConvNeXt-Tiny |
| Seed | 42 |
| Số fold | 5-fold OOF |
| Tập phát triển | 12.611 train chính thức + 1.425 validation chính thức = 14.036 ảnh OOF |
| Tập test | RSNA test: 200 ảnh |
| Ngày hoàn tất | 2026-08-23 |

## 1. Mục tiêu

Hướng A kiểm tra liệu Label Distribution Learning có tạo ra thông tin bổ sung so với mô hình regression E1 hay không. Mục tiêu là đánh giá D3 bằng OOF sạch, sau đó dùng TTA và một ensemble cố định để kiểm tra khả năng giảm MAE mà không tối ưu theo tập test.

## 2. Baseline so sánh

E1 là mô hình ConvNeXt-Tiny baseline đã được huấn luyện trước trên cùng protocol 5-fold. Trong hướng A, E1 được giữ cố định để làm mốc so sánh; không huấn luyện lại hoặc thay đổi E1.

## 3. Dữ liệu và protocol

- Development pool được tạo đúng theo split RSNA đã khóa: **12.611 ảnh train chính thức + 1.425 ảnh validation chính thức = 14.036 ảnh**.
- 12.611 ảnh đến từ `boneage-training-dataset`; 1.425 ảnh đến từ `rsna_official_validation`.
- Hai nguồn này được gộp để thực hiện final 5-fold OOF đã định nghĩa trong protocol P7/D3; 1.425 ảnh không phải dữ liệu ngoài protocol.
- Tổng số prediction OOF: 14.036.
- Mỗi ID xuất hiện đúng một lần trong OOF.
- Development pool không chứa ảnh từ test: không có ID hoặc SHA-256 giao với 200 ảnh test.
- Test gồm 200 ảnh và chỉ được mở ở bước đánh giá cuối.
- Không dùng nhãn test để chọn checkpoint, thay đổi kiến trúc, chọn trọng số hoặc chỉnh bias.
- Các kiểm tra ID, giới tính, target, manifest và mapping ảnh đều được thực hiện trước khi aggregate.

## 4. Cấu hình D3

### Kiến trúc

- Backbone: ConvNeXt-Tiny.
- Sex embedding được đưa vào prediction head.
- Hai đầu ra:
  1. Regression head dự đoán tuổi liên tục.
  2. Distribution head dự đoán phân phối tuổi theo từng tháng.
- Prediction D3 fused:

```text
D3_fused = 0.5 × regression_prediction
          + 0.5 × distribution_prediction
```

- Target được chuẩn hóa theo mean/std của từng training fold.
- Mỗi fold được train độc lập với seed 42.
- Không warm-start từ E1 để giữ tính độc lập của phép so sánh.

### Kiểm soát huấn luyện

- Có preflight kiểm tra config, manifest, ID, target, sex và SHA.
- Có checkpoint tốt nhất, checkpoint resume, metrics, warning log và run state.
- Các fold được kiểm tra trạng thái `completed` hoặc `early_stopped` trước khi aggregate.
- Không phát hiện NaN/Inf, duplicate OOF ID hoặc mismatch target/sex.

## 5. Inference và TTA

Mỗi checkpoint được đánh giá với 10 biến thể TTA:

- Góc xoay: `-10°, -5°, 0°, +5°, +10°`.
- Mỗi góc gồm hai trạng thái: không lật và lật ngang.
- Prediction cuối của mỗi model là trung bình 10 prediction.
- Với test, prediction cuối của mỗi nhánh là trung bình prediction của 5 fold.

Ensemble được khóa trước khi mở test:

```text
Final ensemble = 0.5 × E1-TTA + 0.5 × D3-TTA
```

## 6. Kết quả OOF

### Kết quả tổng thể

| Mô hình | OOF MAE (tháng) |
|---|---:|
| E1 raw | 6.31669 |
| D3 fused raw | 6.38241 |
| E1 + D3 fused, 50/50 | 6.18268 |
| E1-TTA | 6.21045 |
| D3-TTA | 6.24654 |
| E1-TTA + D3-TTA, 50/50 | **6.10134** |

So với E1-TTA, ensemble TTA giảm MAE **0.10910 tháng**. Paired bootstrap 95% CI:

```text
[-0.13365, -0.08420]
```

Khoảng tin cậy nằm hoàn toàn dưới 0, cho thấy cải thiện OOF có tính nhất quán trong protocol đã khóa.

### Kiểm tra độ ổn định

Ensemble TTA cải thiện so với E1-TTA ở cả 5 fold:

| Fold | E1-TTA | D3-TTA | Ensemble 50/50 |
|---|---:|---:|---:|
| 1 | 6.13870 | 6.19073 | 6.05910 |
| 2 | 6.13447 | 6.12869 | 6.03226 |
| 3 | 6.30125 | 6.42416 | 6.23347 |
| 4 | 6.28549 | 6.29264 | 6.15197 |
| 5 | 6.19235 | 6.19651 | 6.02994 |

Theo giới tính:

| Nhóm | E1-TTA | D3-TTA | Ensemble 50/50 |
|---|---:|---:|---:|
| Female | 6.45810 | 6.45938 | **6.32824** |
| Male | 6.00109 | 6.06661 | **5.90953** |

Theo nhóm tuổi:

| Nhóm tuổi | E1-TTA | D3-TTA | Ensemble 50/50 |
|---|---:|---:|---:|
| 0–59 | 5.89530 | 5.80815 | **5.55350** |
| 60–119 | 7.29684 | 7.31955 | **7.17422** |
| 120–179 | 5.75861 | 5.86572 | **5.72144** |
| 180–228 | 5.96215 | 5.66558 | **5.59463** |

Ensemble cải thiện ở toàn bộ nhóm tuổi được báo cáo. Nhóm 60–119 tháng vẫn là nhóm khó nhất và được giữ lại trong báo cáo thay vì dùng để tuning hậu nghiệm.

## 7. Kết quả test cuối

Test được chạy một lần sau khi protocol và trọng số ensemble đã khóa.

| Mô hình | Test MAE (tháng) |
|---|---:|
| E1-TTA | 4.61749 |
| D3-TTA | **4.50846** |
| E1-TTA + D3-TTA, 50/50 | 4.51070 |

D3-TTA tốt hơn E1-TTA **0.10903 tháng** trên test. Ensemble 50/50 chỉ kém D3-TTA **0.00224 tháng**, tức gần như tương đương, nhưng D3-TTA là mô hình có MAE test thấp nhất.

### Test theo giới tính

| Nhóm | E1-TTA | D3-TTA | Ensemble 50/50 |
|---|---:|---:|---:|
| Female, n=100 | 4.74378 | 4.55134 | 4.58479 |
| Male, n=100 | 4.49120 | 4.46558 | **4.43660** |

### Test theo nhóm tuổi

| Nhóm tuổi | E1-TTA | D3-TTA | Ensemble 50/50 |
|---|---:|---:|---:|
| 0–59, n=14 | 5.13860 | 5.56981 | 5.35420 |
| 60–119, n=53 | 6.27339 | 6.04524 | 6.09767 |
| 120–179, n=108 | 3.91260 | 3.91477 | **3.85883** |
| 180–228, n=25 | 3.86030 | **3.22085** | 3.49000 |

## 8. Đóng góp kỹ thuật và học thuật

Hướng A có các đóng góp sau:

1. Xây dựng một nhánh Label Distribution Learning độc lập trên cùng protocol 5-fold với baseline.
2. Tách riêng ảnh hưởng của D3, TTA và ensemble thay vì chỉ báo cáo một mô hình cuối.
3. Sử dụng OOF 14.036 ảnh để đánh giá chính, giúp giảm phụ thuộc vào một validation split nhỏ.
4. Dùng paired bootstrap để kiểm tra độ tin cậy của chênh lệch MAE.
5. Kiểm tra độ ổn định theo fold, giới tính và nhóm tuổi.
6. Khóa ensemble trước khi mở test, tránh chọn trọng số theo nhãn test.
7. Ghi nhận cả kết quả âm tính: D3-TTA riêng lẻ không tốt hơn E1-TTA trên OOF, nhưng dự đoán bổ sung đủ khác biệt để ensemble cải thiện OOF.

## 9. Hạn chế

- Test chỉ có 200 ảnh nên khoảng tin cậy có thể rộng.
- Test MAE 4.50846 vẫn chưa đạt mốc tham khảo 3.68 tháng.
- Cải thiện OOF của ensemble không chuyển thành cải thiện rõ rệt so với D3 riêng trên test.
- D3 có tương quan prediction rất cao với E1-TTA, nên mức đa dạng của ensemble còn hạn chế.
- Chưa có external validation ngoài RSNA.
- Không được kết luận rằng phương pháp vượt qua bài báo tham khảo chỉ từ kết quả hiện tại.

## 10. Artifact tái lập

- D3 code: `d3_oof/tta_oof.py`
- D3 test code: `d3_oof/final_test_tta.py`
- OOF report: `d3_oof/outputs/D3_OOF_V1/D3_OOF_report.json`
- TTA OOF report: `d3_oof/outputs/D3_TTA_OOF_V1/D3_TTA_OOF_report.json`
- TTA subgroup report: `d3_oof/outputs/D3_TTA_OOF_V1/D3_TTA_subgroup_report.json`
- Final test report: `d3_oof/outputs/FINAL_TEST_TTA_V1/FINAL_TEST_TTA_report.json`
- Final test predictions: `d3_oof/outputs/FINAL_TEST_TTA_V1/FINAL_TEST_TTA_predictions.csv`
- D3 checkpoints: `d3_oof/runs/D3_OOF_V1/D3_OOF_V1_FOLD_X/best_mae.ckpt`

## 11. Kết luận

D3-TTA là kết quả test tốt nhất trong hướng A với MAE **4.50846 tháng**. Ensemble 50/50 đạt OOF tốt nhất và gần tương đương D3 trên test, nhưng không tạo thêm cải thiện đáng kể trên test. Vì vậy, kết luận chính của hướng A là D3 + TTA có hiệu quả hơn baseline E1-TTA trên test, còn ensemble được báo cáo như một ablation/robustness result chứ không tuyên bố là mô hình thắng tuyệt đối.
