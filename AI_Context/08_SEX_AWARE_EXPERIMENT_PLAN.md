# Nhận xét P9-B0 và thiết kế thí nghiệm sex-aware

> **Ngày chốt định hướng:** 2026-08-21  
> **Trạng thái:** kế hoạch nghiên cứu đã đề xuất; chưa triển khai E0/E2  
> **Phạm vi lựa chọn mô hình:** chỉ development train/validation hoặc OOF; không dùng nhãn RSNA test  
> **Thuật ngữ:** `sex` trong tài liệu này là giới tính sinh học theo nhãn M/F của bộ RSNA.

## 1. Kết luận điều hành

Hướng ưu tiên tiếp theo của đồ án là:

> **Giữ ConvNeXt-Tiny P7/P10-B0 làm nền tảng, giữ TTA là cải tiến đã được chứng minh, và kiểm định cách chuyên biệt hóa đầu ra theo giới tính bằng backbone chung và hai regression head.**

Không chuyển P9-B0 sang EfficientNet-B4, 1024 hoặc ensemble P9-C. Không dùng RSNA test để cứu, chọn hoặc tinh chỉnh cấu hình mới.

P9-B0 được ghi nhận là **negative screening của một bản tái lập gần Deeplasia**, không phải là bằng chứng rằng EfficientNet-B0 hoặc Deeplasia thất bại về bản chất.

## 2. Bằng chứng hiện tại

### 2.1. Baseline đã khóa

- P10-B0 tái lập chính xác P2 trên 1.425 ảnh validation: MAE **6,184792 tháng**, RMSE 8,4865.
- P7 pooled OOF trên 14.036 ảnh: MAE **6,316691 tháng**, bootstrap 95% CI [6,224638; 6,411327].
- P9-I TTA trên OOF: MAE giảm từ 6,317471 xuống **6,210446 tháng**; paired delta **-0,107025**, CI [-0,135977; -0,078220].
- Bias correction và TTA + bias correction không cải thiện; chỉ giữ TTA.

### 2.2. Kết quả screening P9-B0

Control để ra quyết định là P10-B0 MAE 6,184792.

| Run | Biến thể | Số epoch đã đánh giá | Best validation MAE | Delta so với control |
|---|---|---:|---:|---:|
| `P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_SEED42` | stem 3 kênh, batch 12, accum 2 | 3 | **8,5293** | +2,3445 |
| `P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_1CH_SEED42` | stem 1 kênh, batch 12, accum 2 | 2 | **8,5823** | +2,3975 |
| `P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_1CH_B24_SEED42` | stem 1 kênh, batch 24, accum 1 | 1 | **11,0428** | +4,8580 |

Các run không có NaN/Inf/OOM, checkpoint và prediction artifact được giữ. Cả ba không đạt gate screening nên không có cơ sở chạy OOF, TTA hoặc ensemble cho các candidate này.

### 2.3. Giới hạn của kết luận P9-B0

Không gọi P9-B0 là một reproduction đầy đủ vì:

1. Các candidate dừng sau 1-3 epoch; chưa đi qua toàn bộ scheduler/early stopping của recipe gốc.
2. P9-B0 dùng torchvision EfficientNet, trong khi Deeplasia dùng custom EfficientNet.
3. Cơ chế chuyển pretrained RGB sang grayscale chưa bit-exact với mã Deeplasia.
4. Cache PNG + torchvision augmentation chưa tương đương hoàn toàn xử lý động bằng Albumentations.
5. Hành vi pad/resize/crop và mask còn sai khác.
6. Single EfficientNet-B0 không đại diện cho ensemble dị thể cuối cùng của Deeplasia.

Kết luận được phép sử dụng:

> **Ba candidate EfficientNet-B0 gần Deeplasia không đạt gate trong screening hiện tại; nhánh được dừng để tránh tiếp tục tiêu tốn tài nguyên khi chưa có tín hiệu.**

Kết luận không được sử dụng:

> **EfficientNet hoặc Deeplasia không hiệu quả cho dự đoán tuổi xương.**

## 3. Cơ sở chọn hướng sex-aware

Phân tích trên P7/P9-I OOF cho thấy sai số khác nhau theo giới tính:

| Chỉ số OOF | Nữ | Nam |
|---|---:|---:|
| Số mẫu | 6.430 | 7.606 |
| Tuổi xương trung bình | 117,9 | 135,3 |
| Raw MAE | 6,537 | 6,132 |
| TTA MAE | 6,458 | 6,001 |
| TTA delta so với raw | -0,079 | -0,131 |

- Khoảng cách raw MAE nữ - nam là khoảng **0,405 tháng**, bootstrap CI [0,214; 0,590].
- Khoảng cách sau TTA là khoảng **0,457 tháng**, CI [0,276; 0,646].
- Khi chuẩn hóa thô theo bốn nhóm tuổi, khoảng cách sau TTA giảm còn khoảng 0,26 tháng; vì vậy một phần khoảng cách liên quan đến khác biệt phân bố tuổi.
- Sai số có tương tác giới tính × nhóm tuổi: không được diễn giải đơn giản rằng một giới luôn khó hơn giới kia ở mọi giai đoạn trưởng thành.
- Linear calibration riêng theo giới bằng leave-one-fold-out không cải thiện: TTA nguyên bản 6,210446 so với TTA + sex-specific linear calibration khoảng 6,2283 tháng.

Các kết quả trên tạo ra giả thuyết có thể kiểm định: giới tính có thể cần được dùng để điều kiện hóa ánh xạ feature-to-age ở mức head, thay vì chỉ sửa intercept/slope sau suy luận.

## 4. Câu hỏi và giả thuyết nghiên cứu

### 4.1. Câu hỏi chính

> Chuyên biệt hóa regression head theo giới tính trên một backbone học chung có giảm sai số của nhóm khó hơn mà không làm giảm hiệu năng tổng thể hay không?

### 4.2. Câu hỏi phụ

1. Biến giới tính đóng góp bao nhiêu so với mô hình chỉ dùng ảnh?
2. Sex embedding dùng chung và hai head chuyên biệt khác nhau thế nào về overall MAE, worst-group MAE và sex gap?
3. Lợi ích của TTA có ổn định giữa hai giới và các giai đoạn tuổi không?
4. TTA disagreement có dự báo được absolute error trong từng nhóm giới tính × tuổi hay không?

### 4.3. Giả thuyết

- **H1:** đưa sex vào mô hình cải thiện MAE so với image-only.
- **H2:** shared backbone + sex-specific heads cải thiện female MAE hoặc worst-group MAE so với sex embedding dùng chung.
- **H3:** cải thiện subgroup có thể đạt được mà overall MAE không suy giảm quá ngưỡng non-inferiority đã khóa.
- **H4:** TTA disagreement tương quan dương với absolute error, nhưng độ mạnh tương quan có thể khác giữa các nhóm giới tính × tuổi.

## 5. Ma trận thí nghiệm

| ID | Kiến trúc | Mục đích | Ưu tiên |
|---|---|---|---|
| **E0** | ConvNeXt-Tiny, image-only, một head | Đo giá trị thực của biến sex | Bắt buộc |
| **E1** | ConvNeXt-Tiny + sex embedding, một head | Control hiện tại P10-B0/P7 | Đã có baseline; có thể cần run đối chứng cùng protocol |
| **E2** | ConvNeXt-Tiny backbone chung + hai regression head M/F | Kiểm định sex-aware specialization mà vẫn học feature từ toàn bộ dữ liệu | Ứng viên chính |
| **E3** | Hai ConvNeXt độc lập, mỗi mô hình một giới | Kiểm định full specialization | Chỉ chạy nếu E2 đạt gate |

### 5.1. Kiến trúc E2 đề xuất

```text
Ảnh X-quang
    |
ConvNeXt-Tiny backbone chung
    |
feature vector
    |-----------------------|
female regression head      male regression head
    |                       |
dự đoán tuổi nữ             dự đoán tuổi nam
```

Mỗi mẫu chỉ cập nhật head tương ứng với nhãn sex, nhưng backbone được cập nhật từ cả hai giới. Cấu hình đầu tiên không dùng balanced sampler hoặc oversampling để tránh thay đồng thời nhiều biến; nếu cần cân bằng sẽ là ablation riêng.

## 6. Protocol screening E0-E2

### 6.1. Các thành phần phải giữ cố định

- Official train 12.611 và validation 1.425.
- ConvNeXt-Tiny ImageNet-1K.
- Input 512, grayscale lặp ba kênh.
- Preprocessing `none`.
- A2 augmentation đã khóa.
- SmoothL1 beta 3 tháng.
- AdamW/cosine và các hyperparameter của P10-B0.
- Cùng seed, initialization policy, checkpoint rule và early stopping.
- Không dùng TTA ở primary screening; TTA chỉ được áp dụng sau khi đã khóa candidate raw.
- Không đọc hoặc dùng `rsna_test.csv`.

### 6.2. Thứ tự chạy

1. Chạy E0 seed 42.
2. Chạy E2 seed 42.
3. So sánh E0/E1/E2 trên đúng 1.425 ID validation bằng paired bootstrap.
4. Chỉ ứng viên đạt gate mới được xác nhận thêm seed 17 và 123.
5. Chỉ ứng viên qua xác nhận seed mới được đưa sang 5-fold OOF.
6. E3 chỉ được xem xét sau khi E2 cho tín hiệu ổn định.

### 6.3. Gate giữ E2

Giữ E2 nếu thỏa ít nhất một trong hai điều kiện:

1. Overall MAE cải thiện ít nhất **0,10 tháng** so với E1; hoặc
2. Female MAE cải thiện ít nhất **0,20 tháng**, đồng thời overall MAE không xấu quá **0,05 tháng**.

Điều kiện an toàn bổ sung:

- Không làm male MAE hoặc một age × sex subgroup xấu đi nghiêm trọng.
- Prediction không collapse trong bất kỳ nhóm giới nào.
- Kết quả cùng chiều ở ít nhất 2/3 seed.
- Báo cáo paired delta và bootstrap CI; không chỉ báo cáo MAE điểm.

Ngưỡng trên phải được giữ nguyên sau khi nhìn kết quả seed 42, trừ khi có lỗi kỹ thuật được chứng minh và ghi changelog.

## 7. Xác nhận bằng OOF

Nếu E2 qua gate:

- Dùng development pool 14.036 và split P7 5-fold đã khóa.
- Mỗi ảnh có đúng một prediction OOF.
- So sánh E2 với E1 bằng paired bootstrap trên cùng ID.
- Primary endpoint: pooled OOF MAE.
- Không dùng RSNA test cho quyết định giữ/loại.
- Sau khi khóa mô hình cuối, external holdout chưa chạm là đánh giá xác nhận ưu tiên.

Không được so MAE của một mô hình chỉ dành cho nam hoặc nữ với MAE overall 3,68/3,87 của bài báo. Mọi so sánh subgroup phải dùng cùng quần thể và cùng ảnh.

## 8. Chỉ số phải báo cáo

### 8.1. Primary endpoint

- Overall pooled OOF MAE, đơn vị tháng.

### 8.2. Secondary endpoints

- RMSE và median absolute error.
- Accuracy trong ±6/±12/±18 tháng.
- MAE và signed bias theo sex.
- Chênh lệch MAE nữ - nam và bootstrap CI.
- MAE theo sex × age bin.
- Worst-group MAE.
- Age-standardized MAE theo sex.
- Tỷ lệ sai số >12 và >18 tháng.
- Raw và TTA được báo cáo tách biệt.
- TTA gain và TTA disagreement theo sex × age bin.

### 8.3. Phân tích phụ

- Prediction distribution và collapse checks cho từng giới.
- Correlation giữa TTA disagreement và absolute error.
- Error concentration tại các vùng tuổi hiếm.
- Nếu có artifact stress test: prediction shift paired trên ảnh gốc/ảnh biến đổi, tách theo sex và age bin.

## 9. External validation và test policy

- RSNA test 200 đã được đọc ở P8; không còn là holdout confirmatory cho các quyết định sau P8.
- Không chạy lại RSNA test để chọn E0/E2/E3, TTA setting, checkpoint hoặc ensemble weight.
- Sau khi khóa mô hình bằng validation/OOF, ưu tiên một external holdout chưa chạm như DHA.
- Bộ 200 ảnh artifact/cleaned chỉ là robustness stress test phụ, không thay thế benchmark RSNA hoặc external clinical validation.

## 10. Cách diễn đạt đóng góp khoa học

Nếu E2 cải thiện:

> Shared-backbone sex-specific heads giảm sai số nhóm khó hoặc worst-group MAE mà không làm suy giảm đáng kể overall performance dưới đánh giá OOF leakage-safe.

Nếu E2 không cải thiện:

> Trong recipe ConvNeXt đã khóa, chuyên biệt hóa head theo giới không tạo lợi ích ổn định so với sex embedding; khác biệt sai số theo giới chịu ảnh hưởng đáng kể của phân bố tuổi và không được khắc phục bằng specialization đơn giản.

Cả hai kết quả đều có giá trị nếu protocol được khóa trước, có paired analysis, CI, subgroup analysis và không chọn bằng test.

## 11. Quyết định không thực hiện lúc này

- Không mở rộng P9-B0 sang B4/1024/ensemble.
- Không tiếp tục tăng augmentation mù quáng.
- Không chạy thêm calibration tuyến tính riêng theo giới; phân tích cross-fitted đã không cải thiện.
- Không chỉ lọc nam để lấy MAE thấp hơn.
- Không huấn luyện hai backbone nam/nữ độc lập trước khi E2 chứng minh có tín hiệu.
- Không tuyên bố vượt Bram/Deeplasia từ một subgroup hoặc từ RSNA test đã chạm.

## 12. Artifact nguồn

- `p9_single_model/P9_B0_HANDOFF.md`
- `p9_single_model/runs/P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_SEED42/`
- `p9_single_model/runs/P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_1CH_SEED42/`
- `p9_single_model/runs/P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_1CH_B24_SEED42/`
- `p9_preprocessing/runs/P10_B0_P2_CONTROL_CONVNEXT_TINY_SEED42/`
- `p9_inference/P9_I_HANDOFF.md`
- `p9_inference/outputs/P9_I_TTA_BIAS_OOF/`
- `p7_final_v3/P7_OOF_report.json`

## 13. Bước tiếp theo

1. Khóa specification E0 và E2 trước khi viết code.
2. Thêm unit test xác nhận routing M/F và gradient chỉ đi vào head đúng.
3. Chạy smoke/resume cho E0/E2.
4. Chạy seed 42 trên official validation.
5. Áp dụng gate đã định trước và cập nhật `AI_Context/CHANGELOG.md`.

