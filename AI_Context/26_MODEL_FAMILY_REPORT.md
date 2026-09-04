# Bản đồ các mô hình và nhánh nghiên cứu

> Cập nhật: 2026-09-03
> Mục tiêu: giải thích các mô hình đã thử, điểm mạnh của từng nhánh và quyết định giữ/loại.
> Đơn vị sai số: tháng tuổi xương.
> Endpoint chính: OOF leakage-safe; RSNA test 200 ảnh đã mở nên các điểm test chỉ mang tính thăm dò.

## 1. Quy ước chung

- Dữ liệu chính là RSNA Pediatric Bone Age Assessment.
- Development pool dùng 5-fold OOF; mỗi `image_id` xuất hiện đúng một lần ở validation.
- Mô hình chính dùng ảnh xám lặp thành 3 kênh, kích thước 512 × 512 và sex embedding nếu có.
- OOF được dùng để chọn mô hình và ensemble. Không chọn trọng số theo nhãn test.
- `backbone` là bộ trích xuất đặc trưng; ROI, attention, TTA và label-distribution là các thành phần/pipeline bổ sung, không phải backbone độc lập.

## 2. Bảng tổng quan

| Mã/nhánh | Cấu hình chính | Kết quả/đánh giá | Vai trò và quyết định |
|---|---|---:|---|
| P0 | Audit manifest, ID, hash, split | PASS | Nền tảng chống leakage, không phải model |
| A0 | ConvNeXt-Tiny, không augmentation | MAE 6,640 ở screening | Control để đo tác động augmentation |
| A1 | ConvNeXt-Tiny + flip | MAE 6,514 | Tốt hơn A0 nhưng chưa phải recipe cuối |
| A2/D0 | ConvNeXt-Tiny + augmentation nhẹ + sex embedding + direct regression | MAE 6,185 ở validation | Recipe được khóa làm baseline |
| B1 | E1 + hand-background masking | MAE 6,239 | Không cải thiện; loại khỏi pipeline chính |
| D1 | ConvNeXtV2-Tiny FCMAE | MAE 31,963 | Feature collapse; loại |
| D2 | ConvNeXt-Tiny + multi-scale fusion | MAE 6,297 | Không cải thiện rõ; loại |
| D3 | ConvNeXt-Tiny + regression head + label-distribution head | Fused OOF 6,38241; D3-TTA 6,24654 | Không thắng E1 khi đứng riêng, nhưng có diversity để ensemble |
| E0 | ConvNeXt-Tiny image-only | MAE 7,4717 | Chứng minh sex là tín hiệu quan trọng |
| E1/P7 | ConvNeXt-Tiny + sex embedding + direct regression | **OOF MAE 6,31669** | Baseline chính, ổn định và dễ tái lập |
| E2 | ConvNeXt-Tiny + hai scalar head M/F | MAE 6,1571 ở validation | Cải thiện nhỏ, CI chứa 0; dừng |
| P9-I | E1 + rotate/flip TTA | **OOF MAE 6,21045** | Cải tiến inference có bằng chứng; giữ làm ứng viên |
| P9-B0 | EfficientNet-B0 gần recipe Deeplasia | MAE 8,5293–11,0428 ở screening | Không phải reproduction đầy đủ; không tiếp tục |
| C3-ROI | ConvNeXt-Tiny độc lập trên ROI toàn bàn tay | OOF MAE 6,43735 | Đứng riêng kém E1 nhưng tạo lỗi bổ sung |
| C3-ROI-TTA | C3-ROI + 10-view TTA | **OOF MAE 6,32558** | Tốt hơn C3 raw nhưng gain vừa phải |
| E1-TTA + C3-ROI-TTA | Ensemble 50/50 | **OOF MAE 6,11708** | Ensemble ROI tốt nhất hiện tại |
| C3-Attention | C3-ROI + spatial attention | OOF khoảng 6,4209 | Không có bằng chứng chắc chắn; không giữ làm model cuối |
| C4 | Một ConvNeXt-Tiny chia sẻ cho global + 6 ROI | OOF MAE 6,63931 trên 14.024 mẫu | Nhánh multi-view mới, đang đánh giá riêng |

## 3. E1/P7 — baseline chính

### Cấu hình

```text
Input: ảnh toàn bàn tay
Backbone: ConvNeXt-Tiny pretrained ImageNet-1K
Sex: embedding 16 chiều
Head: hồi quy trực tiếp tuổi theo tháng
Loss: SmoothL1, beta = 3 tháng
Optimizer: AdamW; scheduler cosine
Input size: 512 × 512
Augmentation: A2, hình học/cường độ nhẹ
Protocol: 5-fold OOF
```

### Điểm mạnh

E1 là control công bằng nhất cho mọi nhánh vì kiến trúc, dữ liệu, loss và split đã được kiểm tra. Nó không phụ thuộc ROI hay module phức tạp, nên phù hợp làm baseline trong luận văn.

Kết quả P7 đạt OOF MAE **6,31669 tháng**. P10-B0 tái lập đúng recipe trên validation và cho toàn bộ prediction trùng với run cũ, xác nhận không có drift trong pipeline.

## 4. E0/E1/E2 — nhánh sex-aware

- **E0** bỏ sex hoàn toàn. MAE 7,4717, kém E1 khoảng 1,2869 tháng. Sex là thông tin hữu ích trong bài toán này.
- **E1** đưa sex qua embedding chung với feature ảnh. Đây là lựa chọn ổn định nhất.
- **E2** dùng bottleneck chung và hai scalar head riêng cho nam/nữ. MAE validation 6,1571 nhưng cải thiện chỉ 0,0277 tháng, CI chứa 0 và không thu hẹp sex gap đủ rõ.

**Quyết định:** giữ E1; không mở E3 hai backbone độc lập theo giới vì chi phí cao và E2 chưa chứng minh được lợi ích.

## 5. D1/D2/D3 — nhánh kiến trúc và label distribution

### D1 — ConvNeXtV2-Tiny FCMAE

D1 thử backbone/khởi tạo ConvNeXtV2-Tiny FCMAE. Kết quả bị feature collapse, MAE tăng lên 31,963. Đây là negative result của recipe cụ thể, không được suy rộng rằng mọi ConvNeXtV2 đều thất bại.

### D2 — multi-scale fusion

D2 ghép đặc trưng ở nhiều mức của backbone. MAE 6,297, không cải thiện so với control. Nhánh này tăng độ phức tạp nhưng chưa tạo lợi ích định lượng đủ rõ.

### D3 — Label Distribution Learning

D3 vẫn dùng ConvNeXt-Tiny nhưng có hai đầu:

1. regression head dự đoán tuổi liên tục;
2. distribution head dự đoán phân phối xác suất trên các tháng tuổi.

Dự đoán fused trộn hai đầu. Ý tưởng này phù hợp với nhãn tuổi xương có độ mơ hồ, nhưng D3 không phải ROI và không phải attention.

D3-TTA riêng đạt OOF MAE **6,24654**, chưa tốt hơn E1-TTA **6,21045**. Tuy nhiên, dự đoán D3 có sai khác đủ để ensemble:

```text
0,5 × E1-TTA + 0,5 × D3-TTA = 6,10134 tháng OOF
```

**Điểm mạnh của D3:** bổ sung thông tin phân phối và tạo diversity cho ensemble.
**Điểm yếu:** không thắng chắc khi đứng riêng; không nên gọi D3 là nhánh ROI.

## 6. P9-I — TTA và bias correction

TTA dùng 10 view:

```text
rotation ∈ {-10°, -5°, 0°, 5°, 10°}
× flip/no-flip
```

Trung bình các prediction làm E1 giảm MAE từ khoảng 6,31747 xuống **6,21045 tháng** trên OOF. Bias correction cross-fitted không cải thiện và bị loại khỏi pipeline chính.

**Điểm mạnh:** rẻ hơn train backbone mới, dễ kiểm chứng và cải thiện ổn định trên E1.
**Giới hạn:** TTA không đảm bảo cải thiện mọi nhánh, đặc biệt với ROI nếu crop đã làm mất một phần ngữ cảnh.

## 7. C3-ROI — nhánh ROI toàn bàn tay

C3-ROI dùng hand mask để lấy bounding box bàn tay, mở rộng margin 8%, pad vuông, resize 512 và đưa crop vào một ConvNeXt-Tiny độc lập. Đây là ROI của **toàn bàn tay**, không phải sáu ROI giải phẫu và cũng không chứa attention.

### Kết quả

- C3-ROI raw: **6,43735 tháng OOF**.
- C3-ROI-TTA: **6,32558 tháng OOF**.
- E1-TTA + C3-ROI-TTA 50/50: **6,11708 tháng OOF**.

### Điểm mạnh

C3-ROI tạo prediction khác E1, vì vậy dù C3 đứng riêng không mạnh hơn E1, ensemble vẫn giảm MAE. Đây là bằng chứng cho giá trị của thông tin local/global bổ sung.

### Giới hạn

ROI fallback còn cao: khoảng 18,49% ở development và 33% ở test của run cũ. Vì vậy không nên gọi C3 là local specialist thuần túy. Các kết quả test C3 trước đây chỉ là exploratory.

## 8. C3-ROI-TTA ensemble — ứng viên ROI tốt nhất

Mã chạy: [`c3_roi/tta_oof.py`](../c3_roi/tta_oof.py)
Báo cáo: [`c3_roi/outputs/C3_ROI_TTA_OOF/C3_ROI_TTA_OOF_report.json`](../c3_roi/outputs/C3_ROI_TTA_OOF/C3_ROI_TTA_OOF_report.json)

Kết quả:

| Mô hình | MAE | RMSE | Median AE |
|---|---:|---:|---:|
| E1-TTA | 6,21045 | 8,38244 | 4,74774 |
| E1-TTA + C3-ROI raw | 6,14576 | 8,30944 | 4,65522 |
| **E1-TTA + C3-ROI-TTA** | **6,11708** | **8,27725** | **4,63325** |

Ensemble TTA tốt hơn E1-TTA **0,09337 tháng**, CI paired khoảng `[-0,12037; -0,06743]`. Nó chưa đạt gate cải thiện tối thiểu 0,10 tháng đã đặt trước, nên nên gọi là **ứng viên/ablation dương tính**, chưa tuyên bố là model cuối vượt mọi nhánh.

## 9. EfficientNet/Deeplasia-like screening

Deeplasia có backbone EfficientNet/Inception và code/config công khai, nhưng reproduction đầy đủ còn phụ thuộc preprocessing mask, split, augmentation, ensemble và checkpoint. Thí nghiệm nội bộ chỉ là screening gần recipe Deeplasia:

- EfficientNet-B0, stem 3 kênh: MAE 8,5293.
- EfficientNet-B0, stem 1 kênh: MAE 8,5823.
- Một biến thể batch 24: MAE 11,0428.

Không được kết luận EfficientNet hoặc Deeplasia yếu hơn về bản chất; chỉ kết luận bản screening hiện tại chưa tái lập đúng và không đạt control E1.

## 10. C4 — nhánh multi-view riêng

C4 dùng một ConvNeXt-Tiny chia sẻ trọng số cho bảy view: một ảnh global và sáu ROI giải phẫu. Feature bảy view được nối lại rồi kết hợp với sex embedding. Đây là nhánh khác với C3-ROI.

C4 hiện đạt OOF MAE **6,63931** trên 14.024 mẫu. Do C4 dùng manifest khác C3 và chưa chạy TTA, không được dùng kết quả này để kết luận sáu ROI kém hơn ROI toàn bàn tay về bản chất.

Phân tích đầy đủ nằm ở [`AI_Context/27_C4_GLOBAL_SIX_ROI_REPORT.md`](27_C4_GLOBAL_SIX_ROI_REPORT.md).

## 11. Kết luận theo mục tiêu

| Mục tiêu | Mô hình nên dùng |
|---|---|
| Baseline luận văn | E1/P7 ConvNeXt-Tiny + sex embedding |
| Đo đóng góp sex | E0 so với E1 |
| Đo đóng góp ROI | E1 so với C3-ROI, cùng split |
| Ensemble ROI tốt nhất hiện tại | E1-TTA + C3-ROI-TTA |
| Đo đóng góp Label Distribution | E1 so với D3, sau đó ensemble E1-TTA + D3-TTA |
| Multi-view giải phẫu | C4 global + 6 ROI, báo cáo riêng |
| So sánh bài báo | Chỉ ghi là published benchmark, không gọi là reproduction nếu chưa đủ config |

Kết luận an toàn nhất hiện tại là: **E1 là baseline chính; E1-TTA + C3-ROI-TTA là ứng viên ensemble ROI tốt nhất; D3-TTA ensemble là ứng viên LDL tốt nhất; C4 là nhánh multi-view đang được đánh giá riêng.**

## 12. Tài liệu tham chiếu

- [Chen et al. — Attention-Guided Discriminative Region Localization for Bone Age Assessment](https://github.com/chenchao666/Bone-Age-Assessment)
- [Rassmann et al. — Deeplasia official code](https://github.com/aimi-bonn/Deeplasia)
- [Bram et al. 2025 — Determination of Skeletal Age From Hand Radiographs Using Deep Learning](https://journals.sagepub.com/doi/10.1177/03635465251359618)
