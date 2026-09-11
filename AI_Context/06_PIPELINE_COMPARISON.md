# So sánh pipeline và công trình tham khảo

> Cập nhật: 2026-09-11. MAE tính bằng tháng.

## Pipeline nội bộ

| Pipeline | Input và preprocessing | Backbone/head | Inference | OOF 14.036 | Test 200 |
|---|---|---|---|---:|---:|
| Baseline toàn cảnh | Ảnh toàn cảnh, 512 | ConvNeXt-Tiny; global average pooling; final LayerNorm; sex embedding; direct regression | Mean 5 fold | **6,316691** | **4,730321** |
| C3-ROI V1 | Segmentation bbox bàn tay + margin 8%; fallback toàn ảnh | Như baseline | Mean 5 fold | 6,437349 | **4,337267** |
| C3-ROI V1 + TTA | Như trên | Như baseline | 10 view/fold, mean 5 fold | 6,325576 | **4,331841** |
| C3-R2 Z26 HE | Margin 12% + border rescue; giữ tỷ lệ, resize/padding 512, histogram equalization, canvas đen | Như baseline | Mean 5 fold | 6,391855 | — |
| C3-R2 Z26 NO_HE | Như trên nhưng bỏ histogram equalization | Như baseline | Mean 5 fold | **6,324301** | 4,665987 |
| C0 raw C3-R2 | Raw ROI margin 12%, không Z26 | ConvNeXt-Tiny; global average pooling; final LayerNorm; sex embedding | Development Fold 1–2 | 6,365354 pooled | Không mở |
| Raw C3-R2 + bilinear | Raw ROI margin 12% | ConvNeXt-Tiny; bilinear pooling thay global average pooling; sex embedding | Screening | F1 6,385449; F2 lỗi | Không mở |
| C1 fine-tuning + EMA | Raw ROI margin 12% | ConvNeXt-Tiny GAP; layer-wise LR decay; AdamW; warmup/cosine; EMA | Screening Fold 1–2 | PARTIAL | Không mở |
| C2 ConvNeXt V2 | Raw ROI margin 12% | ConvNeXt V2 pretrained + recipe tương ứng; sex embedding | Chưa chạy | — | Không mở |

## Giải nghĩa kỹ thuật

- **Global average pooling:** lấy trung bình từng feature map theo toàn bộ không gian, tạo một vector đặc trưng gọn trước head hồi quy.
- **Bilinear pooling:** lấy tương tác bậc hai giữa các kênh đặc trưng; biểu diễn mạnh hơn nhưng vector/gradient lớn hơn, dễ bất ổn với AMP và dữ liệu nhỏ.
- **Sex embedding:** mã hóa giới tính thành vector học được rồi nối với đặc trưng ảnh trước hồi quy.
- **EMA:** giữ trung bình trượt của trọng số model trong quá trình train để evaluation ổn định hơn; phải khởi tạo/update đúng mới có ý nghĩa.
- **Layer-wise learning-rate decay:** layer gần input nhận learning rate thấp hơn, layer gần head nhận learning rate cao hơn nhằm giữ pretrained features.
- **TTA 10 view:** xoay −10, −5, 0, +5, +10 độ, mỗi góc có flip/no-flip; trung bình prediction.

## Công trình tham khảo

| Công trình | Train/selection | Test và MAE | Kỹ thuật chính | Ghi chú so sánh |
|---|---|---|---|---|
| Shu & Yu, 2025 | RSNA; protocol chi tiết chưa đầy đủ | RSNA test 200: **4,42** | Attention Eraser, Xception, 8 kích thước xương bàn tay | Cùng test, nhưng có đặc trưng đo bổ sung |
| Rassmann et al., Deeplasia, 2024 | RSNA train 12.611; validation 1.425; ensemble 3 model | RSNA test 200: **3,87** | Segmentation/masking, EfficientNet, sex input, TTA/ensemble theo paper | DHA/GDBD là external test, không trộn vào mốc RSNA |
| Bram et al., 2025 | RSNA train+validation, 5-fold; loại 35 ca bất thường | RSNA test 200: **3,68** | 5-fold ensemble và pipeline dự đoán tuổi xương | External data dùng đánh giá/chuyển miền riêng |
| Zhang et al., 2026 | Chỉ RSNA, official split; view dẫn xuất từ cùng ảnh | Validation 1.425: **4,10** | Preprocessing cường độ, global/local views và fusion | Không so trực tiếp với test 200 |

## Kết luận kỹ thuật hiện tại

- Preprocessing Z26 không chuyển lợi ích sang ConvNeXt/C3-R2 trong thí nghiệm hiện có; histogram equalization là thành phần đáng loại nhất.
- ROI margin 8% vẫn là lựa chọn test tốt nhất, dù OOF không tốt nhất và fallback cao.
- Bilinear pooling chưa chứng minh lợi ích; global average pooling tiếp tục là control.
- Hướng có độ tin cậy hợp lý tiếp theo là tối ưu fine-tuning phù hợp ConvNeXt, rồi ConvNeXt V2; mỗi hướng phải thắng matched Fold 1–2 trước.
