# Lịch sử phương pháp và quyết định

> Cập nhật: 2026-09-11. Chỉ ghi các mốc giúp giải thích pipeline hiện tại.

| Giai đoạn | Thay đổi chính | Kết quả/Quyết định |
|---|---|---|
| Baseline ban đầu | ConvNeXt-Tiny pretrained, ảnh toàn cảnh 512, global average pooling, final LayerNorm, sex embedding, direct regression, augmentation A2 | Khóa thành E1/P7; OOF 6,316691 |
| P8 | Trung bình dự đoán từ 5 fold E1 | Test 200 MAE 4,730321; không tune trên test |
| P9-I | TTA 10 view và bias correction cross-fitted | TTA OOF 6,210446; giữ TTA. Bias correction không cải thiện; loại |
| EfficientNet/Deeplasia reproduction | EfficientNet-B0, sex embedding, MSE, Deeplasia-style augmentation | Screening validation kém; không mở rộng OOF/test |
| Thí nghiệm C3-ROI V1 | Crop ROI bàn tay margin 8% từ segmentation bbox; fallback toàn ảnh; giữ backbone/sex embedding E1 | Standalone OOF 6,437349 nhưng test 4,337267; ensemble với E1 cải thiện OOF |
| C3-ROI TTA | 10 view cho C3-ROI và ensemble với E1-TTA | OOF ensemble 6,117080; test không hơn C3-ROI-TTA |
| C3-R2 | Tái tạo ROI margin 12% + border rescue để giảm fallback | Fallback giảm nhưng chưa tạo lợi ích test |
| Zhang-2026 preprocessing | Giữ tỷ lệ, resize 512, padding canvas đen, thử histogram equalization | Bản không equalization tốt hơn bản có equalization trên OOF; cả pipeline kém C3-ROI V1 trên test |
| Raw C3-R2 C0 | Đối chứng matched: ConvNeXt-Tiny + global average pooling trên raw R2 | Fold 1–2 pooled 6,365354; dùng làm control cho C1/C2 |
| Bilinear pooling | Thay global average pooling bằng bilinear pooling, phần này chạy FP32 để tránh AMP overflow | Fold 1 = 6,385449; Fold 2 bất ổn. Không promote |
| C1 | Fine-tuning phân tầng theo layer + learning rate decay, AdamW, warmup/cosine, EMA | Đang screening; epoch 1 bất thường, chưa kết luận |
| C2 | ConvNeXt V2 với pretrained/recipe phù hợp | Chưa chạy; chỉ bắt đầu nếu C1 không đạt |

## Những điều đã học được

- Preprocessing từ một backbone/paper không mặc nhiên chuyển lợi ích sang ConvNeXt-Tiny.
- Giảm fallback là mục tiêu kỹ thuật tốt nhưng không đồng nghĩa MAE sẽ giảm; hình học crop và miền ảnh phải được kiểm nghiệm riêng.
- C3-ROI có giá trị lớn nhất hiện nay ở test và diversity khi ensemble, không phải standalone OOF.
- Global average pooling là control mạnh; pooling phức tạp hơn phải thắng trên paired Fold 1–2 trước khi mở rộng.
- Test đã truy cập nhiều lần nên mọi lựa chọn tiếp theo phải dựa vào development validation/OOF.

## Báo cáo lịch sử chuyên sâu

- C3-ROI: 24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md
- Họ model và TTA: 26_MODEL_FAMILY_REPORT.md
- C3-R2/fallback: C3_FALLBACK_REBUILD_REPORT.md
- P14 anatomy-diverse: 00_CURRENT_DECISIONS_P14_HANDOFF_2026_08_25.md và artifact vĩnh viễn dưới NghienCuuChinh/p14_anatomy_diverse/
