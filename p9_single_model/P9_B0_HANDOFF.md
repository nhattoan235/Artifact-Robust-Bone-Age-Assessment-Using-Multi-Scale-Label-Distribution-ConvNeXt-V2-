# P9-B0 handoff — Deeplasia single-model screening

Ngày: 2026-08-21  
Trạng thái: **TẠM DỪNG ĐỂ XEM XÉT**  
Phạm vi: development train/validation; không truy cập test, không chạy OOF/ensemble.

## Mục tiêu

Kiểm tra xem một EfficientNet-B0 512 có thể tái lập gần Deeplasia và vượt
control ConvNeXt P10-B0 hay không. Control validation hiện tại là MAE
**6,184792 tháng**; P7 pooled OOF là **6,316691 tháng**.

## Thay đổi đã triển khai

- \`p1_baseline/model.py\`: EfficientNet-B0, sex embedding 32, hidden layer 256,
  ReLU/dropout 0,2, MSE-compatible regression head.
- Stem 1 kênh được thử bằng cách gộp pretrained RGB stem vào một Conv2d
  grayscale; model nhận tensor pipeline 3 kênh nhưng chỉ dùng channel đầu tiên.
- \`p1_baseline/data.py\`: nhánh \`deeplasia_fancy\` dùng affine gần recipe gốc,
  sharpen và chọn tối đa một phép CLAHE/RandomGamma.
- \`p1_baseline/config.py\` và \`trainer.py\`: thêm MSE, Adam, ReduceLROnPlateau;
  mặc định cũ của các phase trước vẫn giữ nguyên.
- Unit tests 16/16 PASS; preflight PASS; cache mask development-only đủ
  12.611 train và 1.425 validation.

## Kết quả screening

| Run | Biến thể | Best epoch | Best val MAE |
|---|---|---:|---:|
| \`P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_SEED42\` | stem 3 kênh, batch 12, accum 2 | 1 | **8,5293** |
| \`P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_1CH_SEED42\` | stem 1 kênh, batch 12, accum 2 | 2 | **8,5823** |
| \`P9_B0_DEEPLASIA_SINGLE_EFFNET_B0_512_1CH_B24_SEED42\` | stem 1 kênh, batch 24, accum 1 | 1 | **11,0428** |

Các run đều không có NaN/Inf/OOM. Các checkpoint, \`metrics.jsonl\`,
\`val_predictions_best.csv\`, config và run state vẫn được giữ lại.

## Kết luận giai đoạn

P9-B0 **chưa đạt gate**: cả ba biến thể đều kém rõ rệt control 6,184792.
Không có cơ sở chạy OOF, TTA hay ensemble cho EfficientNet-B0 này. Kết quả
âm tính có giá trị vì đã tách được ba nghi vấn: stem RGB/gray và batch size.

Không nên mở test để cứu hoặc chọn biến thể. Tạm thời giữ P10-B0/P7 làm mốc
chính và xem xét khoảng cách tái lập Deeplasia trước bước tiếp theo.

## Các sai khác còn lại cần xem xét

1. Deeplasia dùng custom EfficientNet và cách chuyển pretrained weight sang
   \`in_channels=1\`; phép sum stem hiện tại chỉ là xấp xỉ.
2. Deeplasia xử lý mask/crop/augmentation trong Albumentations; pipeline hiện
   dùng cache PNG + torchvision, chưa bit-exact.
3. Deeplasia có nhiều backbone/config và ensemble; single B0 không đại diện cho
   toàn bộ kết quả bài báo.
4. Cần kiểm tra lại preprocessing cache, crop-to-mask và normalization bằng
   paired visual/QC trước khi kết luận về backbone.

## Quyết định

**Dừng P9-B0 tại đây để review.** Chưa chuyển sang P9-C, chưa tạo OOF mới,
chưa mở \`rsna_test.csv\`.
