# Lịch sử phương pháp và quyết định

## P0 – audit dữ liệu

- Xác minh protocol RSNA: train 12.611, validation chính thức 1.425, test 200.
- Không có ID/SHA duplicate giữa split; validation tải từ gói chính thức và khớp annotation Deeplasia.
- Khóa manifest/hash; test chỉ kiểm tra cấu trúc và hash, chưa đọc tuổi.
- Artifact chính: `p0_audit/P0_HANDOFF.md`, các manifest trong `p0_audit/outputs/`.

## P1 – hạ tầng baseline

- ConvNeXt-Tiny ImageNet-1K, ảnh 512, grayscale lặp 3 kênh, sex embedding, direct regression, Smooth L1, AdamW/cosine.
- Xây checkpoint/resume nguyên tử, RNG/config/code/data hash, log metrics/warnings và early-stop.
- BF16 được ưu tiên do FP16 smoke ban đầu sinh gradient Inf; smoke/resume cuối PASS.

## P2 – augmentation

- A0 không augmentation: MAE 6,640.
- A1 flip: 6,514.
- A2 flip + xoay ±7°, tịnh tiến/scale nhẹ, brightness/contrast/gamma nhẹ: **6,185**.
- Paired A2–A0 delta -0,455, CI [-0,658; -0,249]; khóa A2.

## P3 – preprocessing

- B1 full-hand background masking bằng Efficient-UNet/TensorMask fallback, không crop/rotate.
- B1 MAE 6,239 so với B0/D0 6,185; delta +0,054, CI chứa 0.
- Theo quy tắc định trước, loại B1 khỏi pipeline chính; giữ `preprocessing=none`.

## P4 – kiến trúc

- D1 ConvNeXtV2-Tiny FCMAE: MAE 31,963 do feature collapse; loại.
- D2 multi-scale fusion: MAE 6,297, không cải thiện primary; loại.
- D3 label-distribution head: fused MAE 6,145 ở seed 42, nhưng CI chứa 0 và cải thiện nhỏ; giữ làm ứng viên phụ.
- Không được suy rộng rằng mọi ConvNeXtV2/LDL đều thất bại; kết luận chỉ áp dụng cho recipe đã khóa.

## P5 – xác nhận seed

- So sánh D0 và D3 trên seed 17/42/123, mỗi seed 1.425 validation.
- D3 fused mean MAE 6,22508 vs D0 6,26229; delta -0,03721, CI [-0,13521; +0,05863], chỉ 2/3 seed tốt hơn.
- D3 regression-only mean delta -0,01932, CI [-0,11270; +0,07069].
- Không đạt ngưỡng cải thiện thực tiễn 0,10 tháng; chọn D0 đơn giản hơn.

## P6 – độ phân giải

- 512 baseline MAE 6,184792; 768 candidate 6,183421; delta -0,001371, paired CI [-0,190703; +0,192127].
- Chênh lệch không đáng kể; khóa 512 để tiết kiệm VRAM/thời gian.

## P7 – final 5-fold V3

- Dùng D0: ConvNeXt-Tiny + A2 + direct regression + sex embedding, 512.
- Train 5 fold trên development pool 14.036; persistent append-only checkpoints trên Drive, resume qua nhiều tài khoản Colab.
- Smoke step 2 rồi 4 qua tài khoản khác PASS; khắc phục lỗi xác thực email/storage và BF16 T4.
- Hoàn tất OOF và audit; kết quả xem `01_STATUS_RESULTS.md`.

## P8 – ensemble test

- Audit input PASS rồi suy luận 5 fold bằng equal-weight mean trên 200 test.
- Chỉ đánh giá sau khi P7 đã khóa; không dùng nhãn test để chọn checkpoint/weight.
- Kết quả kỹ thuật PASS nhưng chưa vượt Bram/Deeplasia; vì test đã được mở, mọi P9 cần thận trọng về tuyên bố confirmatory.

## Bối cảnh inpainting/generative

Hướng so sánh “200 ảnh sinh/ảnh inpainting so với 200 ảnh gốc” không cùng endpoint với bone-age regression chuẩn. Bộ 200 ảnh cleaned artifact-only có thể dùng cho phân tích robustness/ablation paired, nhưng không được thay thế benchmark RSNA hoặc đưa vào train nếu chưa có giao thức riêng, kiểm tra leakage và nhãn tương ứng.

