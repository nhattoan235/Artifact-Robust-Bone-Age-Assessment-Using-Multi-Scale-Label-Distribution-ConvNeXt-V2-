# Bàn giao P4 – Ablation kiến trúc

**Trạng thái:** D1 early stop hợp lệ và bị loại; D0/ConvNeXt-Tiny thắng, được phép dùng làm backbone cho D2.

## Giao thức đã khóa

- D0 tái sử dụng `P2_A2_LIGHT_FLIP_CONVNEXT_TINY_SEED42`; không train lại.
- D0 tốt nhất: epoch 12, validation MAE `6,18479` tháng, RMSE `8,48647` tháng.
- D1 chỉ đổi backbone thành `convnextv2_tiny.fcmae_ft_in1k`.
- Trọng số D1 dùng ImageNet-1K để không tạo thêm nhiễu do pretraining ImageNet-22K so với D0.
- Cả hai dùng cùng split, seed 42, ảnh 512, preprocessing `none`, augmentation A2, sex embedding, regression head, optimizer, scheduler, loss và early stopping.
- Không sử dụng test set để chọn mô hình.

## D1

- Run ID chính thức: `P4_D1_CONVNEXTV2_TINY_FCMAE_IN1K_B6A6_SEED42`
- Config: `p1_baseline/configs/p4_d1_convnextv2_tiny_b6a6.toml`
- Config hash: `4a32683c6d2662db93ca86ca16a237f67cde7975fd608e6afa6d48ff6460191a`
- Code version: `9ba6b0f118eeac1ce26a65c8bf2fb7c9eba61bcfaf8309ec9f16f53a82db8901`
- Train manifest: `7328667e6822ab074d442155e33eba89606861bb13bbf804be8a1138c78f5285`
- Validation manifest: `f650a20432b557405035d12c52716a6c40fdf20e1343ac3ad0cfb7cd8db21631`
- Thiết bị: RTX 4050 Laptop GPU 6 GB, BF16.
- Checkpoint định kỳ: mỗi 500 optimizer step hoặc 20 phút.
- Early stopping: patience 8 epoch theo validation MAE.
- Preflight: PASS 7/7.
- Unit test: PASS 11/11.
- Resume smoke test: PASS; split/config/code hash và optimizer/scheduler/scaler đều phục hồi.
- Micro-batch `6`, gradient accumulation `6`, giữ nguyên effective batch `36` như D0. Cấu hình thử `12×3` bị dừng sau batch 25 vì chạm khoảng `5.736/6.141 MiB` VRAM và chỉ đạt `2,41 ảnh/s`, cho thấy paging sang shared memory. Run thử này chỉ là bằng chứng kỹ thuật, không dùng để đánh giá. Smoke test `6×6` giảm thời gian một optimizer step từ khoảng 28 xuống 10 giây.
- Run chính thức ổn định đến batch 125: throughput tăng lên `14,94 ảnh/s`, VRAM khoảng `4.658/6.141 MiB`, loss giảm `32,9405 → 26,1462`, grad norm trước clipping giảm còn `9,9666`, skipped batch `0`, không có warning/NaN/Inf/OOM.
- Trọng số timm có giấy phép `CC-BY-NC-4.0`; phù hợp nghiên cứu học thuật, cần ghi attribution trong báo cáo và không mặc định phù hợp triển khai thương mại.

## Điểm kiểm tra sau mỗi epoch

Theo dõi validation MAE/RMSE, MAE theo giới và nhóm tuổi, prediction range/collapse, gradient norm, skipped batches và warning NaN/Inf. Chỉ quyết định bằng validation; test vẫn khóa.

Khi D1 dừng, chạy paired comparison trên đúng 1.425 ảnh validation với D0, bao gồm delta MAE `D1-D0` và bootstrap 95% CI. Backbone thắng mới được đưa sang D2; không mặc định ConvNeXtV2 thắng.

## Kết quả và quyết định D1

- Early stop tại epoch 13 sau 8 epoch không cải thiện; best epoch 5.
- D1: MAE `31,96281`, RMSE `42,43807`, median AE `21,5` tháng.
- D0: MAE `6,18479`, RMSE `8,48647` tháng.
- Delta MAE `D1-D0 = +25,77802` tháng; paired bootstrap 95% CI `[+24,33264; +27,26304]` trên 10.000 lần lấy mẫu.
- D1 tốt hơn D0 trên 206 ảnh, kém hơn trên 1.198 ảnh và hòa 21 ảnh.
- Feature diagnostic: pretrained backbone ban đầu có mean pair distance `8,30168`; checkpoint D1 chỉ còn `0,000630`, xác nhận feature collapse. Dự đoán gần như chỉ còn hai mức theo giới: khoảng `122,31` tháng cho nữ và `149,95` tháng cho nam.
- Kết luận: loại ConvNeXtV2-Tiny khỏi pipeline chính theo đúng ablation đã khóa. Không được diễn giải rằng mọi cách fine-tune ConvNeXtV2 đều thất bại; kết luận chính xác là ConvNeXtV2-FCMAE không tương thích với recipe D0 trong thí nghiệm kiểm soát này.
- D2 phải dùng ConvNeXt-Tiny thắng cuộc + multi-scale fusion; mọi thành phần còn lại giữ theo D0.
- Báo cáo paired: `p4_architecture/D1_vs_D0_paired.json`.

## D2 – Multi-scale fusion

**Trạng thái:** early stop hợp lệ; multi-scale fusion bị loại khỏi pipeline chính.

- Run ID: `P4_D2_CONVNEXT_TINY_MULTISCALE_SEED42`.
- Config: `p1_baseline/configs/p4_d2_convnext_tiny_multiscale.toml`.
- Config hash: `e3308f702cfc7003712ae9369b84473eaa8f7ab1e64e222e28aadd82f44a9b9a`.
- Code version: `4e4d33aed7785c8aad5c0ec46bdcbef3f1f9b02639ae64a0592b5dd2024c73eb`.
- Backbone: ConvNeXt-Tiny ImageNet-1K giống D0.
- Lấy vector sau bốn stage có số chiều `96 + 192 + 384 + 768 = 1.440`.
- Fusion tuyến tính `1.440 → 768`, không bias. Trọng số khởi tạo bằng 0 cho ba stage sớm và identity cho 768 chiều stage cuối; chuẩn hóa stage cuối sao chép trọng số pretrained của D0.
- Regression head sau fusion vẫn nhận 768 chiều ảnh + sex embedding 16 chiều như D0; hidden dimension vẫn 256.
- Batch 12, accumulation 3, effective batch 36; các trường khoa học khác giữ nguyên D0.
- Bổ sung cảnh báo prediction collapse riêng cho từng giới bằng `prediction_std_by_sex`.
- Unit test PASS 12/12, preflight PASS 7/7, smoke test và resume test PASS.
- Khởi động dài đến batch 125 ổn định: loss giảm `28,3343 → 20,9030`, throughput `31,33 ảnh/s`, grad norm trước clipping `11,2775`, reserved VRAM `4.420 MiB`, skipped batch `0`, không warning/NaN/Inf/OOM.
- Checkpoint/log: `p1_baseline/runs/P4_D2_CONVNEXT_TINY_MULTISCALE_SEED42` và `p1_baseline/P4_D2_launcher.stdout.log`.

Sau early stop/kết thúc, bắt buộc so sánh paired D2–D0 trên đúng 1.425 ảnh validation và bootstrap 95% CI. Không tự động giữ multi-scale chỉ vì đây là mô hình phức tạp hơn.

### Kết quả và quyết định D2

- Early stop tại epoch 22 sau 8 epoch không cải thiện; best epoch 14.
- D2: MAE `6,29697`, RMSE `8,49262`, median AE `5,0` tháng.
- D0: MAE `6,18479`, RMSE `8,48647`, median AE `5,0` tháng.
- Delta MAE `D2-D0 = +0,11218` tháng; paired bootstrap 95% CI `[-0,05414; +0,27870]` trên 10.000 lần lấy mẫu.
- D2 tốt hơn D0 trên 616 ảnh, kém hơn trên 644 ảnh và hòa 165 ảnh.
- Accuracy ±6 tháng giảm từ `63,44%` xuống `62,32%`; accuracy ±12 tháng tăng nhẹ từ `86,67%` lên `87,02%` nhưng đây không phải primary endpoint.
- D2 cải thiện nhẹ nhóm nam (`6,01245 → 5,94607`) và tuổi 0–59 (`7,28275 → 6,85563`), nhưng xấu hơn nhóm nữ (`6,38912 → 6,71299`) và phần lớn nhóm tuổi còn lại.
- CI chứa 0 nên không kết luận multi-scale gây hại có ý nghĩa thống kê. Tuy nhiên D2 không cải thiện primary MAE và không đạt tiêu chí giữ đã khóa trước; loại khỏi pipeline chính.
- D3 phải dùng D0/ConvNeXt-Tiny direct-regression làm nền rồi thêm label-distribution head.
- Báo cáo paired: `p4_architecture/D2_vs_D0_paired.json`.

## D3 – Label-distribution learning

**Trạng thái:** early stop hợp lệ; D3 cải thiện số học rất nhỏ nhưng chưa chứng minh vượt D0. P4 hoàn thành, D0 và D3 là top-2 sang P5.

- Run ID: `P4_D3_CONVNEXT_TINY_LDL_SIGMA2_LAMBDA02_SEED42`.
- Config: `p1_baseline/configs/p4_d3_convnext_tiny_ldl.toml`.
- Config hash: `60369b2504308169629001acfacbbd4d5ead47b84e0646da96bcd90fbff9ba91`.
- Code version: `62c26b5b4e05b87bd058ede20e2b5c76e9636af80bd44d9a107f65838496a83f`.
- Nền kiến trúc: D0 ConvNeXt-Tiny direct regression; không dùng D1 hoặc D2.
- Head phụ dự đoán 229 lớp tương ứng tháng tuổi `0–228`.
- Nhãn mềm Gaussian `sigma=2,0` tháng, được chuẩn hóa tổng xác suất bằng 1 kể cả ở biên 0/228.
- Loss train: `SmoothL1 regression + 0,2 × soft cross-entropy`.
- Primary inference đã khóa: `0,5 × regression + 0,5 × distribution expectation`.
- Prediction CSV lưu thêm `regression_months` và `distribution_months` để chẩn đoán; không dùng để thay đổi primary rule sau khi xem validation.
- Unit test PASS 14/14, preflight PASS 7/7, smoke test và resume test PASS.
- Khởi động dài đến batch 175 ổn định: regression loss giảm `30,3930 → 20,3898`, LDL loss giảm `5,3691 → 4,6421`, throughput khoảng `32,70 ảnh/s`, reserved VRAM `4.406 MiB`, skipped batch `0`, không warning/NaN/Inf/OOM.
- Log: `p1_baseline/P4_D3_launcher.stdout.log`.

Sau early stop/kết thúc, paired comparison chính là D3 fused prediction so với D0 trên 1.425 ảnh validation, bootstrap 95% CI. Regression-only và distribution-only chỉ được báo cáo như phân tích cơ chế, không thay primary endpoint đã khóa.

### Kết quả và quyết định D3

- Early stop tại epoch 21 sau 8 epoch không cải thiện; best epoch 13.
- Primary D3 fused: MAE `6,14541`, RMSE `8,42811`, median AE `4,62752` tháng.
- D0: MAE `6,18479`, RMSE `8,48647`, median AE `5,0` tháng.
- Delta MAE primary `D3-D0 = -0,03938` tháng; paired bootstrap 95% CI `[-0,19707; +0,12156]` trên 10.000 lần lấy mẫu.
- D3 fused tốt hơn D0 trên 726 ảnh và kém hơn trên 699 ảnh.
- D3 fused giảm nhẹ MAE/RMSE/median AE nhưng accuracy ±6/±12/±18 đều thấp hơn D0; cải thiện MAE nhỏ hơn ngưỡng thực tiễn 0,10 tháng và CI chứa 0.
- Kết luận primary: chưa có bằng chứng D3 vượt D0; không được tuyên bố label-distribution learning thắng từ seed 42.

Phân tích cơ chế đã định danh trước là secondary/exploratory:

- Regression-only của checkpoint D3: MAE `6,12496`, RMSE `8,43649`, accuracy ±6 `65,05%`.
- Distribution-only: MAE `6,38342`, RMSE `8,60854`.
- Regression-only so với D0: delta MAE `-0,05984`, bootstrap 95% CI `[-0,21211; +0,09625]`; vẫn chứa 0 và dưới ngưỡng 0,10 tháng.
- Điều này gợi ý auxiliary LDL có thể cải thiện biểu diễn của regression head, còn trộn distribution expectation 0,5 làm giảm một phần lợi ích. Vì regression-only được chọn sau khi xem validation seed 42, nó không được tráo thành primary P4.
- Trước khi chạy seed mới ở P5, có thể khóa prospectively D3 regression-only làm candidate inference chính của kiến trúc D3; fused output phải tiếp tục được báo cáo để bảo toàn dấu vết primary P4.
- Báo cáo primary: `p4_architecture/D3_vs_D0_paired.json`.
- Báo cáo exploratory: `p4_architecture/D3_regression_only_vs_D0_exploratory.json`.

## Kết luận P4

| Run | MAE tháng | Delta với D0 | Quyết định |
|---|---:|---:|---|
| D0 ConvNeXt-Tiny | 6,18479 | 0 | Top-2, baseline |
| D1 ConvNeXtV2-Tiny | 31,96281 | +25,77802 | Loại: feature collapse |
| D2 Multi-scale | 6,29697 | +0,11218 | Loại: không cải thiện primary |
| D3 LDL fused | 6,14541 | -0,03938 | Top-2, cần xác nhận nhiều seed |

P4 không tạo ra cải thiện đủ lớn và chắc chắn để thay D0 ngay. P5 phải xác nhận D0 và D3 qua tổng cộng ba seed trước khi giữ hoặc loại LDL.

## Tiếp tục sau khi máy bị dừng

```powershell
& 'C:\Users\ASUS\Documents\tao_sinh_anh_x_ray_hand\.boneage_env\Scripts\python.exe' -m p1_baseline.train --config 'p1_baseline/configs/p4_d1_convnextv2_tiny_b6a6.toml' --resume 'p1_baseline/runs/P4_D1_CONVNEXTV2_TINY_FCMAE_IN1K_B6A6_SEED42/last.ckpt'
```

Không sửa các file Python trong `p1_baseline` hoặc các trường khoa học của config trước khi resume, vì code/config hash sẽ cố ý từ chối checkpoint không tương thích.
