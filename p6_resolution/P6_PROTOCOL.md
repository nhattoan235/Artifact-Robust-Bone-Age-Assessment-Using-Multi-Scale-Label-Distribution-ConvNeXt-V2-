# P6 – So sánh độ phân giải 512 và 768

**Trạng thái:** Hoàn thành; giữ độ phân giải 512 cho final 5-fold.

## Giả thuyết và đối chứng

- Đối chứng 512: `P2_A2_LIGHT_FLIP_CONVNEXT_TINY_SEED42`, validation MAE `6,18479` tháng.
- Candidate 768: `P6_D0_CONVNEXT_TINY_768_SEED42`.
- Chỉ thay đổi kích thước input từ 512 lên 768.
- Giữ ConvNeXt-Tiny, direct regression, sex embedding, augmentation A2, optimizer, scheduler, effective batch 36, seed 42 và official validation 1.425 ảnh.
- Test set tiếp tục bị khóa.

## Cấu hình vận hành

- Micro-batch `4`, gradient accumulation `9`, effective batch `36`.
- Benchmark 3 step: batch 4 đạt khoảng `13,80 ảnh/s`, peak reserved `3.374 MiB`; batch 5 không nhanh hơn và dùng `4.120 MiB`.
- Preflight PASS 7/7; unit test PASS 14/14.
- Stop/resume hai optimizer step PASS; split/config/code hash và optimizer/scheduler/scaler đều phục hồi.
- Config hash: `0e80cf6f780f637216c68262525033f6f76c78f1c11bb27b41f599bb74598115`.
- Config: `p1_baseline/configs/p6_d0_768_seed42.toml`.
- Log: `p1_baseline/P6_D0_768_launcher.stdout.log`.

## Quy tắc quyết định

Sau early stopping, so sánh paired trên cùng 1.425 ảnh và bootstrap 10.000 lần. Chỉ giữ 768 khi MAE cải thiện đủ rõ mà không tạo bất ổn RMSE/giới/nhóm tuổi. Nếu chênh lệch nhỏ hơn 0,10 tháng và CI không thuyết phục, giữ 512 vì đơn giản và nhanh hơn. Không thử 1024 trừ khi 768 cho tín hiệu cải thiện.

## Kết quả và quyết định

- Run 768 early stop tại epoch 32; best epoch 24.
- 512: MAE `6,18479`, RMSE `8,48647` tháng.
- 768: MAE `6,18342`, RMSE `8,37426` tháng.
- Delta MAE `768-512 = -0,00137` tháng; paired bootstrap 95% CI `[-0,19070; +0,19213]` trên 10.000 lần lấy mẫu.
- 768 tốt hơn trên 635 ảnh, kém hơn trên 654 ảnh và hòa 136 ảnh.
- RMSE của 768 giảm `0,11221` tháng nhưng primary MAE gần như không đổi, CI rộng và chứa 0; chi phí tính toán/input tăng 2,25 lần theo số pixel.
- Theo quy tắc khóa trước, **giữ 512** cho final 5-fold. Không thử 1024.
- Báo cáo máy đọc: `p6_resolution/P6_768_vs_512_paired.json`.
