# P9-A — Deeplasia/Bram-inspired preprocessing ablation

P9 là development-only: script không nhận đường dẫn test và không thay đổi P3/P7/P8.

Các biến đã khóa:

1. `deeplasia_mask_v1`: official mask (`eff_unet`, fallback `Tensormask`), nền 0,
   trừ percentile thứ nhất của foreground.
2. `deeplasia_mask_histogram_v1`: các bước trên cộng histogram equalization CDF
   của foreground.

Cache lưu SHA input/output, nguồn mask, percentile và `visual_top24.jpg` để QC.
Chỉ train khi summary PASS và đã xem montage.

Thứ tự: A0 raw L1 -> A1 mask -> A2 mask + histogram. Các run dùng cùng model,
loss, augmentation; chỉ giữ candidate nếu paired validation có CI MAE có lợi, sau đó
mới xác nhận OOF. Không xem test giữa các bước.
