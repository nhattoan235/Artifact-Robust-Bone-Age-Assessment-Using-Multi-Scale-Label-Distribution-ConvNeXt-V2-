# P11 Giai đoạn 1 – sex-aware implementation handoff

> **Ngày:** 2026-08-21  
> **Trạng thái:** PASS kỹ thuật; chưa chạy E0/E2 seed 42 trên official validation  
> **Test policy:** không đọc hoặc dùng nhãn RSNA test

## 1. Phạm vi đã hoàn tất

- Thêm `sex_mode = "none" | "embedding" | "dual_output"` vào config khoa học.
- Giữ `embedding` làm mặc định để config và checkpoint E1/P10/P7 cũ tương thích.
- E0 dùng ConvNeXt-Tiny image-only với regressor `768→256→1`.
- E2 dùng ConvNeXt-Tiny, bottleneck chung `768→256`, sau đó hai scalar head
  `female 256→1` và `male 256→1`.
- Mỗi mẫu E2 route theo nhãn M/F; head không được chọn nhận gradient bằng 0.
- Chỉ `convnext_tiny` và `smoke_cnn` được phép dùng mode mới trong Giai đoạn 1.
- Loader inference cũ mặc định `embedding` nếu resolved config chưa có `sex_mode`.

## 2. Capacity head

| Model | Tham số phần conditioning/regression |
|---|---:|
| E0 image-only | 197.121 |
| E1 sex embedding | 201.249 |
| E2 shared dual output | 197.378 |

E2 gần parameter-matched với E0/E1; lợi ích sau này không được quy đơn giản cho
việc tăng mạnh số tham số head.

## 3. Config đã khóa

| Mục đích | Config | Scientific config hash |
|---|---|---|
| E0 full seed 42 | `p1_baseline/configs/p11_e0_image_only_seed42.toml` | `eb6c43a67bcad42623dc9d3b8ef7408392b2610c5057a814420032c1457b72ce` |
| E2 full seed 42 | `p1_baseline/configs/p11_e2_shared_dual_output_seed42.toml` | `60e44d0710c59129ebe1e5de5f76fda0da40694ea30267d30bbf9bf6bd7955a8` |
| E0 CPU smoke | `p1_baseline/configs/p11_e0_cpu_smoke.toml` | `5339d4e70bf6ffa6173c16f539be70b0a95f52dd3a64206d381fa855a8f48f3c` |
| E2 CPU smoke | `p1_baseline/configs/p11_e2_cpu_smoke.toml` | `d9cdf418f45b337695ac18078f2d327a283ffedea223f7f154b392cd71fab86e` |
| E2 GPU smoke | `p1_baseline/configs/p11_e2_gpu_smoke.toml` | `2437502ecdb40f7bfd03b02ae6b1a482c0b24f512c8f7e51f5bd5ce5407cb491` |

Full E0/E2 được kiểm tra tự động và chỉ khác P10-B0 ở đúng `run_id`,
`output_root` và `sex_mode`.

## 4. Bằng chứng xác minh

### 4.1. Static và unit test

- `compileall`: PASS cho `p1_baseline`, `p8_test_ensemble`, `p9_inference`.
- `python -m unittest -v p1_baseline.test_p1`: **24/24 PASS**.
- Test mới bao phủ E0 bất biến theo sex, E1 backward compatibility, E2 routing,
  gradient isolation, invalid shape, unsupported backbone và state-dict round-trip.

### 4.2. Preflight full config

- E0: PASS train/validation count, manifest hash, test-path absence, sample và
  forward shape; config hash `eb6c43a67bca...`.
- E2: PASS cùng toàn bộ check; config hash `60e44d0710c5...`.

### 4.3. CPU trainer smoke

| Run | Kết quả | Global step | Checkpoint |
|---|---|---:|---|
| `P11_E0_CPU_SMOKE_SEED42` | completed | 8 | last/best PASS |
| `P11_E2_CPU_SMOKE_SEED42` | completed | 8 | last/best PASS |

Hai run đều hoàn tất 2 epoch trên 8 train/8 validation và không có stability
warning. MAE smoke không dùng để quyết định khoa học.

### 4.4. GPU interrupt/resume

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU 6 GB.
- PyTorch `2.7.1+cu128`, torchvision `0.22.1+cu128`.
- AMP resolved: BF16.
- Code version: `cbab2ff6dffcee78c51701c5e885cf83f64083e3659e8ee9063e3d8d25051754`.
- Dừng có chủ ý sau global step 1 tại `8/16` ảnh: PASS.
- Resume xác nhận split/config/code hash: CÓ/CÓ/CÓ.
- Optimizer/scheduler/scaler restore: CÓ.
- Hoàn tất epoch tại global step 2; peak VRAM 1.800 MiB; không warning.
- `last.ckpt` và `best_mae.ckpt` khoảng 336 MB, nằm trong thư mục ignored.

### 4.5. E1 regression thật

- Strict-load `P10_B0_P2_CONTROL_CONVNEXT_TINY_SEED42/best_mae.ckpt`: PASS.
- Dùng đúng batch 12, BF16 và thứ tự đổi normalized output sang tháng của trainer.
- 12/12 prediction khớp `val_predictions_best.csv` tuyệt đối.
- `max_abs_diff_months = 0.0`.

Lưu ý: trainer gốc đổi đơn vị tháng khi tensor còn BF16 rồi mới cast float32. Không
thay thứ tự này trong P11 vì sẽ làm lệch baseline đã khóa.

## 5. Artifact runtime

Runtime smoke nằm tại `p11_sex_aware/smoke_runs/` và bị `.gitignore` loại khỏi
source control. Không commit checkpoint 336 MB.

## 6. Quyết định và bước tiếp theo

Giai đoạn 1 **PASS**. Chưa có kết quả hiệu năng E0/E2 và chưa được phép diễn giải
candidate nào tốt hơn.

Bước tiếp theo là Giai đoạn 2:

1. Chạy E0 seed 42 trên official train 12.611 / validation 1.425.
2. Chạy E2 seed 42 với cùng protocol.
3. So sánh E0/E1/E2 paired trên cùng validation ID.
4. Áp dụng gate đã khóa trong `AI_Context/08_SEX_AWARE_EXPERIMENT_PLAN.md`.
