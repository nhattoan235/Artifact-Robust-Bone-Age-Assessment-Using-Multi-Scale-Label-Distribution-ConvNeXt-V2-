# BÁO CÁO BÀN GIAO P1 – BASELINE CONVNEXT-TINY TÁI LẬP

**Ngày hoàn tất:** 2026-08-13  
**Trạng thái:** PASS – hoàn thành hạ tầng baseline, chưa chạy A0 dài  
**Bước tiếp theo được phép:** P2 – chạy A0–A2 và khóa augmentation  
**Đã đọc hoặc sử dụng nhãn test:** KHÔNG

## 1. Baseline đã khóa cho A0

- ConvNeXt-Tiny pretrained ImageNet-1K V1.
- Input 512×512; ảnh grayscale được pad vuông để không bóp méo tỷ lệ, lặp ba kênh và chuẩn hóa ImageNet.
- Không horizontal flip và không augmentation ở A0.
- Sex embedding 16 chiều.
- Direct regression trên tuổi đã chuẩn hóa bằng mean/SD của train.
- Smooth L1 với beta tương đương 3 tháng.
- AdamW, learning rate `2e-4`, weight decay `0.05`, cosine schedule.
- Batch vật lý 4, gradient accumulation 8; effective batch 32.
- Tối đa 35 epoch; early stopping patience 8 theo validation MAE.
- Config khoa học: `configs/p1_a0.toml`.
- Config hash: `1f55956cc0f8eede0a9752b46d5ce38ff07e72142242c3046676c562649adea0`.

P1 chỉ dùng train 12.611 và validation chính thức 1.425. Source/config P1 không chứa đường dẫn test ground truth.

## 2. Checkpoint và resume

Checkpoint chứa:

- model, optimizer, scheduler và AMP scaler;
- epoch, batch, optimizer step và vị trí mẫu trong epoch;
- tổng/count loss đang tích lũy trong epoch;
- best MAE/epoch và early-stopping counter;
- RNG Python, NumPy, PyTorch CPU/CUDA;
- seed, train/validation fingerprint, scientific config hash và code hash.

`last.ckpt` được lưu nguyên tử và đọc kiểm tra lại:

- sau mỗi epoch;
- mỗi 500 optimizer step;
- hoặc tối đa mỗi 20 phút tại optimizer boundary.

Ngoài ra có `best_mae.ckpt`, tối đa hai periodic checkpoint gần nhất và ba best checkpoint. Resume từ chối khi split, config hoặc code hash không khớp.

Bài kiểm tra dừng sau optimizer step 1 rồi resume đã hoàn tất. So với run không bị ngắt:

```text
model_tensors_bitwise_equal = true
training_state_equal = true
global_step = 8 ở cả hai run
```

## 3. Log và cảnh báo

Mỗi run tạo:

- `train.log`, `warnings.log`, `metrics.jsonl`;
- `environment.txt`, `config_resolved.yaml`, `run_state.json`;
- `val_predictions_best.csv`;
- `last.ckpt`, `best_mae.ckpt`, `periodic/`, `best/`.

Log batch có loss, learning rate, gradient norm, throughput, ETA, GPU allocated/reserved và skipped batches. Log epoch có MAE, RMSE, median AE, accuracy ±6/±12/±18 tháng, subgroup giới tính/tuổi, miền dự đoán, thời gian epoch và peak VRAM.

Cảnh báo đỏ dừng an toàn đối với NaN/Inf, prediction collapse/out-of-range nghiêm trọng hoặc hash checkpoint không khớp. Snapshot sau mỗi epoch đưa ra `TIẾP TỤC`, `THEO DÕI` hoặc `TẠM DỪNG KIỂM TRA`.

## 4. Phát hiện quan trọng từ smoke test

Lần GPU smoke đầu bằng FP16 mặc định tạo gradient Inf ngay step đầu. Cơ chế cảnh báo đỏ đã dừng trước khi optimizer cập nhật trọng số.

Đã sửa AMP:

- ưu tiên BF16 khi GPU hỗ trợ;
- fallback FP16 với initial loss scale 4096 trên GPU không hỗ trợ BF16.

Smoke cuối bằng ConvNeXt-Tiny pretrained, 512×512, batch 4, BF16:

- peak reserved VRAM: 1.570 MiB trên RTX 4050 6 GB;
- gradient cuối hữu hạn: 3,6447;
- không OOM, không skipped batch, không cảnh báo;
- checkpoint đọc lại bằng process mới, đủ key và toàn bộ tensor model hữu hạn.

MAE 53,5625 của smoke chỉ tính trên 8 ảnh sau một optimizer step và **không được dùng làm kết quả khoa học**.

## 5. Kết quả kiểm thử

- 6/6 unit tests đạt.
- Preflight đúng 12.611 train, 1.425 validation và fingerprint P0.
- Forward ConvNeXt-Tiny pretrained trên ảnh thật 512 đạt.
- Stop/resume bitwise đạt.
- GPU forward/backward BF16 batch 4 đạt.
- Đầy đủ checkpoint, log, prediction và environment artifact.
- Không có đường dẫn test bị khóa trong source/config P1.

## 6. Quyết định

P1 hoàn thành. Chưa có MAE baseline thực vì chưa chạy toàn bộ A0; việc đó thuộc P2. Không khởi động run dài tự động để người dùng còn lựa chọn môi trường Colab hoặc RTX 4050 và kiểm tra thời gian.

```text
Phase: P1
Mục tiêu: Xây baseline ConvNeXt-Tiny tái lập
Kết quả: PASS
Run xác minh: P1_SMOKE_RESUME_VERIFIED; P1_SMOKE_UNINTERRUPTED_VERIFIED; P1_GPU_SMOKE_CONVNEXT512_B4_VERIFIED
Data hash: Train 7328667e...f5285; Validation f650a204...21631
Config hash A0: 1f55956c...dea0
Quyết định: Cho phép sang P2
Bước tiếp theo: Chạy A0 đầy đủ, đánh giá validation rồi mới quyết định A1/A2
Test set: KHÔNG chạm nhãn/metric
```
