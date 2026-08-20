# P1 – Baseline ConvNeXt-Tiny tái lập

P1 chỉ dùng `train_manifest.csv` và `validation_manifest.csv` đã khóa ở P0. Không có đường dẫn hoặc code đọc test ground truth.

## Baseline khoa học

- ConvNeXt-Tiny pretrained ImageNet-1K;
- ảnh grayscale được pad vuông, resize 512×512, lặp ba kênh và chuẩn hóa ImageNet;
- không augmentation và không horizontal flip ở A0;
- sex embedding 16 chiều;
- direct regression trên tuổi đã chuẩn hóa;
- Smooth L1, AdamW, cosine schedule;
- chọn checkpoint duy nhất theo validation MAE chính thức.

P1 xây hạ tầng và smoke test. Run dài A0 thuộc P2 và chỉ bắt đầu sau khi người dùng xem cấu hình/thời gian dự kiến.

## Chạy

```powershell
python -m p1_baseline.preflight --config p1_baseline/configs/p1_a0.toml
python -m p1_baseline.train --config p1_baseline/configs/p1_a0.toml
```

Tiếp tục sau khi dừng:

```powershell
python -m p1_baseline.train --config p1_baseline/configs/p1_a0.toml --resume p1_baseline/runs/P1_A0_CONVNEXT_TINY_SEED42/last.ckpt
```

Không sửa config khoa học giữa hai lần chạy. `num_workers`, tần suất log/checkpoint và đường dẫn output là các tham số vận hành duy nhất được phép đổi khi resume.

## Artifact mỗi run

- `config_resolved.yaml`, `environment.txt`;
- `train.log`, `warnings.log`, `metrics.jsonl`;
- `last.ckpt`, `best_mae.ckpt`, `periodic/`, `best/`;
- `run_state.json`, `val_predictions_best.csv`.

Checkpoint được ghi nguyên tử và đọc kiểm tra lại trước khi thay file đích. Mỗi checkpoint chứa model, optimizer, scheduler, scaler, RNG, epoch/step, best metric, early-stop state, split hash, config hash và code version.
