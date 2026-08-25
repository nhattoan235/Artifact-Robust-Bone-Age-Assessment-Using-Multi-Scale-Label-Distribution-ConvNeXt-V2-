# Bone-age baseline v1

Quy trình theo từng giai đoạn: [BASELINE_QUY_TRINH.md](BASELINE_QUY_TRINH.md)

Hướng dẫn chạy Colab: [COLAB_RUN_A2_LIGHT_FLIP.md](COLAB_RUN_A2_LIGHT_FLIP.md)

Baseline này là pipeline độc lập cho vòng thử nghiệm mới. Nó không sửa hoặc ghi đè
artifact P7/P8.

## Mục tiêu

- Dự đoán bone age theo tháng trên RSNA Pediatric Bone Age Challenge.
- Giữ sex embedding, ConvNeXt-Tiny và A2 augmentation làm lõi đã được kiểm chứng.
- Tái lập được `p7_reference` trước khi thử `bram_lite`.
- Chọn cấu hình bằng development 5-fold OOF; không đọc tuổi của RSNA test.
- Ghi checkpoint, manifest, OOF prediction, subgroup metrics và bootstrap CI.

## Hai recipe

| Recipe | Ý nghĩa |
|---|---|
| `p7_reference` | Resize trực tiếp toàn ảnh về 512, không foreground crop; affine nhẹ + ColorJitter; direct Smooth L1 regression. Dùng làm mốc kiểm tra pipeline hiện tại. |
| `a2_light_flip` | Giống `p7_reference` về preprocessing và model, nhưng augmentation chỉ là RandomHorizontalFlip. Dùng để kiểm tra riêng giả thuyết light flip. |
| `bram_lite` | Crop phần foreground không màu nền, giữ tỷ lệ bằng letterbox, autocontrast; vẫn dùng A2 và cùng kiến trúc/loss. Đây là candidate để kiểm tra giả thuyết từ Bram/Deeplasia. |

Không đưa segmentation mask vào baseline mặc định vì nhánh B1 trước đây chưa cải thiện OOF/validation.

## Cấu trúc output

```text
outputs/<run_name>/
  config.json
  data_manifest.json
  fold_0/best.pt, training_log.csv, val_predictions.csv
  ...
  oof_predictions.csv
  oof_report.json
```

## Chạy kiểm tra nhanh ở workspace hiện tại

```powershell
python project/baseline_v1/boneage_baseline.py `
  --mode smoke `
  --data-root data_goc `
  --recipe p7_reference `
  --no-pretrained
```

## Chạy một official train/validation trước

```powershell
python project/baseline_v1/boneage_baseline.py `
  --mode official `
  --device cuda `
  --data-root data_goc `
  --recipe p7_reference `
  --output-root project/baseline_v1/outputs `
  --run-name official_p7_reference_seed42 `
  --pretrained
```

## Chạy OOF 5-fold

```powershell
python project/baseline_v1/boneage_baseline.py `
  --mode oof `
  --data-root data_goc `
  --recipe p7_reference `
  --output-root project/baseline_v1/outputs `
  --run-name oof_p7_reference_seed42 `
  --pretrained
```

Sau khi `p7_reference` chạy xong và kiểm tra được protocol, chạy candidate:

```powershell
python project/baseline_v1/boneage_baseline.py `
  --mode official `
  --device cuda `
  --data-root data_goc `
  --recipe a2_light_flip `
  --output-root project/baseline_v1/outputs `
  --run-name official_a2_light_flip_seed42 `
  --pretrained `
  --amp fp16
```

Candidate `a2_light_flip` chỉ thay đổi augmentation, nên kết quả được so sánh trực tiếp với `p7_reference`.

Sau đó mới chạy candidate foreground:

```powershell
python project/baseline_v1/boneage_baseline.py `
  --mode oof `
  --data-root data_goc `
  --recipe bram_lite `
  --output-root project/baseline_v1/outputs `
  --run-name oof_bram_lite_seed42 `
  --pretrained
```

## Quy tắc quyết định

1. Chỉ giữ candidate nếu pooled OOF MAE giảm, không có subgroup collapse và kết quả
   ổn định khi lặp ít nhất 3 seed/fold.
2. Không dùng `rsna_test.csv`, không dùng file nhãn test và không tối ưu ensemble trên
   test. Script inference test chỉ đọc CSV sex của test và không nhận tham số ground truth.
3. Chưa được gọi là vượt Bram/Deeplasia nếu chưa có OOF/external holdout tương thích.

## Gợi ý cấu hình GPU

Trên T4/P100: `--img-size 512 --batch-size 8 --workers 2 --epochs 100`.
Nếu thiếu VRAM, giảm batch size xuống 4 và tăng `--accum-steps 2`.
BF16 được tự động dùng trên CUDA nếu thiết bị hỗ trợ; CPU smoke không dùng AMP.
