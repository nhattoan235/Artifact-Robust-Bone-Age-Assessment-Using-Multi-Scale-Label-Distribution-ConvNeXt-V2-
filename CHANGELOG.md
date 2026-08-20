# Changelog — Dự án cải thiện mô hình dự đoán tuổi xương

> File này ghi lại lịch sử các thử nghiệm, thay đổi pipeline, và kết quả đo được. Mỗi entry mới thêm lên đầu (mới nhất trên cùng). Tham chiếu chi tiết lộ trình/nguồn ở `PROJECT_CONTEXT.md`.

## Định dạng entry

```
## [YYYY-MM-DD] Giai đoạn N — Tên thử nghiệm

**Thay đổi:** mô tả ngắn gọn thay đổi so với lần chạy trước
**Kết quả:**
| Metric | Giá trị | So với baseline (6.26 mo) |
|---|---|---|
| MAE | | |
| RMSE | | |
| AUC (gender, nếu có) | | |

**Nhận xét:** điều gì hiệu quả / không hiệu quả, giả thuyết cho bước tiếp theo
**File/artifact liên quan:** đường dẫn checkpoint, notebook, log
```

---

## [2026-08-09] Giai đoạn 1 — Ensemble đa kiến trúc + RadImageNet (code, chưa chạy)

**Trạng thái:** Đã triển khai pipeline (`bone_age_phase1/`), CHƯA chạy trên GPU/dataset thật.
Số liệu MAE/RMSE dưới đây sẽ được điền sau khi chạy `train.py` — không dùng số ước tính.

**Thay đổi:**
- Ensemble 3 kiến trúc: ResNet50 + EfficientNet-B4 + DenseNet121, kết hợp bằng
  stacking (LinearRegression trên out-of-fold predictions), có fallback "average".
- Trọng số pretrained: ResNet50 + DenseNet121 dùng RadImageNet (PyTorch .pt chính
  thức từ BMEII-AI/RadImageNet); EfficientNet-B4 dùng ImageNet.
- Giữ nguyên các yếu tố khác của baseline để so sánh công bằng: input 320×320 RGB
  (channel duplication), foreground cropping, augmentation affine/contrast/sharpen,
  Smooth L1 loss, Adam + cosine annealing, gradient accumulation 10 bước, 50 epochs,
  5-fold CV, sex-specific.

**Giới hạn kỹ thuật đã xác minh (quan trọng — cần nêu trong Methods/Limitations):**
Repo chính thức RadImageNet (https://github.com/BMEII-AI/RadImageNet) chỉ phát hành
trọng số cho ResNet50, DenseNet121, InceptionResNetV2, InceptionV3 — không có
EfficientNet-B4. Do đó nhánh EfficientNet-B4 trong ensemble bắt buộc dùng ImageNet
pretrained, không phải RadImageNet như mô tả tổng quát ban đầu trong `PROJECT_CONTEXT.md`.
Phương án thay thế nếu muốn EfficientNet có pretrained y tế: RadiologyNET
(Napravnik et al., Scientific Reports 2025) — đã công bố EfficientNetB4 pretrained và
test trực tiếp trên RSNA Bone Age, nhưng là nguồn dữ liệu khác RadImageNet.

**Kết quả:**
| Metric | Giá trị | So với baseline (6.26 mo) |
|---|---|---|
| MAE | *(chưa chạy)* | |
| RMSE | *(chưa chạy)* | |
| AUC (gender) | — (Giai đoạn 1 không đổi pipeline gender) | |

**Nhận xét:** Chờ chạy thật trên dataset RSNA + GPU của tác giả để điền kết quả.
**File/artifact liên quan:** `bone_age_phase1/{config,dataset,models,ensemble,train}.py`,
output dự kiến: `oof_predictions.csv`, `phase1_report.json`.

---

## [Chưa bắt đầu] Giai đoạn 0 — Tái lập baseline

**Trạng thái:** Chờ triển khai.
**Việc cần làm:** tái tạo ResNet50 ensemble sex-specific theo đúng mô tả Methods của Matsuoka et al., xác nhận MAE ≈ 6.26 tháng trên test set gốc RSNA trước khi so sánh bất kỳ cải tiến nào.

---

*(Các entry kết quả thực nghiệm sẽ được thêm vào đây khi từng giai đoạn được chạy.)*
