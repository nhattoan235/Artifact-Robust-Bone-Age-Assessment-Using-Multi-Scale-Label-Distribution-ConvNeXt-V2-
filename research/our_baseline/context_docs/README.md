# Bone Age Estimation – Cải Tiến Inpainting

**Đề tài**: Dự đoán tuổi xương từ ảnh tạo sinh X-quang bàn tay

## Cấu trúc thư mục

```
project/
├── data/           # RSNA dataset + ảnh synthetic
│   ├── rsna/       # RSNA Bone Age dataset (training + 200 ảnh test)
│   └── synthetic/  # Ảnh tạo sinh (Synthetic Inpainted Hand X-rays)
│
├── baseline/       # Tái lập Cấu hình A (Matsuoka) – MAE baseline 30.11
│
├── huong_B/        # Saliency-Guided Adaptive Masking
│   ├── scripts/    # Grad-CAM++, artifact detection, mask generation
│   ├── masks/      # Saliency masks đã tạo
│   └── outputs/    # Ảnh sau inpainting với saliency mask
│
├── huong_D/        # Frequency-Domain Texture-Preserving Blending
│   ├── scripts/    # Laplacian pyramid blending, Poisson blending
│   └── outputs/    # Ảnh sau frequency blending
│
├── huong_E/        # Anatomical Landmark Consistency QA
│   ├── scripts/    # Landmark detection, NLE computation, correlation
│   ├── landmarks/  # Toạ độ landmark (JSON/CSV)
│   └── qa_reports/ # Báo cáo QA
│
├── models/         # Checkpoint mô hình
│   ├── bone_age/   # Mô hình dự đoán bone age (ResNet50/Inception v3)
│   └── gender/     # Mô hình dự đoán gender
│
├── results/        # Bảng số liệu & biểu đồ
│   ├── tables/     # MAE, RMSE, AUC, NLE, PSD so sánh các hướng
│   └── figures/    # Biểu đồ, confusion matrix, difference maps
│
└── notebooks/      # Debug & EDA (Jupyter notebooks)
```

## Trình tự thực nghiệm

| Bước | Mô tả | Thư mục |
|------|--------|---------|
| 1 | Tái lập baseline (MAE 30.11) | `baseline/` |
| 2 | Hướng E: landmark QA trên baseline → phân tích nguyên nhân | `huong_E/` |
| 3 | Hướng B: saliency mask thay mask trắng → đo MAE/AUC + NLE | `huong_B/` |
| 4 | Hướng D: frequency blending trên kết quả B → đo lại | `huong_D/` |
| 5 | Bảng so sánh tổng: Baseline → +B → +B+D (3 trục đánh giá) | `results/` |

## Ba trục đánh giá

- 📊 **MAE/AUC** – Hiệu năng downstream bone age estimation
- 🦴 **Normalized Landmark Error (NLE)** – Độ toàn vẹn giải phẫu
- 〰️ **PSD dải tần cao (FFT)** – Độ toàn vẹn texture xương

## Tài liệu tham khảo

- `word/Trien_khai_chi_tiet_3_huong_cai_tien.docx` – Chi tiết kỹ thuật 3 hướng
- `word/Huong_cai_tien_toan.docx` – Tổng quan cải tiến
