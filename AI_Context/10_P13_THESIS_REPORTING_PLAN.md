# P13 – Kế hoạch đóng gói kết quả cho luận văn

> **Ngày khóa:** 2026-08-22  
> **Trạng thái:** hoàn tất — build, kiểm thử và visual QA PASS  
> **Mục tiêu:** chuyển bằng chứng P11–P12 thành bảng, hình và văn bản luận văn  
> **Test policy:** không đọc hoặc dùng RSNA test

## 1. Nguồn dữ liệu

- E0 prediction: `p11_sex_aware/runs/P11_E0_IMAGE_ONLY_CONVNEXT_TINY_SEED42/val_predictions_best.csv`.
- E1 prediction: `p9_preprocessing/runs/P10_B0_P2_CONTROL_CONVNEXT_TINY_SEED42/val_predictions_best.csv`.
- E2 prediction: `p11_sex_aware/runs/P11_E2_SHARED_DUAL_OUTPUT_CONVNEXT_TINY_SEED42/val_predictions_best.csv`.
- Paired reports: `p11_sex_aware/analysis/e0_vs_e1_seed42.json` và
  `e2_vs_e1_seed42.json`.
- P12 report/tables: `p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/`.

Mỗi input phải được ghi SHA256 vào manifest. P13 không chạy inference hoặc
huấn luyện lại và không thay đổi quyết định gate.

## 2. Bảng khóa trước

1. So sánh E0/E1/E2: MAE, RMSE, median AE, accuracy ±6/±12, female/male MAE.
2. Forest data: paired delta và bootstrap CI overall/F/M cho E0−E1, E2−E1.
3. Sex × age P12: count, TTA MAE, rho, CI, AUROC và TTA gain.
4. Tóm tắt H1–H4 với trạng thái supported/not supported và giới hạn.

## 3. Hình khóa trước

1. `figure_1_paired_forest`: paired delta MAE cùng 95% CI; đường 0 là không khác biệt.
2. `figure_2_sex_age_heatmaps`: hai panel TTA MAE và disagreement–error rho.
3. `figure_3_disagreement_risk`: disagreement quartile MAE/error rate và
   risk–coverage mô tả.

Mỗi hình xuất PNG 240 dpi và PDF vector, có nhãn/đơn vị, caption và palette
colorblind-friendly. Không dùng trục cắt gây phóng đại khác biệt.

## 4. Văn bản bàn giao

- Methods: dữ liệu, mô hình, endpoint, bootstrap, test policy.
- Results: H1–H4, effect size và CI, không chỉ p-value.
- Discussion: ý nghĩa của sex embedding, negative result E2 và utility hạn chế
  của disagreement.
- Limitations: một seed screening P11, post-hoc P12, test đã chạm, chưa external
  validation và chưa hiệu chuẩn uncertainty.

## 5. Artifact dự kiến

- `p13_reporting/build_thesis_assets.py`
- `p13_reporting/test_reporting.py`
- `p13_reporting/outputs/P13_THESIS_REPORT/`
- `p13_reporting/P13_THESIS_DRAFT_VI.md`
- `p13_reporting/P13_HANDOFF.md`
## 6. Kết quả thực hiện

- Ba nguồn E0/E1/E2 khớp đầy đủ 1.425 ID validation; kiểm tra target và giới
  tính không phát hiện drift.
- Paired bootstrap chạy 10.000 lần với seed 2026.
- Sinh đủ 4 bảng ở định dạng CSV/Markdown và 3 hình ở định dạng PNG/PDF.
- Compile PASS, unit test 5/5 PASS; hai build liên tiếp có 0/15 artifact đổi hash.
- Visual QA cả ba hình PASS: không clipping, chồng chữ hoặc mất nhãn.
- Manifest ghi SHA-256 cho 15 output và xác nhận `test_accessed=false`.
- Bản thảo luận văn và handoff:
  `p13_reporting/P13_THESIS_DRAFT_VI.md`,
  `p13_reporting/P13_HANDOFF.md`.

## 7. Quyết định cuối P13

- H1 được ủng hộ: sex embedding cải thiện rõ so với image-only.
- H2 và H3 không được ủng hộ: dual-output không hơn E1 và không thu hẹp sex gap.
- H4 chỉ được ủng hộ ở mức association/phân tầng; utility phát hiện lỗi lớn còn
  hạn chế và chưa phải uncertainty lâm sàng đã hiệu chỉnh.
- Giữ E1 làm baseline chính để viết luận văn; ưu tiên external holdout chưa bị
  tác động nếu tiếp tục nghiên cứu. Không mở lại RSNA test cho P13.
