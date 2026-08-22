# P13 — Thesis reporting handoff

**Trạng thái:** PASS — bộ bảng/hình đã được build từ nguồn đã khóa; visual QA đạt; RSNA test không được truy cập.

## Mục tiêu và phạm vi

P13 chuyển kết quả P11–P12 thành bộ tài sản tái lập để viết luận văn. Giai đoạn này không huấn luyện mô hình, không chọn lại checkpoint, không thay đổi giả thuyết và không mở thêm nhánh thí nghiệm.

## Nguồn dữ liệu đã khóa

- E0: `p11_sex_aware/runs/P11_E0_IMAGE_ONLY_CONVNEXT_TINY_SEED42/val_predictions_best.csv`
- E1: `p9_preprocessing/runs/P10_B0_P2_CONTROL_CONVNEXT_TINY_SEED42/val_predictions_best.csv`
- E2: `p11_sex_aware/runs/P11_E2_SHARED_DUAL_OUTPUT_CONVNEXT_TINY_SEED42/val_predictions_best.csv`
- Quyết định P11 và các bảng OOF P12 được đọc từ các report/output đã hoàn tất.

Ba tệp validation có 1.425 ID khớp hoàn toàn. Generator kiểm tra schema, ID, nhãn tuổi, giới tính và tính hữu hạn trước khi tính toán.

## Cấu hình tái lập

- Paired bootstrap: 10.000 lần
- Seed gốc: 2026
- Đơn vị lấy mẫu: bệnh nhi/hàng dự đoán đã căn chỉnh
- Chính sách test: `test_accessed=false`
- Manifest: `outputs/P13_THESIS_REPORT/report_manifest.json`

## Kiểm chứng

- Compile: PASS
- Unit tests: 5/5 PASS
- Full deterministic build: PASS; hai lần build liên tiếp có 0/15 artifact đổi hash
- Visual QA ba hình PNG: PASS; không phát hiện clipping, chữ chồng hoặc nhãn bị cắt
- Số output được lập SHA-256 trong manifest: 15

## Tài sản bàn giao

### Bảng

1. `table_1_model_comparison.csv/.md` — hiệu năng E0/E1/E2.
2. `table_2_paired_forest_data.csv/.md` — sáu so sánh tổng thể/nữ/nam.
3. `table_3_p12_sex_age.csv/.md` — bất định theo giới tính–nhóm tuổi.
4. `table_4_hypothesis_summary.csv/.md` — quyết định H1–H4.

### Hình

1. `figure_1_paired_forest.png/.pdf` — chênh lệch MAE theo cặp.
2. `figure_2_sex_age_heatmaps.png/.pdf` — dị biệt theo giới tính–nhóm tuổi.
3. `figure_3_disagreement_risk.png/.pdf` — tứ phân vị, lỗi lớn và risk–coverage.
4. `figure_captions.json` — caption đồng bộ.

### Tài liệu

- `P13_THESIS_DRAFT_VI.md` — bản thảo Methods–Results–Discussion–Limitations, hướng dẫn chèn bảng/hình và quy tắc diễn đạt.
- `build_thesis_assets.py` — generator duy nhất cho bộ tài sản.
- `test_reporting.py` — kiểm thử metric, drift target, bootstrap và số hàng forest plot.

## Kết luận đã khóa

- **H1 — SUPPORTED:** E1 tốt hơn E0 1,287 tháng; 95% CI 1,012–1,570.
- **H2 — NOT SUPPORTED:** E2−E1 = −0,028 tháng; 95% CI −0,179 đến 0,124.
- **H3 — NOT SUPPORTED:** khoảng cách giới không giảm ở E2.
- **H4 — SUPPORTED, LIMITED UTILITY:** Spearman ρ=0,200; AUROC >12 tháng 0,626 và >18 tháng 0,634. Tín hiệu phù hợp để phân tầng/thăm dò, chưa đủ cho tuyên bố lâm sàng.

## Quyết định nghiên cứu tiếp theo

Không mở E3 hoặc nhiều seed cho E2 từ kết quả hiện tại. Hướng có giá trị cao nhất là hoàn thiện luận văn với E1 làm baseline chính; nếu còn tài nguyên, ưu tiên một external holdout chưa bị tác động và protocol hiệu chỉnh/đánh giá referral được khóa trước. Không tái sử dụng RSNA test đã từng bị truy cập để tuyên bố xác nhận độc lập.
