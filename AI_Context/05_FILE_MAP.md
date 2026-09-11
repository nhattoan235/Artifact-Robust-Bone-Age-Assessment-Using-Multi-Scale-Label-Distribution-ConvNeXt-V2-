# Bản đồ tệp

> Cập nhật: 2026-09-11. Đường dẫn tương đối tính từ D:/Hoctap/Doan_totnghiep.

## Canonical context

| Nhu cầu | Tệp |
|---|---|
| Bắt đầu/thuật ngữ/định tuyến | AI_Context/00_START_HERE.md |
| Kết quả và quyết định hiện hành | AI_Context/01_STATUS_RESULTS.md |
| Lịch sử phương pháp | AI_Context/02_METHOD_HISTORY.md |
| Split, leakage và test policy | AI_Context/03_DATA_PROTOCOL.md |
| Kế hoạch C0 → C1 → C2 | AI_Context/04_CURRENT_PLAN.md |
| Bản đồ này | AI_Context/05_FILE_MAP.md |
| So sánh pipeline/bài báo | AI_Context/06_PIPELINE_COMPARISON.md |
| Chỉ mục máy đọc | AI_Context/context_index.json |

## Code và báo cáo đang dùng

| Nội dung | Vị trí |
|---|---|
| Repository chính | doan_totnghiep/ |
| Baseline train/data/model | doan_totnghiep/p1_baseline/ |
| C3-ROI code và outputs | doan_totnghiep/c3_roi/ |
| Phân tích ConvNeXt/C0-C2 | doan_totnghiep/research/CONVNEXT_BACKBONE_ANALYSIS_20260909.md |
| NO_HE OOF audit | doan_totnghiep/artifacts/C3_R2_Z26_NO_HE_E1_V1_RESULTS_AUDIT/NO_HE_OOF_AUDIT_REPORT.json |
| Báo cáo C3-ROI đầy đủ | AI_Context/24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md |
| Báo cáo model family/TTA | AI_Context/26_MODEL_FAMILY_REPORT.md |
| Báo cáo rebuild fallback | AI_Context/C3_FALLBACK_REBUILD_REPORT.md |
| P14 artifacts vĩnh viễn | NghienCuuChinh/p14_anatomy_diverse/ |

## Artifact ngoài repository nhưng đang được tham chiếu

| Artifact | Vị trí |
|---|---|
| C3-R2 Z26 NO_HE test report bundle | C3_R2_Z26_NO_HE_TEST_V1_RESULTS.zip |
| E1/C3 TTA test report bundle | E1_C3_TTA_TEST_V1_RESULTS.zip |
| C3-R2 full Colab/data bundle | C3_ROI_R2_COLAB_FULL_20260906.zip |
| Margin-8 train-ready recreation | C3_ROI_V1_MARGIN8_TRAIN_READY_20260909.zip hoặc bản corrected tương ứng |

Các ZIP chỉ là nguồn artifact; agent không được suy trạng thái từ tên tệp. Phải đọc JSON/report/checkpoint metadata bên trong.

## Tài liệu chuyên sâu/lịch sử

- 07_DEEPLASIA_FOCUSED_ANALYSIS.md — phân tích Deeplasia.
- 08_SEX_AWARE_EXPERIMENT_PLAN.md — sex-aware ablation.
- 09_P12_UNCERTAINTY_PROTOCOL.md — uncertainty protocol.
- 10_P13_THESIS_REPORTING_PLAN.md — thesis reporting.
- 11_* đến 27_* — các báo cáo quyết định/experiment cũ; chỉ mở theo chủ đề.
- _P14_TASKS_TEMP/ — dữ liệu tác vụ tạm, **không canonical**. Chỉ xóa khi điều kiện trong 99_CLEANUP_AFTER_P14.md đạt.

## Cách truy xuất tiết kiệm token

1. Đọc 00_START_HERE.md, context_index.json, 01_STATUS_RESULTS.md.
2. Dùng rg tìm đúng run ID/metric trước khi mở báo cáo dài.
3. Với số liệu dùng trong luận văn, ưu tiên JSON/CSV gốc hơn Markdown tóm tắt.
4. Không đọc _P14_TASKS_TEMP hoặc toàn bộ báo cáo lịch sử nếu tác vụ không liên quan.
