# C3-ROI + Attention — audit gói tải ngày 2026-08-29

## Nguyên nhân có nhiều file trùng

Google Drive đã tạo nhiều thư mục/file cùng tên trong quá trình mirror checkpoint.
Khi tải xuống, Drive thực hiện đồng thời hai việc:

- Chia một lượt tải lớn thành nhiều ZIP gần 1,8 GB với hậu tố `-001`, `-002`, ...
- Gộp các tên trùng và đổi file thành hậu tố `(1)`, `(2)`, ...

Vì vậy một ZIP không tương ứng với một model và các hậu tố giữa
`run_state`, `last.ckpt`, `best_mae.ckpt` không luôn ghép theo cùng số. Bản hợp
lệ được chọn bằng metadata bên trong checkpoint, trạng thái, log, config hash và
MAE tái tính từ prediction; không chọn bằng tên hoặc giờ sửa đổi đơn thuần.

## Kết quả xác minh cuối

| Fold | Kết thúc | Best epoch | Best MAE (tháng) | Prediction |
|---|---:|---:|---:|---:|
| 1 | early-stop epoch 25, step 7800 | 16 | 6,340856 | 2.808 ID |
| 2 | early-stop epoch 20, step 6240 | 11 | 6,382448 | 2.807 ID |
| 3 | early-stop epoch 25, step 7800 | 16 | 6,522994 | 2.807 ID |
| 4 | early-stop epoch 21, step 6552 | 12 (hiển thị epoch 13 trong log) | 6,520042 | 2.807 ID |
| 5 | early-stop epoch 22, step 6864 | 13 | 6,338086 | 2.807 ID |

Kiểm tra cuối: **14.036 dòng, 14.036 ID duy nhất**, không giao nhau giữa các
fold; target và sex khớp C3-ROI. Tất cả checkpoint đọc được và config hash khớp
config fold tương ứng.

Fold 1–2 có code hash cũ và Fold 3–5 có code hash mới do bản sửa vận hành mirror
Google Drive. Thay đổi này không sửa kiến trúc, forward, loss, optimizer, split
hoặc hyperparameter khoa học.

## Phục hồi Fold 4

Fold 4 đã train hoàn chỉnh nhưng `run_state.json` cuối và prediction tốt nhất bị
Drive mirror sai phiên bản. `train.log` cuối xác nhận early stopping sau epoch 21;
checkpoint cuối xác nhận epoch 21/step 6552 và best checkpoint xác nhận MAE
6,520041970965443.

Prediction Fold 4 được tái tạo từ đúng best checkpoint trên cùng validation split,
dùng FP16 như Colab T4. Kết quả có 2.807 ID và MAE khớp checkpoint tuyệt đối.
`run_state.json` được phục hồi từ log và metadata checkpoint, có trường
`recovered_from` để giữ provenance.

## Kết quả OOF 5-fold

- C3-ROI + Attention MAE: **6,420880 tháng**.
- C3-ROI MAE: **6,437349 tháng**.
- Chênh lệch Attention − C3-ROI: **−0,016469 tháng**.
- Paired-bootstrap CI 95%: **[−0,063653; +0,030326] tháng**.

Attention cải thiện point estimate khoảng 0,016 tháng, nhưng CI cắt 0. Vì vậy
không có bằng chứng thống kê rằng attention tốt hơn C3-ROI; kết luận học thuật
phù hợp là **không chứng minh được lợi ích bổ sung của spatial attention sau ROI**.

## Vị trí dữ liệu đã khóa

- Run chuẩn: `c3_roi_attention/runs/C3_ROI_ATTN_V1/`
- Báo cáo OOF: `c3_roi_attention/outputs/C3_ROI_ATTN_V1_OOF/C3_ROI_ATTN_V1_vs_C3_ROI_V1_OOF.json`
- Prediction ghép: `c3_roi_attention/outputs/C3_ROI_ATTN_V1_OOF/C3_ROI_ATTN_V1_vs_C3_ROI_V1_predictions.csv`

Các ZIP tải về và bản staging chưa bị xóa.
