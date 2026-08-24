# Thí nghiệm C — C3-ROI local/global ensemble

> Ngày chốt: 2026-08-24  
> Trạng thái: đã train đủ 5 fold, đã tổng hợp OOF và chạy test thăm dò  
> Endpoint khoa học chính: paired 5-fold OOF trên 14.036 ảnh development

## 1. Tên gọi và phạm vi

`C3_ROI_V1` là mã run của **phương án C dùng ảnh ROI**, không phải kiến trúc
`D3` label-distribution learning. Model C3 thực tế là ConvNeXt-Tiny direct
regression + sex embedding, giữ recipe E1 và chỉ thay đầu vào thành ROI. So sánh
chính là:

- E1: model global nhìn ảnh toàn bàn tay;
- C3-ROI: model độc lập nhìn ảnh ROI;
- ensemble khóa trước: `0,5 × E1 + 0,5 × C3-ROI`.

Không có thí nghiệm D3-LDL train trên ROI trong kết quả này.

## 2. Câu hỏi nghiên cứu

Ảnh toàn bàn tay resize về 512 px có thể làm mất chi tiết cục bộ. Giả thuyết của
phương án C là một model học trên crop ROI sẽ tạo sai số bổ sung cho E1; vì vậy
ensemble local/global có thể giảm MAE ngay cả khi C3-ROI không thắng E1 khi đứng
riêng.

## 3. Dữ liệu và protocol đã khóa

- Development: 14.036 ảnh, 14.036 ID duy nhất.
- Dùng đúng 5 fold P7 đã khóa, seed 42; mỗi ảnh xuất hiện đúng một lần ở OOF.
- Backbone: ConvNeXt-Tiny pretrained ImageNet, input 512.
- Sex embedding: 16 chiều; head 256, dropout 0,2.
- Loss: SmoothL1, beta 3 tháng; AdamW-style recipe với LR `2e-4`, weight decay
  `0,05`, tối đa 35 epoch và early stopping.
- Augmentation nhẹ giống E1; model C3 khởi tạo độc lập, không warm-start E1.
- Ensemble chính cố định 50/50; không tối ưu trọng số theo OOF hoặc test.
- Test 200 ảnh đã từng được mở ở P8, nên chỉ là phân tích thăm dò, không phải
  xác nhận độc lập.

## 4. ROI thực tế và sai lệch so với thiết kế ban đầu

Pipeline thực tế dùng bounding box từ hand segmentation với margin 8%. Khi
segmentation thất bại, ảnh toàn bàn tay được dùng làm fallback; không loại ảnh.
Đây là broad hand-bbox crop, không phải local carpal ROI theo PCA như đặc tả ban
đầu.

| Tập | Tổng ảnh | Bbox ROI | Full-image fallback | Tỷ lệ fallback |
|---|---:|---:|---:|---:|
| Development | 14.036 | 11.441 | 2.595 | **18,49%** |
| Test | 200 | 134 | 66 | **33,00%** |

Toàn bộ 2.595 fallback development có lý do `segment_hand_failed`; gồm 2.300 ảnh
train gốc và 295 ảnh official validation.

Thiết kế C ban đầu yêu cầu fallback không quá 1%. Run này vượt gate đó rất xa.
Vì vậy kết quả chỉ chứng minh hiệu quả của **ROI-preprocessing có fallback**, chưa
đủ để gọi là một anatomy-local specialist thuần túy hoặc khẳng định localization
đã ổn định trên mọi ảnh.

## 5. Kết quả OOF chính

| Mô hình | MAE | RMSE | Median AE |
|---|---:|---:|---:|
| E1/P7 raw | 6,316691 | 8,519420 | 4,8750 |
| C3-ROI raw | 6,437349 | 8,649622 | 4,9375 |
| E1 + C3-ROI 50/50 | **6,176212** | **8,346747** | **4,6875** |

- C3-ROI riêng kém E1 `+0,120658` tháng; paired bootstrap 95% CI
  `[+0,061527; +0,176776]`.
- Ensemble giảm `0,140480` tháng so với E1; paired bootstrap 95% CI của delta
  `[-0,172547; -0,109853]`, hoàn toàn dưới 0.
- Bootstrap 95% CI MAE của ensemble: `[6,084042; 6,266692]`.
- Tương quan Pearson prediction E1/C3 rất cao: `0,995123`, nhưng phần sai khác
  còn đủ để ensemble có lợi.
- ID, target và sex của 14.036 dòng khớp hoàn toàn với E1; không có ID thiếu,
  trùng hoặc non-finite.

Ensemble cải thiện ở cả 5 fold:

| Fold | E1 | C3-ROI | Ensemble 50/50 |
|---:|---:|---:|---:|
| 1 | 6,298710 | 6,375519 | **6,113998** |
| 2 | 6,196835 | 6,372684 | **6,078042** |
| 3 | 6,411474 | 6,591259 | **6,321262** |
| 4 | 6,348768 | 6,426442 | **6,253304** |
| 5 | 6,327675 | 6,420862 | **6,114476** |

OOF ensemble theo nhóm:

- Nữ: n=6.430, MAE 6,382103; nam: n=7.606, MAE 6,002154.
- 0–59: 5,966963; 60–119: **7,061743**; 120–179: 5,872619;
  180–228: 5,515485 tháng.
- Nhóm 60–119 tháng vẫn là nhóm khó nhất.

## 6. Kết quả test thăm dò

| Mô hình | MAE | RMSE | Median AE |
|---|---:|---:|---:|
| E1 | 4,730321 | 6,028745 | 4,133363 |
| C3-ROI | **4,337267** | **5,531101** | **3,394524** |
| E1 + C3-ROI 50/50 | 4,454661 | 5,650689 | 3,734033 |

C3-ROI riêng tốt hơn E1 `0,393055` tháng trên test; ensemble tốt hơn E1
`0,275660` tháng nhưng kém C3-ROI riêng. Không được dùng quan sát này để đổi
trọng số hoặc chọn lại pipeline vì ground truth test đã được truy cập trước đó.

| Nhóm test | E1 | C3-ROI | Ensemble |
|---|---:|---:|---:|
| Nữ (n=100) | 4,8562 | 4,7412 | **4,7130** |
| Nam (n=100) | 4,6044 | **3,9333** | 4,1963 |
| 0–59 (n=14) | 5,1927 | **4,3572** | 4,7750 |
| 60–119 (n=53) | 6,6198 | **5,6182** | 6,0614 |
| 120–179 (n=108) | 3,9324 | 4,0140 | **3,8936** |
| 180–228 (n=25) | 3,9125 | **3,0072** | 3,2928 |

## 7. Kết luận và claim được phép

Kết luận OOF hợp lệ:

> Dưới protocol 5-fold OOF đã khóa, ensemble equal-weight giữa E1 global và
> C3-ROI giảm MAE 0,14048 tháng; paired bootstrap 95% CI hoàn toàn dưới 0 và
> cải thiện ở cả 5 fold.

Không được tuyên bố:

- C3-ROI standalone thắng E1 trên OOF;
- ROI đã đạt gate hình học ban đầu hoặc là local carpal specialist thuần túy;
- test 200 ảnh xác nhận ưu thế tổng quát;
- phương pháp đã đạt/vượt Bram 3,68 tháng.

Quyết định: giữ C3-ROI + E1 50/50 như một **ablation OOF dương tính** và nguồn
diversity có giá trị học thuật. Chưa thay pipeline xác nhận cuối cùng cho đến khi
có holdout ngoài chưa chạm hoặc sửa localization rồi lặp lại theo protocol mới
được khóa trước.

## 8. Artifact tái lập

- Config 5 fold: `c3_roi/configs/fold_1.toml` … `fold_5.toml`.
- Checkpoint: `c3_roi/runs/C3_ROI_V1/C3_ROI_V1_FOLD_X/`.
- Tổng hợp OOF: `c3_roi/aggregate_oof.py`.
- OOF report: `c3_roi/outputs/C3_ROI_V1_OOF/C3_ROI_V1_OOF_report.json`.
- OOF predictions: `c3_roi/outputs/C3_ROI_V1_OOF/C3_ROI_V1_E1_ensemble_OOF_predictions.csv`.
- Chuẩn bị/inference test: `c3_roi/prepare_test_roi.py`,
  `c3_roi/infer_test_ensemble.py`.
- Test report: `c3_roi/outputs/C3_ROI_V1_TEST/C3_E1_50_50_test_report.json`.
- Test predictions: `c3_roi/outputs/C3_ROI_V1_TEST/C3_E1_50_50_test_predictions.csv`.
- Audit development: `c3_roi/_drive_upload/C3_ROI_V1/cpu_artifacts/audit_summary.json`.
- Phân tích ngắn cũ: `c3_roi/outputs/C3_ROI_V1_FINAL_ANALYSIS.md`.

