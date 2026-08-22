# P12 – Protocol phân tích TTA disagreement và sai số OOF

> **Khóa trước phân tích:** 2026-08-22  
> **Trạng thái:** phân tích hoàn tất; H4 association được ủng hộ, utility hạn chế  
> **Loại nghiên cứu:** phân tích hậu nghiệm trên prediction OOF có sẵn  
> **Test policy:** không đọc hoặc dùng RSNA test

## 1. Mục tiêu

P12 kiểm định H4: bất đồng giữa các phép TTA có liên hệ dương với absolute
error hay không, và liên hệ thay đổi thế nào theo sex × age.

P12 không huấn luyện mới và không tối ưu threshold triển khai. Kết quả chỉ đánh
giá khả năng **xếp hạng rủi ro**; không gọi disagreement là uncertainty lâm
sàng đã hiệu chuẩn.

## 2. Dữ liệu khóa

- Nguồn: `p9_inference/outputs/P9_I_TTA_BIAS_OOF/P9_I_TTA_BIAS_OOF_predictions.csv`.
- 14.036 prediction OOF duy nhất, fold 1–5.
- 10 TTA views: rotation `[-10,-5,0,5,10]` × flip `{false,true}`.
- Disagreement: standard deviation 10 prediction, đơn vị tháng.
- Development manifest SHA256:
  `ee8f6ccd4b94b33ee21a11a0c5b6223f4f7adcb28c5cd3fcd3228f0c647776d5`.
- Không dùng bias-corrected prediction trong endpoint chính.

## 3. Endpoint khóa trước

### 3.1. Primary

- Spearman giữa `tta_std_months` và
  `abs(tta_prediction_months - target_months)` trên toàn OOF.
- Paired rank-score bootstrap 95% CI, 5.000 lần, seed 2026.
- H4 association được ủng hộ nếu lower CI > 0.

### 3.2. Secondary

- Spearman và CI theo F/M và 8 nhóm sex × age.
- Age bins: 0–59, 60–119, 120–179, 180–228 tháng.
- Raw/TTA MAE, paired TTA gain và CI theo sex × age.
- Worst-group và age-standardized MAE theo sex.
- Error rate >12/>18 tháng; AUROC disagreement cho hai lỗi lớn.
- Disagreement quartile: count, MAE, error >12/>18.
- Risk–coverage 100%, 90%, 80%, 70%, 50%, giữ disagreement thấp nhất.
- Prediction range/std và finite/collapse checks.

Subgroup là exploratory; p-value được Benjamini–Hochberg. Không chọn lại nhóm
hoặc age bin sau khi xem kết quả.

## 4. Diễn giải khóa trước

- |rho| <0,10: rất yếu; 0,10–0,29: yếu; 0,30–0,49: vừa; ≥0,50: mạnh.
- AUROC <0,60: kém; 0,60–0,69: hạn chế; ≥0,70 mới là tín hiệu tiềm năng,
  vẫn cần external validation và calibration.

Ý nghĩa thống kê không đồng nghĩa khả năng dùng lâm sàng. Không thiết lập
threshold abstention/deployment trên cùng OOF.

## 5. Điều kiện toàn vẹn

- Đúng 14.036 dòng, image ID duy nhất; fold 1–5; sex F/M; target 0–228.
- Prediction/disagreement hữu hạn; disagreement không âm.
- TTA mean khớp trung bình 10 view trong sai số số học cho phép.
- Hash input được ghi vào report.

## 6. Artifact dự kiến

- `p12_uncertainty/analyze_oof_uncertainty.py`
- `p12_uncertainty/test_uncertainty.py`
- `p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/report.json`
- `p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/group_metrics.csv`
- `p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/risk_coverage.csv`
- `p12_uncertainty/P12_HANDOFF.md`
