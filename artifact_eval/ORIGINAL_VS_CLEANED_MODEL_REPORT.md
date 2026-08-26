# Đối chiếu model trên ảnh test gốc và ảnh giảm artifact

> Ngày chạy: 2026-08-26
> Cỡ mẫu: 200 ảnh, cùng `image_id` và cùng ground truth
> Protocol: cùng checkpoint, cùng preprocessing, không TTA; chỉ thay đổi ảnh đầu vào gốc/cleaned

## Kết quả

| Model / ensemble | Ảnh gốc MAE | Cleaned MAE | Cleaned − gốc | Tác động cleaned |
|---|---:|---:|---:|---|
| E1_P7 5-fold | 4,730144 | 4,971330 | +0,241186 | Xấu hơn |
| E0 image-only | 7,310150 | 7,324193 | +0,014043 | Gần như không đổi |
| E1 P10 validation | 4,643414 | 4,786775 | +0,143360 | Xấu hơn |
| E2 dual-output | 4,894432 | 5,094975 | +0,200542 | Xấu hơn |
| D3 5-fold | 4,875168 | 4,988446 | +0,113278 | Xấu hơn |
| C3-ROI 5-fold | 4,336400 | **4,218859** | **−0,117541** | Tốt hơn |
| E1 + D3 50/50 | 4,724717 | 4,923852 | +0,199135 | Xấu hơn |
| E1 + C3 50/50 | 4,454525 | 4,485360 | +0,030835 | Gần như không đổi |
| Trung bình 6 model family | 4,734560 | 4,862957 | +0,128397 | Xấu hơn |

## Kết luận

1. C3-ROI vẫn là model tốt nhất trên cả hai phiên bản ảnh; trên cleaned đạt MAE
   **4,218859 tháng**, cải thiện **0,117541 tháng** so với ảnh gốc.
2. Làm sạch artifact không tự động giúp các model nhận toàn ảnh. E1, E2 và D3
   đều giảm nhẹ, trong khoảng **0,113–0,241 tháng** MAE.
3. E0 gần như không bị ảnh hưởng, chênh lệch chỉ **0,014 tháng**, cho thấy
   pipeline image-only này không khai thác được lợi ích đáng kể từ bước cleaned.
4. Ensemble E1+C3 gần như giữ nguyên (**+0,030835 tháng**), vì C3 được cải thiện
   nhưng E1 lại kém đi.
5. Kết quả ủng hộ giả thuyết: artifact ảnh hưởng chủ yếu qua vùng ngoài ROI hoặc
   bố cục ảnh; ROI giúp model tập trung vào vùng có thông tin xương. Chưa đủ
   bằng chứng để kết luận rằng một bộ cleaned toàn ảnh là tốt hơn cho mọi recipe.

## Giới hạn

- 200 ảnh là bộ đánh giá nội bộ; ground truth ghép từ `data/goc/rsna_test_ground_truth.csv`.
- Ảnh cleaned được tạo từ cùng 200 ảnh nên đây là paired preprocessing comparison,
  không phải external clinical validation.
- C3 dùng tọa độ ROI đã khóa theo `image_id`; không dùng lại ảnh ROI cũ đã sinh từ
  ảnh gốc.
- Không chọn lại checkpoint hoặc tune trọng số ensemble trên 200 ảnh này.

## Artifact

- Predictions: `artifact_eval/outputs/artifact_only_200_manual_v3/original_vs_cleaned_all_model_predictions.csv`
- JSON: `artifact_eval/outputs/artifact_only_200_manual_v3/ORIGINAL_VS_CLEANED_MODEL_REPORT.json`
- Script: `artifact_eval/evaluate_original_and_compare.py`
