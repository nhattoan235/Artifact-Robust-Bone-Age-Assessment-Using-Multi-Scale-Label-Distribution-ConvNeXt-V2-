# Báo cáo chạy toàn bộ model trên `artifact_only_200_manual_v3`

> Ngày chạy: 2026-08-26
> Dataset: `artifact_only_200_manual_v3/cleaned`
> Số ảnh: 200
> Mục đích: đánh giá khả năng tổng quát trên bộ artifact-reduced do nhóm tự xây dựng

## 1. Protocol

- Dùng ảnh trong thư mục `cleaned`, không dùng ảnh gốc RSNA trong lần inference này.
- E1, E0, E2 và D3 nhận ảnh toàn bàn tay đã cleaned.
- C3-ROI dùng cùng ảnh cleaned, nhưng áp dụng tọa độ ROI đã khóa theo `image_id`
  từ C3 test manifest; không dùng ảnh ROI cũ đã sinh từ ảnh gốc.
- Không train lại, không điều chỉnh checkpoint và không tune ensemble weight theo
  bộ 200 ảnh này.
- Nhãn chỉ dùng để tính metric. Nguồn nhãn là
  `data/goc/rsna_test_ground_truth.csv`, ghép theo `image_id`.
- Đây là bộ đánh giá nội bộ mới về preprocessing artifact; không nên gọi là
  external holdout độc lập vì ảnh và nhãn bắt nguồn từ RSNA test pool.

## 2. Model inventory

| Model ID | Nguồn | Số checkpoint |
|---|---|---:|
| E1_P7_5FOLD | P7 E1 ConvNeXt-Tiny + sex embedding + A2 | 5 |
| E0_IMAGE_ONLY | P11 E0 ConvNeXt-Tiny image-only | 1 |
| E1_P10_VALIDATION | P10 E1 validation control | 1 |
| E2_DUAL_OUTPUT | P11 E2 dual-output theo giới tính | 1 |
| D3_5FOLD | D3 ConvNeXt-Tiny + label-distribution head | 5 |
| C3_ROI_5FOLD | C3 ConvNeXt-Tiny trên ROI + sex embedding | 5 |

## 3. Kết quả trên bộ cleaned

| Model / ensemble | MAE (tháng) | RMSE | Median AE | Bias |
|---|---:|---:|---:|---:|
| E1_P7 5-fold | 4,971330 | 6,313834 | 4,236759 | +1,047398 |
| E0 image-only | 7,324193 | 9,750394 | 5,709851 | +0,751979 |
| E1 P10 validation | 4,786775 | 6,010950 | 4,114934 | +1,441980 |
| E2 dual-output | 5,094975 | 6,513651 | 4,060516 | +1,794975 |
| D3 5-fold | 4,988446 | 6,297465 | 3,952867 | +1,420832 |
| C3-ROI 5-fold | **4,218859** | 5,425294 | 3,283991 | +0,347031 |
| E1 + D3 50/50 | 4,923852 | 6,211526 | 4,169602 | +1,234115 |
| E1 + C3-ROI 50/50 | 4,485360 | 5,705609 | 3,765497 | +0,697214 |
| Trung bình 6 model family | 4,862957 | 6,071927 | 4,052443 | +1,134032 |

Metric đầy đủ, bootstrap CI và subgroup được lưu trong
`artifact_only_200_all_model_report.json`; bảng trên nhấn mạnh MAE để dễ đọc.

## 4. Diễn giải

1. C3-ROI là model standalone tốt nhất trên bộ cleaned, MAE 4,218859 tháng.
2. E1 P10 validation control đạt 4,786775 tháng, nhỉnh hơn E1 P7 ensemble
   4,971330 trên bộ này; đây là khác biệt giữa checkpoint/control chứ không phải
   bằng chứng P10 luôn tốt hơn OOF.
3. E1 + C3 50/50 đạt 4,485360 tháng, kém C3 standalone trên bộ này. Vì trọng số
   50/50 đã được khóa từ OOF trước đó, không được đổi trọng số chỉ để tối ưu bộ
   artifact này.
4. D3 không cho lợi ích khi ensemble 50/50 với E1 trên bộ này: 4,923852 so với
   E1 4,971330, mức cải thiện nhỏ hơn C3.
5. E0 image-only yếu nhất; điều này ủng hộ việc đưa thông tin giới tính vào model
   trong các recipe còn lại, nhưng không phải phép thử nhân quả hoàn hảo vì các
   model cũng khác recipe/checkpoint.

## 5. Giới hạn cần ghi trong báo cáo

- Bộ mới chỉ có 200 ảnh và nhãn được ghép từ RSNA test ground truth; không phải
  external clinical dataset.
- Kết quả là đánh giá trên ảnh đã được xử lý artifact theo bộ quy tắc của nhóm.
  Không được so sánh trực tiếp với MAE bài báo nếu protocol ảnh, checkpoint và
  cách chọn test khác nhau.
- Vì bộ này được dùng để tìm model phù hợp với mục tiêu artifact-reduced, mọi
  quyết định cuối cùng cần xác nhận trên một holdout mới chưa dùng để chọn model.
- C3-ROI dùng tọa độ ROI khóa từ pipeline C3; cần ghi rõ tỷ lệ fallback của pipeline
  C3 trong báo cáo C gốc, không coi kết quả này là segmentation hoàn hảo.

## 6. Artifact

- Predictions: `artifact_eval/outputs/artifact_only_200_manual_v3/artifact_only_200_all_model_predictions.csv`
- JSON report: `artifact_eval/outputs/artifact_only_200_manual_v3/artifact_only_200_all_model_report.json`
- Script: `artifact_eval/evaluate_models.py`
- Test: `artifact_eval/test_evaluate_models.py`
- Model inventory đã giải nén tạm tại `data/model_eval_inventory/`.
