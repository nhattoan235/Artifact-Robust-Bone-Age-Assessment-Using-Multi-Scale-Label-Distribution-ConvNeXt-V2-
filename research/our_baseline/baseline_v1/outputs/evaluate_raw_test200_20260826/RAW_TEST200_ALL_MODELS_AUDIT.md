# Đánh giá toàn bộ model trên tập test 200 ảnh đã làm sạch

## Protocol

- Ảnh sạch: `D:\do_an_tot_nghiep\project\data\artifact_only_200_manual_v3\artifact_only_200_manual_v3\cleaned`
- Nhãn: `D:\do_an_tot_nghiep\project\data\rsna\boneage-test-dataset-with-gt.csv`
- Số lượng: 200 ảnh, ID 4360–4559.
- Đã đánh giá 15 checkpoint hiện có.
- Model thuần (`raw`): preprocessing xác định của từng model, không xoay/lật.
- TTA: trung bình 10 views gồm xoay -10°, -5°, 0°, 5°, 10° với ảnh nguyên bản và lật ngang.
- Precision: AMP FP16 trên GPU RTX 3050; checkpoint không bị sửa.
- Đây là đánh giá thăm dò; nhãn test không được dùng để chọn hyperparameter hoặc trọng số.

## Kết quả theo nhóm

| Nhóm | Số model | MAE raw | MAE + TTA | Δ MAE | RMSE raw | RMSE + TTA |
|---|---:|---:|---:|---:|---:|---:|
| original_official | 2 | 4.8178 | 4.6598 | -0.1580 | 6.0582 | 5.9103 |
| p1_standard | 6 | 4.9731 | 4.8170 | -0.1561 | 6.3234 | 6.1161 |
| exp006_five_fold | 5 | 4.7624 | 4.6709 | -0.0915 | 6.0478 | 6.0279 |
| exp009_ldl_two_fold | 2 | 4.9916 | 5.0724 | 0.0808 | 6.4565 | 6.4982 |
| all_15_models | 15 | 4.7664 | 4.6753 | -0.0911 | 5.9747 | 5.9289 |

## Kết quả từng model

| Model | MAE raw | MAE + TTA | Δ MAE | RMSE raw | RMSE + TTA |
|---|---:|---:|---:|---:|---:|
| EXP006_P7_CONTROL_FOLD3_CONVNEXT_TINY_SEX_LDL | 4.5030 | 4.5271 | 0.0241 | 5.7239 | 5.8523 |
| OFFICIAL_P7_REFERENCE_CONVNEXT_TINY_SEX | 4.3004 | 4.5492 | 0.2488 | 5.3536 | 5.8017 |
| EXP004_FRIEND_P7_FRESH_HOLDOUT_CONVNEXT_TINY_SEX | 5.1338 | 4.7506 | -0.3832 | 6.7481 | 6.1779 |
| P7_E1_FOLD4_CONVNEXT_TINY_SEX_A2 | 4.8911 | 4.8518 | -0.0393 | 6.3692 | 6.2191 |
| EXP006_P7_CONTROL_FOLD1_CONVNEXT_TINY_SEX_LDL | 4.8887 | 4.9403 | 0.0516 | 6.3982 | 6.4779 |
| EXP009_LDL_FOLD_1 | 5.1078 | 5.0088 | -0.0990 | 6.6322 | 6.4174 |
| EXP006_P7_CONTROL_FOLD5_CONVNEXT_TINY_SEX_LDL | 5.2898 | 5.0447 | -0.2451 | 6.6499 | 6.5072 |
| P7_E1_FOLD3_CONVNEXT_TINY_SEX_A2 | 5.4998 | 5.1852 | -0.3146 | 7.1075 | 6.6117 |
| P7_E1_FOLD1_CONVNEXT_TINY_SEX_A2 | 5.4491 | 5.1863 | -0.2628 | 7.1425 | 6.9282 |
| OFFICIAL_A2_LIGHT_FLIP_CONVNEXT_TINY_SEX | 6.0472 | 5.2307 | -0.8165 | 7.7112 | 6.7047 |
| P7_E1_FOLD2_CONVNEXT_TINY_SEX_A2 | 5.4852 | 5.2517 | -0.2335 | 6.9498 | 6.5470 |
| P7_E1_FOLD5_CONVNEXT_TINY_SEX_A2 | 5.4588 | 5.4210 | -0.0378 | 6.9343 | 6.9494 |
| EXP006_P7_CONTROL_FOLD2_CONVNEXT_TINY_SEX_LDL | 5.7262 | 5.4231 | -0.3031 | 7.6896 | 7.2912 |
| EXP006_P7_CONTROL_FOLD4_CONVNEXT_TINY_SEX_LDL | 5.3938 | 5.5922 | 0.1984 | 7.0113 | 7.2908 |
| EXP009_LDL_FOLD_2 | 5.4722 | 5.6200 | 0.1478 | 7.0737 | 7.1516 |

## Nhận xét chính

- Ensemble 15 model thuần đạt MAE **4.7664**.
- Ensemble 15 model + TTA đạt MAE **4.6753**.
- TTA thay đổi MAE **-0.0911** tháng; giá trị âm là cải thiện.
- TTA cải thiện ensemble EXP006 5-fold, nhưng mức cải thiện nhỏ hơn nhóm P7 standard.
- TTA làm xấu nhóm EXP009 LDL 2-fold trong phép thử này; không nên mặc định áp dụng TTA cho mọi nhóm.
- Mốc EXP006 TTA 5-fold MAE 4.4669 trước đây thuộc một lần đánh giá khác; không trộn với kết quả ảnh sạch hiện tại.
- Không dùng kết quả test 200 này để chọn trọng số hoặc hyperparameter; quyết định chính thức cần dựa trên OOF/holdout độc lập.

## Kiểm tra dữ liệu sạch

- QC rows: 200
- Pixel preservation pass: 200/200
- Tỷ lệ pixel thay đổi trung bình: 2.6756%
- Trạng thái: `{'MANUAL_MASK_CLEANED': 185, 'AUTO_CLEANED': 15}`

## File kết quả

- [Prediction CSV](cleaned_test200_all_models_raw_vs_tta_predictions.csv)
- [Report JSON](cleaned_test200_all_models_raw_vs_tta_report.json)
- [Script đánh giá](../../scripts/evaluate_all_models_tta_cleaned_test200.py)
- [Script tạo báo cáo](../../scripts/write_all_models_tta_audit_md.py)
