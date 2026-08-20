# Trạng thái và kết quả định lượng

## P7 – 5-fold OOF (kết quả phát triển chính)

Artifact: `p7_final_v3/P7_OOF_report.json`, `p7_final_v3/P7_5FOLD_AUDIT.txt`.

| Chỉ số | Kết quả |
|---|---:|
| Số mẫu OOF / ID duy nhất | 14.036 / 14.036 |
| Pooled MAE | **6,316691 tháng** |
| RMSE | 8,519420 |
| Median absolute error | 4,875 |
| Accuracy ±6 / ±12 / ±18 tháng | 59,01% / 86,73% / 95,53% |
| Bootstrap 95% CI của MAE | [6,224638; 6,411327] |
| Test dùng để chọn mô hình | Không |

MAE từng fold: F1 6,298710; F2 6,196835; F3 6,411474; F4 6,348768; F5 6,327675. Mean 6,316692, SD 0,078775. Tất cả manifest/model tensor audit đều đạt.

## P8 – ensemble 5 fold trên RSNA test

Artifact: `p8_test_ensemble/outputs/P8_test_ensemble_report.json`, `P8_ensemble_predictions.csv`.

Protocol: trung bình đều 5 fold, checkpoint tốt nhất từng fold, **không tune trên test**.

| Chỉ số ensemble | Kết quả |
|---|---:|
| Số ảnh | 200 |
| MAE | **4,730321 tháng** |
| RMSE | 6,028745 |
| Median AE | 4,133363 |
| Accuracy ±6 / ±12 / ±18 | 70,0% / 94,0% / 100,0% |
| Bootstrap 95% CI MAE | [4,222475; 5,263709] |

MAE theo fold: 5,119659; 5,318364; 5,188306; 4,822037; 5,279186. Input audit PASS: 5 model, manifest SHA PASS, 14.036 OOF duy nhất, 200 ảnh test, sex/ID/ground-truth khớp.

## So sánh mốc tham khảo

- Bram et al., *The American Journal of Sports Medicine* (2025): **3,68 tháng** trên RSNA test.
- Rassmann et al., *Pediatric Radiology* (2024), Deeplasia: **3,87 tháng** trên RSNA test.
- P8 tốt hơn baseline nội bộ P7 OOF không phải phép so sánh trực tiếp (OOF 14.036 khác test 200), nhưng **chưa đạt** hai mốc công bố.

## Điểm tốt

- Split chính thức 12.611 train / 1.425 validation / 200 test được kiểm tra ID, ảnh, duplicate, hash và leakage.
- P7 OOF có 14.036 dự đoán, ID duy nhất, bootstrap CI và subgroup metrics.
- Checkpoint/resume, SHA manifest, early stopping và audit storage được thiết kế để tái lập qua nhiều tài khoản Colab.
- P8 là ensemble cố định, không tối ưu trọng số dựa trên nhãn test.

## Điểm chưa tốt/rủi ro

- Sai số P8 còn cao hơn Bram khoảng 1,05 tháng và Deeplasia khoảng 0,86 tháng; chưa thể ghi “vượt trội”.
- Test chỉ có 200 ảnh nên CI rộng; không được lặp lại nhiều lần để chọn mô hình.
- P7 dùng `preprocessing=none`; nhánh mask B1 không cải thiện validation. Có thể còn khoảng cách do thiếu preprocessing/augmentation/recipe tương đương Bram.
- P4 D1 gây feature collapse; D2 không cải thiện; D3/LDL chỉ cải thiện rất nhỏ và CI chứa 0.
- Bộ 200 ảnh đã làm sạch artifact và hướng inpainting/generative trước đây chưa phải phần của protocol P7/P8; không được trộn vào so sánh chính nếu chưa có thiết kế paired và audit độc lập.

