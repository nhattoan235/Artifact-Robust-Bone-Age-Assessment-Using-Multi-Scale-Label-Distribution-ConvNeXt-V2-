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

## P10 – kiểm soát recipe và augmentation

- **P10-B0 control:** tái lập chính xác P2 trên validation, MAE **6,184792**,
  RMSE 8,4865; toàn bộ 1.425 dự đoán giống hệt P2 cũ.
- **P10-B1 Deeplasia augmentation vừa:** MAE **6,185877**, RMSE 8,4255;
  delta so với control +0,001086 tháng, CI chứa 0. Không có bằng chứng
  augmentation mức vừa cải thiện.
- Hai run chỉ dùng validation để quyết định, không mở test/OOF. Baseline hiện
  được khóa; bước tiếp theo là TTA/bias correction và Deeplasia single-model,
  không tăng augmentation mù quáng.

## P9-I – TTA và bias correction trên P7 OOF

- Đã hoàn tất trên toàn bộ 14.036 mẫu OOF; không truy cập test.
- TTA Deeplasia-style đạt MAE **6,210446 tháng**, RMSE 8,382443; paired delta
  so với raw tái suy luận **−0,107025**, CI **[−0,135977; −0,078220]**.
- Raw bias correction cross-fitted đạt MAE 6,324285; delta +0,006815,
  CI [−0,001070; +0,014648], không có bằng chứng cải thiện.
- TTA + bias correction đạt MAE 6,228665, kém TTA đơn độc +0,018219,
  paired CI [+0,012482; +0,023813].
- Quyết định: giữ TTA làm ứng viên inference; loại bias correction khỏi pipeline
  chính hiện tại; chuyển sang Deeplasia single-model EfficientNet-B0.
- Artifact: `p9_inference/P9_I_HANDOFF.md` và
  `p9_inference/outputs/P9_I_TTA_BIAS_OOF/`.

## So sánh mốc tham khảo

- Bram et al., *The American Journal of Sports Medicine* (2025): **3,68 tháng** trên RSNA test.
- Rassmann et al., *Pediatric Radiology* (2024), Deeplasia: **3,87 tháng** trên RSNA test.
- P8 tốt hơn baseline nội bộ P7 OOF không phải phép so sánh trực tiếp (OOF 14.036 khác test 200), nhưng **chưa đạt** hai mốc công bố.

## Thí nghiệm C — C3-ROI local/global (2026-08-24)

- Đã train đủ 5 fold trên đúng split P7, seed 42; OOF khớp 14.036 ID với E1.
- C3-ROI standalone: MAE **6,437349**, kém E1 0,120658 tháng.
- Ensemble cố định `0,5 × E1 + 0,5 × C3-ROI`: MAE **6,176212**, RMSE
  8,346747; cải thiện 0,140480 tháng so với E1.
- Paired bootstrap 95% CI của delta ensemble−E1:
  `[-0,172547; -0,109853]`; ensemble cải thiện ở cả 5 fold.
- Prediction correlation E1/C3 là 0,995123: hai model rất giống nhau nhưng vẫn
  có diversity đủ để ensemble có lợi.
- ROI thực tế là segmentation bounding-box + margin 8%; fallback full-image
  18,49% trên development và 33% trên test. Run **không đạt** gate fallback ≤1%
  của thiết kế C ban đầu và không được gọi là local carpal specialist thuần túy.
- Test 200 ảnh chỉ thăm dò: E1 4,730321; C3-ROI 4,337267; ensemble 4,454661.
  Không dùng kết quả này để đổi trọng số hoặc chọn model.
- Báo cáo đầy đủ: `AI_Context/24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md`.

## Điểm tốt

- Split chính thức 12.611 train / 1.425 validation / 200 test được kiểm tra ID, ảnh, duplicate, hash và leakage.
- P7 OOF có 14.036 dự đoán, ID duy nhất, bootstrap CI và subgroup metrics.
- Checkpoint/resume, SHA manifest, early stopping và audit storage được thiết kế để tái lập qua nhiều tài khoản Colab.
- P8 là ensemble cố định, không tối ưu trọng số dựa trên nhãn test.

## Điểm chưa tốt/rủi ro

- Sai số P8 còn cao hơn Bram khoảng 1,05 tháng và Deeplasia khoảng 0,86 tháng; chưa thể ghi “vượt trội”.
- Test chỉ có 200 ảnh nên CI rộng; không được lặp lại nhiều lần để chọn mô hình.
- P7 dùng `preprocessing=none`; nhánh mask B1 không cải thiện validation. Có thể
  còn khoảng cách do chưa tái lập đầy đủ preprocessing/inference/ensemble của
  Deeplasia; P10 cho thấy augmentation mức vừa đơn độc chưa đủ.
- P4 D1 gây feature collapse; D2 không cải thiện; D3/LDL chỉ cải thiện rất nhỏ và CI chứa 0.
- Bộ 200 ảnh đã làm sạch artifact và hướng inpainting/generative trước đây chưa phải phần của protocol P7/P8; không được trộn vào so sánh chính nếu chưa có thiết kế paired và audit độc lập.

## P9-B0 – Deeplasia EfficientNet-B0 screening (2026-08-21)

- Đã triển khai EfficientNet-B0 512 + sex embedding 32, head 256/dropout 0,2,
  MSE, Adam và ReduceLROnPlateau; unit tests 16/16 và preflight đều PASS.
- Cache deeplasia_mask_v1 còn đủ 12.611 train / 1.425 validation; không dùng test.
- Screening validation không đạt control P10-B0 MAE 6,184792:
  - stem 3 kênh, batch 12: best MAE 8,5293 ở epoch 1;
  - stem 1 kênh, batch 12: best MAE 8,5823 ở epoch 2;
  - stem 1 kênh, batch 24: best MAE 11,0428 ở epoch 1.
- Các run không có NaN/Inf/OOM và được dừng để review; không chạy OOF/TTA/ensemble
  cho EfficientNet-B0 và không mở test.
- Handoff: p9_single_model/P9_B0_HANDOFF.md.
