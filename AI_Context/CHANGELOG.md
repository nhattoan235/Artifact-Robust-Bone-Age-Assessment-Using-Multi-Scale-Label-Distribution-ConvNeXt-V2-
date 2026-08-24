# AI_Context changelog

## 2026-08-21 – C1 implementation complete

- Added `c1_curated` manifest audit, sex × age stratification and deterministic mild weighted sampler.
- Full SHA/image readability audit PASS: 12.611 train + 1.425 official validation kept, 0 exclusions.
- Added C1 primary, local and smoke configs; primary C1 train manifest hash `59b68c4b...e300095`.
- Added weighted sampler resume support; interrupted and uninterrupted smoke predictions were byte-identical.
- Added `c1_curated/C1_HANDOFF.md`; long C1 training has not started and RSNA test was not used.

## 2026-08-21 – Khóa thiết kế curated data + E3 + ensemble

- Thống nhất hai nhánh độc lập: C1 curated-data và E3 hai model nam/nữ.
- Giữ nguyên RSNA official test 200; không dùng test để lọc dữ liệu, chọn model hoặc ensemble weight.
- Tách C1 thành C1-Audit và C1-Balanced; primary candidate giữ toàn bộ ảnh hợp lệ và dùng sampling nhẹ theo sex × age.
- E3-full là đối chứng bắt buộc trước E3-curated; chỉ ensemble bằng prediction OOF cùng ID.
- Ghi protocol, gate, rủi ro và decision log trong `09_CURATED_E3_ENSEMBLE_PLAN.md`.

## 2026-08-19 – Tạo bộ hồ sơ bàn giao AI_Context

- Tạo `00_START_HERE.md`, `01_STATUS_RESULTS.md`, `02_METHOD_HISTORY.md`, `03_DATA_PROTOCOL.md`, `04_NEXT_P9_PLAN.md`, `05_FILE_MAP.md`.
- Ghi rõ P7 OOF PASS (MAE 6,31669) và P8 test ensemble PASS kỹ thuật (MAE 4,73032).
- Ghi rõ kết quả hiện tại chưa vượt Bram 2025 (3,68) hoặc Deeplasia 2024 (3,87).
- Ghi rõ giới hạn quan trọng: test 200 đã được đọc ở P8; P9 phải chọn bằng OOF và cần external hold-out mới nếu muốn tuyên bố confirmatory.
- Ghi lại các nhánh bị loại: mask B1, ConvNeXtV2-FCMAE D1, multi-scale D2, LDL D3 chưa đủ bằng chứng.
- Ghi lại kế hoạch P9/P10 ưu tiên Deeplasia-faithful reproduction, inference
  TTA/bias correction, ensemble dị thể và calibration leakage-safe.
- Bổ sung `06_PIPELINE_COMPARISON.md`: phân biệt ưu thế về reproducibility/audit với ưu thế về mô hình và hiệu năng; kết luận P7/P8 chưa phải state-of-the-art.

## 2026-08-20 – P9 A1 hoàn tất

- Cache `deeplasia_mask_v1` development-only đủ 14.036 ảnh; summary/preflight PASS.
- A1 ConvNeXt-Tiny + L1, 100 epoch tối đa, early-stop epoch 30; best epoch 15.
- Validation MAE 6,307675; paired delta so với P2 A2 là +0,122884 tháng,
  CI [-0,031592; +0,277028], chưa có bằng chứng cải thiện.
- Không mở nhãn test; A1 chưa được đưa sang OOF. Bước tiếp theo là A0 cùng recipe
  để phân tách tác động của loss/hyperparameter khỏi preprocessing.

## 2026-08-20 – P9 A0 reference hoàn tất

- A0 L1/100 epoch early-stop epoch 42, best epoch 27; validation MAE 6,391447.
- So với P2 A2, delta +0,206656 tháng, CI [+0,020061; +0,394780].
- A1 tốt hơn A0 -0,083772 tháng nhưng CI [-0,268072; +0,097423] chứa 0.
- Kết luận: A1 chưa chứng minh cải thiện; không mở test/OOF. Cần xem xét recipe
  Bram/Deeplasia trước khi tiêu tốn tài nguyên cho A2 histogram.

## 2026-08-19 – P8 audit và suy luận ensemble

- Audit input PASS: đủ 5 fold, model tensor/manifest SHA hợp lệ, OOF 14.036 ID duy nhất, test 200 ảnh và ground truth khớp.
- Chạy equal-weight 5-fold ensemble trên RSNA test.
- Kết quả: MAE 4,730321; RMSE 6,028745; median AE 4,133363; bootstrap CI [4,222475; 5,263709].
- Artifact: `p8_test_ensemble/outputs/P8_test_ensemble_report.json`, `P8_ensemble_predictions.csv`.

## 2026-08-17 – P7 final V3 hoàn tất

- Hoàn thành 5 fold với early stopping/checkpoint/resume qua nhiều tài khoản Colab.
- Smoke interruption step 2 → step 4 PASS; xác nhận checkpoint append-only và storage identity.
- OOF PASS: 14.036 dòng, MAE 6,316691, no test used.
- Vấn đề vận hành đã xử lý: Drive đầy do bundle/cache/checkpoint dư; lỗi BF16 T4; mismatch tài khoản storage; dừng Colab và resume.

## 2026-08-16 trở về trước – các phase phát triển

- P0: khóa split 12.611/1.425/200, manifest/hash, không leakage.
- P1: dựng baseline ConvNeXt-Tiny, checkpoint/resume và cảnh báo; chuyển ưu tiên BF16 sau FP16 Inf smoke.
- P2: A2 flip + biến đổi nhẹ thắng A0/A1 trên validation; khóa A2.
- P3: official mask B1 không cải thiện MAE; giữ `preprocessing=none`.
- P4: D1 collapse và bị loại; D2 không cải thiện; D3 chỉ tín hiệu nhỏ.
- P5: D0 thắng quy tắc nhiều seed; không giữ LDL làm model chính.
- P6: 768 không cải thiện có ý nghĩa so với 512; khóa 512.

## Quy ước cập nhật

Mỗi lần chạy mới phải ghi: ngày, phase/run ID, commit/code hash, config hash, data manifest hash, seed, GPU, checkpoint cuối, MAE/RMSE/CI, cảnh báo, quyết định và đường dẫn artifact. Không xóa artifact cũ; nếu dọn Drive, tải archive và ghi hash trước.

## 2026-08-24 – Thí nghiệm C / C3-ROI hoàn tất

- Train đủ 5 fold `C3_ROI_V1` trên split P7 khóa, seed 42; OOF đủ 14.036 ID,
  target/sex khớp E1.
- C3-ROI standalone MAE 6,437349; E1 raw 6,316691.
- Ensemble cố định E1+C3 50/50 đạt MAE **6,176212**, giảm 0,140480 tháng;
  paired bootstrap 95% CI `[-0,172547; -0,109853]`, cải thiện cả 5 fold.
- Test thăm dò 200 ảnh: E1 4,730321; C3 4,337267; ensemble 4,454661.
  Không tune trọng số theo test và không xem đây là xác nhận độc lập.
- Audit phát hiện protocol deviation: full-image fallback 18,49% development
  và 33% test, vượt gate ≤1% của thiết kế C. Kết quả được giữ như ablation OOF
  dương tính, không claim local carpal specialist thuần túy.
- Thêm báo cáo: `AI_Context/24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md`.

## 2026-08-20 – P10 B0 đang chạy: tái lập recipe P2

- Đã tạo config `p1_baseline/configs/p10_b0_p2_control.toml` với đúng recipe P2
  đã thắng trước đây: ConvNeXt-Tiny 512, sex embedding, light augmentation,
  SmoothL1 (beta 3 tháng), LR 2e-4, WD 0,05, dropout 0,2, 35 epoch.
- Preflight PASS và smoke interruption → resume PASS; không có đường dẫn test.
- Run đang chạy tại `p9_preprocessing/runs/P10_B0_P2_CONTROL_CONVNEXT_TINY_SEED42`.
- Mục tiêu là xác nhận pipeline P9 không làm thay đổi baseline trước khi thử
  augmentation Deeplasia mức vừa và preprocessing từng biến một.

## 2026-08-20 – P10 B0 hoàn tất: baseline được tái lập chính xác

- Early-stop ở epoch 20; checkpoint tốt nhất tại epoch 12.
- Validation MAE **6,1847917**, RMSE **8,4865**, median AE **5,0 tháng**.
- Dự đoán của P10-B0 và P2 cũ giống hệt trên 1.425 validation IDs
  (`max_abs_pred_diff = 0`); đây là xác nhận pipeline P9 không gây drift.
- Không mở test/OOF. Baseline đã được khóa; bước tiếp theo là augmentation
  Deeplasia mức vừa, giữ nguyên recipe này và chỉ thay augmentation.

## 2026-08-20 – P10 B1 đang chạy: Deeplasia augmentation mức vừa

- Giữ nguyên P10-B0 control; chỉ đổi augmentation sang `deeplasia_fancy`:
  rotation ±15°, translation ±10%, scale 0,90–1,10, shear ±5°, CLAHE 20%,
  sharpen 15%, brightness/contrast ±15%, gamma 0,85–1,15.
- Preflight PASS và smoke interruption → resume PASS; không có đường dẫn test.
- Run: `p9_preprocessing/runs/P10_B1_DEEPLASIA_MODERATE_CONVNEXT_TINY_SEED42`.
- PID hiện tại: `23168`. Chỉ giữ ứng viên nếu validation MAE cải thiện rõ so với
  control 6,1847917; không dùng test để chọn.

## 2026-08-20 – P10 B1 hoàn tất: augmentation mức vừa chưa cải thiện

- Early-stop ở epoch 22; best epoch 14.
- Validation MAE **6,1858772**, RMSE **8,4255**, median AE **5,0 tháng**.
- So với P10-B0: delta MAE **+0,001086 tháng**, bootstrap 95% CI xấp xỉ
  `[-0,1404; +0,1406]`; RMSE giảm nhẹ nhưng MAE không cải thiện.
- Kết luận: augmentation mức vừa không có bằng chứng giúp ích; không đưa vào
  OOF/test. Cần chuyển sang phân tích lỗi/ROI hoặc một ablation nhỏ có cơ sở,
  không tiếp tục tăng độ mạnh augmentation mù quáng.

## 2026-08-21 – P9-I hoàn tất: TTA có lợi, bias correction không giữ

- Tạo `p9_inference/tta_bias_oof.py` và protocol/handoff tương ứng.
- Chạy đủ 14.036 OOF bằng 5 checkpoint P7, TTA xoay `[-10,-5,0,5,10]` độ,
  có và không flip; không truy cập đường dẫn/nhãn test.
- Raw tái suy luận MAE 6,317471; sai khác so với OOF gốc median 0,02393
  tháng, max 0,36450 tháng.
- TTA đạt MAE **6,210446**; paired delta so với raw `-0,107025`, CI
  `[-0,135977; -0,078220]`.
- Raw + bias correction cross-fitted đạt MAE 6,324285; không cải thiện.
- TTA + bias correction đạt 6,228665 và kém TTA đơn độc; không giữ correction.
- Quyết định: giữ TTA làm ứng viên inference, chuyển bước tiếp theo sang
  Deeplasia single-model EfficientNet-B0 512.

## 2026-08-21 – P9-B0 screening hoàn tất: EfficientNet-B0 chưa đạt gate

- Đã triển khai model/head, MSE, Adam/ReduceLROnPlateau và augmentation
  Deeplasia gần mã gốc; unit tests 16/16, preflight và smoke/resume PASS.
- Ba candidate validation lần lượt đạt best MAE 8,5293; 8,5823; 11,0428 tháng,
  đều kém P10-B0 control 6,184792.
- Không có NaN/Inf/OOM; không dùng test, không chạy OOF/TTA/ensemble.
- Dừng P9-B0 để review. Artifact chính:
  p9_single_model/P9_B0_HANDOFF.md.

## 2026-08-21 – P11 Giai đoạn 1 sex-aware PASS

- Đã khóa và triển khai E0 image-only, E1 sex embedding tương thích ngược và E2
  shared bottleneck + dual scalar output M/F gần parameter-matched.
- Unit test 24/24, full-config preflight E0/E2 và CPU smoke E0/E2 đều PASS.
- E2 GPU interrupt/resume PASS trên RTX 4050: split/config/code hash khớp,
  optimizer/scheduler/scaler phục hồi, peak VRAM 1.800 MiB, không warning.
- E1 strict-load checkpoint P10-B0 và 12/12 prediction khớp tuyệt đối
  (`max_abs_diff_months = 0.0`).
- Chưa chạy official validation seed 42; không có kết luận hiệu năng mới.
- Handoff: `p11_sex_aware/P11_STAGE1_HANDOFF.md`.

## 2026-08-22 – P11 Giai đoạn 2 hoàn tất: E2 không đạt gate

- E0 image-only early-stop epoch 30, best epoch 22, validation MAE 7,4717.
- E2 shared dual-output early-stop epoch 23, best epoch 15, MAE 6,1571.
- Control E1 sex embedding MAE 6,1848; paired trên cùng 1.425 ID, bootstrap
  10.000 lần, seed 2026.
- E0−E1 delta +1,2869 tháng, CI [+1,0114; +1,5667], xác nhận sex mang thông
  tin dự đoán rõ ở cả nữ và nam trong recipe đã khóa.
- E2−E1 delta −0,0277 tháng, CI [−0,1831; +0,1254]; cải thiện nữ chỉ 0,0052
  tháng. E2 không đạt gate overall 0,10 hoặc female 0,20 tháng.
- Giữ E1; không chạy seed bổ sung, OOF hoặc E3 cho E2; không dùng RSNA test.
- Handoff: `p11_sex_aware/P11_STAGE2_HANDOFF.md`.

## 2026-08-22 – Khóa protocol P12 uncertainty OOF

- Chuyển sang phân tích TTA disagreement–absolute error trên 14.036 P7/P9-I
  OOF, không huấn luyện mới và không dùng test.
- Khóa primary Spearman + bootstrap CI; secondary sex × age, TTA gain,
  worst-group, error >12/>18, AUROC và risk–coverage.
- Không gọi disagreement là uncertainty lâm sàng đã hiệu chuẩn và không chọn
  threshold deployment trên cùng OOF.
- Protocol: `AI_Context/09_P12_UNCERTAINTY_PROTOCOL.md`.

## 2026-08-22 – P12 uncertainty OOF hoàn tất

- Integrity PASS 14.036 OOF ID duy nhất; TTA mean/std recompute khớp 10 views;
  compile và unit test 6/6 PASS.
- Primary Spearman giữa disagreement và TTA absolute error ρ=0,2004,
  bootstrap 95% CI [0,1842; 0,2164]; H4 association được ủng hộ nhưng effect yếu.
- AUROC nhận biết lỗi >12/>18 tháng lần lượt 0,6258/0,6341, chỉ mức hạn chế.
- Disagreement quartile cao có MAE 7,6238 so với 4,7587 ở quartile thấp;
  error >12 cao gấp 2,86 lần và error >18 cao gấp 3,29 lần.
- Association mạnh hơn ở nam (ρ=0,2386) so với nữ (ρ=0,1462), không đồng đều
  theo tuổi; F 0–59 không có association, worst-error M 60–119 chỉ ρ=0,0870.
- TTA gain overall +0,1070 tháng, CI [+0,0775; +0,1361], nhưng không đồng đều
  giữa sex × age.
- Giữ disagreement như biến triage nghiên cứu, không gọi là uncertainty lâm
  sàng, không tối ưu threshold trên OOF và không dùng RSNA test.
- Handoff: p12_uncertainty/P12_HANDOFF.md.

## 2026-08-22 – P13 đóng gói kết quả luận văn hoàn tất

- Khóa nguồn E0/E1/E2 trên cùng 1.425 validation ID và nguồn OOF P12; không
  chạy inference/huấn luyện mới và không truy cập RSNA test.
- Tạo generator tái lập `p13_reporting/build_thesis_assets.py`; paired bootstrap
  10.000 lần, seed 2026; manifest ghi SHA-256 đầu vào/đầu ra và
  `test_accessed=false`.
- Sinh 4 bảng CSV/Markdown và 3 hình PNG/PDF; full build PASS, unit test 5/5
  PASS, visual QA 3/3 hình PASS; repeat-build có 0/15 artifact đổi hash.
- Kết luận khóa: H1 supported; H2/H3 not supported; H4 association supported
  nhưng utility hạn chế. Giữ E1 sex embedding làm baseline chính.
- Soạn bản thảo Methods–Results–Discussion–Limitations và quy tắc diễn đạt tại
  `p13_reporting/P13_THESIS_DRAFT_VI.md`.
- Handoff: `p13_reporting/P13_HANDOFF.md`.

## 2026-08-22 – Báo cáo đánh giá đóng góp khoa học

- Tổng hợp toàn bộ bằng chứng P0–P13 và xác định đề tài có đóng góp khoa học rõ
  ở cấp đồ án tốt nghiệp, thuộc nhóm thực nghiệm và phương pháp đánh giá.
- Khóa bốn đóng góp để trình bày trong luận văn: giá trị dự đoán của giới tính;
  kết quả âm dual-output; giá trị/giới hạn của TTA disagreement; và quy trình
  thực nghiệm tái lập, chống leakage.
- Phân biệt rõ đóng góp khoa học với tuyên bố state of the art; ghi nhận các giới
  hạn gồm P11 một seed, chưa có external holdout chưa bị tác động và uncertainty
  chưa được hiệu chỉnh.
- Báo cáo: `AI_Context/11_SCIENTIFIC_CONTRIBUTIONS_REPORT.md`.
