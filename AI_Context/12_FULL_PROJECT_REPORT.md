# Báo cáo tổng hợp toàn bộ dự án dự đoán tuổi xương từ ảnh X-quang bàn tay

**Phạm vi:** P0–P13 của repository `D:\Hoctap\Doan_totnghiep`
**Ngày khóa:** 2026-08-22
**Đơn vị mục tiêu:** tháng tuổi xương
**Vị trí khoa học:** nghiên cứu thực nghiệm có kiểm soát ở cấp đồ án cử nhân; không phải tuyên bố state of the art hay xác nhận khái quát hóa lâm sàng.

## 0. Nguồn đã đọc và các điểm cần giải thích

Đã đọc đầy đủ theo thứ tự bắt buộc: [00_START_HERE.md](../AI_Context/00_START_HERE.md), [context_index.json](../AI_Context/context_index.json), [01_STATUS_RESULTS.md](../AI_Context/01_STATUS_RESULTS.md), [02_METHOD_HISTORY.md](../AI_Context/02_METHOD_HISTORY.md), [03_DATA_PROTOCOL.md](../AI_Context/03_DATA_PROTOCOL.md), [05_FILE_MAP.md](../AI_Context/05_FILE_MAP.md), [08_SEX_AWARE_EXPERIMENT_PLAN.md](../AI_Context/08_SEX_AWARE_EXPERIMENT_PLAN.md), [09_P12_UNCERTAINTY_PROTOCOL.md](../AI_Context/09_P12_UNCERTAINTY_PROTOCOL.md), [10_P13_THESIS_REPORTING_PLAN.md](../AI_Context/10_P13_THESIS_REPORTING_PLAN.md), [11_SCIENTIFIC_CONTRIBUTIONS_REPORT.md](../AI_Context/11_SCIENTIFIC_CONTRIBUTIONS_REPORT.md) và [CHANGELOG.md](../AI_Context/CHANGELOG.md).

Đã kiểm tra chéo handoff/artifact: [P0_HANDOFF.md](../p0_audit/P0_HANDOFF.md), [P1_HANDOFF.md](../p1_baseline/P1_HANDOFF.md), [P2_HANDOFF.md](../p1_baseline/P2_HANDOFF.md), [P3_HANDOFF.md](../p3_preprocessing/P3_HANDOFF.md), [P4_HANDOFF.md](../p4_architecture/P4_HANDOFF.md), [P5_HANDOFF.md](../p5_seed_confirmation/P5_HANDOFF.md), [P6_PROTOCOL.md](../p6_resolution/P6_PROTOCOL.md), [P7_OOF_report.json](../p7_final_v3/P7_OOF_report.json), [P9-I](../p9_inference/P9_I_HANDOFF.md), [P9-B0](../p9_single_model/P9_B0_HANDOFF.md), hai handoff và hai JSON P11, handoff/report/CSV P12, cùng handoff/bản thảo/manifest/bảng P13.

Có bốn điểm cần ghi rõ:

1. `p7_final_v3/P7_5FOLD_AUDIT.txt` không tồn tại ở root; `rg --files` tìm thấy bản tương ứng tại [`P7_5FOLD_AUDIT.txt`](../P7_5FOLD_AUDIT.txt). File này ghi `FINAL FIVE-FOLD AUDIT: FAIL` vì fold 1 thiếu block “best-epoch metrics”, nhưng vẫn ghi các kiểm tra cốt lõi đạt: 14.036 dòng, ID duy nhất, disjoint giữa fold, MAE khớp state và manifest/model integrity. [P7_OOF_report.json](../p7_final_v3/P7_OOF_report.json) ghi `status=PASS`. Vì vậy báo cáo gọi endpoint OOF là **PASS theo báo cáo OOF chính thức, kèm cảnh báo bất nhất hành chính của audit file**; không tự sửa hoặc rerun.
2. Point estimate P11 trong JSON và bảng P13 khớp; CI chênh rất nhỏ do bootstrap/build artifact khác nhau. Vì P13 là bộ báo cáo khóa, báo cáo này ưu tiên [table 1](../p13_reporting/outputs/P13_THESIS_REPORT/table_1_model_comparison.md) và [table 2](../p13_reporting/outputs/P13_THESIS_REPORT/table_2_paired_forest_data.md).
3. P0 chỉ audit cấu trúc/hash của test và chưa đọc tuổi. P8 về sau đã đọc ground truth để tính metric. Do đó P8 là benchmark kỹ thuật trên test đã chạm, không phải external holdout mới cho các quyết định P9–P13.
4. P7/P9/P12 là validation hoặc OOF; P8 là test 200 ảnh. Không dùng chênh lệch P7–P8 như paired improvement.

## 1. Tóm tắt

Đề tài xây dựng và kiểm toán pipeline dự đoán tuổi xương từ X-quang bàn tay RSNA, tập trung vào giá trị của giới tính, chi phí của độ phức tạp mô hình và utility của TTA disagreement. Chuỗi nghiên cứu gồm audit dữ liệu, baseline ConvNeXt-Tiny, augmentation, masking, ablation kiến trúc, seed confirmation, độ phân giải, 5-fold OOF, ensemble test, TTA/bias correction, screening EfficientNet-B0, sex-aware, uncertainty và reporting.

Mô hình được giữ là **ConvNeXt-Tiny ImageNet-1K, 512 px, grayscale lặp ba kênh, A2 augmentation, `preprocessing=none`, sex embedding và direct regression**. TTA mười view được giữ làm cải tiến inference; bias correction bị loại. P7 pooled OOF trên 14.036 ảnh đạt MAE **6,316691 tháng**, bootstrap 95% CI **[6,224638; 6,411327]**. P8 equal-weight ensemble trên 200 ảnh test đạt MAE **4,730321 tháng**, CI **[4,222475; 5,263709]**, nhưng chưa vượt các mốc tham khảo 3,68–3,87 tháng và không được xem là external holdout chưa bị tác động.

P11 cho thấy E0 image-only MAE 7,471711, E1 sex embedding 6,184792 và E2 dual-output 6,157061 trên cùng 1.425 validation ID. E0−E1 là **+1,286919 tháng**, CI P13 **[+1,012226; +1,569611]**; E2−E1 là **−0,027730 tháng**, CI **[−0,179409; +0,124151]**. Như vậy giới tính mang giá trị dự đoán lớn, còn dual-output không đạt gate.

P12 trên 14.036 OOF cho `rho=0,200431` giữa disagreement và absolute error, CI **[0,184238; 0,216411]**; AUROC lỗi >12 và >18 tháng lần lượt 0,625814 và 0,634108. Q4 disagreement có MAE 7,623798 so với 4,758689 ở Q1, lỗi >12 tháng gấp 2,86 lần và >18 tháng gấp 3,29 lần. Tín hiệu phù hợp để phân tầng nghiên cứu, chưa phải uncertainty lâm sàng đã hiệu chỉnh.

> **Thông điệp trung tâm:** Trong protocol được khảo sát, thông tin giới tính tạo ra cải thiện đáng tin cậy lớn hơn hầu hết thay đổi về preprocessing, độ phân giải hoặc kiến trúc; sex embedding đơn giản đạt tỷ lệ hiệu năng–độ phức tạp hợp lý, TTA có lợi ích nhỏ nhưng nhất quán, còn dual-output và disagreement cho thấy giới hạn của việc tăng độ phức tạp hoặc diễn giải tín hiệu hậu nghiệm quá mạnh.

## 2. Bối cảnh và động cơ

Tuổi xương là chỉ dấu hình ảnh của trưởng thành xương, thường được ước lượng từ X-quang bàn tay. Đây là bài toán hồi quy liên tục theo tháng, chịu ảnh hưởng của hình thái xương, giai đoạn phát triển và phân bố dân số. Một MAE thấp từ một lần train không đủ để khẳng định giá trị khoa học, khả năng tái lập hoặc an toàn sử dụng.

Đề tài đặt ra ba vấn đề. Một là các can thiệp như augmentation, masking, resolution, LDL hoặc backbone có thể thay đổi metric nhưng không nhất thiết có effect ổn định. Hai là giới tính có thể là biến điều kiện hóa hữu ích, nhưng giá trị dự đoán không phải causal effect. Ba là TTA disagreement có thể xếp hạng rủi ro, nhưng tương quan thống kê không đồng nghĩa clinical uncertainty.

Vì vậy, mục tiêu không chỉ là săn điểm MAE thấp nhất mà là đo giá trị biên của từng can thiệp bằng cùng ID, paired effect size, bootstrap CI, gate định trước, OOF và audit artifact.

## 3. Mục tiêu và câu hỏi nghiên cứu

### 3.1. Mục tiêu

Xây dựng pipeline có thể tái lập và đánh giá có kiểm soát vai trò của augmentation, preprocessing, kiến trúc, độ phân giải, giới tính và TTA disagreement trong dự đoán tuổi xương RSNA.

### 3.2. Câu hỏi

| Mã | Câu hỏi |
|---|---|
| RQ1 | Bổ sung giới tính có cải thiện so với image-only không? |
| RQ2 | Shared backbone với hai output theo giới có tốt hơn sex embedding không? |
| RQ3 | TTA disagreement có liên hệ với absolute error và làm giàu nhóm sai số cao không? |
| RQ4 | TTA và disagreement có đồng nhất giữa giới tính–nhóm tuổi không? |
| RQ5 | Các thay đổi về augmentation, masking, backbone, LDL và resolution có đủ lợi ích để biện minh cho độ phức tạp không? |

P11 khóa H1–H3; P12 khóa H4. Gate E2: overall cải thiện ít nhất 0,10 tháng, hoặc female cải thiện ít nhất 0,20 tháng và overall không xấu quá 0,05 tháng; đồng thời không collapse, không làm subgroup xấu nghiêm trọng và cần tín hiệu ổn định nếu qua screening. Nguồn: [P11 experiment plan](../AI_Context/08_SEX_AWARE_EXPERIMENT_PLAN.md), [P12 protocol](../AI_Context/09_P12_UNCERTAINTY_PROTOCOL.md).

## 4. Dữ liệu và protocol chống leakage

### 4.1. Split đã khóa

| Split | Số ảnh | Vai trò | Dùng chọn model? |
|---|---:|---|---|
| Official train | 12.611 | Fit trọng số | Có |
| Official validation | 1.425 | Chọn checkpoint/cấu hình development | Có |
| Development pool | 14.036 | 5-fold OOF, P7/P9/P12 | Mỗi prediction do fold không chứa ảnh tạo ra |
| RSNA test | 200 | Benchmark sau khi khóa | Không |

P0 kiểm tra ID, SHA-256, PNG, miền tuổi, giới tính và annotation validation. Không phát hiện duplicate hoặc overlap ID/SHA giữa split trong audit đã biết. Validation có 773 nam và 652 nữ; phân bố tuổi khác nhau theo giới, nên final 5-fold stratify theo sex × age bin.

P0 chưa đọc tuổi test. P8 đọc ground truth sau khi checkpoint/ensemble rule khóa. P9–P13 chỉ dùng train/validation hoặc OOF cho chọn model, TTA, correction, gate và reporting. Manifest P12/P13 có `test_accessed=false` cho chính các phase đó, không phủ nhận P8 đã chạm test.

### 4.2. Quy tắc kiểm soát

- Không tách lại official validation.
- Mỗi run ghi data manifest, config/code hash, seed và environment.
- Paired comparison chỉ trên cùng ID và endpoint.
- P7 phải có đúng một prediction OOF cho mỗi 14.036 ID.
- Không dùng RSNA test để chọn checkpoint, augmentation, loss, ensemble weight, threshold hoặc câu chuyện sau P8.

Nguồn: [P0_HANDOFF.md](../p0_audit/P0_HANDOFF.md), [03_DATA_PROTOCOL.md](../AI_Context/03_DATA_PROTOCOL.md), [P7_OOF_report.json](../p7_final_v3/P7_OOF_report.json), [P12 report.json](../p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/report.json).

## 5. Baseline và hạ tầng tái lập

P1 dùng ConvNeXt-Tiny pretrained ImageNet-1K V1, input 512×512, grayscale pad vuông rồi lặp ba kênh và chuẩn hóa ImageNet. Baseline có sex embedding 16 chiều, direct regression trên tuổi chuẩn hóa theo train mean/SD, Smooth L1 beta tương đương 3 tháng, AdamW `lr=2e-4`, weight decay `0,05`, cosine schedule, batch 4, gradient accumulation 8 ở A0, tối đa 35 epoch và early stopping patience 8 theo validation MAE.

Checkpoint nguyên tử chứa model, optimizer, scheduler, AMP scaler, epoch/step, early-stop state, RNG Python/NumPy/PyTorch, seed và hash. FP16 smoke đầu tạo gradient Inf; chuyển sang BF16 khi hỗ trợ. Smoke BF16 cuối không OOM, không skipped batch, gradient hữu hạn và checkpoint đọc lại được. Stop/resume xác nhận model tensor và training state bitwise equal ở global step 8.

Mỗi run có log loss/MAE/RMSE/median/accuracy/subgroup, warnings, metrics, environment, config, state, prediction CSV và checkpoint. Cảnh báo NaN/Inf, collapse, out-of-range hoặc hash mismatch có cơ chế dừng an toàn. Smoke MAE trên 8 ảnh không phải kết quả khoa học.

Nguồn: [P1_HANDOFF.md](../p1_baseline/P1_HANDOFF.md), [P11_STAGE1_HANDOFF.md](../p11_sex_aware/P11_STAGE1_HANDOFF.md).

## 6. Toàn bộ tiến trình P0–P13

| Phase | Mục tiêu | Can thiệp/protocol | Kết quả chính | Quyết định |
|---|---|---|---|---|
| **P0** | Audit split/leakage/môi trường | 12.611/1.425/200; ID/SHA, PNG, nhãn, manifest | Không duplicate/leakage đã biết; validation khớp annotation | Khóa protocol, cho phép P1 |
| **P1** | Baseline tái lập | ConvNeXt-Tiny 512; sex embedding; direct regression; checkpoint/resume/AMP | Unit, preflight, smoke, resume PASS; FP16 Inf được phát hiện | Cho phép P2 |
| **P2** | Chọn augmentation | A0 none; A1 flip; A2 flip + hình học/cường độ nhẹ | 6,640; 6,514; **6,185**; A2−A0 −0,455, CI [−0,658;−0,249] | Giữ A2 |
| **P3** | Kiểm tra masking | B1 full-hand mask, không crop/rotate | B1 6,23894 so với B0 6,18479; delta +0,05414, CI chứa 0 | `preprocessing=none` |
| **P4** | Ablation kiến trúc | D1 ConvNeXtV2 FCMAE; D2 multi-scale; D3 LDL; D0 | D1 31,96281 do collapse; D2 6,29697; D3 6,14541 nhưng CI cắt 0 | D0/D3 sang P5 |
| **P5** | Seed confirmation | D0/D3 trên seed 17/42/123 | D3 fused mean delta −0,03721, CI [−0,13521;+0,05863], có lợi 2/3 seed | Giữ D0 |
| **P6** | Resolution | Chỉ đổi 512 thành 768 | delta −0,001371, CI [−0,190703;+0,192127]; pixel tăng 2,25 lần | Giữ 512 |
| **P7** | Final OOF | 5 fold trên development 14.036; một prediction/ID | MAE 6,316691; RMSE 8,519420; CI [6,224638;6,411327]; fold SD 0,078775 | OOF phát triển chính |
| **P8** | Test benchmark | Fixed equal-weight 5-fold ensemble trên 200 test | MAE 4,730321; CI [4,222475;5,263709] | Benchmark kỹ thuật; test đã chạm |
| **P9** | Inference và recipe tham khảo | TTA/correction OOF; EfficientNet-B0 screening | TTA giảm 0,107025; correction không giúp; B0 8,5293–11,0428 | Giữ TTA; dừng B0 |
| **P10** | Kiểm soát recipe | B0 tái lập P2; B1 Deeplasia augmentation moderate | B0 6,184792 và prediction khớp; B1 6,185877, delta +0,001086, CI chứa 0 | Khóa baseline |
| **P11** | Sex-aware | E0 image-only; E1 embedding; E2 dual-output; paired gate | E0 7,471711; E1 6,184792; E2 6,157061; E0−E1 +1,286919; E2−E1 −0,027730 | H1 supported; giữ E1; dừng E2 |
| **P12** | Uncertainty OOF | 10 TTA views; Spearman/AUROC/quartile/risk-coverage/sex×age | rho 0,200431; AUROC 0,625814/0,634108; Q4 rủi ro hơn Q1 | Triage nghiên cứu, không clinical uncertainty |
| **P13** | Reporting | Build bảng/hình từ input khóa; hash, visual QA, repeat build | 4 bảng, 3 hình PNG/PDF; 9 input, 15 output; 0/15 hash đổi; test 5/5 PASS | Khóa báo cáo E1 + TTA |

Nguồn: [P0](../p0_audit/P0_HANDOFF.md), [P1/P2](../p1_baseline/P1_HANDOFF.md), [P2](../p1_baseline/P2_HANDOFF.md), [P3](../p3_preprocessing/P3_HANDOFF.md), [P4](../p4_architecture/P4_HANDOFF.md), [P5](../p5_seed_confirmation/P5_HANDOFF.md), [P6](../p6_resolution/P6_PROTOCOL.md), [P7](../p7_final_v3/P7_OOF_report.json), [P9-I](../p9_inference/P9_I_HANDOFF.md), [P9-B0](../p9_single_model/P9_B0_HANDOFF.md), [P11](../p11_sex_aware/P11_STAGE2_HANDOFF.md), [P12](../p12_uncertainty/P12_HANDOFF.md), [P13](../p13_reporting/P13_HANDOFF.md).

## 7. Kết quả định lượng chính

### 7.1. Phân biệt validation, OOF và test

| Endpoint | Tập/vai trò | N | MAE | RMSE | 95% CI | Diễn giải |
|---|---|---:|---:|---:|---|---|
| P10-B0/P2 control | Validation để chọn cấu hình | 1.425 | 6,184792 | 8,486471 | Không phải CI điểm MAE | Baseline được tái lập |
| P7 pooled OOF | Development; mỗi ID ngoài fold | 14.036 | 6,316691 | 8,519420 | [6,224638;6,411327] | Endpoint phát triển chính |
| P9-I raw | Cùng OOF; control inference | 14.036 | 6,317471 | 8,519945 | Không báo CI điểm | Control paired cho TTA |
| P9-I TTA | Cùng OOF; 10 view | 14.036 | 6,210446 | 8,382443 | TTA−raw [−0,135977;−0,078220] | Cải thiện inference |
| P8 ensemble | RSNA test đã đọc ground truth | 200 | 4,730321 | 6,028745 | [4,222475;5,263709] | Benchmark kỹ thuật, không external mới |

P7 accuracy ±6/±12/±18 là 59,0125%/86,7341%/95,5329%; P8 là 70,0%/94,0%/100,0%. Đây không phải paired comparison. Artifact P8 ghi mốc tham khảo nominal test là 3,87 tháng cho Deeplasia và 3,68 tháng cho Bram 2025; P8 4,730321 chưa đạt các mốc này.

Nguồn: [P7_OOF_report.json](../p7_final_v3/P7_OOF_report.json), [P9_I_HANDOFF.md](../p9_inference/P9_I_HANDOFF.md), [P8 report](../p8_test_ensemble/outputs/P8_test_ensemble_report.json).

### 7.2. Effect size của các can thiệp

| So sánh | Tập | Delta MAE (tháng) | Bootstrap 95% CI | Quyết định |
|---|---|---:|---|---|
| A1−A0 | Validation | −0,125 | [−0,315;+0,065] | Chưa chắc chắn |
| A2−A0 | Validation | −0,455 | [−0,658;−0,249] | Giữ A2 |
| B1−B0 | Validation | +0,05414 | [−0,10167;+0,21180] | Loại B1 |
| D1−D0 | Validation | +25,77802 | [+24,33264;+27,26304] | Collapse; loại |
| D2−D0 | Validation | +0,11218 | [−0,05414;+0,27870] | Loại |
| D3 fused−D0, seed 42 | Validation | −0,03938 | [−0,19707;+0,12156] | Chưa đủ |
| D3 fused−D0, mean 3 seed | Validation | −0,03721 | [−0,13521;+0,05863] | Không ổn định |
| 768−512 | Validation | −0,001371 | [−0,190703;+0,192127] | Giữ 512 |
| TTA−raw | OOF | −0,107025 | [−0,135977;−0,078220] | Giữ TTA |
| Raw correction−raw | OOF | +0,006815 | [−0,001070;+0,014648] | Loại correction |
| TTA+correction−TTA | OOF | +0,018219 | [+0,012482;+0,023813] | Correction làm xấu TTA |
| Deeplasia moderate−control | Validation | +0,001086 | CI chứa 0 | Không giữ |

Nguồn: [P2](../p1_baseline/P2_HANDOFF.md), [P3](../p3_preprocessing/P3_HANDOFF.md), [P4](../p4_architecture/P4_HANDOFF.md), [P5](../p5_seed_confirmation/P5_HANDOFF.md), [P6](../p6_resolution/P6_PROTOCOL.md), [P9-I](../p9_inference/P9_I_HANDOFF.md), [P9-B0](../p9_single_model/P9_B0_HANDOFF.md), [01_STATUS_RESULTS](../AI_Context/01_STATUS_RESULTS.md).

## 8. Phương pháp thành công và không thành công

Phương pháp thành công trong phạm vi khảo sát là A2, D0 ConvNeXt-Tiny, E1 sex embedding, TTA 10 views, cùng OOF/paired bootstrap/gate/audit. Đây là các lựa chọn tạo tín hiệu đủ rõ hoặc tạo độ tin cậy phương pháp luận.

| Nhánh không giữ | Kết quả | Kết luận đúng phạm vi |
|---|---|---|
| B1 full-hand masking | MAE cao hơn điểm; CI cắt 0 | Masking không giúp trong recipe B1; không phủ định mọi segmentation |
| D1 ConvNeXtV2 FCMAE | MAE 31,96281; feature collapse; dự đoán gần hai mức theo giới | Recipe D1 collapse; không suy rộng mọi ConvNeXtV2 |
| D2 multi-scale | MAE 6,29697; không cải thiện primary | Multi-scale cấu hình này không đủ bằng chứng |
| D3 LDL | Tốt hơn nhẹ seed 42, không ổn định 3 seed | Recipe sigma/loss/fusion này chưa chứng minh lợi ích |
| 768 px | Delta gần 0; chi phí pixel 2,25 lần | Tăng resolution không đáng trong so sánh này |
| Bias correction | Không cải thiện raw; làm TTA kém hơn | Không có utility trong OOF này |
| Deeplasia moderate | Delta +0,001086; CI chứa 0 | Không có bằng chứng cải thiện |
| EfficientNet-B0 | 8,5293–11,0428; dừng 1–3 epoch | Negative screening gần Deeplasia, chưa phủ định EfficientNet nói chung |
| E2 dual-output | Tốt hơn điểm 0,027730; CI cắt 0; không giảm sex gap | Không hơn E1 trong điều kiện hiện tại |

Không gọi các phương pháp này “hoàn toàn vô dụng”; mọi kết luận đều bị giới hạn bởi recipe, seed, endpoint và dữ liệu đã khảo sát.

## 9. Pipeline cuối cùng được giữ

### 9.1. Huấn luyện

Official train/validation tạo development; final OOF dùng 5 fold trên 14.036 ảnh. Ảnh được pad vuông, resize 512 px, grayscale lặp ba kênh, chuẩn hóa ImageNet, không background masking. A2 gồm flip p=0,5, rotation ±7°, translation tối đa 3%, scale 0,95–1,05, brightness/contrast ±10%, gamma 0,90–1,10; không vertical flip, MixUp, CutMix, elastic deformation. Backbone ConvNeXt-Tiny ImageNet-1K, sex embedding 16 chiều và direct regression; Smooth L1 beta tương đương 3 tháng, AdamW/cosine. Checkpoint chọn bởi validation MAE/early stopping, không bởi test.

### 9.2. Inference

TTA được giữ với 10 view: rotation `[-10,-5,0,5,10]` × flip `{false,true}`, lấy trung bình. Bias correction không giữ. P8 test metric là ensemble theo protocol P8 đã chạy; không tự gắn một metric TTA test chưa được tạo.

E1 được giữ thay E2 vì E2 phức tạp hơn nhưng point estimate chỉ tốt hơn 0,027730 tháng, CI bao gồm 0, không qua gate và không giảm sex gap. Đây là trade-off hiệu năng–độ phức tạp trong recipe hiện tại, không phải tuyên bố tối ưu mọi dataset.

## 10. Phân tích vai trò của giới tính

### 10.1. Bảng E0/E1/E2

| Mô hình | N | MAE | RMSE | Median AE | Accuracy ±6 | Accuracy ±12 | MAE nữ | MAE nam | Sex gap nữ−nam |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 image-only | 1.425 | 7,471711 | 9,927859 | 6,000000 | 53,2632% | 81,7544% | 7,768597 | 7,221297 | 0,547300 |
| E1 sex embedding | 1.425 | 6,184792 | 8,486471 | 5,000000 | 63,4386% | 86,6667% | 6,389115 | 6,012451 | 0,376664 |
| E2 dual-output | 1.425 | 6,157061 | 8,367776 | 4,500000 | 63,0175% | 87,3684% | 6,383915 | 5,965718 | 0,418197 |

### 10.2. Paired effect size và diễn giải

| So sánh | Delta MAE | CI P13 | Diễn giải |
|---|---:|---|---|
| E0−E1 overall | +1,286919 | [+1,012226;+1,569611] | E1 tốt hơn E0 rõ |
| E0−E1 nữ | +1,379481 | [+0,948691;+1,805653] | Cải thiện ở nữ |
| E0−E1 nam | +1,208845 | [+0,840546;+1,581351] | Cải thiện ở nam |
| E2−E1 overall | −0,027730 | [−0,179409;+0,124151] | Không đáng tin cậy |
| E2−E1 nữ | −0,005200 | [−0,220431;+0,209651] | Không đạt gate 0,20 |
| E2−E1 nam | −0,046734 | [−0,263745;+0,164865] | CI cắt 0 |

E1 cải thiện ở cả hai giới, chứng minh giá trị dự đoán bổ sung của nhãn giới trong dữ liệu/recipe, không chứng minh causal effect. E2 không đạt gate overall 0,10 hoặc female 0,20; sex gap tăng từ 0,376664 ở E1 lên 0,418197 ở E2.

Nguồn: [P13 table 1](../p13_reporting/outputs/P13_THESIS_REPORT/table_1_model_comparison.md), [P13 table 2](../p13_reporting/outputs/P13_THESIS_REPORT/table_2_paired_forest_data.md), [e0_vs_e1_seed42.json](../p11_sex_aware/analysis/e0_vs_e1_seed42.json), [e2_vs_e1_seed42.json](../p11_sex_aware/analysis/e2_vs_e1_seed42.json).

## 11. TTA và uncertainty

### 11.1. TTA/bias correction

P9-I trên 14.036 OOF: raw 6,317471; TTA 6,210446; `TTA−raw = −0,107025`, CI [−0,135977;−0,078220]. Raw correction 6,324285, delta +0,006815, CI chứa 0. TTA + correction 6,228665, kém TTA 0,018219, CI [+0,012482;+0,023813]. TTA giảm signed bias trung bình khoảng −0,3922 xuống −0,0930 nhưng không xóa bias mọi nhóm tuổi.

### 11.2. Disagreement

Disagreement là standard deviation của 10 prediction TTA. P12 recompute mean/std từ 10 view, với sai khác tối đa `5,68e-14` và `2,20e-14` tháng; 14.036 dòng và ID duy nhất. Primary là Spearman giữa `tta_std_months` và `abs(tta_prediction_months-target_months)`.

| Chỉ số | Kết quả | Diễn giải |
|---|---:|---|
| Spearman rho | 0,200431 | Association dương nhưng yếu |
| Bootstrap 95% CI | [0,184238;0,216411] | Lower CI >0 |
| AUROC lỗi >12 tháng | 0,625814 | Hạn chế |
| AUROC lỗi >18 tháng | 0,634108 | Hạn chế |

P-value nhỏ do cỡ mẫu lớn không biến effect yếu thành clinical utility. Không có calibration, referral cost hay threshold độc lập.

### 11.3. TTA disagreement/risk

| Quartile | N | TTA MAE | Lỗi >12 | Lỗi >18 |
|---|---:|---:|---:|---:|
| Q1 thấp | 3.509 | 4,758689 | 7,0390% | 2,0804% |
| Q2 | 3.509 | 5,942821 | 11,5417% | 3,3628% |
| Q3 | 3.509 | 6,516477 | 13,5366% | 4,4172% |
| Q4 cao | 3.509 | 7,623798 | 20,1482% | 6,8396% |

Q4 có MAE gấp khoảng 1,60 lần Q1, lỗi >12 gấp 2,86 lần và >18 gấp 3,29 lần. Risk–coverage mô tả trên cùng OOF: coverage 100/90/80/70/50% có MAE 6,210446/5,998069/5,813925/5,673199/5,350755 tháng; không được xem là threshold triển khai.

Nguồn: [P12_HANDOFF.md](../p12_uncertainty/P12_HANDOFF.md), [report.json](../p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/report.json), [disagreement_quartiles.csv](../p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/disagreement_quartiles.csv), [risk_coverage.csv](../p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/risk_coverage.csv).

## 12. Giới tính–nhóm tuổi

P12 dùng các bin 0–59, 60–119, 120–179 và 180–228 tháng; subgroup là exploratory.

| Nhóm | N | TTA MAE | Gain raw−TTA | rho | CI rho | AUROC >12 | Lỗi >12 |
|---|---:|---:|---:|---:|---|---:|---:|
| F:0–59 | 419 | 5,886469 | +0,030907 | −0,017429 | [−0,113609;+0,078368] | 0,507895 | 9,3079% |
| F:60–119 | 2.294 | 6,857886 | −0,052529 | 0,114602 | [0,074309;0,154532] | 0,578600 | 16,5214% |
| F:120–179 | 3.352 | 6,204844 | +0,186069 | 0,153638 | [0,120576;0,186117] | 0,597802 | 12,6492% |
| F:180–228 | 365 | 6,927408 | −0,029548 | 0,356450 | [0,260191;0,444896] | 0,616748 | 18,3562% |
| M:0–59 | 476 | 5,903079 | +0,292019 | 0,145514 | [0,055489;0,230117] | 0,629862 | 11,9748% |
| M:60–119 | 1.587 | 7,931348 | +0,123825 | 0,087041 | [0,038663;0,137164] | 0,573008 | 21,6131% |
| M:120–179 | 4.681 | 5,439070 | +0,130161 | 0,258930 | [0,232515;0,284807] | 0,671326 | 9,4424% |
| M:180–228 | 862 | 5,553432 | +0,060368 | 0,168495 | [0,103003;0,237267] | 0,633342 | 9,6288% |

Association mạnh hơn ở nam (0,238610) so với nữ (0,146217), nhưng vẫn yếu. F:180–228 có rho cao nhất nhưng chỉ n=365. Nhóm khó nhất về MAE là M:60–119 (7,931348), song rho chỉ 0,087041 và AUROC 0,573008; nhóm khó nhất không phải nhóm được proxy phân tầng tốt nhất. F:0–59 không có association đáng tin cậy. Age-standardized TTA MAE là 6,428277 ở nữ và 6,167778 ở nam, gap 0,260499, nhỏ hơn gap chưa chuẩn hóa 0,457011; phân bố tuổi giải thích một phần nhưng không toàn bộ.

Nguồn: [group_metrics.csv](../p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/group_metrics.csv), [P13 table 3](../p13_reporting/outputs/P13_THESIS_REPORT/table_3_p12_sex_age.md), [P12_HANDOFF.md](../p12_uncertainty/P12_HANDOFF.md).

## 13. Bốn đóng góp khoa học

| Đóng góp | Bằng chứng | Mức độ bằng chứng | Phạm vi không được vượt |
|---|---|---|---|
| **C1. Giá trị dự đoán của giới tính** | E1 tốt hơn E0 1,286919; CI [1,012226;1,569611]; cải thiện cả nữ/nam | Mạnh trong validation nội bộ | Một seed/validation; không causal; chưa external |
| **C2. Dual-output không hơn embedding** | E2−E1 −0,027730; CI cắt 0; không qua gate/không giảm gap | Negative result có giá trị | Không phủ định mọi multi-head/dataset; chưa OOF/multi-seed |
| **C3. TTA disagreement** | 14.036 OOF; rho 0,200431; AUROC 0,625814/0,634108; Q4 rủi ro hơn Q1 | Association supported, utility hạn chế | Không clinical uncertainty/calibration/rejection |
| **C4. Protocol tái lập** | Audit hash, paired bootstrap, OOF, gate, checkpoint/resume, P13 9 input/15 output, 0/15 drift | Mạnh ở cấp quy trình nội bộ | Không thay external validation; cần xử lý P7 audit |

Đóng góp hỗ trợ gồm TTA gain nhỏ nhưng nhất quán và bằng chứng âm cho masking, resolution, LDL, multi-scale, bias correction, Deeplasia augmentation moderate, P9-B0 và E2. Giá trị là đo được cái gì có ích và cái gì chưa đủ bằng chứng, không phải số lượng thí nghiệm.

Nguồn: [11_SCIENTIFIC_CONTRIBUTIONS_REPORT.md](../AI_Context/11_SCIENTIFIC_CONTRIBUTIONS_REPORT.md), [P13_HANDOFF.md](../p13_reporting/P13_HANDOFF.md), [report_manifest.json](../p13_reporting/outputs/P13_THESIS_REPORT/report_manifest.json).

## 14. Kết quả âm và ý nghĩa

Masking B1 có mask quality tốt nhưng không cải thiện MAE; D1 collapse cho thấy backbone/pretraining/recipe phải được kiểm tra bằng diagnostics; D2 và D3 chỉ có tín hiệu nhỏ, không ổn định; 768 không đáng chi phí; augmentation mạnh hơn không tự động tốt hơn; bias correction không giúp; EfficientNet-B0 là negative screening gần Deeplasia, chưa phải phủ định EfficientNet/Deeplasia; E2 không giải quyết sex gap; disagreement có association nhưng utility hạn chế.

Chuỗi kết quả âm hỗ trợ kết luận: tăng độ phức tạp không mặc nhiên đem lại cải thiện. Mọi negative claim đều recipe-scoped, không dùng ngôn ngữ tuyệt đối.

## 15. Tái lập và kiểm toán

P0 khóa manifest train/validation/test/development bằng SHA-256, kiểm tra ID/SHA. P1 ghi config/code/data hash, RNG, environment, checkpoint atomic và resume. P3–P6 có preflight, unit test, smoke/resume và paired report. P7 có 14.036/14.036 OOF ID duy nhất; P9-I ghi 10 views và raw control; P12 recompute mean/std; P11 giữ prediction CSV và paired JSON cùng ID.

P13 khóa 9 input, sinh 15 output gồm 4 bảng, 3 hình PNG/PDF và metadata; compile/unit test 5/5, visual QA đạt, hai build liên tiếp 0/15 output đổi hash. Manifest ghi `status=PASS`, `test_accessed=false` cho P13 và H1 supported, H2/H3 not supported, H4 supported association limited utility.

Audit P7 cần được người hướng dẫn xác nhận trước khi nộp bài báo: giải thích/bổ sung block metrics của fold 1 trong bản audit có provenance rõ ràng, không thay metric P7 đã khóa. Đây là threat về provenance, không phải lý do tự động phủ định OOF endpoint.

Nguồn: [P13_HANDOFF.md](../p13_reporting/P13_HANDOFF.md), [report_manifest.json](../p13_reporting/outputs/P13_THESIS_REPORT/report_manifest.json), [P7_5FOLD_AUDIT.txt](../P7_5FOLD_AUDIT.txt).

## 16. Hạn chế và threats to validity

| Hạn chế | Tác động tới phạm vi kết luận |
|---|---|
| P11 là screening một seed/validation; bootstrap không phản ánh đầy đủ variance retraining | C1 cần nhiều seed/OOF hoặc external |
| E2 dừng sau gate, không chạy seed/OOF/E3 | Chỉ nói E2 không đạt gate, không nói E2 kém chắc chắn trên mọi dữ liệu |
| P12 là phân tích hậu nghiệm trên OOF nội bộ | Association cần external validation/calibration |
| Test 200 đã đọc ở P8 | Không gọi P8 là holdout mới; không dùng test chọn P9–P13 |
| Chưa có external holdout/domain shift | Không tuyên bố generalization/clinical transportability |
| Disagreement chưa calibration/referral threshold | Không gọi là clinical uncertainty hay xác suất sai |
| Sex × age cỡ mẫu không đều, nhiều so sánh | Subgroup exploratory; không đặt threshold chung |
| Sex là nhãn M/F sẵn có | Không causal, không đại diện mọi định nghĩa sex/gender |
| Benchmark khác protocol | Không tuyên bố vượt SoTA chỉ từ điểm MAE |
| P7 audit có final FAIL hành chính | Giảm độ sạch provenance; cần kiểm tra trước công bố |
| P9-B0 dừng 1–3 epoch và chưa bit-exact recipe Deeplasia | Không kết luận EfficientNet/Deeplasia thất bại bản chất |

## 17. Hướng phát triển

1. Ưu tiên external holdout chưa bị tác động, với protocol/TTA/calibration/subgroup khóa trước.
2. Xác nhận E0–E1 bằng nhiều seed hoặc OOF để đo training variance.
3. Nghiên cứu calibration/selective prediction: error probability/conformal interval, referral cost, coverage và utility trên dữ liệu độc lập.
4. Đánh giá domain shift theo bệnh viện, thiết bị, chất lượng ảnh, tuổi và giới; báo worst-group.
5. Nếu thử masking/crop, làm paired visual/QC và ablation từng biến.
6. Chỉ tiếp tục Deeplasia/EfficientNet khi tái lập custom model, weight transfer, Albumentations, crop/mask và scheduler đầy đủ.
7. Làm systematic literature review trước khi tuyên bố novelty.
8. Nếu chưa có external data, tập trung luận văn quanh E1 + TTA + subgroup/uncertainty limitations thay vì mở E3 để săn điểm thuận lợi.

## 18. Câu chuyện nghiên cứu thống nhất

> Bài toán cần một hệ thống có thể kiểm toán chứ không chỉ một MAE đơn lẻ. P0 khóa split và loại trừ duplicate/leakage đã biết; P1–P7 xây baseline ConvNeXt-Tiny và OOF, đồng thời cho thấy masking, kiến trúc phức tạp và resolution cao không đem lại lợi ích ổn định. P8 cung cấp benchmark kỹ thuật nhưng đã chạm test, nên quyết định sau đó chuyển về validation/OOF. P11 cho thấy sex embedding cải thiện khoảng 1,29 tháng, lớn hơn nhiều thay đổi kiến trúc, còn dual-output không qua gate. P9-I/P12 cho thấy TTA giúp giảm MAE nhỏ nhưng nhất quán; disagreement làm giàu nhóm sai số cao nhưng utility hạn chế và không đồng nhất theo sex–age. P13 đóng gói bằng hash, manifest, bảng/hình và quy tắc diễn giải, từ đó chọn E1 làm pipeline gọn hợp lý và xác định external validation là bước còn thiếu.

Đây là câu chuyện phù hợp bảo vệ vì trả lời cả “cái gì có tác dụng” và “cái gì chưa đủ bằng chứng”. Điểm mạnh là thiết kế thực nghiệm, negative results, subgroup/uncertainty analysis và reproducibility; không phải state-of-the-art MAE.

## 19. Đánh giá mức độ đề tài

- **Đồ án cử nhân:** có đóng góp khoa học tốt, vượt mức train một model rồi báo MAE; có ablation, paired effect, CI, OOF, negative results, subgroup, uncertainty, gate và audit.
- **Bài báo sinh viên/workshop:** có khả năng nếu tập trung vào vai trò sex embedding, tính đủ dùng của E1 và giới hạn TTA disagreement.
- **Bài báo quốc tế mạnh:** chưa đủ vì thiếu external holdout, P11 một seed, P12 chưa calibration/external, sex nhị phân và chưa systematic review.
- **State of the art:** không tuyên bố. P8 4,730321 chưa đạt mốc 3,68–3,87 và không phải paired comparison với P7.

## 20. Kết luận

1. **Giới tính có giá trị dự đoán lớn:** E1 cải thiện E0 khoảng 1,29 tháng, CI paired không cắt 0, ở cả nữ và nam.
2. **Sex embedding là lựa chọn hợp lý về hiệu năng–độ phức tạp:** E2 chỉ tốt hơn 0,0277 tháng, không qua gate, không giảm sex gap và chưa được OOF xác nhận.
3. **TTA là cải tiến inference hỗ trợ:** giảm MAE OOF khoảng 0,1070 tháng; bias correction không giúp.
4. **Disagreement có giá trị phân tầng nhưng utility hạn chế:** rho 0,2004 và AUROC 0,6258–0,6341; không phải uncertainty lâm sàng và không đồng nhất sex–age.
5. **Đóng góp cốt lõi là quy trình thực nghiệm trung thực và tái lập:** leakage audit, paired comparison, bootstrap CI, gate, OOF, hash/manifest và negative results.

Pipeline giữ cho luận văn là **ConvNeXt-Tiny + A2 + 512 px + `preprocessing=none` + sex embedding + direct regression**, với **TTA** là inference improvement được giữ; bias correction và E2 bị loại. Đề tài có đóng góp tốt ở cấp đồ án cử nhân và có thể phát triển thành bài báo sinh viên/workshop. Chưa được tuyên bố state of the art, causal effect của giới tính, external generalization, clinical uncertainty hay an toàn lâm sàng trước khi có external holdout và xác nhận độc lập.

## Phụ lục. Artifact nguồn chính

| Nội dung | Artifact |
|---|---|
| Data audit | [`p0_audit/P0_HANDOFF.md`](../p0_audit/P0_HANDOFF.md), `p0_audit/outputs/` |
| Baseline/augmentation | [`p1_baseline/P1_HANDOFF.md`](../p1_baseline/P1_HANDOFF.md), [`P2_HANDOFF.md`](../p1_baseline/P2_HANDOFF.md) |
| Preprocessing/architecture/seed/resolution | [`P3_HANDOFF.md`](../p3_preprocessing/P3_HANDOFF.md), [`P4_HANDOFF.md`](../p4_architecture/P4_HANDOFF.md), [`P5_HANDOFF.md`](../p5_seed_confirmation/P5_HANDOFF.md), [`P6_PROTOCOL.md`](../p6_resolution/P6_PROTOCOL.md) |
| P7/P8 | [`P7_OOF_report.json`](../p7_final_v3/P7_OOF_report.json), [`P8 report`](../p8_test_ensemble/outputs/P8_test_ensemble_report.json) |
| P9 | [`P9_I_HANDOFF.md`](../p9_inference/P9_I_HANDOFF.md), [`P9_B0_HANDOFF.md`](../p9_single_model/P9_B0_HANDOFF.md) |
| P11 | [`P11_STAGE1_HANDOFF.md`](../p11_sex_aware/P11_STAGE1_HANDOFF.md), [`P11_STAGE2_HANDOFF.md`](../p11_sex_aware/P11_STAGE2_HANDOFF.md), [P13 tables](../p13_reporting/outputs/P13_THESIS_REPORT/) |
| P12 | [`P12_HANDOFF.md`](../p12_uncertainty/P12_HANDOFF.md), [`report.json`](../p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/report.json), CSV outputs |
| P13 | [`P13_HANDOFF.md`](../p13_reporting/P13_HANDOFF.md), [`P13_THESIS_DRAFT_VI.md`](../p13_reporting/P13_THESIS_DRAFT_VI.md), [`report_manifest.json`](../p13_reporting/outputs/P13_THESIS_REPORT/report_manifest.json) |

## Phụ lục B. Trạng thái giả thuyết

| Giả thuyết | Trạng thái | Bằng chứng |
|---|---|---|
| H1: sex cải thiện image-only | **SUPPORTED** | E0−E1 ΔMAE 1,2869; CI [1,0122;1,5696] |
| H2: dual-output tốt hơn embedding | **NOT SUPPORTED** | Improvement 0,0277 < gate 0,10; CI cắt 0 |
| H3: subgroup cải thiện dưới gate | **NOT SUPPORTED** | Female improvement 0,0052 < gate 0,20; gap không giảm |
| H4: disagreement liên hệ absolute error | **SUPPORTED, LIMITED UTILITY** | rho 0,2004; CI [0,1842;0,2164]; AUROC >12 0,6258 |

Nguồn khóa: [`table_4_hypothesis_summary.md`](../p13_reporting/outputs/P13_THESIS_REPORT/table_4_hypothesis_summary.md).
