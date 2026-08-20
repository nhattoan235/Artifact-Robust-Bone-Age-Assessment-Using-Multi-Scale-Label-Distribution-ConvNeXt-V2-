# AI_Context changelog

## 2026-08-19 – Tạo bộ hồ sơ bàn giao AI_Context

- Tạo `00_START_HERE.md`, `01_STATUS_RESULTS.md`, `02_METHOD_HISTORY.md`, `03_DATA_PROTOCOL.md`, `04_NEXT_P9_PLAN.md`, `05_FILE_MAP.md`.
- Ghi rõ P7 OOF PASS (MAE 6,31669) và P8 test ensemble PASS kỹ thuật (MAE 4,73032).
- Ghi rõ kết quả hiện tại chưa vượt Bram 2025 (3,68) hoặc Deeplasia 2024 (3,87).
- Ghi rõ giới hạn quan trọng: test 200 đã được đọc ở P8; P9 phải chọn bằng OOF và cần external hold-out mới nếu muốn tuyên bố confirmatory.
- Ghi lại các nhánh bị loại: mask B1, ConvNeXtV2-FCMAE D1, multi-scale D2, LDL D3 chưa đủ bằng chứng.
- Ghi lại kế hoạch P9 ưu tiên tái lập recipe Bram có preprocessing/augmentation/hyperparameter search được khóa trước.
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

## 2026-08-20 – P10 B0 đang chạy: tái lập recipe P2

- Đã tạo config `p1_baseline/configs/p10_b0_p2_control.toml` với đúng recipe P2
  đã thắng trước đây: ConvNeXt-Tiny 512, sex embedding, light augmentation,
  SmoothL1 (beta 3 tháng), LR 2e-4, WD 0,05, dropout 0,2, 35 epoch.
- Preflight PASS và smoke interruption → resume PASS; không có đường dẫn test.
- Run đang chạy tại `p9_preprocessing/runs/P10_B0_P2_CONTROL_CONVNEXT_TINY_SEED42`.
- Mục tiêu là xác nhận pipeline P9 không làm thay đổi baseline trước khi thử
  augmentation Deeplasia mức vừa và preprocessing từng biến một.
