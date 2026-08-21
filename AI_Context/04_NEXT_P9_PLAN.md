# Kế hoạch P9 – cải thiện có kiểm soát

## Lý do phải làm P9

P8 ensemble đạt 4,73032 tháng, chưa đạt Bram 2025 (3,68) hoặc Deeplasia 2024 (3,87). Vì vậy chưa đủ để báo cáo “vượt bài báo”. Deeplasia là trục tái lập chính; Bram là baseline đối chứng và nguồn ý tưởng phụ. Cần ưu tiên các thành phần có thể kiểm chứng: inference, preprocessing, training recipe và ensemble.

## Trạng thái triển khai P9-A (2026-08-19)

- Đã tạo `p9_preprocessing/deeplasia_preprocess.py`: cache development-only,
  provenance/QC riêng và không có đường dẫn test.
- Hai ablation đã khóa: `deeplasia_mask_v1` và
  `deeplasia_mask_histogram_v1`. P3 cũ vẫn bất biến.
- Đã tạo ba config A0/A1/A2 trong `p1_baseline/configs/p9_*.toml`.
  A0 là reference L1 100 epoch; A1/A2 chỉ khác preprocessing.
- Cache A1 development-only đã hoàn tất và audit PASS; A2 histogram mới chỉ có
  smoke cache, chưa được train đầy đủ vì A0/A1 không cho thấy recipe hiện tại có lợi.
- P10-B0/P10-B1 đã được chạy sau đó để xác nhận control và kiểm tra augmentation;
  xem kết quả bên dưới.

### Kết quả A1 (2026-08-20)

- Run `P9_A1_DEEPLASIA_MASK_L1_CONVNEXT_TINY_SEED42` đã early-stop ở epoch 30,
  best epoch 15.
- Validation MAE **6,307675**, RMSE **8,546865**, median AE 5,0 tháng; không có
  NaN/Inf/OOM. Cảnh báo chính là validation xấu đi trong khi train loss tiếp tục giảm.
- Paired so với P2 A2 trên cùng 1.425 ảnh: delta `+0,122884` tháng (A1 xấu hơn),
  bootstrap 95% CI `[-0,031592; +0,277028]`; CI chứa 0, chưa có bằng chứng A1 cải thiện.
- Quyết định: **không mở test và chưa xác nhận A1 bằng OOF**. Chạy A0 cùng recipe
  L1/100 epoch trước để tách ảnh hưởng của preprocessing khỏi loss/hyperparameter;
  chỉ tiếp tục A2 histogram nếu có lý do từ A0/A1 và gate được khóa trước.

### Kết quả A0 (2026-08-20)

- Run `P9_A0_REFERENCE_L1_CONVNEXT_TINY_SEED42` early-stop ở epoch 42,
  best epoch 27.
- Validation MAE **6,391447**, RMSE **8,674613**, median AE 5,0 tháng.
- So với P2 A2 cũ (MAE 6,184792), A0 kém hơn `+0,206656` tháng;
  paired bootstrap 95% CI `[+0,020061; +0,394780]`.
- So với A0, A1 tốt hơn `-0,083772` tháng nhưng CI `[-0,268072; +0,097423]`
  chứa 0. Kết luận: A1 có tín hiệu nhỏ nhưng chưa đủ bằng chứng; recipe L1/100
  với cấu hình hiện tại không thắng baseline P2.
- Quyết định: không mở test; không đưa A1/A0 vào OOF. Cần điều chỉnh recipe
  theo Bram/Deeplasia hoặc dừng nhánh này trước khi thử A2 histogram tốn tài nguyên.

### Kết quả P10-B0 control (2026-08-20)

- Giữ nguyên recipe P2: SmoothL1, LR `2e-4`, WD `0,05`, dropout `0,2`, light
  augmentation, ConvNeXt-Tiny 512 và preprocessing none.
- Early-stop epoch 20, best epoch 12; validation MAE **6,1847917** và RMSE
  **8,4865**.
- So với P2 cũ, toàn bộ 1.425 validation predictions giống hệt nhau (max abs
  diff = 0). Pipeline P9 được xác nhận không làm thay đổi baseline.
- Quyết định: khóa P10-B0 làm control; ứng viên kế tiếp chỉ thay augmentation
  sang Deeplasia mức vừa, không thay loss/LR/preprocessing đồng thời.

### Kết quả P10-B1 augmentation vừa (2026-08-20)

- Early-stop epoch 22, best epoch 14; validation MAE **6,1858772**, RMSE
  **8,4255**.
- So với control P10-B0, delta MAE **+0,001086 tháng**, CI bootstrap xấp xỉ
  `[-0,1404; +0,1406]`; RMSE giảm nhẹ nhưng MAE không cải thiện.
- Quyết định: loại B1 khỏi OOF/test. Không tăng độ mạnh augmentation một cách
  mù quáng; ưu tiên phân tích lỗi/ROI hoặc ablation preprocessing có giả thuyết.

## Nhánh ưu tiên cao nhất: Deeplasia-faithful reproduction và inference

Deeplasia là bài báo nền chính của đề tài: EfficientNet đa cấu hình, preprocessing/mask, TTA, bias correction và ensemble dị thể. Bram (ConvNeXt + sex, 5-fold RSNA, 3,68 tháng) được giữ làm mốc đối chứng. Không coi các con số test là phép so sánh công bằng nếu protocol khác nhau.

### P9-I – inference chi phí thấp (đã hoàn tất)

- Chạy TTA xoay/flip và bias correction trên OOF hiện có trước khi train backbone mới.
- Fit correction bằng train/OOF cross-fitting; tuyệt đối không dùng nhãn test.
- So sánh raw, TTA, bias correction và TTA + bias correction bằng MAE/RMSE,
  age-bin bias và bootstrap CI.

Kết quả đầy đủ và quyết định nằm trong `p9_inference/P9_I_HANDOFF.md`:

- TTA giảm MAE từ 6,317471 xuống **6,210446** tháng trên raw tái suy luận;
  paired delta `-0,107025`, CI `[-0,135977; -0,078220]`.
- Bias correction đơn độc không cải thiện MAE.
- TTA + bias correction kém TTA đơn độc.
- Giữ TTA làm inference candidate; không giữ bias correction trong pipeline
  chính hiện tại.

### P9-A – preprocessing có kiểm soát

- Xây pipeline riêng theo mã/miêu tả Deeplasia: mask, loại nền, normalization và
  các biến đổi hình học có thể tái lập; ghi rõ mọi sai khác so với mã chính thức.
- Không dùng mask B1 hiện tại một cách mặc định vì B1 đã không thắng; mỗi biến phải có ablation riêng.
- Audit 24–50 ảnh trực quan trước, sau đó chạy validation/OOF; giữ nguyên ID và hash.

### P9-B – Deeplasia single-model recipe

- Bắt đầu bằng EfficientNet-B0 512 + sex input và recipe gần mã Deeplasia;
  chưa ensemble ngay.
- Giữ P10-B0 làm control ConvNeXt; không thay nhiều biến cùng lúc.
- Khi báo cáo model đơn, giữ raw và TTA song song để tách đóng góp backbone
  khỏi đóng góp inference.
- Mỗi ứng viên chỉ được đánh giá trên OOF/validation; chọn một cấu hình cuối
  trước khi mở bất kỳ đánh giá test nào.

### P9-C – ensemble và báo cáo

- Nếu nhiều seed/fold có OOF dự đoán, thử equal-weight trước; chỉ thử trọng số học từ OOF nếu khóa công thức trước.
- Báo cáo MAE, RMSE, median AE, accuracy ±6/±12/±18, CI bootstrap, sex/age subgroup và calibration/bias theo tuổi.
- So sánh Bram/Deeplasia chỉ khi dataset, đơn vị tháng, test split và protocol
  tương thích; không coi 4,73 trên 200 là bằng chứng vượt.

## Tiêu chí dừng/giữ

- Giữ P9 chỉ khi OOF MAE cải thiện rõ so với P7 6,31669, không có subgroup collapse và cải thiện lặp lại ở ít nhất 3 seed/fold.
- Không dùng điểm test để quyết định “giữ hay bỏ”. Nếu P9 không cải thiện OOF, dừng và trình bày phân tích âm tính thay vì train vô hạn.
- Nếu vẫn muốn tuyên bố vượt Bram, cần một external hold-out mới chưa được xem; test RSNA 200 hiện tại đã dùng cho P8.

## Vận hành Colab

- Dữ liệu nguồn nén lưu một lần ở tài khoản A; tài khoản B/C chỉ mount shortcut đọc.
- Checkpoint append-only lưu ở Drive; local `/content` chỉ là cache tạm.
- Mỗi session phải chạy preflight, xem `run_state.json`, rồi resume đúng fold; không chạy đồng thời hai tài khoản trên cùng run.
- Giữ tối đa 2 checkpoint mới nhất + best model; log và manifest phải tải về local sau mỗi fold.

### P9-B0 – screening đã tạm dừng (2026-08-21)

- Đã kiểm tra EfficientNet-B0 512 theo recipe gần Deeplasia ở ba biến thể:
  stem 3 kênh/batch 12, stem 1 kênh/batch 12 và stem 1 kênh/batch 24.
- Best validation MAE lần lượt là 8,5293; 8,5823; 11,0428 tháng, đều kém
  P10-B0 control 6,184792.
- Quyết định: không mở test, không chạy OOF hoặc ensemble; dừng để review
  khoảng cách custom EfficientNet, mask/crop và Albumentations.
- Handoff: p9_single_model/P9_B0_HANDOFF.md.
