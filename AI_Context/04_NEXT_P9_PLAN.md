# Kế hoạch P9 – cải thiện có kiểm soát

## Lý do phải làm P9

P8 ensemble đạt 4,73032 tháng, thấp hơn Bram 2025 (3,68) và Deeplasia 2024 (3,87). Vì vậy chưa đủ để báo cáo “vượt bài báo”. Cần ưu tiên khác biệt có cơ sở: P7 chưa tái lập đầy đủ preprocessing/augmentation/training recipe của Bram.

## Trạng thái triển khai P9-A (2026-08-19)

- Đã tạo `p9_preprocessing/deeplasia_preprocess.py`: cache development-only,
  provenance/QC riêng và không có đường dẫn test.
- Hai ablation đã khóa: `deeplasia_mask_v1` và
  `deeplasia_mask_histogram_v1`. P3 cũ vẫn bất biến.
- Đã tạo ba config A0/A1/A2 trong `p1_baseline/configs/p9_*.toml`.
  A0 là reference L1 100 epoch; A1/A2 chỉ khác preprocessing.
- Chưa chạy cache, preflight, smoke hoặc train P9. Chỉ chạy A1/A2 sau visual
  QC của cache và preflight PASS.

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

## Nhánh ưu tiên cao nhất: Bram-faithful reproduction

Bram báo cáo ConvNeXt + sex, 5-fold RSNA, 100 epoch và ablation: không preprocessing/augmentation/ensemble 4,97; thêm preprocessing 4,46; thêm augmentation 3,74; ensemble 3,68. Bài: [DOI 10.1177/03635465251359618](https://doi.org/10.1177/03635465251359618).

### P9-A – preprocessing có kiểm soát

- Xây pipeline riêng theo mô tả Bram: loại nền/định hướng bàn tay, cân bằng histogram và chuẩn hóa hình học nếu có thể tái lập từ bài/mã nguồn.
- Không dùng mask B1 hiện tại một cách mặc định vì B1 đã không thắng; mỗi biến phải có ablation riêng.
- Audit 24–50 ảnh trực quan trước, sau đó chạy validation/OOF; giữ nguyên ID và hash.

### P9-B – training recipe

- ConvNeXt-Tiny + sex embedding, A2 làm baseline công bằng.
- Cho phép tối đa 100 epoch với early stopping patience 12–15; ghi rõ best epoch.
- Tìm nhỏ, có khóa trước: learning rate `{1e-4, 5e-5, 1e-5}`, weight decay `{0, 0.1, 0.01, 0.001, 0.0001}`, dropout `{0, 0.1, 0.2}` như mô tả Bram; không thử toàn bộ tổ hợp nếu không đủ tài nguyên.
- Mỗi ứng viên chỉ được đánh giá trên OOF/validation; chọn một cấu hình cuối trước khi mở test.

### P9-C – ensemble và báo cáo

- Nếu nhiều seed/fold có OOF dự đoán, thử equal-weight trước; chỉ thử trọng số học từ OOF nếu khóa công thức trước.
- Báo cáo MAE, RMSE, median AE, accuracy ±6/±12/±18, CI bootstrap, sex/age subgroup và calibration/bias theo tuổi.
- So sánh Bram/Deeplasia chỉ khi dataset, đơn vị tháng, test split và protocol tương thích; không coi 4,73 trên 200 là bằng chứng vượt.

## Tiêu chí dừng/giữ

- Giữ P9 chỉ khi OOF MAE cải thiện rõ so với P7 6,31669, không có subgroup collapse và cải thiện lặp lại ở ít nhất 3 seed/fold.
- Không dùng điểm test để quyết định “giữ hay bỏ”. Nếu P9 không cải thiện OOF, dừng và trình bày phân tích âm tính thay vì train vô hạn.
- Nếu vẫn muốn tuyên bố vượt Bram, cần một external hold-out mới chưa được xem; test RSNA 200 hiện tại đã dùng cho P8.

## Vận hành Colab

- Dữ liệu nguồn nén lưu một lần ở tài khoản A; tài khoản B/C chỉ mount shortcut đọc.
- Checkpoint append-only lưu ở Drive; local `/content` chỉ là cache tạm.
- Mỗi session phải chạy preflight, xem `run_state.json`, rồi resume đúng fold; không chạy đồng thời hai tài khoản trên cùng run.
- Giữ tối đa 2 checkpoint mới nhất + best model; log và manifest phải tải về local sau mỗi fold.
