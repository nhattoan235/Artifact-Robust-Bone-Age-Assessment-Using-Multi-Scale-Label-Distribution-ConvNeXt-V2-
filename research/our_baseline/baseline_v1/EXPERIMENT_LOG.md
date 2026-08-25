# Nhật ký thực nghiệm baseline

Mục đích của file này là ghi lại mọi hướng đã thử trong quá trình cải thiện mô hình dự đoán tuổi xương. Mỗi thử nghiệm phải có mã, giả thuyết, dữ liệu, cấu hình, kết quả, kết luận và giới hạn. Thử nghiệm không thành công vẫn được giữ lại.

## Danh mục

| Mã | Hướng thử nghiệm | Dữ liệu đánh giá | Trạng thái |
|---|---|---|---|
| EXP-001 | Blend baseline P7 với OOF P7 của bạn bạn | Official validation, 1.425 ảnh | Có tín hiệu cải thiện; cần xác nhận trên OOF/holdout độc lập |
| EXP-002 | Xác nhận blend 50/50 trên split holdout cố định | 391 ảnh holdout nội bộ | Blend 50/50 tiếp tục tốt nhất; chưa phải holdout độc lập tuyệt đối |
| EXP-003 | Tái tạo inference checkpoint P7 của bạn bạn | 10 ảnh validation để kiểm tra | Lần đầu sai do thiếu pad_square; kiểm tra lại bằng code gốc đã đạt |
| EXP-004 | Train lại bằng pipeline P7 của bạn bạn trên holdout mới | 1.260 ảnh holdout mới | Đã train; blend 25% tốt nhất, 50/50 chỉ cải thiện rất nhỏ |
| EXP-005 | Đánh giá P7 retrain và các blend trên test 200 | 200 ảnh test có nhãn | Friend P7 ensemble tốt nhất; chỉ đánh giá khám phá |
| EXP-006 | Chuẩn bị P7 control 5-fold trên local data | 14.036 development, OOF 5-fold | Đã tạo manifest/config; preflight PASS, chưa train |
| EXP-007 | D3 LDL regression-only 5-fold | 14.036 development, OOF 5-fold | Đã tạo config; chỉ chạy sau EXP-006 |
| EXP-008 | D3 LDL fused 5-fold | 14.036 development, OOF 5-fold | Đã tạo config; chỉ chạy sau EXP-007 |

## Quy ước đánh giá

- MAE được tính theo tháng tuổi xương; thấp hơn là tốt hơn.
- Tập test 200 ảnh không được dùng để chọn trọng số hoặc hyperparameter.
- Các kết quả trên test 200 ảnh trước đây chỉ được giữ làm thông tin tham khảo và không được coi là bằng chứng xác nhận cuối cùng nếu chưa có quy trình đánh giá độc lập.
- Mọi file đầu vào và file kết quả quan trọng cần được lưu cùng hash SHA-256 khi có thể.

## EXP-006/007/008 — Chuẩn bị lộ trình 5-fold, TTA và LDL

### Trạng thái hiện tại

Đã triển khai các script:

- `scripts/prepare_exp006_p7_5fold.py`: đọc duy nhất development manifest của bạn bạn, ánh xạ sang ảnh local, tạo 5 cặp train/validation manifest và config cho P7, D3 regression-only, D3 fused.
- `scripts/merge_exp006_oof.py`: kiểm tra ID trùng, gộp 5 file `val_predictions_best.csv`, tính metric pooled và theo fold.
- `scripts/evaluate_exp006_tta_oof.py`: đánh giá TTA xoay `[-10,-5,0,5,10]` độ, có/không flip trên OOF; không đọc test.
- `EXP006_COLAB_RUN.md`: quy trình chạy GPU, resume và lưu mirror lên Drive.

### Kiểm tra đã hoàn thành

- Development manifest: 14.036 dòng, ID duy nhất.
- 5 fold được phân tầng theo giới + age bins `[0,60,120,180,229]`.
- Mỗi fold có 2.807–2.808 validation rows và 11.228–11.229 train rows.
- Preflight P7 fold 1: **PASS**.
- Preflight D3 regression-only fold 1: **PASS**.
- Hash train/validation khớp đúng implementation `p1_baseline.data.manifest_hash`.
- Kiểm tra `test_path_absent`: **PASS**.
- Chưa train mô hình và chưa đọc nhãn test 200 trong EXP-006/007/008.

### Quy tắc cổng đánh giá

1. Chạy đủ EXP-006 P7 5 fold và gộp OOF.
2. Chạy TTA trên chính các checkpoint fold-held-out; chỉ giữ nếu OOF cải thiện.
3. Chỉ sau đó chạy EXP-007 D3 regression-only và EXP-008 D3 fused.
4. Chọn blend bằng OOF/holdout; test 200 chỉ dùng tham khảo sau khi khóa phương án.

## EXP-001 — Blend prediction của baseline với OOF của bạn bạn

### 1. Mục tiêu

Kiểm tra nhanh xem mô hình P7 baseline hiện tại và mô hình P7 của bạn bạn có sai số bổ sung cho nhau hay không. Nếu có, một blend cố định có thể cải thiện MAE mà chưa cần train lại mô hình.

### 2. Giả thuyết

Prediction trung bình giữa hai mô hình có thể giảm nhiễu và giảm sai số cực đoan. Các trọng số friend được thử là 0%, 25%, 50%, 75% và 100%.

### 3. Dữ liệu và nguồn đầu vào

- Prediction của baseline: `project/baseline_v1/outputs/official_gpu/official/results/outputs/official_fp16/official/val_predictions.csv`.
- Prediction của bạn bạn: năm file `P7_FINAL_V3_FOLD_1` đến `P7_FINAL_V3_FOLD_5`, mỗi file là `val_predictions_best.csv`, trong `project/p7_results/results`.
- Baseline có 1.425 mẫu validation.
- Năm file P7 có tổng cộng 14.036 OOF prediction, không có ID trùng nhau theo lần kiểm tra trước.
- Sau khi chuẩn hóa ID, phần giao nhau là 1.425/1.425 mẫu.
- Nhãn tuổi giữa hai nguồn khớp hoàn toàn; sai khác lớn nhất là 0 tháng.
- Nhãn test 200 ảnh không được đọc trong thí nghiệm này.

### 4. Công thức và quy trình

Với `w` là trọng số prediction của bạn bạn:

`prediction_blend = (1 - w) * prediction_baseline + w * prediction_friend`

Script tái lập:

```powershell
python project\baseline_v1\scripts\audit_friend_blend.py `
  --own-predictions project\baseline_v1\outputs\official_gpu\official\results\outputs\official_fp16\official\val_predictions.csv `
  --friend-root project\p7_results\results `
  --output-dir project\baseline_v1\outputs\exp001_friend_blend
```

Script tự động kiểm tra cột dữ liệu, ID trùng, số mẫu giao nhau, độ khớp nhãn, hash SHA-256 và xuất bảng metric.

### 5. Kết quả

| Mô hình | Trọng số friend | N | MAE | RMSE | Median AE | Trong ±6 tháng | Trong ±12 tháng | Trong ±18 tháng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline P7 | 0.00 | 1425 | 6.471143 | 8.878041 | 4.750000 | 0.597193 | 0.860351 | 0.953684 |
| Blend | 0.25 | 1425 | 6.263659 | 8.546222 | 4.593750 | 0.608421 | 0.865263 | 0.953684 |
| Blend | 0.50 | 1425 | **6.177155** | **8.396693** | **4.562500** | **0.618947** | **0.865965** | 0.952982 |
| Blend | 0.75 | 1425 | 6.226381 | 8.439151 | 4.656250 | 0.607018 | 0.864561 | 0.950877 |
| Friend P7 OOF | 1.00 | 1425 | 6.408334 | 8.670775 | 4.875000 | 0.589474 | 0.860351 | 0.947368 |

So với baseline, blend 50/50 cải thiện:

- MAE: giảm `0.293988` tháng, tương đương khoảng `4.54%`.
- RMSE: giảm `0.481348` tháng.
- Median absolute error: giảm `0.1875` tháng.
- Tỷ lệ đúng trong ±6 tháng: tăng khoảng `2.1754` điểm phần trăm.

### 6. File bằng chứng

- Bảng metric: `project/baseline_v1/outputs/exp001_friend_blend/blend_metrics.csv`.
- Prediction đã merge và prediction cho ba blend chính: `project/baseline_v1/outputs/exp001_friend_blend/blend_merged_predictions.csv`.
- Audit nguồn, hash, số mẫu và cảnh báo phương pháp: `project/baseline_v1/outputs/exp001_friend_blend/blend_audit.json`.
- Script tái lập: `project/baseline_v1/scripts/audit_friend_blend.py`.

### 7. Kết luận

Đây là tín hiệu cải thiện có ý nghĩa cho hướng ensemble/blend. Blend 50/50 được giữ làm ứng viên phát triển tiếp, nhưng chưa được tuyên bố là kết quả cuối cùng vì trọng số được khảo sát trên cùng official validation. Cần xác nhận bằng OOF độc lập hoặc một holdout phát triển khác trước khi dùng làm kết luận chính thức.

### 8. Giới hạn và rủi ro

- Hai mô hình được train bằng pipeline khác nhau; blend đang tận dụng tính bổ sung, chưa chứng minh mô hình mới tốt hơn về mặt kiến trúc.
- Official validation được dùng để so sánh trọng số, vì vậy kết quả 6.177155 là kết quả phát triển, không phải estimate không thiên lệch.
- Chưa đánh giá blend này trên tập test 200 ảnh trong EXP-001 để tránh dùng test cho lựa chọn phương án.
- Nếu sau này đánh giá test 200 ảnh, phải ghi thành một mục thực nghiệm riêng và nêu rõ đây là đánh giá tham khảo hay xác nhận cuối cùng.

### 9. Quyết định tiếp theo

1. Giữ blend 50/50 làm mốc ứng viên.
2. Tạo quy trình OOF cùng split cho baseline và mô hình bạn bạn để xác nhận blend mà không phụ thuộc một official validation split duy nhất.
3. Nếu tín hiệu vẫn giữ được, mới cân nhắc train lại hoặc distill/finetune một mô hình đơn để đạt mục tiêu dưới 3.5 MAE.

## EXP-002 — Xác nhận blend 50/50 trên holdout nội bộ

### 1. Mục tiêu

Kiểm tra xem ưu thế của blend 50/50 còn giữ được trên một phần dữ liệu không được dùng trong bảng kết quả xác nhận của EXP-002. Trọng số được cố định trước ở mức 50/50; không dùng tập test 200 ảnh.

### 2. Quy trình

- Nguồn là `blend_merged_predictions.csv` của EXP-001, gồm 1.425 prediction out-of-sample của baseline và OOF prediction của bạn bạn.
- Chia ID xác định bằng SHA-256 để kết quả tái lập trên mọi máy:
  - 1.034 ảnh development: giá trị hash chuẩn hóa `< 0.70`.
  - 391 ảnh holdout: các ID còn lại.
- Tính MAE, RMSE, median absolute error và accuracy trong ±6/±12/±18 tháng cho các trọng số 0%, 25%, 50%, 75%, 100%.
- Script chạy:

```powershell
python project\baseline_v1\scripts\confirm_blend_holdout.py `
  --merged-predictions project\baseline_v1\outputs\exp001_friend_blend\blend_merged_predictions.csv `
  --output-dir project\baseline_v1\outputs\exp002_blend_holdout
```

### 3. Kết quả

| Split | Phương án | N | MAE | RMSE | Median AE | Trong ±6 tháng |
|---|---|---:|---:|---:|---:|---:|
| Development | Baseline | 1034 | 6.622737 | 9.189129 | 4.875000 | 0.581238 |
| Development | Blend 50/50 | 1034 | 6.334318 | 8.666189 | 4.687500 | 0.605416 |
| Holdout nội bộ | Baseline | 391 | 6.070253 | 7.997264 | 4.500000 | 0.639386 |
| Holdout nội bộ | Blend 50/50 | 391 | **5.761539** | **7.638326** | **4.312500** | **0.654731** |

Trên 391 ảnh holdout nội bộ, blend 50/50:

- Giảm MAE `0.308714` tháng so với baseline.
- Tăng accuracy trong ±6 tháng khoảng `1.5345` điểm phần trăm.
- Là trọng số tốt nhất trong năm trọng số được kiểm tra trên chính holdout này.

### 4. Kết luận

Tín hiệu blend không chỉ xuất hiện trên toàn bộ validation; nó vẫn giữ được trên phần holdout nội bộ. Vì vậy blend 50/50 được giữ làm ứng viên mạnh cho bước tiếp theo.

### 5. Giới hạn cần ghi rõ

Đây chưa phải xác nhận độc lập tuyệt đối: toàn bộ official validation đã được nhìn thấy trong EXP-001 trước khi tạo split này. Do đó, kết quả này là **internal split-confirmation**, không phải estimate hoàn toàn chưa từng được xem.

Để có xác nhận nghiêm ngặt, cần một trong hai phương án:

1. Train lại baseline trên một split mới, không chứa holdout, rồi dự đoán holdout đó và blend với friend OOF.
2. Dùng một holdout chưa từng được dùng trong bất kỳ bước chọn mô hình nào; nếu dùng test 200 ảnh thì phải có prediction của cả hai mô hình và ghi riêng như một đánh giá độc lập.

### 6. File bằng chứng

- Metric: `project/baseline_v1/outputs/exp002_blend_holdout/holdout_blend_metrics.csv`.
- Phân chia ID: `project/baseline_v1/outputs/exp002_blend_holdout/holdout_assignment.csv`.
- Audit và SHA-256: `project/baseline_v1/outputs/exp002_blend_holdout/holdout_blend_audit.json`.
- Script: `project/baseline_v1/scripts/confirm_blend_holdout.py`.

## EXP-003 — Kiểm tra tái tạo inference checkpoint P7

### 1. Mục tiêu

Tái tạo prediction trực tiếp từ năm `best_model.pt` của bạn bạn để có thể tạo prediction trên tập test 200 ảnh và xác nhận blend bằng một holdout chưa từng dùng.

### 2. Phương pháp đã thử

- Dựng lại ConvNeXt-Tiny bằng `torchvision`.
- Tái tạo sex embedding 16 chiều và regression head 784 → 256 → 1 theo `config_resolved.yaml` và state dict.
- Resize ảnh về 512×512, chuyển ảnh grayscale thành RGB, chuẩn hóa ImageNet.
- Khôi phục tuổi theo `prediction = raw * target_std + target_mean`.
- Chạy trên 10 ảnh đầu của `P7_FINAL_V3_FOLD_1/val_predictions_best.csv`.

### 3. Kết quả và bằng chứng

Prediction tái tạo của fold 1 không khớp prediction đã lưu:

- Sai khác tuyệt đối lớn nhất: `7.09657` tháng.
- MAE của sai khác giữa hai dãy prediction: `2.67998` tháng.
- Ví dụ ID `1378`: lưu sẵn `9.3671875`, tái tạo `7.13097`.
- Ví dụ ID `1393`: lưu sẵn `124.25`, tái tạo `131.34657`.

File kiểm tra: `project/baseline_v1/outputs/exp001_friend_holdout/friend_reproduction_check.csv`.
Script: `project/baseline_v1/scripts/infer_friend_p7.py`.

### 4. Kết luận lần thử đầu

Lần tái tạo đầu **không hợp lệ** vì script kiểm tra đã resize trực tiếp mà bỏ qua `pad_square` của code gốc. Sai khác tối đa khi đó là `7.09657` tháng; kết quả này bị loại và không dùng để đánh giá.

### 5. Kiểm tra lại bằng code gốc

Đã dùng trực tiếp `p1_baseline/data.py` và `p1_baseline/model.py` của repo, trong đó có `pad_square`, resize bicubic có antialias và chuẩn hóa đúng pipeline. Trên 10 ảnh của fold 1:

- Sai khác tuyệt đối lớn nhất so với CSV đã lưu: `0.054260` tháng.
- Sai khác này phù hợp với khác biệt precision/inference; pipeline đã được coi là tái lập đạt để tiếp tục.
- File bằng chứng: `project/baseline_v1/outputs/exp004_friend_holdout/inference_verify_fold1.csv`.
- Script: `project/baseline_v1/scripts/verify_friend_p7_inference.py`.

Không dùng kết quả kiểm tra 10 ảnh này để chọn mô hình; nó chỉ xác minh code inference.

## EXP-004 — Train lại bằng pipeline P7 của bạn bạn trên holdout mới

### 1. Mục tiêu

Tạo một baseline mới bằng đúng pipeline P7 của bạn bạn, nhưng huấn luyện trên tập train mới và đánh giá trên holdout mới không trùng official validation cũ. Prediction holdout sau đó sẽ được kết hợp với prediction OOF của bạn bạn bằng trọng số 50/50 đã đăng ký trước.

### 2. Source code được dùng

- Repo: `project/friend_repo`.
- Commit clone: `70658f6a0901741dfec40270bcac4459eab4da9f`.
- Trainer: `p1_baseline/trainer.py`.
- Model: `p1_baseline/model.py` — ConvNeXt-Tiny ImageNet-1K, sex embedding 16, head 784 → 256 → 1.
- Data/preprocessing: `p1_baseline/data.py` — pad vuông, resize 512, lặp grayscale thành 3 kênh, ImageNet normalization, light augmentation.
- Thay đổi tương thích tối thiểu: `timm` được import tùy chọn; chỉ kiến trúc ConvNeXt-V2 cần `timm`, ConvNeXt-Tiny P7 không cần. Thay đổi này không đổi model P7.

### 3. Split và cấu hình

- Nguồn 14.036 ảnh development của repo bạn bạn.
- Chỉ lấy 10% từ 12.611 ảnh original train để làm holdout bằng quy tắc SHA-256 cố định với seed 42.
- 1.260 ảnh holdout; 12.776 ảnh train mới = phần train còn lại + toàn bộ 1.425 official validation.
- Train/holdout không có ID giao nhau.
- Nhãn test không được đọc.
- Cấu hình chính: `friend_p7_fresh_holdout.toml`, ConvNeXt-Tiny, 512, batch 12, accumulation 3, 35 epoch, AdamW, LR 2e-4, weight decay .05, Smooth L1 beta 3 tháng, AMP FP16.

### 4. Kiểm tra đã hoàn thành

- Manifest hash train: `e6a49b89a52dd029eb341555a878e50c8f17f267fc6c43f40a409db44965c8a2`.
- Manifest hash holdout: `b11701e8ef111b673dbb8fcb3c892bc4c00e9cfcca713fe28332564cd4a7cd40`.
- Preflight: PASS, đúng 12.776/1.260 mẫu, forward shape đúng, không chứa đường dẫn test.
- Smoke trainer: PASS trên CPU với 8 train, 4 holdout, 1 epoch; checkpoint, log và prediction được tạo.
- Kết quả smoke không dùng để đánh giá khoa học.

Trong lúc chạy smoke, lần đầu trainer không tự tạo được thư mục output tuyệt đối do quyền ghi của process khi chạy từ thư mục clone. Đây là lỗi môi trường, không phải lỗi model; tạo sẵn thư mục đích và chạy process từ workspace root thì smoke hoàn tất PASS. Sự cố này và cách xử lý được giữ lại trong log terminal của phiên làm việc.

### 5. File bằng chứng hiện có

- Audit split/config: `project/baseline_v1/outputs/exp004_friend_holdout/fresh_holdout_audit.json`.
- Train manifest: `project/baseline_v1/outputs/exp004_friend_holdout/train_manifest.csv`.
- Holdout manifest: `project/baseline_v1/outputs/exp004_friend_holdout/holdout_manifest.csv`.
- Config full: `project/baseline_v1/outputs/exp004_friend_holdout/friend_p7_fresh_holdout.toml`.
- Config smoke: `project/baseline_v1/outputs/exp004_friend_holdout/friend_p7_smoke.toml`.
- Smoke artifacts: `project/baseline_v1/outputs/exp004_friend_holdout/smoke_runs/EXP004_FRIEND_P7_SMOKE`.
- Script tạo split: `project/baseline_v1/scripts/prepare_friend_holdout.py`.

### 6. Kết quả train đầy đủ

- Run kết thúc early stopping ở epoch 27, global step 9585.
- Best epoch: 19.
- Best holdout MAE của mô hình retrain: `6.213796` tháng.
- Holdout RMSE: `8.450559` tháng.
- Median absolute error: `4.750000` tháng.
- Accuracy trong ±6/±12/±18 tháng: `0.600000 / 0.873016 / 0.954762`.
- Không có ảnh test hoặc nhãn test được dùng.

### 7. Xác nhận blend trên holdout mới

Prediction của mô hình retrain được merge với OOF prediction P7 của bạn bạn theo 1.260 ID holdout. Nhãn giữa hai nguồn khớp hoàn toàn.

| Phương án | Trọng số friend | N | MAE | RMSE | Median AE | Trong ±6 tháng |
|---|---:|---:|---:|---:|---:|---:|
| Mô hình retrain | 0.00 | 1260 | 6.213796 | 8.450559 | 4.750000 | 0.600000 |
| Blend | 0.25 | 1260 | **6.179959** | 8.364744 | 4.734375 | 0.597619 |
| Blend | 0.50 | 1260 | 6.204244 | **8.366013** | 4.812500 | **0.606349** |
| Blend | 0.75 | 1260 | 6.299853 | 8.454326 | 4.976562 | 0.592063 |
| Friend P7 OOF | 1.00 | 1260 | 6.460169 | 8.627011 | 5.125000 | 0.576984 |

Kết luận:

- Blend 50/50 vẫn giảm MAE so với mô hình retrain, nhưng chỉ `0.009552` tháng; mức cải thiện rất nhỏ.
- Blend 25/75 (25% friend, 75% mô hình retrain) là phương án tốt nhất theo MAE, giảm `0.033837` tháng.
- Vì vậy không thể xác nhận 50/50 là trọng số tối ưu cố định trên holdout mới. Ta giữ 50/50 làm mốc đã đăng ký, nhưng ứng viên hiện tại là 25% friend + 75% mô hình retrain.
- Đây là holdout phát triển được tạo sau các thực nghiệm trước, nên vẫn cần một đánh giá cuối cùng trên protocol chưa dùng để chọn trọng số trước khi báo cáo kết quả chính thức.

### 8. File bằng chứng sau train

- Checkpoint mirror: `project/baseline_v1/outputs/exp004_friend_holdout/checkpoint_mirror/checkpoint_mirror/EXP004_FRIEND_P7_FRESH_HOLDOUT_SEED42`.
- Prediction best: `.../val_predictions_best.csv`.
- Log/metrics/checkpoint: cùng thư mục trên.
- Bảng blend: `project/baseline_v1/outputs/exp004_friend_holdout/blend_holdout/blend_metrics.csv`.
- Prediction đã merge: `project/baseline_v1/outputs/exp004_friend_holdout/blend_holdout/blend_merged_predictions.csv`.
- Audit nguồn và hash: `project/baseline_v1/outputs/exp004_friend_holdout/blend_holdout/blend_audit.json`.

### 9. Trạng thái

EXP-004 hoàn tất. Hướng P7 friend code đã được tái lập và train thành công trên GPU Colab. Hướng blend 50/50 được đánh giá là có cải thiện nhưng không đủ mạnh; không được dùng như bằng chứng rằng 50/50 là tối ưu.

## EXP-005 — Đánh giá test 200 cho P7 retrain và blend

### 1. Mục tiêu

Đánh giá các phương án đã thử trên cùng 200 ảnh test có nhãn:

1. P7 retrain của EXP-004.
2. Friend P7 five-fold ensemble.
3. Blend 50/50.
4. Blend 75% P7 retrain + 25% friend P7.

### 2. Quy tắc sử dụng test

Người dùng yêu cầu chạy đánh giá test nên nhãn test được đọc trong EXP-005. Kết quả này là **exploratory**; không được dùng để chọn lại trọng số, hyperparameter hoặc tuyên bố kết quả xác nhận cuối cùng. Các quyết định phát triển vẫn dựa trên validation/holdout.

### 3. Kết quả

| Phương án | MAE | RMSE | Median AE | Trong ±6 tháng | Trong ±12 tháng | Trong ±18 tháng |
|---|---:|---:|---:|---:|---:|---:|
| P7 retrain EXP-004 | 5.214580 | 6.828400 | 3.635758 | 0.655 | 0.920 | 0.990 |
| Friend P7 ensemble | **4.730865** | **6.029187** | 4.134196 | **0.700** | **0.940** | **1.000** |
| Blend 50/50 | 4.933367 | 6.324887 | 4.134396 | 0.680 | 0.935 | 0.990 |
| Blend 75% retrain + 25% friend | 5.065745 | 6.553198 | 3.844749 | 0.670 | 0.930 | 0.990 |

So với P7 retrain, friend P7 ensemble giảm MAE `0.483715` tháng. Trên test, cả hai blend đều kém friend P7 ensemble:

- Blend 50/50 kém `0.202502` tháng.
- Blend 75/25 kém `0.334880` tháng.

Kết quả P8 đã có trước đó là `4.730321` tháng; sai khác nhỏ so với `4.730865` ở EXP-005 do P8 report dùng nhãn test được làm tròn 2 chữ số, còn EXP-005 dùng đầy đủ nhãn trong `boneage-test-dataset-with-gt.csv`. Kết luận thứ hạng không thay đổi.

### 4. Kết luận

Trên test 200, friend P7 five-fold ensemble là phương án tốt nhất trong nhóm được đánh giá. Blend 50/50 có lợi trên holdout phát triển nhưng không chuyển thành cải thiện trên test; đây là lý do không được chọn trọng số dựa trên test. P7 retrain mới chưa vượt được friend ensemble và chưa đạt mục tiêu MAE dưới 3.5.

### 5. File bằng chứng

- Report: `project/baseline_v1/outputs/exp005_test200_friend_methods/test200_method_report.json`.
- Toàn bộ prediction: `project/baseline_v1/outputs/exp005_test200_friend_methods/test200_method_predictions.csv`.
- Script: `project/baseline_v1/scripts/evaluate_test200_friend_methods.py`.
