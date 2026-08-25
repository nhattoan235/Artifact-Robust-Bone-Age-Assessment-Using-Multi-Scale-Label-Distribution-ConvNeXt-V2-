# Tổng hợp toàn bộ quá trình nghiên cứu dự đoán tuổi xương

Ngày cập nhật: 2026-08-23

## 1. Mục tiêu nghiên cứu

Mục tiêu là xây dựng và cải thiện mô hình dự đoán tuổi xương theo tháng từ ảnh
X-quang bàn tay của bộ RSNA Pediatric Bone Age.

Các mốc mục tiêu:

1. Tái lập được pipeline baseline và quy trình đánh giá đáng tin cậy.
2. Vượt kết quả tốt nhất hiện tại của bạn cùng nhóm, khoảng MAE 4.73 tháng trên
   200 ảnh test tham khảo.
3. Hướng tới MAE dưới 3.5 tháng.

Mục tiêu dưới 3.5 tháng cần giảm khoảng 1.23 tháng, tương đương hơn 26% so với
MAE 4.73. Vì vậy, một thay đổi nhỏ như tăng epoch hoặc đổi một augmentation sẽ
không đủ; cần kết hợp cải thiện preprocessing, học biểu diễn, ensemble và đánh
giá OOF nghiêm ngặt.

## 2. Dữ liệu và quy ước đánh giá

### 2.1. Các phần dữ liệu

| Phần dữ liệu | Số ảnh | Vai trò |
|---|---:|---|
| Official train | 12.611 | Huấn luyện ban đầu |
| Official validation | 1.425 | Validation chính thức |
| Development pool | 14.036 | Gộp train + validation để chạy 5-fold OOF |
| RSNA test tham khảo | 200 | Chỉ đo MAE/RMSE tham khảo |

### 2.2. Quy tắc sử dụng test 200

Tập test 200 được người dùng xác nhận chỉ dùng để thử mô hình và xem MAE là bao
nhiêu. Tập này không được dùng để:

- chọn hyperparameter;
- chọn checkpoint;
- chọn recipe hoặc preprocessing;
- chọn trọng số blend;
- quyết định giữ hay loại một hướng nghiên cứu.

Các quyết định phát triển phải dựa trên official validation, fresh holdout hoặc
5-fold OOF. Test 200 chỉ được ghi như một phép đo tham khảo sau khi phương án đã
được khóa.

### 2.3. Kiểm soát leakage

- Chuẩn hóa ID ảnh và kiểm tra ID duy nhất.
- Kiểm tra đường dẫn ảnh tồn tại.
- Lưu manifest, split hash, config hash và SHA-256 của file nguồn.
- Không đọc nhãn test trong các pipeline train/OOF.
- Với blend, kiểm tra ID giao nhau và giá trị target khớp hoàn toàn.

## 3. Các ý tưởng tham khảo từ bài báo và repo liên quan

Các bài báo người dùng đã cung cấp và các tài liệu gần đây được dùng để hình
thành các nhóm hướng sau:

- Transfer learning với backbone CNN hiện đại đã pretrained trên ImageNet.
- Kết hợp thông tin sex với đặc trưng ảnh thay vì chỉ dùng ảnh.
- Giữ tỷ lệ ảnh, crop/pad và loại bớt nền không liên quan.
- Học đặc trưng đa scale hoặc global-local khi có bằng chứng phù hợp.
- Label Distribution Learning để biểu diễn sự không chắc chắn giữa các tuổi lân
  cận thay vì coi tuổi là một nhãn rời rạc hoàn toàn chính xác.
- Robust loss, augmentation nhẹ và regularization để giảm overfit.
- Ensemble nhiều fold hoặc nhiều model có sai số bổ sung.
- Test-time augmentation và uncertainty để khai thác nhiều dự đoán ở inference.

Các mốc văn liệu được dùng làm tham khảo gồm các bài người dùng đã gửi:

- [Scientific Reports 2022](https://www.nature.com/articles/s41598-022-10292-y).
- [ScienceDirect 2022](https://www.sciencedirect.com/science/article/abs/pii/S1746809422005055).
- [Springer 2023 – s11517-023-03013-8](https://link.springer.com/article/10.1007/s11517-023-03013-8).
- [Springer 2023 – s00247-023-05789-1](https://link.springer.com/article/10.1007/s00247-023-05789-1).

Ngoài ra, các tài liệu trong repo của bạn cùng nhóm tổng hợp Bram 2025,
Deeplasia 2024, MMANet 2023, BoNet+ 2026 và các hướng uncertainty-aware.

Các MAE từ bài báo chỉ là mốc tham khảo vì khác bộ dữ liệu, preprocessing, split,
test protocol và cách tính. Không dùng chúng để tuyên bố mô hình hiện tại đã
vượt bài báo.

## 4. Baseline ban đầu của chúng ta

### 4.1. Kiến trúc

```text
Ảnh grayscale
→ lặp thành 3 kênh RGB
→ ConvNeXt-Tiny pretrained ImageNet
→ global average pooling
→ sex embedding
→ LayerNorm / Linear / GELU / Dropout
→ Linear 1
→ bone age theo tháng
```

Mô hình ban đầu dùng ConvNeXt-Tiny, sex embedding 32 chiều và direct regression.

### 4.2. Loss và tối ưu hóa ban đầu

| Thành phần | Cấu hình ban đầu |
|---|---|
| Loss | Smooth L1 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| Warm-up | 3 epoch |
| Scheduler | Cosine decay |
| Resolution | 512×512 |
| Epoch tối đa | 100 |
| Early stopping | Patience 15 |
| Gradient accumulation | ban đầu thấp, batch nhỏ |

### 4.3. Recipe đã chạy

#### P7 reference của baseline v1

- Resize trực tiếp về 512×512.
- Không giữ tỷ lệ bằng `pad_square`.
- ImageNet normalization.
- ConvNeXt-Tiny + sex embedding.
- Direct Smooth L1 regression.

Official validation:

| Metric | Kết quả |
|---|---:|
| MAE | **6.471143** |
| RMSE | 8.878041 |
| Median AE | 4.750000 |
| Trong ±6 tháng | 0.597193 |
| Trong ±12 tháng | 0.860351 |
| Trong ±18 tháng | 0.953684 |

File bằng chứng: `outputs/official_gpu/official/results/outputs/official_fp16/official/report.json`.

#### A2 light flip

A2 thêm horizontal flip nhẹ trong lúc train.

Official validation:

| Metric | Kết quả |
|---|---:|
| MAE | **6.706371** |
| RMSE | 9.025617 |
| Median AE | 5.062500 |
| Trong ±6 tháng | 0.581754 |

Kết luận: A2 light flip trong implementation baseline cũ không cải thiện; MAE
xấu hơn P7 reference. Không nên kết luận horizontal flip luôn có hại, vì pipeline
P7 của bạn cùng nhóm dùng preprocessing và augmentation khác.

#### Test 200 của baseline P7 cũ

Một run P7 reference cũ trên 200 ảnh có nhãn đạt khoảng MAE 5.113332. Đây là phép
đo tham khảo, không dùng để chọn model.

File: `outputs/test200_p7_reference_best/test200_report.json`.

## 5. Pipeline P7 của bạn cùng nhóm

### 5.1. Preprocessing chính xác

```text
Ảnh grayscale
→ pad_square giữ nguyên tỷ lệ
→ resize bicubic 512×512, antialias
→ lặp thành 3 kênh
→ ImageNet normalization
```

Đây là khác biệt quan trọng so với baseline cũ của chúng ta, vì resize trực tiếp
có thể làm biến dạng hình thái bàn tay.

### 5.2. Kiến trúc và huấn luyện

| Thành phần | P7 của bạn cùng nhóm |
|---|---|
| Backbone | ConvNeXt-Tiny ImageNet pretrained |
| Sex embedding | 16 chiều |
| Head | 784 + sex → 256 → 1 |
| Target | Chuẩn hóa mean/std |
| Loss | Smooth L1, beta 3 tháng |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.05 |
| Batch size | 12 |
| Gradient accumulation | 3, batch hiệu dụng khoảng 36 |
| Epoch | 35 |
| Patience | 8 |
| Augmentation | light affine, flip, brightness/contrast/gamma nhẹ |
| Precision | AMP FP16/auto |

Mặc dù config P7 có các trường liên quan Label Distribution Learning, checkpoint
P7 thực tế chỉ có `architecture = convnext_tiny` và không có
`distribution_head`. Vì vậy P7 thực tế vẫn là direct regression.

### 5.3. Kết quả P7 five-fold ensemble

OOF của bạn cùng nhóm:

| Fold | MAE |
|---:|---:|
| 1 | 6.298710 |
| 2 | 6.196835 |
| 3 | 6.411474 |
| 4 | 6.348768 |
| 5 | 6.327675 |
| Pooled OOF | **6.316691** |

Kết quả test 200 của ensemble P7:

| Metric | Kết quả |
|---|---:|
| MAE | **4.730865** |
| RMSE | 6.029187 |
| Median AE | 4.134196 |
| Trong ±6 tháng | 0.700 |
| Trong ±12 tháng | 0.940 |
| Trong ±18 tháng | 1.000 |

Kết quả P8 report là 4.730321 do sử dụng nhãn test làm tròn 2 chữ số; khi tính
với nhãn đầy đủ, kết quả là 4.730865. Thứ hạng không thay đổi.

## 6. Các thử nghiệm đã thực hiện

### EXP-001 — Blend baseline với OOF của bạn cùng nhóm

Mục tiêu: kiểm tra các prediction có sai số bổ sung hay không.

Công thức:

```text
blend = (1 - w) × own_prediction + w × friend_prediction
```

Kết quả trên 1.425 official validation rows:

| Phương án | MAE |
|---|---:|
| Baseline own | 6.471143 |
| Friend OOF trên cùng rows | 6.408334 |
| Blend 25% friend | 6.263659 |
| Blend 50/50 | **6.177155** |
| Blend 75% friend | 6.226381 |

Blend 50/50 giảm 0.293988 tháng so với baseline. Tuy nhiên, trọng số được khảo
sát trên official validation nên đây chỉ là kết quả phát triển.

Files:

- `outputs/exp001_friend_blend/blend_metrics.csv`.
- `outputs/exp001_friend_blend/blend_merged_predictions.csv`.
- `outputs/exp001_friend_blend/blend_audit.json`.

### EXP-002 — Holdout nội bộ cho blend 50/50

Trọng số 50/50 được giữ cố định và split lại official validation thành:

- 1.034 development rows.
- 391 internal holdout rows.

Kết quả trên 391 rows:

| Phương án | MAE |
|---|---:|
| Baseline | 6.070253 |
| Blend 50/50 | **5.761539** |

Tín hiệu blend vẫn còn, nhưng đây chưa phải holdout độc lập hoàn toàn vì toàn bộ
official validation đã được nhìn thấy trong EXP-001 trước đó.

### EXP-003 — Tái tạo inference checkpoint P7

Lần đầu tái tạo sai vì bỏ qua `pad_square`:

- Sai khác lớn nhất: 7.09657 tháng.
- Sai khác trung bình tuyệt đối: 2.67998 tháng.

Sau khi dùng đúng `p1_baseline/data.py` và `p1_baseline/model.py` của bạn cùng
nhóm:

- Sai khác lớn nhất trên 10 ảnh: 0.054260 tháng.
- Pipeline inference được xác nhận đủ đúng để tiếp tục.

Kết luận kỹ thuật: `pad_square`, bicubic antialias và target normalization là các
chi tiết phải giữ nguyên khi tái lập.

### EXP-004 — Retrain P7 trên fresh holdout

Split:

- 10% original train làm holdout mới: 1.260 ảnh.
- Train mới: 12.776 ảnh.
- Không có ID giao nhau.
- Không đọc nhãn test.

Cấu hình dùng đúng pipeline P7 của bạn cùng nhóm.

Kết quả mô hình retrain:

| Metric | Kết quả |
|---|---:|
| Best epoch | 19 |
| Early stopped | Epoch 27 |
| Holdout MAE | **6.213796** |
| RMSE | 8.450559 |
| Median AE | 4.750000 |
| Trong ±6/±12/±18 | 0.600000 / 0.873016 / 0.954762 |

Blend trên fresh holdout:

| Phương án | MAE |
|---|---:|
| Retrain | 6.213796 |
| 25% friend + 75% retrain | **6.179959** |
| 50/50 | 6.204244 |
| 75% friend + 25% retrain | 6.299853 |
| Friend only | 6.460169 |

Kết luận: blend 50/50 không phải trọng số tối ưu cố định. Trên holdout mới,
25% friend + 75% retrain tốt nhất nhưng mức cải thiện nhỏ.

Files:

- `outputs/exp004_friend_holdout/fresh_holdout_audit.json`.
- `outputs/exp004_friend_holdout/blend_holdout/blend_metrics.csv`.
- `outputs/exp004_friend_holdout/checkpoint_mirror/.../val_predictions_best.csv`.

### EXP-005 — Thử các phương án trên test 200

Đây là phép đo exploratory theo yêu cầu người dùng; không dùng để tối ưu.

| Phương án | MAE | RMSE |
|---|---:|---:|
| P7 retrain EXP-004 | 5.214580 | 6.828400 |
| Friend P7 ensemble | **4.730865** | **6.029187** |
| Blend 50/50 | 4.933367 | 6.324887 |
| 75% retrain + 25% friend | 5.065745 | 6.553198 |

Kết luận: trên test 200, friend P7 five-fold ensemble đang là mốc tốt nhất.
Blend không vượt friend ensemble dù blend có tín hiệu trên holdout.

Files:

- `outputs/exp005_test200_friend_methods/test200_method_report.json`.
- `outputs/exp005_test200_friend_methods/test200_method_predictions.csv`.

## 7. Các kết quả đã có trong repo của bạn cùng nhóm

### P9-I — TTA

TTA sử dụng rotation `[-10, -5, 0, 5, 10]` và có/không horizontal flip, tổng cộng
10 prediction cho mỗi ảnh.

Kết quả OOF:

- P7 raw: khoảng 6.317.
- TTA: khoảng 6.210.
- Cải thiện: khoảng 0.107 tháng.
- Khoảng tin cậy của chênh lệch không đi qua 0.

Bias correction không cải thiện thêm; TTA + bias correction kém hơn TTA đơn độc.

Quyết định: giữ TTA làm candidate inference chính, không giữ bias correction.

Evidence: `project/friend_repo/p9_inference/P9_I_HANDOFF.md` và thư mục
`project/friend_repo/p9_inference/outputs/P9_I_TTA_BIAS_OOF`.

### P10 — Augmentation

Augmentation Deeplasia moderate không cải thiện có ý nghĩa so với control.

Kết luận: không tăng augmentation một cách mù quáng. Augmentation phải được
đánh giá bằng paired OOF và kiểm tra subgroup.

### P4/P5 — Label Distribution Learning

Kết quả thử nghiệm của repo bạn cùng nhóm:

- D0 direct regression: MAE khoảng 6.18479.
- D3 fused: MAE khoảng 6.14541.
- D3 regression-only exploratory: MAE khoảng 6.12496.

Mức cải thiện nhỏ; khoảng tin cậy của D3 fused chưa đủ để tuyên bố thắng chắc.
Tuy vậy đây là hướng có cơ sở vì P7 hiện tại chưa kích hoạt distribution head.

### P4 — Multi-scale và ConvNeXtV2

- ConvNeXtV2 feature collapse: MAE khoảng 31.96.
- Multi-scale: MAE khoảng 6.29697, xấu hơn D0.
- Resolution 768: MAE khoảng 6.18342 so với 512 khoảng 6.18479, cải thiện không
  đáng kể so với chi phí.

Quyết định: tạm không ưu tiên ConvNeXtV2, multi-scale hoặc tăng resolution.

### P9-B0 — EfficientNet-B0

Screening ban đầu thất bại:

- Validation MAE khoảng 8.5293, 8.5823 và 11.0428 ở các run.

Không được kết luận rằng mọi EfficientNet đều thất bại, vì implementation chưa
phải tái lập đầy đủ pipeline Deeplasia. Tuy nhiên đây không phải hướng ưu tiên
hiện tại.

### Preprocessing mask

Một số mask/preprocessing biến thể làm validation xấu hơn control. Khả năng cao
implementation đã làm mất thông tin xương hoặc chưa tái lập đúng phương pháp gốc.

Quyết định: nếu thử lại phải có visual QC ảnh gốc, crop, mask và output cuối; chỉ
giữ nếu OOF cải thiện rõ ràng.

## 8. Các điểm yếu hiện tại của baseline

1. Baseline cũ resize trực tiếp, còn P7 của bạn cùng nhóm dùng `pad_square`.
2. Baseline cũ là một model đơn, còn kết quả 4.73 là ensemble 5 fold.
3. P7 config có trường LDL nhưng checkpoint thực tế chưa có distribution head.
4. Validation thường xấu đi trong khi train loss tiếp tục giảm; có overfit sau best
   epoch khoảng 19.
5. Nhóm tuổi 60–119 tháng thường có MAE cao nhất.
6. MAE nữ thường cao hơn MAE nam trong một số run.
7. Ensemble hiện tại chủ yếu đồng nhất ConvNeXt-Tiny; chưa có ensemble dị thể
   giữa P7, LDL và preprocessing khác.

## 9. EXP-006/007/008 đang triển khai

### EXP-006 — P7 control 5 fold với local data

Đã tạo:

- 5 fold stratified theo sex + age bins `[0,60,120,180,229]`.
- Mỗi fold khoảng 11.228–11.229 train và 2.807–2.808 validation.
- Config sử dụng đúng pipeline P7 của bạn cùng nhóm.
- Chạy trên development pool 14.036 ảnh.
- Không đọc test labels.

Preflight đã PASS cho cả 5 fold:

- train count đúng;
- validation count đúng;
- train/validation hash đúng;
- forward shape đúng;
- test path absent đúng.

Các file chuẩn bị:

- `outputs/exp006_roadmap/roadmap_preparation_audit.json`.
- `outputs/exp006_roadmap/fold_1` đến `fold_5`.
- `scripts/prepare_exp006_p7_5fold.py`.
- `scripts/merge_exp006_oof.py`.

Hiện fold 1 đã train thử 2 epoch trên Colab. Đã có:

```text
last.ckpt
best_mae.ckpt
```

Checkpoint đã được sao lưu thủ công lên Drive tại:

```text
boneage_colab/outputs/exp006_roadmap/EXP006_P7_CONTROL_FOLD_1/
```

Chưa có kết luận MAE cho EXP-006 vì fold chưa train hoàn chỉnh và OOF chưa được
gộp đủ 5 fold.

Sau đó fold 1 đã chạy đến epoch 9 trước khi Colab ngắt kết nối:

- Validation MAE hiện tại: `6.587100`.
- Best epoch hiện tại: 9.
- Global step: 2808.
- Age bin yếu nhất: `60–119`, MAE `8.644725`.
- Best checkpoint gốc và last checkpoint đều đã được lưu.
- Thư mục snapshot `best/` trên bản mirror còn bản epoch 3 cũ; đây là artifact phụ
  chưa đồng bộ, không phải lỗi của `best_mae.ckpt`.

Chi tiết kiểm tra: `EXP006_FOLD1_CHECKPOINT_AUDIT.md`.

### EXP-007 — D3 LDL regression-only

Config đã được tạo cho 5 fold nhưng chưa chạy. Cấu hình dùng:

- `architecture = convnext_tiny_ldl`;
- auxiliary label distribution loss;
- `regression_inference_weight = 1.0` để đánh giá regression-only;
- sigma 2.0;
- label distribution weight 0.2.

Chỉ chạy sau khi EXP-006 P7 control có OOF.

### EXP-008 — D3 LDL fused

Config đã được tạo cho 5 fold nhưng chưa chạy. Cấu hình dùng:

- `architecture = convnext_tiny_ldl`;
- kết hợp regression và distribution expectation với trọng số 0.5/0.5.

Chỉ chạy sau EXP-007 hoặc khi cần kiểm tra trực tiếp cơ chế fused.

## 10. Quy trình hiện tại đang theo

```text
EXP-006 P7 5-fold control
        ↓
Gộp OOF 14.036 ảnh
        ↓
Đánh giá raw vs TTA trên OOF
        ↓
EXP-007 D3 LDL regression-only
        ↓
EXP-008 D3 LDL fused
        ↓
Ensemble dị thể P7 + TTA + LDL
        ↓
Preprocessing crop/mask có visual QC
        ↓
Khóa phương án bằng OOF/holdout
        ↓
Đánh giá test 200 tham khảo
```

## 11. Quy tắc giữ hoặc loại một hướng

Một hướng chỉ được giữ khi:

- pooled OOF MAE giảm;
- cải thiện xuất hiện ổn định ở nhiều fold;
- không làm subgroup collapse;
- không tạo prediction ngoài miền quá nghiêm trọng;
- có checkpoint, config, split hash và log đầy đủ;
- không dùng test 200 để chọn.

Không ưu tiên:

- tăng epoch đơn thuần;
- augmentation mạnh không có ablation;
- ConvNeXtV2/multi-scale đã có tín hiệu âm;
- tăng resolution chỉ vì nghĩ ảnh lớn hơn luôn tốt hơn;
- chọn trọng số blend từ test 200.

## 12. Các vấn đề vận hành đã xử lý

### Chạy nhầm đường dẫn trên Windows

Đã phân biệt thư mục làm việc `D:\do_an_tot_nghiep` và đường dẫn script lồng
nhầm `project\baseline_v1\project\baseline_v1`.

### GPU không được sử dụng

Đã thêm `--device {auto,cuda,cpu}` vào baseline v1 và cơ chế báo lỗi rõ ràng nếu
người dùng yêu cầu CUDA nhưng PyTorch không thấy GPU.

### Colab shortcut Drive

Shortcut được giải quyết qua:

```text
/content/drive/.shortcut-targets-by-id/<shortcut-id>/boneage_colab
```

Code nhỏ được copy vào `/content/friend_repo`; dữ liệu lớn có thể đọc từ Drive
shortcut hoặc copy vào `/content` để tăng tốc I/O.

### Checkpoint Drive

Trainer chạy nhanh trên `/content`, còn `best_mae.ckpt` và `last.ckpt` được mirror
sang Drive nếu config có `checkpoint_mirror_root`. Nếu không có tham số này,
checkpoint chỉ nằm trong `/content` và sẽ mất khi runtime reset.

### Hết phiên Colab

Có thể resume từ `last.ckpt`. Không đổi các tham số khoa học giữa hai lần resume;
nếu đổi learning rate, batch, augmentation, model hoặc loss thì tạo run ID mới.

### Chuyển sang Kaggle

Đã tạo quy trình riêng tại `EXP006_KAGGLE_RUN.md`. Kaggle dùng:

- input read-only dưới `/kaggle/input`;
- code và run tạm dưới `/kaggle/working`;
- checkpoint/output cần được Save Version hoặc đóng gói thành Kaggle Dataset
  trước khi runtime hết thời gian.

Fold 1 hiện có thể resume từ `last.ckpt` của epoch 9 thay vì train lại từ đầu.

## 13. Mốc kết quả hiện tại

| Mốc | MAE | Ý nghĩa |
|---|---:|---|
| Baseline own official | 6.471143 | Mốc baseline cũ |
| A2 own official | 6.706371 | Không cải thiện |
| Friend P7 OOF | 6.316691 | Mốc OOF của bạn cùng nhóm |
| EXP-004 retrain holdout | 6.213796 | Một model trên fresh holdout |
| Friend P7 test 200 | 4.730865 | Mốc test tham khảo tốt nhất hiện tại |
| Mục tiêu cuối | < 3.5 | Chưa đạt |

Điểm cần lưu ý: MAE 4.73 của bạn cùng nhóm là ensemble trên test 200, trong khi
6.21 của EXP-004 là một model trên holdout. Đây không phải phép so sánh hoàn toàn
đồng cấp. EXP-006 P7 5-fold control sẽ giúp tạo phép so sánh công bằng hơn.

## 14. Kết luận hiện tại

- Pipeline P7 của bạn cùng nhóm là baseline mạnh hơn baseline v1 cũ nhờ
  `pad_square`, target normalization, batch hiệu dụng lớn và ensemble 5 fold.
- Blend có tín hiệu trên official validation và fresh holdout nhưng không vượt
  friend ensemble trên test 200.
- TTA là cải tiến inference có bằng chứng tốt nhất hiện tại, khoảng -0.107 MAE
  trên OOF.
- LDL là cải tiến train có tiềm năng nhưng chưa được triển khai đầy đủ trong P7
  thực tế.
- Preprocessing mask, ConvNeXtV2, multi-scale, EfficientNet screening và tăng
  resolution chưa cho bằng chứng đủ mạnh để ưu tiên.
- EXP-006 đang là bước quan trọng nhất: tạo control 5 fold công bằng trên local
  data trước khi chạy LDL hoặc ensemble mới.

## 15. File bằng chứng chính

- Quy trình baseline cũ: `BASELINE_QUY_TRINH.md`.
- Nhật ký thực nghiệm: `EXPERIMENT_LOG.md`.
- Quy trình Colab EXP-006: `EXP006_COLAB_RUN.md`.
- Chuẩn bị 5 fold: `scripts/prepare_exp006_p7_5fold.py`.
- Gộp OOF: `scripts/merge_exp006_oof.py`.
- Đánh giá TTA: `scripts/evaluate_exp006_tta_oof.py`.
- P7 của bạn cùng nhóm: `../friend_repo/p1_baseline` và `../friend_repo/p7_results`.
- P9 TTA: `../friend_repo/p9_inference/P9_I_HANDOFF.md`.
- P4 LDL: `../friend_repo/p4_architecture/P4_HANDOFF.md`.
- So sánh pipeline: `../friend_repo/AI_Context/06_PIPELINE_COMPARISON.md`.
