# KẾ HOẠCH TỔNG THỂ – DỰ ĐOÁN TUỔI XƯƠNG TỪ ẢNH X-QUANG BÀN TAY

**Tên đề tài đề xuất:** Mô hình ConvNeXt V2 đa mức kết hợp phân phối nhãn nhằm dự đoán tuổi xương chính xác và bền vững trước vật thể phi giải phẫu trên ảnh X-quang bàn tay  
**Tên tiếng Anh:** Artifact-Robust Bone Age Assessment Using Multi-Scale Label-Distribution ConvNeXt V2  
**Tên mô hình đề xuất:** MSLD–ConvNeXtV2  
**Phiên bản kế hoạch:** 3.0  
**Ngày khóa kế hoạch ban đầu:** 2026-08-12  
**Trạng thái:** P6 hoàn thành – P7 final 5-fold đã khóa code/split và sẵn sàng chạy trên Colab

> Đây là nguồn thông tin chính cho toàn bộ đồ án. Mỗi hội thoại mới phải đọc hết file này trước khi sửa code, train, đánh giá hoặc viết báo cáo. Nếu thay đổi protocol, phải ghi rõ lý do và cập nhật CHANGE LOG ở cuối file.

---

## 1. Mục tiêu nghiên cứu

### 1.1. Câu hỏi nghiên cứu chính

Việc kết hợp:

1. tiền xử lý bảo toàn toàn bộ giải phẫu bàn tay;
2. ConvNeXt V2-Tiny;
3. hợp nhất đặc trưng đa mức;
4. sex embedding;
5. hai đầu ra direct regression và label-distribution;

có làm giảm sai số dự đoán tuổi xương và giảm sự phụ thuộc của mô hình vào chữ, marker, viền ảnh và các vật thể phi giải phẫu hay không?

### 1.2. Ba đóng góp dự kiến

1. **Đóng góp kiến trúc:** MSLD–ConvNeXtV2 kết hợp đặc trưng nhiều stage và phân phối nhãn tuổi.
2. **Đóng góp robustness:** đánh giá tính bất biến trên 200 cặp ảnh RSNA gốc và ảnh đã giảm artifact.
3. **Đóng góp thực nghiệm:** ablation có kiểm soát, test được khóa, bootstrap CI, phân tích theo giới tính và nhóm tuổi.

### 1.3. Phạm vi kết luận

- Nếu chỉ đánh giá trên RSNA: chỉ kết luận về hiệu năng nội bộ trên RSNA và robustness đối với artifact.
- Nếu có thêm DHA: có thể kết luận thêm về khả năng tổng quát hóa ngoài bộ dữ liệu huấn luyện.
- Bộ 200 ảnh artifact-reduced vẫn là cùng ảnh test RSNA, **không phải external validation**.

---

## 2. Bài báo mốc và ngưỡng so sánh

### 2.1. Mốc tái lập: Deeplasia

Rassmann et al., *Deeplasia: deep learning for bone age assessment validated on skeletal dysplasias*, Pediatric Radiology, 2024.  
DOI: https://doi.org/10.1007/s00247-023-05789-1  
Mã nguồn: https://github.com/aimi-bonn/Deeplasia

- RSNA train: 12.611 ảnh.
- RSNA validation: 1.425 ảnh.
- RSNA test: 200 ảnh.
- Test MAD/MAE: 3,87 tháng.
- Test RMSE: 5,14 tháng.
- Accuracy trong 12 tháng: 98,5%.
- Mô hình: ensemble ba EfficientNet được chọn từ nhiều điều kiện train.

### 2.2. Mốc hiệu năng: Bram et al.

Bram et al., *Determination of Skeletal Age From Hand Radiographs Using Deep Learning*, The American Journal of Sports Medicine, 2025.  
DOI: https://doi.org/10.1177/03635465251359618

- RSNA test MAE: 3,68 tháng, 95% CI 3,24–4,14.
- RSNA test RMSE: 4,92 tháng, 95% CI 4,23–5,73.
- DHA với Model 1 chỉ dùng ảnh + giới tính: MAE 5,66, RMSE 7,96.
- DHA với Model 2 dùng thêm chronological age và RHPE transfer: MAE 4,65, RMSE 6,38.

Ablation của Bram trên RSNA test:

| Preprocessing | Augmentation | Ensemble | MAE (tháng) |
|---|---|---|---:|
| Không | Không | Không | 4,97 |
| Có | Không | Không | 4,46 |
| Có | Có | Không | 3,74 |
| Có | Có | Có | 3,68 |

**Quy tắc so sánh công bằng:**

- Trên RSNA, mục tiêu chính là MAE < 3,68 và RMSE < 4,92.
- Trên DHA, mô hình chỉ dùng ảnh + giới tính chỉ được so với Bram Model 1: MAE 5,66.
- Không so mô hình của ta với mốc DHA 4,65 nếu ta không dùng chronological age và RHPE.
- Không thể tuyên bố vượt khả năng trên skeletal dysplasia nếu không có bộ test dysplasia tương ứng.

### 2.3. Cơ sở của backbone mới

Woo et al., *ConvNeXt V2: Co-Designing and Scaling ConvNets With Masked Autoencoders*, CVPR 2023.  
Bài báo: https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html

ConvNeXt V2 là một giả thuyết cần kiểm chứng bằng D0–D1; không mặc định V2 sẽ tốt hơn V1 trên RSNA.

---

## 3. Dữ liệu và quy tắc chống leakage

### 3.1. Bộ chia phát triển

- Train chính thức: 12.611 ảnh.
- Validation chính thức: 1.425 ảnh.
- Test chính thức: 200 ảnh.
- Tất cả lựa chọn mô hình, augmentation, loss và checkpoint chỉ dựa trên validation.
- Phải lưu manifest gồm: image ID, path, sex, bone age, split và hash file.
- Kiểm tra trùng ID, trùng hash, thiếu ảnh, ảnh hỏng, nhãn ngoài miền và metadata sai.

Đường dẫn validation chính thức đã xác minh:

- CSV: `D:\Hoctap\Doan_totnghiep\Dataset\RSNA\boneage-validation-dataset.csv`.
- Ảnh: `D:\Hoctap\Doan_totnghiep\Dataset\RSNA\boneage-validation-dataset\boneage-validation-dataset`.
- Thư mục ảnh có hai thư mục con tương ứng hai gói gốc; dataloader đọc đường dẫn chính xác từ manifest, không cần làm phẳng.

Fingerprint khóa sau P0:

```text
train_manifest_sha256=7328667e6822ab074d442155e33eba89606861bb13bbf804be8a1138c78f5285
validation_manifest_sha256=f650a20432b557405035d12c52716a6c40fdf20e1343ac3ad0cfb7cd8db21631
test_locked_manifest_sha256=fd22bf5b44397a32826cfedad805b31ba727637047a48be51cc8954bb51229db
development_manifest_14036_sha256=db1d62d768aca44016e9434c0d3c3b64e5f695094338cff0dbf6da2077e5925a
three_split_registry_sha256=234d368e37dc908c4e30d271a5dee2e3a853e4a9920628b476c418e457b6b7e6
```

P0 xác nhận validation có đúng 1.425 ảnh, ID không giao với train/test, không có exact duplicate theo SHA-256 và tuổi/giới tính khớp hoàn toàn annotation Deeplasia. Không tự chia lại validation từ train.

### 3.2. Bộ chia train cuối

Chỉ sau khi đã khóa toàn bộ cấu hình:

1. Gộp train và validation thành 14.036 ảnh.
2. Chia stratified 5-fold theo sex và age bin.
3. Train đúng một config trên năm fold.
4. Lấy trung bình năm dự đoán làm kết quả cuối.
5. Lấy sample standard deviation giữa năm fold làm chỉ số uncertainty.

### 3.3. Khóa test

Cấm thực hiện các hành động sau trước khi khóa mô hình:

- Chọn checkpoint dựa trên 200 ảnh test.
- Chọn hyperparameter dựa trên MAE test.
- Train hoặc fine-tune bằng ảnh test gốc hay ảnh test đã làm sạch.
- Dùng nhãn test trong bất kỳ bước chọn mô hình nào.
- Trung bình dự đoán ảnh gốc và ảnh sạch để hạ MAE chính thức.
- Sửa mô hình sau khi đã xem kết quả test cuối, rồi vẫn gọi đó là một lần test độc lập.

Test gốc và test artifact-reduced chỉ được chạy sau khi config và quy tắc chọn checkpoint đã khóa.

---

## 4. Bộ 200 ảnh artifact-reduced hiện có

### 4.1. Đường dẫn

- Ảnh đã làm sạch: `C:\Users\ASUS\Documents\tao_sinh_anh_x_ray_hand\outputs\artifact_only_200_manual_v3\cleaned`
- Ảnh gốc: `C:\Users\ASUS\Documents\tao_sinh_anh_x_ray_hand\data\original_200`
- Artifact masks: `C:\Users\ASUS\Documents\tao_sinh_anh_x_ray_hand\outputs\artifact_only_200_manual_v3\artifact_masks`
- Protected masks: `C:\Users\ASUS\Documents\tao_sinh_anh_x_ray_hand\outputs\artifact_only_200_manual_v3\protected_masks`
- QC CSV: `C:\Users\ASUS\Documents\tao_sinh_anh_x_ray_hand\outputs\artifact_only_200_manual_v3\artifact_only_qc.csv`

### 4.2. Kết quả QC đã biết

- Có đủ 200 ảnh gốc, 200 ảnh sạch, 200 artifact mask và 200 protected mask.
- ID liên tục từ 4360 đến 4559; không trùng ID.
- Kích thước và image mode giữa ảnh gốc/sạch khớp nhau.
- 185 mask thủ công, 15 mask tự động.
- Diện tích artifact trung bình: 3,123% ảnh.
- Pixel thay đổi trung bình: 2,676%; median 2,680%; p95 4,129%.
- Pixel thay đổi lớn nhất: 11,579%, ID 4364.
- PSNR trung bình khoảng 30,02 dB; nhỏ nhất 22,49 dB.
- 200/200 đạt pixel preservation check.
- Không có pixel thay đổi ngoài artifact mask.
- Không có pixel thay đổi trong protected mask.
- 110/200 ảnh được đánh dấu `detector_review_required`.
- Vẫn có thể còn artifact nhỏ hoặc viền collimation; vì vậy gọi là **artifact-reduced**, không gọi là artifact-free.

### 4.3. Cách sử dụng khoa học

Tên thí nghiệm: **Artifact Invariance Test**.

| Điều kiện | Đầu vào |
|---|---|
| T1 | Ảnh test RSNA gốc |
| T2 | Ảnh test artifact-reduced |
| T3 | Ảnh gốc sau full-hand masking |
| T4 | Ảnh artifact-reduced sau cùng full-hand masking |

Báo cáo:

- MAE, RMSE trên T1 và T2.
- Delta MAE = MAE(T2) − MAE(T1).
- Mean/median của `abs(pred_clean - pred_original)`.
- Tỷ lệ cặp ảnh thay đổi dự đoán >3, >6 và >12 tháng.
- Spearman correlation giữa artifact area và prediction shift.
- Paired bootstrap confidence interval.
- Wilcoxon signed-rank nếu phù hợp.
- Bland–Altman giữa dự đoán gốc và sạch.
- Phân tích theo manual/auto mask, review flag và artifact area.
- Grad-CAM hoặc occlusion trên các ảnh có prediction shift lớn.

**Không dùng bộ 200 cặp ảnh này để train.** Nếu cần artifact augmentation, phải tạo artifact tổng hợp label-independent trên train, chỉ ở ngoài hand mask.

---

## 5. Mô hình đề xuất MSLD–ConvNeXtV2

### 5.1. Đầu vào và tiền xử lý

- Input ban đầu: 512 × 512.
- Ảnh grayscale được lặp thành ba kênh.
- Chuẩn hóa intensity cố định và giống nhau cho train/validation/test.
- Full-hand segmentation/masking, padding và alignment nếu B1 chứng minh có lợi.
- Giữ đầy đủ distal radius/ulna, carpal, metacarpal và phalanges.
- Không dùng hard ROI chỉ cắt một phần bàn tay.
- Không dùng vertical flip.
- Không dùng phép biến đổi làm thay đổi morphology quá mạnh.

### 5.2. Backbone và fusion

- Backbone chính: ConvNeXt V2-Tiny pretrained.
- Lấy feature maps từ stage 2, stage 3 và stage 4.
- Global average pooling từng stage.
- Project các vector về cùng feature dimension.
- Gated/weighted fusion để mô hình học mức đóng góp của từng scale.
- Mã hóa sex bằng embedding, sau đó nối với image feature.

### 5.3. Hai đầu ra

1. **Direct regression head:** dự đoán tuổi liên tục theo tháng.
2. **Label-distribution head:** dự đoán phân phối tuổi trên miền 0–228 tháng.

Dự đoán từ label-distribution:

`age_ldl = sum(probability_i * age_i)`

Dự đoán cuối ban đầu:

`age_final = 0.5 * age_reg + 0.5 * age_ldl`

Trọng số này chỉ được thay đổi dựa trên validation trước khi khóa model.

### 5.4. Loss ban đầu

`L = L_reg + lambda_ldl * L_ldl + lambda_cons * L_cons`

- `L_reg`: Smooth L1 giữa tuổi thật và direct prediction.
- `L_ldl`: cross-entropy/KL với Gaussian label distribution.
- `L_cons`: ràng buộc direct prediction và expected age không lệch quá lớn.
- Khởi tạo: `sigma = 3` tháng, `lambda_ldl = 1`, `lambda_cons = 0.1`.
- Nếu cần, chỉ thử sigma trong tập nhỏ `{2, 3, 4}`.

---

## 6. Chuỗi thí nghiệm bắt buộc

### 6.1. Phase A – Audit và augmentation

| Run | Cấu hình | Mục đích |
|---|---|---|
| A0 | ConvNeXt-Tiny + direct regression, không horizontal flip | Baseline augmentation |
| A1 | A0 + horizontal flip p=0,5 | Đo đóng góp riêng của flip |
| A2 | Cấu hình tốt hơn A0/A1 + augmentation nhẹ đầy đủ | Kiểm chứng augmentation kiểu Bram |

Augmentation nhẹ dự kiến:

- horizontal flip p=0,5 nếu A1 tốt hơn;
- rotation tối đa ±7°;
- translation tối đa 3%;
- scale 0,95–1,05;
- brightness/contrast/gamma nhẹ;
- không vertical flip;
- chưa dùng MixUp, CutMix, elastic deformation hay TTA.

### 6.2. Phase B – Tiền xử lý

| Run | Cấu hình | Mục đích |
|---|---|---|
| B0 | Cấu hình augmentation tốt nhất | Baseline không full-hand masking |
| B1 | B0 + official full-hand masking + background neutralization; không crop, không xoay | Đo đóng góp riêng của masking |

Lưu ý: full-hand masking không đồng nghĩa với hard ROI. Giao thức B1 được khóa trước khi xem MAE validation:

- mask chính thức từ Rassmann et al., Zenodo DOI `10.5281/zenodo.7611677`;
- chọn `eff_unet` làm nguồn chính vì Dice trung bình trên mask thủ công cao hơn `Tensormask`;
- `Tensormask` chỉ bù ID thiếu; nếu cả hai thiếu thì giữ ảnh gốc và ghi fallback;
- đặt nền ngoài mask bằng 0, sau đó pad vuông và resize 512 giống B0;
- không crop/rotate trên RSNA vì Deeplasia dùng `mask_crop_size=-1`, đồng thời hard ROI trước đó đã cho tín hiệu xấu;
- alignment/crop chỉ được xem là thí nghiệm riêng trong tương lai, không gộp vào B1.

Kết quả P3:

| Run | Validation MAE | RMSE | Accuracy ±6 | Kết luận |
|---|---:|---:|---:|---|
| B0 = A2, không mask | 6,18479 | 8,48647 | 63,44% | Được chọn |
| B1, official mask | 6,23894 | 8,44963 | 62,32% | Không chọn cho pipeline chính |

`MAE_B1 - MAE_B0 = +0,05414` tháng; paired bootstrap 95% CI `[-0,10167; +0,21180]`. CI chứa 0 nên không thể kết luận B1 gây hại có ý nghĩa thống kê, nhưng B1 không chứng minh được cải thiện primary endpoint. P4 dùng `preprocessing=none`; B1 chỉ giữ cho robustness/phân tích phụ.

### 6.3. Phase C – Kiến trúc

| Run | Cấu hình | Mục đích |
|---|---|---|
| D0 | ConvNeXt-Tiny + direct regression | Baseline gần Bram |
| D1 | ConvNeXtV2-Tiny + direct regression | Đo đóng góp của backbone V2 |
| D2 | Backbone thắng D0–D1 + multi-scale fusion | Đo đóng góp của đặc trưng đa mức |
| D3 | Backbone/head thắng D0–D2 + label-distribution head | Đo đóng góp của học phân phối nhãn |

D0–D3 phải giữ nguyên split, seed, input size, preprocessing, augmentation, optimizer, scheduler và checkpoint rule.

### 6.4. Phase D – Xác nhận seed

- Chọn hai config tốt nhất từ Phase C.
- Chạy tổng cộng ba seed cho mỗi config.
- So sánh paired absolute error trên cùng validation images.
- Giữ một thành phần nếu mean improvement khoảng ≥0,10 tháng và ổn định qua seed/bootstrap.
- Loại thành phần nếu cải thiện quá nhỏ, không ổn định, hoặc làm xấu rõ rệt một sex/age subgroup.

### 6.5. Phase E – Độ phân giải

- Chỉ thử sau khi model 512 đã ổn định.
- So sánh 512 với 768 cho duy nhất config tốt nhất.
- Chỉ thử 1024 nếu GPU/thời gian cho phép và 768 cho tín hiệu cải thiện.
- Không lặp lại toàn bộ ablation ở mỗi resolution.

### 6.6. Phase F – Final 5-fold ensemble

- Đóng băng config.
- Gộp 14.036 train + validation images.
- Tạo 5 fold stratified theo sex và age bin.
- Train năm fold cùng một recipe.
- Ensemble mean để dự đoán.
- Ensemble standard deviation để ước lượng uncertainty.
- Chạy test gốc, artifact-reduced và DHA nếu có đúng một lần đánh giá cuối.

---

## 7. Tiêu chí chọn và dừng thí nghiệm

Mỗi run phải lưu:

- config đầy đủ;
- git/code snapshot hoặc hash;
- split manifest/hash;
- seed;
- best epoch và checkpoint;
- validation prediction CSV theo từng ảnh;
- MAE/RMSE tổng và theo subgroup;
- learning curves;
- GPU, VRAM và thời gian train.

Quy tắc chọn:

1. Ưu tiên validation MAE.
2. Nếu chênh lệch <0,10 tháng và bootstrap/seed không ổn định, chọn model đơn giản hơn.
3. Kiểm tra RMSE để tránh một số ảnh có sai số rất lớn.
4. Kiểm tra male/female và age bins; không chỉ xem MAE tổng.
5. Chỉ top-2 config mới được chạy ba seed.
6. Chỉ một config khóa cuối mới được train 5-fold.

### 7.1. Checkpoint và khả năng tiếp tục train

Mọi script train phải hỗ trợ **resume chính xác**, vì quá trình huấn luyện có thể bị dừng giữa chừng. Không được thiết kế theo kiểu phải train lại từ đầu sau khi Colab ngắt phiên hoặc người dùng chủ động dừng.

Mỗi run bắt buộc lưu:

- `last.ckpt`: trạng thái mới nhất, cập nhật sau mỗi epoch;
- `best_mae.ckpt`: checkpoint có validation MAE tốt nhất;
- `periodic/epoch_XXX.ckpt`: checkpoint định kỳ để quay lại khi checkpoint mới nhất bị lỗi;
- checkpoint tạm theo số optimizer step hoặc thời gian đối với epoch dài;
- file `run_state.json` ghi epoch, global step, best metric và trạng thái run;
- `config_resolved.yaml`: toàn bộ cấu hình thực tế sau khi resolve default;
- prediction CSV của validation ở mỗi epoch tốt nhất.

Một checkpoint dùng để resume phải chứa tối thiểu:

- model state;
- optimizer state;
- scheduler state;
- AMP gradient-scaler state;
- epoch và global optimizer step;
- best metrics và early-stopping counter;
- random states của Python, NumPy, PyTorch CPU và CUDA;
- seed, split hash, config hash và code/version identifier;
- trạng thái sampler cần thiết để tiếp tục đúng protocol.

Quy tắc lưu đề nghị:

- lưu `last.ckpt` sau mỗi epoch;
- với epoch dài, lưu thêm mỗi 500 optimizer steps hoặc tối đa khoảng 20–30 phút một lần;
- lưu `best_mae.ckpt` ngay khi validation MAE cải thiện;
- giữ tối thiểu hai checkpoint periodic gần nhất và ba checkpoint tốt nhất nếu dung lượng cho phép;
- ghi file theo cách an toàn: lưu file tạm, kiểm tra thành công rồi mới thay checkpoint đích;
- sau khi lưu, kiểm tra checkpoint có thể đọc lại và có đầy đủ các key bắt buộc;
- đồng bộ checkpoint, config và log ra nơi lưu bền vững sau mỗi epoch khi chạy Colab.

Khi resume, chương trình phải in rõ:

```text
RESUME CHECK
Run ID:
Checkpoint path:
Epoch/global step tiếp tục:
Best validation MAE trước đó:
Split hash khớp: CÓ/KHÔNG
Config hash khớp: CÓ/KHÔNG
Optimizer/scheduler/scaler đã phục hồi: CÓ/KHÔNG
```

Nếu split hash hoặc các tham số khoa học quan trọng không khớp, mặc định phải từ chối resume. Chỉ cho phép thay đổi các tham số vận hành không làm đổi thí nghiệm, chẳng hạn số worker hoặc tần suất in log.

### 7.2. Log bắt buộc

Log phải vừa in ra màn hình, vừa được lưu thành file máy đọc được. Mỗi run tối thiểu có:

- `train.log`: log văn bản đầy đủ;
- `metrics.csv` hoặc `metrics.jsonl`: một dòng cho mỗi epoch;
- TensorBoard log nếu môi trường hỗ trợ;
- `warnings.log`: chỉ chứa các cảnh báo bất ổn;
- `environment.txt`: GPU, CUDA, driver và phiên bản thư viện;
- `val_predictions_best.csv`: ID, tuổi thật, tuổi dự đoán, sai số tuyệt đối, sex và age bin.

Log theo một số batch phải có:

- epoch, batch/optimizer step;
- learning rate;
- train loss tổng và từng thành phần loss;
- gradient norm;
- GPU memory allocated/reserved;
- tốc độ ảnh/giây và thời gian dự kiến còn lại;
- số batch bị bỏ qua do dữ liệu lỗi, NaN hoặc OOM.

Log cuối mỗi epoch phải có:

- train loss trung bình;
- validation loss;
- validation MAE, RMSE và median AE;
- accuracy trong ±6, ±12 và ±18 tháng;
- MAE nam, MAE nữ và khoảng cách giữa hai nhóm;
- MAE theo age bin;
- mean/std/min/max của prediction;
- learning rate, epoch time và peak VRAM;
- best epoch/best MAE hiện tại;
- số epoch không cải thiện.

### 7.3. Cờ cảnh báo bất ổn

Code phải in cảnh báo nổi bật với tiền tố `STABILITY WARNING` và ghi vào `warnings.log`. Các điều kiện ban đầu:

| Mức | Dấu hiệu | Hành động mặc định |
|---|---|---|
| Đỏ | Loss, gradient hoặc prediction có NaN/Inf | Dừng an toàn và giữ checkpoint gần nhất |
| Đỏ | Checkpoint hỏng, split/config hash không khớp khi resume | Không tiếp tục train |
| Đỏ | Prediction sụp về gần một hằng số hoặc nằm ngoài miền hợp lý nghiêm trọng | Dừng để kiểm tra pipeline/nhãn |
| Cam | Gradient norm tăng đột biến hoặc vượt ngưỡng cấu hình nhiều lần liên tiếp | Lưu checkpoint, in batch/step liên quan và đề nghị review |
| Cam | Validation MAE xấu đi liên tiếp trong khi train loss vẫn giảm | Cảnh báo overfitting; chưa tự động kết luận |
| Cam | Validation MAE không cải thiện trong số epoch bằng patience | Kích hoạt early stopping sau khi lưu checkpoint |
| Cam | Một thành phần loss lớn bất thường hoặc lấn át các loss còn lại | In riêng từng loss và tỷ lệ giữa chúng |
| Cam | Chênh lệch MAE nam–nữ hoặc một age bin tăng mạnh | In subgroup table để review |
| Vàng | Loss/MAE tăng vọt so với moving median các epoch gần nhất | Đánh dấu bất thường, tiếp tục tới mốc validation/checkpoint kế tiếp |
| Vàng | GPU OOM, dataloader lỗi hoặc throughput giảm mạnh | Ghi đầy đủ lỗi và số lần xuất hiện |

Ngưỡng cụ thể phải đặt trong config, không hard-code rải rác. Các ngưỡng ban đầu sẽ được hiệu chỉnh sau smoke test; không tự động dừng chỉ vì một dao động nhỏ ở một batch.

### 7.4. Điểm người dùng quyết định tiếp tục

Sau mỗi epoch validation hoặc khi xuất hiện cảnh báo cam/đỏ, log phải in một khối tóm tắt có thể sao chép để gửi sang hội thoại khác:

```text
TRAINING REVIEW SNAPSHOT
Run ID / config hash / split hash:
Epoch / global step:
Train loss và các thành phần:
Validation MAE / RMSE / median AE:
MAE nam / nữ / age bins:
Learning rate / gradient norm / peak VRAM:
Best epoch / best MAE / epochs không cải thiện:
Cảnh báo hiện tại:
Checkpoint gần nhất:
Checkpoint tốt nhất:
Đề xuất tự động: TIẾP TỤC / THEO DÕI / TẠM DỪNG KIỂM TRA
```

`Đề xuất tự động` chỉ là gợi ý dựa trên quy tắc, không thay thế đánh giá nghiên cứu. Nếu có dấu hiệu bất ổn, người dùng sẽ gửi snapshot cùng đoạn `warnings.log` và một phần learning curve để phân tích trước khi quyết định tiếp tục.

### 7.5. Kiểm thử bắt buộc trước run dài

Trước A0 và trước mỗi thay đổi lớn của trainer:

1. Chạy smoke test trên tập nhỏ trong 1–2 epoch.
2. Dừng có chủ ý giữa run.
3. Resume từ `last.ckpt`.
4. Xác nhận epoch, optimizer, scheduler, scaler và metric tiếp tục đúng.
5. Thử đọc `best_mae.ckpt` bằng một process mới.
6. Xác nhận log, prediction CSV và warning mechanism được tạo đầy đủ.
7. Chỉ khi bài kiểm tra này đạt mới được phép chạy run dài trên Colab.

---

## 8. Chỉ số đánh giá cuối

Bắt buộc báo cáo:

- MAE.
- RMSE.
- Median absolute error.
- Accuracy trong ±6, ±12 và ±18 tháng.
- 95% bootstrap confidence interval, 3.000–10.000 bootstrap samples.
- MAE theo sex.
- MAE theo age bin.
- Scatterplot prediction–ground truth.
- Bland–Altman.
- Error theo ensemble uncertainty.
- Calibration/retention curve nếu phù hợp.
- Grad-CAM và/hoặc occlusion analysis.
- Artifact Invariance Test T1–T4.
- Số tham số, thời gian inference và cấu hình GPU.

Ngưỡng mục tiêu:

| Mức | RSNA test MAE |
|---|---:|
| Vượt Deeplasia | < 3,87 |
| Vượt point estimate của Bram | < 3,68 |
| Kết quả mạnh cho đồ án | ≤ 3,55 |
| Kết quả rất mạnh | 3,40–3,50 |

Thêm:

- Mục tiêu RMSE: <4,92 để vượt point estimate của Bram.
- Nếu có DHA và chỉ dùng image + sex: mục tiêu MAE <5,66.
- MAE 3,65 so với 3,68 chưa đủ để nói “vượt trội có ý nghĩa” do test chỉ có 200 ảnh.
- Cần paired bootstrap với baseline tái lập; không thể paired-test trực tiếp với Bram nếu không có per-image predictions của họ.

---

## 9. Các kết quả sơ bộ đã biết

Kết quả do người dùng cung cấp từ một pipeline khác:

| Model | Validation MAE |
|---|---:|
| ConvNeXt-Tiny + flip | 6,173 |
| ConvNeXt-Tiny + flip + ROI | 6,321 |
| ResNet50 + flip | 6,375 |
| ResNet50 + flip + Mean Teacher | 6,482 |
| EfficientNet-B0 + flip | 6,644 |

Diễn giải tạm thời:

- ConvNeXt-Tiny là backbone có tín hiệu tốt nhất trong bảng.
- Hard ROI làm xấu 0,148 tháng, khoảng 2,4%.
- Mean Teacher làm xấu ResNet50 0,107 tháng.
- EfficientNet-B0 kém nhất trong bảng này.
- Không thể kết luận flip có lợi vì không có dòng `không flip` cùng điều kiện.
- Không so trực tiếp validation MAE 6,173 với RSNA test MAE 3,68; hai split có độ khó khác nhau.

Metadata cần lấy trước khi tái sử dụng run 6,173:

- danh sách ID và hash split;
- official hay random split, kích thước validation;
- input size;
- pretrained weights;
- flip chỉ train hay có TTA;
- ROI được tạo/cắt như thế nào;
- có đưa sex vào model không;
- loss, optimizer, scheduler;
- epoch và checkpoint rule;
- seed/fold;
- đơn vị nhãn;
- test đã từng bị dùng để chọn model hay chưa.

---

## 10. Các hướng không ưu tiên

- Không tiếp tục Mean Teacher ở giai đoạn chính.
- Không dùng hard ROI làm phương pháp chính.
- Không tìm kiếm rộng ResNet, DenseNet, EfficientNet và nhiều backbone khác.
- Không tiếp tục stacking trên OOF Phase 1 chưa hoàn chỉnh.
- Không dùng 200 ảnh sạch làm training augmentation.
- Không gọi ảnh đã xử lý là hoàn toàn artifact-free.
- Không báo cáo chỉ mỗi MAE; phải có RMSE, CI và subgroup.
- Không hứa chắc chắn vượt 3,68 trước khi test.

---

## 11. Thứ tự thực hiện chính thức

- [x] P0. Audit dữ liệu, split, metadata và leakage.
- [x] P1. Xây baseline ConvNeXt-Tiny có khả năng tái lập.
- [x] P2. Chạy A0–A2 và khóa augmentation.
- [x] P3. Chạy B0–B1 và khóa preprocessing `none`.
- [x] P4. Chạy D0–D3 trên cùng protocol.
- [ ] P5. Xác nhận top-2 bằng ba seed.
- [ ] P6. Thử 512–768 cho config tốt nhất nếu tài nguyên cho phép.
- [ ] P7. Khóa model và train final stratified 5-fold.
- [ ] P8. Đánh giá RSNA original test.
- [ ] P9. Chạy Artifact Invariance Test T1–T4.
- [ ] P10. Đánh giá DHA nếu có thể truy cập hợp lệ.
- [ ] P11. Thống kê, hình ảnh, ablation và viết báo cáo.

---

## 12. Mẫu bàn giao giữa các hội thoại

Mỗi hội thoại chỉ nên xử lý một phase. Khi kết thúc, cập nhật một file kết quả riêng và điền mẫu sau:

```text
Phase:
Mục tiêu:
Code/config đã thay đổi:
Dữ liệu/split hash:
Run IDs:
Kết quả chính:
Kết quả subgroup:
Lỗi hoặc hạn chế:
Quyết định giữ/loại:
Bước tiếp theo được phép:
Test set đã được chạm hay chưa: KHÔNG/CÓ
```

Prompt để bắt đầu một hội thoại mới:

```text
Hãy đọc toàn bộ file BONE_AGE_MASTER_PLAN.md trước. Chỉ thực hiện Phase [mã phase].
Không sử dụng test set để chọn mô hình. Trước khi sửa code, hãy audit hiện trạng và nếu
có mâu thuẫn với Master Plan thì báo rõ. Khi kết thúc, lưu config, prediction validation,
metrics, run metadata và cập nhật trạng thái phase; không tự động chuyển sang phase sau.
```

---

## 13. Quyết định kiến trúc đã chốt và chưa chốt

### Đã chốt

- Bài báo mốc tái lập: Deeplasia.
- Bài báo mốc hiệu năng: Bram et al. 2025.
- Backbone nghiên cứu: ConvNeXt V2-Tiny.
- Input ban đầu: 512.
- Có sex embedding.
- Có ablation multi-scale và label-distribution.
- Final evaluation dùng 5-fold ensemble.
- 200 ảnh sạch chỉ dùng paired robustness test.
- Official test được khóa đến cuối.
- Augmentation từ P2: horizontal flip p=0,5; rotation ±7°; translation 3%; scale 0,95–1,05; brightness/contrast ±10%; gamma 0,90–1,10.
- Preprocessing chính từ P3: `none`; official masking không thắng MAE và chỉ giữ cho robustness branch.

### Chỉ được chốt sau ablation

- Có giữ ConvNeXt V2 thay ConvNeXt V1 hay không.
- Có giữ multi-scale fusion hay không.
- Có giữ label-distribution head hay không.
- Resolution cuối là 512 hay 768.
- Trọng số kết hợp regression/LDL.

---

## 14. CHANGE LOG

### Phiên bản 3.0 – 2026-08-16

- Khóa P7 final 5-fold bằng `StratifiedKFold`, seed chia fold 2026, strata giới × bốn nhóm tuổi.
- Năm validation fold phủ đúng 14.036 ID, không trùng; mỗi fold train 11.228/11.229 ảnh và validation 2.808/2.807 ảnh.
- Cấu hình cuối: ConvNeXt-Tiny direct regression, input 512, augmentation A2, effective batch 36, seed train 42.
- Mỗi fold tính target mean/std chỉ từ phần train tương ứng để tránh dùng nhãn held-out trong preprocessing.
- Thêm đường dẫn ảnh tương đối cho Colab và cơ chế mirror nguyên tử checkpoint/log từ SSD tạm sang Google Drive.
- Thêm tự động resume với fallback periodic checkpoint, staging SHA-256 đủ 14.036 ảnh, preflight chống test leakage và tổng hợp OOF.
- Validation setup PASS toàn bộ năm fold; unit test core+P7 PASS 18/18; bundle ZIP kiểm tra toàn vẹn PASS.
- Notebook: `p7_final/P7_COLAB.ipynb`; bundle SHA-256 `d7e5879b99022275839bee6cceeac6e34fe17496406f456ef4ecd1c9ac6c2329`.

### Phiên bản 2.9 – 2026-08-16

- P6 768 early stop epoch 32; best epoch 24, MAE `6,18342`, RMSE `8,37426` tháng.
- Đối chứng 512 đạt MAE `6,18479`, RMSE `8,48647` tháng.
- Delta MAE 768−512 `-0,00137`, paired bootstrap 95% CI `[-0,19070; +0,19213]`; cải thiện không đạt ngưỡng `0,10` tháng.
- Khóa input 512 cho final 5-fold vì hiệu năng MAE hòa nhưng chi phí 768 cao hơn; không thử 1024.
- Báo cáo: `p6_resolution/P6_768_vs_512_paired.json`.

### Phiên bản 2.8 – 2026-08-16

- Hoàn tất P5 trên ba seed 17/42/123; không sử dụng test set.
- D0 đạt MAE trung bình `6,26229`; D3 fused đạt `6,22508`, cải thiện trung bình `0,03721` tháng với bootstrap 95% CI `[-0,13521; +0,05863]`.
- D3 regression-only adaptive cải thiện trung bình `0,01932` tháng với CI `[-0,11270; +0,07069]`.
- Cả hai endpoint D3 không đạt ngưỡng thực tiễn `0,10` tháng; fused còn xấu hơn trung bình `0,22067` tháng ở nhóm tuổi 180–228.
- Khóa D0 ConvNeXt-Tiny direct regression làm cấu hình đi tiếp; mở P6 so sánh duy nhất 512 với 768.
- Báo cáo: `p5_seed_confirmation/P5_HANDOFF.md`; dữ liệu máy đọc: `p5_seed_confirmation/P5_AGGREGATE.json`.

### Phiên bản 2.7 – 2026-08-15

- D3 seed 17 early stop epoch 19; best epoch 11, fused MAE `6,24066` tháng.
- So với D0-17, fused delta `-0,07884`, CI `[-0,21105; +0,05364]`; regression-only adaptive delta `-0,02426`, CI `[-0,16177; +0,11447]`.
- Cặp seed 17 đều hợp lệ; D3 có lợi số học nhưng chưa đạt ngưỡng 0,10 và CI chứa 0.
- D0 seed 123 preflight/smoke/resume PASS; bắt đầu run `P5_D0_CONVNEXT_TINY_SEED123`.

### Phiên bản 2.6 – 2026-08-15

- D0 seed 17 early stop epoch 19; best epoch 11, MAE `6,31950` tháng; run hợp lệ, không lỗi đỏ.
- D3 seed 17 preflight/smoke/resume PASS và đã bắt đầu run `P5_D3_CONVNEXT_TINY_LDL_SEED17`.

### Phiên bản 2.5 – 2026-08-15

- Khóa P5 với seed `17/42/123`, top-2 D0 và D3, cùng official validation 1.425 ảnh.
- Primary giữ D3 fused của P4; regression-only là endpoint secondary adaptive được khai báo trước khi chạy seed mới.
- Khóa quy tắc quyết định: mean improvement ít nhất 0,10 tháng, có lợi ở tối thiểu 2/3 seed và ổn định theo cluster bootstrap/subgroup.
- Tái sử dụng cặp seed 42; cần bốn run mới theo thứ tự D0-17, D3-17, D0-123, D3-123.
- Preflight/smoke/resume D0-17 PASS; bắt đầu run `P5_D0_CONVNEXT_TINY_SEED17`.

### Phiên bản 2.4 – 2026-08-15

- D3 early stop tại epoch 21; best epoch 13, primary fused MAE `6,14541`, RMSE `8,42811` tháng.
- Delta MAE D3−D0 `-0,03938` tháng; paired bootstrap 95% CI `[-0,19707; +0,12156]` trên 10.000 lần lấy mẫu.
- Cải thiện nhỏ hơn ngưỡng thực tiễn 0,10 tháng và CI chứa 0; chưa tuyên bố D3 vượt D0.
- Regression-only exploratory đạt MAE `6,12496`, nhưng delta `-0,05984` và CI `[-0,21211; +0,09625]` vẫn chưa đủ bằng chứng.
- Đóng P4: loại D1 và D2; chọn D0 cùng D3 làm top-2 sang P5 xác nhận tổng cộng ba seed.

### Phiên bản 2.3 – 2026-08-15

- Triển khai D3 trên D0 với 229 lớp tuổi tháng, Gaussian sigma 2 tháng và loss `SmoothL1 + 0,2 × soft cross-entropy`.
- Khóa primary inference trước khi xem validation: trung bình 0,5 regression và 0,5 kỳ vọng phân phối.
- Prediction artifact lưu riêng hai đầu ra để chẩn đoán cơ chế nhưng không dùng để đổi primary rule hậu nghiệm.
- Unit test PASS 14/14, preflight PASS 7/7, smoke/resume PASS; bắt đầu run `P4_D3_CONVNEXT_TINY_LDL_SIGMA2_LAMBDA02_SEED42`.

### Phiên bản 2.2 – 2026-08-15

- D2 early stop tại epoch 22; best epoch 14, MAE `6,29697`, RMSE `8,49262` tháng.
- Delta MAE D2−D0 `+0,11218` tháng; paired bootstrap 95% CI `[-0,05414; +0,27870]` trên 10.000 lần lấy mẫu.
- CI chứa 0 nên không kết luận multi-scale gây hại có ý nghĩa thống kê; tuy nhiên D2 không cải thiện primary endpoint và bị loại theo tiêu chí đã khóa.
- D3 không xây trên D2; D3 dùng D0 ConvNeXt-Tiny direct-regression rồi chỉ thêm label-distribution head.

### Phiên bản 2.1 – 2026-08-15

- Triển khai D2 lấy đặc trưng bốn stage ConvNeXt-Tiny: `96 + 192 + 384 + 768` chiều và fusion về 768 chiều.
- Fusion được khởi tạo sao chép stage cuối của D0; ba stage sớm bắt đầu với đóng góp bằng 0 để giảm rủi ro phá baseline.
- Thêm cảnh báo collapse theo từng giới nhằm phát hiện kiểu thất bại đã bị độ lệch chuẩn tổng che giấu ở D1.
- Unit test PASS 12/12, preflight PASS 7/7, smoke/resume PASS; bắt đầu run `P4_D2_CONVNEXT_TINY_MULTISCALE_SEED42`.

### Phiên bản 2.0 – 2026-08-15

- D1 early stop tại epoch 13; best epoch 5, MAE `31,96281`, RMSE `42,43807` tháng.
- So với D0, delta MAE D1−D0 `+25,77802` tháng; paired bootstrap 95% CI `[+24,33264; +27,26304]` trên 10.000 lần lấy mẫu.
- Chẩn đoán xác nhận feature collapse: mean pair distance của đặc trưng giảm từ `8,30168` ở pretrained backbone xuống `0,000630` ở best checkpoint; đầu ra gần như chỉ dựa vào giới tính.
- Loại D1/ConvNeXtV2-FCMAE khỏi pipeline chính trong recipe đã khóa; không khái quát kết luận sang mọi chiến lược fine-tune ConvNeXtV2.
- D2 dùng backbone thắng là ConvNeXt-Tiny của D0, sau đó chỉ thêm multi-scale fusion.

### Phiên bản 1.9 – 2026-08-15

- Khóa D0 là A2 đã có, không train lại: validation MAE `6,18479`, RMSE `8,48647` tháng.
- D1 chỉ đổi backbone sang `convnextv2_tiny.fcmae_ft_in1k`; dùng pretrained ImageNet-1K để kiểm soát nguồn dữ liệu pretraining so với D0.
- Preflight PASS 7/7, unit test PASS 11/11 và resume smoke test PASS trên RTX 4050 6 GB với BF16.
- Cấu hình micro-batch `12×3` bị tràn sang shared GPU memory và chỉ đạt `2,41 ảnh/s`; chuyển sang `6×6` nhưng giữ effective batch 36.
- Bắt đầu run chính thức `P4_D1_CONVNEXTV2_TINY_FCMAE_IN1K_B6A6_SEED42`; chưa chạy D2/D3 trước paired comparison D1–D0.

### Phiên bản 1.8 – 2026-08-15

- B1 early stop hợp lệ sau 22 epoch; best epoch 14, MAE 6,23894, RMSE 8,44963.
- B0/A2 có MAE 6,18479; delta B1−B0 `+0,05414`, paired bootstrap 95% CI `[-0,10167; +0,21180]`.
- B1 không chứng minh cải thiện primary endpoint; khóa `preprocessing=none` cho P4.
- Không tuyên bố masking gây hại có ý nghĩa thống kê vì CI chứa 0; giữ B1 làm robustness/phân tích phụ.
- D0 ở P4 tái sử dụng B0/A2, không train lại; bước huấn luyện mới kế tiếp là D1 ConvNeXtV2-Tiny.

### Phiên bản 1.7 – 2026-08-14

- Loại prototype threshold cổ điển trước khi train vì fallback 37,5% sau khi loại các mask nền giả.
- Tải và xác minh gói mask RSNA chính thức: 306.332.153 byte, MD5 `a692de2799d99fd4bfec8cd535cd7979`.
- Trên 528 mask thủ công, `eff_unet` đạt mean Dice 0,98878 và `Tensormask` đạt 0,98810 ở kích thước audit 512; khóa `eff_unet` làm nguồn chính.
- Audit/cache đủ 12.611 train + 1.425 validation: 14.013 `eff_unet`, 22 `Tensormask`, một raw fallback ID 3100; không mask rỗng hoặc sai kích thước.
- Khóa B1 là masking + zero background, không crop/rotate; B0 tái sử dụng A2, không train lại.
- Preflight, 10 unit test và checkpoint/resume hai optimizer step đều PASS; chưa xem MAE validation B1 và chưa dùng test.

### Phiên bản 1.6 – 2026-08-14

- Hoàn tất A0 không augmentation: validation MAE 6,640, RMSE 8,884.
- Hoàn tất A1 flip p=0,5: MAE 6,514; cải thiện 0,125 tháng nhưng paired CI còn chứa 0.
- Hoàn tất A2 augmentation nhẹ có flip: MAE 6,185, RMSE 8,486.
- A2 tốt hơn A0 0,455 tháng, paired 95% CI [−0,658; −0,249].
- A2 tốt hơn A1 0,329 tháng, paired 95% CI [−0,509; −0,152].
- Khóa augmentation A2 cho P3–P4; chưa dùng test ground truth hoặc test metric.
- Đóng P2 với trạng thái PASS; cho phép bắt đầu P3.

### Phiên bản 1.5 – 2026-08-13

- Xây baseline A0 ConvNeXt-Tiny pretrained, input 512, sex embedding và direct regression; chưa chạy huấn luyện dài.
- Trainer chỉ nhận manifest train/validation đã khóa, không chứa đường dẫn test ground truth.
- Bổ sung checkpoint nguyên tử theo epoch, 500 optimizer step hoặc tối đa 20 phút; lưu đầy đủ RNG, optimizer, scheduler, scaler, vị trí trong epoch và hash.
- Xác minh stop/resume cho model và training state giống bitwise với run không bị ngắt.
- Phát hiện FP16 gradient Inf ở smoke test đầu; chuyển AMP auto sang BF16 trên RTX 4050 và FP16 init scale 4096 khi cần fallback.
- GPU smoke ConvNeXt-Tiny 512 batch 4 đạt với peak reserved 1.570 MiB và không cảnh báo.
- Đóng P1 với trạng thái PASS; cho phép bắt đầu P2.

### Phiên bản 1.4 – 2026-08-13

- Bổ sung và audit bộ validation chính thức gồm 1.425 ảnh.
- Xác nhận ID, tuổi và giới tính validation khớp hoàn toàn annotation Deeplasia.
- Xác nhận không có ID hoặc SHA-256 giao nhau giữa train, validation và test.
- Khóa manifest/fingerprint của ba split và development pool 14.036 ảnh.
- Đóng P0 với trạng thái PASS; cho phép bắt đầu P1.

### Phiên bản 1.3 – 2026-08-13

- Bắt đầu P0 và tạo manifest/hash cho 12.611 train cùng 200 test bị khóa nhãn.
- Xác nhận train/test hiện có không thiếu ảnh, không ảnh hỏng và không có exact duplicate/leakage theo SHA-256.
- Xác nhận môi trường RTX 4050 chạy được synthetic forward/backward ở 512×512, batch 1.
- Ghi nhận blocker: data root chưa có bộ validation chính thức 1.425 ảnh.
- Chưa đánh dấu P0 hoàn thành và chưa chuyển P1.

### Phiên bản 1.2 – 2026-08-13

- Bổ sung checkpoint phục hồi đầy đủ theo epoch, step/thời gian và best validation MAE.
- Bổ sung kiểm tra hash khi resume và quy tắc lưu checkpoint an toàn.
- Bổ sung log theo batch/epoch, file cảnh báo và `TRAINING REVIEW SNAPSHOT`.
- Bổ sung các cờ bất ổn NaN/Inf, gradient, overfitting, prediction collapse, subgroup và OOM.
- Bắt buộc kiểm thử stop/resume trước mọi run dài.

### Phiên bản 1.1 – 2026-08-12

- Chuyển toàn bộ nội dung sang tiếng Việt có dấu chuẩn UTF-8 để dễ đọc.
- Giữ nguyên protocol và các quyết định khoa học của phiên bản 1.0.

### Phiên bản 1.0 – 2026-08-12

- Tạo kế hoạch tổng thể.
- Khóa protocol train/validation/test và nguyên tắc chống leakage.
- Chốt chuỗi A0–A2, B0–B1, D0–D3, seed confirmation và final 5-fold.
- Thêm Artifact Invariance Test cho 200 cặp ảnh gốc/artifact-reduced.
- Ghi nhận kết quả sơ bộ của pipeline khác và các metadata còn thiếu.
