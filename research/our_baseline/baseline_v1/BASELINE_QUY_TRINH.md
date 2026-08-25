# Quy trình baseline dự đoán tuổi xương RSNA

## 1. Mục tiêu và nguyên tắc chung

Mục tiêu của quy trình là xây dựng một hệ thống dự đoán tuổi xương theo tháng từ
ảnh X-quang bàn tay, sau đó từng bước cải thiện MAE hướng tới mốc **dưới 3.5 tháng**.

Baseline được thiết kế theo nguyên tắc:

1. Mỗi thay đổi phải có một baseline đối chứng rõ ràng.
2. Không sử dụng nhãn của 200 ảnh RSNA test để chọn mô hình, loss, preprocessing,
   checkpoint hoặc trọng số ensemble.
3. Mọi quyết định chính dựa trên development pool gồm 14.036 ảnh và 5-fold OOF.
4. Không gộp nhiều kỹ thuật mới vào cùng một lần chạy nếu chưa biết từng kỹ thuật
   đóng góp gì.
5. Mỗi run phải lưu config, seed, manifest dữ liệu, checkpoint, log và báo cáo MAE.

Các bài tham khảo chính cho thiết kế gồm Bram 2025, Deeplasia 2024, MMANet 2023,
BoNet+ 2026 và mô hình uncertainty-aware 2026. Các kết quả MAE trong bài báo chỉ
dùng làm mốc tham khảo; không coi là so sánh trực tiếp nếu protocol khác nhau.

---

## 2. Sơ đồ tổng quát

```text
Audit dữ liệu
    ↓
Chuẩn hóa CSV và đường dẫn ảnh
    ↓
Preprocessing / augmentation
    ↓
ConvNeXt-Tiny + sex embedding
    ↓
Smooth L1 direct regression
    ↓
Official train/validation smoke
    ↓
5-fold OOF trên 14.036 ảnh
    ↓
Phân tích MAE, RMSE, subgroup, CI và residual
    ↓
So sánh p7_reference với bram_lite
    ↓
Chọn candidate bằng OOF
    ↓
Mở rộng global-local / uncertainty / expectation head
```

---

## 3. Giai đoạn 0 — Khóa dữ liệu và protocol

### Việc cần làm

- Xác nhận đúng ba phần dữ liệu:

  | Phần | Số ảnh | Vai trò |
  |---|---:|---|
  | Official train | 12.611 | Huấn luyện |
  | Official validation | 1.425 | Validation chính thức |
  | RSNA test | 200 | Chỉ đánh giá cuối |

- Gộp train và validation thành development pool 14.036 ảnh cho 5-fold OOF.
- Chuẩn hóa các tên cột:
  - `id` hoặc `Image ID` → `id`;
  - `boneage` hoặc `Bone Age (months)` → `boneage`;
  - `male` → biến sex nhị phân.
- Kiểm tra mỗi ID có đúng một ảnh.
- Tạo SHA/hash cho CSV và manifest ID.

### Tác dụng

Ngăn leakage, nhầm đường dẫn, mất ảnh hoặc đánh giá sai do trộn hai loại split.
Đây là điều kiện để MAE có ý nghĩa và có thể tái lập.

### Đóng góp cho mục tiêu MAE < 3.5

Không trực tiếp giảm MAE, nhưng bảo đảm mọi cải thiện sau đó là cải thiện thật,
không phải do lỗi protocol.

### Điều kiện đạt

- Train: 12.611 ảnh.
- Validation: 1.425 ảnh.
- Development pool: 14.036 ID duy nhất.
- Không đọc `rsna_test.csv` trong quá trình chọn mô hình.

---

## 4. Giai đoạn 1 — Thiết lập môi trường tính toán

### Việc cần làm

- Ưu tiên GPU NVIDIA có CUDA.
- Kiểm tra:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- Trên GPU 4GB, bắt đầu với batch size 1 hoặc 2 và dùng gradient accumulation.
- Dùng BF16 autocast nếu GPU hỗ trợ.
- Kiểm tra checkpoint có thể lưu và resume sau khi tiến trình bị gián đoạn.

### Tác dụng

ConvNeXt-Tiny ở 512×512 trên CPU rất chậm. GPU không làm thay đổi thuật toán,
nhưng giúp chạy đủ epoch, nhiều fold và nhiều seed trong thời gian thực tế.

### Đóng góp cho mục tiêu

Cho phép thực hiện 5-fold OOF và lặp 3 seed. Đây là điều cần thiết để phân biệt
một cải thiện bền vững với một kết quả ngẫu nhiên.

### Điều kiện đạt

- `torch.cuda.is_available()` là `True` khi chạy thực nghiệm chính.
- Smoke forward và smoke train không có NaN/Inf/OOM.

---

## 5. Giai đoạn 2 — Chuẩn hóa ảnh đầu vào

### Baseline `p7_reference`

- Đọc ảnh grayscale.
- Resize trực tiếp về 512×512.
- Lặp kênh grayscale thành RGB để dùng pretrained ImageNet.
- Normalize theo ImageNet:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Candidate `bram_lite`

- Tách vùng foreground dựa trên ngưỡng pixel không màu nền.
- Giữ tỷ lệ hình ảnh bằng letterbox thay vì kéo méo hình.
- Dùng autocontrast nhẹ.
- Sau đó resize về 512×512.

### Candidate `a2_light_flip`

- Giữ nguyên preprocessing `p7_reference`.
- Chỉ thêm `RandomHorizontalFlip(p=0.5)` trong lúc train.
- Không dùng affine, translation, scale hoặc ColorJitter trong recipe này.
- Validation và inference không dùng flip ngẫu nhiên; TTA flip sẽ được đánh giá ở giai đoạn riêng.

Mục tiêu của recipe là kiểm tra độc lập giả thuyết light horizontal flip, không trộn
đồng thời nhiều augmentation nên có thể quy nguyên nhân nếu MAE thay đổi.

### Tác dụng

Giảm sự khác biệt do biên đen, khoảng trống nền, kích thước ảnh và contrast giữa
các máy chụp. Mục tiêu là để mô hình tập trung vào hình thái xương thay vì nền ảnh.

### Đóng góp kỳ vọng

Đây là nhóm thay đổi có khả năng cải thiện trực tiếp MAE vì các bài Bram và Deeplasia
đều cho thấy preprocessing có ảnh hưởng lớn. Tuy nhiên, mask xương đầy đủ không được
dùng mặc định vì nhánh mask trước đây chưa chứng minh cải thiện.

### Điều kiện đạt

- Kiểm tra trực quan ít nhất 24–50 ảnh.
- Không làm mất đầu ngón tay, cổ tay hoặc vùng carpal.
- So sánh `p7_reference` và `bram_lite` bằng cùng seed/fold.

---

## 6. Giai đoạn 3 — Augmentation A2

### Việc cần làm

Trong lúc train, áp dụng các biến đổi nhẹ:

- affine rotation khoảng ±7°;
- translation khoảng 3%;
- scale nhẹ khoảng 0.97–1.03;
- brightness/contrast nhẹ.

Validation và inference không dùng augmentation ngẫu nhiên.

Recipe `p7_reference` giữ affine/brightness/contrast như protocol hiện tại.
Recipe `a2_light_flip` là ablation riêng cho horizontal flip; không gộp hai nhóm
biến đổi trong cùng một kết luận. Sau khi chạy `a2_light_flip`, chỉ giữ flip nếu
MAE giảm trên cùng official split và không làm xấu subgroup rõ rệt.

### Tác dụng

Giúp mô hình bớt phụ thuộc vào vị trí, góc chụp và cường độ ảnh cụ thể. Nó cũng
giảm overfitting vào các đặc điểm không liên quan đến độ trưởng thành xương.

### Đóng góp kỳ vọng

Giảm variance và cải thiện khả năng tổng quát. A2 đã từng tốt hơn augmentation
đơn giản trong các thử nghiệm trước, nên được khóa làm augmentation chính.

### Điều kiện đạt

- Phân phối tuổi sau augmentation không thay đổi.
- Không tạo ảnh phi lâm sàng quá mức.
- Chạy cùng preprocessing với validation để tránh train/validation mismatch.

---

## 7. Giai đoạn 4 — Kiến trúc mô hình baseline

### Kiến trúc

```text
Ảnh 3 kênh 512×512
        ↓
ConvNeXt-Tiny pretrained ImageNet
        ↓
Global pooled image feature
        ↓
Sex embedding: 1 → 32
        ↓
Concatenate image feature + sex feature
        ↓
LayerNorm → Linear 256 → GELU → Dropout → Linear 1
        ↓
Bone age theo tháng
```

### Tác dụng

- ConvNeXt-Tiny học các đặc trưng hình thái xương ở nhiều cấp độ.
- Sex embedding mô hình hóa khác biệt sinh học giữa nam và nữ.
- Direct regression cho phép dự đoán liên tục theo tháng, không bị giới hạn ở
  các mốc tuổi rời rạc.

### Đóng góp kỳ vọng

Đây là backbone baseline để đo mọi cải tiến. Bản thân kiến trúc đã gần với hướng
Bram và đủ mạnh để xác định preprocessing, loss hoặc ensemble có tạo ra cải thiện
hay không.

### Điều kiện đạt

- Forward output có kích thước `[batch]`.
- Không có NaN/Inf.
- Checkpoint load lại cho kết quả nhất quán.

---

## 8. Giai đoạn 5 — Loss và tối ưu hóa

### Cấu hình baseline

| Thành phần | Cấu hình |
|---|---|
| Loss | Smooth L1 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| Scheduler | Warm-up 3 epoch + cosine decay |
| Epoch tối đa | 100 |
| Early stopping | Patience 15 |
| Gradient clipping | 5.0 |

### Tác dụng

Smooth L1 ít nhạy với các nhãn lệch lớn hơn MSE. AdamW và cosine decay giúp
fine-tune pretrained backbone ổn định hơn. Early stopping hạn chế overfit.

### Đóng góp kỳ vọng

Giảm các outlier lớn và giúp mô hình học ổn định. Giai đoạn này chủ yếu làm nền
cho các cải tiến sau, không nên tune quá rộng trước khi có baseline sạch.

### Điều kiện đạt

- Training log có train loss, train MAE, validation MAE và learning rate.
- Lưu best checkpoint theo validation MAE.
- Có resume checkpoint sau mỗi epoch.

---

## 9. Giai đoạn 6 — Official smoke và kiểm tra vận hành

### Việc cần làm

Chạy trước với số lượng nhỏ:

- 1 epoch;
- 64×64 hoặc 128×128;
- vài trăm ảnh;
- không dùng để kết luận hiệu năng.

Kiểm tra:

- loader đọc đúng ảnh;
- sex và target khớp ID;
- model forward/backward được;
- checkpoint và resume hoạt động;
- không có lỗi OOM, NaN hoặc đường dẫn.

### Tác dụng

Phát hiện lỗi code trước khi tiêu tốn nhiều giờ GPU.

### Đóng góp cho mục tiêu

Không giảm MAE trực tiếp, nhưng tránh mất toàn bộ một lần chạy dài do lỗi kỹ thuật.

---

## 10. Giai đoạn 7 — Official train/validation

### Việc cần làm

Chạy một mô hình trên 12.611 train và đánh giá trên 1.425 official validation.

Mục đích của bước này là:

- kiểm tra tốc độ thực tế;
- kiểm tra best epoch;
- kiểm tra độ ổn định loss;
- xác định batch size và accumulation phù hợp;
- phát hiện preprocessing làm ảnh hỏng.

### Tác dụng

Đây là vòng đánh giá nhanh trước khi chạy 5-fold OOF.

### Đóng góp kỳ vọng

Cho biết candidate có tiềm năng vượt baseline nội bộ hay không. Tuy nhiên, không
dùng kết quả một lần chạy để khẳng định vượt bài báo.

### Điều kiện chuyển bước

- Không có subgroup sụp đổ rõ ràng.
- Validation curve không có dấu hiệu overfit nghiêm trọng.
- Candidate đáng chạy OOF nếu MAE tốt hơn hoặc có lý do phương pháp rõ ràng.

---

## 11. Giai đoạn 8 — 5-fold OOF chính thức

### Việc cần làm

- Gộp 14.036 ảnh train + validation thành development pool.
- Chia 5 fold stratified theo sex và nhóm tuổi.
- Mỗi ảnh chỉ xuất hiện một lần trong OOF validation.
- Huấn luyện 5 model độc lập.
- Ghép các dự đoán thành `oof_predictions.csv`.

### Chỉ số cần ghi

- Pooled MAE.
- RMSE.
- Median absolute error.
- Accuracy trong ±6, ±12, ±18 tháng.
- Bootstrap 95% CI của MAE.
- MAE theo sex.
- MAE theo nhóm tuổi.
- Bias trung bình theo tuổi.
- MAE từng fold và độ lệch giữa các fold.

### Tác dụng

OOF cung cấp đánh giá ổn định hơn một validation nhỏ. Nó cũng tạo prediction
đồng nhất để tính residual correlation và chọn ensemble.

### Đóng góp cho mục tiêu

Đây là tiêu chí chính để quyết định một thay đổi có thật sự đưa mô hình gần MAE
3.5 hay không.

### Điều kiện đạt

- 14.036 dòng OOF.
- ID duy nhất.
- Không có NaN/Inf.
- CI và subgroup metrics được lưu.
- Không đọc nhãn test.

---

## 12. Giai đoạn 9 — So sánh hai baseline recipe

### Run A: `p7_reference`

Mục tiêu là tái lập mốc nội bộ P7 với cùng backbone, augmentation và direct
regression.

Mốc tham khảo hiện có:

```text
P7 OOF MAE = 6.316691 tháng
```

### Run B: `bram_lite`

Chỉ thay preprocessing, giữ nguyên:

- seed;
- fold assignment;
- backbone;
- loss;
- optimizer;
- epoch budget;
- evaluation code.

### Run C: `a2_light_flip`

Chỉ thay augmentation bằng horizontal flip, giữ nguyên preprocessing, model, loss,
optimizer và split. Đây là thí nghiệm ưu tiên trước khi thử kiến trúc mới.

### Quy tắc kết luận

Giữ `bram_lite` chỉ khi:

1. Pooled OOF MAE giảm rõ ràng.
2. Cải thiện không chỉ đến từ một fold.
3. Không có subgroup bị xấu nghiêm trọng.
4. Kết quả có thể lặp lại ở ít nhất 3 seed nếu mức cải thiện đủ lớn.

Không mở RSNA test chỉ vì một candidate có validation tốt hơn.

---

## 13. Giai đoạn 10 — Ensemble OOF có kiểm soát

### Việc cần làm

- Bắt đầu bằng equal-weight mean giữa các fold/seed.
- Tính MAE OOF của từng model.
- Tính tương quan giữa residual của các model.
- Chỉ dùng model có lỗi bổ sung cho nhau.
- Nếu học trọng số, chỉ học trọng số từ OOF và khóa công thức trước khi chạy test.

### Tác dụng

Ensemble giúp giảm variance khi các model có lỗi khác nhau. Tuy nhiên, ensemble
các model có prediction tương quan gần như hoàn toàn sẽ không đem lại nhiều lợi ích.

### Đóng góp kỳ vọng

Có thể giảm thêm khoảng cách giữa single model và mốc MAE 3.5, đặc biệt khi kết hợp
khác seed, preprocessing hoặc resolution.

### Điều kiện đạt

- Ensemble OOF tốt hơn model tốt nhất.
- Trọng số không được tối ưu trên 200 ảnh test.
- Báo cáo riêng single model và ensemble.

---

## 14. Giai đoạn 11 — Mở rộng global-local

Chỉ triển khai sau khi baseline sạch đã có kết quả OOF.

### Thiết kế

```text
Global stream: toàn bàn tay → ConvNeXt/Transformer
Local stream: patch hoặc vùng ROI → attention/MIL
Global feature + local feature + sex embedding
        ↓
Fusion head → bone age
```

### Tác dụng

Global stream học mức độ trưởng thành tổng thể. Local stream tập trung vào carpal,
epiphysis, phalanges và metacarpals. Đây là hướng phù hợp với BoNet+ và MMANet.

### Đóng góp kỳ vọng

Có khả năng giảm lỗi ở nhóm tuổi 8–15, nơi thay đổi hình thái tinh tế và là vùng
khó nhất của bài toán.

### Cảnh báo

Không dùng STN học vị trí ngay từ đầu nếu chưa có cơ chế chống collapse. Patch/MIL
attention hoặc crop cố định có kiểm soát an toàn hơn cho thử nghiệm đầu tiên.

---

## 15. Giai đoạn 12 — Uncertainty và expectation head

### Uncertainty head

Mô hình dự đoán đồng thời:

- tuổi xương trung bình;
- độ bất định theo từng ảnh.

Ảnh có độ bất định cao có thể được giảm trọng số trong ensemble hoặc chuyển sang
chế độ cần chuyên gia xem lại.

### Expectation/distribution head

Thay vì chỉ dự đoán một số tháng, mô hình học phân phối tuổi rồi lấy expectation.
Điều này tận dụng tính thứ tự của tuổi xương và có thể giảm ảnh hưởng của nhãn
không chính xác tuyệt đối.

### Đóng góp kỳ vọng

Hai hướng này có thể giảm outlier và làm dự đoán ổn định hơn, nhưng phải được so
sánh với direct regression trên cùng folds.

---

## 16. Giai đoạn 13 — Robustness và bias audit

### Kiểm tra robustness

Đánh giá lại trên các biến đổi:

- brightness;
- contrast;
- rotation nhỏ;
- resolution thấp;
- marker/lề ảnh;
- crop nhẹ.

### Kiểm tra bias

Báo cáo riêng theo:

- male/female;
- nhóm tuổi;
- khoảng tuổi xương;
- ảnh có chất lượng hoặc background khác biệt.

### Tác dụng

Một mô hình có MAE trung bình thấp nhưng sai nghiêm trọng ở một subgroup vẫn chưa
đủ tin cậy.

### Đóng góp cho mục tiêu

Giúp tránh việc đạt MAE thấp do chỉ tối ưu nhóm dữ liệu đông nhất, đồng thời xác
định nhóm còn gây ra phần lớn sai số để ưu tiên cải tiến.

---

## 17. Giai đoạn 14 — Đánh giá cuối và báo cáo

Chỉ sau khi khóa recipe, seed, checkpoint policy, preprocessing và ensemble formula
mới chạy đánh giá cuối.

Báo cáo cần có:

- cấu hình đầy đủ;
- data manifest/hash;
- số fold và seed;
- MAE/RMSE/median AE;
- CI 95%;
- subgroup metrics;
- residual plot và calibration/bias;
- số ảnh lỗi lớn;
- so sánh với P7, Bram và Deeplasia với ghi chú protocol;
- giới hạn do RSNA test chỉ có 200 ảnh và đã từng được mở trong P8.

Không dùng cụm “vượt bài báo” nếu chưa có protocol tương thích hoặc external
hold-out chưa từng được chạm.

---

## 18. Thứ tự chạy thực tế

```text
1. Smoke CPU/GPU
2. Official p7_reference, 1 seed
3. Official bram_lite, 1 seed
4. OOF p7_reference, 5 fold
5. OOF bram_lite, 5 fold
6. So sánh pooled MAE và subgroup
7. Lặp candidate tốt hơn ở 3 seed
8. Chọn equal-weight ensemble bằng OOF
9. Thử global-local
10. Thử uncertainty hoặc expectation head
11. Robustness/bias audit
12. Khóa protocol và đánh giá cuối
```

## 19. Tiêu chí thành công theo tầng

| Tầng | Tiêu chí |
|---|---|
| Kỹ thuật | Pipeline chạy ổn định, không NaN/OOM, resume được |
| Baseline | Tái lập được P7 reference trên cùng protocol |
| Cải thiện 1 | Candidate giảm OOF MAE rõ ràng so với baseline |
| Cải thiện 2 | Kết quả lặp lại ở nhiều seed/fold |
| Mục tiêu | OOF/external MAE tiến tới hoặc dưới 3.5 tháng |
| Báo cáo | Có CI, subgroup, robustness và giới hạn rõ ràng |

## 20. File triển khai

- Script chính: `project/baseline_v1/boneage_baseline.py`
- Hướng dẫn chạy: `project/baseline_v1/README.md`
- Output mặc định: `project/baseline_v1/outputs/`
- Hồ sơ trạng thái chung: `word/AI_Context/AI_Context/`
