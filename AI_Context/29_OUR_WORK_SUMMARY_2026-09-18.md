# Tổng hợp hướng nghiên cứu, cải tiến và kết quả hiện có

**Ngày cập nhật:** 2026-09-18
**Phạm vi:** toàn bộ các hướng đã triển khai hoặc đã được lên kế hoạch trong nhánh nghiên cứu của người dùng, đối chiếu với các báo cáo đã lưu trong repository chung.
**Đơn vị:** tháng tuổi xương; MAE càng thấp càng tốt.

> Tài liệu này là bản bàn giao ngắn gọn để tiếp tục công việc. Các số liệu được gắn trạng thái rõ ràng: `VERIFIED` là có artifact/report để kiểm tra; `PARTIAL` là mới có một phần fold hoặc pilot; `EXPLORATORY` là đã đọc test-200 nên không được dùng để chọn mô hình; `PLANNED` là chưa train hoặc chưa có kết quả.

## 1. Mục tiêu và nguyên tắc chung

Mục tiêu cuối là xây dựng mô hình dự đoán tuổi xương từ ảnh X-quang bàn tay có thể tái lập, giảm phụ thuộc vào artifact/marker/viền ảnh, tận dụng thông tin giới tính và vùng giải phẫu, rồi so sánh công bằng với các mốc trong bài báo.

Các mốc tham khảo chính:

- Deeplasia, Pediatric Radiology 2024: MAE/MAD công bố trên RSNA test-200 khoảng **3,87 tháng**.
- Bram et al. 2025: MAE công bố trên RSNA test-200 khoảng **3,68 tháng**.
- Shu & Yu 2025: MAE khoảng **4,42 tháng**, nhưng có thêm 8 đặc trưng kích thước xương được đo.
- Zhang et al. 2026: MAE **4,10 tháng trên RSNA validation 1.425**, không đặt ngang trực tiếp với test-200.

Mục tiêu thực nghiệm đang theo đuổi là cải thiện kết quả của nhóm trên test-200 xuống quanh hoặc dưới **4,2 tháng**, nhưng không được dùng nhãn test-200 để chọn kiến trúc, checkpoint, TTA, trọng số ensemble hoặc hyperparameter.

## 2. Dữ liệu và giao thức đã khóa

| Thành phần | Quy mô/vai trò | Ghi chú |
|---|---:|---|
| RSNA train chính thức | 12.611 | Dùng để fit model |
| RSNA validation chính thức | 1.425 | Development; dùng để chọn trong giai đoạn tái lập |
| Development gộp | 14.036 | Dùng cho 5-fold OOF; mỗi ảnh chỉ được dự đoán bởi fold không train ảnh đó |
| RSNA test | 200 | Đã truy cập nhiều lần; chỉ còn exploratory/robustness |
| Artifact-reduced test | 200 cặp gốc/sạch | Chỉ dùng Artifact Invariance Test, không dùng train |

Các quy tắc quan trọng:

1. Manifest phải khóa ID, target, sex, fold và hash.
2. Kết quả OOF phải đủ đúng 14.036 ID, mỗi ID đúng một lần, fold 1–5 và không có giá trị NaN/Inf.
3. Không dùng ảnh test hoặc ảnh test đã làm sạch để huấn luyện.
4. Khi so sánh hai hướng, dùng cùng ID, cùng fold, paired delta và bootstrap CI.
5. Checkpoint cần lưu `last`, `best`, optimizer, scheduler, scaler, RNG, early-stop counter, config và provenance để resume không làm mất trạng thái.

## 3. Các vấn đề môi trường và dữ liệu đã xử lý

### 3.1. Colab/Kaggle và resume

Đã xử lý nhiều lỗi khác nhau giữa môi trường local, Colab và Kaggle:

- xung đột NumPy/OpenCV/Pandas;
- `pkg_resources` thiếu khi dùng PyTorch cũ;
- Lightning/jsonargparse phiên bản mới không tương thích với code Deeplasia cũ;
- Albumentations mới đổi API `RandomResizedCrop` và `CLAHE`;
- Kaggle `/kaggle/input` chỉ đọc nên log phải chuyển sang `/kaggle/working`;
- checkpoint PyTorch mới thay đổi mặc định `weights_only`;
- resume sai do nhầm `split_0`, nhầm candidate hoặc dùng đường dẫn code/data chưa tồn tại;
- copy ảnh/mask từ Drive quá chậm hoặc bị ngắt kết nối.

Giao thức đã rút ra: code và dữ liệu chạy từ `/content` hoặc `/kaggle/working`, checkpoint/log ghi trực tiếp lên Drive hoặc output của Kaggle; mask/data lớn nên nén ZIP rồi giải nén cục bộ; trước train phải chạy preflight, GPU pilot và resume smoke test.

### 3.2. Mask và ROI

- Mask chính thức đã được audit; `eff_unet` được chọn làm nguồn chính, `Tensormask` dùng bù các ID thiếu.
- Có một ngoại lệ mask đã biết là `3100.png`; không nên làm hỏng toàn bộ run vì một mask thiếu, phải ghi fallback rõ ràng.
- Việc giảm fallback hoặc cắt ROI không mặc nhiên làm MAE tốt hơn; cần kiểm tra hash ảnh, tỷ lệ foreground, kích thước và fallback rate trước khi train.
- Pilot C3-Z26 500 ảnh phát hiện `roi_whole` trùng byte-for-byte với ảnh gốc ở **500/500 cặp**, nên kết quả Original và ROI-whole giống nhau và **không có giá trị để kết luận ROI**.

## 4. Baseline và những cải tiến đã thực hiện

### 4.1. Tái lập Deeplasia/EfficientNet

Đã chuẩn bị và chạy các cấu hình EfficientNet theo hướng tái lập Deeplasia: EfficientNet-B0/B4, ảnh 512, sex embedding, regression head, checkpoint/resume và early stopping. Có các lần train model 1–6; model 5 từng bị upload nhầm code/config của model 4 nên phải đánh dấu **chưa train hợp lệ**.

Kết luận hiện tại:

- Có thể tái lập pipeline và checkpoint của tác giả ở mức code/data protocol.
- Các lỗi môi trường khiến một số run cũ không phản ánh hiệu năng model.
- Không dùng kết quả các run EfficientNet chưa có report OOF/test đầy đủ làm bằng chứng vượt baseline.
- Mốc Deeplasia 3,87 tháng là mốc của bài báo, không phải kết quả đã chứng minh lại hoàn toàn bởi run EfficientNet nội bộ.

### 4.2. E1/P7 — ConvNeXt-Tiny global baseline

Kiến trúc: ảnh toàn cảnh 512×512, ConvNeXt-Tiny pretrained, global average pooling, sex embedding và direct regression. Augmentation nhẹ được kiểm soát; không vertical flip.

| Run | Tập | MAE | Trạng thái |
|---|---:|---:|---|
| Friend P7 ensemble | OOF 14.036 | **6,316691** | `VERIFIED`, shared reference |
| EXP-006 P7 control | OOF 14.036 | **6,323629** | `VERIFIED` |
| P7 + TTA | OOF 14.036 | **6,296203** | `VERIFIED`; giữ làm inference improvement |
| P7 ensemble | Test-200 | **4,730321** | `EXPLORATORY` |
| EXP-006 P7 + TTA | Test-200 | **4,466886** | `EXPLORATORY` |

Bài học: ConvNeXt-Tiny là control mạnh và phù hợp làm mốc paired; tăng số epoch đơn thuần không giải quyết được overfitting validation.

### 4.3. TTA

TTA đã thử các góc xoay nhỏ và flip ngang. Với P7, TTA giảm OOF MAE từ 6,323629 xuống 6,296203. Với C3 ConvNeXtV2, TTA 10 biến thể gồm 5 góc `-10,-5,0,+5,+10` và hai trạng thái flip:

| C3 biến thể | OOF MAE | Delta/nhận xét |
|---|---:|---|
| Raw | 6,616847 | Control |
| TTA 10 biến thể | **6,559268** | Delta `-0,057579`; bootstrap CI `[-0,085732; -0,028821]` |
| Chỉ rotation | 6,604990 | Kém TTA đầy đủ |
| Chỉ flip variants | 6,584654 | Có lợi nhỏ hơn TTA đầy đủ |

TTA có tín hiệu tích cực trên OOF, nhưng test-200 không được dùng để chọn số view hoặc góc.

### 4.4. C3-ROI V1 và C3-R2

C3-ROI V1 dùng bbox bàn tay từ segmentation, margin 8%, fallback về ảnh toàn cảnh nếu segmentation thất bại. C3-R2 mở rộng margin 12% và border rescue để giảm fallback; Z26 thử resize/padding và có/không histogram equalization.

| Pipeline | OOF MAE | Test-200 MAE | Kết luận |
|---|---:|---:|---|
| C3-ROI V1, ConvNeXt-Tiny + sex embedding | 6,437349 | **4,337267** | `VERIFIED` OOF/test exploratory; tốt trên test nhưng standalone OOF kém P7 |
| C3-ROI V1 + TTA | — | **4,331841** | TTA giảm rất ít; CI OOF/test không đủ để khẳng định lợi ích riêng trên test |
| C3-R2 Z26, không equalization | 6,324301 | 4,665987 | Gần baseline OOF, kém ROI margin 8% trên test |
| C3-R2 Z26, có equalization | cao hơn bản NO_HE | — | Không promote |

Fallback là yếu tố lớn: C3-ROI V1 fallback khoảng 18,49% development và 33% test; C3-R2 giảm còn khoảng 14,06% development và 28% test nhưng nhóm fallback vẫn có MAE cao hơn nhóm mask/bbox. Vì vậy “ROI” hiện chưa phải một chuyên gia giải phẫu thuần túy; nhiều mẫu vẫn đi qua ảnh toàn cảnh.

### 4.5. C3 ConvNeXtV2-Base + bilinear pooling + uncertainty

Kiến trúc C3 đã thử:

```text
Ảnh 512×512
  → ConvNeXtV2-Base pretrained
  → 1×1 Conv C→512→128 + GELU
  → bilinear pooling 128×128 = 16.384 chiều
  → signed square-root + L2 normalization
  → ghép sex embedding
  → mean prediction + log-variance uncertainty
```

Kết quả OOF 5-fold:

- Raw: **6,616847** MAE; RMSE 8,945885.
- TTA 10 biến thể: **6,559268** MAE; RMSE 8,872458.
- TTA cải thiện đồng đều ở cả 5 fold, mạnh nhất ở fold 4.

Kết luận: bilinear pooling và uncertainty chưa làm C3 đơn lẻ vượt P7, nhưng C3 tạo diversity hữu ích cho ensemble.

### 4.6. Ensemble và calibrated stacking

Đã có hai lớp ensemble:

1. **Blend cố định:** P7/P7-TTA kết hợp C3 theo tỷ lệ thử nghiệm 50/50 hoặc 66/34.
2. **C4 calibrated stacking:** Ridge regression cross-fit, có thêm P7 raw, P7-TTA, C3 raw, uncertainty, độ lệch TTA và sex.

Các kết quả phải đọc kèm provenance vì C3 có nhiều checkpoint generation khác nhau:

| Ensemble | Tập | MAE | Trạng thái |
|---|---:|---:|---|
| P7 global TTA + C3 ROI TTA, 50/50 | OOF 14.036 | 6,117080 | `VERIFIED` theo registry C3-ROI |
| EXP-008 fixed 50/50 | OOF 14.036 | **6,120448** | `VERIFIED` trong research bundle |
| Blend reference 66% P7-TTA + 34% C3 raw | OOF 14.036 | khoảng **6,1758** | Control đang khóa cho kế hoạch V4/stacking |
| C4 calibrated stacking | OOF 14.036 | **6,181123** | `VERIFIED`; CI MAE `[6,088988; 6,273582]` |

Không được trộn các số 6,120448, 6,1758 và 6,181123 thành một kết quả duy nhất: chúng dùng các nguồn C3/checkpoint hoặc quy tắc khác nhau. Bước tiếp theo phải chuẩn hóa provenance về một bảng OOF duy nhất rồi chạy cross-fitted simplex stacking.

### 4.7. LDL và label-distribution

EXP-009 đã chuẩn bị cùng split 5-fold với ConvNeXt-Tiny, sigma khoảng 2, auxiliary label-distribution weight 0,2 và inference kết hợp regression/distribution. Mục tiêu gate được đặt trước là phải tốt hơn OOF blend hiện tại khoảng 6,120448.

Trạng thái: **đang/đã chuẩn bị nhưng chưa có kết quả cuối đủ để promote**. Không được lấy kết quả 2 fold hoặc test-200 làm kết luận 5-fold.

### 4.8. Anatomical ROI đa view

Hướng này dùng bảy view đồng bộ:

- một ảnh toàn bàn tay;
- sáu ROI theo vùng: wrist/carpal, thumb, index, middle, ring, little;
- MobileNetV3-Small dùng chung cho ROI;
- region embedding + attention pooling để mô hình biết vùng nào quan trọng;
- sex embedding hoặc Sex FiLM;
- mean regression + log-variance uncertainty.

Pilot 500 ảnh ban đầu không hợp lệ để so sánh vì `roi_whole` giống ảnh gốc 100%. Fold 3 còn bị prediction collapse và toàn pilot chỉ có 400/500 prediction hợp lệ. Đây là lỗi dữ liệu/pipeline, không phải bằng chứng rằng anatomical ROI kém.

### 4.9. V3-A/B/C — ConvNeXt-Tiny anatomical ROI Fold 1

Ba candidate đã được thiết kế theo single-variable ablation trên cùng Fold 1, cùng manifest và cùng pretrained weights:

| Candidate | Thay đổi | Fold 1 validation MAE | Trạng thái |
|---|---|---:|---|
| V3-A | baseline anatomical ROI, không train augmentation | khoảng **7,004163** | `VERIFIED` từ metadata V3-B |
| V3-B | bật train augmentation | **6,660061** | `VERIFIED` từ metadata V3-C |
| V3-C | V3-B + giảm uncertainty NLL weight xuống 0,05 | **6,609694** | `VERIFIED`; control Fold 1 cho V4 |

Một giá trị **7,023846** trong `metadata.json` là baseline comparator được ghi trong package, không nên nhầm với MAE V3-A thực tế `7,004163` được ghi ở metadata kế tiếp.

Kết luận: augmentation và giảm trọng số uncertainty có tín hiệu cải thiện trên Fold 1, nhưng chưa đủ để mở rộng 5-fold nếu chưa qua gate/spec tương ứng. Kết quả này cũng chưa chứng minh mục tiêu test-200 4,2 tháng.

### 4.10. Sex embedding và Sex FiLM

Đã xem xét hai cách đưa giới tính vào mô hình:

- `sex embedding`: mã hóa nam/nữ thành vector rồi nối với image feature;
- `Sex FiLM`: dùng giới tính để điều chế feature theo kênh bằng scale/shift, có thể áp dụng riêng cho global và local stream.

Sex FiLM có tiềm năng hơn khi hai giới cần cách diễn giải đặc trưng khác nhau; sex embedding đơn giản, rẻ và dễ làm control. Do phần so sánh trực tiếp đã có người khác trong nhóm triển khai, hướng hiện tại không mở thêm ablation này, mà quay về V4 ConvNeXtV2-Base + anatomical ROI.

## 5. Hướng đang triển khai và kế hoạch tiếp theo

### 5.1. Nhánh không cần GPU: leakage-safe OOF ensemble

Đây là việc ưu tiên trước khi train V4 dài:

1. Chuẩn hóa P7 raw, P7-TTA, C3 raw, C3-TTA về schema chung.
2. Kiểm tra đúng 14.036 ID, target/sex/fold, finite values và provenance.
3. Tính metric từng nguồn, correlation, error complementarity, subgroup và oracle chỉ như chẩn đoán.
4. Fit simplex weights không âm, tổng bằng 1 bằng cross-fitting: mỗi fold đích chỉ được dùng trọng số fit từ bốn fold còn lại.
5. So candidate với control cố định `0,66 × P7-TTA + 0,34 × C3-raw` bằng paired bootstrap.
6. Chỉ nhận stack mới nếu OOF MAE tốt hơn control và CI 95% của delta có upper bound < 0.

Task schema OOF và diagnostics đã được triển khai/kiểm thử trong worktree thí nghiệm. Cross-fitted simplex stacker, CLI chạy thật và báo cáo quyết định vẫn là phần cần hoàn thiện; tuyệt đối chưa đọc test-200 ở nhánh này.

### 5.2. V4 — ConvNeXtV2-Base + anatomical ROI trước, ensemble sau

V4-A dự kiến:

- global ConvNeXtV2-Base pretrained;
- compact bilinear global descriptor `C→512→128`, 16.384 chiều rồi project 256;
- sáu ROI qua MobileNetV3-Small, region embedding 16 chiều và attention pooling;
- Sex FiLM cho global/local, sex embedding 32 chiều;
- fusion `[256 global + 128 local + 32 sex] = 416`;
- head mean + log-variance;
- loss Smooth L1 + heteroscedastic NLL, NLL weight 0,05;
- synchronized augmentation cho toàn bộ bảy view, không horizontal flip;
- warm-up freeze encoder, sau đó AdamW ba parameter groups và checkpoint/resume đầy đủ.

Gate đã định trước:

- Fold 1 MAE `≤6,5`: promote V4-A lên 5-fold, không chạy V4-B.
- `6,5 < MAE ≤6,7`: chạy đúng một V4-B với ordinal age-stage auxiliary head.
- `MAE >6,7` hoặc subgroup safety gate thất bại: dừng nhánh V4 Fold 1, không tinh chỉnh ngẫu nhiên.

V4 chỉ được thêm vào ensemble sau khi có OOF 5-fold hợp lệ. Không lấy checkpoint V3-C làm pretrained initialization cho V4.

### 5.3. Sau khi có V4 OOF

1. Chạy raw OOF trước.
2. Chỉ chạy V4-TTA nếu raw OOF đã hợp lệ và TTA thắng bằng paired bootstrap.
3. Đưa V4 raw/TTA vào evaluator simplex hiện tại.
4. Chốt duy nhất một estimator cuối cùng bằng OOF cross-fit.
5. Ký manifest gồm code commit, checkpoint hash, split hash, preprocessing, TTA và weights.
6. Chỉ sau đó mới chạy một lần test-200 chính thức; nếu cần ảnh sạch thì báo cáo là robustness phụ, không dùng để tune.

## 6. Các hướng đã loại hoặc không ưu tiên

- Hard ROI/crop một phần bàn tay làm phương pháp chính: từng làm xấu OOF hoặc có fallback cao.
- Histogram equalization: bản NO_HE tốt hơn trên OOF; không mặc định dùng HE.
- Bilinear pooling đơn lẻ trên C3-R2: Fold 1 không tốt hơn control và Fold 2 bất ổn.
- ConvNeXtV2-FCMAE-Tiny đơn giản: từng có dấu hiệu feature collapse trong nhánh D1; không suy rộng kết luận sang V4-Base có kiến trúc/recipe khác.
- Mean Teacher: không tiếp tục vì kết quả lịch sử không tốt hơn control.
- Tăng epoch đơn thuần: không giải quyết được pattern overfitting.
- Dùng 200 ảnh sạch để train: cấm vì gây leakage và không phản ánh khả năng tổng quát hóa.
- Chọn model/weights từ test-200: cấm trong protocol chính.

## 7. Kết luận bàn giao

Hiện tại có ba tín hiệu mạnh nhất:

1. **P7/E1 ConvNeXt-Tiny** là baseline global ổn định.
2. **C3-ROI/C3-TTA** có diversity và hiệu quả test exploratory tốt, dù standalone OOF chưa thắng P7.
3. **Blending và TTA** có bằng chứng OOF tốt hơn các thay đổi preprocessing đơn lẻ.

> Vì vậy thứ tự hợp lý là hoàn thiện OOF stacking leakage-safe trước, rồi triển khai V4 ConvNeXtV2-Base + anatomical ROI theo gate Fold 1. Chưa có bằng chứng hiện tại cho phép khẳng định sẽ đạt MAE 4,2 trên test-200; mục tiêu đó vẫn là mục tiêu cần kiểm chứng sau khi estimator được khóa.

## 8. Các artifact chính để tiếp tục

- `AI_Context/01_STATUS_RESULTS.md` — trạng thái và số liệu hiện hành của nhóm.
- `AI_Context/02_METHOD_HISTORY.md` — lịch sử quyết định phương pháp.
- `AI_Context/03_DATA_PROTOCOL.md` — split, leakage và đánh giá.
- `research/our_baseline/RESULTS_SUMMARY.md` — kết quả P7/EXP-006/EXP-008/EXP-009.
- `research/our_baseline/EXPERIMENT_REGISTRY.md` — registry thí nghiệm.
- `d3_oof/` — OOF, TTA và calibration.
- `c4_multi_roi/` — calibrated stacking và multi-ROI.
- `project/C3_Z26_ANATOMICAL_ROI_FOLD1_MAE65_ABLATION` ở máy local — package V3-A/B/C và kế hoạch V4.

### Trạng thái truy cập test-200

Các số liệu test-200 trong tài liệu này chỉ để đối chiếu lịch sử/exploratory. Từ thời điểm này, mọi quyết định chọn tiếp phải dựa trên validation/OOF; test-200 không được dùng lại để điều chỉnh estimator.
