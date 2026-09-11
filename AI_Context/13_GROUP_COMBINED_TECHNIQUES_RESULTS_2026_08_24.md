# Tổng hợp kỹ thuật và kết quả nghiên cứu dự đoán tuổi xương của nhóm

**Ngày tổng hợp:** 2026-08-25
**Bài toán:** Dự đoán tuổi xương theo tháng từ ảnh X-quang bàn tay
**Dữ liệu chính:** RSNA Pediatric Bone Age
**Phạm vi:** Hồ sơ `AI_Context` P0-P13, D3 OOF/TTA, Experiment C (C3-ROI) và bundle thực nghiệm EXP-001 đến EXP-009
**Vị trí khoa học:** Nghiên cứu thực nghiệm có kiểm soát ở cấp đồ án tốt nghiệp; chưa phải tuyên bố state of the art hoặc xác nhận lâm sàng.

## 1. Phạm vi và quy ước nguồn

Báo cáo này tổng hợp bốn lớp bằng chứng:

1. **Nhánh chính P0-P13:** các kết quả đã được lưu trong `D:\Hoctap\Doan_totnghiep\AI_Context` và các artifact tương ứng của repository.
2. **Bundle baseline của thành viên nhóm:** baseline v1 và EXP-001 đến EXP-009 trong `research/our_baseline/`, gồm registry, code, config, log, prediction CSV và report JSON.
3. **Nhánh D3 Label Distribution Learning + TTA:** báo cáo trong `AI_Context/19_REPORT_PLAN_A_D3_FINAL.md` và artifact trực tiếp trong `d3_oof/`.
4. **Experiment C — C3-ROI local/global:** báo cáo trong `AI_Context/24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md` và artifact trực tiếp trong `c3_roi/`.

Các số liệu mới trong bản cập nhật này được đối chiếu với report JSON trong repository ở commit `a75b130e` (2026-08-25). Các kết quả trên test 200 vẫn được đánh dấu **thăm dò**, vì nhãn test đã được truy cập từ P8 và không còn là holdout xác nhận mới.

## 2. Kết luận điều hành

Pipeline chính được giữ trong nhánh P0-P13 là:

```text
Ảnh X-quang grayscale
→ pad vuông giữ tỷ lệ
→ resize bicubic 512×512
→ lặp thành 3 kênh
→ ImageNet normalization
→ augmentation A2
→ ConvNeXt-Tiny pretrained ImageNet-1K
→ sex embedding
→ direct regression bằng Smooth L1
→ TTA 10 views khi suy luận
```

Các kết quả quan trọng nhất:

- P7 pooled OOF trên 14.036 ảnh: MAE **6,316691 tháng**.
- P8 equal-weight ensemble 5 fold trên 200 ảnh test: MAE **4,730321 tháng**.
- TTA giảm OOF MAE từ 6,317471 xuống **6,210446 tháng**.
- Sex embedding giảm MAE **1,286919 tháng** so với image-only trên cùng 1.425 ảnh validation.
- EXP-006 P7 + TTA đạt OOF MAE **6,296203**; blend cố định EXP-008 đạt **6,120448** trên 14.036 OOF.
- Ensemble cố định `0,5 × E1-TTA + 0,5 × D3-TTA` đạt OOF MAE **6,10134 tháng**, là điểm OOF thấp nhất được báo cáo trong các nguồn tổng hợp.
- C3-ROI đứng riêng kém E1 trên OOF, nhưng ensemble cố định `0,5 × E1 + 0,5 × C3-ROI` đạt **6,176212**, cải thiện 0,140480 tháng với CI hoàn toàn dưới 0.
- C3-ROI đạt test-200 MAE **4,337267 tháng**, là số đo test thăm dò thấp nhất hiện có; kết quả này không được dùng để thay pipeline hoặc chọn lại trọng số.
- Kết quả hiện tại vẫn chưa đạt các mốc tham khảo khoảng 3,68-3,87 tháng và chưa có external holdout hoàn toàn chưa bị tác động.

Thông điệp nghiên cứu thống nhất là: **thông tin giới tính, augmentation nhẹ có kiểm soát, TTA và diversity của ensemble tạo lợi ích đáng tin cậy hơn việc tăng độ phức tạp kiến trúc, độ phân giải hoặc masking một cách đơn lẻ.**

## 3. Dữ liệu và protocol đánh giá

| Tập dữ liệu | Số ảnh | Vai trò |
|---|---:|---|
| Official train | 12.611 | Huấn luyện trọng số |
| Official validation | 1.425 | Chọn cấu hình và checkpoint trong development |
| Development pool | 14.036 | Train/đánh giá 5-fold OOF |
| RSNA test | 200 | Đánh giá cuối hoặc benchmark tham khảo |

P0 kiểm tra ID, SHA-256, đường dẫn ảnh, duplicate và giao nhau giữa các split. Không phát hiện leakage hoặc duplicate theo các kiểm tra đã thực hiện.

P8 đã đọc ground truth của test 200 để tính metric. Do đó:

- P8 là benchmark kỹ thuật sau khi khóa P7;
- test 200 không còn là confirmatory holdout hoàn toàn mới cho các quyết định sau P8;
- không được dùng test để chọn checkpoint, augmentation, preprocessing, ensemble weight hoặc threshold;
- các so sánh phát triển sau P8 phải dựa trên validation, OOF hoặc external holdout mới.

## 4. Baseline và hạ tầng tái lập

P1 xây dựng baseline ConvNeXt-Tiny với các thành phần:

- input 512×512, grayscale lặp ba kênh;
- pretrained ImageNet-1K;
- sex embedding;
- direct regression trên tuổi đã chuẩn hóa;
- Smooth L1 beta tương đương 3 tháng;
- AdamW, cosine schedule và early stopping;
- checkpoint nguyên tử chứa model, optimizer, scheduler, scaler, RNG, epoch/step và early-stop state;
- hash dữ liệu, cấu hình và mã nguồn;
- log metric, warning, prediction và trạng thái run.

FP16 smoke ban đầu tạo gradient Inf. Pipeline chuyển sang ưu tiên BF16 khi phần cứng hỗ trợ. Smoke, stop/resume và kiểm tra tensor/training state cuối cùng đều PASS.

Đóng góp kỹ thuật quan trọng của P1 không phải một kiến trúc mới, mà là khả năng train dài và resume có kiểm toán qua nhiều phiên hoặc tài khoản Colab.

## 5. Tổng hợp toàn bộ kỹ thuật P0-P13

### 5.1. Tổng hợp các kỹ thuật theo nhóm

| Nhóm kỹ thuật | Kỹ thuật đã thử | Kết quả nổi bật | Trạng thái |
|---|---|---|---|
| Kiểm soát dữ liệu | Audit ID/SHA-256, duplicate, overlap và khóa manifest | 12.611 train, 1.425 validation, 200 test; không phát hiện overlap đã biết | **Bắt buộc giữ** |
| Hình học ảnh | `pad_square`, resize bicubic 512×512 | Bỏ `pad_square` từng làm prediction lệch trung bình 2,68 tháng | **Giữ** |
| Chuẩn hóa ảnh | Grayscale lặp 3 kênh, ImageNet normalization | Tương thích tốt với pretrained ConvNeXt | **Giữ** |
| Tiền xử lý | Full-hand mask; Deeplasia mask/normalization | Mask B1 6,23894, kém control 6,18479; A1 6,307675 | **Không giữ trong recipe hiện tại** |
| Augmentation | A0 none, A1 flip, A2 nhẹ, Deeplasia moderate | A2 đạt 6,185 và cải thiện 0,455 tháng; moderate gần như không đổi | **Giữ A2** |
| Backbone | ConvNeXt-Tiny, ConvNeXtV2-FCMAE, EfficientNet-B0 | ConvNeXt-Tiny ổn định; ConvNeXtV2 collapse; EfficientNet screening kém | **Giữ ConvNeXt-Tiny** |
| Đặc trưng/độ phân giải | Multi-scale; 512 so với 768 px | Multi-scale 6,29697; 768 chỉ cải thiện 0,001371 tháng | **Giữ single-scale 512** |
| Thông tin giới tính | Image-only, sex embedding, dual-output M/F | Sex embedding cải thiện 1,286919 tháng; dual-output chỉ thêm 0,027730, CI cắt 0 | **Giữ sex embedding** |
| Đầu ra/loss | Direct Smooth L1; Label Distribution Learning | Direct regression ổn định; D3 tạo diversity hữu ích cho ensemble | **Giữ direct regression; xem D3 là nhánh bổ sung** |
| ROI/local-global | Hand-bbox crop + margin 8%, fallback full image; blend E1/C3 | C3 OOF 6,437349; blend 50/50 đạt 6,176212, CI delta dưới 0; fallback development 18,49% | **Ablation dương tính; localization chưa đạt gate** |
| Tối ưu/tái lập | AdamW, cosine, early stopping, BF16, atomic checkpoint và hash | P10-B0 tái lập chính xác P2; smoke/resume PASS | **Giữ** |
| Xác nhận | Multi-seed, 5-fold OOF, paired bootstrap | P7 OOF 6,316691; CI giúp loại các cải thiện nhỏ không chắc chắn | **Bắt buộc giữ** |
| Ensemble | Equal-weight 5 fold; P7+EXP-006; E1+D3; E1+C3 | EXP-008 OOF 6,120448; E1-TTA+D3-TTA OOF 6,10134; E1+C3 OOF 6,176212 | **Giữ các kết quả diversity đã khóa; chưa chọn bằng test** |
| Inference | TTA 10 views | OOF 6,317471→6,210446; cải thiện 0,107025, CI không cắt 0 | **Giữ TTA** |
| Calibration | Linear bias correction cross-fitted | Không cải thiện raw và làm TTA xấu hơn | **Loại** |
| Uncertainty | TTA disagreement và risk-coverage | ρ=0,200431; AUROC 0,625814/0,634108 | Chỉ phân tầng nghiên cứu |
| Subgroup | Phân tích sex × age | Nhóm khó và utility disagreement không đồng nhất | **Giữ cho fairness/safety** |
| Reporting | Manifest SHA, bảng/hình tự động, repeat build | 4 bảng, 3 hình; test 5/5; 0/15 output đổi hash | **Giữ** |

### 5.2. Chuỗi kỹ thuật được khuyến nghị

```text
Audit dữ liệu và khóa split
→ pad_square + resize bicubic 512×512
→ grayscale lặp 3 kênh + ImageNet normalization
→ augmentation A2
→ ConvNeXt-Tiny pretrained + sex embedding
→ direct Smooth L1 regression
→ AdamW + cosine + early stopping
→ checkpoint/resume + hash
→ 5-fold OOF + paired bootstrap + subgroup analysis
→ TTA 10 views
→ ensemble cố định trên OOF khi nhánh bổ sung chứng minh diversity
→ TTA disagreement chỉ dùng cho phân tầng nghiên cứu
```

### 5.3. Kết quả chi tiết theo phase

| Phase | Kỹ thuật/can thiệp | Kết quả chính | Quyết định |
|---|---|---|---|
| P0 | Audit split, ID, SHA-256, duplicate và leakage | 12.611/1.425/200 hợp lệ; không phát hiện overlap đã biết | Khóa protocol |
| P1 | ConvNeXt-Tiny + sex embedding + direct regression; checkpoint/resume/AMP | Unit, preflight, smoke và resume PASS | Dùng làm hạ tầng baseline |
| P2-A0 | Không augmentation | MAE 6,640 | Baseline augmentation |
| P2-A1 | Horizontal flip | MAE 6,514 | Có tín hiệu nhỏ nhưng chưa chắc chắn |
| P2-A2 | Flip + rotation ±7° + translation/scale nhẹ + brightness/contrast/gamma | MAE **6,185**; A2−A0 = −0,455, CI [−0,658; −0,249] | **Giữ A2** |
| P3-B1 | Full-hand background masking | 6,23894 so với control 6,18479; delta +0,05414, CI cắt 0 | Giữ `preprocessing=none` |
| P4-D1 | ConvNeXtV2-Tiny FCMAE | MAE **31,96281**, feature collapse | Loại recipe D1 |
| P4-D2 | Multi-scale fusion | MAE 6,29697; delta +0,11218, CI cắt 0 | Loại |
| P4-D3 | Label Distribution Learning, fused prediction | Seed 42 MAE 6,14541; tốt hơn D0 0,03938 nhưng CI cắt 0 | Kiểm tra nhiều seed |
| P5 | D0/D3 trên seed 17, 42, 123 | D3 fused mean 6,22508 so với D0 6,26229; delta −0,03721, CI [−0,13521; +0,05863] | Giữ D0 đơn giản |
| P6 | Tăng resolution 512→768 | 6,183421 so với 6,184792; delta −0,001371, CI [−0,190703; +0,192127] | Giữ 512 |
| P7 | Final 5-fold OOF | N=14.036; MAE **6,316691**; RMSE 8,519420; CI [6,224638; 6,411327] | Endpoint development chính |
| P8 | Equal-weight 5-fold test ensemble | N=200; MAE **4,730321**; RMSE 6,028745; CI [4,222475; 5,263709] | Benchmark kỹ thuật |
| P9-A0 | L1/100 epoch, không mask | MAE 6,391447; kém P2 0,206656, CI nằm phía dương | Loại |
| P9-A1 | Deeplasia mask + L1 recipe | MAE 6,307675; tốt hơn A0 nhẹ nhưng không hơn P2 | Không chạy OOF/test |
| P10-B0 | Tái lập chính xác P2 | MAE **6,184792**; 1.425 prediction giống hệt P2 | Khóa control |
| P10-B1 | Deeplasia augmentation mức vừa | MAE 6,185877; delta +0,001086, CI cắt 0 | Không giữ |
| P9-I | TTA 10 views | Raw 6,317471 → TTA **6,210446**; delta −0,107025, CI [−0,135977; −0,078220] | **Giữ TTA** |
| P9-I | Linear bias correction cross-fitted | Raw corrected 6,324285; TTA+correction 6,228665 | Loại correction |
| P9-B0 | EfficientNet-B0 512 gần Deeplasia | Best validation MAE 8,5293; 8,5823; 11,0428 | Dừng screening |
| P11-E0 | ConvNeXt-Tiny image-only | MAE **7,471711** | Đối chứng giá trị sex |
| P11-E1 | ConvNeXt-Tiny + sex embedding | MAE **6,184792** | **Mô hình được giữ** |
| P11-E2 | Shared backbone + dual output M/F | MAE 6,157061; E2−E1 = −0,027730, CI cắt 0 | Không đạt gate |
| P12 | TTA disagreement làm uncertainty proxy | Spearman ρ=0,200431; AUROC lỗi >12/>18 = 0,625814/0,634108 | Chỉ dùng phân tầng nghiên cứu |
| P13 | Build bảng/hình, manifest và bản thảo | 4 bảng, 3 hình; test 5/5 PASS; repeat build 0/15 artifact đổi hash | Hoàn tất reporting |
| EXP-006/007 | P7 control 5-fold và TTA | Control 6,323629; TTA **6,296203** trên 14.036 OOF | Giữ TTA trong nhánh baseline bundle |
| EXP-008 | Blend cố định friend P7 + EXP-006 TTA | OOF **6,120448**; test-200 4,553021 (thăm dò) | OOF selection gate hiện tại của bundle |
| D3 final | LDL hai đầu ra + TTA + blend E1/D3 | D3-TTA 6,246542; blend **6,101345** OOF | Diversity dương tính, artifact PASS |
| C3-ROI | Hand-bbox crop/fallback + blend global/local | C3 6,437349; E1+C3 **6,176212** OOF | Ablation dương tính; fallback vượt gate |

## 6. Kết quả P7 OOF và P8 test

### 6.1. P7 pooled OOF

| Chỉ số | Kết quả |
|---|---:|
| Số prediction/ID duy nhất | 14.036/14.036 |
| MAE | **6,316691** |
| RMSE | 8,519420 |
| Median absolute error | 4,875 |
| Accuracy ±6 tháng | 59,01% |
| Accuracy ±12 tháng | 86,73% |
| Accuracy ±18 tháng | 95,53% |
| Bootstrap 95% CI MAE | [6,224638; 6,411327] |

MAE từng fold:

| Fold | MAE |
|---:|---:|
| 1 | 6,298710 |
| 2 | 6,196835 |
| 3 | 6,411474 |
| 4 | 6,348768 |
| 5 | 6,327675 |

Mean fold MAE là 6,316692, SD 0,078775, cho thấy mức dao động giữa fold tương đối nhỏ.

### 6.2. P8 test ensemble

| Chỉ số | Kết quả |
|---|---:|
| Số ảnh test | 200 |
| MAE | **4,730321** |
| RMSE | 6,028745 |
| Median AE | 4,133363 |
| Accuracy ±6 tháng | 70,0% |
| Accuracy ±12 tháng | 94,0% |
| Accuracy ±18 tháng | 100,0% |
| Bootstrap 95% CI MAE | [4,222475; 5,263709] |

Một tài liệu thành viên tính bằng nhãn test đầy đủ cho kết quả 4,730865 thay vì 4,730321 của artifact dùng nhãn làm tròn. Sai khác này không thay đổi thứ hạng phương pháp.

Không được dùng chênh lệch P7 6,3167 và P8 4,7303 để tuyên bố ensemble cải thiện 1,59 tháng, vì hai metric được tính trên hai tập dữ liệu khác nhau.

### 6.3. Lưu ý audit P7

`P7_OOF_report.json` ghi endpoint OOF là PASS. Tuy nhiên, file `P7_5FOLD_AUDIT.txt` ở root từng ghi `FINAL FIVE-FOLD AUDIT: FAIL` do fold 1 thiếu block best-epoch metrics. Các kiểm tra cốt lõi vẫn đạt: đủ 14.036 dòng, ID duy nhất, fold disjoint, metric khớp state và manifest/model integrity.

Do đó, kết luận đúng là: **P7 OOF PASS theo báo cáo OOF chính thức, kèm bất nhất hành chính/provenance cần được giải thích trước khi công bố hoặc bảo vệ.**

## 7. Vai trò của giới tính

### 7.1. Kết quả E0/E1/E2

| Mô hình | MAE tổng | RMSE | Median AE | MAE nữ | MAE nam | Sex gap nữ−nam |
|---|---:|---:|---:|---:|---:|---:|
| E0 image-only | 7,471711 | 9,927859 | 6,0000 | 7,768597 | 7,221297 | 0,547300 |
| E1 sex embedding | **6,184792** | 8,486471 | 5,0000 | 6,389115 | 6,012451 | 0,376664 |
| E2 dual-output | 6,157061 | 8,367776 | 4,5000 | 6,383915 | 5,965718 | 0,418197 |

### 7.2. Paired effect

| So sánh | Delta MAE | 95% CI | Diễn giải |
|---|---:|---|---|
| E0−E1 overall | +1,286919 | [+1,012226; +1,569611] | E1 tốt hơn rõ ràng |
| E0−E1 nữ | +1,379481 | [+0,948691; +1,805653] | Cải thiện ở nữ |
| E0−E1 nam | +1,208845 | [+0,840546; +1,581351] | Cải thiện ở nam |
| E2−E1 overall | −0,027730 | [−0,179409; +0,124151] | Không đáng tin cậy |
| E2−E1 nữ | −0,005200 | CI cắt 0 | Không đạt gate 0,20 tháng |
| E2−E1 nam | −0,046734 | CI cắt 0 | Không đủ bằng chứng |

Kết luận được phép: trong recipe ConvNeXt đã khóa, nhãn giới tính cung cấp giá trị dự đoán bổ sung lớn và nhất quán so với image-only.

Không được diễn giải đây là bằng chứng nhân quả sinh học. Kết quả chỉ chứng minh giá trị dự đoán của biến sex trong dữ liệu và protocol được khảo sát.

E2 có point estimate thấp hơn E1 nhưng không qua gate, CI cắt 0 và sex gap không giảm. Vì vậy E1 có tỷ lệ hiệu năng-độ phức tạp hợp lý hơn.

## 8. TTA, bias correction và uncertainty proxy

### 8.1. TTA

TTA sử dụng 10 view:

```text
rotation [-10°, -5°, 0°, +5°, +10°]
×
không flip / horizontal flip
```

| Phương án | OOF MAE |
|---|---:|
| Raw tái suy luận | 6,317471 |
| TTA | **6,210446** |
| Raw + bias correction | 6,324285 |
| TTA + bias correction | 6,228665 |

TTA cải thiện 0,107025 tháng, CI [0,078220; 0,135977] nếu biểu diễn theo mức giảm. Linear bias correction không cải thiện raw và làm TTA xấu thêm khoảng 0,018219 tháng.

### 8.2. Disagreement

Disagreement là standard deviation của 10 prediction TTA.

| Chỉ số | Kết quả |
|---|---:|
| Spearman với absolute error | 0,200431 |
| Bootstrap 95% CI | [0,184238; 0,216411] |
| AUROC lỗi >12 tháng | 0,625814 |
| AUROC lỗi >18 tháng | 0,634108 |

| Quartile disagreement | TTA MAE | Lỗi >12 | Lỗi >18 |
|---|---:|---:|---:|
| Q1 thấp | 4,758689 | 7,0390% | 2,0804% |
| Q2 | 5,942821 | 11,5417% | 3,3628% |
| Q3 | 6,516477 | 13,5366% | 4,4172% |
| Q4 cao | 7,623798 | 20,1482% | 6,8396% |

Q4 có lỗi >12 tháng gấp 2,86 lần và lỗi >18 tháng gấp 3,29 lần Q1. Disagreement phù hợp để làm giàu nhóm nguy cơ trong nghiên cứu, nhưng AUROC chỉ ở mức hạn chế và chưa thể dùng như xác suất sai, khoảng tin cậy lâm sàng hoặc threshold tự động từ chối dự đoán.

## 9. Kết quả theo giới tính và nhóm tuổi

Trong P7/P9-I OOF:

- nữ: raw MAE khoảng 6,537; TTA 6,458;
- nam: raw MAE khoảng 6,132; TTA 6,001;
- gap nữ-nam sau TTA khoảng 0,457 tháng;
- sau chuẩn hóa thô theo nhóm tuổi, gap giảm còn khoảng 0,260 tháng.

Điều này cho thấy khác biệt phân bố tuổi giải thích một phần nhưng không toàn bộ sex gap.

Các nhóm đáng chú ý:

- M 60-119 tháng có TTA MAE cao nhất, khoảng 7,9313, nhưng disagreement chỉ có ρ=0,0870 và AUROC 0,5730;
- F 180-228 có ρ=0,3564 nhưng chỉ n=365;
- F 0-59 không có association đáng tin cậy;
- uncertainty proxy không hoạt động đồng đều giữa sex × age.

Nhóm khó nhất về sai số không nhất thiết là nhóm được disagreement nhận diện tốt nhất. Đây là giới hạn quan trọng về fairness và safety.

## 10. Nhánh baseline và blend của thành viên nhóm

### 10.1. Baseline v1

Baseline v1 sử dụng ConvNeXt-Tiny, sex embedding 32 chiều, direct Smooth L1 regression và resize trực tiếp 512×512, không dùng `pad_square`.

| Phương án | Validation/test MAE |
|---|---:|
| Baseline P7 reference v1, official validation | **6,471143** |
| A2 light flip của baseline v1 | 6,706371 |
| Một model baseline v1 trên test 200 | 5,113332 |

Light flip làm xấu baseline v1, nhưng không được suy rộng rằng horizontal flip luôn có hại vì preprocessing và augmentation của pipeline P7 khác.

### 10.2. Khác biệt giữa baseline v1 và P7 mạnh hơn

Pipeline P7 của nhánh chính sử dụng:

- `pad_square` giữ tỷ lệ hình thái bàn tay;
- resize bicubic 512×512 có antialias;
- sex embedding 16 chiều;
- target normalization mean/std;
- Smooth L1 beta 3 tháng;
- LR `2e-4`, weight decay `0,05`;
- batch hiệu dụng lớn hơn;
- A2 augmentation có hình học và cường độ nhẹ;
- ensemble 5 fold.

Khi tái tạo inference mà bỏ qua `pad_square`, sai khác prediction tối đa là 7,09657 tháng và trung bình tuyệt đối 2,67998 tháng. Sau khi dùng đúng `data.py`, `model.py`, target normalization và preprocessing, sai khác tối đa trên 10 ảnh giảm còn 0,054260 tháng.

Kết quả này cho thấy preprocessing hình học và target normalization là thành phần không thể bỏ qua khi tái lập model.

### 10.3. EXP-001 đến EXP-005

| Thí nghiệm | Phương án | MAE | Diễn giải |
|---|---|---:|---|
| EXP-001 | Baseline own | 6,471143 | Official validation |
| EXP-001 | Friend OOF cùng 1.425 hàng | 6,408334 | Model bổ sung |
| EXP-001 | Blend 50/50 | **6,177155** | Tốt nhất trên validation đã dùng chọn weight |
| EXP-002 | Baseline trên internal holdout 391 ảnh | 6,070253 | Holdout không hoàn toàn độc lập |
| EXP-002 | Blend 50/50 | **5,761539** | Tín hiệu blend còn tồn tại |
| EXP-004 | Retrain trên fresh holdout 1.260 ảnh | 6,213796 | Một model mới |
| EXP-004 | 75% retrain + 25% friend | **6,179959** | Blend tốt nhất nhưng cải thiện nhỏ |
| EXP-004 | Blend 50/50 | 6,204244 | Weight tối ưu thay đổi theo tập |
| EXP-005 | Retrain trên test 200 | 5,214580 | Exploratory |
| EXP-005 | Friend P7 ensemble | **4,730865** | Tốt nhất trong EXP-005 |
| EXP-005 | Blend 50/50 | 4,933367 | Không vượt friend ensemble |
| EXP-005 | 75% retrain + 25% friend | 5,065745 | Không vượt friend ensemble |

Kết luận: blend có tín hiệu trên validation và fresh holdout, nhưng trọng số tối ưu thay đổi theo tập và không vượt P7 ensemble trên test 200. Không nên chọn blend weight từ test.

### 10.4. EXP-006 đến EXP-009 sau lần hợp nhất repository

| Thí nghiệm | Protocol | MAE | Quyết định |
|---|---|---:|---|
| EXP-006 P7 control | 5-fold pooled OOF, n=14.036 | 6,323629 | Hoàn tất |
| EXP-007 P7 + TTA | 5-fold pooled OOF, n=14.036 | **6,296203** | Giữ; cải thiện 0,027426 tháng |
| EXP-008 fixed 50/50 blend | Friend P7 + EXP-006 TTA OOF | **6,120448** | OOF gate của bundle |
| EXP-007 P7 + TTA | Test 200 | **4,466886** | Chỉ thăm dò |
| EXP-008 fixed blend | Test 200 | 4,553021 | Chỉ thăm dò |
| EXP-009 ConvNeXt-Tiny LDL fused | Cùng split 5-fold | Chưa có kết quả cuối | Phải thấp hơn 6,120448 |

EXP-008 là bằng chứng OOF sạch hơn các blend EXP-001/002 vì dùng hai nguồn prediction OOF đã căn hàng trên toàn bộ 14.036 ảnh và khóa trọng số 50/50 trước khi đọc test. Không gộp EXP-009 với D3 final: đây là roadmap LDL riêng trong bundle baseline và vẫn chưa có endpoint cuối.

## 11. Nhánh D3 Label Distribution Learning + TTA của thành viên nhóm

### 11.1. Trạng thái nguồn

Kết quả phần này lấy từ báo cáo D3 final ngày 2026-08-23. Sau lần pull ngày 2026-08-25, các artifact `d3_oof/...` đã có trong repository. Đối chiếu trực tiếp cho thấy `D3_OOF_report.json`, `D3_TTA_OOF_report.json` và `FINAL_TEST_TTA_report.json` đều ghi `status=PASS`, đúng 14.036 OOF và 200 test; các MAE khớp báo cáo.

EXP-007/008 của bundle baseline và nhánh D3 là các chuỗi thí nghiệm khác nhau dù tên giai đoạn có thể gây nhầm. Báo cáo này phân biệt chúng bằng đường dẫn artifact và công thức ensemble, không suy diễn trạng thái của nhánh này từ log của nhánh kia.

### 11.2. Cấu hình D3

- Backbone ConvNeXt-Tiny.
- Sex embedding đưa vào prediction head.
- Regression head dự đoán tuổi liên tục.
- Distribution head dự đoán phân phối tuổi theo từng tháng.
- Prediction fused:

```text
D3_fused = 0,5 × regression_prediction
         + 0,5 × distribution_prediction
```

- 5 fold độc lập, seed 42.
- Không warm-start từ E1.
- TTA 10 views giống P9-I.
- Ensemble được khóa trước khi mở test:

```text
Final ensemble = 0,5 × E1-TTA + 0,5 × D3-TTA
```

### 11.3. Kết quả OOF

| Mô hình | OOF MAE |
|---|---:|
| E1 raw | 6,31669 |
| D3 fused raw | 6,38241 |
| E1 + D3 fused raw 50/50 | 6,18268 |
| E1-TTA | 6,21045 |
| D3-TTA | 6,24654 |
| E1-TTA + D3-TTA 50/50 | **6,10134** |

So với E1-TTA, ensemble TTA cải thiện **0,10910 tháng** với paired bootstrap 95% CI `[-0,13365; -0,08420]` theo quy ước ensemble trừ E1.

Ensemble cải thiện ở cả 5 fold:

| Fold | E1-TTA | D3-TTA | Ensemble 50/50 |
|---:|---:|---:|---:|
| 1 | 6,13870 | 6,19073 | **6,05910** |
| 2 | 6,13447 | 6,12869 | **6,03226** |
| 3 | 6,30125 | 6,42416 | **6,23347** |
| 4 | 6,28549 | 6,29264 | **6,15197** |
| 5 | 6,19235 | 6,19651 | **6,02994** |

Theo giới:

| Nhóm | E1-TTA | D3-TTA | Ensemble |
|---|---:|---:|---:|
| Nữ | 6,45810 | 6,45938 | **6,32824** |
| Nam | 6,00109 | 6,06661 | **5,90953** |

Theo tuổi:

| Tuổi | E1-TTA | D3-TTA | Ensemble |
|---|---:|---:|---:|
| 0-59 | 5,89530 | 5,80815 | **5,55350** |
| 60-119 | 7,29684 | 7,31955 | **7,17422** |
| 120-179 | 5,75861 | 5,86572 | **5,72144** |
| 180-228 | 5,96215 | 5,66558 | **5,59463** |

D3 riêng lẻ xấu hơn E1 trên OOF, nhưng prediction có phần sai số bổ sung. Vì vậy ensemble tốt hơn cả hai nhánh. Đây là lợi ích của **error diversity**, không phải bằng chứng rằng D3 riêng lẻ thắng E1 trên development data.

### 11.4. Kết quả test 200

| Mô hình | Test MAE |
|---|---:|
| E1-TTA | 4,61749 |
| D3-TTA | **4,50846** |
| E1-TTA + D3-TTA 50/50 | 4,51070 |

D3-TTA thấp hơn E1-TTA 0,10903 tháng trên test. Ensemble chỉ kém D3-TTA 0,00224 tháng.

Theo giới:

| Nhóm | E1-TTA | D3-TTA | Ensemble |
|---|---:|---:|---:|
| Nữ, n=100 | 4,74378 | **4,55134** | 4,58479 |
| Nam, n=100 | 4,49120 | 4,46558 | **4,43660** |

Theo tuổi:

| Tuổi | E1-TTA | D3-TTA | Ensemble |
|---|---:|---:|---:|
| 0-59, n=14 | **5,13860** | 5,56981 | 5,35420 |
| 60-119, n=53 | 6,27339 | **6,04524** | 6,09767 |
| 120-179, n=108 | 3,91260 | 3,91477 | **3,85883** |
| 180-228, n=25 | 3,86030 | **3,22085** | 3,49000 |

Trong riêng nhánh D3, MAE 4,50846 là điểm test thấp nhất và tốt hơn P8 khoảng 0,222 tháng. Sau khi bổ sung Experiment C, số đo test thấp nhất toàn repository là C3-ROI 4,337267. Tuy nhiên:

- D3-TTA không tốt hơn E1-TTA trên OOF;
- test chỉ có 200 ảnh;
- test đã được truy cập từ P8;
- chưa có external validation;
- chưa có paired CI test được báo cáo trong tài liệu D3.

Do đó, D3-TTA nên được gọi là **kết quả test thăm dò tốt nhất của nhánh D3**, không phải tốt nhất toàn repository. Bằng chứng khoa học mạnh hơn của nhánh D3 là ensemble đạt OOF 6,10134 và cải thiện nhất quán ở 5/5 fold.

## 12. Experiment C — C3-ROI local/global

C3-ROI dùng segmentation bounding-box của bàn tay, nới margin 8%, crop rồi resize 512×512. Khi segmentation thất bại, pipeline fallback về ảnh full; đây là broad hand-bbox crop, không phải carpal ROI theo PCA như đặc tả ban đầu.

| Tập | Bbox ROI | Full-image fallback | Tỷ lệ fallback |
|---|---:|---:|---:|
| Development, n=14.036 | 11.441 | 2.595 | **18,49%** |
| Test, n=200 | 134 | 66 | **33,00%** |

Fallback vượt xa gate ≤1%, nên không được gọi C3 là anatomy-local specialist thuần túy.

| Mô hình | OOF MAE | Delta so với E1 | Diễn giải |
|---|---:|---:|---|
| E1/P7 raw | 6,316691 | — | Global baseline |
| C3-ROI raw | 6,437349 | +0,120658; CI [+0,061527; +0,176776] | Đứng riêng kém hơn |
| E1 + C3-ROI 50/50 | **6,176212** | **−0,140480; CI [−0,172547; −0,109853]** | Cải thiện ở 5/5 fold |

Prediction correlation E1/C3 là 0,995123, nhưng sai số vẫn có diversity đủ để blend giảm MAE. Trên test 200, E1 đạt 4,730321, C3-ROI 4,337267 và blend 4,454661. Đây chỉ là số đo thăm dò; không được dùng để đổi trọng số hoặc chọn C3 thay cho pipeline đã khóa.

## 13. Kỹ thuật thành công, chưa thành công và chưa hoàn tất

### 13.1. Có bằng chứng tốt

1. **Sex embedding:** cải thiện khoảng 1,29 tháng so với image-only.
2. **A2 augmentation nhẹ:** cải thiện khoảng 0,455 tháng so với không augmentation.
3. **TTA 10 views:** cải thiện khoảng 0,107 tháng trên OOF.
4. **E1-TTA + D3-TTA ensemble:** OOF 6,10134, tốt hơn E1-TTA 0,10910 tháng và cùng chiều ở 5/5 fold.
5. **EXP-008 P7 + EXP-006-TTA blend:** OOF 6,120448 với trọng số 50/50 khóa trước test.
6. **E1 + C3-ROI ensemble:** OOF 6,176212, delta −0,140480 tháng với CI hoàn toàn dưới 0 và cùng chiều ở 5/5 fold.
7. **5-fold OOF và paired bootstrap:** giúp tách cải thiện thực khỏi dao động một validation split.
8. **Checkpoint/resume, manifest/hash và audit:** tạo khả năng tái lập và kiểm toán.

### 13.2. Không có đủ bằng chứng để giữ

- Full-hand masking B1.
- Deeplasia augmentation mức vừa.
- ConvNeXtV2-FCMAE recipe D1.
- Multi-scale fusion D2.
- D3 fused ở thử nghiệm P4/P5 cũ khi đứng riêng.
- Tăng resolution 512 lên 768.
- Linear bias correction.
- EfficientNet-B0 screening gần Deeplasia.
- Dual-output E2 theo giới.
- Blend model với weight chọn trên validation hoặc test.

Các kết luận âm chỉ áp dụng cho recipe, seed và protocol đã khảo sát; không được suy rộng rằng mọi dạng masking, EfficientNet, LDL, multi-scale hoặc calibration đều vô ích.

### 13.3. Kế hoạch hoặc trạng thái chưa được xác nhận đầy đủ

- EXP-009 LDL trong bundle baseline đã chuẩn bị đủ 5 fold nhưng chưa có kết quả cuối.
- C3-ROI chưa đạt gate fallback ≤1%; cần sửa localization rồi lặp lại dưới protocol khóa trước nếu muốn gọi là local specialist.
- External validation, calibration/selective prediction và systematic literature review vẫn chưa hoàn tất.

## 14. So sánh các mốc kết quả

| Mốc | MAE | Loại đánh giá | Ghi chú |
|---|---:|---|---|
| Baseline v1 official validation | 6,471143 | Validation | Resize trực tiếp, một model |
| P10-B0/E1 official validation | 6,184792 | Validation | Baseline chính có sex embedding |
| P7 pooled OOF | 6,316691 | OOF 14.036 | Endpoint development chính |
| P9-I E1-TTA OOF | 6,210446 | OOF 14.036 | TTA cải thiện 0,107 tháng |
| EXP-006 P7+TTA OOF | 6,296203 | OOF 14.036 | Nhánh baseline bundle |
| EXP-008 fixed blend OOF | 6,120448 | OOF 14.036 | Friend P7 + EXP-006 TTA |
| D3-TTA OOF | 6,24654 | OOF 14.036 | Xấu hơn E1-TTA khi đứng riêng |
| E1-TTA + D3-TTA OOF | **6,10134** | OOF 14.036 | OOF thấp nhất được báo cáo |
| C3-ROI OOF | 6,437349 | OOF 14.036 | Đứng riêng kém E1 |
| E1 + C3-ROI OOF | 6,176212 | OOF 14.036 | Ablation diversity dương tính |
| Baseline v1 test | 5,113332 | Test 200 | Một model |
| EXP-004 retrain test | 5,214580 | Test 200 | Một model |
| P8 friend ensemble test | 4,730321/4,730865 | Test 200 | Khác biệt do độ chính xác nhãn |
| E1-TTA test | 4,61749 | Test 200 | Báo cáo D3 final |
| EXP-006 P7+TTA test | 4,466886 | Test 200 | Thăm dò |
| D3-TTA test | 4,50846 | Test 200 | Tốt nhất trong nhánh D3 |
| Ensemble E1/D3-TTA test | 4,51070 | Test 200 | Gần tương đương D3-TTA |
| C3-ROI test | **4,337267** | Test 200 | Thấp nhất hiện có; chỉ thăm dò |
| E1 + C3-ROI test | 4,454661 | Test 200 | Không dùng để chọn lại weight |
| Bram 2025 tham khảo | 3,68 | Benchmark công bố | Không cùng toàn bộ protocol |
| Deeplasia 2024 tham khảo | 3,87 | Benchmark công bố | Không cùng toàn bộ recipe |

Không so sánh trực tiếp validation, OOF và test như các endpoint đồng cấp. Mỗi con số phải luôn đi kèm tập dữ liệu và protocol.

## 15. Đóng góp khoa học đề xuất

### C1. Định lượng giá trị dự đoán của giới tính

Sex embedding cải thiện MAE khoảng 1,29 tháng so với image-only, với paired CI không cắt 0 và lợi ích xuất hiện ở cả nữ và nam.

### C2. Chứng minh tăng chuyên biệt hóa theo giới không mặc nhiên tốt hơn

Dual-output E2 chỉ tốt hơn E1 0,0277 tháng, không qua gate, CI cắt 0 và không giảm sex gap. Đây là negative result có giá trị cho lựa chọn kiến trúc.

### C3. Xác định lợi ích inference của TTA

TTA tạo cải thiện nhỏ nhưng nhất quán khoảng 0,107 tháng trên 14.036 OOF. Bias correction không tạo thêm lợi ích.

### C4. Khai thác diversity của LDL

D3-TTA không tốt hơn E1-TTA khi đứng riêng trên OOF, nhưng ensemble cố định của hai nhánh đạt 6,10134 và cải thiện ở 5/5 fold. Điều này cho thấy LDL có thể tạo representation/prediction bổ sung hữu ích cho ensemble ngay cả khi single-model MAE không thấp hơn baseline.

### C5. Đánh giá giới hạn của TTA disagreement

Disagreement có liên hệ dương với sai số và làm giàu nhóm lỗi lớn, nhưng effect yếu, AUROC hạn chế và không đồng nhất giữa sex × age. Vì vậy chưa phải uncertainty lâm sàng.

### C6. Quy trình thực nghiệm có khả năng tái lập

Audit split/leakage, paired comparison, bootstrap CI, OOF, gate định trước, checkpoint/resume, SHA manifest và công bố kết quả âm tạo ra dấu vết kiểm toán tốt hơn cách chỉ báo cáo một MAE từ một lần train.

### C7. Định lượng diversity của biểu diễn ROI/global

C3-ROI đứng riêng kém E1 nhưng blend cố định cải thiện 0,140480 tháng trên OOF, CI hoàn toàn dưới 0 và cùng chiều ở 5/5 fold. Đồng thời, fallback 18,49% cho thấy lợi ích ensemble không đồng nghĩa localization đã đạt yêu cầu giải phẫu.

## 16. Hạn chế

1. P11 E0/E1/E2 mới là screening một seed trên official validation; bootstrap không phản ánh đầy đủ variance do retraining.
2. Test 200 đã được truy cập ở P8 và tiếp tục xuất hiện trong các đánh giá nhóm; không còn là untouched confirmatory holdout.
3. Test chỉ có 200 ảnh, các subgroup test rất nhỏ, đặc biệt nhóm 0-59 chỉ n=14.
4. Chưa có external validation theo bệnh viện, thiết bị hoặc quần thể độc lập.
5. D3 đã có report/prediction artifact trong repository, nhưng vẫn thiếu external replication độc lập.
6. P7 audit có bất nhất hành chính liên quan block metrics fold 1.
7. EfficientNet-B0 dừng sau 1-3 epoch và chưa bit-exact với Deeplasia; không được dùng để phủ định EfficientNet nói chung.
8. Uncertainty proxy chưa được calibration thành xác suất hoặc khoảng dự đoán.
9. Sex là nhãn nhị phân M/F có sẵn; kết quả không chứng minh quan hệ nhân quả và không bao phủ mọi khái niệm sex/gender.
10. C3-ROI có fallback 18,49% development và 33% test, vượt gate thiết kế ≤1%; không được diễn giải là local specialist thuần túy.
11. Chưa thực hiện systematic literature review để chứng minh novelty tuyệt đối.

## 17. Pipeline và kết quả nên dùng khi viết luận văn

### 17.1. Baseline chính

```text
ConvNeXt-Tiny
+ A2 augmentation
+ 512 px, pad_square
+ preprocessing=none
+ sex embedding
+ direct Smooth L1 regression
```

### 17.2. Cải tiến inference được xác nhận

```text
E1 + TTA 10 views
```

OOF MAE: **6,210446 tháng**.

### 17.3. Kết quả ensemble nhóm mới

```text
0,5 × E1-TTA + 0,5 × D3-TTA
```

OOF MAE báo cáo: **6,10134 tháng**, paired improvement 0,10910 tháng so với E1-TTA.

Các ablation OOF dương tính bổ sung:

- EXP-008 P7 + EXP-006-TTA 50/50: **6,120448**.
- E1 + C3-ROI 50/50: **6,176212**, paired delta −0,140480 và CI hoàn toàn dưới 0.

### 17.4. Điểm test thấp nhất được quan sát

C3-ROI: **4,337267 tháng**. Chỉ nên gọi là số đo test thăm dò thấp nhất hiện có; không phải endpoint dùng chọn mô hình, không thay thế bằng chứng OOF và chưa phải bằng chứng state of the art.

## 18. Cách diễn đạt khuyến nghị

### Nên dùng

- “cung cấp bằng chứng trong dữ liệu và protocol được khảo sát”;
- “sex embedding mang giá trị dự đoán bổ sung”;
- “TTA tạo cải thiện nhỏ nhưng nhất quán”;
- “D3 cung cấp error diversity hữu ích cho ensemble”;
- “C3-ROI cung cấp diversity hữu ích nhưng localization chưa đạt gate fallback”;
- “ensemble cải thiện OOF ở cả 5 fold”;
- “disagreement có giá trị phân tầng nhưng utility hạn chế”;
- “các kết quả test-200 cần được xác nhận trên external holdout”.

### Không nên dùng

- “đã vượt state of the art”;
- “đã vượt Bram hoặc Deeplasia”;
- “D3 chắc chắn tốt hơn E1” chỉ dựa vào test 200;
- “C3-ROI là local carpal specialist” khi fallback còn vượt gate;
- “giới tính gây ra sai khác tuổi xương”;
- “uncertainty phát hiện chính xác ca mô hình sai”;
- “EfficientNet, masking hoặc LDL không có ích nói chung”;
- “ensemble cải thiện P7 từ 6,31 xuống 4,73” vì hai metric ở hai tập khác nhau.

## 19. Kết luận cuối

Nghiên cứu của nhóm cho thấy các cải tiến có hiệu quả nhất không đến từ việc liên tục tăng độ phức tạp mô hình. Sex embedding tạo effect lớn nhất; A2 augmentation và TTA tạo lợi ích ổn định; nhiều can thiệp như masking, resolution cao, multi-scale, ConvNeXtV2-FCMAE, bias correction và dual-output không đủ bằng chứng để giữ.

Các nhánh D3, EXP-008 và C3 cùng củng cố một kết luận quan trọng: một model có MAE riêng lẻ không tốt hơn baseline vẫn có thể tạo giá trị nếu lỗi của nó đủ bổ sung. Ensemble E1-TTA + D3-TTA đạt OOF MAE 6,10134; EXP-008 đạt 6,120448; E1 + C3-ROI đạt 6,176212 với paired CI hoàn toàn dưới 0. C3-ROI đạt test MAE 4,337267, là số đo test thấp nhất hiện có, nhưng chỉ là kết quả thăm dò và localization còn không đạt gate fallback.

Vị trí khoa học phù hợp của đề tài là một nghiên cứu thực nghiệm có kiểm soát về:

1. giá trị của thông tin giới tính;
2. lợi ích và giới hạn của độ phức tạp kiến trúc;
3. TTA, ensemble diversity và uncertainty proxy;
4. quy trình chống leakage và tái lập;
5. giá trị của các kết quả âm được đánh giá bằng paired effect và confidence interval.

## 20. Nguồn chính

- `AI_Context/00_START_HERE.md`
- `AI_Context/01_STATUS_RESULTS.md`
- `AI_Context/02_METHOD_HISTORY.md`
- `AI_Context/03_DATA_PROTOCOL.md`
- `AI_Context/06_PIPELINE_COMPARISON.md`
- `AI_Context/07_DEEPLASIA_FOCUSED_ANALYSIS.md`
- `AI_Context/08_SEX_AWARE_EXPERIMENT_PLAN.md`
- `AI_Context/09_P12_UNCERTAINTY_PROTOCOL.md`
- `AI_Context/10_P13_THESIS_REPORTING_PLAN.md`
- `AI_Context/11_SCIENTIFIC_CONTRIBUTIONS_REPORT.md`
- `AI_Context/12_FULL_PROJECT_REPORT.md`
- `AI_Context/CHANGELOG.md`
- `AI_Context/19_REPORT_PLAN_A_D3_FINAL.md`
- `AI_Context/24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md`
- `d3_oof/outputs/D3_OOF_V1/D3_OOF_report.json`
- `d3_oof/outputs/D3_TTA_OOF_V1/D3_TTA_OOF_report.json`
- `d3_oof/outputs/FINAL_TEST_TTA_V1/FINAL_TEST_TTA_report.json`
- `c3_roi/outputs/C3_ROI_V1_OOF/C3_ROI_V1_OOF_report.json`
- `c3_roi/outputs/C3_ROI_V1_TEST/C3_E1_50_50_test_report.json`
- `research/our_baseline/RESULTS_SUMMARY.md`
- `research/our_baseline/EXPERIMENT_REGISTRY.md`
- `research/our_baseline/baseline_v1/BANG_SO_SANH_KY_THUAT_DA_SU_DUNG.md`

## 21. Việc cần làm để khóa báo cáo nhóm chính thức

1. Tái tính độc lập metric D3, EXP-008 và C3 từ prediction CSV trong một audit thống nhất.
2. Giải thích hoặc bổ sung provenance cho cảnh báo audit P7 fold 1.
3. Sửa localization C3 để giảm fallback, rồi lặp lại theo protocol khóa trước nếu tiếp tục nhánh ROI.
4. Chỉ chạy EXP-009 nếu đủ tài nguyên và dùng gate OOF 6,120448 đã định trước.
5. Khóa câu chuyện luận văn quanh E1, TTA, diversity của D3/C3, giới hạn của uncertainty và kết quả âm.
6. Ưu tiên external holdout chưa bị tác động thay vì mở thêm biến thể trên test RSNA 200.
