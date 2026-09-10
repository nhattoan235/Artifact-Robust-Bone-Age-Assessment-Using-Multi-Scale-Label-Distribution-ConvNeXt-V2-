# Nhật ký thí nghiệm bone-age: C3-ROI và các nhánh liên quan

Ngày cập nhật: 2026-09-09

Tài liệu này gom các thí nghiệm đã được thực hiện hoặc đã được chuẩn bị trong
đoạn làm việc hiện tại. Mỗi mục ghi rõ kỹ thuật, tập đánh giá và trạng thái để
không trộn lẫn validation, OOF và RSNA test.

## Quy ước đọc kết quả

| Nhãn | Ý nghĩa |
|---|---|
| Fold 1 validation | 2.808 ảnh validation của cùng split Fold 1; được dùng để sàng lọc pilot |
| Development OOF | 14.036 ảnh development, mỗi ID có đúng một dự đoán OOF; dùng để khóa TTA/ensemble |
| RSNA test | 200 ảnh test; chỉ dùng sau khi recipe đã khóa, không dùng để chọn mô hình |
| Exploratory | Kết quả thăm dò, test hoặc split đã được dùng lại; không xem là xác nhận độc lập |
| Đã chạy | Đã có checkpoint/prediction/report trong project hoặc đã được báo cáo trong chat |
| Đã chuẩn bị | Có config/code nhưng chưa có MAE thực nghiệm |

## C3-ROI dùng những kỹ thuật gì?

C3-ROI là một nhánh local bổ sung cho model global E1. C3-ROI **không phải**
sáu ROI nhỏ và bản C3-ROI gốc **không dùng LDL**.

### Tiền xử lý ảnh

1. Tạo hand mask để tìm vùng bàn tay.
2. Lấy bounding box bao quanh toàn bộ bàn tay.
3. Mở rộng bounding box bằng margin 12% trong cache Z26 V2.
4. Crop từ ảnh X-quang gốc.
5. Chuyển grayscale, pad thành hình vuông và resize về `512×512`.
6. Lặp ảnh grayscale thành 3 kênh để dùng backbone pretrained.
7. Chuẩn hóa ImageNet.
8. Nếu segmentation thất bại, dùng `global_fallback` thay vì loại ảnh.

### Mô hình và huấn luyện C3-ROI gốc

- Backbone: ConvNeXt-Tiny pretrained trên ImageNet-1K.
- Feature ảnh: 768 chiều.
- Sex embedding: `1 → 16` chiều rồi nối với feature ảnh.
- Regression head: `784 → 256 → 1`, GELU và dropout 0,2.
- Đầu ra: direct regression theo tháng, không có distribution head.
- Loss: Smooth L1, beta tương đương 3 tháng.
- Optimizer: AdamW, learning rate `2×10⁻⁴`, cosine scheduler.
- Augmentation: lật ngang, xoay ±7°, translation, scale, brightness,
  contrast và gamma.
- Input `512×512`, batch hiệu dụng 36, AMP FP16 trên T4.
- Train 5 fold; mỗi ảnh được dự đoán bởi 5 checkpoint rồi lấy trung bình.

### TTA của C3-ROI V2

TTA không tạo nhãn mới và không thay đổi trọng số model. Mỗi ảnh được chạy qua
10 view:

- góc xoay `−10°, −5°, 0°, +5°, +10°`;
- mỗi góc có bản lật ngang và không lật;
- lấy trung bình đều các dự đoán.

## Danh sách thí nghiệm

| ID | Thí nghiệm | Kỹ thuật chính | Tập đánh giá | Kết quả chính | Trạng thái |
|---|---|---|---|---:|---|
| C3-V1 | C3-ROI local/global đời đầu | Broad ROI, ConvNeXt-Tiny, sex embedding, direct regression, 5-fold | OOF 14.036 | C3 6,437349; E1+C3 50/50: 6,176212 | Đã chạy |
| C3-V2-F1 | C3-Z26 C3-ROI V2 Fold 1 | Z26 broad ROI, ConvNeXt-Tiny, sex embedding, Smooth L1 | Validation 2.808 | **6,253462** | Đã chạy; baseline A |
| C3-V2-OOF | C3-Z26 V2 5-fold | Cùng C3-V2, checkpoint best/late2, trung bình 5 fold | OOF 14.036 | baseline 6,330082; best+late2 6,288125 | Đã chạy |
| C3-V2-SWA | SWA late checkpoints | Trung bình checkpoint cuối theo fold, không train lại | OOF 14.036 | 6,288125; gain −0,041957, chưa đạt gate −0,1 | Đã chạy |
| C3-V2-TTA | C3-Z26 V2 + TTA | 5 fold × 10 view, trung bình 50 dự đoán | RSNA test 200 | raw 4,382809; TTA **4,251692** | Exploratory test |
| LDL | LDL pilot Fold 1 | Thêm label-distribution head, Gaussian sigma 2, LDL weight 0,2 | Validation 2.808 | 50/50: 6,3459; 66/34: 6,3421; regression-only: 6,3630 | Đã chạy; không vượt baseline |
| Seed-2026 | Đổi seed | Giữ nguyên C3-V2, chỉ đổi seed từ 42 sang 2026 | Validation 2.808 | 6,359650 | Đã chạy; không đạt gate |
| Seed ensemble | Seed 42 + 2026 | Trung bình hai dự đoán Fold 1 | Validation 2.808 | 6,187530 | Đã chạy; cần 5-fold để kết luận |
| Ordinal | Ordinal auxiliary head | Thêm head học thứ tự tuổi, inference regression-only | Validation 2.808 | 6,337441 | Đã chạy; không đạt gate |
| Ordinal ensemble | Baseline + ordinal | Trung bình prediction baseline và ordinal | Validation 2.808 | **6,163007**, gain 0,090456 | Đã chạy; dưới gate gain 0,1 |
| E1-TTA | Model global + TTA | Global/full-hand ConvNeXt-Tiny, sex embedding, 10-view TTA | OOF/test | OOF 6,210446; test 4,617492 | Đã chạy |
| D3-TTA | Global LDL + TTA | ConvNeXt-Tiny, sex embedding, label distribution, 10-view TTA | OOF/test | OOF 6,246542; test 4,508460 | Đã chạy |
| OOF ensemble | C3 + E1 + D3 | Trung bình 1/3 dự đoán OOF | OOF/development | **6,044192** | Đã chạy |
| Test ensemble | C3 + E1 + D3 | Trung bình 1/3 dự đoán test 200 ảnh | RSNA test 200 | 4,348207 | Exploratory test |
| Rescue V1/V2 | Sửa ảnh fallback | Candidate rescue cho ảnh segmentation lỗi, giữ split/model | OOF fallback audit | V1 6,368042 → V2 6,330082; delta −0,037960 | Đã chạy; CI chứa 0 |
| Cleaned artifact | So sánh ảnh gốc/làm sạch | Đánh giá ảnh cleaned/inpainting trên test cũ | RSNA test 200 | khoảng 4,3364 → 4,2189 | Exploratory; không dùng chọn model |
| Pilot B | Artifact augmentation | Ghép ảnh sạch và ảnh có artifact nhẹ, supervised loss trên cả hai view | Chưa chạy | Chưa có MAE | Đã chuẩn bị |
| Pilot C | Artifact + consistency | Pilot B + phạt sai khác prediction giữa hai view, weight 0,30 | Chưa chạy | Chưa có MAE | Đã chuẩn bị |

## Chi tiết từng thí nghiệm

### C3-V1 — broad ROI local/global đời đầu

Đây là C3-ROI trước Z26 V2. Pipeline dùng hand mask, bounding box toàn bàn tay,
margin 8%, fallback toàn ảnh, ConvNeXt-Tiny pretrained, sex embedding và direct
regression. C3 được train độc lập trên 5 fold với Smooth L1 và AdamW.

| Mô hình | OOF MAE |
|---|---:|
| E1 global | 6,316691 |
| C3-ROI riêng | 6,437349 |
| E1 + C3-ROI, 50/50 | **6,176212** |

C3 đứng riêng kém E1 nhưng có prediction diversity đủ để ensemble cải thiện.
Fallback development là 18,49% và test là 33%, nên đây không phải local
specialist thuần túy.

### C3-V2-F1 — baseline A đã khóa

C3-Z26 V2 dùng cùng ý tưởng broad hand ROI nhưng cache Z26 V2, margin 12%,
fallback được ghi trong manifest. Model là ConvNeXt-Tiny direct regression với
sex embedding 16 chiều, light augmentation, Smooth L1, AdamW và cosine
scheduler. Baseline Fold 1, seed 42 đạt:

```text
MAE = 6.253462484419516 tháng
```

Đây là baseline để so sánh LDL, seed 2026, ordinal và hai pilot artifact.

### C3-V2-OOF và SWA

SWA được kiểm tra bằng cách trung bình các checkpoint cuối mỗi fold, không train
lại và không đọc test. Kết quả `best_late2` giảm OOF MAE từ 6,330082 xuống
6,288125 tháng, nhưng gain chỉ 0,041957 tháng, nhỏ hơn gate 0,1 tháng. Vì vậy
SWA không được chọn làm cải tiến chính.

### C3-V2-TTA trên RSNA test

Recipe được khóa từ OOF rồi mới chạy test. C3-V2 dùng 5 checkpoint, mỗi ảnh có
10 view TTA, tổng cộng 50 dự đoán và lấy trung bình:

| Phiên bản | MAE |
|---|---:|
| C3-V2 raw 5-fold | 4,382809 |
| C3-V2 + TTA | **4,251692** |

TTA giảm 0,131117 tháng trên 200 ảnh. Bootstrap paired CI là
`[-0,311471; 0,046141]`, nên điểm ước lượng tốt nhưng chưa đủ để khẳng định
TTA luôn tốt hơn. So với mốc bài báo 2025 khoảng 4,42 tháng, C3-V2 TTA thấp
hơn khoảng 0,1683 tháng, tương đương khoảng 5 ngày; đây vẫn là kết quả test
thăm dò vì RSNA test chỉ có 200 ảnh.

### LDL pilot

LDL giữ backbone/ROI/sex embedding của C3 nhưng thêm distribution head 229 lớp.
Tuổi thật được biến thành Gaussian label distribution với sigma 2; loss tổng hợp
regression loss và distribution loss với LDL weight 0,2. Khi inference, regression
và phân phối tuổi được trộn theo regression weight.

Kết quả Fold 1:

| Cấu hình | MAE |
|---|---:|
| Baseline | 6,2535 |
| LDL 50/50 | 6,3459 |
| LDL 66/34 | 6,3421 |
| LDL regression-only | 6,3630 |

LDL làm xấu MAE trong recipe này; không nên gọi C3-ROI gốc là LDL model.

### Seed 2026

Seed 2026 giữ nguyên split, kiến trúc, augmentation và loss của C3-V2, chỉ đổi
random seed. Best epoch 22 đạt MAE 6,359650, kém baseline 0,106188 tháng.
Paired bootstrap CI của candidate − baseline là `[0,0025577; 0,2134585]`, nên
không đạt gate cải thiện.

Trung bình seed 42 và seed 2026 đạt 6,187530, tốt hơn baseline 0,0659326 tháng
với CI `[-0,120317; -0,010090]`. Tuy nhiên đây mới là Fold 1; chưa đủ để dùng
làm kết quả chính 5-fold.

### Ordinal auxiliary head

Ordinal pilot thêm một head học các mốc thứ tự tuổi. Head này cung cấp tín hiệu
"tuổi lớn hơn/nhỏ hơn" trong lúc train, còn inference được đánh giá bằng
regression-only để tách ảnh hưởng của head phụ.

| Mô hình | MAE |
|---|---:|
| Baseline | 6,253462 |
| Ordinal | 6,337441 |
| Baseline + Ordinal 50/50 | **6,163007** |

Ordinal đứng riêng kém baseline. Ensemble có lợi 0,090456 tháng nhưng dưới gate
0,1 tháng và Fold 1 đã được dùng để chọn/đánh giá, nên chưa phải bằng chứng
generalization.

### E1-TTA, D3-TTA và ensemble

E1 là model global/full-hand. D3 là nhánh ConvNeXt-Tiny có sex embedding và
label-distribution auxiliary head. Cả hai dùng TTA 10 view.

Trên OOF development:

| Model | OOF MAE |
|---|---:|
| C3-V2 | 6,288125 |
| E1-TTA | 6,210446 |
| D3-TTA | 6,246542 |
| C3 + E1, 50/50 | 6,094127 |
| C3 + D3, 50/50 | 6,094049 |
| C3 + E1 + D3, 1/3 | **6,044192** |

Trên RSNA test 200 ảnh:

| Model | Test MAE |
|---|---:|
| C3-V2 TTA | **4,251692** |
| E1-TTA | 4,617492 |
| D3-TTA | 4,508460 |
| C3 + E1 + D3, 1/3 | 4,348207 |

OOF ensemble rất tốt nhưng test triple ensemble lại kém C3-V2 TTA riêng. Vì
vậy không được suy ra rằng ensemble OOF chắc chắn tốt hơn trên test.

### Rescue segmentation V1/V2

Nhánh rescue chỉ tác động vào ảnh bị segmentation lỗi hoặc fallback. V2 thay
candidate rescue và đánh giá paired trên cùng OOF, không thay đổi model backbone.

- V1: 6,368042 tháng.
- V2: 6,330082 tháng.
- Delta: −0,037960 tháng.
- Bootstrap CI: `[-0,088376; 0,012097]`.
- V2 cứu được 588 ảnh; vẫn còn 1.386 fallback trên 14.036 ảnh.

Kết quả gợi ý giảm fallback có ích về mặt pipeline nhưng chưa phải hướng chính
để đạt mục tiêu MAE.

### Pilot B và C đã chuẩn bị

Hai pilot dùng cùng Fold 1, seed 42 và không dùng RSNA test để chọn model.

**Pilot B — augmentation-only control**

- Tạo ảnh sạch và một ảnh artifact nhẹ từ cùng sample.
- Artifact gồm thay đổi brightness/contrast/gamma, blur/noise nhỏ và band ở
  rìa ảnh; không che vùng trung tâm bàn tay.
- Tính supervised regression/LDL loss trên cả hai view.
- `consistency_weight = 0`.

**Pilot C — consistency regularization**

- Giữ nguyên toàn bộ Pilot B.
- Thêm `|prediction_clean − prediction_artifact|` vào loss.
- `consistency_weight = 0,30`.

Output được tách thành:

```text
C3_Z26_C3_ROI_V2_PILOTS/runs/
  C3_Z26_C3_ROI_V2_PILOT_B_FOLD_1_SEED_42/
  C3_Z26_C3_ROI_V2_PILOT_C_FOLD_1_SEED_42/
```

Hai pilot chưa có MAE; cần đánh giá clean validation và paired artifact
validation trước khi mở rộng sang các fold còn lại.

## Kết luận hiện tại

1. **Kết quả đơn mạnh nhất trên RSNA test hiện tại:** C3-V2 + TTA, MAE
   4,251692 tháng.
2. **Kết quả OOF tốt nhất:** C3 + E1 + D3, 1/3, MAE 6,044192 tháng; chưa
   chuyển thành kết quả test tốt hơn C3-V2 TTA.
3. **Cải tiến có bằng chứng ổn định nhất trong development:** TTA và ensemble
   C3/E1/D3 trên OOF, nhưng phải giữ tách biệt với test.
4. **LDL, seed 2026 và ordinal standalone** chưa vượt baseline Fold 1.
5. **SWA và rescue** có cải thiện nhỏ nhưng chưa đạt gate 0,1 tháng.
6. **Pilot B/C** là hướng tiếp theo có ý nghĩa phương pháp: kiểm tra robustness
   với artifact tổng hợp và consistency, thay vì chỉ điều chỉnh prediction để
   giảm MAE.

## Các giới hạn phải ghi trong báo cáo

- RSNA test chỉ có 200 ảnh; CI của MAE rất rộng.
- Một số kết quả test đã được chạy sau nhiều vòng phát triển, nên phải gọi là
  exploratory nếu recipe không được khóa trước từ OOF.
- C3 có global fallback; không nên mô tả là local-only model.
- OOF ensemble và test ensemble không được trộn thành một chỉ số.
- Không dùng kết quả trên RSNA test để chọn trọng số, chọn checkpoint hoặc quyết
  định giữ pilot.

## Artifact và mã liên quan

- C3-Z26 V2 TTA report: `c3_roi/artifacts/RSNA_DATA_BACKUP_20260909/C3_Z26_C3_ROI_V2_TTA_TEST/`
- C3-Z26 V2 SWA report: `c3_roi/artifacts/RSNA_DATA_BACKUP_20260909/C3_Z26_C3_ROI_V2_SWA_OOF_V1/`
- LDL run: `c3_roi/artifacts/RSNA_DATA_BACKUP_20260909/C3_Z26_C3_ROI_V2_LDL_V1/`
- Seed comparison: `c3_roi/artifacts/RSNA_DATA_BACKUP_20260909/C3_Z26_C3_ROI_V2/seed_pilot/seed_2026_fold1_comparison.json`
- Ordinal comparison: `c3_roi/artifacts/RSNA_DATA_BACKUP_20260909/loose_downloads/fold1_comparison.json`
- C3-ROI method explanation: `docs/c3-roi-explainer.md`
- Pilot B/C bundle: `C3_Z26_C3_ROI_V2_PILOT_BC_CODE_V1.zip`
