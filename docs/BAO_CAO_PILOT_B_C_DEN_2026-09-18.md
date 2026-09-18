# Báo cáo Pilot B/C của C3-ROI V2 đến 18/09/2026

## Đọc nhanh

Hai pilot kiểm tra cách làm mô hình dự đoán tuổi xương bền hơn khi ảnh X-quang bị thay đổi chất lượng. **B** học từ cặp ảnh sạch và ảnh có artifact tổng hợp nhẹ; **C** làm giống B và thêm ràng buộc để hai dự đoán của cùng một ảnh gần nhau. B đã train ở **Fold 1 và Fold 5**; C đã train **đủ 5 fold**. Đây là những model mới được train riêng, không phải TTA và không phải tiếp tục train checkpoint baseline.

Trên 14.036 ảnh validation OOF, C giảm MAE khi có artifact từ **7,2378 xuống 6,5227 tháng** và giảm độ lệch dự đoán giữa ảnh sạch/artifact **64,1%**. Đổi lại, MAE ảnh sạch tăng từ **6,3300 lên 6,3707** và trẻ dưới 120 tháng chịu thiệt, rõ nhất ở nhóm 0–59 tháng. Blend cố định `27% baseline + 73% C` đạt **6,2725 clean / 6,4879 artifact**, nhưng sạch ở 0–59 tháng vẫn xấu hơn baseline **0,1960 tháng**. B Fold 5 chứng minh thêm rằng chỉ artifact augmentation cũng có thể gây hại cho trẻ rất nhỏ; consistency weight 0,30 của C còn làm nhóm 60–119 tháng Fold 5 tệ hơn B.

**Quyết định hiện tại:** giữ C và blend như kết quả nghiên cứu về *synthetic artifact robustness*, chưa thay baseline C3-V2 bằng C cho mọi tuổi. Test 200 ảnh cùng TTA 10-view vừa chạy cho MAE B/C cao hơn các đối chứng tương ứng, nhưng CI của mọi chênh lệch đều chứa 0; đây là kết quả thăm dò. Gói YOUNG2 được chuẩn bị sau đó là ý tưởng train **khác**, chưa chạy và không thuộc kết quả B/C.

## 1. Mốc so sánh và ý nghĩa các tập dữ liệu

Baseline là C3-Z26 C3-ROI V2 đã khóa: ConvNeXt-Tiny nhận ảnh ROI bàn tay 512 px và giới tính, rồi **hồi quy trực tiếp tuổi xương**. Baseline và B/C không dùng head phân phối nhãn; các trường LDL còn có trong một số config B/C không hoạt động khi `architecture = "convnext_tiny"`. Cùng một chia 5 fold và seed 42 được dùng để đối chiếu B/C. Mỗi ảnh chỉ được chấm bởi model của fold mà nó thuộc validation; ghép 5 tập validation tạo thành **OOF 14.036 ảnh**. `Clean` là ảnh chuẩn; `artifact` là phiên bản cùng ảnh có suy giảm tổng hợp `mild_v1`, cùng nhãn tuổi.

MAE **6,3300** là của baseline trên *clean OOF*. MAE **4,251692** là kết quả khác: baseline 5-fold với **10-view TTA** trên *RSNA test 200 ảnh*, đã từng được xem nhiều lần nên chỉ coi là test thăm dò. Không so sánh trực tiếp 6,3300 với 4,251692 để kết luận phương pháp nào thắng: khác tập ảnh và khác cách inference. Phần validation OOF của B/C không dùng test; kết quả test thăm dò được trình bày riêng ở dưới.

`Delta` dưới đây luôn là `MAE ứng viên − MAE baseline/đối chứng`: **âm là tốt hơn**. CI 95% là khoảng bootstrap ghép cặp theo cùng `image_id`; một CI chứa 0 chưa đủ bằng chứng cho chênh lệch rõ ràng. `Disagreement` là trung bình `|dự đoán ảnh artifact − dự đoán ảnh sạch|`, không phải MAE so với tuổi thật.

## 2. B và C đã dùng kỹ thuật gì?

| Model | Khi train | Điểm khác nhau |
|---|---|---|
| Baseline | Ảnh sạch, augmentation thông thường | Mốc so sánh |
| Pilot B | Mỗi mẫu có ảnh sạch và ảnh artifact `mild_v1`; cả hai đều chịu loss dự đoán tuổi thật | Artifact augmentation, `consistency_weight = 0` |
| Pilot C | Giống B | Thêm `0,30 × |prediction_clean − prediction_artifact|` vào loss, sau warm-up/ramp |

`mild_v1` thay đổi nhẹ độ sáng, tương phản, gamma, blur, nhiễu và một dải ở mép ảnh; không cố ý che phần giữa của bàn tay. Tác động artifact được warm-up 3 epoch rồi tăng dần trong 5 epoch. B và C giữ cùng architecture, dữ liệu, chia fold, seed 42, optimizer, lịch learning rate và tiêu chí chọn checkpoint theo **clean validation MAE**. Vì vậy so sánh B với C trên cùng fold giúp tách tác động của consistency penalty; so sánh B với baseline đo tác động chung của cách train bằng artifact. Đây là artifact giả lập, chưa kiểm nghiệm ảnh từ bệnh viện khác.

**TTA là kỹ thuật lúc inference**, xoay/lật ảnh rồi trung bình dự đoán; nó không phải artifact augmentation và cũng không phải consistency loss. Các chỉ số B/C clean/artifact ở đây là đánh giá validation theo view đã khóa, không phải test TTA 10-view.

## 3. Diễn biến và kết quả theo thứ tự

### B và C trên Fold 1: kiểm tra ý tưởng

| Fold 1, n = 2.808 | Baseline | B | C |
|---|---:|---:|---:|
| Clean MAE (tháng) | 6,2535 | 6,2984 | 6,2751 |
| Artifact MAE (tháng) | 7,1521 | 6,4835 | 6,3925 |

B làm artifact MAE tốt hơn baseline **0,6686 tháng** (CI delta B − baseline khoảng `[-0,8340; -0,5041]`), nhưng clean kém hơn **0,0449 tháng**. C tốt hơn baseline trên artifact **0,7597 tháng**, nhưng so trực tiếp C với B thì mức tốt hơn **0,0910 tháng** có CI `[-0,1968; +0,0163]`: chưa kết luận consistency là phần đem lại lợi ích riêng. C được mở rộng lên 5 fold để kiểm tra độ ổn định.

### C trên đủ 5 fold: đánh giá OOF chính

| Trên 14.036 ảnh | Baseline | C | Delta C − baseline | CI 95% delta |
|---|---:|---:|---:|---|
| Clean MAE | 6,3300 | 6,3707 | +0,0406 | `[-0,0065; +0,0863]` |
| Artifact MAE | 7,2378 | **6,5227** | **−0,7151** | `[-0,7984; −0,6332]` |
| Clean–artifact disagreement | 3,0933 | **1,1090** | giảm 64,1% | — |

C cải thiện rõ với artifact tổng hợp, nhưng **không cải thiện clean MAE**. Gate về độ bền và độ ổn định đạt; gate theo nhóm tuổi không đạt. Trên nhóm 0–59 tháng (895 ảnh), C clean xấu hơn baseline **+0,4976 tháng**, artifact **+0,3015**. Nhóm 60–119 tháng (3.881 ảnh) clean xấu hơn **+0,2358**, dù artifact tốt hơn **−0,3054**. Chỉ số toàn bộ dễ che các nhóm này vì nhóm 120–179 tháng có 8.033 ảnh.

### Thử blend baseline và C trên chính OOF

Blend dự đoán bằng `0,27 × baseline + 0,73 × C`. Tỷ lệ 73% được tìm trên OOF nên kết quả cố định 0,73 là **thăm dò**, không phải xác nhận trên dữ liệu độc lập. Cross-fit chọn tỷ lệ từ bốn fold rồi chấm fold còn lại cho ra trọng số C `0,80 / 0,77 / 0,71 / 0,65 / 0,71`; clean OOF gộp **6,2728**, artifact **6,4985**. Cross-fit cũng không giải quyết hết vấn đề nhóm tuổi: clean delta tệ nhất **+0,2084 tháng**, vượt ngưỡng +0,20 đã đặt.

| OOF, 14.036 ảnh | Baseline | Blend 27/73 | Delta |
|---|---:|---:|---:|
| Clean MAE theo ảnh | 6,3300 | **6,2725** | **−0,0575**, CI `[-0,0925; −0,0229]` |
| Artifact MAE theo ảnh | 7,2378 | **6,4879** | **−0,7499**, CI `[-0,8151; −0,6859]` |
| Clean–artifact disagreement | 3,0933 | 1,3722 | giảm 55,6% |
| Clean MAE cân bằng bốn nhóm tuổi | **6,2597** | 6,3056 | +0,0458, CI `[-0,0180; +0,1093]` |
| Artifact MAE cân bằng bốn nhóm tuổi | 6,9645 | **6,5090** | −0,4555 |

MAE “theo ảnh” cho mỗi ảnh trọng số bằng nhau; MAE “cân bằng tuổi” cho mỗi nhóm 0–59, 60–119, 120–179, 180–228 trọng số **25%**. Blend có lợi cho MAE clean theo ảnh, nhưng **không chứng minh lợi ích clean khi cân bằng tuổi**. Chi tiết: 0–59 clean `+0,1960` (CI `[+0,0064; +0,3845]`), 60–119 `+0,0827` (CI `[+0,0110; +0,1536]`), 120–179 `−0,1743`, 180–228 `+0,0790` (CI chứa 0).

### Kiểm tra ba seed artifact

Giữ nguyên các model và blend 27/73, chỉ sinh lại artifact bằng ba seed. Đây **không phải train ba bộ model mới**.

| Seed sinh artifact | Baseline MAE | Blend MAE | Lợi ích blend |
|---:|---:|---:|---:|
| 20260912 | 7,2378 | 6,4879 | 0,7499 tháng |
| 20260913 | 7,3070 | 6,5333 | 0,7736 tháng |
| 20260914 | 7,2604 | 6,5055 | 0,7549 tháng |

Lợi ích artifact lặp lại ở ba lần sinh thuộc **cùng họ `mild_v1`**. Điều này hỗ trợ tính ổn định trong họ artifact đó, chưa phải bằng chứng khái quát sang thiết bị/bệnh viện khác.

### Vì sao kiểm tra thêm Fold 5?

Phân tích OOF cho thấy Fold 5 là điểm xấu nhất của blend trên ảnh sạch: baseline **6,1823**, C **6,4305**, blend **6,2598**. Riêng 60–119 tháng, baseline **7,4161**, C **8,2461**, blend **7,9140**. Sự xấu đi trải trên nhiều ảnh, không chỉ vài outlier. Vì vậy đã train thêm **B Fold 5** làm đối chứng với C Fold 5, cùng split và seed, chỉ bỏ consistency penalty; không train lại C.

Theo kết quả Colab bạn gửi ngày 18/09, so sánh ghép cặp trên Fold 5 như sau:

| Fold 5, toàn bộ 2.807 ảnh | Baseline | B | C | Nhận xét |
|---|---:|---:|---:|---|
| Clean MAE | ~6,1822 | 6,2954 | 6,4305 | B − baseline `+0,1132`, CI `[+0,0203; +0,2070]`; C − B `+0,1350`, CI `[+0,0210; +0,2523]` |
| Artifact MAE | ~7,1458 | 6,5291 | 6,6374 | B cải thiện baseline `−0,6167`; C − B `+0,1083`, CI `[-0,0328; +0,2467]` |

Ở **60–119 tháng**, B clean **7,3863**, gần baseline **7,4161** (delta `−0,0298`, CI chứa 0), còn C clean **8,2461**: C − B **+0,8598** (CI `[+0,6423; +1,0764]`). Artifact cũng vậy: B **7,4905**, C **8,5176**, C − B **+1,0271** (CI `[+0,7466; +1,3029]`). Trong lát cắt này, consistency `0,30` là ứng viên giải thích sự sa sút thêm của C so với B; không nên kết luận từ riêng Fold 5 rằng nó luôn có hại ở mọi fold.

Nhưng **0–59 tháng** cho thấy chỉ giảm consistency vẫn chưa sửa được: ở Fold 5, B − baseline clean **+1,0130** (CI `[+0,4755; +1,5564]`), artifact **+1,7681** (CI `[+0,7617; +2,8083]`). MAE cân bằng tuổi Fold 5 trên ảnh sạch là baseline **6,3593**, B **6,6250**, C **6,8006**; trên artifact lần lượt **7,0513 / 7,0505 / 7,1256**. Đây là lý do không chọn B hay C thay baseline chỉ vì MAE artifact toàn bộ đẹp.

### Phân tích lỗi trẻ nhỏ và xem ảnh

Với 0–59 tháng ở Fold 5, B có signed bias (dự đoán trừ tuổi thật) **+5,983 tháng clean / +6,888 artifact**, so với baseline **+3,982 / +0,589**. B dự đoán quá tuổi **79,9%** ảnh clean và **77,7%** ảnh artifact. Nhóm **0–23 tháng chỉ có 23 ảnh** ở Fold 5 nhưng B − baseline artifact lên tới **+6,655 tháng**; nhóm 24–59 tháng có 156 ảnh, artifact **+1,048**. Fold 1 không lặp mức lỗi này: B − baseline nhóm 0–59 là clean **+0,101**, artifact **−0,138**; do đó có biến thiên giữa fold đáng kể.

Số ảnh 0–59 dùng `global_fallback` gần bằng nhau ở Fold 1 và Fold 5 (**42/179** và **41/179**), nên Fold 5 không xấu đơn giản vì có nhiều fallback hơn. Ở Fold 5, B − baseline artifact là **+2,444** trên 41 ảnh fallback và **+1,567** trên 138 ảnh `mask_bbox`: vấn đề xuất hiện ở cả hai đường ROI. Trên ảnh montage do bạn gửi, ví dụ ID 4119 (3 tháng) B đổi từ **18,0 clean** thành **52,2 artifact**, còn ID 7110 (15 tháng) B đã dự đoán **36,5 clean** và **37,1 artifact**. Như vậy có cả lỗi nhạy với biến đổi ảnh và lỗi đã tồn tại ở clean. Phần nền, marker và độ phơi sáng trên vài ảnh fallback là **giả thuyết** về nguồn tín hiệu sai; montage không đủ để chứng minh quan hệ nhân quả.

### Test RSNA 200 ảnh với cùng TTA 10-view (18/09)

Sau khi hoàn tất phân tích validation, B/C được chạy inference trên đúng 200 ảnh test với 5 góc xoay × lật/không lật. C có đủ 5 fold; B chỉ có Fold 1 và 5, nên bảng dùng các đối chứng có **cùng số fold**. Delta dương là ứng viên có MAE cao hơn.

| So sánh | MAE đối chứng | MAE ứng viên | Delta | CI 95% delta |
|---|---:|---:|---:|---|
| C 5-fold so với baseline 5-fold | 4,2517 | 4,4029 | +0,1512 | `[-0,0553; +0,3584]` |
| B Fold 1+5 so với baseline Fold 1+5 | 4,2053 | 4,4128 | +0,2076 | `[-0,0508; +0,4635]` |
| C Fold 1+5 so với B Fold 1+5 | 4,4128 | 4,4690 | +0,0562 | `[-0,1861; +0,2992]` |
| C Fold 1+5 so với baseline Fold 1+5 | 4,2053 | 4,4690 | +0,2637 | `[-0,0227; +0,5551]` |

Baseline có MAE thấp nhất trong mỗi so sánh với B/C theo số fold tương ứng. Tuy nhiên **cả bốn CI đều cắt 0**, nên trên 200 ảnh này chưa chứng minh được mức khác biệt rõ ràng. Kết quả clean test không xóa đi lợi ích artifact tổng hợp trên OOF, nhưng cũng không cho thấy B/C cải thiện clean test. MAE 4,2053 của baseline hai fold không được xem là model mới tốt hơn baseline năm fold: chỉ số này được tính để đối chiếu công bằng với B và không được dùng để chọn ensemble bằng nhãn test.

Các con số test ở đây được chép từ output Colab bạn gửi. CSV và report cuối được xác nhận tồn tại trên Drive tại `RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/TEST_TTA_10VIEW_EXPLORATORY/`; **chưa nhập bản gốc hai file này vào project**. Test 200 ảnh đã được xem trong các vòng nghiên cứu trước, vì vậy kết quả chỉ mang tính thăm dò, không phải xác nhận độc lập.

## 4. Những gì đã làm, chưa làm và cách đọc kết quả

| Việc | Trạng thái đến hiện tại |
|---|---|
| Train B Fold 1, C Fold 1–5, B Fold 5 | Đã chạy trên Colab theo kết quả bạn gửi |
| Đánh giá B/C clean và artifact cùng Fold 1, Fold 5 | Đã chạy; Fold 5 có paired bootstrap B/C/baseline |
| C 5-fold OOF, fixed blend, cross-fit blend, balanced MAE, ba seed artifact | Đã chạy và có report/prediction được lưu trong project |
| Tách từng loại artifact trên Fold 5 | **Bỏ qua theo lựa chọn của bạn**; chưa có ma trận loại artifact |
| B đủ 5-fold OOF | **Chưa có**; thiếu B Fold 2–4 |
| B/C trên RSNA test 200 ảnh, cùng recipe TTA 10-view | Đã chạy và lưu CSV/report trên Drive; số liệu từ output bạn gửi. Chưa có bản gốc trong project. Kết quả là thăm dò |
| External validation bệnh viện khác | Chưa có |
| Gói YOUNG2 | Chỉ mới chuẩn bị local, **chưa train, chưa có MAE**; không phải bước của B/C |

Hướng dẫn và code inference nằm tại [PILOT_BC_TEST_TTA_10VIEW_README.md](../c3_roi/PILOT_BC_TEST_TTA_10VIEW_README.md). Script kiểm tra baseline 200 ID và lưu cache từng fold lên Drive.

Các bảng Fold 5, tuổi 0–23 và nhận xét montage phía trên lấy từ **output Colab và ảnh bạn gửi trong cuộc trò chuyện**. Hiện `results/pilot_bc_20260914/` trong project là gói evidence đến 14/09, **chưa bao gồm report/prediction gốc của B Fold 5**. Vì vậy số Fold 5 cần được đối chiếu lại khi tải `pilot_b_c_baseline_fold_5_report.json`, file predictions và `artifact_robustness_report.json` về project; đừng coi bản chép số trong báo cáo này là bản lưu dữ liệu gốc.

## 5. Quyết định và việc tiếp theo

1. **Giữ baseline C3-V2 làm mốc an toàn cho clean/nhóm tuổi nhỏ.** Báo cáo C và blend như cải thiện độ bền với artifact giả lập, đồng thời công bố đầy đủ clean subgroup và MAE cân bằng tuổi. Chưa nâng B, C hay blend thành model thắng chung.
2. **Lưu bằng chứng Fold 5 vào project:** report/prediction của B và bảng paired comparison B/C/baseline; lưu cả montage đã xem. Điều này giúp kiểm tra lại số liệu và audit được các ảnh lỗi.
3. **Nếu muốn sửa lỗi trẻ nhỏ, tạo giả thuyết mới và đánh giá bằng validation/OOF trước:** xem sampling theo tuổi, chất lượng ROI và hai nguồn lỗi riêng (bias trên ảnh clean, độ nhạy artifact). Một thí nghiệm mới phải có cấu hình/tiêu chí dừng khóa trước và đo theo cả 0–23, 24–59, 60–119 tháng, MAE toàn bộ và MAE cân bằng tuổi. Gói YOUNG2 hiện chỉ là đề xuất thử sampling, không có kết quả và không tự động được chọn.
4. **Lưu hai file test 200 ảnh từ Drive vào project** cùng hash/checkpoint provenance khi thuận tiện. Không dùng kết quả này để chỉnh trọng số, view TTA hay checkpoint; cần tập xác nhận độc lập để chứng minh cải thiện thực sự.

## Nguồn kiểm tra trong project

- [Báo cáo B/C đến 14/09](PILOT_BC_FINAL_RESULTS_20260914.md): cấu hình, Fold 1, C 5-fold, blend, multi-seed, subgroup.
- [Phân tích lỗi C và kế hoạch Fold 5](PILOT_C_FAILURE_ANALYSIS_AND_FOLLOWUP_20260914.md): clean degradation, balanced MAE, lý do train B Fold 5. Phần “prepared” trong file cũ đã được thực hiện sau đó theo output bạn gửi.
- [Bằng chứng đã lưu](../results/pilot_bc_20260914/VALIDATION_SUMMARY.json), [C OOF](../results/pilot_bc_20260914/oof/C3_Z26_C3_ROI_V2_PILOT_C_OOF_report.json), [blend cross-fit](../results/pilot_bc_20260914/oof/blend/pilot_c_blend_crossfit_report.json), [balanced OOF](../results/pilot_bc_20260914/balanced_oof/balanced_oof_report.json), [multi-seed](../results/pilot_bc_20260914/oof/multiseed/multiseed_artifact_report.json).
- [Config B](../c3_roi/pilot_configs/C3_Z26_C3_ROI_V2_PILOT_B_FOLD_5_SEED_42.toml), [config C](../c3_roi/pilot_configs/C3_Z26_C3_ROI_V2_PILOT_C_FOLD_5_SEED_42.toml), [cách tính loss](../p1_baseline/trainer.py), [cách tạo artifact](../p1_baseline/artifacts.py), [script so B/C](../c3_roi/compare_pilot_b_c_fold.py).
