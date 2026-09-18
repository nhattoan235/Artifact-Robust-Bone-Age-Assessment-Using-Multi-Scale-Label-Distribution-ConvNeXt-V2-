# Chốt nghiên cứu C3-ROI V2 và Pilot B/C

**Ngày chốt: 18/09/2026 · Trạng thái: kết luận nội bộ trên dữ liệu hiện có.** Tài liệu này hợp nhất [báo cáo B/C chi tiết](BAO_CAO_PILOT_B_C_DEN_2026-09-18.md) với [audit baseline và kế hoạch MAE 4,2](AUDIT_VA_KE_HOACH_MAE_4_2_2026-09-18.md). Hai file nguồn vẫn được giữ để truy xuất phép đo, code và lịch sử. “Chốt” ở đây nghĩa là khóa cách diễn giải kết quả hiện tại và quyết định nghiên cứu tiếp theo; **không** có nghĩa đã chứng minh mô hình đạt MAE ≤4,2 hoặc đã được xác nhận ngoài bệnh viện.

## Câu hỏi, phương pháp và mốc đúng

Câu hỏi B/C là: *train bằng ảnh có artifact tổng hợp có giúp mô hình bền hơn mà không làm hại độ chính xác trên ảnh sạch, đặc biệt ở trẻ nhỏ không?* Mốc đối chứng **C3-Z26 C3-ROI V2** dùng ROI bàn tay 512 px (fallback ảnh toàn cục khi mask không đạt), ConvNeXt-Tiny pretrained, sex embedding 16 chiều, một head **hồi quy trực tiếp** tuổi xương, Smooth L1, AdamW, cosine LR, 5 fold seed 42. Đây **không phải** ConvNeXt V2 hay multi-scale LDL. Các tham số LDL nằm trong vài config pilot không kích hoạt head LDL khi `architecture = "convnext_tiny"`. [Config baseline đóng băng](../c3_roi/outputs/C3_Z26_C3_ROI_T4_CODE_V2/C3_Z26_C3_ROI_V2/configs_t4_b36/fold_1.toml), [model](../p1_baseline/model.py) và [config C Fold 5](../c3_roi/pilot_configs/C3_Z26_C3_ROI_V2_PILOT_C_FOLD_5_SEED_42.toml) là nguồn xác minh.

- **A / baseline:** train ảnh sạch với augmentation nhẹ.
- **B:** train cả ảnh sạch và phiên bản `mild_v1` cùng nhãn tuổi; không có consistency penalty. Đã train Fold 1 và Fold 5.
- **C:** giống B nhưng thêm penalty giữ dự đoán clean/artifact gần nhau, `consistency_weight = 0,30`. Đã train đủ 5 fold.

Artifact `mild_v1` là thay đổi độ sáng, tương phản, gamma, blur, nhiễu và dải mép ảnh; warm-up 3 epoch, ramp 5 epoch. Checkpoint B/C được chọn bằng **clean validation MAE**. Các model B/C được train mới, không phải TTA. TTA 10 view chỉ dùng lúc inference: 5 góc xoay × lật/không lật, lấy trung bình prediction. Baseline test dùng 5 fold × 10 view; so sánh B hai fold dùng baseline và C đúng hai fold tương ứng.

`OOF` gồm 14.036 ảnh phát triển: mỗi ảnh được dự đoán bởi model của fold mà nó nằm trong validation. `Test` là 200 ảnh RSNA riêng. MAE của hai tập **không so trực tiếp** vì tập ảnh và cách suy luận khác nhau. Delta luôn là `MAE ứng viên − MAE đối chứng`; âm là cải thiện.

## Bảng kết quả để trích dẫn

| Đánh giá | Đối chứng | Ứng viên | Kết quả | Điều được phép kết luận |
|---|---:|---:|---|---|
| Fold 1 clean, n=2.808 | A 6,2535 | B 6,2984 / C 6,2751 | B +0,0449; C +0,0216 tháng | Chưa có gain clean trên pilot đầu |
| Fold 1 artifact `mild_v1` | A 7,1521 | B 6,4835 / C 6,3925 | B cải thiện 0,6686; C 0,7597 | Train artifact giúp trong bài kiểm tra tổng hợp này |
| **OOF 5 fold clean**, n=14.036 | A **6,3300** | C 6,3707 | C − A = **+0,0406**, CI 95% [−0,0065; +0,0863] | **Không chứng minh C tốt hơn A trên clean** |
| **OOF 5 fold artifact** | A 7,2378 | C **6,5227** | C − A = **−0,7151**, CI [−0,7984; −0,6332] | C cải thiện rõ trên artifact cùng họ `mild_v1` |
| OOF clean/artifact disagreement | A 3,0933 | C 1,1090 | giảm **64,1%** | Dự đoán của C ít đổi hơn giữa hai view; đây không phải MAE |
| OOF blend cố định A 27% + C 73%, clean | A 6,3300 | blend 6,2725 | −0,0575, CI [−0,0925; −0,0229] | Tín hiệu OOF thăm dò; weight 0,73 được chọn trên OOF |
| OOF blend cố định, artifact | A 7,2378 | blend 6,4879 | −0,7499, CI [−0,8151; −0,6859] | Bền hơn trên artifact tổng hợp; chưa là gain external |
| **RSNA test 200, 5 fold, TTA 10 view** | A **4,2517** | C 4,4029 | C − A = +0,1512, CI [−0,0553; +0,3584] | Không thấy C thắng A; test đã được xem nhiều lần |
| RSNA test 200, chỉ Fold 1+5, TTA 10 view | A 4,2053 | B 4,4128 / C 4,4690 | B − A +0,2076; C − A +0,2637; CI đều cắt 0 | Chỉ là đối chứng cùng số fold cho B; không chọn A hai fold như model mới |

Các số OOF 5 fold nằm trong [report C](../results/pilot_bc_20260914/oof/C3_Z26_C3_ROI_V2_PILOT_C_OOF_report.json), [prediction từng ảnh](../results/pilot_bc_20260914/oof/C3_Z26_C3_ROI_V2_PILOT_C_OOF_predictions.csv) và [report blend](../results/pilot_bc_20260914/oof/blend/fixed_weight_073_report.json). [Validation summary](../results/pilot_bc_20260914/VALIDATION_SUMMARY.json) kiểm tra công thức blend trên 14.036 hàng. Lưu ý một lỗi trình bày trong JSON blend: trường `summary.weight_c` ghi `1.0`, mâu thuẫn với top-level `weight_pilot_c = 0.73`; **công thức và CSV prediction đã được kiểm tra là 0,27 A + 0,73 C**. Không sử dụng trường nested đó để mô tả phương pháp.

MAE test baseline và protocol/hashes nằm trong [report test gốc](../c3_roi/artifacts/RSNA_DATA_BACKUP_20260909/C3_Z26_C3_ROI_V2_TTA_TEST/C3_Z26_C3_ROI_V2_TTA_TEST_report.json). Số B/C test ở bảng là output Colab đã lưu trong [báo cáo B/C](BAO_CAO_PILOT_B_C_DEN_2026-09-18.md); **CSV/report gốc B/C test chưa nằm trong project**, nên hiện chưa thể tái tính từng dòng từ máy local. Đừng trích dẫn chúng như một xác nhận độc lập.

## Vì sao không chọn B/C hoặc blend làm mô hình chính?

**Lợi ích artifact đi kèm tổn thất ở trẻ nhỏ.** Trong OOF C, nhóm 0–59 tháng (895 ảnh) có delta clean **+0,4976** và artifact **+0,3015**; nhóm 60–119 tháng (3.881 ảnh) có clean **+0,2358**. Blend A/C tuy giảm clean MAE chung 0,0575 nhưng nhóm 0–59 sạch xấu hơn **+0,1960** (CI [+0,0064; +0,3845]), nhóm 60–119 **+0,0827** (CI [+0,0110; +0,1536]). MAE clean cân bằng bốn nhóm tuổi của blend là **6,3056**, xấu hơn A **6,2597**. Cross-fit chọn trọng số từ bốn fold rồi đo fold còn lại đạt clean 6,2728 / artifact 6,4985, nhưng mức hại clean cao nhất theo nhóm tuổi vẫn **+0,2084**, vượt gate +0,20. Không thể dùng MAE chung để che nhược điểm này.

**B Fold 5 là đối chứng giải thích một phần, không cứu được vấn đề.** Toàn Fold 5, clean A khoảng 6,1822, B 6,2954, C 6,4305. Ở nhóm 60–119 tháng, B 7,3863 gần A 7,4161, nhưng C 8,2461; C − B **+0,8598**, CI [+0,6423; +1,0764]. Consistency weight 0,30 có thể góp phần làm xấu riêng lát cắt đó. Nhưng ở 0–59 tháng, **B cũng xấu hơn A**: clean +1,0130 và artifact +1,7681. Điều đó cho thấy không thể chỉ bỏ consistency để giải quyết toàn bộ vấn đề trẻ nhỏ. Fold 1 ít hại hơn Fold 5, nên không suy rộng hiệu ứng Fold 5 thành quy luật mọi fold. Chi tiết bias, 0–23/24–59 và montage ở [báo cáo B/C](BAO_CAO_PILOT_B_C_DEN_2026-09-18.md).

**Ba seed artifact chỉ xác nhận trong cùng họ biến đổi.** Giữ nguyên các checkpoint, blend 27/73 đạt gain artifact 0,7499 / 0,7736 / 0,7549 tháng cho seed 20260912/13/14. Đây không phải ba lần train độc lập, càng không phải kiểm chứng trên máy X-quang hay bệnh viện khác. [Report multi-seed](../results/pilot_bc_20260914/oof/multiseed/multiseed_artifact_report.json).

**Test 200 không ủng hộ chọn C để đạt 4,2.** A 5 fold + TTA là **4,251692 tháng**, còn cách 4,2 **0,051692 tháng (~1,57 ngày)**; C cùng protocol là 4,402925. Khoảng bootstrap của MAE A là [3,8014; 4,7187]; mọi CI delta B/C test đã báo đều chứa 0. Test này đã được xem và dùng để so nhiều phương án, vì vậy một phép tinh chỉnh mới để thấy 4,19 ở đây sẽ không phải bằng chứng cải thiện độc lập.

## Audit tính đúng đắn và giới hạn phải công bố

[Audit có thể chạy lại](../c3_roi/audit_v2_protocol.py) cho [kết quả JSON](C3_V2_PROTOCOL_AUDIT_2026-09-18.json): 5 fold validation phủ **14.036 ID/hash duy nhất**, không trùng ID/hash giữa train và validation trong từng fold, hash manifest khớp config, 200 test ID không trùng OOF. Điều này kiểm tra metadata, **chưa loại trừ cùng bệnh nhân hoặc ảnh gần trùng** vì thiếu patient ID và chưa làm so sánh pixel/perceptual đầy đủ.

1. **Chuẩn hóa nhãn không fit riêng theo fold:** fold 2–5 đang dùng mean/std tuổi của **train Fold 1**. Tập train Fold 1 có nhãn thuộc validation của các fold 2–5. Đây là rò rỉ thống kê nhãn ở mức protocol; ảnh validation vẫn không nằm trực tiếp trong train của chính fold. A/B/C trên cùng fold dùng cùng hằng số nên các so sánh ghép cặp vẫn mô tả được kết quả *trong protocol cũ*, nhưng chưa định lượng tác động của việc sửa hằng số. **Không viết đè** config/checkpoint cũ để che vấn đề; recipe train mới phải tính mean/std từ train riêng từng fold và so bằng đối chứng retrain cùng protocol.
2. **Metadata ROI cũ:** 1.974 ảnh phát triển vẫn ghi `global_fallback` trong manifest, nhưng V2 đã thay pixel **588** ảnh bằng rescue; fallback thực còn **1.386**. Các phép chia subgroup `global_fallback`/`mask_bbox` trước đây dựa trực tiếp vào `roi_mode` có thể trộn hai trạng thái. Không dùng các so sánh ROI đó làm kết luận nhân quả về nguyên nhân B/C hỏng; ghép `rescue_status` với `image_id` trước khi phân tích lại.
3. **Bằng chứng local chưa đầy đủ:** repo có gói OOF C/blend/multi-seed đã kiểm tra 72 file theo manifest, nhưng thiếu B Fold 5 report/prediction gốc và B/C test report/prediction gốc trong project. Số tương ứng hiện là bản chép từ output người dùng; nên nhập bản gốc và hash khi làm phụ lục khoa học. Không cần tải checkpoint lên Git.
4. **Thiếu external validation và audit chất lượng pixel toàn diện:** artifact `mild_v1` không đại diện mọi loại hỏng ảnh. Hiện không thể kết luận độ bền trên DHA, loạn sản xương hay bệnh viện khác.

## Quyết định nghiên cứu đã khóa

**Giữ A/C3-V2 làm đối chứng chính cho ảnh sạch; lưu C và blend 27/73 như kết quả phụ về robustness với artifact tổng hợp. Dừng mở rộng B/C để săn clean MAE ≤4,2.** Không tuyên bố B/C là model tốt hơn tổng thể, không kết luận C đã chứng minh khái quát lâm sàng, không tối ưu tiếp trên test 200. Kết quả C vẫn có giá trị khoa học nếu báo cả lợi ích artifact **và** tổn thất ở trẻ nhỏ. Nhận định này phù hợp với nghiên cứu gốc về [stress test bone age](https://pmc.ncbi.nlm.nih.gov/articles/PMC11140516/) và [bias theo nhóm tuổi/giới khi đánh giá ngoài](https://pubs.rsna.org/doi/10.1148/radiol.220505); đây là căn cứ cho **cách đánh giá**, không phải bằng chứng những paper đó đã chứng minh C3-V2.

Việc tiếp theo được chốt theo thứ tự:

1. **Không train:** lưu bổ sung artifact gốc B Fold 5 và B/C test vào project, kèm SHA-256 và checkpoint/config hash; audit lỗi ảnh lớn trên OOF, tách 0–23/24–59/60–119 tháng, sex và ROI *sau rescue*. Chọn mẫu lỗi lẫn mẫu ngẫu nhiên đối chứng, xem ảnh gốc và ROI, không chọn theo test.
2. **Protocol mới, không sửa lịch sử:** tạo phiên bản config/dataset mới với target mean/std fit riêng từng fold, metadata ROI hiệu lực và kiểm tra split. Train đối chứng lại ở fold cần so. Chỉ khi đối chứng này ổn định mới quy tác động của thí nghiệm mới cho biến cần thử.
3. **Một pilot ưu tiên:** age-balanced sampling **nhẹ** so permutation, giữ architecture/preprocessing/loss và tổng số update; thử Fold 1 và Fold 5, mỗi fold có đối chứng cùng protocol. Không đồng thời đổi CLAHE, ROI margin, LDL hay consistency. Theo dõi clean MAE tổng, MAE cân bằng tuổi, 0–23/24–59/60–119, sex, bias có dấu và artifact MAE phụ.
4. **Gate trước khi mở 5 fold:** trên validation ghép Fold 1+5, clean MAE gain ≥0,10 tháng, CI 95% của delta ứng viên − đối chứng có cận trên <0, không có harm rõ ràng ở 0–59/60–119 và kết quả không phụ thuộc một fold duy nhất. Nếu không qua, dừng pilot; chuyển sang ablation ROI/intensity từng biến như [kế hoạch chi tiết](AUDIT_VA_KE_HOACH_MAE_4_2_2026-09-18.md). Nếu qua, mở 5 fold và kiểm tra lại cùng gate trên OOF đầy đủ; chỉ sau khi khóa toàn bộ recipe mới đánh giá trên **cohort xác nhận độc lập chưa từng dùng để chọn model**.

**Trạng thái mục tiêu:** chưa có bằng chứng đáng tin cậy đạt MAE ≤4,2. Số tốt nhất của baseline trên RSNA test đã xem là 4,2517; khoảng cách số học rất nhỏ nhưng không thể xem việc lặp thử trên cùng 200 ảnh là thành công khoa học.
