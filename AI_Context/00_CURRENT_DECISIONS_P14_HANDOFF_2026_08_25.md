# QUYẾT ĐỊNH HIỆN HÀNH VÀ HANDOFF THỰC THI P14

> Ngày khóa quyết định: 2026-08-25
> Phạm vi: toàn bộ dự án tại D:\Hoctap\Doan_totnghiep
> Trạng thái: kế hoạch đã chốt, chưa triển khai P14
> Đối tượng đọc: AI hoặc thành viên tiếp tục nghiên cứu và huấn luyện mô hình

## 0. Tài liệu này có thẩm quyền gì?

Đây là tài liệu phải đọc đầu tiên trước khi tiếp tục thí nghiệm mới. Nó tổng hợp quyết định mới nhất của chủ đồ án sau khi:

- kiểm tra toàn bộ AI_Context và các artifact hiện có;
- đối chiếu kết quả P7, P8, P9-I, EXP006, D3 và C3;
- kiểm tra khả năng tái lập Deeplasia và Bram từ nguồn công khai;
- xác nhận chủ đồ án muốn ưu tiên một hướng tự phát triển, thay vì dành tài nguyên chính cho việc tái lập bài báo.

Tài liệu này thay thế phần “việc tiếp theo”, “next action” hoặc “khuyến nghị ưu tiên” trong các tài liệu cũ nếu có mâu thuẫn. Các số liệu lịch sử trong tài liệu cũ vẫn được giữ làm bằng chứng.

Các tài liệu sau chỉ còn vai trò lịch sử hoặc bằng chứng, không còn quyết định hướng nghiên cứu chính:

- 04_NEXT_P9_PLAN.md: hướng tái lập Deeplasia/Bram là kế hoạch cũ.
- 07_DEEPLASIA_FOCUSED_ANALYSIS.md: còn hữu ích để hiểu Deeplasia, nhưng không còn là roadmap chính.
- 13_GROUP_COMBINED_TECHNIQUES_RESULTS_2026_08_24.md: báo cáo tổng hợp lịch sử, không phải kế hoạch triển khai tiếp theo.
- 19_REPORT_PLAN_A_D3_FINAL.md, 20_D3_CALIBRATION_RESULT.md và 24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md: bằng chứng đầu vào cho P14.
- context_index.json hiện có dấu hiệu merge chưa xử lý, khóa trùng và nội dung không phải JSON hợp lệ. Không dùng file này làm nguồn máy đọc cho tới khi được sửa riêng.
- 00_START_HERE.md và 01_STATUS_RESULTS.md có một số “next action” cũ hoặc chưa chứa toàn bộ kết quả mới. Khi mâu thuẫn, tài liệu này được ưu tiên.

## 1. Tóm tắt quyết định đã chốt

### 1.1. Hướng chính

Hướng chính tiếp theo là:

**P14 — Anatomy-Diverse Global–Local Bone Age Ensemble**

Tên tiếng Việt đề xuất:

**Mô hình đa góc nhìn giải phẫu toàn cục–cục bộ cho dự đoán tuổi xương.**

Đây là hướng tự phát triển. Deeplasia và Bram chỉ được dùng làm:

- mốc so sánh;
- nguồn tham khảo cho preprocessing, augmentation, backbone và ensemble;
- đối chứng trong phần Related Work và Discussion.

Không gọi P14 là “Deeplasia reproduction”, “Bram reproduction” hoặc “Bram-faithful”.

### 1.2. Vai trò của việc giữ và phục hồi artifact

Giữ và phục hồi artifact là Phase 0 bắt buộc để bảo vệ tính tái lập, nhưng không phải đóng góp mới của P14.

Quyết định cụ thể:

- giữ nguyên mọi checkpoint, config, manifest, prediction, report và log đã hoàn tất;
- không sửa đè artifact P0–P13, D3, EXP006 hoặc C3;
- phục hồi những artifact còn thiếu hoặc provenance chưa đầy đủ;
- tạo P14 trong namespace/thư mục mới;
- mọi candidate P14 phải được so sánh bằng validation/OOF, không dùng nhãn RSNA test để chọn.

### 1.3. Thứ tự ưu tiên

Thứ tự đã chốt:

1. Đóng băng, kiểm kê và phục hồi artifact hiện tại.
2. Sửa localization và tạo các góc nhìn giải phẫu ổn định.
3. Huấn luyện các specialist toàn cục/cục bộ độc lập.
4. Tạo ensemble không âm bằng nested cross-fitting.
5. Chỉ sau khi toàn bộ protocol đã khóa mới chạy benchmark test thăm dò.
6. Việc xin code/checkpoint từ tác giả Deeplasia/Bram tạm hoãn; không để việc này chặn P14.

## 2. Trạng thái khoa học hiện tại

## 2.1. Các kết quả OOF quan trọng

| Nhánh | Tập đánh giá | MAE tháng | Cách diễn giải |
|---|---:|---:|---|
| P7/E1 raw | OOF 14.036 | 6,316691 | Baseline 5-fold chính |
| E1-TTA | OOF 14.036 | 6,210446 | TTA cải thiện 0,107025 tháng so với raw tái suy luận; CI paired dưới 0 |
| EXP006-TTA | OOF 14.036 | 6,296203 | Nhánh độc lập có lỗi bổ sung |
| D3-TTA | OOF 14.036 | 6,246542 | Đứng riêng kém E1-TTA |
| E1-TTA + D3-TTA 50/50 | OOF 14.036 | 6,101345 | Ensemble báo cáo tốt; cải thiện ở 5/5 fold |
| C3-ROI raw | OOF 14.036 | 6,437349 | Đứng riêng kém E1 |
| E1 + C3 50/50 | OOF 14.036 | 6,176212 | Diversity dương tính; CI paired hoàn toàn dưới 0 |
| Nested E1-TTA + EXP006-TTA + C3 | OOF 14.036 | 6,075569 | Baseline kỹ thuật mạnh nhất có thể kiểm toán từng dòng từ CSV hiện có |

Không được so trực tiếp P7 OOF 6,3167 với P8 test 4,7303 để nói ensemble cải thiện 1,59 tháng, vì hai số được tính trên hai tập khác nhau.

### 2.2. Baseline nested OOF 6,075569 được tính như thế nào?

Ba nguồn dự đoán:

1. E1-TTA:

   D:\Hoctap\Doan_totnghiep\p9_inference\outputs\P9_I_TTA_BIAS_OOF\P9_I_TTA_BIAS_OOF_predictions.csv

   SHA-256:

   659837337640EAB62CF5B225B88367CD098A374D87A965188B3A8DFB8C82F88A

2. EXP006-TTA:

   D:\Hoctap\Doan_totnghiep\research\our_baseline\baseline_v1\outputs\exp006_p7_tta\exp006_p7_tta\results\tta_oof_predictions.csv

   SHA-256:

   80B2B5F01AC061DF940B8C6F33FFC0B3644C6A1490A4DADCE2762952603565E8

3. C3 raw:

   D:\Hoctap\Doan_totnghiep\c3_roi\outputs\C3_ROI_V1_OOF\C3_ROI_V1_E1_ensemble_OOF_predictions.csv

   SHA-256:

   B2131168FDE86BE7A9A100FA757E1DCB495AA35F65A245E8D9CC3037EF64CA0A

Quy trình đúng:

- join one-to-one theo image_id;
- xác nhận đủ 14.036 ID duy nhất;
- xác nhận target khớp tuyệt đối giữa ba nguồn;
- dùng fold của E1/P7 làm outer fold;
- với mỗi outer fold f, học trọng số không âm, tổng bằng 1 trên bốn fold còn lại bằng cách tối thiểu hóa MAE;
- chỉ áp dụng trọng số đó lên fold f chưa được dùng để học trọng số;
- ghép năm prediction outer-fold để tính pooled MAE.

Trọng số theo outer fold đã tái tính:

| Outer fold | E1-TTA | EXP006-TTA | C3 |
|---:|---:|---:|---:|
| 1 | 0,428739 | 0,325740 | 0,245521 |
| 2 | 0,396526 | 0,360373 | 0,243101 |
| 3 | 0,412376 | 0,316215 | 0,271409 |
| 4 | 0,460593 | 0,280082 | 0,259325 |
| 5 | 0,452472 | 0,323096 | 0,224432 |
| Trung bình | 0,430141 | 0,321101 | 0,248758 |

Kết quả:

- pooled MAE: 6,075568541;
- delta so với E1-TTA: −0,134877672 tháng;
- paired bootstrap 10.000 lần, seed 2026: CI xấp xỉ [−0,1603; −0,1092].

Lưu ý bắt buộc:

- công thức 0,43/0,32/0,25 chỉ là trọng số trung bình để mô tả, không phải trọng số đã áp dụng giống nhau cho mọi fold;
- nếu tối ưu một trọng số duy nhất trên toàn OOF rồi đánh giá trên chính OOF đó, metric sẽ lạc quan;
- P14 phải dùng nested cross-fitting tương tự hoặc chặt hơn;
- 6,075569 là baseline kỹ thuật để P14 phải vượt, không phải tuyên bố benchmark quốc tế.

### 2.3. Các kết quả test chỉ được xem là thăm dò

| Nhánh | RSNA test 200 MAE tháng | Trạng thái |
|---|---:|---|
| P8/E1 5-fold ensemble | 4,730321 | Test đã được đọc |
| E1-TTA | 4,61749 | Báo cáo nhánh D3 |
| D3-TTA | 4,50846 | Tốt nhất trong nhánh D3 |
| E1-TTA + D3-TTA | 4,51070 | Gần D3-TTA |
| C3-ROI | 4,337267 | Thấp nhất hiện có, nhưng chỉ thăm dò |
| E1 + C3 | 4,454661 | Không được dùng để tune weight |
| Deeplasia | 3,87 | Mốc công bố |
| Bram | 3,68 | Mốc công bố |

Khoảng cách điểm số hiện tại:

- C3 4,337267 còn cao hơn Deeplasia khoảng 0,467 tháng;
- C3 còn cao hơn Bram khoảng 0,657 tháng;
- P8 còn cao hơn Deeplasia khoảng 0,860 tháng.

RSNA test 200 đã được mở và được xem nhiều lần. Vì vậy:

- không dùng test để chọn preprocessing, ROI, backbone, loss, seed, checkpoint, TTA hoặc ensemble weight;
- mọi lần test tiếp theo chỉ là benchmark thăm dò;
- muốn tuyên bố xác nhận độc lập cần external holdout chưa bị tác động, đánh giá mù hoặc server/challenge không trả nhãn theo từng thử nghiệm.

## 3. Kết luận về khả năng tái lập Deeplasia và Bram

## 3.1. Deeplasia

Nguồn chính thức đã kiểm tra:

- bài báo: https://pmc.ncbi.nlm.nih.gov/articles/PMC10776485/
- repository: https://github.com/aimi-bonn/Deeplasia
- commit công khai được kiểm tra: b010cff8693f64712e65dfba2c817438f2da09f5
- hand-segmentation repository: https://github.com/aimi-bonn/hand-segmentation
- mask archive: https://zenodo.org/records/7611677

Những gì công khai đủ để xây một reproduction gần phương pháp:

- EfficientNet-B0/B4;
- input 512 và 1024;
- sex input và các fully connected head được chọn;
- MSE, Adam, learning rate, weight decay, scheduler, dropout và batch size;
- augmentation khá chi tiết;
- unweighted ensemble ba model;
- các hệ số chuẩn hóa/calibration trong file public parameters.yml;
- dependency được pin tương đối đầy đủ.

Những gì còn thiếu để exact reproduction:

- ba checkpoint cuối được notebook gọi tên;
- đúng FastSurferCNN checkpoint hoặc đúng mask fscnn_cos dùng cho đánh giá cuối;
- exact split manifest mà README/code tham chiếu;
- per-image golden predictions;
- package/release gắn chính xác với phiên bản dùng để sinh số 3,87;
- bằng chứng không mơ hồ về exact inference path sinh metric cuối.

Kết luận đã chốt:

- exact inference reproduction: hiện bị chặn;
- exact training reproduction: không thể bảo đảm;
- method-faithful reproduction: khả thi ở mức trung bình đến khá;
- P9 EfficientNet-B0 screening MAE 8,53–11,04 không phải reproduction Deeplasia và không được dùng để kết luận Deeplasia không tái lập được;
- không dành tài nguyên chính cho reproduction Deeplasia ở giai đoạn này.

TTA của Deeplasia đã được thử trong nghiên cứu nhưng theo supplement không được dùng trong final inference vì chi phí. Không được mô tả sai rằng MAE 3,87 chắc chắn là kết quả có TTA.

## 3.2. Bram

Nguồn chính thức đã kiểm tra:

- bài báo: https://pmc.ncbi.nlm.nih.gov/articles/PMC12381387/
- bản đầy đủ: https://journals.sagepub.com/doi/full/10.1177/03635465251359618
- supplement: https://journals.sagepub.com/doi/suppl/10.1177/03635465251359618/suppl_file/sj-pdf-1-ajs-10.1177_03635465251359618.pdf

Những gì bài báo công bố:

- ConvNeXt + sex;
- gộp RSNA train 12.611 và validation 1.425;
- loại 35 ảnh bất thường;
- 5-fold;
- train 100 epoch bằng MAE loss;
- preprocessing có background segmentation, histogram equalization và alignment;
- augmentation gần Deeplasia;
- trung bình năm model;
- Model 1 RSNA đạt 3,68 tháng.

Những gì thiếu:

- code chính thức;
- ConvNeXt variant;
- input resolution, pretraining và normalization;
- optimizer, scheduler, batch, precision;
- cấu hình cuối được chọn từ grid;
- seed và five-fold manifest;
- ID của 35 ảnh bị loại;
- exact segmentation, histogram, alignment, crop và augmentation;
- checkpoint và per-image predictions.

Kết luận đã chốt:

- exact reproduction Bram: không khả thi từ artifact công khai;
- method-faithful Bram: cũng không thể tuyên bố;
- triển khai hiện có hoặc tương lai chỉ được gọi là “Bram-inspired”;
- Bram 3,68 là point estimate; CI 95% được báo cáo [3,24; 4,14] bao gồm 3,87 của Deeplasia, nên không có đủ bằng chứng paired để nói Bram vượt Deeplasia có ý nghĩa thống kê.

## 4. Vì sao kết quả hiện tại chỉ gần bài báo?

Không có một nguyên nhân duy nhất. Các bằng chứng hiện tại ủng hộ sáu nguyên nhân chính.

### 4.1. Ensemble hiện tại còn quá tương quan

P8 chủ yếu trung bình năm fold của cùng một ConvNeXt-Tiny và cùng kiểu input. Cách này giảm variance nhưng không tạo diversity lớn như một ensemble khác kiến trúc, khác độ phân giải và khác vùng giải phẫu.

Các residual/prediction hiện có tương quan cao, xấp xỉ 0,88–0,92 giữa nhiều nhánh; E1 và C3 có prediction correlation 0,995123. Do đó chỉ tối ưu lại trọng số khó tạo bước nhảy đủ lớn để vượt 3,87.

### 4.2. C3 chưa phải local anatomy specialist đúng nghĩa

C3 hiện dùng segmentation bounding box toàn bàn tay + margin 8%, không phải carpal ROI riêng:

- fallback development: 18,49%;
- fallback test: 33%;
- gate thiết kế ban đầu: không quá 1%.

C3 cho thấy tín hiệu quan trọng: thay đổi góc nhìn có thể tạo error diversity. Nhưng localization hiện tại chưa đủ ổn định và chưa cô lập đúng vùng giải phẫu.

### 4.3. Pipeline chưa chuẩn hóa hình học tốt

Ảnh RSNA khác nhau về rotation, scale, vị trí tay, khoảng nền và tương phản. Resize toàn ảnh có thể dành nhiều pixel cho nền và làm thay đổi kích thước tương đối của cấu trúc xương.

Masking đơn lẻ từng thất bại không chứng minh mọi dạng anatomy normalization đều vô ích. Mask + canonical orientation + fixed anatomical ROI là một can thiệp khác với mask/crop cũ.

### 4.4. Các mô hình nhìn cùng một tín hiệu

Global full-hand có xu hướng học tín hiệu tổng thể. Tuổi xương còn có các tín hiệu khác nhau ở:

- carpal bones;
- distal radius/ulna;
- metacarpals;
- phalanges;
- mức đóng growth plate.

Một model toàn ảnh có thể không phân bổ đủ độ phân giải cho mọi vùng. Specialist theo vùng có khả năng tạo lỗi bổ sung ngay cả khi MAE standalone không thắng.

### 4.5. Một số hướng đã thử không phải nguyên nhân chính

- augmentation Deeplasia mức vừa gần như không đổi MAE;
- tăng resolution 512 lên 768 không cải thiện ổn định;
- calibration tuyến tính làm xấu MAE 6,10134 lên 6,11641;
- D3-TTA đứng riêng không thắng E1-TTA trên OOF;
- EfficientNet-B0 P9 chỉ là screening bị dừng sớm, không phải kiểm chứng đầy đủ Deeplasia.

Do đó không tiếp tục tăng augmentation, resolution, calibration hoặc complexity một cách mù quáng.

### 4.6. Nhóm khó còn rõ

Nhóm 60–119 tháng là nhóm lỗi khó nổi bật. Sai số nữ nhìn chung cao hơn trong các phân tích hiện có. P14 phải báo subgroup theo sex × age và không được tối ưu hậu nghiệm riêng trên nhóm này bằng cùng OOF.

## 5. Giả thuyết khoa học của P14

Giả thuyết chính:

> Các góc nhìn giải phẫu được chuẩn hóa và huấn luyện độc lập sẽ tạo ra sai số bổ sung cho mô hình toàn bàn tay; một ensemble nested, bị ràng buộc không âm, có thể giảm MAE OOF ổn định hơn việc chỉ thay backbone hoặc tối ưu lại trọng số các model rất tương quan.

Đóng góp dự kiến:

1. Pipeline localization có gate hình học và audit rõ ràng.
2. Thiết kế multi-view gồm global, wrist/carpal và metacarpal/phalangeal.
3. Ablation tách riêng tác động của chuẩn hóa hình học, ROI, backbone và ensemble.
4. Ensemble nested cross-fitted tránh leakage do học weight trên toàn OOF.
5. Phân tích lợi ích theo sex × age và mối quan hệ giữa diversity với MAE.

## 6. Kiến trúc P14 đã chốt

### 6.1. G0 — Global raw baseline

Không train lại ở bước đầu.

Sử dụng E1 hiện có:

- ConvNeXt-Tiny;
- input 512;
- sex embedding;
- direct regression;
- A2 augmentation;
- TTA đã kiểm chứng.

Vai trò:

- baseline khóa;
- một thành phần ensemble;
- chuẩn để so paired từng ID.

### 6.2. G1 — Global anatomy-normalized

Input:

- hand foreground được segment ổn định;
- canonical left/right orientation;
- canonical rotation;
- crop theo hand mask với margin khóa trước;
- letterbox/pad nhất quán;
- resize đúng một lần;
- có thể giữ nền bằng zero/constant mask, nhưng phải khóa trước khi train.

Model đầu tiên:

- giữ cùng ConvNeXt-Tiny, sex embedding và direct regression như E1;
- mục đích là cô lập tác động của normalization, không thay nhiều biến cùng lúc.

### 6.3. L1 — Wrist/carpal + distal radius/ulna specialist

ROI phải chứa:

- toàn bộ carpal bones;
- distal radius;
- distal ulna;
- phần gần của metacarpals đủ làm mốc.

Không dùng broad hand bounding box rồi gọi là carpal ROI.

Vòng đầu dùng cùng recipe/model với G0 để đo tác động ROI. Nếu L1 tạo diversity nhưng standalone chưa đủ, vòng tiếp theo dùng một backbone khác, ưu tiên EfficientNetV2-S hoặc EfficientNet-B3 ở resolution phù hợp VRAM.

### 6.4. L2 — Metacarpal + phalangeal specialist

ROI phải chứa:

- metacarpals;
- proximal/middle/distal phalanges;
- growth plates liên quan;
- có thể loại phần lớn wrist nếu định nghĩa crop vẫn ổn định.

Vòng đầu giữ recipe giống G0. Chỉ thay backbone sau khi ROI đạt gate và có bằng chứng diversity.

### 6.5. D3 — nhánh tùy chọn

D3 không phải trục chính của P14.

Chỉ đưa D3-TTA vào ensemble P14 sau khi:

- phục hồi được prediction CSV OOF từng dòng;
- xác nhận image_id, target, sex, fold và checkpoint provenance;
- recompute đúng MAE 6,246542 và ensemble 6,101345;
- không suy đoán prediction từ report tổng hợp.

Nếu không phục hồi được CSV, D3 chỉ xuất hiện trong bảng so sánh report-verified, không được dùng cho row-level optimizer P14.

### 6.6. Cách fusion

Ưu tiên late fusion prediction-level trước feature fusion.

Lý do:

- dễ audit từng nhánh;
- có OOF prediction rõ;
- tách được lỗi localization khỏi lỗi fusion;
- giảm rủi ro overfit;
- cho phép giữ specialist có standalone MAE không thắng nhưng tạo diversity.

Feature-level fusion hoặc attention/gating chỉ được thử sau khi late fusion đạt baseline và có ngân sách. Không dùng age-dependent hoặc sex-dependent gating ở vòng đầu.

## 7. Phase 0 — đóng băng và phục hồi artifact

AI tiếp theo phải hoàn thành Phase 0 trước khi train dài.

### 7.1. Không được làm

- không sửa config của run đã hoàn tất;
- không ghi đè best_mae.ckpt hoặc last.ckpt;
- không đổi tên/move artifact mà không có manifest ánh xạ;
- không tạo lại report cũ rồi thay file gốc;
- không xóa cache/checkpoint cũ trước khi archive và hash;
- không push remote nếu chưa có yêu cầu cụ thể của người dùng.

### 7.2. Inventory bắt buộc

Tạo inventory máy đọc chứa tối thiểu:

- absolute path;
- relative project path;
- file size;
- SHA-256;
- run_id;
- phase;
- fold;
- seed;
- artifact_type;
- created_at hoặc LastWriteTime;
- status: original, recovered, regenerated hoặc missing;
- source/provenance note.

Tối thiểu phải kiểm kê:

- P7 five-fold checkpoints/config/manifests/OOF;
- P9-I E1-TTA predictions/report;
- EXP006 checkpoints/config/OOF/TTA predictions;
- D3 checkpoints/reports/predictions;
- C3 checkpoints/config/ROI audit/OOF/test exploratory;
- environment/dependency files;
- preprocessing caches và mask metadata.

### 7.3. Hai mục cần phục hồi ưu tiên

1. D3 OOF row-level prediction:

   - hiện có report PASS nhưng chưa thấy CSV OOF từng dòng trong audit hiện tại;
   - nếu checkpoint còn đủ, tái inference đúng code/config khóa;
   - file recovered phải có hash và ghi rõ được regenerate ngày nào;
   - không sửa report gốc.

2. P7 Fold 1 best-epoch metrics provenance:

   - P7_OOF_report.json ghi PASS;
   - P7_5FOLD_AUDIT.txt từng ghi final FAIL hành chính do Fold 1 thiếu block best-epoch metrics;
   - cần bổ sung một provenance note hoặc recovered evidence riêng;
   - không thay đổi metric P7 đã khóa;
   - nếu không chứng minh được, giữ trạng thái “OOF PASS kèm cảnh báo provenance”.

### 7.4. Run manifest chuẩn cho P14

Mỗi run mới bắt buộc lưu:

- run_id duy nhất;
- parent experiment;
- git commit hoặc source tree hash;
- config file + SHA-256;
- train/validation manifest + SHA-256;
- preprocessing version + SHA-256;
- fold và seed;
- GPU, CUDA, Python, PyTorch, timm và package versions;
- pretrained weight name/hash;
- start/end time;
- best epoch và selection metric;
- last/best checkpoint hash;
- prediction CSV hash;
- metric report;
- warnings, OOM, NaN/Inf, resume history;
- test_accessed=false cho mọi run phát triển.

## 8. Phase 1 — localization V2

Đây là bước có xác suất tạo giá trị cao nhất và là blocker trước train.

### 8.1. Pipeline hình học đề xuất

1. Đọc ảnh nguyên gốc và giữ image_id.
2. Chuẩn hóa hướng trái/phải về một quy ước duy nhất.
3. Segment foreground bàn tay.
4. Giữ connected component hợp lệ nhất; xử lý lỗ nhỏ/mảnh nhiễu bằng tham số khóa.
5. Xác định trục dài của bàn tay hoặc landmarks ổn định.
6. Xoay về canonical orientation.
7. Tạo G1 crop theo mask.
8. Tạo L1 và L2 từ tọa độ canonical, không từ tọa độ ảnh thô.
9. Pad/letterbox theo quy tắc cố định.
10. Resize một lần ở bước cuối.
11. Cache ảnh và metadata hình học.

Nếu chưa có landmark model đáng tin cậy, dùng mask geometry/PCA như phiên bản đầu nhưng phải visual-audit kỹ. Không được che giấu fallback.

### 8.2. Gate bắt buộc trước train

| Gate | Ngưỡng |
|---|---:|
| ID xử lý thành công | 100% development |
| Missing/empty ROI | 0% |
| Tổng fallback hình học | ≤1% |
| Foreground bị cắt khỏi G1 | ≤0,5% theo diện tích mask |
| L1 thiếu wrist/carpal nhìn thấy | ≤0,5% trong audit |
| L2 thiếu phần lớn phalanges/metacarpals | ≤0,5% trong audit |
| Duplicate output ID | 0 |
| Non-finite metadata | 0 |
| Source/output checksum có mặt | 100% |

Visual audit tối thiểu:

- mẫu stratified theo sex × age;
- toàn bộ ca fallback;
- toàn bộ ca có crop-margin nhỏ nhất/lớn nhất;
- các ca rotation cực trị;
- ít nhất 200 ảnh nếu đủ nguồn lực.

Nếu fallback trên 1%:

- dừng trước train;
- sửa localization;
- không dùng full-image fallback âm thầm như C3;
- không loại ảnh chỉ để làm đẹp metric nếu không có protocol loại ảnh khóa trước.

### 8.3. Output metadata tối thiểu

Mỗi image_id cần:

- source_sha256;
- mask_sha256;
- orientation_flip;
- rotation_degrees;
- hand_bbox;
- G1_bbox;
- L1_bbox;
- L2_bbox;
- crop/pad parameters;
- foreground_retention_ratio;
- fallback flag và reason;
- preprocessing_version.

## 9. Phase 2 — screening có kiểm soát

Không chạy 5-fold cho mọi ý tưởng.

### 9.1. Run matrix tối thiểu

| Run | Input | Backbone | Mục đích |
|---|---|---|---|
| P14-G0 | raw full hand | ConvNeXt-Tiny | baseline có sẵn, không train lại ban đầu |
| P14-G1 | normalized global | ConvNeXt-Tiny | đo riêng tác động normalization |
| P14-L1-C | wrist/carpal | ConvNeXt-Tiny | đo riêng tác động ROI L1 |
| P14-L2-C | metacarpal/phalanges | ConvNeXt-Tiny | đo riêng tác động ROI L2 |
| P14-LX-H | ROI tốt nhất | EfficientNetV2-S hoặc EfficientNet-B3 | tạo backbone diversity |

Chỉ chọn một backbone dị thể trong vòng đầu. Không mở grid lớn nhiều kiến trúc.

### 9.2. Recipe primary

Primary:

- pretrained ImageNet;
- direct regression;
- sex embedding như E1;
- SmoothL1 hoặc MAE recipe đã ổn định của E1;
- augmentation nhẹ, hợp lý giải phẫu;
- 512 cho ConvNeXt global/local nếu VRAM cho phép;
- resolution của backbone dị thể phải khóa bằng screening không dùng test;
- early stopping và checkpoint theo validation MAE;
- seed 42 cho screening;
- không calibration.

LDL chỉ là auxiliary candidate sau khi direct regression hoàn tất. Không thay đồng thời ROI, backbone, loss và augmentation trong một ablation.

### 9.3. Promotion gate từ screening lên full OOF

Một candidate được đưa lên 5-fold nếu thỏa ít nhất một điều:

- standalone cải thiện khoảng 0,15–0,20 tháng trên paired development comparison và chiều cải thiện nhất quán; hoặc
- standalone không thắng nhưng residual correlation thấp hơn rõ và một fixed blend cải thiện ít nhất 0,10 tháng trên dữ liệu không dùng để chọn weight.

Operational failure phải dừng ngay:

- NaN/Inf;
- collapse;
- resume sai;
- checkpoint/config hash mismatch;
- localization gate fail;
- OOM lặp lại sau cấu hình an toàn;
- prediction ID/target mismatch.

Không kết luận khoa học từ một fold duy nhất. Nếu dùng Fold 1 làm pilot thì chỉ để kiểm tra vận hành.

## 10. Phase 3 — full 5-fold specialist

Chỉ chạy full OOF cho tối đa hai candidate tốt nhất ở vòng đầu:

- một candidate anatomy-normalized hoặc local tốt nhất;
- một candidate tạo backbone diversity tốt nhất.

Quy tắc:

- dùng đúng 14.036 development ID và split P7 đã khóa;
- mỗi prediction OOF phải đến từ model không train trên ID đó;
- config khác fold chỉ được khác đường dẫn và fold id;
- seed primary cố định;
- lưu prediction từng fold và merged OOF;
- recompute metric độc lập từ CSV;
- báo pooled, per-fold, sex, age và sex × age;
- báo residual correlation với E1-TTA, EXP006-TTA và C3;
- giữ cả kết quả âm.

Candidate final nên được xác nhận thêm bằng ít nhất một seed phụ hoặc một replication plan có ngân sách rõ ràng. Không dùng test để quyết định seed nào được giữ.

## 11. Phase 4 — ensemble nested và đánh giá thống kê

### 11.1. Thành phần đầu vào

Thành phần có thể xét:

- G0/E1-TTA;
- G1 nếu qua gate;
- L1 nếu qua gate;
- L2 nếu qua gate;
- backbone dị thể nếu qua gate;
- EXP006-TTA như baseline diversity;
- C3 để đối chứng lịch sử;
- D3-TTA chỉ khi phục hồi được OOF CSV.

### 11.2. Quy tắc học weight

Primary ensemble:

- weight không âm;
- tổng weight bằng 1;
- tối thiểu hóa MAE;
- nested theo outer fold;
- weight của outer fold chỉ học từ bốn fold còn lại;
- có thể thêm regularization kéo về equal weight nếu weight không ổn định.

Phải báo:

- weight từng outer fold;
- mean và SD weight;
- pooled nested OOF metric;
- per-fold metric;
- paired delta so với E1-TTA;
- paired delta so với baseline nested 6,075569;
- bootstrap CI;
- tỷ lệ fold thắng;
- subgroup;
- correlation và error diversity.

Không được:

- tối ưu weight trên toàn OOF rồi báo lại metric cùng OOF như unbiased;
- chọn weight bằng RSNA test;
- chọn model chỉ vì standalone test thấp;
- thử nhiều optimizer/weight grid rồi chỉ báo cáo kết quả tốt nhất mà không nested.

### 11.3. Gate giữ P14

Minimum success:

- nested OOF MAE ≤5,95;
- paired CI của delta so với baseline liên quan phải hoàn toàn dưới 0;
- không có subgroup lớn xấu đi nghiêm trọng;
- cải thiện không chỉ đến từ một fold.

Strong success:

- cải thiện ít nhất 0,25 tháng so với 6,075569, tức MAE khoảng ≤5,83.

Stretch target:

- cải thiện khoảng 0,30 tháng, tức MAE khoảng ≤5,78.

Nếu P14 chỉ cải thiện dưới 0,10 tháng hoặc CI cắt 0:

- giữ như ablation;
- không gọi là mô hình vượt trội;
- không mở thêm test chỉ để tìm một số đẹp.

## 12. Phase 5 — freeze và benchmark cuối

Trước khi chạy bất kỳ test nào:

- khóa code hash;
- khóa config;
- khóa preprocessing version;
- khóa fold checkpoints;
- khóa TTA;
- khóa ensemble components;
- khóa nested/final weight rule;
- khóa metric implementation;
- tạo signed/frozen manifest;
- ghi rõ không còn quyết định nào thay đổi sau khi xem test.

RSNA test 200:

- chỉ chạy một final frozen P14 package nếu thật sự cần cho luận văn;
- kết quả vẫn phải gọi là exploratory vì test đã bị chạm trước đây;
- engineering aspiration có thể đặt ở MAE ≤3,70 để có cơ hội thấp hơn point estimate Deeplasia 3,87;
- không được gọi là confirmatory superiority chỉ từ số này.

Để tuyên bố mạnh:

- cần external holdout chưa chạm;
- hoặc đánh giá mù bởi bên thứ ba;
- báo CI và nếu có prediction đối chứng thì dùng paired test;
- cùng endpoint, cùng label precision và cùng inclusion criteria.

## 13. Tổ chức nhiều GPU/tài khoản

Người dùng có GPU local và nhiều tài khoản Google. Có thể phân phối fold để giảm thời gian nếu tuân thủ điều khoản của nền tảng. Không thiết kế quy trình nhằm né quota hoặc hạn chế dịch vụ.

Nguyên tắc:

- không lưu credential trong repo/config/log;
- mỗi worker chỉ nhận immutable experiment bundle;
- một run_id chỉ có một owner tại một thời điểm;
- mỗi fold có output directory riêng;
- bundle phải chứa code/config/manifest hash;
- upload checkpoint nguyên tử;
- sau mỗi lần ngắt phải kiểm tra state trước resume;
- không trộn checkpoint từ hai tài khoản;
- sau khi tải về phải verify SHA-256;
- merge OOF chỉ khi đủ 5 fold và schema khớp.

Phân phối gợi ý:

- local GPU: smoke, preprocessing audit, inference audit và một pilot;
- GPU account 1–5: mỗi account một outer fold cho candidate đã khóa;
- CPU/local: merge, bootstrap, nested ensemble và report.

## 14. Cấu trúc thư mục P14 đề xuất

Tạo mới, không dùng chung output directory với phase cũ:

    p14_anatomy_diverse/
      P14_HANDOFF.md
      README.md
      configs/
        base_locked.toml
        g1_fold_1.toml ... g1_fold_5.toml
        l1_fold_1.toml ... l1_fold_5.toml
        l2_fold_1.toml ... l2_fold_5.toml
        heterogeneous_fold_1.toml ... heterogeneous_fold_5.toml
      manifests/
        development_locked.csv
        fold_assignments_locked.csv
        preprocessing_inventory.json
      preprocessing/
        version.json
        audit_report.json
        audit_samples/
      runs/
        RUN_ID/
      outputs/
        oof/
        nested_ensemble/
        reports/
      tests/

Nếu cần sửa shared training code, phải:

- tạo versioned change;
- chạy regression test xác nhận E1 prediction không drift ngoài chủ đích;
- ghi code hash;
- không sửa lịch sử output.

## 15. Checklist cho AI bắt đầu thực hiện

AI tiếp theo phải làm theo thứ tự sau.

### Bước 1 — xác nhận phạm vi và không làm hỏng lịch sử

- đọc file này;
- đọc 03_DATA_PROTOCOL.md;
- đọc 13_GROUP_COMBINED_TECHNIQUES_RESULTS_2026_08_24.md;
- đọc 19_REPORT_PLAN_A_D3_FINAL.md;
- đọc 24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md;
- kiểm tra git status và user changes;
- không push remote;
- không sửa artifact cũ.

### Bước 2 — Phase 0

- tạo inventory/hash;
- xác minh ba CSV dùng cho baseline nested;
- phục hồi D3 OOF CSV nếu có thể;
- giải thích P7 Fold 1 provenance bằng file bổ sung;
- tạo P14_HANDOFF và run schema.

### Bước 3 — localization V2

- triển khai preprocessing theo version mới;
- cache development only;
- audit 100% metadata;
- visual audit stratified;
- dừng nếu fallback trên 1%.

### Bước 4 — screening

- G1 same-backbone;
- L1 same-backbone;
- L2 same-backbone;
- chọn đúng một ROI tốt nhất cho heterogeneous backbone;
- không dùng test.

### Bước 5 — full OOF

- chỉ promote tối đa hai candidate;
- chạy 5 fold;
- merge OOF;
- paired/bootstrap/subgroup/correlation audit;
- giữ candidate theo gate đã khóa.

### Bước 6 — ensemble

- nested nonnegative sum-to-one;
- so với 6,075569;
- không full-OOF tune rồi self-evaluate;
- khóa final package.

### Bước 7 — reporting

- ghi cả kết quả dương và âm;
- ghi rõ test status;
- phân biệt benchmark, method reproduction và exact reproduction;
- không claim state of the art nếu chưa có external confirmation.

## 16. Những hướng không ưu tiên ở vòng đầu

Không ưu tiên:

- exact Bram reproduction;
- full Deeplasia reproduction;
- grid search rộng nhiều backbone;
- tăng augmentation mạnh không có ablation;
- tăng resolution đơn thuần;
- calibration tuyến tính hậu nghiệm;
- model sex-specific riêng biệt khi chưa có bằng chứng mới;
- feature fusion phức tạp;
- age/sex conditional gating;
- generative/inpainting trên test;
- tune ensemble theo test;
- loại ảnh khó sau khi xem error.

Các hướng này chỉ được mở lại nếu P14 localization + specialist + nested ensemble đã hoàn tất hoặc có bằng chứng mới.

## 17. Cách diễn đạt được phép trong luận văn

Có thể viết:

- “P14 là phương pháp tự phát triển lấy cảm hứng từ global/local modeling và ensemble diversity.”
- “C3 cho thấy ROI preprocessing tạo diversity, nhưng chưa đạt gate localization.”
- “D3 có giá trị bổ sung cho ensemble dù standalone OOF không tốt hơn E1-TTA.”
- “TTA cải thiện nhỏ nhưng nhất quán trên OOF.”
- “Nested ensemble được dùng để giảm optimism khi học trọng số.”
- “RSNA test sau P8 chỉ được dùng như benchmark thăm dò.”

Không được viết:

- “Đã tái lập hoàn toàn Deeplasia.”
- “Đã tái lập Bram.”
- “P9 chứng minh EfficientNet/Deeplasia không hiệu quả.”
- “C3 là carpal specialist.”
- “D3 chắc chắn tốt hơn E1” chỉ từ test.
- “Đã vượt Deeplasia/Bram” nếu chỉ có một point estimate trên test đã bị chạm.
- “State of the art” khi chưa có external/blinded confirmation.

## 18. Tiêu chí dừng và quyết định cuối

Tiếp tục P14 khi:

- localization đạt gate;
- ít nhất một specialist có gain hoặc diversity định lượng;
- full OOF integrity PASS;
- nested ensemble cải thiện có CI dưới 0.

Dừng hoặc thu hẹp khi:

- localization không đạt fallback ≤1% sau các vòng sửa hợp lý;
- specialist không tạo gain lẫn diversity;
- residual vẫn tương quan gần như tuyệt đối;
- nested gain dưới 0,10 tháng và CI cắt 0;
- artifact/provenance không đủ để audit;
- ngân sách GPU tăng nhưng không có bằng chứng promotion gate.

Nếu không đạt target vượt Deeplasia, đồ án vẫn có thể có đóng góp tốt nếu trình bày trung thực:

- pipeline anatomy normalization được audit;
- ablation global/local;
- nested ensemble chống leakage;
- phân tích diversity và subgroup;
- kết quả âm có kiểm soát;
- giới hạn của benchmark test đã bị chạm.

## 19. Kết luận một câu

**Giữ và phục hồi toàn bộ artifact hiện có là bước nền bắt buộc; hướng nghiên cứu chính là P14 tự phát triển với global raw + global normalized + hai local anatomy specialist + backbone diversity, sau đó fusion bằng nested OOF. Mốc kỹ thuật cần vượt trước tiên là OOF 6,075569; Deeplasia 3,87 và Bram 3,68 chỉ là benchmark test, không phải lý do để tuyên bố reproduction hoặc superiority khi chưa có đánh giá độc lập.**
