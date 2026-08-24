# Báo cáo đánh giá đóng góp khoa học của đề tài

> **Ngày đánh giá:** 2026-08-22
> **Phạm vi:** toàn bộ bằng chứng từ P0 đến P13
> **Kết luận tổng quát:** đề tài có đóng góp khoa học rõ ràng ở cấp độ đồ án tốt nghiệp; đóng góp chính thuộc nhóm thực nghiệm, phương pháp đánh giá và phân tích giới hạn, không phải một tuyên bố vượt state of the art.

## 1. Kết luận điều hành

Đề tài đã hình thành một câu chuyện khoa học nhất quán:

> Trong dự đoán tuổi xương từ X-quang bàn tay, việc đưa đúng thông tin sinh học
> phụ trợ — giới tính — mang lại cải thiện lớn hơn đáng kể so với việc tăng độ
> phức tạp kiến trúc, độ phân giải hoặc tiền xử lý. Một sex embedding đơn giản
> đã đủ hiệu quả trong recipe được khảo sát; kiến trúc đầu ra kép theo giới tính
> không tạo thêm lợi ích đáng tin cậy. Bất đồng giữa các dự đoán TTA có thể làm
> giàu nhóm ca sai số cao, nhưng liên hệ còn yếu và không đồng nhất giữa giới
> tính–nhóm tuổi, vì vậy chưa thể xem là uncertainty lâm sàng đã hiệu chỉnh.

Giá trị khoa học của đề tài không phụ thuộc vào việc vượt MAE của tác giả tham
chiếu. Nó nằm ở việc trả lời có kiểm soát các câu hỏi: thông tin nào thực sự có
giá trị, mức độ phức tạp nào là cần thiết, cải tiến nào không đem lại hiệu quả,
và một uncertainty proxy đơn giản có giới hạn ở đâu.

## 2. Cơ sở để xem đây là đóng góp khoa học

Một kết quả được xem là đóng góp khi nó tạo thêm tri thức có thể kiểm tra, không
chỉ tạo thêm mã nguồn hoặc một con số MAE. Chuỗi P0–P13 đáp ứng tiêu chí này nhờ:

- câu hỏi nghiên cứu và biến can thiệp được xác định rõ;
- so sánh trên cùng bệnh nhi và cùng recipe khi có thể;
- effect size đi kèm bootstrap confidence interval;
- gate đi/không đi tiếp được khóa trước;
- công bố cả kết quả dương và kết quả âm;
- OOF coverage đủ 14.036 ID cho phân tích TTA/uncertainty;
- tách kết quả phát triển khỏi tập test đã bị truy cập;
- artifact, hash, cấu hình và quyết định được lưu để kiểm toán.

## 3. Các đóng góp khoa học chính

### C1. Định lượng vai trò của giới tính trong dự đoán tuổi xương

Đây là đóng góp mạnh nhất của đề tài.

P11 so sánh E0 image-only và E1 sex embedding trên cùng 1.425 bệnh nhi, giữ
cố định split, augmentation, backbone, loss, optimizer, scheduler và tiêu chí
chọn checkpoint.

| Mô hình | MAE (tháng) | Nữ | Nam |
|---|---:|---:|---:|
| E0 — chỉ ảnh | 7,4717 | 7,7686 | 7,2213 |
| E1 — sex embedding | 6,1848 | 6,3891 | 6,0125 |

- E0−E1 overall: **+1,2869 tháng**, 95% CI **[+1,0114; +1,5667]**.
- Cải thiện ở nữ: +1,3795 tháng; cải thiện ở nam: +1,2088 tháng.
- Khoảng tin cậy của cả hai giới đều hoàn toàn lớn hơn 0.

Kết luận được phép:

> Trong recipe ConvNeXt đã khóa, thông tin giới tính mang lại giá trị dự đoán
> bổ sung lớn và nhất quán so với mô hình chỉ dùng ảnh.

Không được suy diễn rằng thí nghiệm đã chứng minh một cơ chế nhân quả sinh học.
Kết quả chứng minh giá trị dự đoán của biến giới tính trong dữ liệu và protocol
được khảo sát.

### C2. Chứng minh kiến trúc chuyên biệt theo giới không mặc nhiên tốt hơn

E2 sử dụng backbone chung và hai đầu ra riêng theo giới tính. So với E1:

- E2 MAE: **6,1571 tháng**.
- E2−E1: **−0,0277 tháng**, 95% CI **[−0,1831; +0,1254]**.
- Cải thiện ở nữ chỉ 0,0052 tháng và ở nam 0,0467 tháng; các CI đều cắt 0.
- Khoảng cách MAE nữ–nam tăng từ 0,377 tháng ở E1 lên 0,418 tháng ở E2.
- E2 không đạt gate cải thiện overall 0,10 tháng hoặc female 0,20 tháng.

Đây là một kết quả âm có giá trị: sex embedding đơn giản đạt tỷ lệ hiệu
năng–độ phức tạp hợp lý hơn dual-output trong điều kiện hiện tại. Việc dừng E3,
seed bổ sung và OOF cho E2 sau khi không đạt gate giúp tránh chọn lọc kết quả có
lợi sau khi đã nhìn dữ liệu.

### C3. Đánh giá giá trị và giới hạn của TTA disagreement như uncertainty proxy

P12 phân tích 14.036 dự đoán OOF, không dùng RSNA test. Disagreement được tính
từ 10 TTA views và được đánh giá như tín hiệu xếp hạng rủi ro.

- Spearman giữa disagreement và absolute error: **ρ=0,2004**.
- Bootstrap 95% CI: **[0,1842; 0,2164]**.
- AUROC phát hiện sai số >12 tháng: **0,6258**.
- AUROC phát hiện sai số >18 tháng: **0,6341**.
- Q4 disagreement có MAE 7,6238 so với 4,7587 ở Q1.
- Q4 có tỷ lệ lỗi >12 tháng gấp 2,86 lần và lỗi >18 tháng gấp 3,29 lần Q1.

Kết quả tạo ra hai tri thức đồng thời:

1. disagreement có giá trị phân tầng rủi ro ở mức quần thể;
2. utility phân biệt từng ca còn hạn chế, chưa đủ làm uncertainty lâm sàng hoặc
   cơ chế tự động từ chối dự đoán.

Đây là đóng góp có ý nghĩa vì nghiên cứu không đồng nhất “có tương quan thống kê”
với “đủ hữu ích cho ứng dụng”.

### C4. Phát hiện tính không đồng nhất theo giới tính và nhóm tuổi

P12 cho thấy một uncertainty proxy chung không hoạt động đồng đều:

- nữ: ρ=0,1462, AUROC >12 tháng 0,5888;
- nam: ρ=0,2386, AUROC >12 tháng 0,6576;
- F 180–228 có ρ=0,3564 nhưng chỉ n=365;
- F 0–59 không có association đáng tin cậy;
- nhóm có sai số cao nhất, M 60–119, chỉ có ρ=0,0870 và AUROC 0,5730.

Age-standardized sex gap nhỏ hơn gap quan sát chưa chuẩn hóa, cho thấy phân bố
tuổi giải thích một phần nhưng không toàn bộ khác biệt giữa hai giới.

Đóng góp này liên quan trực tiếp tới fairness và safety: hiệu năng tổng thể có
thể che khuất các phân nhóm mà tín hiệu cảnh báo hoạt động kém.

### C5. Xây dựng bằng chứng thực nghiệm về các cải tiến không hiệu quả

Các kết quả âm không được xem là thất bại kỹ thuật nếu thí nghiệm hợp lệ và câu
kết luận được giới hạn đúng phạm vi.

| Giai đoạn | Can thiệp | Kết quả chính | Quyết định |
|---|---|---|---|
| P2 | augmentation nhẹ A2 | cải thiện 0,455 tháng, CI [0,249; 0,658] | giữ A2 |
| P3 | full-hand background masking | kém hơn 0,054 tháng, CI cắt 0 | loại B1 |
| P4–P5 | multi-scale/LDL | cải thiện nhỏ, không ổn định qua seed | giữ D0 đơn giản |
| P6 | tăng 512 lên 768 px | cải thiện 0,001 tháng, CI rất rộng quanh 0 | giữ 512 |
| P10 | Deeplasia augmentation vừa | delta +0,001 tháng, CI cắt 0 | không giữ |
| P9-I | cross-fitted bias correction | không cải thiện; làm TTA kém hơn | loại correction |
| P9-B0 | EfficientNet-B0 Deeplasia screening | MAE 8,529–11,043, kém control | dừng screening |
| P11 | dual-output theo giới tính | cải thiện 0,028 tháng, không qua gate | giữ E1 |

Chuỗi kết quả này hỗ trợ kết luận tổng quát ở mức đề tài: tăng độ phức tạp không
mặc nhiên cải thiện dự đoán; biến điều kiện hóa phù hợp và thiết kế đánh giá chặt
có giá trị hơn việc liên tục mở rộng kiến trúc.

### C6. Đóng góp về quy trình thực nghiệm và khả năng tái lập

Đề tài đã xây dựng một protocol nghiên cứu mạnh hơn cách báo cáo một lần train:

- audit 12.611 train / 1.425 validation / 200 test, kiểm tra duplicate và hash;
- 5-fold OOF đủ 14.036 ID, pooled MAE 6,3167 tháng, CI [6,2246; 6,4113];
- paired bootstrap ở cấp bệnh nhi;
- xác nhận nhiều seed cho LDL trước khi loại;
- checkpoint/resume kèm RNG, config, code và data hash;
- gate định trước và dừng candidate không đạt;
- manifest P13 với 9 input, 15 output, hash khớp và build lặp lại 0/15 artifact
  thay đổi;
- ghi rõ `test_accessed=false` cho P11–P13.

Đây là đóng góp phương pháp ở cấp đồ án: kết luận có dấu vết kiểm toán, tái lập
và ít phụ thuộc hơn vào một run thuận lợi.

## 4. Vai trò của các kết quả hỗ trợ

### P7 và P8

- P7 cung cấp baseline OOF ổn định: MAE 6,3167 tháng trên 14.036 bệnh nhi, SD
  giữa năm fold 0,0788 tháng.
- P8 ensemble trên 200 test đạt MAE 4,7303 tháng, nhưng chưa đạt các mốc tác giả
  tham chiếu 3,68–3,87 tháng.

P8 là bằng chứng kỹ thuật và benchmark hỗ trợ, không phải đóng góp khoa học trung
tâm. Không được dùng chênh lệch P7 OOF và P8 test để tuyên bố ensemble cải thiện,
vì hai kết quả đến từ hai tập dữ liệu khác nhau.

### TTA

TTA cải thiện paired MAE **0,1070 tháng**, 95% CI **[0,0775; 0,1361]** trên
14.036 OOF. Đây là cải thiện nhỏ nhưng nhất quán, phù hợp làm đóng góp hỗ trợ về
inference. Nó không nên được đặt ngang với effect 1,2869 tháng của sex embedding.

## 5. Những nội dung không nên gọi là đóng góp khoa học chính

- Việc chỉ triển khai ConvNeXt, EfficientNet hoặc sex embedding có sẵn.
- Một con số MAE đơn lẻ không có baseline hoặc uncertainty interval.
- P8 MAE 4,7303 như một tuyên bố state of the art.
- Số lượng lớn thí nghiệm nếu không gắn với câu hỏi nghiên cứu.
- Disagreement TTA như xác suất mô hình sai hoặc độ tin cậy lâm sàng.
- Dual-output như mô hình tốt hơn chỉ vì point estimate thấp hơn 0,0277 tháng.

## 6. Mức độ đóng góp theo chuẩn công bố

### Đồ án tốt nghiệp

**Đạt rõ ràng.** Đề tài vượt mức một đồ án chỉ huấn luyện mô hình và báo MAE nhờ
có ablation, OOF, confidence interval, negative result, subgroup analysis,
uncertainty analysis và artifact tái lập.

### Bài báo sinh viên, workshop hoặc hội nghị trong nước

**Có khả năng hình thành bài báo**, nếu tập trung vào một câu chuyện duy nhất:
vai trò của giới tính, tính đủ dùng của sex embedding và giới hạn của TTA
disagreement. Không nên viết như một bài thi đua state of the art.

### Tạp chí hoặc hội nghị quốc tế yêu cầu cao

**Chưa đủ mạnh ở trạng thái hiện tại.** Các giới hạn chính gồm:

- P11 mới là screening một seed trên một validation split;
- chưa có external holdout hoàn toàn chưa bị tác động;
- RSNA test 200 ảnh đã được truy cập nên không còn là confirmatory holdout mới;
- P12 dùng cùng development OOF và chưa có xác nhận ngoài miền;
- disagreement chưa được calibration thành xác suất và chưa có referral threshold;
- giới tính trong dữ liệu là nhãn nhị phân sẵn có;
- chưa thực hiện systematic literature review để chứng minh novelty tuyệt đối.

Những giới hạn này làm giảm phạm vi tuyên bố nhưng không phủ nhận đóng góp khoa
học ở cấp độ luận văn.

## 7. Bốn đóng góp đề xuất ghi trong luận văn

> **Thứ nhất**, luận văn thực hiện đánh giá có kiểm soát vai trò của thông tin
> giới tính trong dự đoán tuổi xương và cho thấy sex embedding cải thiện MAE
> khoảng 1,29 tháng so với mô hình chỉ sử dụng ảnh.
>
> **Thứ hai**, luận văn chỉ ra rằng kiến trúc đầu ra kép theo giới tính không tạo
> ra cải thiện đáng tin cậy so với sex embedding đơn giản, qua đó cung cấp một
> kết quả âm có ý nghĩa cho lựa chọn kiến trúc.
>
> **Thứ ba**, luận văn đánh giá bất đồng giữa các dự đoán TTA như một tín hiệu
> phân tầng nguy cơ trên 14.036 dự đoán OOF, đồng thời xác định khả năng phân
> biệt sai số lớn còn hạn chế và không đồng nhất giữa giới tính–nhóm tuổi.
>
> **Thứ tư**, luận văn xây dựng quy trình thực nghiệm có khả năng tái lập, sử
> dụng kiểm tra rò rỉ dữ liệu, phân tích cặp, bootstrap confidence interval,
> tiêu chí dừng định trước và manifest kiểm toán.

## 8. Cách diễn đạt khi viết và bảo vệ

### Nên dùng

- “cung cấp bằng chứng trong protocol/dữ liệu được khảo sát”;
- “thông tin giới tính mang giá trị dự đoán bổ sung”;
- “sex embedding đạt tỷ lệ hiệu năng–độ phức tạp tốt hơn dual-output”;
- “kết quả âm theo gate định trước”;
- “disagreement có giá trị phân tầng nhưng utility còn hạn chế”;
- “cần external validation trước khi suy rộng”.

### Không nên dùng

- “vượt state of the art”;
- “mô hình mới tốt hơn tác giả”;
- “đã chứng minh khả năng tổng quát hóa lâm sàng”;
- “giới tính gây ra sai khác tuổi xương”;
- “uncertainty phát hiện chính xác ca mô hình sai”;
- “dual-output tốt hơn E1”.

## 9. Hướng nâng cấp đóng góp

Theo thứ tự ưu tiên:

1. xác nhận E0–E1 trên external holdout chưa bị tác động;
2. nếu không có external data, xác nhận hiệu ứng E0–E1 qua nhiều seed hoặc OOF;
3. khóa trước protocol calibration/selective prediction và đánh giá trên dữ
   liệu độc lập;
4. thực hiện systematic literature review để xác định novelty chính xác;
5. đánh giá domain shift theo bệnh viện, thiết bị và nhóm tuổi nếu có dữ liệu.

Không nên ưu tiên mở thêm kiến trúc phức tạp chỉ để tìm MAE thấp hơn khi đóng góp
trung tâm hiện tại đã rõ và các giới hạn xác nhận quan trọng hơn.

## 10. Kết luận cuối

Đề tài **có đóng góp khoa học**. Đóng góp mạnh nhất là bằng chứng định lượng rằng
giới tính là thông tin dự đoán quan trọng, trong khi sex embedding đơn giản đã
đủ hiệu quả hơn về tính gọn so với dual-output. Đóng góp thứ hai là đánh giá
trung thực TTA disagreement: có khả năng làm giàu nhóm nguy cơ nhưng chưa đủ cho
uncertainty lâm sàng và hoạt động không đồng đều giữa giới–tuổi. Chuỗi ablation
âm cùng protocol tái lập củng cố giá trị của kết luận.

Vị trí khoa học đúng của đề tài là một nghiên cứu thực nghiệm có kiểm soát về
**vai trò của giới tính, tính cần thiết của độ phức tạp kiến trúc và giới hạn của
uncertainty proxy** trong dự đoán tuổi xương — không phải một tuyên bố vượt mô
hình tốt nhất đã công bố.

## 11. Nguồn bằng chứng nội bộ

- `AI_Context/01_STATUS_RESULTS.md`
- `AI_Context/02_METHOD_HISTORY.md`
- `AI_Context/08_SEX_AWARE_EXPERIMENT_PLAN.md`
- `AI_Context/09_P12_UNCERTAINTY_PROTOCOL.md`
- `AI_Context/10_P13_THESIS_REPORTING_PLAN.md`
- `p11_sex_aware/P11_STAGE2_HANDOFF.md`
- `p12_uncertainty/P12_HANDOFF.md`
- `p13_reporting/P13_HANDOFF.md`
- `p13_reporting/P13_THESIS_DRAFT_VI.md`
