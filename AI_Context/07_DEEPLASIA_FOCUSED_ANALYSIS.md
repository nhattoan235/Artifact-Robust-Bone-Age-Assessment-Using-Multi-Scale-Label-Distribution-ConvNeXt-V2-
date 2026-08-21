# Phân tích trọng tâm Deeplasia và định hướng vượt mốc tham khảo

> **Trạng thái:** Tài liệu định hướng nghiên cứu, cập nhật theo trạng thái P0–P10 hiện tại.
> P10-B0 và P10-B1 đã hoàn tất; các kết quả dưới đây không dùng nhãn test để chọn mô hình.
>
> **Kết luận ngắn:** Deeplasia là bài báo nền chính của dự án. Bram chỉ nên được xem là mốc đối chứng và nguồn ý tưởng mở rộng; không nên tiếp tục dùng Bram làm trục tái lập chính khi recipe của Bram không đủ khả năng tái lập.

## 1. Định vị đúng bài toán

Mục tiêu thực tế của đồ án không nên được diễn đạt là “thử thật nhiều mô hình để tìm MAE thấp hơn”. Câu hỏi nghiên cứu có giá trị hơn là:

> **Một quy trình tái lập chặt chẽ các thành phần cốt lõi của Deeplasia, sau đó bổ sung calibration, ensemble và khả năng chống artifact một cách leakage-safe, có thể cải thiện hiệu năng và độ tin cậy trên dự đoán tuổi xương hay không?**

Theo cách định vị này:

- **Deeplasia** là baseline chính cần tái lập và phân tích.
- **P0–P8** là nền tảng audit dữ liệu, split, preprocessing, huấn luyện và đánh giá.
- **Bram** là baseline đối chứng/nguồn ý tưởng phụ, không phải mục tiêu tái lập bắt buộc.
- **P9/P10 trở đi** nên tập trung trả lời vì sao pipeline hiện tại còn cách Deeplasia và thành phần nào có thể tạo ra cải thiện có ý nghĩa.

Deeplasia báo cáo MAE khoảng **3,87 tháng** trên RSNA test. Mô hình hiện tại đạt **4,73 tháng** trên 200 ảnh test bằng ensemble P8. Tuy nhiên, hai con số này chưa phải là phép so sánh “cùng recipe”: Deeplasia dùng ensemble dị thể, test-time augmentation và bias correction; P8 chủ yếu là ensemble nhiều fold của cùng một cấu hình ConvNeXt-Tiny. Vì vậy, chưa thể kết luận ConvNeXt-Tiny hoặc hướng nghiên cứu hiện tại yếu hơn về bản chất. [Bài báo Deeplasia](https://link.springer.com/article/10.1007/s00247-023-05789-1)

**Cập nhật P10:** P10-B0 đã tái lập chính xác P2 trên validation (MAE 6,1847917; toàn bộ 1.425 dự đoán giống hệt). P10-B1 thay riêng augmentation Deeplasia mức vừa nhưng đạt MAE 6,1858772, không cải thiện có ý nghĩa. Vì vậy, vấn đề hiện tại không nằm ở việc tăng augmentation nhẹ; các bước tiếp theo phải tập trung vào inference Deeplasia, backbone/ensemble dị thể và calibration.

## 2. Những gì P0–P8 đã làm tốt

P0–P8 đã tạo được một nền thực nghiệm tương đối chắc:

| Thành phần | Trạng thái hiện tại | Ý nghĩa |
|---|---|---|
| Split RSNA | Đã audit | Không phát hiện leakage trùng ảnh theo kiểm tra hiện có |
| Baseline | ConvNeXt-Tiny, grayscale lặp 3 kênh, sex embedding, direct regression | Có pipeline huấn luyện và resume ổn định |
| Augmentation | A2 tốt nhất trong các cấu hình đã thử | Cho thấy augmentation có ảnh hưởng, nhưng chưa phải augmentation Deeplasia |
| Official hand mask | Đã kiểm tra | Masking đơn độc chưa cải thiện trong recipe ConvNeXt hiện tại |
| Resolution | 768 không vượt 512 | Không có lý do tiếp tục tăng độ phân giải nếu chưa đổi recipe |
| Multi-seed | Đã xác nhận | Chênh lệch các seed nhỏ, tránh kết luận từ một seed đơn lẻ |
| OOF 5-fold | Đã hoàn tất | Có cơ sở tốt để chọn mô hình và calibration mà không chạm test |
| Test ensemble | Đã chạy | Có kết quả thăm dò tốt, nhưng không được dùng để chọn các thử nghiệm tiếp theo |

Kết quả P7 OOF là **6,31669 tháng MAE**, còn P8 đạt **4,73032 tháng MAE** trên 200 ảnh test. Khoảng cách giữa OOF và test khá lớn, do test chỉ có 200 ảnh và có thể có phân phối thuận lợi hơn. Do đó, điểm mạnh đáng tin cậy nhất hiện nay là quy trình OOF; điểm P8 nên được báo cáo là đánh giá thăm dò, không phải bằng chứng xác nhận cuối cùng.

Chi tiết số liệu nằm trong [báo cáo P7 OOF](../p7_final_v3/P7_OOF_report.json), [báo cáo P8 test ensemble](../p8_test_ensemble/outputs/P8_test_ensemble_report.json) và [tổng hợp trạng thái dự án](01_STATUS_RESULTS.md).

## 3. Khoảng cách quan trọng giữa Deeplasia và P0–P8

Điểm cốt lõi là P0–P8 mới tái lập **triết lý** của Deeplasia, chưa tái lập đầy đủ **performance recipe** của Deeplasia.

### 3.1. Backbone và ensemble chưa tương đương

Deeplasia sử dụng các mô hình EfficientNet ở nhiều điều kiện:

- EfficientNet-B0 ở kích thước 512;
- EfficientNet-B4 ở kích thước 512;
- EfficientNet-B0 ở kích thước 1024;
- nhiều cấu hình fully connected head;
- chọn các mô hình tốt từ tập ứng viên rồi trung bình dự đoán của ba mô hình được chọn.

P8 dùng năm fold của cùng một cấu hình ConvNeXt-Tiny. Đây là ensemble theo **data split**, còn Deeplasia là ensemble theo **kiến trúc, độ phân giải và head**. Hai cơ chế này có thể giảm sai số theo những cách khác nhau. Đặc biệt, nếu các mô hình P8 có dự đoán tương quan cao, thêm fold mới sẽ đem lại lợi ích nhỏ hơn so với các mô hình có lỗi bổ sung cho nhau.

### 3.2. Augmentation hiện tại nhẹ hơn đáng kể

Augmentation A2 hiện dùng flip, xoay nhỏ, dịch chuyển nhỏ, scale nhẹ, brightness/contrast và gamma. Code Deeplasia chính thức có recipe mạnh hơn, gồm affine transform với phạm vi lớn hơn, rotation khoảng ±30°, shear, sharpen, random resized crop và nhánh CLAHE hoặc gamma. Đây là một khác biệt có thể kiểm chứng trực tiếp, không phải suy đoán kiến trúc.

Không nên kết luận “augmentation mạnh chắc chắn tốt hơn”. Cách đúng là chạy một thí nghiệm Deeplasia-faithful có kiểm soát, sau đó so sánh OOF theo cùng seed/fold và khoảng tin cậy. [Mã nguồn Deeplasia chính thức](https://github.com/aimi-bonn/Deeplasia)

### 3.3. Inference chưa tái lập đủ

Deeplasia có hai thành phần dễ bị bỏ sót:

1. **Test-time augmentation (TTA):** dự đoán ở nhiều góc xoay và tùy chọn flip rồi lấy trung bình.
2. **Linear bias correction:** hiệu chỉnh độ dốc và intercept của dự đoán dựa trên dữ liệu train/validation.

Nếu P8 chưa áp dụng hai thành phần này theo đúng cách, việc so sánh 4,73 với 3,87 vẫn chưa công bằng. Đây là nhóm thử nghiệm có chi phí thấp nhưng tiềm năng cao hơn việc tiếp tục đổi backbone ngẫu nhiên.

Bias correction cần được fit bằng train/OOF hoặc validation theo quy trình cross-fitting. Không được fit trực tiếp từ nhãn test, kể cả khi nhãn test đã được đọc trong P8.

### 3.4. Training recipe khác nhau

Deeplasia dùng EfficientNet, head và optimizer/scheduler riêng trong code chính thức. Pipeline hiện tại dùng SmoothL1, AdamW-like training, cosine schedule, learning rate và weight decay khác. Vì vậy, các kết quả P3–P6 chủ yếu trả lời câu hỏi:

> “Thành phần này có tốt trong recipe ConvNeXt hiện tại không?”

chứ chưa trả lời:

> “Thành phần này có tốt trong recipe Deeplasia không?”

Đây là lý do kết quả masking không cải thiện không đủ để bác bỏ preprocessing của Deeplasia.

## 4. Điểm yếu khoa học của Deeplasia có thể khai thác

Các điểm dưới đây là cơ hội để tạo đóng góp mới, nhưng cần phân biệt rõ “hạn chế được bài báo nêu hoặc quan sát từ thiết kế” với “lỗi của bài báo”.

### 4.1. Chưa phân rã đóng góp của từng thành phần

Deeplasia có nhiều thành phần cùng lúc: mask, EfficientNet, sex input, nhiều head, nhiều độ phân giải, ensemble, TTA và bias correction. Kết quả cuối cùng tốt, nhưng nếu không có ablation đầy đủ thì chưa biết thành phần nào tạo ra bao nhiêu lợi ích.

Đây là khoảng trống tốt nhất cho đồ án: xây dựng ablation có kiểm soát để đo riêng:

- backbone/độ phân giải;
- head;
- augmentation;
- TTA;
- bias correction;
- ensemble dị thể.

Đóng góp như vậy có giá trị hơn việc chỉ thay ConvNeXt bằng một backbone khác rồi báo một con số MAE.

### 4.2. Nguy cơ overfit validation khi chọn nhiều mô hình

Deeplasia đánh giá nhiều cấu hình rồi chọn các mô hình tốt để ensemble. Nếu tất cả quyết định chọn đều dựa trên một validation split, có nguy cơ hiệu năng được tối ưu một phần cho split đó. Đây chưa phải bằng chứng có leakage, nhưng là nguy cơ lựa chọn mô hình nhiều lần trên cùng validation set.

Quy trình cải tiến nên dùng:

- OOF hoặc nested selection để chọn mô hình;
- quy tắc chọn được định nghĩa trước;
- báo cáo cả kết quả từng model và ensemble;
- bootstrap confidence interval;
- nếu có thể, một external holdout chưa từng chạm.

### 4.3. Test RSNA nhỏ

RSNA test chỉ có 200 ảnh. Một vài ca sai số lớn có thể thay đổi MAE đáng kể. Vì vậy, chênh lệch 0,1–0,2 tháng không nên được diễn giải là cải thiện thực chất nếu không có khoảng tin cậy hoặc kiểm định paired trên cùng ảnh.

Mục tiêu thực nghiệm hợp lý là đạt cải thiện ổn định trên OOF, sau đó kiểm tra test như một đánh giá cuối cùng. Không được dùng test để tiếp tục điều chỉnh ensemble weight, preprocessing hoặc checkpoint.

### 4.4. Khả năng tổng quát hóa ngoài RSNA còn hạn chế

Deeplasia có đánh giá thêm DHA và German Dysplastic Bone Dataset, nhưng các tập này nhỏ hơn, nhãn có thể đến từ số lượng người đánh giá khác nhau, và nhóm bệnh lý chỉ bao phủ một số rối loạn. Bài báo cũng ghi nhận cần cohort lớn hơn và nhiều bệnh hơn.

Đây là hướng vượt bài báo có tính thuyết phục: không chỉ giảm MAE RSNA mà còn giảm độ nhạy với domain shift, artifact, dân số và bệnh lý. Nếu chưa có external dataset phù hợp, có thể dùng stress test artifact như bằng chứng phụ, nhưng phải gọi đúng là robustness evaluation chứ không phải external validation.

### 4.5. Test–retest chưa phải lặp ảnh trực tiếp

Độ chính xác test–retest trong Deeplasia được ước lượng từ dữ liệu dọc, với giả định tiến triển tuổi xương tuyến tính. Đây là chỉ số hữu ích nhưng không hoàn toàn tương đương với hai ảnh chụp gần như cùng thời điểm của cùng bệnh nhân.

Đồ án có thể bổ sung phân tích uncertainty và prediction disagreement để xem mô hình có tự nhận biết những ca không chắc chắn hay không. Tuy nhiên, cần tránh gọi disagreement giữa các fold là uncertainty lâm sàng đã được hiệu chuẩn.

### 4.6. Attention map mới mang tính định tính

Việc attention tập trung vào khớp đốt ngón, xương bàn tay và xương cổ tay là hợp lý về mặt chuyên môn, nhưng heatmap không chứng minh mô hình thật sự dựa vào vùng đó. Một đóng góp tốt hơn là đánh giá faithfulness bằng occlusion hoặc mask vùng giải phẫu và đo thay đổi dự đoán.

## 5. Diễn giải đúng các kết quả hiện tại

### 5.1. Không nên kết luận “masking thất bại”

P3 cho thấy official full-hand mask không cải thiện trong pipeline ConvNeXt hiện tại. Kết luận an toàn là:

> **Masking đơn độc, khi ghép với recipe hiện tại, chưa tạo ra cải thiện có ý nghĩa thống kê.**

Kết luận không an toàn là:

> **Masking không có ích cho dự đoán tuổi xương.**

Vì Deeplasia dùng mask cùng với normalization, augmentation, EfficientNet, ensemble và inference correction. Cần kiểm tra lại mask trong một reproduction gần Deeplasia trước khi loại bỏ nó.

### 5.2. P8 cho thấy hiệu năng test tốt nhưng bias theo tuổi còn rõ

Trong P8, nhóm 60–119 tháng có MAE khoảng 6,62 tháng và bias dương; nhóm 180–228 tháng có bias âm. Điều này gợi ý hiện tượng shrinkage về trung bình: mô hình có xu hướng dự đoán cao hơn ở nhóm nhỏ tuổi và thấp hơn ở nhóm lớn tuổi.

Đây là cơ hội cải thiện có cơ sở hơn việc đổi backbone:

- linear calibration;
- isotonic calibration nếu đủ kiểm soát overfit;
- age-bin calibration cross-fitted;
- sampling hoặc loss cân bằng theo tuổi, nhưng chỉ sau khi có baseline Deeplasia-faithful.

Calibration phải được đánh giá đồng thời trên MAE, RMSE, bias theo age-bin, giới tính và sai số lớn; không tối ưu duy nhất một subgroup rồi làm xấu toàn bộ phân phối.

### 5.3. Kết luận sau P10

P10-B0 đã hoàn tất và tái lập chính xác P2, xác nhận pipeline mới không gây drift. P10-B1 augmentation Deeplasia mức vừa có MAE gần như không đổi và CI chênh lệch chứa 0. Do đó, không tiếp tục tăng độ mạnh augmentation một cách mù quáng; mọi ứng viên mới phải có giả thuyết riêng và được đánh giá trên OOF.

## 6. Lộ trình ưu tiên có xác suất thành công cao nhất

### Giai đoạn 0 — kiểm soát tính đúng của pipeline (đã hoàn tất)

P10-B0 đã tái lập P2 chính xác trên validation; P10-B1 cho thấy augmentation mức vừa chưa tạo cải thiện. Cổng kiểm soát đạt, không cần chạy thêm biến thể augmentation nhẹ.

### Giai đoạn 1 — inference chi phí thấp trên OOF

Trước khi train backbone mới, triển khai trên các dự đoán OOF hiện có:

- TTA xoay/flip theo Deeplasia;
- linear bias correction fit bằng train/OOF cross-fitting;
- so sánh raw, TTA, bias correction và TTA + bias correction.

Đây là bước rẻ, leakage-safe và cho biết khoảng cách còn lại có đến từ inference hay không.

### Giai đoạn 2 — tái lập một model Deeplasia đơn

Chạy EfficientNet-B0 ở 512 với:

- official hand mask;
- masked-image normalization;
- sex embedding 32 chiều;
- augmentation gần code Deeplasia;
- loss, scheduler và early stopping gần recipe chính thức;
- cùng split RSNA, cùng metric MAE/MAD và RMSE.

Chưa cần ensemble ngay. Mục tiêu là biết một model đơn tái lập được bao nhiêu phần khoảng cách.

### Giai đoạn 3 — ensemble dị thể kiểu Deeplasia

Sau khi model đơn ổn định, chạy lần lượt B0-512, B4-512 và B0-1024. Có thể sàng lọc trước bằng một head đơn giản, sau đó mở rộng các head nếu tài nguyên cho phép.

Quy tắc đề xuất:

- chọn bằng OOF, không chọn bằng test;
- equal-weight trước, tối ưu weight sau;
- báo cáo prediction correlation và diversity;
- chỉ giữ ensemble nếu cải thiện có độ ổn định qua fold/seed.

### Giai đoạn 4 — calibration và age-bias correction

Dùng OOF prediction để fit calibration cross-fitted. So sánh linear, isotonic và age-bin calibration có regularization. Chỉ giữ cách hiệu chỉnh nếu cải thiện MAE tổng thể mà không làm tăng mạnh sai số nhóm tuổi hoặc giới tính.

### Giai đoạn 5 — robustness với artifact

Artifact augmentation chỉ được tạo trên development train. 200 ảnh stress test hiện có nên giữ cố định để đánh giá thăm dò, không dùng để chọn model. Báo cáo:

- MAE ảnh sạch và ảnh có artifact;
- prediction shift trên từng cặp;
- tỷ lệ sai số vượt 3/6/12 tháng;
- thay đổi theo giới tính và age-bin;
- tương quan giữa disagreement/uncertainty và absolute error.

### Giai đoạn 6 — chỉ sau đó mới phát triển kiến trúc mới

ConvNeXtV2, multi-scale hoặc LDL chỉ nên quay lại nếu có giả thuyết rõ và đạt cải thiện ổn định tối thiểu khoảng 0,1 tháng trên OOF qua nhiều seed, không làm xấu subgroup. Nếu không, các hướng này nên được ghi nhận là thí nghiệm âm tính, không tiếp tục tiêu tốn nguồn lực chính.

## 7. Tiêu chí để tuyên bố “vượt Deeplasia”

Một tuyên bố mạnh cần thỏa ít nhất:

- cùng dataset/split hoặc giải thích rõ khác biệt protocol;
- không dùng test để chọn mô hình;
- so sánh trên cùng 200 ảnh nếu tái đánh giá được;
- báo cáo MAE, RMSE, median AE, accuracy ±6/±12/±18;
- paired bootstrap confidence interval;
- kết quả OOF và test được tách rõ;
- có ablation chứng minh nguồn cải thiện;
- có phân tích subgroup và robustness;
- nếu có thể, xác nhận trên external holdout.

Với tình trạng hiện tại, mục tiêu an toàn nhất không phải là hứa chắc chắn đạt dưới 3,87 tháng ngay lập tức. Mục tiêu có khả năng thực hiện cao hơn là chứng minh theo từng tầng:

1. tái lập được Deeplasia single-model;
2. tái lập được lợi ích của TTA/bias correction;
3. tái lập được ensemble dị thể;
4. bổ sung calibration hoặc robustness có lợi ích đo được;
5. chỉ sau đó mới kết luận có vượt mốc Deeplasia hay chưa.

## 8. Kết luận định hướng

Điểm yếu lớn nhất hiện tại không phải là thiếu một backbone mới. Điểm yếu là chưa biết khoảng cách 4,73–3,87 đến từ đâu vì pipeline hiện tại chưa tái lập đầy đủ recipe Deeplasia. Vì vậy, hướng có xác suất thành công và giá trị khóa luận cao nhất là:

> **Deeplasia-faithful reproduction → component ablation → heterogeneous ensemble → cross-fitted calibration → artifact robustness.**

Nếu kết quả cuối cùng chưa vượt 3,87 tháng, đồ án vẫn có thể có đóng góp tốt nếu chứng minh được thành phần nào thực sự tạo lợi ích, thành phần nào không tái lập được, và mô hình nào ổn định hơn dưới artifact hoặc domain shift. Một kết luận âm tính có kiểm soát vẫn có giá trị hơn một con số thấp nhưng được chọn sau nhiều lần chạm test.

## Tài liệu liên quan trong project

- [Trạng thái kết quả hiện tại](01_STATUS_RESULTS.md)
- [Lịch sử phương pháp](02_METHOD_HISTORY.md)
- [Protocol dữ liệu](03_DATA_PROTOCOL.md)
- [Kế hoạch P9 hiện tại](04_NEXT_P9_PLAN.md)
- [So sánh các pipeline](06_PIPELINE_COMPARISON.md)
- [Báo cáo P7 OOF](../p7_final_v3/P7_OOF_report.json)
- [Báo cáo P8 test ensemble](../p8_test_ensemble/outputs/P8_test_ensemble_report.json)
- [Mã testing và TTA của Deeplasia](../external/Deeplasia/lib/testing.py)
- [Mã dataset/preprocessing của Deeplasia](../external/Deeplasia/lib/datasets.py)
