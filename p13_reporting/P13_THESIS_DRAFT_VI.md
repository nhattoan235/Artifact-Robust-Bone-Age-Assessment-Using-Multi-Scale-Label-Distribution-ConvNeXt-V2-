# P13 — Bản thảo Methods–Results–Discussion cho luận văn

**Trạng thái:** Bản thảo dựa trên các kết quả đã khóa của P11 và P12  
**Phạm vi dữ liệu:** Chỉ tập development/validation và các dự đoán OOF đã được xác định trong protocol; không sử dụng RSNA test để lựa chọn mô hình, hiệu chỉnh, đặt ngưỡng hoặc viết kết luận.

## 1. Thông điệp khoa học đề xuất

Nghiên cứu này không đặt trọng tâm vào việc tuyên bố vượt qua kết quả tốt nhất đã công bố. Đóng góp chính là đánh giá có kiểm soát vai trò của giới tính và giá trị của một tín hiệu bất định hậu nghiệm trong dự đoán tuổi xương trẻ em trên X-quang bàn tay.

Ba kết luận trung tâm có thể bảo vệ được là:

1. Thông tin giới tính mang lại cải thiện lớn và nhất quán so với mô hình chỉ dùng ảnh.
2. Trong điều kiện thí nghiệm hiện tại, đầu ra kép theo giới tính không cải thiện có ý nghĩa so với sex embedding đơn giản; đây là một kết quả âm có giá trị, giúp loại bỏ một kiến trúc phức tạp không cần thiết.
3. Mức bất đồng giữa dự đoán ảnh gốc và các biến thể TTA có liên hệ với sai số, nhưng liên hệ còn yếu và khả năng phát hiện ca sai số lớn chỉ ở mức hạn chế; vì vậy chưa thể xem đây là thước đo bất định lâm sàng đã được hiệu chỉnh.

### Câu hỏi nghiên cứu

- **RQ1:** Việc bổ sung giới tính có cải thiện dự đoán tuổi xương so với mô hình chỉ dùng ảnh không?
- **RQ2:** Một kiến trúc đầu ra kép theo giới tính có tốt hơn sex embedding dùng chung không?
- **RQ3:** Mức bất đồng giữa các dự đoán TTA có thể nhận diện các trường hợp có nguy cơ sai số cao không?
- **RQ4:** Hiệu quả và tín hiệu bất định có đồng nhất giữa giới tính và các nhóm tuổi hay không?

## 2. Phương pháp

### 2.1. Thiết kế nghiên cứu và nguyên tắc chống rò rỉ

Các so sánh kiến trúc P11 được thực hiện trên cùng phân hoạch development/validation, cùng quy trình tiền xử lý, backbone, kích thước ảnh, augmentation, lịch tối ưu và tiêu chí chọn checkpoint. Khác biệt chủ đích duy nhất giữa các nhánh là cách sử dụng thông tin giới tính. Mọi so sánh được thực hiện theo từng bệnh nhi trên cùng tập validation gồm 1.425 trường hợp.

Tập RSNA test không được truy cập trong P13. Tập này cũng không được dùng để chọn kiến trúc, điều chỉnh siêu tham số, đặt ngưỡng bất định hoặc xây dựng câu chuyện kết quả. Vì tập test đã từng được xem trong một giai đoạn trước của toàn bộ dự án, nghiên cứu không trình bày nó như một external holdout chưa bị tác động. Kết luận chính vì vậy chỉ giới hạn ở development/validation và OOF.

### 2.2. Các mô hình so sánh

- **E0 — Image only:** mô hình chỉ nhận ảnh X-quang, không sử dụng giới tính.
- **E1 — Sex embedding:** backbone ảnh dùng chung; biểu diễn giới tính được nhúng và kết hợp với đặc trưng ảnh trước tầng hồi quy. Đây là baseline có giới tính.
- **E2 — Shared dual output:** backbone dùng chung và hai đầu ra theo giới tính; đầu ra tương ứng được chọn theo nhãn giới tính của mẫu.

E0 được dùng để kiểm định giá trị tăng thêm của giới tính. E2 được so trực tiếp với E1 để đánh giá liệu tăng độ chuyên biệt của kiến trúc có đem lại lợi ích thực tế hay không. Các tiêu chí chấp nhận và trình tự dừng/đi tiếp đã được khóa trước khi đọc kết quả cuối.

### 2.3. Chỉ số đánh giá và phân tích cặp

Chỉ số chính là mean absolute error (MAE, tháng). Các chỉ số bổ sung gồm RMSE, median absolute error, tỷ lệ dự đoán nằm trong ±6 tháng và ±12 tháng, MAE theo nữ/nam và khoảng cách MAE tuyệt đối giữa hai giới.

Do các mô hình được đánh giá trên cùng các bệnh nhi, chênh lệch MAE được tính theo cặp ở cấp mẫu. Khoảng tin cậy 95% được ước lượng bằng paired bootstrap 10.000 lần với seed 2026. Chênh lệch được định nghĩa là MAE của mô hình ứng viên trừ MAE của E1; giá trị dương của E0−E1 biểu thị E1 tốt hơn E0, còn giá trị âm của E2−E1 biểu thị E2 tốt hơn E1.

### 2.4. Phân tích bất định hậu nghiệm

P12 sử dụng dự đoán out-of-fold (OOF) của toàn bộ tập development gồm 14.036 trường hợp. Với mỗi ảnh, mô hình tạo dự đoán từ ảnh chuẩn và các biến thể test-time augmentation (TTA). Điểm bất đồng được xây dựng từ độ phân tán của các dự đoán này và được xem như một tín hiệu xếp hạng rủi ro, không phải xác suất sai số đã hiệu chỉnh.

Mối liên hệ giữa bất đồng và sai số tuyệt đối được đánh giá bằng tương quan hạng Spearman. Khả năng nhận diện ca sai số lớn được đánh giá bằng AUROC tại các ngưỡng sai số >12 và >18 tháng. Phân tích bổ sung gồm so sánh tứ phân vị bất đồng, risk–coverage, phân tầng theo giới tính và nhóm tuổi, cùng chênh lệch giữa dự đoán TTA trung bình và dự đoán chuẩn.

### 2.5. Giả thuyết đã khóa

- **H1:** E1 tốt hơn E0, chứng minh giá trị bổ sung của giới tính.
- **H2:** E2 tốt hơn E1 đủ để vượt qua cổng cải thiện đã định trước.
- **H3:** E2 giảm chênh lệch sai số giữa nữ và nam mà không làm suy giảm hiệu năng chung.
- **H4:** Bất đồng TTA có liên hệ dương với sai số tuyệt đối và có ích trong phân tầng rủi ro.

## 3. Kết quả

### 3.1. Hiệu năng tổng thể và theo giới tính

E1 đạt MAE 6,185 tháng, cải thiện rõ rệt so với E0 với MAE 7,472 tháng. Chênh lệch E0−E1 là 1,287 tháng (95% CI 1,012–1,570), và khoảng tin cậy không cắt 0. Cải thiện xuất hiện ở cả nữ, 1,379 tháng (95% CI 0,949–1,806), và nam, 1,209 tháng (95% CI 0,841–1,581). Kết quả này ủng hộ H1: giới tính là biến phụ trợ quan trọng cho bài toán dự đoán tuổi xương.

E2 đạt MAE 6,157 tháng, chỉ thấp hơn E1 0,028 tháng. Chênh lệch E2−E1 là −0,028 tháng (95% CI −0,179 đến 0,124), do đó không có bằng chứng cho thấy đầu ra kép tốt hơn sex embedding. Kết quả theo nữ là −0,005 tháng (95% CI −0,220 đến 0,210) và theo nam là −0,047 tháng (95% CI −0,264 đến 0,165); cả hai khoảng tin cậy đều cắt 0. H2 không được ủng hộ.

Khoảng cách MAE nữ–nam là 0,377 tháng ở E1 và 0,418 tháng ở E2. E2 không làm giảm chênh lệch này, nên H3 không được ủng hộ. Theo protocol, các thử nghiệm E3, nhiều seed và OOF cho E2 không được mở tiếp sau khi E2 thất bại ở cổng chính.

| Mô hình | MAE | RMSE | Median AE | Trong ±6 tháng | Trong ±12 tháng | MAE nữ | MAE nam | Khoảng cách giới |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 — chỉ ảnh | 7,472 | 9,928 | 6,000 | 53,26% | 81,75% | 7,769 | 7,221 | 0,547 |
| E1 — sex embedding | 6,185 | 8,486 | 5,000 | 63,44% | 86,67% | 6,389 | 6,012 | 0,377 |
| E2 — dual output | 6,157 | 8,368 | 4,500 | 63,02% | 87,37% | 6,384 | 5,966 | 0,418 |

### 3.2. Bất đồng TTA và nguy cơ sai số

Trên 14.036 dự đoán OOF, tương quan Spearman giữa điểm bất đồng TTA và sai số tuyệt đối là ρ=0,200 (95% CI 0,184–0,216). Mối liên hệ có hướng đúng và ổn định về thống kê, nhưng độ lớn yếu. AUROC để phát hiện sai số >12 tháng là 0,626 và để phát hiện sai số >18 tháng là 0,634, cho thấy khả năng phân biệt chỉ ở mức hạn chế.

Dù vậy, phân tầng theo tứ phân vị cho thấy tín hiệu có ý nghĩa mô tả: nhóm bất đồng cao nhất có MAE gấp 1,60 lần nhóm thấp nhất; tỷ lệ sai số >12 tháng gấp 2,86 lần và tỷ lệ sai số >18 tháng gấp 3,29 lần. Vì vậy H4 được ủng hộ ở mức **có liên hệ và có giá trị phân tầng**, nhưng không được diễn giải thành một bộ phát hiện thất bại đáng tin cậy cho lâm sàng.

Trung bình hóa TTA cải thiện MAE 0,107 tháng so với dự đoán chuẩn (95% CI 0,078–0,136). Lợi ích này nhỏ nhưng nhất quán, và nên được mô tả tách biệt với khả năng dùng bất đồng làm tín hiệu bất định.

### 3.3. Dị biệt theo giới tính và nhóm tuổi

Các heatmap theo giới tính–nhóm tuổi cho thấy cả sai số và mối liên hệ giữa bất đồng với sai số không đồng nhất giữa các phân nhóm. Do kích thước mẫu và phân bố tuổi khác nhau giữa các ô, các kết quả này mang tính thăm dò. Không nên dùng một ô riêng lẻ để tuyên bố ưu thế mô hình hoặc thiết lập ngưỡng lâm sàng. Giá trị chính của phân tích là chỉ ra rằng hiệu năng tổng thể có thể che khuất các vùng tuổi–giới tính khó dự đoán hơn.

## 4. Thảo luận

### 4.1. Ý nghĩa của thông tin giới tính

Mức cải thiện khoảng 1,29 tháng của E1 so với E0 lớn hơn đáng kể so với chênh lệch giữa hai cách tích hợp giới tính. Kết quả phù hợp với đặc điểm sinh học của trưởng thành xương: cùng tuổi theo lịch, tiến trình cốt hóa ở trẻ nam và nữ không hoàn toàn giống nhau. Về mặt mô hình hóa, giới tính giúp điều kiện hóa ánh xạ từ hình thái X-quang sang tuổi xương.

### 4.2. Vì sao kiến trúc phức tạp hơn không thắng

E2 có điểm ước lượng tốt hơn E1 rất ít, nhưng mức cải thiện nhỏ so với độ bất định và không qua cổng đã khóa. Một lời giải thích hợp lý là backbone dùng chung kết hợp sex embedding đã đủ khả năng biểu diễn phần lớn tương tác ảnh–giới tính; tách đầu ra làm giảm số mẫu hiệu dụng cho mỗi nhánh mà không bổ sung đủ tín hiệu. Đây là diễn giải hậu nghiệm, không phải cơ chế đã được chứng minh.

Kết quả âm này vẫn có giá trị khoa học: nó cho thấy việc thêm cấu trúc chuyên biệt theo giới không mặc nhiên tốt hơn một cơ chế điều kiện hóa đơn giản. Với dữ liệu hiện tại, E1 hợp lý hơn về tỷ lệ hiệu năng–độ phức tạp.

### 4.3. Giá trị và giới hạn của bất đồng TTA

Bất đồng TTA tập trung nhiều ca sai số lớn hơn ở các tứ phân vị cao, nhưng tương quan và AUROC còn thấp. Tín hiệu này phù hợp hơn cho mục đích xếp hạng ca cần rà soát hoặc hỗ trợ phân tích lỗi hơn là tự động từ chối dự đoán. Trước khi ứng dụng, cần hiệu chỉnh trên dữ liệu độc lập, xác định utility/cost của referral, và kiểm tra tính ổn định giữa thiết bị, bệnh viện, giới tính và nhóm tuổi.

### 4.4. Đóng góp thực tế của đồ án

Giá trị của đồ án nằm ở thiết kế so sánh có kiểm soát, quy tắc cổng định trước, phân tích cặp kèm khoảng tin cậy, công bố kết quả âm, và phân tích bất định OOF không phóng đại kết quả. Câu chuyện này chặt chẽ hơn việc chỉ tìm kiếm một điểm MAE thấp hơn tác giả tham chiếu trên một lần chia dữ liệu.

## 5. Hạn chế

1. P11 là giai đoạn sàng lọc trên một seed và một tập validation; khoảng tin cậy bootstrap phản ánh biến thiên do lấy mẫu bệnh nhi, không phản ánh đầy đủ biến thiên do huấn luyện lại mô hình.
2. P12 dùng dự đoán OOF và có protocol khóa trước khi tổng hợp chính thức, nhưng vẫn là phân tích hậu nghiệm phát triển từ pipeline hiện hữu.
3. Bootstrap cho Spearman và các chỉ số phân tầng đánh giá độ ổn định nội bộ; nó không thay thế xác nhận trên một quần thể độc lập.
4. Tập RSNA test đã từng được truy cập ở giai đoạn trước của dự án nên bị loại khỏi việc chứng minh khả năng khái quát chưa thiên lệch trong P13.
5. Chưa có external holdout hoàn toàn chưa bị tác động, và chưa đánh giá dịch chuyển miền theo bệnh viện, thiết bị chụp hoặc quần thể.
6. Bất đồng TTA chưa được hiệu chỉnh thành xác suất sai số, chưa có ngưỡng referral được xác nhận và không nên gọi là “độ tin cậy lâm sàng”.
7. Các phân tích giới tính sử dụng nhãn nhị phân sẵn có của bộ dữ liệu; nghiên cứu không đánh giá được các trường hợp thiếu nhãn hoặc sự khác biệt giữa giới tính ghi nhận và các yếu tố sinh học liên quan đến trưởng thành xương.
8. Phân tích theo nhóm tuổi có thể chịu ảnh hưởng của mất cân bằng số mẫu và nhiều phép so sánh; các kết quả phân nhóm chỉ nên xem là thăm dò.

## 6. Kết luận đề xuất

Trong thiết kế đánh giá hiện tại, bổ sung giới tính bằng sex embedding cải thiện rõ rệt dự đoán tuổi xương so với mô hình chỉ dùng ảnh. Việc thay bằng đầu ra kép theo giới tính không tạo ra cải thiện đáng tin cậy và cũng không giảm khoảng cách sai số giữa hai giới. Bất đồng giữa các dự đoán TTA có liên hệ với sai số và giúp phân tầng nhóm nguy cơ, nhưng khả năng phân biệt ca sai số lớn còn hạn chế. Kết quả ủng hộ một mô hình gọn với sex embedding làm baseline chính, đồng thời xác định external holdout chưa bị tác động và hiệu chỉnh bất định là các bước cần thiết trước khi đưa ra kết luận về khả năng khái quát hoặc ứng dụng lâm sàng.

## 7. Hướng dẫn chèn bảng và hình

### Bảng

- **Bảng 1 — Hiệu năng E0/E1/E2:** `outputs/P13_THESIS_REPORT/table_1_model_comparison.md`
- **Bảng 2 — So sánh bootstrap theo cặp:** `outputs/P13_THESIS_REPORT/table_2_paired_forest_data.md`
- **Bảng 3 — Bất định theo giới tính–nhóm tuổi:** `outputs/P13_THESIS_REPORT/table_3_p12_sex_age.md`
- **Bảng 4 — Kết luận giả thuyết:** `outputs/P13_THESIS_REPORT/table_4_hypothesis_summary.md`

### Hình

- **Hình 1 — Forest plot:** `outputs/P13_THESIS_REPORT/figure_1_paired_forest.png` hoặc `.pdf`. Dùng ngay sau mục 3.1 để minh họa chênh lệch MAE theo cặp và khoảng tin cậy 95%.
- **Hình 2 — Heatmap giới tính–nhóm tuổi:** `outputs/P13_THESIS_REPORT/figure_2_sex_age_heatmaps.png` hoặc `.pdf`. Dùng trong mục 3.3; ghi rõ đây là phân tích thăm dò.
- **Hình 3 — Bất đồng và rủi ro:** `outputs/P13_THESIS_REPORT/figure_3_disagreement_risk.png` hoặc `.pdf`. Dùng trong mục 3.2 để trình bày tứ phân vị sai số, tỷ lệ sai số lớn và risk–coverage.

Caption chính thức, đồng bộ với generator, nằm trong `outputs/P13_THESIS_REPORT/figure_captions.json`.

## 8. Quy tắc diễn đạt khi bảo vệ

Nên dùng:

- “ủng hộ giả thuyết trong tập validation/OOF hiện tại”;
- “cải thiện có khoảng tin cậy bootstrap không cắt 0”;
- “tín hiệu bất định có liên hệ nhưng utility còn hạn chế”;
- “kết quả âm theo cổng định trước”;
- “chưa được xác nhận trên external holdout chưa bị tác động”.

Không nên dùng:

- “vượt state of the art”;
- “đã chứng minh tổng quát hóa”;
- “mô hình an toàn cho lâm sàng”;
- “bất đồng TTA là xác suất mô hình sai”;
- “E2 tốt hơn E1” chỉ dựa trên chênh lệch điểm 0,028 tháng.

## 9. Tái lập kết quả

```powershell
python -m unittest -v p13_reporting.test_reporting
python -m p13_reporting.build_thesis_assets --output-dir p13_reporting/outputs/P13_THESIS_REPORT --bootstrap 10000 --seed 2026
```

`report_manifest.json` lưu cấu hình, SHA-256 của đầu vào/đầu ra và cờ `test_accessed=false` để kiểm toán P13.
