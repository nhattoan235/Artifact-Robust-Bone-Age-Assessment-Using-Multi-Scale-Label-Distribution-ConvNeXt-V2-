# So sánh pipeline hiện tại với Bram và Deeplasia

## Kết luận một câu

Pipeline hiện tại **tiên tiến hơn về tính tái lập, kiểm soát thí nghiệm và audit**; nhưng **chưa tiên tiến hơn về phương pháp dự đoán cốt lõi** và chưa đạt hiệu năng của hai bài tham khảo.

## Ma trận so sánh

| Thành phần | Pipeline P0–P8 hiện tại | Bram 2025 | Deeplasia 2024 | Đánh giá |
|---|---|---|---|---|
| Backbone | ConvNeXt-Tiny | ConvNeXt | EfficientNet b0/b4, có điều kiện 1024 | Hiện tại cùng họ với Bram, không mới hơn |
| Sex input | Có sex embedding | Có sex input | Có sex representation | Tương đương |
| Head/loss | Direct regression, Smooth L1 | Regression, MAE loss | Regression FC | Hiện tại chưa có lợi thế rõ |
| Preprocessing | P7 chính: `none`; mask B1 đã thử nhưng bị loại | Segmentation + histogram equalization + alignment | Hand-background masking tự động | Hiện tại **thua về mức hoàn thiện preprocessing** |
| Augmentation | A2: flip + hình học/cường độ nhẹ | Augmentation mở rộng, có ablation | Nhiều điều kiện train, chi tiết trong supplement | Hiện tại có nền tốt nhưng chưa tái lập đầy đủ Bram |
| Resolution/model diversity | 512, 5 fold cùng cấu hình | 5 fold cùng ConvNeXt | EfficientNet b0 512, b4 512, b0 1024; chọn ensemble đa điều kiện | Deeplasia đa dạng model hơn |
| Ensemble | Equal-weight 5 fold | Equal mean 5 fold | Ensemble 3 model được chọn từ 9 model | P8 đúng chuẩn nhưng chưa phong phú bằng Deeplasia |
| Hyperparameter search | Ablation có khóa, phạm vi hẹp | Search weight decay/LR/dropout | Nhiều cấu hình FC/backbone/resolution | Bram/Deeplasia rộng hơn |
| External validation | Chưa có trong P8 | RHPE, DHA, institutional 200 | 568 ảnh skeletal dysplasia + test–retest 149 | Hiện tại yếu hơn rõ |
| Uncertainty/CI | Bootstrap CI, fold dispersion, subgroup metrics | Fold SD uncertainty + bootstrap/Bland–Altman | Test–retest và validation dysplasia | P8 mạnh về audit CI nhưng chưa có clinical external validation |
| Reproducibility | Manifest SHA, code/config hash, resume qua Colab, storage audit | Có mô tả training nhưng không cùng mức handoff vận hành | Open-source code/masks, reproducible baseline | Đây là **điểm mạnh nhất của chúng ta** |
| Leakage protocol | Khóa test, OOF 14.036, audit ID/SHA | 5-fold RSNA; đánh giá thêm nhiều dataset | RSNA train/val/test và cohort dysplasia | Hiện tại nghiêm ngặt cho development, nhưng test P8 đã mở |

## Những gì thực sự có thể gọi là “tiên tiến” của chúng ta

1. **Kỷ luật thực nghiệm:** mọi phase đều có baseline, paired comparison, bootstrap CI, tiêu chí giữ/loại được đặt trước.
2. **Tái lập khi train dài:** checkpoint chứa optimizer/scheduler/scaler/RNG, hash dữ liệu–config–code, resume được giữa nhiều phiên Colab/tài khoản.
3. **Đánh giá OOF quy mô development:** 14.036 dự đoán OOF, ID duy nhất, subgroup theo sex/age và kiểm tra prediction collapse.
4. **Phân tích thất bại có hệ thống:** đã định danh feature collapse của D1 và loại các nhánh D2/B1/D3 bằng tiêu chí định lượng, thay vì giữ mô hình phức tạp chỉ vì “mới”.

Đây là đóng góp về **engineering/research methodology**, phù hợp để trình bày trong đồ án. Tuy nhiên không nên gọi là đóng góp kiến trúc mới.

## Những gì chưa thể gọi là tiên tiến

- ConvNeXt-Tiny + sex + direct regression + 5-fold mean về cơ bản là baseline gần Bram.
- P7 chưa dùng preprocessing mà Bram báo cáo là nguồn giảm lỗi lớn nhất trong ablation.
- P8 chưa có external validation như Bram (RHPE/DHA/institutional) hoặc Deeplasia (skeletal dysplasia/test–retest).
- Không có chronological-age branch vì RSNA không cung cấp biến này; không được tạo giả hoặc dùng tuổi nhãn làm input.
- P8 MAE 4,73032 tháng chưa vượt Bram 3,68 hoặc Deeplasia 3,87 trên RSNA test.

## Quyết định nghiên cứu

Không nên tiếp tục gọi P7/P8 là “mô hình tiên tiến hơn bài báo”. Cách diễn đạt đúng là: **một pipeline tái lập và audit nghiêm ngặt, hiện đang là baseline mạnh; P9 sẽ bổ sung preprocessing và protocol Bram-faithful để kiểm tra khả năng cải thiện có kiểm soát.**

Nguồn: [Bram et al. 2025](https://doi.org/10.1177/03635465251359618), [Rassmann et al. 2024](https://doi.org/10.1007/s00247-023-05789-1).
