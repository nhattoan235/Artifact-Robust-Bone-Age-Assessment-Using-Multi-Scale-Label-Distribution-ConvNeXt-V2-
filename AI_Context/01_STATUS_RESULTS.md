# Trạng thái và kết quả hiện hành

> Cập nhật: 2026-09-11
> Đơn vị MAE: tháng. Đây là nguồn trạng thái chính; báo cáo dài chỉ dùng để kiểm chứng chi tiết.

## Nhãn độ tin cậy

- **VERIFIED:** có prediction/report hoặc log đủ để kiểm tra lại.
- **PARTIAL:** mới có một số fold hoặc log giữa chừng; không dùng để kết luận.
- **HISTORICAL:** kết quả hợp lệ của nhánh cũ, không còn là quyết định hiện hành.

## Kết quả khóa

| Pipeline | Tập đánh giá | MAE | Trạng thái | Ý nghĩa |
|---|---:|---:|---|---|
| Ảnh toàn cảnh + ConvNeXt-Tiny + global average pooling + sex embedding | OOF 14.036 | **6,316691** | VERIFIED | Baseline phát triển chính |
| Pipeline trên, trung bình 5 fold | Test RSNA 200 | **4,730321** | VERIFIED, exploratory | Test đã được đọc; không dùng để chọn tiếp |
| ROI bàn tay margin 8% + ConvNeXt-Tiny + sex embedding | OOF 14.036 | **6,437349** | VERIFIED | Đơn lẻ kém baseline OOF |
| Pipeline ROI margin 8%, trung bình 5 fold | Test RSNA 200 | **4,337267** | VERIFIED, exploratory | Baseline test tốt nhất ổn định |
| Pipeline ROI margin 8% + 10-view TTA | Test RSNA 200 | **4,331841** | VERIFIED, exploratory | Point estimate thấp nhất; chênh với raw không có ý nghĩa |
| ROI tái tạo margin 12% + resize/padding Z26, không histogram equalization | OOF 14.036 | **6,324301** | VERIFIED | Gần baseline toàn cảnh |
| Pipeline trên, trung bình 5 fold | Test RSNA 200 | **4,665987** | VERIFIED, exploratory | Kém ROI margin 8% rõ ràng |

### Khoảng tin cậy và so sánh quan trọng

- Baseline toàn cảnh OOF: CI bootstrap 95% **[6,224638; 6,411327]**.
- ROI margin 12% Z26 không histogram equalization OOF: CI **[6,233724; 6,419696]**.
- Không histogram equalization so với cùng pipeline có histogram equalization: delta MAE **−0,067554**, CI **[−0,122398; −0,014280]**; bỏ equalization tốt hơn trên OOF.
- Không histogram equalization so với baseline toàn cảnh: delta **+0,007610**, CI **[−0,050270; +0,066445]**; chưa có khác biệt.
- Trên test 200, ROI margin 12% Z26 không equalization kém ROI margin 8% **+0,328720**, CI **[+0,056389; +0,612503]**.
- ROI margin 8% + TTA so với raw: delta **−0,005463**, CI **[−0,162908; +0,156851]**; chưa có bằng chứng TTA cải thiện pipeline này.

## Ensemble đã xác minh

| Ensemble | Tập | MAE | Kết luận |
|---|---:|---:|---|
| 0,5 × baseline toàn cảnh raw + 0,5 × ROI margin 8% raw | OOF 14.036 | **6,176212** | Tốt hơn baseline toàn cảnh 0,140480; CI delta [−0,172547; −0,109853] |
| Baseline toàn cảnh TTA + ROI margin 8% TTA, trọng số 0,5/0,5 | OOF 14.036 | **6,117080** | Tốt hơn baseline TTA 0,093366; CI [−0,120367; −0,067429] |
| Baseline toàn cảnh TTA + ROI margin 8% TTA, trọng số 0,5/0,5 | Test RSNA 200 | **4,400603** | Kém C3-ROI-TTA 0,068762; CI chứa 0 |
| Nested ensemble lịch sử: baseline TTA + EfficientNet TTA + ROI raw | OOF 14.036 | **6,075569** | Point estimate OOF tốt nhất; cần đọc báo cáo nguồn trước khi tái sử dụng |

## Sàng lọc đang diễn ra trên cùng raw C3-R2

Chỉ Fold 1–2; chưa được phép mở test hay tự động chạy Fold 3–5.

| Ứng viên | Fold 1 | Fold 2 | Pooled/Trạng thái | Quyết định |
|---|---:|---:|---|---|
| C0: ConvNeXt-Tiny + global average pooling chuẩn | 6,352325 | 6,378387 | **6,365354** | Đối chứng matched đã sẵn sàng |
| Đối chứng lịch sử C3-ROI V1 trên cùng hai fold | 6,375519 | 6,372684 | 6,374102 | C0 delta −0,008748; CI [−0,094621; +0,080216] |
| Bilinear pooling thay global average pooling | **6,385449** | Không hợp lệ | PARTIAL | Không promote; Fold 2 dừng vì bất ổn |
| C1: fine-tuning phân tầng + EMA | Epoch 1: EMA 31,4944; raw 15,7176 | Chưa có | PARTIAL/abnormal | Phải kiểm tra EMA/khởi tạo/training trước khi đánh giá |
| C2: ConvNeXt V2 pretrained mạnh hơn | Chưa chạy | Chưa chạy | PLANNED | Chỉ chạy sau quyết định C1 |

## Dữ liệu và fallback

- C3-ROI V1: margin 8%; fallback toàn ảnh **18,49% development**, **33% test**.
- C3-R2: margin 12% + border rescue; hard fallback **14,06% development**, **28% test**.
- Trong OOF C3-R2 Z26 không equalization, nhóm fallback có MAE **6,9723** (1.974 mẫu), cao hơn nhóm mask/bbox **6,2183** (12.062 mẫu).
- Không gọi C3-ROI là chuyên gia carpal thuần túy: input là ROI bàn tay theo bbox segmentation và một phần đáng kể fallback toàn ảnh.

## Benchmark bài báo

| Công trình | Dữ liệu đánh giá | MAE công bố | Có thể so trực tiếp? |
|---|---|---:|---|
| Shu & Yu, 2025 | RSNA test 200 | **4,42** | Gần trực tiếp; có thêm 8 đặc trưng kích thước xương bàn tay được đo |
| Rassmann et al., Deeplasia, 2024 | RSNA test 200 | **3,87** | Có; ensemble 3 model, DHA/GDBD chỉ dùng đánh giá ngoài |
| Bram et al., 2025 | RSNA test 200 | **3,68** | Có; ensemble 5 fold, loại 35 ca bất thường khỏi train/validation |
| Zhang et al., 2026 | RSNA validation 1.425 | **4,10** | Không đặt ngang test 200; chỉ dùng RSNA và các view dẫn xuất |

## Quyết định hiện hành

1. Giữ ROI margin 8% + ConvNeXt-Tiny + sex embedding làm baseline test.
2. Không promote histogram equalization, C3-R2 Z26 hoặc bilinear pooling từ bằng chứng hiện có.
3. Hoàn tất chẩn đoán và sàng lọc C1 trên Fold 1–2; nếu thất bại, chuyển C2 ConvNeXt V2.
4. Chọn ứng viên bằng validation/OOF. Test 200 chỉ báo cáo thăm dò vì đã được truy cập nhiều lần.
5. Không tuyên bố state-of-the-art; so sánh bài báo phải kèm split, ensemble/single model và khác biệt dữ liệu.

## Rủi ro cần nhớ

- OOF 14.036 và test 200 không cùng phân phối/kích thước; thứ hạng pipeline đã đảo chiều.
- Test 200 nhỏ và không còn untouched holdout.
- Run C1 có dấu hiệu EMA bất thường; không diễn giải log epoch 1 như kết quả cuối.
- Các checkpoint định kỳ lớn có thể làm đầy Google Drive; vẫn phải giữ ít nhất checkpoint tốt nhất và checkpoint gần nhất đã xác minh.
