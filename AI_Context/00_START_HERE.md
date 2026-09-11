# AI Context — bắt đầu tại đây

> Cập nhật: 2026-09-11
> Mục đích: giúp AI/agent nắm đúng trạng thái dự án với số token tối thiểu.

## Đường đọc mặc định

Chỉ đọc theo thứ tự sau, dừng khi đã đủ thông tin cho tác vụ:

1. `AI_Context/00_START_HERE.md` — phạm vi, thuật ngữ và định tuyến.
2. `AI_Context/context_index.json` — trạng thái máy đọc và số liệu khóa.
3. `AI_Context/01_STATUS_RESULTS.md` — kết quả đã xác minh và trạng thái run mới.
4. Chỉ mở thêm một tệp theo nhu cầu:
   - lịch sử quyết định: `02_METHOD_HISTORY.md`;
   - dữ liệu/leakage/test policy: `03_DATA_PROTOCOL.md`;
   - việc cần làm tiếp: `04_CURRENT_PLAN.md`;
   - tìm code/artifact: `05_FILE_MAP.md`;
   - so sánh pipeline và bài báo: `06_PIPELINE_COMPARISON.md`.

Không đọc toàn bộ `AI_Context` theo mặc định. Các tệp còn lại là báo cáo lịch sử hoặc bằng chứng chuyên sâu.

## Thứ tự ưu tiên khi tài liệu mâu thuẫn

1. `context_index.json` và `01_STATUS_RESULTS.md` cho trạng thái/số liệu hiện hành.
2. JSON/CSV/log gốc được chỉ ra trong `05_FILE_MAP.md` để kiểm chứng.
3. Báo cáo chuyên sâu theo ngày cho lịch sử và provenance.

Tên `00_CURRENT_DECISIONS_P14_HANDOFF_2026_08_25.md` là tên legacy: đó là snapshot P14 ngày 2026-08-25, **không phải trạng thái toàn dự án hiện tại**.

## Mục tiêu hiện tại

Dự đoán tuổi xương theo tháng từ ảnh X-quang bàn tay của bộ RSNA. Mục tiêu ngắn hạn là cải thiện MAE trên đúng protocol RSNA, nhưng mọi lựa chọn mô hình phải dựa trên validation/OOF; bộ test 200 ảnh đã được truy cập nhiều lần nên chỉ còn là benchmark thăm dò, không phải holdout xác nhận mới.

## Pipeline và tên gọi cần hiểu đúng

- **E1/P7:** ảnh toàn cảnh; ConvNeXt-Tiny pretrained ImageNet-1K; global average pooling, final LayerNorm, sex embedding, direct regression, augmentation A2, ảnh 512.
- **C3-ROI V1:** ROI bàn tay margin 8%; ConvNeXt-Tiny và sex embedding như E1; fallback toàn ảnh 18,49% ở development và 33% ở test.
- **C3-R2:** tái tạo ROI margin 12% với bước border-rescue; hard fallback 14,06% ở development và 28% ở test.
- **Z26:** họ preprocessing lấy cảm hứng từ Zhang et al. 2026. Phải ghi rõ `HE` hoặc `NO_HE`; không dùng “Z26” một mình để suy ra histogram equalization.
- **TTA:** trung bình 10 biến thể xoay {-10, -5, 0, 5, 10 độ} × flip/no-flip.
- **OOF:** mỗi ảnh development chỉ được dự đoán bởi fold không dùng ảnh đó để fit.

## Chốt trạng thái trong một phút

- Baseline E1 OOF: **6,316691** tháng; test: **4,730321**.
- C3-ROI V1 OOF: **6,437349**; test: **4,337267**.
- C3-ROI-TTA test có point estimate thấp nhất hiện có: **4,331841**, nhưng gần như ngang C3 raw và CI paired chứa 0.
- C3-R2 Z26 NO_HE OOF: **6,324301**; test: **4,665987**. OOF gần E1 nhưng test kém C3-ROI V1 rõ ràng.
- Đối chứng raw C3-R2 + GAP trên Fold 1–2: pooled **6,365354**; không khác có ý nghĩa so với C3-ROI V1 trên cùng hai fold.
- Raw C3-R2 + bilinear: mới có Fold 1 hợp lệ, MAE **6,385449**; Fold 2 bị dừng vì instability nên chưa có kết luận hai fold.
- Fine-tuning mới + EMA đang ở giai đoạn screening; log epoch 1 bất thường, chưa có kết quả cuối để so sánh.
- Benchmark bài báo trực tiếp trên RSNA test 200: Shu & Yu **4,42**, Deeplasia **3,87**, Bram et al. **3,68**. Zhang et al. 2026 **4,10** là validation 1.425 ảnh, không đặt trong bảng test-200.

## Quy tắc bắt buộc

1. Không dùng nhãn/MAE test để chọn preprocessing, model, checkpoint, TTA hoặc trọng số ensemble.
2. Luôn phân biệt validation 1.425, OOF 14.036 và test 200; không so trực tiếp như cùng một tập.
3. Chỉ gọi một kết quả `VERIFIED` khi có prediction/report hoặc log đủ để tái tính; log giữa chừng là `PARTIAL`.
4. Không sửa đè artifact đã chạy. Run mới phải có namespace, config hash, manifest hash, code version và resume state riêng.
5. Khi nói “vượt bài báo”, phải nêu cùng split hay không, single model hay ensemble, và test có còn độc lập hay không.

## Tài liệu lịch sử

Các tệp có tên P9, P14, C4, D3, Deeplasia analysis hoặc artifact checklist vẫn có giá trị bằng chứng, nhưng không còn là nguồn quyết định hiện hành. `_P14_TASKS_TEMP` chưa được xóa vì điều kiện cleanup trong chính thư mục đó chưa đạt; không dùng nó làm nguồn trạng thái chính.
