# AI Context – Đọc trước khi tiếp tục

> **Nguồn sự thật hiện tại:** thư mục `AI_Context/`. Các file `PROJECT_CONTEXT.md` và `CHANGELOG.md` ở thư mục gốc là lịch sử cũ, không phải trạng thái mới nhất.

## Mục tiêu đồ án

Dự đoán tuổi xương (tháng) từ ảnh X-quang bàn tay RSNA. Mục tiêu học thuật là xây dựng quy trình có thể tái lập, đánh giá đúng trên split RSNA và phấn đấu vượt các mốc tham khảo Bram et al. (2025) và Rassmann/Deeplasia (2024).

## Trạng thái hiện tại

- **P0–P7:** đã hoàn tất; dữ liệu, leakage, augmentation, preprocessing, kiến trúc, seed, độ phân giải và 5-fold final đã được audit.
- **P7 OOF:** PASS, 14.036 mẫu, MAE **6,31669 tháng**; test không được dùng để chọn mô hình.
- **P8 test ensemble:** PASS về mặt kỹ thuật; ensemble 5 fold trên 200 ảnh có MAE **4,73032 tháng**.
- **Thí nghiệm C / C3-ROI:** đã hoàn tất 5-fold OOF. C3-ROI riêng đạt MAE 6,43735; ensemble cố định E1 + C3-ROI 50/50 đạt **6,17621**, giảm 0,14048 tháng so với E1 với paired CI hoàn toàn dưới 0 và thắng cả 5 fold. Tuy nhiên ROI fallback 18,49% ở development và 33% ở test, nên run không đạt gate hình học ≤1% của thiết kế C ban đầu.
- **So với công bố:** Bram 2025 báo cáo 3,68 tháng; Deeplasia 2024 báo cáo 3,87 tháng trên RSNA test. Kết quả hiện tại **chưa vượt** hai mốc này.
- Vì test ground truth đã được đọc ở P8, mọi thử nghiệm chọn mô hình sau đây phải dùng OOF/validation; P8 phải được gọi là đánh giá thăm dò nếu dùng để định hướng P9. Muốn có tuyên bố xác nhận cuối cùng cần một test ngoài/đánh giá mới chưa chạm.

## Việc tiếp theo được khuyến nghị

1. Đọc `01_STATUS_RESULTS.md` để nắm số liệu chính xác.
2. Đọc `02_METHOD_HISTORY.md` để biết vì sao các nhánh đã bị loại.
3. Đọc `03_DATA_PROTOCOL.md` trước khi chạm dữ liệu.
4. Làm theo `04_NEXT_P9_PLAN.md`: tái lập công bằng recipe Bram (preprocessing + augmentation + 100 epoch/early stop) và chỉ chọn cấu hình bằng OOF.
5. Dùng `05_FILE_MAP.md` để tìm artifact; ghi mọi thay đổi vào `CHANGELOG.md`.
6. Đọc `24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md` trước khi diễn giải hoặc tái sử dụng kết quả phương án C.

## Quy tắc bắt buộc

- Không dùng `rsna_test.csv` hoặc metric test để chọn checkpoint, preprocessing, augmentation, loss, ensemble weight hay hyperparameter.
- Không thay đổi manifest/hash/config của run đã hoàn tất.
- Train dài phải có checkpoint nguyên tử, resume, log `metrics`/`warnings`, và kiểm tra trạng thái sau mỗi lần ngắt Colab.
- Không tuyên bố “vượt bài báo” chỉ từ một lần chạy; cần cùng dataset, cùng protocol và khoảng tin cậy.

