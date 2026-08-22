# AI Context – Đọc trước khi tiếp tục

> **Nguồn sự thật hiện tại:** thư mục `AI_Context/`. Các file `PROJECT_CONTEXT.md` và `CHANGELOG.md` ở thư mục gốc là lịch sử cũ, không phải trạng thái mới nhất.

## Mục tiêu đồ án

Dự đoán tuổi xương (tháng) từ ảnh X-quang bàn tay RSNA. Mục tiêu học thuật là xây dựng quy trình có thể tái lập, đánh giá đúng trên split RSNA và phấn đấu vượt các mốc tham khảo Bram et al. (2025) và Rassmann/Deeplasia (2024).

## Trạng thái hiện tại

- **P0–P10:** đã hoàn tất audit dữ liệu, baseline, augmentation, preprocessing,
  kiến trúc, seed, độ phân giải, 5-fold OOF, test ensemble và các nhánh tái lập
  Deeplasia/TTA.
- **P7 OOF:** PASS, 14.036 mẫu, MAE **6,31669 tháng**; test không được dùng để
  chọn mô hình.
- **P8 test ensemble:** PASS kỹ thuật, MAE **4,73032 tháng** trên 200 ảnh, nhưng
  chưa vượt hai mốc công bố 3,68–3,87 tháng. Test đã bị truy cập nên không còn là
  confirmatory holdout mới.
- **P11 sex-aware:** E1 sex embedding tốt hơn E0 image-only **1,2869 tháng**,
  CI [1,0114; 1,5667]. E2 dual-output không hơn E1 và không giảm sex gap; giữ E1.
- **P12 uncertainty OOF:** TTA disagreement liên hệ yếu với sai số
  (ρ=0,2004); AUROC lỗi lớn 0,6258–0,6341. Có giá trị phân tầng nghiên cứu nhưng
  chưa phải uncertainty lâm sàng.
- **P13 reporting:** bảng/hình, manifest và bản thảo luận văn đã hoàn tất; báo
  cáo đóng góp khoa học nằm tại `11_SCIENTIFIC_CONTRIBUTIONS_REPORT.md`.
## Việc tiếp theo được khuyến nghị

1. Đọc `01_STATUS_RESULTS.md` và `02_METHOD_HISTORY.md` để nắm bằng chứng nền.
2. Đọc `08_SEX_AWARE_EXPERIMENT_PLAN.md`, `09_P12_UNCERTAINTY_PROTOCOL.md` và
   `10_P13_THESIS_REPORTING_PLAN.md` để hiểu các quyết định đã khóa.
3. Dùng `11_SCIENTIFIC_CONTRIBUTIONS_REPORT.md` để viết phần đóng góp, phạm vi
   tuyên bố và giới hạn của luận văn.
4. Hoàn thiện luận văn với E1 làm baseline chính. Nếu tiếp tục nghiên cứu, ưu
   tiên external holdout chưa bị tác động hoặc xác nhận E0–E1 qua nhiều seed/OOF.
5. Ghi mọi thay đổi tiếp theo vào `CHANGELOG.md`.
## Quy tắc bắt buộc

- Không dùng `rsna_test.csv` hoặc metric test để chọn checkpoint, preprocessing, augmentation, loss, ensemble weight hay hyperparameter.
- Không thay đổi manifest/hash/config của run đã hoàn tất.
- Train dài phải có checkpoint nguyên tử, resume, log `metrics`/`warnings`, và kiểm tra trạng thái sau mỗi lần ngắt Colab.
- Không tuyên bố “vượt bài báo” chỉ từ một lần chạy; cần cùng dataset, cùng protocol và khoảng tin cậy.

