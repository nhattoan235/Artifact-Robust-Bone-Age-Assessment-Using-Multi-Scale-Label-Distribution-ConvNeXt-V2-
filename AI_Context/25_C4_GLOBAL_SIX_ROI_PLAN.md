# C4 — Global + sáu ROI giải phẫu

## Mục tiêu

Kiểm tra liệu việc kết hợp ảnh toàn bàn tay với sáu vùng giải phẫu gồm một vùng
carpal/wrist và năm vùng MCP có cải thiện dự đoán tuổi xương so với ConvNeXt-Tiny
ảnh toàn bàn tay hay không.

## Kiến trúc khóa trước

- Một mẫu tương ứng một `image_id`, gồm một ảnh global và sáu ROI.
- Một ConvNeXt-Tiny chia sẻ trọng số xử lý bảy view.
- Feature của các view được nối theo thứ tự cố định, ghép với sex embedding rồi
  đưa qua regression head.
- So sánh `global_only`, `six_roi_only` và `global_plus_six`.
- Không thêm spatial attention trong C4 V1.

## Kiểm soát leakage

Sáu ROI không được xem là sáu mẫu độc lập. Tất cả view từ cùng ảnh giữ chung
`image_id`, target, sex và fold. ROI được tạo sau khi split nguồn đã khóa và
không dùng tuổi, prediction hoặc test để chọn vị trí.

## Quy trình

1. Pilot image-only 100 ảnh để kiểm tra hình học ROI.
2. Khóa ROI protocol và tạo cache 14.036 ảnh development.
3. Screening một fold trên development.
4. Chỉ cấu hình vượt gate mới chạy fresh 5-fold.
5. Tổng hợp OOF và paired bootstrap CI với E1 và E1+C3.
6. Chỉ sau khi khóa OOF mới chạy test 200 dưới nhãn exploratory.

Chi tiết triển khai nằm tại
`docs/plans/2026-08-30-global-six-roi-bone-age.md`.
