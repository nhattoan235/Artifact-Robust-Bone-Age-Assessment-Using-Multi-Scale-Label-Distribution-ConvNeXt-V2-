# Phương án B — Artifact Consistency Training

- Ngày: 2026-08-23
- Trạng thái: thiết kế; chưa code/train.
- Vai trò: phương án dự phòng về robustness, không tự động thay thế A và không mặc định là hướng giảm clean MAE tốt nhất.

## 1. Mục tiêu

Đo xem việc buộc model ổn định trước các biến dạng ảnh nhẹ, hợp lý với artifact chụp X-quang, có giúp giảm lỗi trên dữ liệu sạch hoặc dữ liệu bị suy giảm hay không.

## 2. Giới hạn phương pháp

Không dùng B để:

- lọc các hard case theo prediction;
- tuyên bố robustness lâm sàng từ perturbation tổng hợp;
- thử nhiều config rồi chọn kết quả tốt nhất;
- thay A chỉ vì stress MAE giảm;
- biến consistency thành cơ chế làm mất tín hiệu tuổi.

## 3. Thiết kế model và loss

- Backbone E1 ConvNeXt-Tiny + sex embedding.
- Mỗi ảnh có clean view và một artifact view nhẹ.
- Perturbation gồm blur, gamma, contrast và noise trong khoảng được khóa trước.
- Khoảng perturbation chỉ được hiệu chuẩn từ quality metric của training split.
- Không dùng central occlusion hoặc biến đổi làm phá xương.
- Giữ nguyên split, backbone, sex protocol và target protocol của E1.
- Loss gồm clean regression, artifact regression và consistency giữa hai prediction.
- Warmup 3 epoch chỉ với clean loss.
- Từ epoch 4–8 mới ramp artifact/consistency loss.
- Early stopping dựa trên clean validation MAE, không dựa riêng stress MAE.

## 4. Kiểm soát confound

Nếu B cho kết quả tốt, phải chạy thêm augmentation-only control với consistency weight bằng 0. Chỉ khóa một cấu hình trước khi train; không search nhiều hệ số trên cùng OOF.

Nếu augmentation-only cũng cải thiện, không được quy toàn bộ lợi ích cho consistency. Nếu chỉ consistency cải thiện, phải báo cáo rõ đó là đóng góp của consistency.

## 5. Đánh giá

Primary endpoint:

- clean pooled OOF MAE theo đúng 5 fold.

Secondary endpoints:

- synthetic stress MAE;
- độ lệch prediction giữa clean và artifact view;
- MAE theo sex và age bins;
- calibration của disagreement;
- tỷ lệ case thay đổi prediction quá lớn.

## 6. Điều kiện giữ B

Chỉ giữ B như ứng viên nếu:

- clean MAE không giảm có ý nghĩa;
- stress MAE cải thiện;
- không có subgroup bị hại;
- kết quả không chỉ đến từ một fold hoặc một loại perturbation;
- augmentation-only control không giải thích toàn bộ kết quả.

Nếu clean MAE không cải thiện nhưng stress MAE cải thiện, kết luận đúng là synthetic robustness, không phải cải thiện clinical performance.

## 7. Rủi ro chính

- Perturbation tổng hợp không đại diện hoàn toàn cho artifact ngoài đời.
- Chi phí train gần gấp đôi vì hai view.
- Consistency quá mạnh có thể xóa tín hiệu tuổi.
- Teacher/model ban đầu yếu có thể truyền sai lầm.
- Gain có thể thực chất đến từ augmentation chứ không phải consistency.

Tài liệu liên quan: AI_Context/12_ABC_EXPERIMENT_ROADMAP.md và AI_Context/13_ABC_EXPERIMENT_SPECIFICATION.md.

