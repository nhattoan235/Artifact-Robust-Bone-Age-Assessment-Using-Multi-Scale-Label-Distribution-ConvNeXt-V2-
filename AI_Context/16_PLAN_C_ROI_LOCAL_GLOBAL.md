# Phương án C — C-ROI Local/Global Ensemble

- Ngày: 2026-08-23
- Trạng thái: thiết kế đã khóa; chưa cache/train.
- Vai trò: hướng dự phòng khi A không đạt, tập trung vào thông tin giải phẫu độ phân giải cao.

## 1. Mục tiêu

Bổ sung cho E1 global một model local độ phân giải cao quanh vùng carpal, metacarpal và MCP, đặc biệt để kiểm tra nhóm lỗi ở khoảng 60–132 tháng.

## 2. Nguyên tắc khoa học

- Không dùng nhãn bone-age bên ngoài.
- Không dùng annotation keypoint bên ngoài.
- Không routing theo tuổi.
- Không lọc prediction sau khi xem nhãn.
- Không post-hoc tune ensemble weight.
- Có thể dùng Deeplasia mask đã audit như fixed preprocessing geometry; phải công khai việc mask có external segmentation supervision.
- ROI model phải được báo cáo như một nhánh độc lập, không bắt buộc đánh bại E1 standalone.

## 3. Kiến trúc

- E1 whole-hand được khóa.
- C-ROI là ConvNeXt-Tiny độc lập trên ROI.
- Ensemble chính khóa ở 0.5 E1 + 0.5 C-ROI.
- Cùng direct regression, sex embedding, SmoothL1 và recipe với E1.
- C-ROI khởi tạo độc lập từ ImageNet, không warm-start từ E1.
- Version 1 chỉ dùng một ROI rộng, không learned gate và không nhiều ROI.
- ROI standalone vẫn được đánh giá để biết nguồn cải thiện.

## 4. Quy tắc tạo ROI

1. Restore hoặc regenerate audited mask và khóa source SHA.
2. Chọn hand component lớn nhất.
3. Dùng PCA để lấy trục dài của hand.
4. Phân biệt phía finger và wrist bằng profile của mask, không dùng target hoặc prediction.
5. Crop từ khoảng 28% chiều dài tính từ phía finger đến hết wrist, giữ toàn bộ bề rộng và thêm khoảng 8% margin.
6. Chỉ thực hiện một affine/resize từ ảnh gốc; không mask pixel.
7. Pad thành ảnh vuông rồi resize 512.

Tọa độ ROI chỉ được kiểm tra bằng anatomical coverage trên training-QC, không tối ưu theo MAE.

## 5. Fallback và audit

- Mask thiếu, rỗng hoặc sai kích thước: fallback full-hand.
- Có hai component chính: fallback, không tự chọn một component.
- Orientation không chắc: crop rộng đối xứng, sau đó fallback full-hand nếu vẫn không chắc.
- Không loại ảnh khỏi dataset.
- Nếu mask thiếu và geometric fallback vượt 1%, phải dừng trước train để sửa localization.
- Cache image SHA, mask SHA, coordinates, angle, fallback reason và output SHA.

## 6. Pilot và train

- Giữ 5 fold P7, seed 42.
- Local: microbatch 6, accumulation 6, effective batch 36.
- Colab T4: có thể thử batch 12, accumulation 3 sau khi kiểm tra ổn định.
- Fold 1 chỉ là operational pilot: preflight, forward/backward và interrupt/resume.
- Không rút ra kết luận khoa học từ riêng Fold 1.
- Nếu không có NaN/Inf, OOM lặp lại, collapse, lỗi resume hoặc fallback vượt 1% thì chạy Fold 2–5.
- Ước tính: local khoảng 3.5–4.5 giờ/fold; T4 khoảng 2.5–4 giờ/fold tùy I/O.

## 7. Gate đánh giá

Endpoint 1, raw:

- So sánh 0.5 E1 raw + 0.5 C-ROI raw với E1 raw 6.316691.
- Cần giảm ít nhất 0.10 MAE.
- Paired bootstrap 95% CI phải nằm hoàn toàn dưới 0.
- Ít nhất 4/5 fold cải thiện.

Endpoint 2, TTA:

- Chạy P9 cho C-ROI.
- So sánh 0.5 E1-TTA + 0.5 C-ROI-TTA với E1-TTA 6.210446.
- Cần giảm ít nhất 0.10 MAE.

Cả hai endpoint phải đồng thời tránh:

- sex MAE giảm quá 0.15;
- age-bin MAE tăng quá 0.20;
- cell sex × age có n ít nhất 200 tăng quá 0.30;
- prediction collapse hoặc mean shrinkage;
- ID thiếu/trùng/non-finite.

Nếu raw dương nhưng TTA thất bại, giữ C như một ablation dương, không thay thế E1-TTA.

## 8. Failure matrix

- Mask SHA/count mismatch: dừng, restore đúng mask.
- ROI sai anatomy: fallback và kiểm tra lại; fallback trên 1% thì không train.
- Affine gây blur: sửa về một lần resampling.
- Collapse/overfit: early stop, không sửa loss hậu nghiệm để cứu kết quả.
- C quá giống E1: không tự tune weight; báo cáo redundancy.
- TTA làm xấu ROI: giữ raw result, không dùng TTA.
- Mất điện/runtime: resume từ checkpoint atomic, không overwrite prediction cũ.

## 9. Claim policy

Claim được phép nếu đạt gate:

An anatomy-aware local/global ensemble improved paired five-fold OOF MAE on the RSNA development set under a locked protocol.

Không tuyên bố xác nhận vượt mốc Bram 3.68 bằng test đã từng được truy cập trước đó; test chỉ dùng một lần cho báo cáo cuối cùng theo protocol đã khóa.

Tài liệu liên quan: AI_Context/11_C_ROI_FALLBACK_PLAN.md.

