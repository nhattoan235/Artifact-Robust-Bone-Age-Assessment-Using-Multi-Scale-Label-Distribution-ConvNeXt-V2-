# Kế Hoạch Triển Khai Thử Nghiệm — Tách Bàn Tay & Ghép Nền Mới (Segment-and-Composite)

**Mục tiêu:** Kiểm chứng tính khả thi của phương án thay thế cho ControlNet Canny Inpainting: tách chính xác bàn tay khỏi nền bằng segmentation (không dùng generative model), ghép vào nền mới có texture nhiễu tự nhiên — loại bỏ hoàn toàn rủi ro "Text Hallucination" mà không cần retrain LoRA.

**Cách dùng file này:** Copy từng khối TASK theo thứ tự dán vào Antigravity. Mỗi Task có "Điều kiện dừng để duyệt" — không bỏ qua, vì đây là thử nghiệm mới, chưa có tiền lệ trong project.

---

## ĐIỀU KIỆN TIÊN QUYẾT

- [ ] Đã có sẵn 5-10 ảnh mẫu RSNA có nhãn dán ở nền (dùng lại đúng ảnh mẫu bạn của bạn đã dùng cho Cấu hình C để dễ so sánh chéo kết quả, ví dụ case 4360, 4364)
- [ ] Môi trường Python có OpenCV, NumPy, scikit-image (`pip install scikit-image` — dùng cho SSIM đánh giá sau này)
- [ ] Đã đọc qua `mask_generator.py` của bạn cùng nhóm để biết cấu trúc code hiện có (tái sử dụng phần Otsu, tránh viết lại từ đầu)

---

## TASK 1 — Segmentation chính xác bàn tay (không dùng Convex Hull)

```
Trong project/segment_composite/scripts/hand_segmentation.py:

1. Viết hàm segment_hand_precise(img_path):
   - Đọc ảnh grayscale, Gaussian blur nhẹ (kernel 5x5) để giảm nhiễu hạt.
   - Otsu threshold để tách bàn tay (sáng) khỏi nền (tối).
   - Morphological closing (kernel 7x7) để lấp lỗ nhỏ do nhiễu, KHÔNG
     dùng Convex Hull ở bước lấy silhouette chính (Convex Hull sẽ lấp
     đầy khoảng trống giữa các ngón tay, làm sai vùng cần ghép nền mới).
   - Tìm contour lớn nhất (giả định là bàn tay), fill thành mask nhị
     phân đầy đủ.
2. Chạy thử trên 5-10 ảnh mẫu, lưu overlay (mask màu xanh chồng lên ảnh
   gốc) ra project/segment_composite/outputs/mask_preview/<id>_mask.png
   để kiểm tra bằng mắt.
3. In cảnh báo nếu diện tích mask quá nhỏ (<15% ảnh) hoặc quá lớn (>60%
   ảnh) — dấu hiệu Otsu threshold bị lỗi trên ảnh đó (do độ tương phản
   khác thường).
4. Dừng lại chờ tôi duyệt.
```

**Điều kiện dừng để duyệt:** Xem `mask_preview` — silhouette phải bám sát đúng viền ngón tay thật (kể cả khoảng trống giữa các ngón), không bị "lấp đầy" như hình Convex Hull.

---

## TASK 2 — Chuẩn bị 2 phương án nền thay thế, so sánh song song

```
Trong project/segment_composite/scripts/background_generator.py:

1. Phương án A — Nền tổng hợp từ thống kê nhiễu:
   - Với mỗi ảnh, lấy mean/std cường độ pixel của vùng nền GỐC (ngoài
     mask bàn tay), sinh nền mới bằng nhiễu Gaussian cùng mean/std, làm
     mượt bằng GaussianBlur(15,15) để giống texture X-quang thật.
2. Phương án B — Nền lấy từ ảnh RSNA sạch khác:
   - Chọn 2-3 ảnh RSNA không có nhãn dán ở vùng nền làm "ngân hàng nền".
   - Trước khi ghép, dùng histogram matching (skimage.exposure.
     match_histograms) để đồng bộ độ sáng/tương phản nền nguồn với ảnh
     đích.
3. Với mỗi ảnh mẫu (5-10 ảnh từ Task 1), sinh cả 2 phương án nền, lưu
   riêng vào project/segment_composite/outputs/bg_option_A/ và
   bg_option_B/ để so sánh trực quan.
4. Dừng lại chờ tôi duyệt, chọn 1 phương án (hoặc giữ cả 2 để thử nghiệm
   tiếp) trước khi sang bước ghép.
```

**Điều kiện dừng để duyệt:** So sánh 2 phương án bằng mắt — phương án nào trông tự nhiên hơn, ít giống "nền phẳng nhân tạo" hơn.

---

## TASK 3 — Ghép ảnh với feathering (tránh lộ vết cắt cứng)

```
Trong project/segment_composite/scripts/composite.py:

1. Viết hàm composite_hand_on_new_bg(orig_img, precise_hand_mask, new_bg):
   - Làm mềm biên mask bằng GaussianBlur (kernel 21x21) để tạo soft_mask
     liên tục 0-1, mô phỏng gradient tán xạ tia X tự nhiên quanh mép bàn
     tay (không dùng biên cứng nhị phân).
   - Ghép: composited = orig * soft_mask + new_bg * (1 - soft_mask)
2. QUAN TRỌNG — nếu vết ghép vẫn lộ rõ sau feathering đơn giản, thử
   thêm Laplacian Pyramid Blending (tái sử dụng đúng code đã thiết kế ở
   Hướng D trước đó — module này áp dụng được cho cả bài toán ghép nền,
   không chỉ ghép vùng inpainted).
3. Chạy trên 5-10 ảnh mẫu (cả 2 phương án nền A và B), lưu kết quả ra
   project/segment_composite/outputs/composited_preview/
4. Xuất ảnh so sánh 4 panel mỗi case: Ảnh gốc | Mask | Nền mới | Kết quả
   ghép cuối, để dễ đánh giá toàn bộ quy trình trong 1 hình.
5. Dừng lại chờ tôi duyệt.
```

**Điều kiện dừng để duyệt:** Đây là bước quan trọng nhất để đánh giá bằng mắt — kiểm tra kỹ xem có còn "đường viền lộ rõ" quanh bàn tay không, đặc biệt ở vùng khe giữa các ngón tay (nơi dễ lộ nhất do mask silhouette phức tạp).

---

## TASK 4 — Đánh giá định lượng, so sánh trực tiếp với Cấu hình C

```
Trong project/segment_composite/scripts/evaluate.py:

1. Tính SSIM giữa ảnh gốc và ảnh đã ghép, CHỈ TÍNH TRONG VÙNG BÀN TAY
   (dùng precise_hand_mask để crop trước khi tính SSIM) — đây phải là
   gần 1.0 tuyệt đối vì vùng xương hoàn toàn không đổi (khác với Cấu
   hình C, nơi ControlNet chỉ "cố gắng" giữ nguyên chứ không đảm bảo
   tuyệt đối).
2. Tính "Bone Diff" giống hệt công thức bạn cùng nhóm đã dùng (sai lệch
   pixel trung bình CHỈ tại vùng xương) để so sánh trực tiếp trên cùng
   thang đo — kỳ vọng bằng 0 hoặc gần 0 tuyệt đối, vì vùng bàn tay được
   giữ y nguyên pixel-for-pixel (không qua bất kỳ mô hình sinh ảnh nào).
3. Kiểm tra định tính: xác nhận KHÔNG có hiện tượng Text Hallucination
   nào ở nền mới (quan sát bằng mắt trên toàn bộ 5-10 ảnh mẫu) — vì bản
   chất phương pháp này không dùng generative model nên về lý thuyết
   không thể xảy ra hiện tượng này, cần xác nhận thực tế đúng như vậy.
4. Xuất bảng so sánh: Cấu hình C (ControlNet) vs Segment-and-Composite,
   theo 3 tiêu chí: Bone Diff, có/không Text Hallucination, thời gian xử
   lý mỗi ảnh (phương pháp này không cần GPU/diffusion model nên kỳ vọng
   nhanh hơn rất nhiều).
5. Dừng lại chờ tôi duyệt.
```

**Điều kiện dừng để duyệt:** Xem bảng so sánh — nếu Bone Diff ≈ 0 và không còn Text Hallucination, đây là bằng chứng thuyết phục cho thấy phương pháp khả thi và có ưu điểm rõ so với Cấu hình C.

---

## TASK 5 — Kiểm tra rủi ro "Shortcut Learning đảo chiều" (bắt buộc, không bỏ qua)

```
Đây là bước kiểm tra rủi ro quan trọng nhất đã lường trước — PHẢI làm
trước khi coi phương pháp này là hướng chính thức.

1. Trong project/segment_composite/scripts/check_shortcut_risk.py:
   - So sánh phân phối thống kê (histogram cường độ, độ lệch chuẩn local
     texture) giữa vùng nền MỚI (đã ghép) và vùng nền THẬT của các ảnh
     RSNA sạch khác (không qua xử lý gì).
   - Nếu vùng nền mới có độ mượt/đồng đều bất thường so với nền thật
     (std thấp hơn đáng kể, thiếu texture nhiễu hạt đặc trưng của phim
     X-quang), CẢNH BÁO rõ ràng — đây là dấu hiệu mô hình downstream có
     thể học được shortcut mới "nền mượt = ảnh đã xử lý" thay vì học đặc
     trưng xương.
2. Nếu phát hiện rủi ro, quay lại Task 2 điều chỉnh Phương án A (tăng
   thêm nhiễu high-frequency mô phỏng hạt phim thật, ví dụ Perlin noise
   thay vì Gaussian noise thuần) hoặc ưu tiên dùng hẳn Phương án B (nền
   từ ảnh RSNA thật).
3. Dừng lại, báo cáo kết luận rõ ràng: phương pháp này có an toàn về mặt
   không tạo shortcut mới hay không.
```

**Điều kiện dừng để duyệt:** Đây là điều kiện bắt buộc để quyết định có tiếp tục mở rộng phương pháp này ra full dataset hay không — không được bỏ qua bước này dù kết quả Task 4 có đẹp đến đâu.

---

## TASK 6 — Xử lý trường hợp dị vật/nhãn nằm ĐÈ LÊN bàn tay (giới hạn đã biết trước)

```
1. Rà soát nhanh trong 200 ảnh test (hoặc một mẫu lớn hơn từ tập train)
   xem có bao nhiêu ảnh có nhãn dán/dị vật kim loại nằm chồng lên chính
   vùng bàn tay (không phải ở nền) — ước tính tỷ lệ phần trăm.
2. Với các ảnh này, phương pháp segment-and-composite KHÔNG xử lý được
   (vì giữ nguyên toàn bộ pixel trong mask bàn tay) — cần báo cáo rõ tỷ
   lệ ảnh bị giới hạn này để biết mức độ ảnh hưởng thực tế.
3. Đề xuất hướng xử lý bổ sung cho nhóm nhỏ này: giữ lại một bước
   inpainting có kiểm soát rất hẹp (tương tự "Selective Label Masking"
   của Cấu hình C) chỉ áp dụng riêng cho các ảnh thuộc nhóm này, không
   áp dụng đại trà.
4. Dừng lại, báo cáo tổng kết toàn bộ thử nghiệm.
```

---

## Bảng tổng hợp Task

| Task | Nội dung | Cỡ mẫu | Cần duyệt? |
|---|---|---|---|
| 1 | Segmentation chính xác (không Convex Hull) | 5-10 ảnh | Có |
| 2 | 2 phương án nền thay thế | 5-10 ảnh | Có |
| 3 | Ghép ảnh + feathering/Laplacian blending | 5-10 ảnh | Có — quan trọng nhất về mặt thị giác |
| 4 | Đánh giá định lượng, so sánh Cấu hình C | 5-10 ảnh | Có |
| 5 | **Kiểm tra rủi ro shortcut learning đảo chiều** | — | **Có — bắt buộc, không bỏ qua** |
| 6 | Rà soát giới hạn (dị vật đè lên bàn tay) | Mẫu lớn hơn | Có |

## Sau khi hoàn tất — bước tiếp theo

Nếu Task 1-5 đều cho kết quả tốt (Bone Diff ≈ 0, không hallucination, không phát hiện rủi ro shortcut mới), phương pháp này đủ điều kiện để:
1. Trình bày như một **hướng đối chứng/thay thế song song** với Cấu hình C của bạn cùng nhóm — điểm khác biệt rõ ràng, không trùng lặp.
2. Mở rộng chạy full-scale trên 200 ảnh test, dùng chính model bone-age đã freeze (`boneage_v1_FROZEN.pth`) để đo MAE/AUC thực tế, so sánh với baseline và với Hướng B đã làm trước đó.
3. Cân nhắc **kết hợp** với Hướng B (Saliency Masking) — dùng segmentation để loại bỏ hoàn toàn generative artifact ở vùng nền, đồng thời dùng Grad-CAM mask để xử lý tinh hơn các trường hợp nhãn/dị vật nằm sát hoặc đè lên vùng xương (Task 6).
