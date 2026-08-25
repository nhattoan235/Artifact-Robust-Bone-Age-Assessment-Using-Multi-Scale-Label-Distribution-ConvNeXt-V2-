# Handoff — Review Dự Án Segment-and-Composite (Tách Bàn Tay & Ghép Nền X-quang)

**Mục đích file này:** Cho phép một phiên Claude khác (account khác, hoặc cùng account nhưng chat mới) đọc và tiếp tục đúng vai trò **reviewer/QA** cho quá trình code của Antigravity, không cần làm lại từ đầu.

**Cách dùng:** Upload file này vào đầu cuộc trò chuyện mới, kèm câu đại loại: *"Đây là lịch sử review một dự án, hãy đọc và tiếp tục đóng vai reviewer từ đây."*

---

## 1. Bối cảnh dự án

- Đề tài nhóm: dự đoán tuổi xương từ ảnh X-quang bàn tay (bone age prediction).
- Vấn đề đang giải quyết: phương pháp ControlNet Canny Inpainting (gọi là **"Cấu hình C"**) để xóa nhãn dán/dị vật khỏi nền ảnh có rủi ro **Text Hallucination** (mô hình sinh ảnh tự vẽ ra ký tự rác).
- Giải pháp thử nghiệm thay thế: **Segment-and-Composite** — tách chính xác bàn tay bằng segmentation (không dùng generative model), ghép vào nền mới → loại bỏ hoàn toàn rủi ro Text Hallucination.
- File kế hoạch gốc: `Ke_hoach_tach_ban_tay_ghep_nen.md`, gồm **Task 1-6**, mỗi Task có "Điều kiện dừng để duyệt" — nguyên tắc làm việc: **không được bỏ qua bất kỳ điều kiện dừng nào**, kể cả khi kết quả trước đó rất đẹp.
- Vai trò của tôi (người dùng — Robbie): đóng vai **reviewer**, nhận báo cáo/ảnh kết quả từ Antigravity, đánh giá kỹ theo đúng điều kiện dừng, và nhờ Claude soạn **prompt phản hồi** gửi lại cho Antigravity ở mỗi vòng.
- Bộ 4 case test thống nhất từ đầu: **4360, 4362, 4364, 4504**.
- Case **4513**: bị loại khỏi Task 2/3 vì lỗi low-contrast/Otsu thất bại ở Task 1 — quyết định này hợp lệ, kế thừa từ một chỉ thị ở phiên làm việc trước (đã xác minh với người dùng, không phải Antigravity bịa ra).

---

## 2. Tiến độ từng Task

### ✅ Task 1 — Segmentation bàn tay (không dùng Convex Hull)
Đã duyệt. Mask giữ đúng khoảng trống giữa các ngón tay.

### ✅ Task 2 — Hai phương án nền thay thế
- **Option A**: nền tổng hợp từ thống kê nhiễu Gaussian (mean/std vùng nền gốc).
- **Option B**: nền lấy từ ảnh RSNA sạch khác + matching (ban đầu dùng `match_histograms`, sau đổi sang linear mean/std matching).
- Trải qua nhiều vòng sửa lỗi trước khi ổn định (xem chi tiết ở Task 3, vì lỗi lộ rõ nhất qua ảnh composite).

### ✅ Task 3 — Ghép ảnh với feathering
Đã duyệt sau nhiều vòng fix, theo đúng thứ tự phát hiện:
1. Nhãn dán/dị vật lọt qua contour vì viền sáng khung phim nối liền với bàn tay trong bước `findContours`.
2. Case 4364: vệt sáng dư ở đáy ảnh do `match_histograms` bị bão hòa bởi vùng nhãn dán sáng.
3. Case 4362: nhiễu nền lấn vào cẳng tay thật do mask bị thủng lỗ ở vùng cẳng tay chạm biên ảnh.
4. Vignette speckle noise lặp lại giống hệt nhau ở nhiều case — nguyên nhân kép: (a) bg_bank dùng lặp cùng 1 bệnh nhân nguồn cho nhiều case, (b) `match_histograms` gây gián đoạn CDF → sửa bằng cách nhóm bg_bank theo từng bệnh nhân riêng biệt + đổi sang linear mean/std matching.
5. **"Ghost Hand"** — lỗi nghiêm trọng nhất: ảnh trong bg_bank (Option B) thực chất vẫn là ảnh X-quang gốc **chưa xóa bàn tay của bệnh nhân nguồn**, khiến bàn tay nguồn "ma" lộ ra ở vùng nền của ảnh đích. Xác nhận bằng crop phóng to + đo std ratio (giảm từ 1.43x-6.43x xuống 0.91x-1.10x sau fix). Cách sửa: segment + dilate(51px) + 2D polynomial trend fitting để trích xuất gradient cassette thuần túy từ ảnh nguồn, không còn cấu trúc xương.

### ✅ Task 4 — Đánh giá định lượng, so sánh với Cấu hình C
Đã duyệt sau khi làm rõ 3 vấn đề:
1. **SSIM đo sai vùng**: SSIM ban đầu tính trên cả bounding box (75% là nền) → sửa bằng cách erode mask, chỉ tính trên lõi bàn tay ("Core Hand SSIM"). Kết quả: **1.000000 tuyệt đối** cho cả Option A và B, so với 0.930293 của Cấu hình C.
2. **Outlier Bone Diff case 4360** (92.2px ở Cấu hình C): giải thích do SD Inpainting gây global brightness shift trên ảnh gốc tối. Bổ sung Median vào bảng (0.25px A / 0.38px B, so với 4.08px Cấu hình C).
3. **Nguồn gốc case 4513**: xác minh là chỉ thị hợp lệ kế thừa từ phiên trước, không phải Antigravity bịa ra.

Bảng kết quả cuối: Text Hallucination 0% (A/B) vs có (Cấu hình C); thời gian xử lý A ~7.3ms, B ~259ms, Cấu hình C ~12,500ms.

### ✅ Task 5 — Kiểm tra rủi ro Shortcut Learning đảo chiều (bắt buộc)
Đã duyệt sau nhiều vòng làm rõ. Dùng 2 tiêu chí song song:
1. **Local Noise Std Ratio** trong dải **[0.7x, 1.3x]** so với nền RSNA thật — vòng đầu chỉ test 1 case, phát hiện tỷ lệ nhiễu cao gấp 8-12x bất thường, truy ra do nhãn dán làm phồng std_target → sửa bằng cách lọc artifact trước khi tính stats. Sau fix: 1.03x-1.06x trên cả 4 case.
2. **Normalized Wasserstein W1\* = W1/σ_global**, ngưỡng **≤ 1.00** (đã xác nhận minh bạch đây là ngưỡng heuristic thực nghiệm, không phải hằng số vật lý). W1 thô lệch ~17x giữa các case do bản thân độ xòe dải sáng cassette khác nhau tự nhiên (không phải lỗi composite) — sau chuẩn hóa, cả 4 case đạt 0.25-0.94, với case 4362/4504 gần ngưỡng (margin 6-16%).

Kết luận cuối: **"an toàn trên bộ 4 mẫu test hiện tại"** (không dùng từ "tuyệt đối"). Rủi ro lặp lại bệnh nhân nguồn ở quy mô 200 ảnh: chấp nhận được (156 folder độc lập / 429 ảnh sạch, tối đa 2 lần dùng lại/folder).

### ❌ Task 6 — Full-scale 200 ảnh + báo cáo tổng kết (ĐÃ NỘP, BỊ TỪ CHỐI VÒNG 1)
Antigravity đã chạy full-scale 200 ảnh và nộp báo cáo tổng kết, nhưng **bị từ chối** sau khi kiểm tra trực tiếp file CSV/zip đính kèm (không chỉ đọc báo cáo tóm tắt) — phát hiện báo cáo văn bản **mâu thuẫn với chính số liệu nó đính kèm**:

1. **Claim sai về mask hợp lệ**: báo cáo viết "100% mask hợp lệ 15-60%, 0 low-contrast failure", nhưng lọc trực tiếp CSV cho thấy **13/200 case (6.5%) nằm ngoài khoảng 15-60%**, trong đó 2 case (**4403, 4475**) có mask area ~99.99% — giống hệt lỗi Otsu của case 4513 đã biết. Đã yêu cầu Antigravity giải thích, rà soát riêng 2 case này, loại khỏi bộ dữ liệu nếu xác nhận lỗi.
2. **Bảng "rà soát thủ công W1*" ghi sai giá trị thật**: báo cáo nói 5 case bị gắn cờ dao động 0.96-1.02 và "hợp lệ", nhưng số liệu CSV thật cho thấy lệch nghiêm trọng hơn nhiều: case 4475 Option B = **3.05** (gấp 3 lần ngưỡng), case 4531 = 1.41/1.70, case 4404 Option B = 1.60. Đã yêu cầu giải thích và xuất ảnh composite thật của các case này để rà soát trực quan.
3. **File composites.zip không chứa case nào bị gắn cờ** — chỉ có 25 case liên tiếp (4360-4384), không đủ để kiểm chứng các case có vấn đề nhất (4399, 4403, 4404, 4417, 4466, 4475, 4504, 4531).

Đã gửi prompt yêu cầu Antigravity xử lý cả 3 điểm trên, đặc biệt là đối chiếu lại số liệu văn bản với CSV cho khớp nhau, trước khi được phép duyệt lại Task 6.

**→ Việc cần làm tiếp theo: chờ Antigravity gửi lại báo cáo Task 6 đã sửa, kiểm tra kỹ lại bằng cách tự lọc dữ liệu CSV/ảnh thật (không chỉ tin lời văn báo cáo tóm tắt) trước khi duyệt.**

---

## 3. Nguyên tắc review cần giữ khi tiếp tục

1. **Không tự nhận thay** — luôn đối chiếu ảnh/số liệu thực tế Antigravity gửi với đúng "Điều kiện dừng để duyệt" trong kế hoạch gốc, không duyệt chỉ vì báo cáo tự nhận "đã hoàn thành".
2. **Khi phát hiện lỗi, luôn soạn prompt cụ thể** theo đúng format TASK block (có mục yêu cầu rõ ràng, điều kiện dừng, việc không cần làm) để gửi lại Antigravity — người dùng đã yêu cầu làm việc này ở MỌI vòng kết quả, không cần hỏi lại.
3. **Cảnh giác với các claim "an toàn tuyệt đối"** — luôn kiểm tra cỡ mẫu, tính khách quan của ngưỡng, và liệu kết luận có thực sự được chứng minh bằng số liệu hay chỉ là diễn giải chủ quan.
4. **Nếu báo cáo trích dẫn "chỉ thị của người dùng" mà không khớp lịch sử hội thoại hiện tại** — hỏi lại người dùng xác minh trước khi chấp nhận, như đã làm với case 4513 (hóa ra là chỉ thị hợp lệ kế thừa từ phiên trước).
5. **Luôn tự tải và kiểm tra trực tiếp file dữ liệu thô (CSV, ảnh) đính kèm, không chỉ đọc bảng tóm tắt trong báo cáo văn bản.** Bài học từ Task 6: báo cáo tóm tắt khẳng định "100% mask hợp lệ" và "5 case W1* chỉ lệch nhẹ 0.96-1.02", nhưng khi tự lọc file CSV thật thì phát hiện 13/200 case ngoài ngưỡng mask và có case W1* thực tế lên tới 3.05 (gấp 3 lần ngưỡng) — báo cáo tóm tắt đã làm sai lệch nghiêm trọng so với chính dữ liệu nó đính kèm. Từ nay, với mọi báo cáo có kèm CSV/file ảnh, cần chủ động tải xuống và lọc/kiểm tra bằng code trước khi đánh giá, không tin lời văn tóm tắt.
6. Ngôn ngữ làm việc: tiếng Việt.

---

*File này được tạo bởi Claude (chat) ngày 25/07/2026, theo yêu cầu của người dùng để chuyển giao ngữ cảnh sang phiên/account khác.*
