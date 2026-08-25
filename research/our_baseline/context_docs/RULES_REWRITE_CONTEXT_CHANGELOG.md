# Bộ quy tắc tái cấu trúc AI Context & Changelog

Đưa nguyên văn tài liệu này cho AI (kèm 2 file gốc `AI_CONTEXT.md` và `CHANGELOG.md`) và yêu cầu: *"Viết lại theo đúng các quy tắc dưới đây, giữ nguyên toàn bộ thông tin nghiệp vụ, không được tự suy diễn hay bỏ sót chi tiết kỹ thuật nào."*

---

## 0. Mục tiêu tái cấu trúc

- Giảm token phải nạp mỗi lần AI Agent làm việc, bằng cách chỉ nạp đúng phần liên quan đến task.
- Loại bỏ trùng lặp thông tin giữa các file.
- Tách rõ "quy tắc bền vững" khỏi "trạng thái tạm thời/tiến độ".
- Giữ khả năng mở rộng khi dự án lớn thêm nhiều giai đoạn mà không phải viết lại từ đầu.

---

## 1. Nguyên tắc phân vai giữa các loại file (KHÔNG được trộn lẫn)

| Loại file | Nội dung | Được phép chứa | KHÔNG được chứa |
|---|---|---|---|
| **Core context** (`00_core.md`) | Nguyên tắc nghiệp vụ + kỹ thuật bất biến, áp dụng toàn dự án | Business invariants (Ledger, soft delete, currency...), kiến trúc tổng thể | Tiến độ, số liệu test, trạng thái theo ngày |
| **Rule file theo domain** (`0X_rules_<domain>.md`) | Quy tắc bắt buộc khi làm việc trong 1 domain/feature cụ thể | Chỉ dẫn code, pattern bắt buộc, lý do (rationale) | Nội dung của domain khác, tường thuật quá trình |
| **Trạng thái hiện tại** (`STATUS.md` hoặc mục riêng trong core) | Snapshot "đang ở đâu" tại thời điểm hiện tại | Giai đoạn nào đã xong/đang làm, con số hiện tại (số test, % hoàn thành) | Diễn biến từng bước đã làm thế nào |
| **Changelog** (`CHANGELOG.md`) | Lịch sử thay đổi, append-only | Mỗi entry: cái gì thay đổi + ảnh hưởng, có ngày/version | Chi tiết debug, thiết bị test, log kỹ thuật từng bước |
| **Báo cáo kỹ thuật / spike report** (`docs/spikes/*.md`, `GATE_REPORT.md`) | Chi tiết điều tra, test case, bằng chứng | Toàn bộ tường thuật quá trình, số liệu test chi tiết, thiết bị, lỗi gặp phải | — (đây là nơi chứa chi tiết, không giới hạn) |

**Quy tắc vàng**: Một thông tin chỉ được sống ở **đúng một file**. Nếu cần nhắc ở nơi khác, dùng liên kết/tham chiếu (`Xem: docs/spikes/phase7_gate_report.md`), không copy lại nội dung.

---

## 2. Cấu trúc thư mục đề xuất

```
docs/ai-context/
  00_core.md                 # Luôn nạp — nguyên tắc bất biến toàn dự án
  01_status.md                # Luôn nạp — trạng thái hiện tại, ngắn gọn, cập nhật mỗi lần
  02_db_schema.md              # Nạp khi đụng bảng/DAO/migration
  03_folder_map.md              # Nạp khi cần định vị file
  04_rules_codegen_state.md      # Nạp khi sửa Drift/Riverpod (5.1, 5.2 cũ)
  05_rules_performance.md         # Nạp khi làm reports/dashboard/list/query
  06_rules_ai_assistant.md         # Nạp khi làm feature ai_assistant / STT-TTS
  07_rules_multiuser_rbac_sync.md   # Nạp khi làm giai đoạn 7 (auth/RBAC/sync)
CHANGELOG.md                        # Lịch sử ngắn gọn, append-only
docs/spikes/
  phase7_gate_report.md              # Chi tiết kỹ thuật spike, không giới hạn độ dài
```

---

## 3. Quy tắc viết Core Context (`00_core.md`)

- Chỉ chứa thứ **hiếm khi thay đổi**: mục tiêu sản phẩm, triết lý thiết kế, nguyên tắc kỹ thuật cốt lõi (Single Source of Truth, Ledger Pattern, Soft Delete, kiểu dữ liệu tiền tệ...).
- Không vượt quá ~100-150 dòng. Nếu vượt, tách bớt sang rule file domain.
- Mỗi nguyên tắc: **1 câu định nghĩa + lý do tồn tại** (nếu không hiển nhiên).

## 4. Quy tắc viết Status (`01_status.md`)

- Là **ảnh chụp hiện tại**, không phải nhật ký. Mỗi lần cập nhật là **ghi đè**, không cộng dồn.
- Cấu trúc chuẩn mỗi giai đoạn:
  ```
  ## Giai đoạn X — <tên>
  Trạng thái: <Hoàn tất / Đang làm — %, mốc cụ thể / Chưa bắt đầu>
  Đang chặn bởi: <nếu có>
  Chi tiết đầy đủ: <link tới spike report/changelog nếu cần>
  ```
- KHÔNG liệt kê từng việc đã làm theo trình tự thời gian — đó là việc của changelog.
- Số liệu dễ lỗi thời (số test đạt, % hoàn thành) chỉ nên xuất hiện **duy nhất ở đây**, không lặp ở rule file khác.

## 5. Quy tắc viết Rule file theo domain

- Đặt tên rõ domain, gắn được với glob pattern nếu công cụ AI hỗ trợ (vd: `lib/features/ai_assistant/**`).
- Format bắt buộc:
  - Heading H2/H3 phân đoạn theo chủ đề nhỏ.
  - Bullet ngắn, mỗi bullet 1 ý — không nhồi 3-4 yêu cầu vào 1 câu dài.
  - Có ví dụ code "Preferred/Avoid" khi là rule kỹ thuật cụ thể.
  - Nếu rule có lý do không hiển nhiên, thêm 1 dòng "Lý do:" ngay dưới.
- Tách quy tắc **bắt buộc/bất biến** ra khỏi quy tắc **có điều kiện/đang thử nghiệm** (ví dụ: "PowerSync có điều kiện, chưa chốt" phải ghi rõ trạng thái *provisional*, không viết chung giọng với rule đã chốt).

## 6. Quy tắc viết Changelog

- Mỗi entry: **tối đa 2-3 dòng** — cái gì thay đổi, tác động gì tới hệ thống/API/schema.
- KHÔNG ghi: thiết bị test cụ thể, số liệu debug từng bước, log lỗi HTTP, quá trình thử-sai.
- Nếu một hạng mục cần nhiều chi tiết (spike, migration phức tạp), changelog chỉ ghi 1 dòng tóm tắt + link sang spike report riêng.
- Định kỳ (khi file vượt ~300-400 dòng): archive phần cũ sang `CHANGELOG_ARCHIVE_<mốc thời gian>.md`, chỉ giữ lại các bản gần nhất trong file chính.

## 7. Quy tắc format & metadata áp dụng cho MỌI file

- Đầu mỗi file, thêm khối metadata:
  ```yaml
  ---
  last_updated: YYYY-MM-DD
  scope: <core | db | performance | ai_assistant | rbac_sync | changelog>
  ---
  ```
- Dùng bảng (table) khi liệt kê dữ liệu có nhiều thuộc tính song song (vd. danh sách bảng DB) — đã làm tốt, giữ nguyên.
- Dùng cây thư mục có chú thích mục đích từng file — đã làm tốt, giữ nguyên, chỉ tách thành file riêng (`03_folder_map.md`).
- Không dùng đoạn văn dài liên tục cho rule — luôn bullet hóa.

## 8. Quy tắc loại bỏ trùng lặp (bước bắt buộc khi viết lại)

- Trước khi viết file mới, đối chiếu nội dung giữa `AI_CONTEXT.md` và `CHANGELOG.md` gốc: đoạn nào xuất hiện ở cả hai (dù diễn đạt khác nhau) → chỉ giữ lại ở **đúng một nơi** theo bảng phân vai ở mục 1.
- Ưu tiên giữ bản tóm tắt ở Status/Core, giữ bản chi tiết ở Changelog/Spike report — không giữ cả hai bản chi tiết.

## 9. Checklist tự kiểm tra sau khi viết lại

- [ ] Không file rule nào vượt 150 dòng (trừ changelog/spike report).
- [ ] Không có đoạn nội dung nào lặp lại ở 2 file trở lên.
- [ ] Mọi số liệu tiến độ (test count, %...) chỉ xuất hiện ở `01_status.md`.
- [ ] Mọi file có metadata `last_updated`.
- [ ] Rule bất biến và trạng thái tạm thời không nằm chung một khối/bullet.
- [ ] Có thể trả lời "nếu tôi chỉ sửa feature X, tôi cần đọc file nào?" bằng 1-2 file, không phải toàn bộ context.
