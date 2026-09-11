# Kế hoạch hiện tại

> Cập nhật: 2026-09-11. Mục tiêu: giảm MAE bằng thay đổi phù hợp ConvNeXt, chọn hoàn toàn trên development.

## Thứ tự bắt buộc

### 1. Hoàn tất kiểm tra C1 trên Fold 1–2

C1 dùng cùng raw C3-R2 và cùng split với C0, gồm:

- ConvNeXt-Tiny + global average pooling + final LayerNorm + sex embedding;
- discriminative fine-tuning theo stage;
- learning-rate decay theo độ sâu layer;
- AdamW không weight decay cho bias/norm;
- warmup 2 epoch rồi cosine schedule;
- exponential moving average (EMA) decay 0,999 để chọn checkpoint.

Việc trước mắt: chẩn đoán vì sao epoch 1 có raw MAE 15,7176 nhưng EMA MAE 31,4944. Kiểm tra khởi tạo EMA, thời điểm update, validation path, checkpoint selection và learning rate. Không chấp nhận run nếu EMA không hội tụ về raw hoặc có sai logic.

### 2. So C1 với C0 bằng paired Fold 1–2

C0 pooled MAE hiện là 6,365354. Chỉ promote C1 nếu:

- cả hai fold hợp lệ và ổn định;
- paired MAE giảm nhất quán, không chỉ một fold;
- không đổi data/split/inference giữa candidate và control;
- có prediction CSV để bootstrap paired.

Nếu C1 không đạt, dừng nhánh; không chạy Fold 3–5 và không mở test.

### 3. Chuyển sang C2 nếu C1 thất bại

C2 là ConvNeXt V2 pretrained mạnh hơn với recipe tương ứng, nhưng giữ raw C3-R2, split và evaluation giống C0. Trước full training cần:

1. preflight manifest/hash;
2. forward/backward smoke test;
3. kiểm tra pretrained weights, normalization và classifier head;
4. Fold 1–2 screening;
5. paired report so với C0.

Chỉ ứng viên vượt gate development mới được mở rộng 5 fold.

## Nhánh không ưu tiên

- Bilinear pooling: mới một fold hợp lệ 6,385449, Fold 2 bất ổn; không promote.
- Histogram equalization: bằng chứng OOF hiện tại cho thấy bỏ equalization tốt hơn.
- Chạy lại C3-R2 Z26 trên test: không làm; test đã đủ bằng chứng bất lợi.
- Tuning ensemble/TTA trên test: cấm. Có thể nghiên cứu trên OOF với trọng số/quy tắc khóa trước.

## Tiêu chí dừng và báo cáo

Mỗi candidate cần bảng: fold MAE, pooled MAE, paired delta với C0, CI 95%, stability warnings, config/split hash và quyết định PROMOTE, STOP hoặc REPAIR_AND_REPEAT.

Khi một candidate được promote:

1. khóa config;
2. chạy đủ 5 fold;
3. tạo OOF report;
4. quyết định ensemble trên OOF;
5. chỉ sau đó mới báo cáo test exploratory một lần theo rule đã khóa.
