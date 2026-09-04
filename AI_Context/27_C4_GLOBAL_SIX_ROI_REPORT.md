# Báo cáo C4 — Global + sáu ROI giải phẫu

> Cập nhật: 2026-09-03
> Trạng thái: đã chạy đủ 5-fold OOF raw; C4-TTA chưa chạy.
> Endpoint chính: OOF development, không dùng nhãn test để chọn mô hình.

## 1. Câu hỏi nghiên cứu

C4 kiểm tra liệu một mô hình nhìn đồng thời:

1. ảnh toàn bàn tay (`global`);
2. một ROI carpal/wrist;
3. năm ROI vùng MCP/ngón;

có tốt hơn mô hình global hoặc ROI toàn bàn tay hay không.

C4 là nhánh multi-view giải phẫu. Nó khác C3-ROI: C3 dùng một crop ROI toàn bàn tay, còn C4 dùng sáu crop cục bộ cộng với ảnh global.

## 2. Cấu hình hệ thống

```text
Input mỗi mẫu: 1 global + 6 ROI = 7 view
Backbone: một ConvNeXt-Tiny pretrained dùng chung cho cả 7 view
Feature fusion: nối feature theo thứ tự view đã khóa
Metadata: sex embedding
Head: regression head dự đoán tuổi theo tháng
Split: 5-fold đã khóa
```

Các ROI không được xem là bảy mẫu độc lập. Chúng giữ chung `image_id`, target, sex và fold với ảnh gốc.

## 3. Tạo ROI và chống leakage

- Vị trí ROI được tạo từ hình học/mask, không dùng target, prediction hoặc sex để chọn vị trí.
- Việc tạo cache được thực hiện sau khi split nguồn đã được khóa.
- Mỗi mẫu có manifest và các file view tương ứng.
- C4 không được đánh giá bằng cách nhân số view thành số mẫu.

## 4. Kết quả OOF hiện tại

Artifact: [`C4_MULTI_ROI_V2_OOF_report.json`](../c4_multi_roi/outputs/C4_MULTI_ROI_V2_OOF_LOCAL/C4_MULTI_ROI_V2_OOF_report.json)

| Chỉ số | Kết quả |
|---|---:|
| Số mẫu OOF | 14.024 |
| Số fold | 5 |
| MAE | **6,63931 tháng** |
| RMSE | 8,81567 tháng |
| Test dùng để chọn | Không |

C4 raw hiện chưa tốt hơn E1/C3 trong các run đã có. Tuy nhiên, kết quả này chưa phải phép so sánh hoàn toàn đồng nhất với C3 vì C4 dùng 14.024 mẫu, trong khi C3/E1 cũ dùng 14.036 mẫu.

## 5. Đối chiếu với các nhánh liên quan

| Nhánh | Input | OOF MAE |
|---|---|---:|
| E1/P7 | Global | 6,31669 |
| C3-ROI raw | Một ROI toàn bàn tay | 6,43735 |
| C3-ROI-TTA | Một ROI toàn bàn tay + TTA | 6,32558 |
| C4 raw | Global + 6 ROI | **6,63931** |

Bảng này chỉ là đối chiếu hiện trạng. Không được diễn giải ngay rằng ROI từng điểm kém ROI toàn bàn tay vì:

- C4 hiện tại là `global + 6 ROI`, chưa phải `six_roi_only`;
- C4 chưa chạy TTA;
- manifest C4 có 14.024 mẫu, không trùng hoàn toàn với C3/E1;
- C4 dùng shared-backbone multi-view và cơ chế fusion khác C3.

## 6. C4 có điểm mạnh gì?

- Giữ được thông tin toàn cục và chi tiết cục bộ trong cùng một mẫu.
- Có khả năng học các vùng xương liên quan riêng biệt.
- Cấu trúc view rõ ràng, dễ audit thứ tự và nguồn của từng ROI.
- Phù hợp để nghiên cứu multi-view và phân tích đóng góp từng vùng.

## 7. Vì sao C4 chưa thắng trong run hiện tại?

Chưa thể quy toàn bộ cho chất lượng ROI. Các khả năng cần phân biệt:

1. sáu ROI cục bộ làm mất ngữ cảnh và làm feature fusion khó hơn;
2. shared backbone nhận bảy view nhưng chưa có cơ chế attention/fusion thích nghi;
3. số mẫu C4 khác manifest E1/C3;
4. C4 mới được đánh giá raw, trong khi C3 đã có TTA;
5. một số ROI có thể chứa fallback hoặc crop chưa tối ưu.

Vì vậy, kết luận hiện tại chỉ nên là: **C4 raw chưa cải thiện trong cấu hình và protocol đang thử**, không phải sáu ROI chắc chắn kém về bản chất.

## 8. Có nên chạy C4-TTA không?

Có thể chạy như một ablation. TTA phải biến đổi đồng bộ cả bảy view:

```text
rotation ∈ {-10°, -5°, 0°, 5°, 10°}
× flip/no-flip
```

Không được xoay/flip từng ROI độc lập theo các góc khác nhau vì sẽ phá vỡ tính nhất quán của một mẫu global–local.

C4-TTA cần được đánh giá trên OOF trước. Chỉ giữ nếu cải thiện rõ so với C4 raw và không làm xấu các nhóm tuổi/giới. Không được dùng test để chọn TTA hoặc trọng số ensemble.

## 9. Nếu làm lại sau khi loại 12 ảnh

Nếu 12 ảnh bị loại do tiêu chí chất lượng khách quan, quy trình đúng là:

1. khóa manifest mới còn 14.024 ảnh;
2. loại 12 ảnh trước khi chia fold;
3. tạo lại fold cho E1, C3 và C4 bằng cùng ID;
4. train/evaluate lại các nhánh cần so sánh;
5. so sánh raw với raw và TTA với TTA trên cùng ID;
6. chỉ sau khi khóa OOF mới đánh giá test một lần.

Không được loại ảnh chỉ vì model dự đoán sai hoặc làm MAE tăng.

## 10. Kết luận

C4 là một hướng multi-view có ý nghĩa về mặt nghiên cứu nhưng kết quả raw hiện tại chưa vượt baseline. C4 vẫn nên được giữ trong báo cáo như một nhánh thử nghiệm có kiểm soát, kèm kết quả âm tính và giới hạn protocol.

Mô hình C4 hiện tại không được dùng để thay thế E1-TTA + C3-ROI-TTA. Nếu tiếp tục C4, bước hợp lý nhất là khóa lại manifest 14.024 ảnh, chạy C4-TTA đồng bộ bảy view trên OOF, rồi mới quyết định có đưa C4 vào ensemble hay không.
