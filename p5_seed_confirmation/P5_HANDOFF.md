# Bàn giao P5 – Xác nhận nhiều seed

**Trạng thái:** Hoàn thành; chọn D0 ConvNeXt-Tiny direct regression cho P6.

## Giao thức

- So sánh D0 và D3 trên cùng official validation 1.425 ảnh với các seed `17`, `42`, `123`.
- Primary D3: fused prediction `0,5 regression + 0,5 distribution expectation`.
- Secondary adaptive: regression-only từ cùng checkpoint D3.
- Delta được định nghĩa là `MAE_D3 - MAE_D0`; số âm có lợi cho D3.
- Giữ D3 chỉ khi cải thiện trung bình ít nhất `0,10` tháng, có lợi ở ít nhất 2/3 seed và không có tín hiệu bất ổn rõ rệt từ bootstrap/nhóm con.
- Test set chưa được sử dụng.

## Kết quả theo seed

| Seed | D0 MAE | D3 fused MAE | Delta fused | D3 regression-only | Delta regression |
|---:|---:|---:|---:|---:|---:|
| 17 | 6,31950 | 6,24066 | -0,07884 | 6,29524 | -0,02426 |
| 42 | 6,18479 | 6,14541 | -0,03938 | 6,12496 | -0,05984 |
| 123 | 6,28259 | 6,28917 | +0,00659 | 6,30873 | +0,02614 |

## Tổng hợp ba seed

| Endpoint | MAE trung bình | SD | Delta trung bình với D0 | Bootstrap 95% CI | Seed có lợi |
|---|---:|---:|---:|---:|---:|
| D0 | 6,26229 | 0,06961 | 0 | — | — |
| D3 fused | 6,22508 | 0,07313 | -0,03721 | [-0,13521; +0,05863] | 2/3 |
| D3 regression-only adaptive | 6,24297 | 0,10243 | -0,01932 | [-0,11270; +0,07069] | 2/3 |

## Nhóm con

Delta trung bình D3 fused so với D0:

- Nữ: `-0,01260`; nam: `-0,05797` tháng.
- Tuổi 0–59: `-0,34468`; 60–119: `-0,09758`; 120–179: `-0,01289`; 180–228: `+0,22067` tháng.

D3 fused cải thiện nhất quán ở nhóm 0–59 nhưng xấu hơn ở cả ba seed trong nhóm 180–228. Đây là tín hiệu đổi lợi ích giữa nhóm tuổi, không phù hợp để thay D0 làm mô hình chính.

## Quyết định

D3 không đạt ngưỡng cải thiện thực tiễn `0,10` tháng; cả hai CI đều chứa 0. Regression-only adaptive cũng không đạt. Theo quy tắc khóa trước, chọn mô hình đơn giản hơn: **D0 ConvNeXt-Tiny + sex embedding + direct regression**.

Không tuyên bố LDL không có tác dụng nói chung. Kết luận chỉ giới hạn ở cấu hình sigma 2, trọng số loss 0,2, fused 0,5/0,5 và recipe hiện tại trên RSNA validation.

## Bước tiếp theo

P6 chỉ so sánh D0 ở input 512 với input 768, seed 42, giữ nguyên các trường khoa học còn lại. Chỉ khi 768 cải thiện đủ rõ mới cân nhắc độ phân giải cao hơn; chưa mở test set.

Artifact máy đọc: `p5_seed_confirmation/P5_AGGREGATE.json`.
