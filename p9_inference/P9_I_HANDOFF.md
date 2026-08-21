# P9-I Handoff — TTA và bias correction trên P7 OOF

## Kết luận

Trên 14.036 mẫu OOF của P7, Deeplasia-style TTA là thành phần có lợi ích rõ ràng
trong pipeline hiện tại. Linear bias correction làm bias trung bình gần 0 nhưng
không cải thiện MAE; TTA + bias correction cũng kém hơn TTA đơn độc.

Quyết định:

- Giữ TTA làm ứng viên inference chính cho các bước tiếp theo.
- Không giữ raw bias correction đơn độc.
- Không giữ TTA + bias correction trong pipeline chính hiện tại.
- Chưa mở nhãn test và chưa dùng test để chọn quyết định nào.

## Protocol

- Năm best checkpoint P7 final, ConvNeXt-Tiny 512 + sex embedding + A2.
- Toàn bộ development OOF 14.036 mẫu, ID duy nhất.
- Raw: xoay 0°, không flip.
- TTA: xoay `[-10, -5, 0, 5, 10]`°, có và không flip ngang; trung bình 10 dự đoán.
- Bias correction: hồi quy signed error theo prediction, fit cross-fitted; correction
  cho fold `f` chỉ fit trên bốn fold còn lại.
- Không có đường dẫn hoặc nhãn RSNA test trong tool P9-I.

## Kết quả chính

| Endpoint | MAE (tháng) | RMSE | Median AE | Bias pred−true |
|---|---:|---:|---:|---:|
| P7 OOF reference | 6,316691 | 8,519420 | 4,875 | −0,3899 |
| Raw tái suy luận | 6,317471 | 8,519945 | 4,8529 | −0,3922 |
| TTA | **6,210446** | **8,382443** | **4,7477** | −0,0930 |
| Raw + bias correction | 6,324285 | 8,525744 | 4,8521 | ≈0 |
| TTA + bias correction | 6,228665 | 8,396016 | 4,7610 | ≈0 |

Paired bootstrap 95% CI của delta MAE so với raw tái suy luận:

| So sánh | Delta MAE (tháng) | Paired bootstrap 95% CI |
|---|---:|---:|
| TTA − raw | **−0,107025** | **[−0,135977; −0,078220]** |
| Raw + correction − raw | +0,006815 | [−0,001070; +0,014648] |
| TTA + correction − raw | **−0,088806** | **[−0,118558; −0,059390]** |
| TTA + correction − TTA | +0,018219 | [+0,012482; +0,023813] |

TTA cải thiện thêm so với P7 OOF reference khoảng 0,106245 tháng về MAE. Đây
là kết quả phát triển trên OOF, chưa phải kết quả test xác nhận.

## Subgroup và bias

So với raw tái suy luận, TTA giảm MAE ở cả hai giới:

- Female: 6,536692 → 6,458097 tháng.
- Male: 6,132144 → 6,001086 tháng.

Theo age-bin, TTA cải thiện rõ nhất ở nhóm 0–59 tháng (6,065081 → 5,895303)
và 120–179 tháng (5,912101 → 5,758611). Nhóm 60–119 gần như không đổi
(7,316426 → 7,296841); nhóm 180–228 thay đổi rất nhỏ.

TTA giảm bias trung bình từ khoảng −0,3922 xuống −0,0930 tháng, nhưng bias
theo age-bin chưa biến mất; nhóm 180–228 vẫn có bias âm khoảng −3,0 tháng.

Độ bất đồng giữa các biến thể TTA: mean SD 1,709 tháng, median SD 1,559 tháng,
P95 SD 3,252 tháng. Đây mới là tín hiệu disagreement, chưa phải uncertainty
lâm sàng đã được hiệu chuẩn.

## Kiểm tra tái lập raw

So với CSV OOF gốc, raw tái suy luận có mean absolute difference 0,02795 tháng,
median 0,02393 tháng, maximum 0,36450 tháng; 13.956/14.036 mẫu sai khác không
quá 0,1 tháng. Khi so sánh ablation, dùng `raw_recomputed` làm control trực tiếp.

## Ý nghĩa và bước tiếp theo

P9-I cho thấy khoảng cách đến Deeplasia có một phần đến từ inference: chỉ riêng
TTA đã giảm khoảng 0,107 tháng MAE trên OOF. Tuy nhiên, đây chưa phải
reproduction đầy đủ Deeplasia vì preprocessing/mask, EfficientNet và ensemble
dị thể vẫn chưa được thay đổi.

1. Đóng băng TTA như một thành phần inference ứng viên.
2. Không train lại ConvNeXt chỉ để thử thêm bias correction.
3. Chuyển sang P9-B: Deeplasia single-model EfficientNet-B0 ở 512.
4. Giữ raw và TTA song song để tách đóng góp backbone khỏi inference.
5. Chỉ sau khi single-model ổn định mới làm ensemble dị thể.

## Artifact

- [Script P9-I](tta_bias_oof.py)
- [Protocol P9-I](P9_I_PROTOCOL.md)
- [Predictions P9-I](outputs/P9_I_TTA_BIAS_OOF/P9_I_TTA_BIAS_OOF_predictions.csv)
- [Report P9-I](outputs/P9_I_TTA_BIAS_OOF/P9_I_TTA_BIAS_OOF_report.json)
