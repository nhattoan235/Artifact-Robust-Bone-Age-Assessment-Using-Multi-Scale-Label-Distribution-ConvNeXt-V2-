# P12 – TTA disagreement và sai số OOF handoff

> **Ngày hoàn tất:** 2026-08-22  
> **Trạng thái:** PASS phân tích; H4 association được ủng hộ nhưng utility hạn chế  
> **Phạm vi:** 14.036 prediction OOF P7/P9-I  
> **Test policy:** `test_accessed=false`

## 1. Protocol và integrity

- Protocol được khóa tại `AI_Context/09_P12_UNCERTAINTY_PROTOCOL.md` trước khi
  đọc correlation.
- Primary: Spearman giữa TTA disagreement và TTA absolute error; rank-score
  bootstrap 5.000 lần, seed 2026.
- Đủ 14.036 dòng và 14.036 image ID duy nhất; fold 1–5 đúng 2.808/2.807×4;
  nữ 6.430, nam 7.606.
- TTA mean/std được recompute từ 10 views, max difference lần lượt
  `5,68e-14` và `2,20e-14` tháng.
- Compile và unit test **6/6 PASS**.

## 2. Primary endpoint

- Spearman rho: **0,2004**.
- Bootstrap 95% CI: **[0,1842; 0,2164]**.
- p-value: `3,76e-127`.
- H4 association: **SUPPORTED** vì lower CI >0.
- Theo mức diễn giải đã khóa, effect chỉ **yếu**, không phải uncertainty lâm
  sàng đã hiệu chuẩn.

## 3. Khả năng nhận biết lỗi lớn

- AUROC cho absolute error >12 tháng: **0,6258**.
- AUROC cho absolute error >18 tháng: **0,6341**.
- Cả hai chỉ ở mức hạn chế, dưới ngưỡng tín hiệu tiềm năng 0,70 đã khóa.

| Disagreement quartile | TTA MAE | Error >12 | Error >18 |
|---|---:|---:|---:|
| Q1 thấp | 4,7587 | 7,04% | 2,08% |
| Q2 | 5,9428 | 11,54% | 3,36% |
| Q3 | 6,5165 | 13,54% | 4,42% |
| Q4 cao | 7,6238 | 20,15% | 6,84% |

Q4 so với Q1 có MAE gấp 1,60 lần, error >12 gấp 2,86 lần và error >18 gấp
3,29 lần. Disagreement có giá trị phân tầng rủi ro ở mức quần thể dù chưa đủ
chính xác cho quyết định từng ca.

## 4. Sex × age heterogeneity

- Nữ: rho 0,1462, CI [0,1224; 0,1700], AUROC >12 là 0,5888.
- Nam: rho 0,2386, CI [0,2166; 0,2599], AUROC >12 là 0,6576.
- Mạnh nhất: F 180–228, rho 0,3564, CI [0,2602; 0,4449], nhưng n=365.
- Không có association: F 0–59, rho −0,0174, CI [−0,1136; 0,0784].
- Worst error group: M 60–119, TTA MAE 7,9313; disagreement rho chỉ 0,0870
  và AUROC >12 chỉ 0,5730.

Kết luận: độ mạnh association phụ thuộc nhóm; một uncertainty proxy chung
không hoạt động đồng đều giữa giới tính và giai đoạn trưởng thành.

## 5. TTA gain

- Overall raw−TTA MAE gain: **+0,1070 tháng**, CI [+0,0775; +0,1361].
- Nữ: +0,0786, CI [+0,0354; +0,1227].
- Nam: +0,1311, CI [+0,0928; +0,1702].
- TTA gain có ý nghĩa ở F 120–179, M 0–59, M 60–119 và M 120–179.
- TTA không có lợi ích chắc chắn ở F 0–59, F 60–119, F 180–228 và M 180–228.

Age-standardized TTA MAE là 6,4283 ở nữ và 6,1678 ở nam; gap khoảng 0,2605
tháng, nhỏ hơn gap quan sát chưa chuẩn hóa 0,4570 tháng. Phân bố tuổi giải
thích một phần, nhưng không toàn bộ, chênh lệch theo giới.

## 6. Risk–coverage mô tả

| Coverage giữ lại | Số ca | TTA MAE | Error >12 |
|---:|---:|---:|---:|
| 100% | 14.036 | 6,2104 | 13,07% |
| 90% | 12.632 | 5,9981 | 11,92% |
| 80% | 11.228 | 5,8139 | 11,05% |
| 70% | 9.825 | 5,6732 | 10,38% |
| 50% | 7.018 | 5,3508 | 9,29% |

Đây chỉ là mô tả trên cùng OOF. Không dùng các cutoff disagreement trong bảng
làm threshold triển khai; cần external validation và thiết kế selective
prediction riêng.

## 7. Kết luận khoa học và quyết định

> TTA disagreement liên hệ dương nhưng yếu với absolute error trên OOF. Nó làm
> giàu các ca sai số lớn ở quartile cao, song hiệu năng nhận biết lỗi và độ ổn
> định subgroup chưa đủ để xem là uncertainty lâm sàng hoặc cơ chế tự động từ
> chối dự đoán.

- Giữ TTA vì cải thiện MAE paired đã được xác nhận.
- Giữ disagreement như biến phân tích/triage nghiên cứu, không là output lâm sàng.
- Không tối ưu threshold trên OOF và không dùng RSNA test đã chạm.
- Bước nghiên cứu tiếp theo hợp lý: xác nhận association/risk–coverage trên một
  external holdout chưa chạm, hoặc nghiên cứu uncertainty được hiệu chuẩn.

## 8. Artifact

- `p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/report.json`
- `p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/group_metrics.csv`
- `p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/disagreement_quartiles.csv`
- `p12_uncertainty/outputs/P12_OOF_UNCERTAINTY/risk_coverage.csv`
