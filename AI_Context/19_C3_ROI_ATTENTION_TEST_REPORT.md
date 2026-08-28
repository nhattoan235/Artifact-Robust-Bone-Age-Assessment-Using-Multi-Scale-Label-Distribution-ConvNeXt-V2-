# Báo cáo C3-ROI + Spatial Attention trên RSNA test 200

**Ngày khóa kết quả:** 2026-08-29

**Trạng thái:** hoàn tất đánh giá 5-fold; không tuning trên test
**Phạm vi:** ablation ROI so với ROI + spatial attention

## 1. Câu hỏi nghiên cứu

Sau khi ảnh đã được crop theo ROI bàn tay, spatial attention có tiếp tục cải thiện
khả năng dự đoán tuổi xương hay không?

So sánh chính là:

- `C3_ROI_V1`: ConvNeXt-Tiny + sex embedding, đầu vào ROI.
- `C3_ROI_ATTN_V1`: cùng pipeline ROI và recipe, thêm spatial attention có cổng
  khởi tạo identity.

Attention là thay đổi duy nhất có chủ đích. Thí nghiệm này không dùng để thay đổi
ROI, loss, fold, epoch, checkpoint hoặc trọng số ensemble.

## 2. Protocol đánh giá

- Bộ đánh giá: 200 ảnh RSNA test gốc.
- Đầu vào: ROI cache cố định `C3_ROI_V1_TEST`.
- Mỗi phương pháp dùng trung bình prediction của 5 fold.
- Không TTA.
- Không chọn checkpoint hay hyperparameter bằng test.
- Chênh lệch MAE được tính paired theo cùng `image_id`.
- CI 95% dùng paired bootstrap 5.000 lần, seed 42.
- Vì bộ test đã được dự án sử dụng trước đây, kết quả được gọi là
  **locked exploratory re-evaluation**, không phải holdout hoàn toàn mới.

## 3. Kiểm tra tính toàn vẹn

- Attention: 5/5 checkpoint hợp lệ.
- Mỗi fold tạo đúng 200 prediction.
- Sau ensemble có 200 dòng và 200 `image_id` duy nhất.
- Tập ID Attention, C3-ROI và E1 tham chiếu giống nhau.
- Không dùng test để điều chỉnh mô hình sau khi xem kết quả.

## 4. Kết quả chính

| Model | MAE (tháng) | RMSE (tháng) | Median AE (tháng) |
|---|---:|---:|---:|
| C3-ROI | **4,3373** | **5,5311** | **3,3945** |
| C3-ROI + Attention | 4,4119 | 5,6391 | 3,5871 |
| E1 tham chiếu | 4,7303 | 6,0287 | 4,1334 |

Chênh lệch paired `Attention − C3-ROI`:

- Point estimate: **+0,0747 tháng**.
- Paired-bootstrap CI 95%: **[−0,1238; +0,2617] tháng**.

Giá trị dương nghĩa là Attention có MAE cao hơn. CI cắt 0 nên chưa thể khẳng
định hai phương pháp khác nhau về mặt thống kê.

## 5. Phân tích theo giới

| Giới | Số ảnh | C3-ROI MAE | Attention MAE | Attention − C3 |
|---|---:|---:|---:|---:|
| Nữ | 100 | 4,7412 | 4,7700 | +0,0288 |
| Nam | 100 | 3,9333 | 4,0538 | +0,1205 |

Attention không cải thiện MAE ở cả hai nhóm giới. Mức suy giảm lớn hơn ở nhóm
nam, nhưng phân tích phân nhóm này chỉ mang tính mô tả vì chưa thực hiện kiểm
định riêng có hiệu chỉnh đa so sánh.

## 6. Đối chiếu với OOF development

| Đánh giá | C3-ROI | Attention | Attention − C3 | CI 95% |
|---|---:|---:|---:|---:|
| OOF 14.036 ảnh | 6,4373 | 6,4209 | −0,0165 | [−0,0637; +0,0303] |
| Test 200 ảnh | 4,3373 | 4,4119 | +0,0747 | [−0,1238; +0,2617] |

OOF cho point estimate tốt hơn rất nhỏ, còn test cho point estimate xấu hơn.
Cả hai CI đều cắt 0. Kết quả nhất quán với nhận định rằng hiệu quả thật của
spatial attention sau ROI gần bằng 0 và không ổn định giữa development/test.

## 7. Kết luận khoa học

Không có bằng chứng cho thấy spatial attention bổ sung cải thiện C3-ROI. ROI đã
loại phần lớn nền và tập trung đầu vào vào bàn tay; attention bổ sung có thể trở
nên dư thừa, đồng thời tăng độ tự do của mô hình mà không cung cấp tín hiệu mới.

Quyết định:

1. Giữ `C3_ROI_V1` là phương án ROI chính.
2. Không chọn `C3_ROI_ATTN_V1` làm model cuối chỉ dựa trên test.
3. Báo cáo Attention như một **negative ablation** có giá trị: thêm attention sau
   ROI không tạo cải thiện MAE đáng tin cậy.
4. Không tuning attention hoặc ensemble weight trên 200 test.

## 8. Tái lập

```powershell
& .\.venv-c2\Scripts\python.exe -m c3_roi_attention.infer_test `
  --batch-size 4 `
  --num-workers 0
```

Đầu ra:

- `c3_roi_attention/outputs/C3_ROI_ATTN_V1_TEST/C3_ROI_ATTN_V1_TEST_report.json`
- `c3_roi_attention/outputs/C3_ROI_ATTN_V1_TEST/C3_ROI_ATTN_V1_TEST_predictions.csv`
- `c3_roi_attention/outputs/C3_ROI_ATTN_V1_TEST/ATTN_fold_1_predictions.csv` đến
  `ATTN_fold_5_predictions.csv`
