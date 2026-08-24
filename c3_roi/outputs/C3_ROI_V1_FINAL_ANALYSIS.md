# C3-ROI — phân tích OOF và test

## Kết luận ngắn

C3-ROI là ứng viên tốt hơn E1 trên tập test 200 ảnh, nhưng chưa tốt hơn trên OOF phát triển. Vì vậy không được kết luận C3 đã cải thiện tổng quát; kết quả test chỉ là đánh giá thăm dò do test đã được mở trước đó.

## OOF development — 14.036 ảnh

| Mô hình | MAE (tháng) |
|---|---:|
| E1-full | 6,31669 |
| C3-ROI | 6,43735 |
| E1 + C3 50/50 | **6,17621** |

Ensemble 50/50 cải thiện so với E1 `0,14048` tháng; paired bootstrap 95% CI của delta là `[-0,17255; -0,10985]`. Đây là bằng chứng OOF tích cực, dù C3 đơn lẻ kém E1.

## Test thăm dò — 200 ảnh

| Mô hình | MAE (tháng) | RMSE (tháng) |
|---|---:|---:|
| E1-full | 4,73032 | 6,02875 |
| C3-ROI | **4,33727** | **5,53110** |
| E1 + C3 50/50 | 4,45466 | 5,65069 |

C3 tốt hơn E1 `0,39305` tháng trên 200 ảnh. Ensemble 50/50 tốt hơn E1 `0,27566` tháng nhưng không tốt bằng C3 riêng. Điều này không được dùng để đổi trọng số sau khi xem test.

## Subgroup test

| Nhóm | E1 | C3 | Ensemble |
|---|---:|---:|---:|
| Nữ (n=100) | 4,8562 | 4,7412 | 4,7130 |
| Nam (n=100) | 4,6044 | **3,9333** | 4,1963 |
| 0–59 (n=14) | 5,1927 | **4,3572** | 4,7750 |
| 60–119 (n=53) | 6,6198 | **5,6182** | 6,0614 |
| 120–179 (n=108) | 3,9324 | 4,0140 | **3,8936** |
| 180–228 (n=25) | 3,9125 | **3,0072** | 3,2928 |

## Quyết định

1. Giữ C3-ROI là một nhánh có tín hiệu tốt và giữ ensemble 50/50 như kết quả OOF đã khóa.
2. Không tối ưu thêm trọng số dựa trên test.
3. Không tuyên bố đã vượt mốc Bram 3,68 tháng; C3 test đạt 4,33727 tháng.
4. Kết quả có giá trị học thuật chính là: ROI/mask branch tạo dự đoán bổ sung đủ để ensemble cải thiện OOF leakage-safe, nhưng hiệu quả phụ thuộc phân phối và cần external holdout chưa chạm để xác nhận.
