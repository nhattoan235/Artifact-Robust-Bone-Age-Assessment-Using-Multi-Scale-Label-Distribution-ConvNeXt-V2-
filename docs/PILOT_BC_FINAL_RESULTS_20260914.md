# Báo cáo kết quả Pilot B/C — 2026-09-14

## Phạm vi

Báo cáo này tổng hợp riêng nhánh thí nghiệm B/C của C3-Z26 ROI V2. Tất cả kết quả chính được tính trên validation OOF của 5 fold đã khóa, gồm 14.036 ảnh. Nhánh B chỉ được dùng làm pilot trên Fold 1; nhánh C được huấn luyện và đánh giá đủ 5 fold. Không truy cập tập test 200 ảnh trong các thí nghiệm B/C này.

Nguồn số liệu là gói evidence tải từ Google Drive ngày 2026-09-14. Các file đã được nhập có chọn lọc vào [`results/pilot_bc_20260914`](../results/pilot_bc_20260914); checkpoint và ZIP đóng gói không được đưa vào Git.

## Thiết kế thí nghiệm

| Phiên bản | Artifact augmentation | Consistency weight | Mục tiêu |
|---|---|---:|---|
| Baseline C3-V2 | Không | 0,00 | Mốc so sánh clean và artifact |
| Pilot B | `mild_v1`, xác suất 1,0, warm-up 3 epoch và ramp 5 epoch | 0,00 | Đo tác động của artifact augmentation |
| Pilot C | Giống Pilot B | 0,30 | Buộc dự đoán clean và artifact của cùng ảnh gần nhau |

`mild_v1` gồm các suy giảm tổng hợp nhẹ như brightness/contrast/gamma, blur/noise và edge band. Vì vậy kết quả đo độ bền trước họ artifact tổng hợp này, chưa phải external validation trên ảnh bệnh viện khác.

## Pilot B — Fold 1

| Metric | Baseline Fold 1 | Pilot B | Thay đổi |
|---|---:|---:|---:|
| Clean MAE | 6,2535 | 6,2984 | +0,0449 tháng |
| Artifact MAE | 7,1521 | 6,4835 | **−0,6686 tháng** |
| Clean/artifact disagreement | 3,2306 | 1,4223 | **−56,0%** |

Paired bootstrap cho artifact delta `B − baseline` có CI 95% xấp xỉ `[-0,8340; -0,5041]`. Pilot B cải thiện độ bền trước artifact rõ ràng nhưng làm clean MAE xấu hơn nhẹ, nên không được mở rộng thành mô hình chính.

## Pilot C — Fold 1

| So sánh | Clean gain | CI 95% | Artifact gain | CI 95% |
|---|---:|---:|---:|---:|
| C so với B | 0,0233 | `[-0,1190; 0,0747]` | 0,0910 | `[-0,1968; 0,0163]` |
| C so với baseline | −0,0216 | `[-0,0758; 0,1220]` | **0,7597** | `[-0,9258; -0,5897]` |

Trên Fold 1, C chưa tốt hơn B một cách có ý nghĩa thống kê. So với baseline, C cải thiện artifact rõ ràng nhưng clean không khác rõ. Kết quả này đủ cơ sở để chạy C trên 5 fold nhằm kiểm tra tính ổn định.

## Pilot C độc lập — OOF 5 fold

### Kết quả từng fold

| Fold | Số ảnh | Clean MAE | Artifact MAE | Disagreement | Best epoch | Trạng thái |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.808 | 6,2751 | 6,3925 | 1,0919 | 19 | Early stopped |
| 2 | 2.807 | 6,2770 | 6,4174 | 1,0888 | 17 | Early stopped |
| 3 | 2.807 | 6,4482 | 6,5565 | 1,0628 | 14 | Early stopped |
| 4 | 2.807 | 6,4226 | 6,6099 | 1,0480 | 11 | Early stopped |
| 5 | 2.807 | 6,4305 | 6,6374 | 1,2536 | 7 | Early stopped |

### Kết quả gộp

| Metric | Baseline | Pilot C | Delta `C − baseline` | CI 95% |
|---|---:|---:|---:|---:|
| Clean MAE | 6,3300 | 6,3707 | +0,0406 | `[-0,0065; 0,0863]` |
| Artifact MAE | 7,2378 | 6,5227 | **−0,7151** | `[-0,7984; -0,6332]` |
| Disagreement | 3,0933 | 1,1090 | **giảm 64,15%** | delta `[-2,0504; -1,9188]` tháng |

Pilot C độc lập đạt gate clean non-inferiority, artifact superiority và stability. Gate subgroup không đạt. Kết luận phù hợp là C giữ dự đoán ổn định hơn rất rõ khi ảnh bị artifact, đổi lại clean MAE tăng nhẹ.

## Blend baseline/Pilot C

Phân tích grid đầy đủ trên cùng OOF chọn trọng số 27% baseline và 73% Pilot C. Vì trọng số được chọn từ chính OOF development, kết quả fixed 0,73 là exploratory và phải được khóa trước khi dùng trên một tập xác nhận mới.

Lưu ý kiểm toán: trường `summary.weight_c` trong hai report blend đang ghi `1.0` vì hàm tổng hợp nhận prediction đã blend sẵn. Trọng số thực được xác minh từ trường top-level và tính lại từng dòng prediction là `0.27 × baseline + 0.73 × Pilot C`, sai số số học tối đa dưới `9e-14`.

| Metric | Baseline | Blend 27/73 | Delta | CI 95% |
|---|---:|---:|---:|---:|
| Clean MAE | 6,3300 | **6,2725** | **−0,0575** | `[-0,0925; -0,0229]` |
| Artifact MAE | 7,2378 | **6,4879** | **−0,7499** | `[-0,8151; -0,6859]` |
| Disagreement | 3,0933 | 1,3722 | **giảm 55,64%** | — |

Cross-fit chọn trọng số C theo fold là `0,80; 0,77; 0,71; 0,65; 0,71`. Kết quả held-out gộp đạt clean MAE 6,2728 và artifact MAE 6,4985. Tuy nhiên constraint subgroup gộp không đạt do clean delta của nhóm 0–59 tháng là +0,2084, vượt margin +0,20 khoảng 0,0084 tháng.

## Kiểm tra multi-seed artifact

Trọng số 0,73 được giữ cố định. Ba seed dưới đây chỉ thay đổi lần sinh artifact, không phải ba lần huấn luyện model.

| Artifact seed | Baseline MAE | Blend MAE | Gain | CI 95% của delta |
|---:|---:|---:|---:|---:|
| 20260912 | 7,2378 | 6,4879 | 0,7499 | `[-0,8150; -0,6857]` |
| 20260913 | 7,3070 | 6,5333 | 0,7736 | `[-0,8404; -0,7077]` |
| 20260914 | 7,2604 | 6,5055 | 0,7549 | `[-0,8226; -0,6894]` |
| Trung bình | **7,2684** | **6,5089** | **0,7595** | `[-0,8159; -0,7048]` |

Gain ổn định trong khoảng 0,750–0,774 tháng ở cả ba artifact seed. Mức trung bình 0,7595 tháng tương đương khoảng 23 ngày và giảm artifact MAE khoảng 10,45% so với baseline.

## Subgroup của blend 27/73

### Theo giới tính

| Nhóm | N | Clean delta | CI 95% | Artifact delta | CI 95% |
|---|---:|---:|---:|---:|---:|
| Nữ | 6.430 | −0,0350 | `[-0,0861; 0,0174]` | **−0,7072** | `[-0,8045; -0,6107]` |
| Nam | 7.606 | **−0,0765** | `[-0,1233; -0,0297]` | **−0,7859** | `[-0,8734; -0,6983]` |

Artifact cải thiện rõ ở cả hai giới. Clean cải thiện rõ ở nam; ở nữ, CI cắt 0 nên chưa kết luận có khác biệt.

### Theo nhóm tuổi

| Nhóm tuổi | N | Clean delta | CI 95% | Artifact delta | CI 95% | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| 0–59 | 895 | **+0,1960** | `[0,0064; 0,3845]` | −0,0809 | `[-0,3399; 0,1821]` | Clean xấu hơn; artifact chưa rõ |
| 60–119 | 3.881 | **+0,0827** | `[0,0110; 0,1536]` | **−0,5096** | `[-0,6465; -0,3769]` | Clean xấu hơn; artifact tốt hơn |
| 120–179 | 8.033 | **−0,1743** | `[-0,2142; -0,1342]` | **−1,0232** | `[-1,1071; -0,9420]` | Cả hai tốt hơn |
| 180–228 | 1.227 | +0,0790 | `[-0,0674; 0,2230]` | **−0,2083** | `[-0,3916; -0,0300]` | Clean chưa rõ; artifact tốt hơn |

Điểm yếu chính là clean performance dưới 120 tháng, đặc biệt nhóm 0–59. Đây là vấn đề cần xử lý trước khi tuyên bố phương pháp tốt đồng đều trên mọi lứa tuổi.

## Kết luận

1. Pilot B chứng minh artifact augmentation có hiệu quả nhưng đánh đổi clean MAE.
2. Pilot C thêm consistency loss giúp giảm clean/artifact disagreement từ 3,0933 xuống 1,1090 tháng và giảm artifact MAE 0,7151 tháng trên OOF 5-fold.
3. Blend 27/73 là cấu hình sử dụng thực tế tốt nhất hiện tại cho nhánh B/C: clean MAE 6,2725 và artifact MAE 6,4879.
4. Multi-seed xác nhận artifact gain ổn định, nhưng mới bao phủ ba lần sinh của cùng họ artifact `mild_v1`.
5. Kết quả B/C chưa phải external validation và chưa chứng minh khả năng khái quát tới ảnh bệnh viện khác.
6. Bước nghiên cứu tiếp theo nên tập trung sửa clean subgroup dưới 120 tháng và đánh giá trên artifact family hoặc dữ liệu ngoài chưa dùng để chọn trọng số.

## File bằng chứng trong Git

- `results/pilot_bc_20260914/runs/`: báo cáo, prediction, config và log của B Fold 1, C Fold 1–5.
- `results/pilot_bc_20260914/oof/`: OOF 5-fold và subgroup.
- `results/pilot_bc_20260914/oof/blend/`: fixed blend, cross-fit blend và grid exploratory.
- `results/pilot_bc_20260914/oof/multiseed/`: report và prediction 42.108 dòng của ba artifact seed.
- `results/pilot_bc_20260914/CURATED_MANIFEST.json`: nguồn, quy tắc chọn file và SHA-256 của từng file.

Checkpoint B/C vẫn được lưu ngoài Git trong `C3_Z26_PILOT_BC_MODELS_20260914.zip`. Không đưa checkpoint hoặc ZIP vào repository để tránh làm tăng kích thước lịch sử Git.
