# BÁO CÁO BÀN GIAO P0 – AUDIT DỮ LIỆU, LEAKAGE VÀ MÔI TRƯỜNG

**Ngày hoàn tất:** 2026-08-13  
**Trạng thái P0:** HOÀN THÀNH – PASS  
**Được phép chuyển sang:** P1 – xây dựng baseline ConvNeXt-Tiny có khả năng tái lập  
**Test đã dùng để chọn mô hình hoặc siêu tham số:** KHÔNG  
**Tuổi xương/metric của test đã được xem hoặc tính:** KHÔNG

## 1. Ba split chính thức đã khóa

| Split | Số ảnh | CSV/manifest | Vai trò |
|---|---:|---|---|
| Train chính thức | 12.611 | `outputs/train_manifest.csv` | Huấn luyện và ablation |
| Validation chính thức | 1.425 | `outputs/validation_manifest.csv` | Chọn mô hình, checkpoint và siêu tham số |
| Test chính thức | 200 | `outputs/test_manifest_LOCKED_NO_AGE.csv` | Chỉ đánh giá một lần sau khi khóa mô hình |

Đường dẫn dữ liệu chuẩn:

- Root: `D:\Hoctap\Doan_totnghiep\Dataset\RSNA`
- Train CSV: `boneage-training-dataset.csv`
- Train images: `boneage-training-dataset\boneage-training-dataset`
- Validation CSV: `boneage-validation-dataset.csv`
- Validation images: `boneage-validation-dataset\boneage-validation-dataset`
- Test metadata: `boneage-test-dataset.csv`
- Test images: `boneage-test-dataset\boneage-test-dataset`
- Test ground truth bị khóa: `D:\Hoctap\Doan_totnghiep\Dataset\rsna_test.csv`

Thư mục validation chứa hai thư mục con của hai gói ảnh gốc. Không cần làm phẳng; manifest lưu chính xác đường dẫn từng ảnh và dataloader phải đọc từ manifest.

## 2. Bằng chứng validation tương thích

Gói tải về `Bone+Age+Validation+Set.zip` có SHA-256:

```text
683ea5398dd9d0730af883e673176a5c6817fd35a5a59331a500fa7affcd36f0
```

Audit validation đạt toàn bộ 15 kiểm tra:

- CSV có đúng 1.425 dòng và có đúng 1.425 ảnh PNG;
- ID trong CSV và ảnh khớp hoàn toàn, không trùng ID;
- không giao ID với train 12.611 hoặc test 200;
- 1.425/1.425 ảnh đọc được và đều là PNG grayscale;
- nhãn tuổi hợp lệ, trong khoảng 3–228 tháng;
- giới tính gồm 773 nam và 652 nữ;
- tuổi và giới tính khớp hoàn toàn annotation công khai của Deeplasia;
- không có ảnh trùng SHA-256 trong validation;
- không có ảnh trùng SHA-256 giữa validation với train hoặc test.

Kết luận: đây là split validation chính thức tương thích với protocol 12.611/1.425/200. Không tự tách validation từ train.

## 3. Kết quả audit toàn bộ dữ liệu

- Development pool: 14.036 ảnh = 12.611 train + 1.425 validation.
- Tổng cả ba split: 14.236 ảnh.
- Không thiếu ảnh hoặc metadata.
- Không có ảnh hỏng.
- Không có ID trùng giữa ba split.
- Không có SHA-256 trùng giữa ba split.
- Toàn bộ tuổi train/validation hợp lệ trong miền 0–228 tháng.
- File manifest test và workbook P0 không chứa tuổi xương test.

Phân bố validation:

- toàn bộ: mean 127,156 tháng; median 132; SD 41,722;
- nam: 773 ảnh, mean 134,916 tháng;
- nữ: 652 ảnh, mean 117,957 tháng.

Vì phân bố tuổi khác nhau theo giới tính, final 5-fold phải stratify theo `sex × age_bin` như Master Plan.

## 4. Fingerprint bất biến

```text
train_manifest_sha256=7328667e6822ab074d442155e33eba89606861bb13bbf804be8a1138c78f5285
validation_manifest_sha256=f650a20432b557405035d12c52716a6c40fdf20e1343ac3ad0cfb7cd8db21631
test_locked_manifest_sha256=fd22bf5b44397a32826cfedad805b31ba727637047a48be51cc8954bb51229db
development_manifest_14036_sha256=db1d62d768aca44016e9434c0d3c3b64e5f695094338cff0dbf6da2077e5925a
three_split_registry_sha256=234d368e37dc908c4e30d271a5dee2e3a853e4a9920628b476c418e457b6b7e6
```

Mọi run sau phải kiểm tra fingerprint trước khi train. Nếu thay đổi, phải dừng và giải thích nguyên nhân trước khi tiếp tục.

## 5. Khóa test

File `rsna_test.csv` chỉ được P0 dùng để kiểm tra số dòng, ID, sex, miền hợp lệ và fingerprint. Script P0 không xuất tuổi xương test, không tính phân bố tuổi và không tính metric.

Từ P1 đến P7, code phát triển không được nhận đường dẫn test ground truth. Chỉ bước đánh giá cuối P8 mới được phép đọc file này sau khi config, checkpoint rule và ensemble đã khóa.

## 6. Môi trường GPU

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6.141 MiB.
- Python: 3.12.13.
- PyTorch: 2.7.1+cu128.
- CUDA build: 12.8; cuDNN: 90701.
- timm: 1.0.28.
- Synthetic forward/backward FP16, 512×512, batch 1: ConvNeXt-Tiny 556 MiB; ConvNeXtV2-Tiny 812 MiB peak reserved.

Đây là capability check, không phải dự báo chính xác VRAM của trainer hoàn chỉnh. P1 vẫn phải chạy smoke test dataloader, checkpoint và resume trước run dài.

## 7. Artifact P0

- `outputs/P0_AUDIT_REPORT.xlsx`: workbook audit đã kiểm tra trực quan.
- `outputs/P0_FINAL_REPORT.json`: kết luận máy đọc được và các gate P0.
- `outputs/P0_FINAL_FINGERPRINTS.txt`: toàn bộ fingerprint đã khóa.
- `outputs/three_split_registry_LOCKED.csv`: registry ba split.
- `outputs/development_manifest_14036.csv`: manifest train + validation cho final 5-fold sau này.
- `outputs/validation_audit_report.json`: chi tiết 15 kiểm tra validation.
- `outputs/validation_manifest.csv`: manifest validation chính thức.

Archive tải về và thư mục `_incoming_validation` vẫn được giữ làm nguồn đối chiếu; chưa xóa dữ liệu nào.

## 8. Gate chuyển P1

- [x] Train 12.611 ảnh đã audit.
- [x] Validation chính thức 1.425 ảnh đã audit và khớp Deeplasia.
- [x] Test 200 ảnh đã audit và khóa nhãn.
- [x] Không phát hiện exact duplicate/leakage giữa ba split.
- [x] Manifest và fingerprint ba split đã khóa.
- [x] Môi trường 512 smoke capability đạt.
- [x] Master Plan đã cập nhật.

## 9. Bàn giao

```text
Phase: P0
Mục tiêu: Audit dữ liệu, split, metadata, leakage và môi trường
Kết quả: PASS; protocol 12.611/1.425/200 đã được xác minh
Data hash: Train 7328667e...f5285; Validation f650a204...21631; Test fd22bf5b...129db
Development hash: db1d62d7...925a; Registry hash: 234d368e...b7e6
Quyết định: Đóng P0 và cho phép bắt đầu P1
Bước tiếp theo: Xây baseline ConvNeXt-Tiny tái lập, có checkpoint/resume/log/cảnh báo
Test set: chỉ audit ID/hash; KHÔNG dùng metric hoặc chọn model
```
