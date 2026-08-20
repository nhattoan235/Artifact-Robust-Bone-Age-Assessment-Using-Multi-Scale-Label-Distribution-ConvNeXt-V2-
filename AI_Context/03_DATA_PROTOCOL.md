# Giao thức dữ liệu bất biến

## Split đã khóa

| Split | Số ảnh | Vai trò |
|---|---:|---|
| Train chính thức | 12.611 | Fit trọng số |
| Validation chính thức | 1.425 | Chọn cấu hình/checkpoint trong development |
| Development 5-fold | 14.036 | OOF cho P7/P9 |
| RSNA test | 200 | Đánh giá cuối; không dùng chọn mô hình |

Manifest/hash chi tiết: `p0_audit/P0_HANDOFF.md` và `p0_audit/outputs/`.

## Đường dẫn cục bộ

- RSNA root: `D:\Hoctap\Doan_totnghiep\Dataset\RSNA`
- Train images: `Dataset\RSNA\boneage-training-dataset\boneage-training-dataset`
- Train CSV: `Dataset\RSNA\boneage-training-dataset.csv`
- Validation images: `Dataset\RSNA\boneage-validation-dataset\boneage-validation-dataset`
- Validation CSV: `Dataset\RSNA\boneage-validation-dataset.csv`
- Test images: `Dataset\RSNA\boneage-test-dataset\boneage-test-dataset`
- Test sex CSV: `Dataset\RSNA\boneage-test-dataset.csv`
- Test ground truth: `Dataset\rsna_test.csv`
- Deeplasia semantic copy: `external\Deeplasia\data\rsna_test.csv`

## Quy tắc sử dụng

1. P9 model/config selection chỉ dùng train/validation hoặc OOF; không đọc tuổi test.
2. Không tự tách lại validation khi đã có split chính thức.
3. Trước mỗi run ghi manifest SHA, config hash, code version, seed và preprocessing version.
4. Nếu muốn dùng 200 ảnh cleaned artifact-only, phải coi là nhánh robustness riêng; không trộn vào OOF chính.
5. Sau khi P9 khóa, chỉ chạy test một lần cho mỗi protocol đã định trước. P8 đã chạm test nên không gọi P8 là hold-out hoàn toàn mới.

