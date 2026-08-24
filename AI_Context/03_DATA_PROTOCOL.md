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

- Workspace data root: `data\`
- Train images: `data\goc\boneage-training-dataset\boneage-training-dataset`
- Train CSV: `data\goc\boneage-training-dataset.csv`
- Validation images: `data\rsna_official_validation\images`
- Validation CSV: `data\rsna_official_validation\validation.csv`
- Test images: `data\goc\boneage-test-dataset\boneage-test-dataset`
- Test sex CSV: `data\goc\boneage-test-dataset.csv`
- Test ground truth: `data\goc\rsna_test_ground_truth.csv`
- C1 manifests: `c1_curated\outputs\C1_MANIFEST_V1\`

P0 manifests giữ nguyên fingerprint lịch sử và còn chứa absolute path cũ; C1 remap path sang workspace mà không đổi ID/nhãn/SHA.

## Quy tắc sử dụng

1. P9 model/config selection chỉ dùng train/validation hoặc OOF; không đọc tuổi test.
2. Không tự tách lại validation khi đã có split chính thức.
3. Trước mỗi run ghi manifest SHA, config hash, code version, seed và preprocessing version.
4. Nếu muốn dùng 200 ảnh cleaned artifact-only, phải coi là nhánh robustness riêng; không trộn vào OOF chính.
5. Sau khi P9 khóa, chỉ chạy test một lần cho mỗi protocol đã định trước. P8 đã chạm test nên không gọi P8 là hold-out hoàn toàn mới.

