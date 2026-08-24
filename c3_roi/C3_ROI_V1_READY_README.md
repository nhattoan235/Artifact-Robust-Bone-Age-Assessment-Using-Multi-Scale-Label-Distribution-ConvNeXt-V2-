# C3 ROI V1 — ready-to-run package

Sau khi giải nén package vào `/content/d3_workspace` hoặc thư mục workspace tương
đương, cấu trúc `c3_roi/...` và `p1_baseline/...` phải được giữ nguyên.

## Đã bao gồm

- Data-independent mask cache: 14.036 mask.
- ROI cache: 14.036 ảnh tại `c3_roi/cache/C3_ROI_V1/roi`.
- Manifest/config/registry cho 5 fold.
- Code trainer và preflight.
- Phân bổ: Colab Fold 1–3; local Fold 4–5.

## Chạy preflight

```text
python -m p1_baseline.preflight --config c3_roi/configs/fold_1.toml
```

## Chạy Fold 1

```text
python -m p1_baseline.train --config c3_roi/configs/fold_1.toml
```

Đổi `fold_1` thành `fold_2` hoặc `fold_3` cho Colab. Checkpoint phải mirror sang
Drive trước khi runtime Colab reset.

## Trạng thái khoa học

Package này chạy được về mặt kỹ thuật, nhưng là **candidate experiment**. ROI được
sinh từ mask candidate có 2.595 fallback (18,5%) và cần re-audit test-blind trước
khi dùng làm kết quả chính thức. Không dùng test 200 để chọn checkpoint hoặc blend.

