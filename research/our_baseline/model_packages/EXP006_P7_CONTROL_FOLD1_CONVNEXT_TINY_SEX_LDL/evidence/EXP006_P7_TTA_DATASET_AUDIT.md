# EXP006 — Audit Dataset TTA

Dataset local:

```text
D:\do_an_tot_nghiep\project\baseline_v1\outputs\results\exp006-p7-tta-results
```

## Kết luận

Dataset đã đủ các thành phần kết quả cần thiết cho TTA trên Kaggle.

- Tổng file: **12**
- Dung lượng: khoảng **1.569 GB**
- OOF rows: **14.036**
- Image ID duy nhất: **14.036**
- ID trùng: **0**
- Fold counts: 2.808 / 2.807 / 2.807 / 2.807 / 2.807
- `test_labels_used`: **false**
- Có đủ `best_mae.ckpt` và `config_resolved.yaml` cho cả 5 fold.
- Cả 5 checkpoint đều đọc được bằng PyTorch, mỗi checkpoint có 186 model
  state keys.

## Cấu trúc đã kiểm tra

```text
oof/
├── oof_predictions.csv
└── oof_report.json
runs/
├── EXP006_P7_CONTROL_FOLD_1/{best_mae.ckpt,config_resolved.yaml}
├── EXP006_P7_CONTROL_FOLD_2/{best_mae.ckpt,config_resolved.yaml}
├── EXP006_P7_CONTROL_FOLD_3/{best_mae.ckpt,config_resolved.yaml}
├── EXP006_P7_CONTROL_FOLD_4/{best_mae.ckpt,config_resolved.yaml}
└── EXP006_P7_CONTROL_FOLD_5/{best_mae.ckpt,config_resolved.yaml}
```

## Lưu ý khi upload

Dataset này **không chứa ảnh gốc, mã nguồn friend repo hoặc script TTA**.
Khi chạy Kaggle phải attach đồng thời Dataset `boneage-exp006-assets`, vì
Dataset đó chứa `data_goc`, `friend_repo` và `baseline_scripts`.

Không cần thêm `last.ckpt`, log, metrics hoặc các ảnh raw vào Dataset TTA.
