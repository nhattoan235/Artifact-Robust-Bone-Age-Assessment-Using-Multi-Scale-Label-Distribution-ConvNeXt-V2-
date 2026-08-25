# EXP-006 Fold 1 — Checkpoint audit

Ngày kiểm tra: 2026-08-23

## Kết luận

Checkpoint chính **không bị mất và không bị lưu sai**.

Hai file ở thư mục gốc là artifact cần dùng:

- `best_mae.ckpt`: best MAE hiện tại `6.587100471866097`, tương ứng epoch 9.
- `last.ckpt`: trạng thái mới nhất sau epoch 9, dùng để resume.

Thư mục `best/` chỉ đang chứa bản snapshot cũ:

```text
best/epoch_003_mae_6.9650.ckpt
```

Bản này tương ứng epoch 3, MAE `6.965027377136752`, không phải best hiện tại.

## Bằng chứng

Run state:

- Run ID: `EXP006_P7_CONTROL_FOLD_1`.
- Epoch đã hoàn tất: 9.
- Global step: 2808.
- Best MAE: `6.587100471866097`.
- Best epoch trong state dùng chỉ số zero-based là 8, hiển thị thành epoch 9 trong
  review snapshot.
- Checkpoint chính: `best_mae.ckpt` và `last.ckpt` đều tồn tại.

Payload checkpoint đã đọc trực tiếp:

| File | epoch trong payload | best_mae | best_epoch trong payload | global_step |
|---|---:|---:|---:|---:|
| `best_mae.ckpt` | 8 | 6.587100 | 8 | 2808 |
| `last.ckpt` | 9 | 6.587100 | 8 | 2808 |
| `best/epoch_003_mae_6.9650.ckpt` | 2 | 6.965027 | 2 | 936 |

Review epoch 9:

- Validation MAE: `6.5871`.
- Validation RMSE: `8.8844`.
- Median AE: `5.0000`.
- MAE nữ: `6.890625`.
- MAE nam: `6.330272`.
- Age bin yếu nhất: `60–119`, MAE `8.644725`.
- Prediction range: `[-1.8867, 216.1250]`.
- Peak VRAM: `4426 MiB`.

## Nguyên nhân thư mục `best/` bị cũ

Trainer lưu checkpoint chính ở thư mục gốc và mirror các file nhỏ. Snapshot dạng:

```text
best/epoch_XXX_mae_YYY.ckpt
```

được tạo trong run directory local, nhưng không được đồng bộ đầy đủ sang Drive
mirror trong cơ chế mirror hiện tại. Vì vậy thư mục `best/` trên bản mirror có thể
thấp hơn `best_mae.ckpt` hiện tại.

Đây là lỗi không nhất quán của artifact phụ, không làm hỏng checkpoint chính.

## Quyết định sử dụng

- Resume bằng `last.ckpt`.
- Inference/evaluation bằng `best_mae.ckpt`.
- Không dùng file trong thư mục `best/` để chọn model ở run này.
- Không sửa code trainer giữa lúc resume vì checkpoint có kiểm tra code version.

## Đường dẫn đã kiểm tra

Thư mục mirror local:

```text
D:\do_an_tot_nghiep\project\baseline_v1\outputs\outputs\outputs\exp006_roadmap\EXP006_P7_CONTROL_FOLD_1
```

Các artifact chính:

- `best_mae.ckpt`.
- `last.ckpt`.
- `metrics.jsonl`.
- `run_state.json`.
- `val_predictions_best.csv`.
- `train.log`.
- `warnings.log`.

## Hành động tiếp theo

Không cần train lại từ đầu. Khi GPU hoạt động lại, resume từ `last.ckpt`. Sau khi
fold hoàn tất, có thể tạo một snapshot best dễ đọc trong thư mục `best/` từ
`best_mae.ckpt`, nhưng việc này chỉ phục vụ tổ chức file, không thay đổi model.
