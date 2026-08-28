# Chạy C3-ROI + attention trên Colab

## File cần có trong `MyDrive/data`

1. `C3_ROI_V1_READY_TO_TRAIN_512_JPEG_V2.zip` — ROI data cũ đã upload.
2. `C3_ROI_ATTENTION_COLAB_CODE_FINAL_V3.zip` — gói code mới.

Nếu thư mục do tài khoản khác chia sẻ, mỗi tài khoản cần chọn **Organize → Add
shortcut → My Drive** và giữ shortcut là `MyDrive/data`. Nếu dùng tên khác, chỉ
sửa biến `DRIVE_DATA_ROOT` trong notebook.

Mở `C3_ROI_ATTENTION_COLAB.ipynb`, bật GPU T4 và chạy lần lượt từ trên xuống.
Mỗi tài khoản chỉ đổi:

```python
FOLD = 1
```

thành fold được giao từ 1 đến 5. Các cell còn lại giữ nguyên.

## Checkpoint

Checkpoint tự mirror vào:

```text
MyDrive/data/c3_attention_runs/C3_ROI_ATTN_V1_FOLD_<FOLD>/
```

Lần đầu runner tự chạy fresh. Nếu `last.ckpt` đã tồn tại, runner tự resume đúng
fold. Không thêm `--resume` thủ công.

## Sau khi đủ 5 fold

Tải năm thư mục fold về `c3_roi_attention/runs/C3_ROI_ATTN_V1/`, sau đó chạy:

```text
python -m c3_roi_attention.aggregate_oof
```

Script kiểm tra đủ 14.036 OOF IDs và xuất paired-bootstrap comparison với C3-ROI.
