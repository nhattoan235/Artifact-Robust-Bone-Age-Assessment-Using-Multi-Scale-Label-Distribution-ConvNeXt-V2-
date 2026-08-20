# P7 V3 — hướng dẫn sạch từ đầu

## Chỉ upload một lần

Trên Drive được chọn làm nơi lưu, tạo đúng cấu trúc:

```text
MyDrive/bone_age_p7_v3/
├── P7_COLAB_BUNDLE_V3.zip
└── sources/
    ├── train_source.zip
    └── validation_source.zip
```

Hai file source có thể là shortcut tới ZIP đang nằm ở tài khoản khác. Không đặt dữ liệu test vào đây.

## Điều đã bỏ hoàn toàn

- Không còn `EXPECTED_STORAGE_EMAIL`.
- Không còn `STORAGE_ROOT_FOLDER_ID`.
- Không gọi `auth.authenticate_user()`.
- Không ghi checkpoint trực tiếp khi từng batch/epoch lên Drive.
- Không overwrite checkpoint lớn và không tự xóa file lớn.

## Cơ chế lưu

- Train, cache và ảnh đã giải nén nằm ở `/content`.
- Full checkpoint local khoảng mỗi 5 phút.
- Mỗi 60 phút tạo một checkpoint persistent có tên mới trên Drive.
- Mỗi khi validation MAE tốt hơn, lưu riêng `best_model` dạng gọn (~110 MiB); vì vậy reset Colab không làm mất best epoch.
- Prediction của best epoch được đồng bộ cùng best-model và là file bắt buộc trước khi finalize OOF.
- Mỗi file được kiểm tra đủ byte, SHA-256 và khả năng `torch.load`.
- Resume thử từ bản mới nhất; nếu bản đó hỏng sẽ tự lùi về bản trước.
- Tối đa 10 full checkpoint/fold (xấp xỉ 3,3 GiB) và 15 best-model gọn. Đạt giới hạn thì train dừng an toàn, không làm đầy Drive.
- Khi fold hoàn tất, `results/P7_FINAL_V3_FOLD_n` chứa model tốt nhất dạng gọn, prediction, log và manifest SHA-256.

## Dùng nhiều tài khoản Colab Free

Tài khoản mở Colab có thể thay đổi. Ở cell mount, luôn chọn Drive có `bone_age_p7_v3`. Chạy lại notebook với cùng `FOLD`; runner tự tìm và kiểm tra checkpoint, không nhập email hay ID.

Trước train thật phải làm hai lượt smoke: global step 2 ở tài khoản thứ nhất, sau đó global step 4 ở tài khoản thứ hai. Chỉ tiếp tục khi log xác nhận resume từ step 2 và phục hồi optimizer/scheduler/scaler.

## Dung lượng Drive

Notebook không thể bảo đảm xóa vĩnh viễn qua Drive mount, nên cố ý không tự xóa checkpoint lớn. Khi đủ 10 bản hoặc fold hoàn tất, giữ hai cặp `.ckpt`/`.json` mới nhất, xóa các cặp cũ trong giao diện Drive và xóa vĩnh viễn khỏi Thùng rác. Đây là thao tác dọn duy nhất, có phạm vi đúng một thư mục fold.
