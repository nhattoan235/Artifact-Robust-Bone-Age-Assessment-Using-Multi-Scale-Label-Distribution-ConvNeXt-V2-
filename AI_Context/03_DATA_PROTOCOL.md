# Giao thức dữ liệu và đánh giá

> Cập nhật: 2026-09-11. Các quy tắc này ưu tiên hơn mô tả cũ trong notebook/log.

## Split chuẩn

| Split | Số ảnh | Vai trò |
|---|---:|---|
| RSNA train chính thức | 12.611 | Fit model |
| RSNA validation chính thức | 1.425 | Development/kiểm tra ban đầu |
| Development gộp | 14.036 | 5-fold OOF; mỗi ảnh chỉ do fold không fit ảnh đó dự đoán |
| RSNA test chính thức | 200 | Benchmark thăm dò; nhãn đã được đọc nhiều lần |

Khóa theo patient/image ID, không theo đường dẫn tương đối. Trước khi so sánh phải kiểm tra count, ID duy nhất, target, sex, split hash và manifest hash.

## Chính sách test

Test 200 **không còn là untouched holdout**. Không được dùng MAE/nhãn test để:

- chọn preprocessing hoặc hình học ROI;
- chọn backbone, checkpoint, epoch hay hyperparameter;
- quyết định TTA, trọng số ensemble hoặc calibration;
- quyết định chạy tiếp Fold 3–5.

Chỉ dùng test để báo cáo exploratory với disclosure rõ. Xác nhận tương lai cần external holdout hoặc protocol mới khóa trước.

## Các họ input

| Tên | Nguồn ảnh | Biến đổi quyết định |
|---|---|---|
| E1/P7 global | Ảnh RSNA toàn cảnh | Resize về 512 theo recipe baseline; ConvNeXt normalization |
| C3-ROI V1 | Segmentation bbox bàn tay, margin 8% | Fallback toàn ảnh khi ROI lỗi; development 18,49%, test 33% |
| C3-R2 raw | ROI tái tạo margin 12% + border rescue | Hard fallback development 14,06%, test 28%; chưa có Z26 mặc định |
| C3-R2 Z26 HE | C3-R2 | Giữ tỷ lệ, resize/padding 512, histogram equalization, canvas đen |
| C3-R2 Z26 NO_HE | C3-R2 | Như trên nhưng không histogram equalization |

Không suy HE từ chữ Z26. Tên artifact/run phải ghi rõ HE hoặc NO_HE.

## So sánh công bằng

1. So candidate và control trên cùng image IDs, folds, targets và seed/split.
2. Dùng paired delta và paired bootstrap CI khi có dự đoán từng ảnh.
3. Screening hiện tại chỉ Fold 1–2 trên raw C3-R2; C0 là control matched.
4. Chỉ promote khi lợi ích đủ lớn, nhất quán qua hai fold và không có lỗi ổn định.
5. Khi mở rộng 5 fold, khóa trước config, code version, checkpoint rule và ensemble rule.
6. Không so MAE validation 1.425 với OOF 14.036 hoặc test 200 như cùng một cohort.

## Tính tái lập

Mỗi run cần lưu:

- config resolved và config hash;
- train/validation manifest cùng hash;
- code version;
- seed/fold;
- best_mae.ckpt, last.ckpt, metrics và predictions tốt nhất;
- log resume xác nhận split/config/code/optimizer/scheduler/scaler khớp.

Không sửa đè artifact đã hoàn thành. Khi chạy trên nhiều tài khoản Drive, phải xác minh đúng tài khoản sở hữu và checkpoint path thật trước khi resume.

## Dữ liệu bài báo để đối chiếu

- Deeplasia: train RSNA 12.611, chọn trên validation 1.425, test RSNA 200; DHA/GDBD là external test.
- Bram et al.: RSNA train+validation theo 5 fold, loại 35 trường hợp bất thường; test RSNA 200. Dữ liệu ngoài dùng đánh giá/chuyển miền riêng.
- Shu & Yu: báo cáo trên RSNA test 200, bổ sung 8 đặc trưng kích thước xương bàn tay được đo.
- Zhang et al. 2026: chỉ RSNA; các view là dẫn xuất từ cùng ảnh; mốc 4,10 được xác nhận trên validation 1.425, không phải test 200.
