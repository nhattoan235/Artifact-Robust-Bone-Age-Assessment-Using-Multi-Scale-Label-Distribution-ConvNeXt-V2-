# BÁO CÁO BÀN GIAO P2 – KHÓA AUGMENTATION

**Ngày hoàn tất:** 2026-08-14  
**Trạng thái:** PASS  
**Test ground truth/metric đã sử dụng:** KHÔNG  
**Bước tiếp theo:** P3 – B0/B1 preprocessing

## 1. Ba run chính thức

| Run | Augmentation | Best epoch | Validation MAE | RMSE | ±6 tháng |
|---|---|---:|---:|---:|---:|
| A0 | Không | 10 | 6,640 | 8,884 | 59,09% |
| A1 | Horizontal flip p=0,5 | 5 | 6,514 | 8,787 | 60,14% |
| A2 | Flip + geometric/intensity nhẹ | 12 | **6,185** | **8,486** | **63,44%** |

Tất cả dùng cùng train 12.611, validation chính thức 1.425, seed 42, ConvNeXt-Tiny pretrained, input 512, optimizer/scheduler và checkpoint rule.

## 2. So sánh paired trên cùng 1.425 ảnh

Delta được định nghĩa là `AE(candidate) − AE(baseline)`; số âm có lợi cho candidate.

| So sánh | Delta MAE | Paired bootstrap 95% CI | Kết luận |
|---|---:|---:|---|
| A1 − A0 | −0,125 | [−0,315; 0,065] | Tín hiệu có lợi, chưa chắc chắn |
| A2 − A0 | **−0,455** | **[−0,658; −0,249]** | A2 tốt hơn rõ |
| A2 − A1 | **−0,329** | **[−0,509; −0,152]** | A2 tốt hơn rõ |

A2 giảm MAE 6,85% so với A0 và 5,06% so với A1.

## 3. Augmentation được khóa

- Horizontal flip p=0,5.
- Rotation tối đa ±7°.
- Translation tối đa 3%.
- Scale 0,95–1,05.
- Brightness ±10%.
- Contrast ±10%.
- Gamma 0,90–1,10.
- Không vertical flip, MixUp, CutMix, elastic deformation hoặc TTA.

Augmentation được sinh xác định từ `seed + epoch + image ID`, nên stop/resume không làm đổi ảnh của cùng epoch.

## 4. Kết quả subgroup của A2

- MAE nữ: 6,389 tháng.
- MAE nam: 6,012 tháng.
- Khoảng cách giới tính: 0,377 tháng.
- 0–59 tháng: 7,283.
- 60–119 tháng: 6,838.
- 120–179 tháng: 5,930.
- 180–228 tháng: 4,977.

So với A0, A2 cải thiện cả hai giới và ba nhóm tuổi đầu. Nhóm 180–228 xấu nhẹ 0,055 tháng; cần theo dõi ở các phase sau.

## 5. Diễn giải thận trọng

- A2 có bằng chứng paired tốt hơn trên validation chính thức; đây là kết luận về augmentation trong protocol hiện tại.
- Accuracy ±12/±18 của A2 không cao hơn A1 dù MAE/RMSE tốt hơn; không được tuyên bố A2 thắng mọi metric.
- Ba run mới dùng một seed. P5 sẽ xác nhận độ ổn định nhiều seed cho các cấu hình kiến trúc tốt nhất.
- Validation MAE 6,185 không được so trực tiếp với test MAE 3,68 của Bram.
- Chưa được chạy test RSNA hoặc bộ artifact-reduced để chọn mô hình.

## 6. Bàn giao

```text
Phase: P2
Mục tiêu: Chạy A0–A2 và khóa augmentation
Kết quả: PASS
Best run: P2_A2_LIGHT_FLIP_CONVNEXT_TINY_SEED42
Best validation MAE/RMSE: 6,1848 / 8,4865 tháng
Best epoch: 12
Data hash: Train 7328667e...f5285; Validation f650a204...21631
Quyết định: Khóa augmentation A2 có flip
Bước tiếp theo: P3 – B0 giữ A2, B1 thêm full-hand preprocessing
Test set: KHÔNG chạm nhãn/metric
```
