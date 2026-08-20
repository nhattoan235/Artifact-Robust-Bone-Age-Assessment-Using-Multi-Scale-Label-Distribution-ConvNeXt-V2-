# Giai đoạn 1 — Ensemble đa kiến trúc + RadImageNet

Triển khai đúng 2 kỹ thuật mô tả trong `PROJECT_CONTEXT.md`:
1. Ensemble ResNet50 + EfficientNet-B4 + DenseNet121 (thay vì 5-fold cùng ResNet50)
2. Trọng số khởi tạo RadImageNet thay ImageNet — **chỉ áp dụng được cho ResNet50 và
   DenseNet121** (xem giới hạn bên dưới)

## 1. Cài đặt

```bash
pip install torch torchvision albumentations scikit-learn pandas opencv-python-headless --break-system-packages
```

## 2. Chuẩn bị dữ liệu

Cấu trúc mong đợi trong `config.py::data_root`:
```
data_root/
  boneage-training-dataset.csv     # cột: id, boneage, male
  boneage-training-dataset/{id}.png
```
Sửa `config.py` (`data_root`, `output_dir`) theo đường dẫn thật của bạn.

## 3. Trọng số RadImageNet — giới hạn quan trọng cần ghi vào khóa luận

Repo chính thức **https://github.com/BMEII-AI/RadImageNet** chỉ phát hành trọng số
PyTorch (.pt) cho **ResNet50, DenseNet121, InceptionResNetV2, InceptionV3** —
**không có EfficientNet-B4**. Vì vậy trong Giai đoạn 1:
- ResNet50, DenseNet121 → RadImageNet pretrained (cần tải thủ công, repo yêu cầu
  điền form xin quyền sử dụng qua Google Drive link trong README của họ)
- EfficientNet-B4 → giữ ImageNet pretrained (torchvision), là lựa chọn hợp lý duy
  nhất với nguồn public hiện có, không phải sai sót thiết kế

Sau khi tải, đặt đường dẫn vào `config.py`:
```python
radimagenet_resnet50_path = "/path/to/RadImageNet_ResNet50.pt"
radimagenet_densenet121_path = "/path/to/RadImageNet_DenseNet121.pt"
```
Để chạy nhánh không dùng RadImageNet (ví dụ để so sánh ablation), đặt các giá trị
trên về `None` — code sẽ tự fallback sang ImageNet.

> Nếu muốn EfficientNet-B4 cũng có trọng số pretrained y tế, phương án thay thế là
> RadiologyNET (Napravnik et al., 2025, Scientific Reports) — dataset y tế khác,
> đã công bố EfficientNetB3/B4 pretrained và **đã test trực tiếp trên RSNA Bone Age**.
> Đây là hướng mở rộng hợp lệ nhưng KHÔNG nằm trong phạm vi "RadImageNet" như
> `PROJECT_CONTEXT.md` đã chỉ định — nên nêu là lựa chọn thay thế trong phần Discussion
> nếu bạn muốn dùng, chứ không âm thầm thay nguồn.

## 4. Chạy trên Google Colab (khuyến nghị cho GPU VRAM nhỏ như RTX 4050 6GB local)

**Bắt buộc lưu `output_dir` trên Google Drive** (không lưu ở `/content/` local của
Colab) — vì đó là nơi checkpoint + `progress.json` + `oof_predictions.csv` được ghi,
cần tồn tại xuyên suốt nhiều session Colab bị ngắt.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Sửa `config.py`:
```python
data_root  = "/content/drive/MyDrive/bone_age/rsna_boneage"
output_dir = "/content/drive/MyDrive/bone_age/phase1_outputs"
```

Chạy:
```bash
python train.py
```

**Cơ chế resume đã tích hợp sẵn** — nếu Colab ngắt kết nối/hết giờ giữa chừng, chỉ
cần chạy lại `python train.py` (session Colab mới, mount lại Drive), script sẽ:
1. Đọc `progress.json` trong `output_dir` để biết (kiến trúc, giới tính, fold) nào
   đã xong hoàn toàn -> bỏ qua, không train lại từ đầu.
2. Với fold đang train dở, đọc checkpoint `_last.pt` (lưu sau mỗi epoch) để tiếp tục
   đúng từ epoch tiếp theo, không mất tiến độ trong fold đó.

**VRAM 6GB:** đã bật sẵn AMP (`use_amp=True`) và batch size riêng cho từng kiến trúc
trong `config.py::per_arch_batch_size` (EfficientNet-B4 = 6, thấp hơn ResNet50/DenseNet121
vì tốn VRAM hơn). `grad_accum_steps` được tự tính lại theo từng kiến trúc để effective
batch size (batch × accumulation) vẫn giữ khớp baseline (~160) dù batch vật lý khác nhau.
Nếu vẫn OOM, giảm tiếp `per_arch_batch_size` (script tự bù bằng cách tăng accumulation).

Output tại `output_dir`:
- `{arch}_{sex_tag}_fold{n}.pt` — checkpoint từng fold/kiến trúc/giới tính
- `oof_predictions.csv` — dự đoán out-of-fold, dùng để fit stacking ensemble
- `phase1_report.json` — MAE/RMSE cuối cùng, so với baseline 6.26/7.79 tháng,
  và trọng số stacking của từng kiến trúc

## 5. Sau khi có kết quả

Điền số liệu vào `CHANGELOG.md` theo đúng format entry đã định nghĩa sẵn trong file
đó (mục "Giai đoạn 1"), thay các ô để trống bằng số liệu thật từ `phase1_report.json`.
KHÔNG copy số liệu ước tính/kỳ vọng vào CHANGELOG — chỉ ghi số đã chạy thật.



### Đường dẫn git của tác giả
Truy cập: https://github.com/BMEII-AI/RadImageNet.git

### Đường dẫn lấy file trọng số
https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing
