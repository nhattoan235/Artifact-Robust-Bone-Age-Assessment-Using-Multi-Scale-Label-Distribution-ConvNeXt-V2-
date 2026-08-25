# Bàn Giao Công Việc — Dự Án Cải Thiện Mô Hình Dự Đoán Tuổi Xương

*Cập nhật lần cuối: theo tiến trình làm việc gần nhất trong phiên trao đổi*

---

## 1. Tổng quan dự án

**Mục tiêu hiện tại:** tự đề xuất và cải tiến mô hình dự đoán tuổi xương (bone age estimation) trên bộ dữ liệu **RSNA Pediatric Bone Age Challenge 2017** (train 12.611 / validation 1.425 / test 200 ảnh), với mốc cần vượt qua ban đầu là **MAE 6.26 tháng** (Matsuoka et al., arXiv:2511.23066), và mốc tham chiếu ngành cao hơn vừa xác định là **MAE ~3.87–4.2 tháng** (Deeplasia, Springer 2023).

**Lưu ý quan trọng:** đề tài đã **đổi hướng giữa chừng**. Ban đầu tập trung vào đánh giá tác động của GPT-image-1 generative inpainting lên độ chính xác bone age (dựa trên bài Matsuoka). Sau đó chuyển hẳn sang **tự đề xuất cải tiến mô hình bone age**, dùng lộ trình 8 giai đoạn (0–7) tham khảo từ một dự án tương tự của bạn cùng nhóm — bạn đó đang phụ trách Giai đoạn 1 (ensemble đa kiến trúc + RadImageNet), nên phần việc của bạn tập trung vào các giai đoạn còn lại, hiện đang ở **Giai đoạn 3 → 3.1 → 6**.

---

## 2. Dòng thời gian tóm tắt

### Giai đoạn trước khi đổi hướng (bối cảnh, không còn là trọng tâm)
- Xây dựng model ResNet50 bone-age (v1, v2, v2.1) làm công cụ đánh giá (downstream evaluator) + nguồn Grad-CAM cho 3 hướng cải tiến pipeline inpainting GPT-image-1 (Hướng B — Saliency-Guided Masking, Hướng D — Frequency Blending, Hướng E — Landmark QA).
- Kết quả: v1 test MAE 6.91 tháng; v2 (thêm progressive unfreezing + regularization mạnh) val MAE 8.06 — **tệ hơn v1** do over-regularization; v2.1 (giảm bớt regularization) đang cải thiện tốt hơn khi đổi hướng đề tài (khoảng 7.86 ở epoch 29, chưa có kết luận cuối).
- Có xây thêm 1 notebook thử nghiệm ControlNet Canny + Stable Diffusion Inpainting (hướng thay thế GPT-image-1 bằng model mã nguồn mở) — **chưa chạy thử, để tham khảo nếu quay lại hướng này**.

### Giai đoạn 3 — ROI/Landmark (Global-Local Attention Network)
- Thiết kế kiến trúc 2 nhánh: Global stream (toàn ảnh) + Local stream (crop vùng ROI qua Spatial Transformer Network — STN, tự học vị trí, không cần landmark annotation thủ công).
- **Phát hiện quan trọng:** đo chính xác toạ độ khung crop trên 6 ảnh validation khác nhau → sai lệch vị trí **chưa tới 2%** dù ảnh rất khác nhau → **STN bị "collapse"**, học ra 1 vị trí gần như cố định thay vì thích ứng theo từng ảnh. Best Val MAE = 6.639 tháng nhưng **không đáng tin** vì cơ chế cốt lõi không hoạt động đúng ý đồ.

### Giai đoạn 3.1 — Sửa lỗi collapse + Ablation kiểm chứng
- Thêm 3 kỹ thuật chống collapse: noise injection (giảm dần theo epoch), discriminative learning rate cho localization head, diversity regularization (phạt nếu độ lệch chuẩn vị trí trong batch quá thấp).
- Thiết kế **thí nghiệm đối chứng (ablation)**: `USE_LEARNED_LOCALIZATION = True` (học vị trí) vs `False` (crop cố định tại đúng vị trí đã collapse).
- **Kết quả:** `fixed` (6.8233 tháng) **thắng** `learned` (6.9596 tháng). Biểu đồ độ đa dạng vị trí cho thấy: đa dạng chỉ tồn tại trong lúc còn nhiễu nhân tạo, ngay khi tắt nhiễu (annealing xong) model lập tức quay về đúng 1 điểm cố định — **collapse tái diễn dù đã có 3 lớp phòng vệ**.
- **Kết luận khoa học:** với kiến trúc học "mù" từ tín hiệu loss (không có giám sát trực tiếp), cơ chế ROI/Landmark tự học **không mang lại lợi ích thực sự** trên bộ dữ liệu này (RSNA vốn đã được foreground-crop khá đồng đều).

### Chuyển đổi hạ tầng chạy (Kaggle ↔ Local)
- Kaggle hết giờ chạy giữa chừng → chuyển code sang chạy local (script Python độc lập, có checkpoint resume, an toàn cho Windows/`num_workers`).
- Sang tuần mới, quota Kaggle reset → chuyển ngược lại thành notebook Kaggle, giữ nguyên các cải tiến (resume checkpoint, chống collapse).

### Sửa lỗi split dữ liệu — dùng đúng chuẩn RSNA gốc
- Phát hiện: trước đó tự chia 90/10 từ 12.611 ảnh train, **chưa dùng** tập validation chính thức (1.425 ảnh) mà challenge RSNA công bố riêng.
- Chuyển sang dùng đúng 3 tập tách biệt (train 12.611 / validation 1.425 / test 200) — khớp với cách hầu hết literature làm, tận dụng trọn vẹn dữ liệu train.
- Gặp và xử lý các vấn đề thực tế: dataset Kaggle username/tên bị gõ sai, thư mục ảnh lồng cấp trùng tên, **validation set bị chia làm 2 thư mục con** (`-1`/`-2`, cần gộp lại), **CSV validation dùng tên cột khác CSV train** (`Image ID`/`Bone Age (months)` thay vì `id`/`boneage`) — đã viết code tự nhận diện và map tên cột.

### Đọc bài báo tham khảo thứ 2 — Deeplasia (Springer, 2023)
- MAD 3.87 tháng (ensemble 3 model), ~4.2 tháng (model đơn) — trên đúng test set 200 ảnh RSNA.
- **Xác nhận độc lập** kết luận ở Giai đoạn 3.1: Deeplasia dùng kiến trúc *prior-free* (không ROI, không landmark), và trong Discussion họ khẳng định cách này mạnh ngang các phương pháp có prior.
- Xác định khoảng cách 6.8 (mình) → 3.87 (họ) đến từ: (1) độ phân giải cao hơn (512→1024 vs 320), (2) model segmentation tay chuyên biệt (mình chỉ threshold-crop thô), (3) backbone EfficientNet (mình dùng ResNet50), (4) ensemble 3 model (mình 1 model).

### Giai đoạn 6 — Tăng độ phân giải đầu vào
- Đơn giản hoá kiến trúc: **bỏ hẳn nhánh Local/STN** (theo kết luận ablation ở Giai đoạn 3.1), quay về 1 backbone đơn (ResNet50 + gender embedding) — nhẹ hơn, giải phóng VRAM.
- Thêm Mixed Precision (AMP) để giảm VRAM/tăng tốc khi lên độ phân giải cao.
- Test `RESOLUTION=320` và `RESOLUTION=512`.
- **Kết quả:**
  - 320px: Best Val MAE = **6.6208** tháng — tốt hơn Stage 3.1 `fixed` (6.82), xác nhận thêm lần nữa việc bỏ nhánh Local/STN là đúng.
  - 512px: Best Val MAE = **6.7798** tháng — **KHÔNG tốt hơn** 320px, trái kỳ vọng từ Deeplasia.
- **Kết luận:** độ phân giải một mình không đủ để cải thiện — nhiều khả năng do bước tách nền/segmentation còn thô (threshold-crop đơn giản), phóng to ảnh chỉ phóng to luôn cả nhiễu nền/viền/nhãn, không tăng tỷ lệ tín hiệu hữu ích.

---

## 3. Bảng tổng hợp toàn bộ kết quả thực nghiệm

| Model / Giai đoạn | Kiến trúc | Split dữ liệu | Best Val MAE (tháng) | Ghi chú |
|---|---|---|---|---|
| v1 (trước đổi hướng) | ResNet50 đơn | tự chia | 6.91 (test) | Baseline cũ, dùng cho Hướng B/D/E |
| v2 | + progressive unfreeze, reg. mạnh | tự chia | 8.06 | Tệ hơn v1 — over-regularization |
| v2.1 | reg. vừa phải hơn | tự chia | ~7.86 (chưa chốt) | Dở dang khi đổi hướng đề tài |
| Stage 3 | Global-Local + STN | tự chia 90/10 | 6.639 | **Không đáng tin** — STN collapse |
| Stage 3.1 `learned` | + chống collapse | tự chia 90/10 | 6.9596 | Vẫn collapse khi hết nhiễu |
| Stage 3.1 `fixed` | Crop cố định (đối chứng) | tự chia 90/10 | **6.8233** | Thắng bản `learned` |
| Stage 6 `res320` | Single-stream (bỏ STN) | **split gốc RSNA** | **6.6208** | Tốt nhất tính đến hiện tại |
| Stage 6 `res512` | Single-stream, 512px | split gốc RSNA | 6.7798 | Không cải thiện so với 320px |

**Mốc tham chiếu:**

| Nguồn | MAE/MAD (tháng) |
|---|---|
| Baseline Matsuoka (bài báo gốc dự án) | 6.26 |
| BoNet (Escobar et al., MICCAI 2019) | 6.3 |
| 16Bit — vô địch RSNA Challenge 2017 | 4.265 |
| Deeplasia — model đơn (EfficientNet) | ~4.2 |
| **Deeplasia — ensemble 3 model** | **3.87** |

---

## 4. Bài học / phát hiện quan trọng cần nhớ

1. **STN/ROI tự học không giúp ích** trên bộ dữ liệu này khi học hoàn toàn "mù" từ loss cuối — đã kiểm chứng 2 lần độc lập (ablation thực nghiệm + đối chiếu với Deeplasia). Nếu muốn thử lại hướng ROI trong tương lai, cần cho localization head một tín hiệu giám sát yếu ban đầu (vd Grad-CAM từ model bone-age cũ) thay vì học từ đầu.
2. **Độ phân giải một mình không đủ** — cần đi kèm cải thiện tách nền/segmentation hoặc đổi backbone mới phát huy tác dụng (dựa theo kinh nghiệm Deeplasia).
3. **Luôn dùng đúng split gốc RSNA** (12.611/1.425/200) thay vì tự chia — quan trọng cho khả năng so sánh với literature.
4. **Luôn kiểm tra bằng số liệu, không chỉ nhìn ảnh** — STN collapse ban đầu tưởng ổn khi nhìn ảnh bằng mắt, chỉ phát hiện được khi đo toạ độ khung crop chính xác bằng code.
5. **Thiết kế ablation (đối chứng) rất hiệu quả** để kiểm chứng một ý tưởng kiến trúc có thực sự đóng góp hay không — nên tiếp tục dùng cách này cho các thử nghiệm tiếp theo (vd: EfficientNet vs ResNet50, có/không segmentation).

---

## 5. Vấn đề kỹ thuật đã gặp và cách xử lý (tránh lặp lại)

| Vấn đề | Cách xử lý |
|---|---|
| Lỗi hiển thị `tqdm.notebook` widget trên Kaggle khi chạy dài | Chuyển sang `from tqdm import tqdm` (bản text thuần) |
| Kaggle hết giờ chạy giữa chừng, mất tiến trình | Thêm checkpoint resume: lưu đầy đủ state (model/optimizer/scheduler + lịch sử) mỗi epoch, cờ `RESUME=True` để tiếp tục |
| `DataLoader(num_workers>0)` lỗi vòng lặp vô hạn trên Windows (khi chạy local) | Bọc toàn bộ code trong `if __name__ == "__main__":` |
| Validation set bị chia 2 thư mục con (`-1`/`-2`) | Dataset class nhận `list` nhiều đường dẫn, tự gộp map id→path |
| CSV validation dùng tên cột khác CSV train (`Image ID` vs `id`, `Bone Age (months)` vs `boneage`) | Hàm chuẩn hoá tên cột tự động nhận diện các biến thể phổ biến |
| Tên dataset/username Kaggle gõ sai trong code mẫu ban đầu | Luôn đối chiếu với ảnh chụp màn hình cây thư mục thật trước khi chạy dài |

---

## 6. Danh sách file đã tạo (deliverables)

| File | Mục đích | Trạng thái |
|---|---|---|
| `train_boneage_v2_1_kaggle.ipynb` | v2.1 — bối cảnh cũ (trước đổi hướng) | Dở dang, không còn ưu tiên |
| `controlnet_canny_inpaint_kaggle.ipynb` | Thử nghiệm SD+ControlNet thay GPT-image-1 | Chưa chạy, để tham khảo nếu quay lại hướng inpainting |
| `stage3_global_local_attention_kaggle.ipynb` | Giai đoạn 3 gốc (bị STN collapse) | Đã có kết luận, không dùng tiếp |
| `stage3_1_fix_collapse_kaggle.ipynb` | Giai đoạn 3.1, split tự chia 90/10 | Đã có kết luận (ablation) |
| `train_stage3_1_local.py` + `README_local.md` + `requirements.txt` | Bản chạy local (Windows/Linux, GPU NVIDIA) | Dự phòng khi Kaggle hết giờ |
| `stage3_1_kaggle_official_split.ipynb` | Giai đoạn 3.1 + split gốc RSNA (bản user tự sửa path đúng) | Bản tham chiếu đường dẫn chuẩn |
| `stage6_resolution_kaggle.ipynb` | Giai đoạn 6 — độ phân giải, kiến trúc đơn giản hoá | **Đang dùng, active** |

---

## 7. Trạng thái theo lộ trình 8 giai đoạn gốc

| Giai đoạn | Nội dung | Trạng thái | Người phụ trách |
|---|---|---|---|
| 0 | Tái lập baseline (MAE 6.26) | **Chưa làm — cần làm sớm** để có căn cứ so sánh hợp lệ | Bạn |
| 1 | Ensemble đa kiến trúc + RadImageNet | Đang làm | Bạn cùng nhóm |
| 2 | CORAL/CORN ordinal loss | Chưa làm | — |
| 3 | ROI/Landmark | **Đã thử, đã bác bỏ bằng ablation** — không tiếp tục đầu tư | Bạn |
| 3.1 | Sửa lỗi + kiểm chứng Giai đoạn 3 | Hoàn tất, có kết luận rõ ràng | Bạn |
| 4 | Multi-task + FiLM conditioning | Chưa làm (phụ thuộc kết quả Giai đoạn 1) | — |
| 5 | Uncertainty + TTA | Chưa làm (phụ thuộc Giai đoạn 1) | — |
| 6 | Độ phân giải & multi-scale | **Đang làm** — 320px xong, 512px xong (chưa đạt kỳ vọng) | Bạn |
| 7 | Đánh giá tổng hợp cuối cùng | Chưa làm (cần các giai đoạn khác xong trước) | — |

**Lưu ý quan trọng:** Giai đoạn 0 (tái lập baseline) vẫn **chưa được thực hiện** trong suốt quá trình — mọi so sánh MAE ở trên hiện chỉ đối chiếu với con số 6.26 công bố trong bài báo Matsuoka, chưa có baseline tự tái lập trên đúng môi trường/pipeline của nhóm để đảm bảo so sánh hoàn toàn hợp lệ. Nên ưu tiên làm sớm.

---

## 8. Quyết định đang chờ / bước tiếp theo

Sau kết quả Giai đoạn 6 (độ phân giải không giúp ích như kỳ vọng), có 3 hướng đang cân nhắc, **chưa chốt**:

1. **Cải thiện tách nền/segmentation bàn tay** — thay `foreground_crop()` thô hiện tại bằng phương pháp chính xác hơn (vd: một model segmentation nhẹ, hoặc kỹ thuật threshold thích ứng tốt hơn), theo đúng hướng Deeplasia đã làm.
2. **Đổi backbone sang EfficientNet** — giữ nguyên 320px để cô lập biến, kiểm chứng riêng tác động của backbone (đúng phương pháp ablation đã dùng xuyên suốt dự án).
3. **Thử lại 512px với patience/epoch cao hơn** — kiểm tra xem model có thực sự hội tụ chưa, hay bị early-stop hơi sớm.

*(Khuyến nghị cá nhân: nên ưu tiên hướng 1 hoặc 2 trước, vì hướng 3 ít khả năng giải quyết được gốc rễ vấn đề đã phân tích ở mục 2.)*

---

## 9. Tài liệu tham khảo chính đã dùng

1. Matsuoka, F.A. et al. *"Evaluating the Clinical Impact of Generative Inpainting on Bone Age Estimation"*. arXiv:2511.23066, 2025. — baseline MAE 6.26
2. Halabi, S.S. et al. *"The RSNA Pediatric Bone Age Machine Learning Challenge"*. Radiology 290(2), 2019. — nguồn split gốc 12.611/1.425/200
3. Chen, C. et al. *"Attention-Guided Discriminative Region Localization and Label Distribution Learning for Bone Age Assessment"*. arXiv:2006.00202, 2020. — căn cứ lý thuyết ban đầu cho Giai đoạn 3
4. Wang, D. et al. *"Improve Bone Age Assessment by Learning from Anatomical Local Regions"* (ALA-Net). arXiv:2005.13452, 2020.
5. Jaderberg, M. et al. *"Spatial Transformer Networks"*. NeurIPS 2015. — cơ chế STN dùng trong Giai đoạn 3/3.1
6. Fu, J. et al. *"Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition"* (RA-CNN). CVPR 2017. — ý tưởng nền cho kiến trúc global-local
7. **Deeplasia** — *"Deep learning for bone age assessment validated on skeletal dysplasias"*. Pediatric Radiology, Springer, 2023. https://link.springer.com/article/10.1007/s00247-023-05789-1 — mốc tham chiếu mới (MAD 3.87), xác nhận hướng prior-free
