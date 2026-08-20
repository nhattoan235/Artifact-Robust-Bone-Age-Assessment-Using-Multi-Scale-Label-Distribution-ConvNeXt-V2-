# Dự án: Cải thiện mô hình dự đoán tuổi xương (Bone Age Estimation)

## 1. Bối cảnh & Mục tiêu

**Bài báo gốc:** Matsuoka et al., *"Evaluating the Clinical Impact of Generative Inpainting on Bone Age Estimation"*, arXiv:2511.23066, 28/11/2025.

Bài báo gốc **không** nhằm tối ưu mô hình bone age — trọng tâm của nó là chứng minh rằng inpainting bằng gpt-image-1 phá vỡ các đặc trưng giải phẫu tinh vi (đặc biệt vùng carpus và khớp bàn–ngón) mà mô hình downstream dùng để suy luận tuổi xương. Tuy nhiên, mô hình downstream mà họ dùng làm công cụ đánh giá — một ensemble ResNet50 theo giới tính — chính là điểm khởi đầu (baseline) cho dự án này.

**Mục tiêu của dự án:** xây dựng hoặc kết hợp các kỹ thuật/mô hình đã có trong literature để đạt độ chính xác dự đoán tuổi xương **tốt hơn** baseline của tác giả trên cùng bộ dữ liệu RSNA Bone Age.

### Baseline cần vượt qua (từ bài báo gốc)

| Chỉ số | Ảnh gốc (baseline) |
|---|---|
| MAE (bone age) | **6.26 tháng** (95% CI 5.60–6.89) |
| RMSE (bone age) | 7.79 tháng (95% CI 7.02–8.65) |
| AUC (gender) | 0.956 (95% CI 0.925–0.979) |

**Kiến trúc/pipeline baseline:**
- Ensemble ResNet50 theo giới tính, 5-fold CV, PyTorch 2.7.1
- Pretrained ImageNet
- Input 320×320, RGB hoá qua channel duplication
- Preprocessing: MONAI + Albumentations, foreground cropping tự động (nonzero-pixel bbox), intensity normalization
- Augmentation: affine (rotation ±15°, translation, scaling), contrast/brightness, sharpening
- Loss: Smooth L1 (Huber)
- Optimizer: Adam, cosine annealing LR, gradient accumulation (10 steps), 50 epochs
- Gender embedding: qua auxiliary linear layer (optional)

### Mốc tham chiếu ngành (để biết "tốt" là bao nhiêu)
- 16 Bit (RSNA 2017 winner): MAD **4.265 tháng** trên test set gốc của cuộc thi
- BoNet (Escobar et al., MICCAI 2019): MAE **6.3 tháng** trên bộ RHPE (tự công bố)
- Model đánh giá lại 16Bit trên external test set (Beheshtian et al. 2023): MAD **6.8–6.9 tháng**

→ Mục tiêu thực tế: đưa MAE từ 6.26 xuống **dưới 5.5 tháng** ở các giai đoạn đầu, hướng tới vùng **4.0–4.5 tháng** nếu triển khai đầy đủ pipeline ROI/landmark.

---

## 2. Lộ trình theo giai đoạn

> Mỗi giai đoạn là một thử nghiệm độc lập, có thể đo lường được so với baseline 6.26 tháng. Kết quả mỗi giai đoạn được ghi vào `CHANGELOG.md`.

### Giai đoạn 0 — Thiết lập & tái lập baseline
**Mục tiêu:** tái tạo pipeline gốc của tác giả để có con số baseline đáng tin cậy trên cùng môi trường/hardware của mình trước khi so sánh bất kỳ cải tiến nào.
**Việc cần làm:**
- Tải RSNA Bone Age dataset (train 12,611 / test 200)
- Dựng lại ResNet50 ensemble (sex-specific, 5-fold) đúng theo mô tả Methods của bài báo
- Xác nhận MAE ≈ 6.26 tháng trên test set gốc
**Nguồn:** Halabi et al., *"The RSNA Pediatric Bone Age Machine Learning Challenge"*, Radiology 290(2), 2019. https://pubs.rsna.org/doi/10.1148/radiol.2018180736

### Giai đoạn 1 — Quick wins: Ensemble đa kiến trúc + Pretrained domain y tế
**Mục tiêu:** cải thiện nhanh, chi phí thấp, không đổi cấu trúc pipeline lớn.
**Kỹ thuật:**
1. Thay 5-fold cùng ResNet50 bằng ensemble đa kiến trúc: ResNet50 + EfficientNet-B4 + DenseNet121 (average hoặc stacking bằng meta-learner tuyến tính)
2. Đổi trọng số khởi tạo từ ImageNet sang RadImageNet
**Nguồn:**
- Tan, M., Le, Q., *"EfficientNet: Rethinking Model Scaling for CNNs"*, ICML 2019 (Google Brain)
- Mei, X., Yang, Y., Deyer, T., et al., *"RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"*, Radiology: AI, 2022. https://pubs.rsna.org/doi/10.1148/ryai.210315 — cải thiện AUC 0.9–9.4% so với ImageNet trên 8 bài toán y tế độc lập
- Trọng số RadImageNet: https://github.com/BMEII-AI/RadImageNet

### Giai đoạn 2 — Đổi hàm loss: Ordinal regression (CORAL/CORN)
**Mục tiêu:** khai thác tính thứ tự tự nhiên của tuổi xương (các giai đoạn cốt hoá) thay vì coi đó là hồi quy liên tục thuần tuý.
**Kỹ thuật:** thay Huber loss bằng CORAL hoặc CORN loss — chỉ cần đổi output head, tương thích với bất kỳ backbone nào ở Giai đoạn 1.
**Nguồn:**
- Cao, W., Mirjalili, V., Raschka, S., *"Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation"* (CORAL), Pattern Recognition Letters 140, 2020. https://arxiv.org/abs/1901.07884
- Shi, X., Cao, W., Raschka, S., *"Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities"* (CORN), arXiv:2111.08851, 2021
- Code sẵn dùng: https://github.com/Raschka-research-group/coral-pytorch

### Giai đoạn 3 — Pipeline ROI/landmark (thay đổi kiến trúc lớn nhất)
**Mục tiêu:** thay foreground-cropping thô hiện tại bằng module định vị đúng vùng giải phẫu quan trọng (carpus, khớp bàn–ngón) trước khi đưa vào backbone hồi quy — đây là thay đổi mang lại cải thiện lớn nhất theo toàn bộ literature bone-age.
**Kỹ thuật (chọn 1 trong 2 hướng):**
- **Có annotation landmark:** BoNet-style — module detect tay + pose estimation, theo Escobar et al.
- **Không cần annotation thêm:** attention tự học vùng phân biệt, theo Chen et al.
**Nguồn:**
- Escobar, M. et al., *"Hand Pose Estimation for Pediatric Bone Age Assessment"* (BoNet), MICCAI 2019. https://link.springer.com/chapter/10.1007/978-3-030-32226-7_59 — MAE 6.3 tháng trên RHPE
- Chen, C. et al., *"Attention-Guided Discriminative Region Localization and Label Distribution Learning for Bone Age Assessment"*, arXiv:2006.00202
- Wang, D. et al., *"Improve Bone Age Assessment by Learning from Anatomical Local Regions"* (ALA-Net), arXiv:2005.13452, 2020
- Iglovikov, V. et al., *"Paediatric Bone Age Assessment Using Deep CNNs"*, in Deep Learning in Medical Image Analysis, Springer 2018

### Giai đoạn 4 — Multi-task learning + FiLM conditioning theo giới tính
**Mục tiêu:** gộp 2 pipeline tách biệt (bone age riêng, gender riêng) thành 1 backbone dùng chung, 2 head, với giới tính là biến điều kiện thay vì phải train 2 ensemble riêng theo giới như hiện tại.
**Kỹ thuật:** shared backbone (từ Giai đoạn 1 hoặc 3) + FiLM layer để điều biến đặc trưng theo giới tính + multi-task loss (bone age regression + gender classification).
**Nguồn:** Perez, E. et al., *"FiLM: Visual Reasoning with a General Conditioning Layer"*, AAAI 2018 (Element AI/MILA)

### Giai đoạn 5 — Uncertainty quantification + Test-time augmentation
**Mục tiêu:** không chỉ cải thiện điểm dự đoán trung bình mà còn cung cấp khoảng tin cậy — giá trị lâm sàng cao, đồng thời giúp phát hiện các ca bất thường (liên hệ trực tiếp tới phát hiện của bài báo gốc về việc mô hình "tự tin sai" trên ảnh bị nhiễu).
**Kỹ thuật:** MC Dropout hoặc Deep Ensembles (đã có sẵn từ Giai đoạn 1) + test-time augmentation (multi-crop, flip, TTA-averaging).
**Nguồn:**
- Gal, Y., Ghahramani, Z., *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"*, ICML 2016 (University of Cambridge)
- Lakshminarayanan, B., Pritzel, A., Blundell, C., *"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"*, NeurIPS 2017 (DeepMind)

### Giai đoạn 6 — Độ phân giải đầu vào & multi-scale
**Mục tiêu:** nâng từ 320×320 lên độ phân giải cao hơn hoặc kiến trúc two-stream (ảnh toàn cảnh + crop độ phân giải cao vùng cổ tay) để giữ chi tiết sụn tăng trưởng nhỏ.
**Việc cần làm:** thử nghiệm 512×512 hoặc 640×640, so sánh chi phí tính toán vs. cải thiện MAE; nếu dùng two-stream, kết hợp với module ROI ở Giai đoạn 3.

### Giai đoạn 7 — Đánh giá tổng hợp & kiểm tra độ bền vững
**Mục tiêu:** so sánh toàn diện mô hình cuối cùng với baseline 6.26 tháng, đồng thời — theo đúng tinh thần bài báo gốc — kiểm tra độ bền của mô hình mới trước nhiễu loạn kiểu inpainting/artifact để xem liệu các cải tiến có làm mô hình ổn định hơn không.
**Việc cần làm:** báo cáo MAE/RMSE/AUC có CI 95%, so sánh với baseline và với các mốc SOTA (4.265–6.3 tháng), viết kết luận.

---

## 3. Trạng thái hiện tại

| Giai đoạn | Trạng thái | MAE đạt được | Ghi chú |
|---|---|---|---|
| 0 — Tái lập baseline | Chưa bắt đầu | — | — |
| 1 — Ensemble đa kiến trúc + RadImageNet | Code xong, chờ chạy | — | RadImageNet chỉ áp dụng cho ResNet50/DenseNet121; EfficientNet-B4 giữ ImageNet (xem CHANGELOG) |
| 2 — CORAL/CORN loss | Chưa bắt đầu | — | — |
| 3 — ROI/landmark pipeline | Chưa bắt đầu | — | — |
| 4 — Multi-task + FiLM | Chưa bắt đầu | — | — |
| 5 — Uncertainty + TTA | Chưa bắt đầu | — | — |
| 6 — Độ phân giải cao hơn | Chưa bắt đầu | — | — |
| 7 — Đánh giá tổng hợp | Chưa bắt đầu | — | — |

Chi tiết từng lần chạy thử nghiệm và kết quả cụ thể được ghi trong `CHANGELOG.md`.

## 4. Tài liệu tham khảo đầy đủ

1. Matsuoka, F.A. et al. *"Evaluating the Clinical Impact of Generative Inpainting on Bone Age Estimation"*. arXiv:2511.23066, 2025. (Bài báo gốc / baseline)
2. Halabi, S.S. et al. *"The RSNA Pediatric Bone Age Machine Learning Challenge"*. Radiology 290(2):498–503, 2019.
3. Beheshtian, E. et al. *"Generalizability and Bias in a Deep Learning Pediatric Bone Age Prediction Model Using Hand Radiographs"*. Radiology 306(2):e220505, 2023.
4. Tan, M., Le, Q. *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"*. ICML 2019.
5. Liu, Z. et al. *"A ConvNet for the 2020s"* (ConvNeXt). CVPR 2022.
6. Mei, X. et al. *"RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"*. Radiology: AI, 2022.
7. Cao, W., Mirjalili, V., Raschka, S. *"Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation"* (CORAL). Pattern Recognition Letters 140, 2020.
8. Shi, X., Cao, W., Raschka, S. *"Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities"* (CORN). arXiv:2111.08851, 2021.
9. Escobar, M. et al. *"Hand Pose Estimation for Pediatric Bone Age Assessment"* (BoNet). MICCAI 2019.
10. Chen, C. et al. *"Attention-Guided Discriminative Region Localization and Label Distribution Learning for Bone Age Assessment"*. arXiv:2006.00202.
11. Wang, D. et al. *"Improve Bone Age Assessment by Learning from Anatomical Local Regions"* (ALA-Net). arXiv:2005.13452, 2020.
12. Iglovikov, V. et al. *"Paediatric Bone Age Assessment Using Deep Convolutional Neural Networks"*. In: Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support, Springer, 2018.
13. Perez, E. et al. *"FiLM: Visual Reasoning with a General Conditioning Layer"*. AAAI 2018.
14. Gal, Y., Ghahramani, Z. *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"*. ICML 2016.
15. Lakshminarayanan, B., Pritzel, A., Blundell, C. *"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"*. NeurIPS 2017.
