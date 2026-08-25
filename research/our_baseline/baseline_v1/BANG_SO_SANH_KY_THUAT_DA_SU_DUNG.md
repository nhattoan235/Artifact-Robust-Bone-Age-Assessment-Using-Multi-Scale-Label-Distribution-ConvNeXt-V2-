# Bảng so sánh các kỹ thuật đã sử dụng trong nghiên cứu bone-age

Ngày cập nhật: 2026-08-24

## Quy ước đọc kết quả

- MAE thấp hơn là tốt hơn.
- Chỉ so sánh trực tiếp các kết quả có cùng protocol.
- Kết quả test 200 ảnh chỉ là exploratory, không dùng để chọn kỹ thuật.
- `Friend` nghĩa là kết quả từ repo/thí nghiệm của bạn cùng nhóm; chưa đồng
  nghĩa là kỹ thuật đó đã được tái lập đầy đủ trên baseline riêng của chúng ta.

## 1. Các kỹ thuật đã chạy và có kết quả

| Kỹ thuật / thành phần | Nguồn | Cách áp dụng | Kết quả đo được | Đánh giá |
|---|---|---|---|---|
| Official train/validation | Pipeline baseline v1 | Train 12.611 ảnh, validate 1.425 ảnh | P7 reference MAE **6.471143** | Mốc baseline cũ trên official validation |
| ConvNeXt-Tiny pretrained ImageNet | Baseline v1; friend repo; các bài báo tham khảo | Ảnh grayscale lặp thành 3 kênh, backbone ConvNeXt-Tiny pretrained, global pooling | Là backbone của P7 reference, A2 và EXP006 | Nền tảng ổn định; chưa đủ để đạt 3.5 nếu dùng model đơn |
| Sex embedding + image feature | Baseline v1: 32 chiều; friend P7: 16 chiều | Mã hóa giới tính, nối với image feature trước regression head | Được dùng xuyên suốt P7; không có ablation riêng | Có cơ sở sinh học, nên giữ |
| Direct regression theo tháng | Baseline v1 và friend P7 | Head cuối trả về một số thực bone age theo tháng | P7 reference **6.471143**; friend P7 OOF **6.316691** | Tốt làm control; P7 thực tế chưa kích hoạt distribution head |
| Smooth L1 loss | Baseline v1 và friend repo | Regression loss với beta 3 tháng trong friend P7; baseline cũ dùng Smooth L1 | Được dùng trong mọi run chính | Giảm ảnh hưởng outlier; chưa có ablation loss riêng |
| AdamW + cosine decay + warm-up | Baseline v1 và friend repo | Fine-tune pretrained model; friend P7 dùng LR 2e-4, weight decay 0.05 | Được dùng trong P7/A2/EXP004/EXP006 | Giúp train ổn định, không phải cải tiến độc lập đã đo |
| AMP FP16, gradient accumulation, gradient clipping | Trainer friend repo / baseline v1 | FP16 trên GPU, accumulation 3 ở P7, clip norm 5 | Cho phép train 512×512 trên T4; peak khoảng 4.3–4.4 GiB | Kỹ thuật vận hành, không trực tiếp làm giảm MAE |
| Resize trực tiếp 512×512 | Baseline v1 `p7_reference` | Kéo toàn ảnh về 512, có thể làm biến dạng tỷ lệ bàn tay | MAE **6.471143** | Baseline cũ; yếu hơn pipeline pad-square |
| Horizontal flip / A2 light flip | Baseline v1 | Thêm RandomHorizontalFlip trong lúc train | MAE **6.706371**, xấu hơn P7 reference 0.235228 | Không giữ trong baseline cũ; không kết luận flip luôn có hại |
| Pad-square giữ tỷ lệ + bicubic antialias | Friend repo `p1_baseline/data.py` | Pad ảnh vuông, resize bicubic 512, antialias, lặp grayscale thành RGB | Tái lập inference sai khác tối đa chỉ **0.054260 tháng** sau khi sửa đúng pipeline | Thành phần preprocessing quan trọng nhất đã học được |
| ImageNet normalization | Baseline v1 và friend repo | Mean/std ImageNet sau khi đưa ảnh về 3 kênh | Dùng trong P7/A2/EXP004/EXP006 | Cần giữ nhất quán giữa train và inference |
| Target normalization | Friend P7 | Chuẩn hóa bone age bằng mean/std train, sau đó đổi ngược về tháng | Friend P7 OOF **6.316691**; EXP006 OOF **6.323629** | Giúp regression ổn định; nên giữ |
| Light affine/brightness/contrast/gamma | Friend P7 A2/light pipeline | Rotation, translation, scale và biến đổi cường độ nhẹ trong train | EXP004 retrain holdout MAE **6.213796** | Có lợi trong pipeline friend; cần giữ mức nhẹ để tránh ảnh phi lâm sàng |
| Friend P7 five-fold ensemble | Repo bạn cùng nhóm | 5 model ConvNeXt-Tiny, prediction fold-held-out/ensemble | Friend test 200 MAE **4.730865**; friend OOF **6.316691** | Mốc tốt nhất hiện tại trên test 200, nhưng test chỉ tham khảo |
| EXP004 retrain theo pipeline friend | Repo bạn cùng nhóm, train lại trên fresh holdout | Giữ pad-square, ConvNeXt, sex embedding, target normalization, light augmentation | Fresh holdout MAE **6.213796**; test 200 MAE **5.214580** | Chưa vượt friend ensemble; chứng minh pipeline có thể tái lập/train lại |
| Blend prediction 25/50/75% | Kết hợp prediction baseline của chúng ta và friend P7 | `blend=(1-w)*own+w*friend` | Official validation: 25% **6.263659**, 50/50 **6.177155**, 75% **6.226381** | Có tín hiệu nhưng trọng số được khảo sát trên cùng validation |
| Blend 50/50 trên holdout nội bộ | EXP002 | Khóa trọng số 50/50 trước khi tách 391 holdout | Baseline **6.070253** → blend **5.761539** | Tín hiệu còn giữ, nhưng holdout chưa hoàn toàn độc lập vì validation đã từng được nhìn thấy |
| Blend trên fresh holdout EXP004 | EXP004 | So sánh 0/25/50/75/100% friend | Retrain **6.213796**; tốt nhất 25% friend **6.179959**; 50/50 **6.204244** | 50/50 không phải trọng số tối ưu cố định |
| 5-fold OOF stratified | Script `prepare_exp006_p7_5fold.py` | Gộp 14.036 development ảnh; stratify theo sex + age bins; mỗi ảnh validation đúng một lần | EXP006 P7 Control pooled OOF **6.323629**, RMSE **8.529248** | Protocol đánh giá đáng tin cậy nhất hiện tại; không dùng test 200 |
| Early stopping theo validation MAE | Friend trainer | Dừng sau 8 epoch không cải thiện; giữ `best_mae.ckpt` | EXP006 các fold dừng epoch 15–29; Fold 1 best **6.239962** | Giảm overfit; log cho thấy train loss thường còn giảm khi val MAE xấu đi |
| Test-time augmentation (TTA) | Friend repo P9-I; ý tưởng từ các hướng augmentation/uncertainty | 5 rotations `[-10,-5,0,5,10]` × flip/no-flip, lấy trung bình 10 prediction | Friend P7 OOF raw **6.317471** → TTA **6.210446**, cải thiện **0.107025** | Candidate inference mạnh; TTA trên EXP006 của chúng ta đang được chạy trên Kaggle |
| Bias correction | Friend repo P9-I | Fit correction cross-fitted trên 4 fold còn lại | Raw + correction **6.324285**; TTA + correction **6.228665** | Không giữ; làm MAE xấu hơn TTA đơn độc |

## 2. Kỹ thuật đã tham khảo hoặc chuẩn bị nhưng chưa có kết quả chính thức của chúng ta

| Kỹ thuật | Nguồn | Cách dự kiến áp dụng | Kết quả hiện có | Trạng thái |
|---|---|---|---|---|
| Label Distribution Learning (LDL) regression-only | Friend repo, các hướng uncertainty/ordinal age | Thêm distribution head quanh tuổi thật; dùng expectation/regression output để dự đoán | Friend regression-only khoảng **6.12496** | EXP007 đã tạo config 5 fold nhưng chưa train |
| LDL fused regression + expectation | Friend repo | Kết hợp regression output và expectation của distribution, trọng số dự kiến 0.5/0.5 | Friend fused khoảng **6.14541** | EXP008 đã tạo config nhưng chưa train |
| Multi-scale feature | Friend repo / bài báo multi-scale | Ghép đặc trưng ở nhiều tầng ConvNeXt hoặc global-local | Friend khoảng **6.29697** | Tín hiệu kém D0; chưa ưu tiên |
| ConvNeXt-V2 | Friend repo / hướng hiện đại | Thay ConvNeXt-Tiny bằng ConvNeXt-V2 | Friend feature-collapse khoảng **31.96** | Không ưu tiên cho vòng hiện tại |
| Resolution 768 | Friend repo | Tăng ảnh từ 512 lên 768 | Friend khoảng **6.18342** so với 512 khoảng 6.18479 | Cải thiện nhỏ, chi phí cao |
| EfficientNet-B0 | Friend repo / Deeplasia screening | Backbone khác ở 512, có thể kết hợp preprocessing Deeplasia | Friend screening khoảng **8.5293–11.0428** | Kết quả âm; chưa tái lập đầy đủ nên chưa kết luận tuyệt đối |
| Foreground crop / letterbox / autocontrast `bram_lite` | Bram/Deeplasia tham khảo; baseline v1 README | Tách foreground, giữ tỷ lệ bằng letterbox, autocontrast nhẹ | Chưa có OOF chính thức của baseline v1 | Candidate chưa được chạy đầy đủ |
| Mask/crop preprocessing | Deeplasia, Bram và repo bạn cùng nhóm | Loại nền hoặc lấy vùng xương trước khi đưa vào backbone | Một số mask của friend làm xấu control | Chỉ thử lại khi có visual QC ảnh gốc/crop/mask |
| Global-local / patch / MIL attention | MMANet, BoNet+ và hướng global-local | Global stream toàn bàn tay + local stream vùng carpal/epiphysis/phalanges, sau đó fusion | Chưa chạy | Hướng có tiềm năng nhưng cần thêm code và OOF |
| Uncertainty head | Các hướng uncertainty-aware | Dự đoán tuổi và độ bất định; dùng uncertainty để weighting/flag outlier | Chưa chạy độc lập | Để sau LDL/TTA control |

## 3. Bảng mốc kết quả tổng hợp

| Mốc | Protocol | MAE | Có thể so sánh trực tiếp với EXP006 OOF? |
|---|---|---:|---|
| Baseline P7 cũ | Official validation, 1.425 ảnh | 6.471143 | Không hoàn toàn, khác split |
| A2 light flip cũ | Official validation, 1.425 ảnh | 6.706371 | Có thể so với P7 cũ cùng split |
| Friend P7 | 5-fold OOF, 14.036 ảnh | 6.316691 | Gần nhất, nhưng khác pipeline/split hash |
| EXP006 P7 Control | 5-fold OOF, 14.036 ảnh | **6.323629** | Mốc control hiện tại |
| Friend P7 TTA | 5-fold OOF, 14.036 ảnh | 6.210446 | Kết quả friend, chưa phải EXP006 TTA |
| EXP004 retrain | Fresh holdout 1.260 ảnh | 6.213796 | Không cùng protocol |
| Friend P7 test 200 | Test 200 exploratory | **4.730865** | Chỉ tham khảo, không dùng chọn model |
| Mục tiêu cuối | Protocol phải được khóa trước | **< 3.5** | Chưa đạt |

## 4. Kết luận thực nghiệm

1. Thành phần có bằng chứng mạnh nhất hiện tại là `pad_square + bicubic
   antialias + target normalization + ConvNeXt-Tiny + sex embedding` của
   pipeline friend.
2. TTA là kỹ thuật inference có cải thiện rõ trên OOF của friend, khoảng
   **0.107 tháng**; cần chờ kết quả TTA trên EXP006 để xác nhận trên control
   của chúng ta.
3. Blend có tín hiệu trên validation/holdout nhưng chưa vượt friend ensemble
   trên test 200; không được chọn trọng số bằng test 200.
4. LDL là hướng có cơ sở và có kết quả thăm dò tốt trong repo friend, nhưng
   EXP007/EXP008 của chúng ta chưa được train nên chưa được tuyên bố là cải thiện.
5. ConvNeXt-V2, multi-scale, EfficientNet và mask chưa cho tín hiệu đủ tốt để
   ưu tiên trước TTA/LDL.

## 5. Nguồn chính

- [Baseline process của chúng ta](BASELINE_QUY_TRINH.md)
- [Nhật ký thực nghiệm](EXPERIMENT_LOG.md)
- [Tổng hợp toàn bộ tiến trình](TONG_HOP_TOAN_BO_TIEN_TRINH.md)
- [Repo P7/P9 của bạn cùng nhóm](https://github.com/nhattoan235/Artifact-Robust-Bone-Age-Assessment-Using-Multi-Scale-Label-Distribution-ConvNeXt-V2-)
- [P9-I TTA handoff](../friend_repo/p9_inference/P9_I_HANDOFF.md)
- [Scientific Reports 2022](https://www.nature.com/articles/s41598-022-10292-y)
- [ScienceDirect 2022](https://www.sciencedirect.com/science/article/abs/pii/S1746809422005055)
- [Springer 2023 — s11517-023-03013-8](https://link.springer.com/article/10.1007/s11517-023-03013-8)
- [Springer 2023 — s00247-023-05789-1](https://link.springer.com/article/10.1007/s00247-023-05789-1)
