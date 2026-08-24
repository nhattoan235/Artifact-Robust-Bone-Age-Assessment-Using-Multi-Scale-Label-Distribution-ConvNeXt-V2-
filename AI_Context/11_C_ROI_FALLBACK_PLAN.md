# Phương án C-ROI: anatomy-aware local specialist fallback

> **Ngày khóa thiết kế:** 2026-08-22  
> **Trạng thái:** thiết kế đã được người dùng chấp nhận; chưa triển khai code, tạo cache hoặc train  
> **Vai trò:** phương án dự phòng nếu D3-OOF ở phương án A không đạt gate  
> **Primary endpoint:** pooled 5-fold OOF MAE trên đúng 14.036 ảnh development  
> **Test policy:** không dùng RSNA test để thiết kế, chọn checkpoint, crop, TTA hoặc ensemble weight

## 1. Khuyến nghị thứ tự A -> C

Không bỏ phương án A để chuyển thẳng sang C.

- A/D3-OOF đã có registry, năm config, kiểm thử và preflight; không còn chi phí thiết kế lớn.
- Screening trên official validation cho thấy `E1 + D3` equal-weight đạt khoảng 6,0481 so với E1 6,1848, tức có tín hiệu cải thiện khoảng 0,137 tháng dù chưa phải bằng chứng OOF.
- C-ROI có giả thuyết giải phẫu mạnh hơn nhưng chưa có screening nội bộ, còn phụ thuộc vào việc khôi phục mask và xác minh crop.
- Vì vậy, chạy A đúng protocol đã khóa. Nếu A không đạt gate thì không chỉnh D3 hậu nghiệm; đóng A và kích hoạt C-ROI.
- Nếu A đạt gate, C tiếp tục là dự phòng/extension và chỉ chạy khi ngân sách GPU còn phù hợp.

## 2. Understanding summary

- Mục tiêu là giảm overall MAE thật sự, có giá trị học thuật và không đạt kết quả bằng lọc ca khó, đổi test hoặc tối ưu metric hậu nghiệm.
- C phải khác về cơ chế so với A/LDL và B/artifact consistency.
- Điểm yếu thực tế của E1 tập trung ở giai đoạn phát triển xương, đặc biệt khoảng 60-132 tháng, không chủ yếu do ảnh mờ/tối.
- E1/P7 giữ vai trò global model đã khóa và không train lại.
- C-ROI là local specialist độc lập, nhìn vùng giải phẫu quan trọng ở độ phân giải hiệu dụng cao hơn.
- Mọi kết luận dùng đúng năm split P7 và paired OOF; official test không tham gia model selection.
- Pipeline phải chạy được trên RTX 3050 Ti 4 GB hoặc Colab, hỗ trợ ngắt quãng, checkpoint và resume.

## 3. Bằng chứng nội bộ dẫn tới C-ROI

E1/P7 pooled OOF có MAE 6,316691 tháng. Phân tích sâu trên 14.036 OOF prediction cho thấy:

| Nhóm | MAE E1 OOF (tháng) |
|---|---:|
| 60-119 tháng | 7,3168 |
| 96-107 tháng | 7,926 |
| 108-119 tháng | 8,334 |
| 120-131 tháng | 8,031 |
| Nam 60-119 tháng | 8,055 |
| Nữ toàn bộ | 6,536 |
| Nam toàn bộ | 6,131 |

- 10% ca có lỗi lớn nhất đóng góp khoảng 30,3% tổng absolute error.
- Các chỉ số sharpness, intensity và contrast chỉ tương quan rất yếu với absolute error; hard filtering không xử lý đúng nguồn lỗi chính.
- TTA disagreement có tương quan vừa phải với lỗi nhưng chưa đủ để trở thành model router an toàn.
- Nghiên cứu trước cho thấy carpal, metacarpal, MCP và các vùng lân cận có vai trò thay đổi theo tuổi/giới; tuy nhiên nhiều mô hình công bố dùng keypoint annotations, split khác hoặc tài nguyên GPU lớn nên không được sao chép trực tiếp.

## 4. Giả định, phạm vi và non-goals

### Giả định

- Dùng ảnh, tuổi xương và giới tính RSNA hiện có; ImageNet initialization được phép.
- Không dùng external bone-age labels hoặc external keypoint annotations.
- Mask Deeplasia đã audit được phép dùng như một công cụ segmentation preprocessing cố định. Phải công khai rằng mask có segmentation supervision bên ngoài; không được gọi toàn bộ C-ROI là annotation-free tuyệt đối.
- Mask chỉ dùng để xác định hình học crop; pixel model nhận vẫn lấy từ ảnh X-quang gốc.

### Non-goals

- Không loại ảnh vì khó, mờ hoặc dự đoán sai.
- Không chia lại fold, oversample tuổi/giới hoặc train model nam/nữ riêng.
- Không học ensemble weight bằng official validation/test.
- Không hứa đạt 3,68 tháng từ OOF; mốc Bram là kết quả trên RSNA test 200 ảnh và không cùng endpoint.
- Không dùng Grad-CAM hay qualitative examples để chọn checkpoint/crop sau khi xem kết quả.

### Yêu cầu vận hành

- Quy mô: 14.036 ảnh development, năm fold P7 đã khóa.
- Phần cứng: local RTX 3050 Ti 4 GB hoặc Colab; mỗi phiên có thể ngắt sau khoảng 8 giờ.
- Reliability: atomic checkpoint; lưu optimizer, scheduler, scaler và RNG; cache/config/manifest có SHA.
- Privacy: RSNA là dữ liệu công khai đã ẩn danh; không thêm dữ liệu bệnh nhân riêng.
- Ownership: người dùng quyết định gate và lịch GPU; workspace giữ artifact/handoff; không push remote khi chưa được phép.

## 5. Các hướng C đã xem xét

| Hướng | Quyết định | Lý do |
|---|---|---|
| C-ROI local specialist | Chọn | Tác động trực tiếp lên chi tiết giải phẫu, khác D3, phù hợp lỗi 60-132 tháng và có thể dùng E1 global sẵn có |
| C-Stage soft mixture-of-experts | Không chọn primary | Stage label vẫn suy ra từ tuổi, có ranh giới nhân tạo và gần với rủi ro mean-shrinkage/ordinal của D3 |
| C-Arch TinyViT/ResNet ensemble | Không chọn primary | Có diversity nhưng recipe/tài nguyên rủi ro; EfficientNet screening trước đã cho thấy đổi backbone không tự động cải thiện |

## 6. Kiến trúc C-ROI v1

C-ROI v1 chỉ dùng một ROI giải phẫu rộng để giữ thiết kế và attribution đơn giản.

```text
Ảnh toàn bàn tay -> E1 đã khóa ------------------+
                                                  +-> mean cố định 0,5/0,5
ROI giải phẫu -> C-ROI ConvNeXt-Tiny độc lập ----+
```

- ROI giữ proximal phalanges/MCP, metacarpal, carpal và distal radius-ulna; loại phần lớn đầu ngón và nền.
- C-ROI dùng ConvNeXt-Tiny ImageNet initialization độc lập, không warm-start E1.
- Giữ direct regression, sex embedding, SmoothL1 và recipe E1 để crop/field-of-view là thay đổi khoa học chính.
- Không dùng learned gate, age-prediction routing hoặc nhiều ROI trong v1.
- C-ROI standalone được báo cáo nhưng không bắt buộc thắng E1; primary candidate là equal-weight global/local ensemble.

## 7. Protocol tạo ROI

1. Khôi phục hoặc tái sinh đúng mask artifact đã audit; khóa source/model version và SHA.
2. Chọn thành phần bàn tay lớn nhất và tìm trục dài bằng PCA.
3. Phân biệt đầu ngón/cổ tay bằng profile mặt cắt mask: phía ngón thường có nhiều đoạn, phía cổ tay thường liên tục.
4. Chuẩn hóa hướng chỉ để lấy crop; thực hiện đúng một phép affine/resize trực tiếp từ ảnh gốc.
5. Lấy ROI từ khoảng 28% chiều dài tính từ đầu ngón đến hết cổ tay, phủ toàn bộ chiều ngang và margin 8%.
6. Giữ grayscale/intensity gốc; pad vuông và resize 512 như E1.

Tỷ lệ 28% và margin 8% chỉ được xác minh bằng anatomical coverage trên một tập training QC khóa trước. Không dùng prediction hoặc validation MAE để chọn lại tọa độ.

### Fallback

- Mask thiếu/rỗng/sai kích thước: full-hand input như E1.
- Hai thành phần lớn gần tương đương: full-hand; không tự chọn một tay.
- Không xác định chắc orientation: crop đối xứng rộng; nếu vẫn fail QC thì full-hand.
- Không loại ảnh vì crop lỗi.
- Mask missing phải dưới 1%; tổng fallback hình học phải không quá 1%. Nếu vượt gate thì dừng trước train và sửa localization, không chữa bằng loại ảnh.

Cache ROI phải lưu ID, source image SHA, mask source/SHA, tọa độ crop, góc, fallback reason và output SHA.

## 8. Train và pilot

- Năm config dùng nguyên fold assignment P7 và target normalization riêng của từng fold.
- Seed primary 42; micro-batch local 6, accumulation 6, effective batch 36.
- Không thay crop, learning rate, loss hoặc augmentation sau khi xem Fold 1/OOF MAE.
- Mỗi run có run ID/output riêng; không ghi đè artifact đã hoàn tất.

Pilot Fold 1:

1. Preflight ID/path/hash/cache.
2. Hai optimizer step forward/backward.
3. Interrupt/resume một lần và kiểm tra state.
4. Train Fold 1 hoàn chỉnh.

Fold 1 chỉ là operational gate. Chỉ dừng sớm toàn hướng khi có NaN/Inf, OOM lặp lại, resume sai, prediction collapse, fallback >1%, hoặc C-ROI Fold 1 kém E1 Fold 1 trên 0,75 tháng. Nếu không có failure rõ ràng thì chạy bốn fold còn lại; không kết luận khoa học từ một fold.

Ước tính dựa trên run C2 local: khoảng 3,5-4,5 giờ/fold, tổng 18-23 giờ GPU cho năm fold; ROI cache chủ yếu dùng CPU và tạo một lần.

## 9. OOF aggregation và statistical gate

Báo cáo chỉ được tạo khi:

- đủ đúng 14.036 OOF rows và 14.036 ID duy nhất;
- mỗi ID thuộc đúng một validation fold;
- target, sex, fold và image SHA khớp E1;
- không có missing/duplicate/non-finite prediction.

### Endpoint 1: primary kiến trúc

So sánh `0.5 * E1_raw + 0.5 * CROI_raw` với E1 raw MAE 6,316691.

### Endpoint 2: utility bắt buộc để thay pipeline

Áp dụng đúng TTA P9 đã khóa cho C-ROI. So sánh `0.5 * E1_TTA + 0.5 * CROI_TTA` với E1-TTA MAE 6,210446.

C-ROI chỉ được nâng thành pipeline chính khi đồng thời:

- primary delta MAE không lớn hơn -0,10 tháng;
- paired bootstrap 95% CI của primary delta hoàn toàn dưới 0;
- ít nhất 4/5 fold cải thiện;
- final TTA ensemble cải thiện ít nhất 0,10 tháng so với E1-TTA;
- MAE toàn bộ nam hoặc nữ không tăng quá 0,15 tháng;
- không age bin lớn nào tăng quá 0,20 tháng;
- không sex x age cell có ít nhất 200 ảnh nào tăng quá 0,30 tháng;
- prediction spread hợp lý, không collapse/mean shrinkage.

Residual/prediction correlation và error overlap được báo cáo để giải thích diversity nhưng không thay thế primary MAE gate.

Nếu raw qua gate nhưng TTA không qua, C-ROI là ablation dương tính nhưng không thay E1-TTA. Nếu primary fail, không điều chỉnh crop/loss/weight hậu nghiệm trên cùng OOF.

## 10. Failure matrix

| Failure | Phát hiện | Hành động |
|---|---|---|
| Sai/mất mask version | SHA/count mismatch | Dừng và khôi phục đúng artifact; không đổi segmentation model |
| ROI lộn đầu/cắt anatomy | Geometry test + contact sheet training-only | Fallback; nếu >1% thì không train |
| Hai tay/tay bị cắt | Multiple components hoặc border QC | Full-hand fallback; không loại ảnh |
| Affine làm mờ | Sharpness/edge energy audit | Chỉ warp/resize một lần; dừng nếu suy giảm bất thường |
| Overfit/mean shrinkage | Prediction SD, gap và age bias | Early stop; không cứu bằng đổi loss hậu nghiệm |
| C-ROI quá giống E1 | Correlation + ensemble delta | Kết luận thiếu diversity; không tune weight |
| TTA cắt anatomy | Coverage test đủ 10 views | Giữ raw; không chọn view theo MAE |
| Train bị ngắt/hết ổ | Checkpoint/free-space audit | Resume đúng state; không ghi đè run hoàn tất |

## 11. Testing strategy

Trước GPU:

- cùng input SHA luôn sinh cùng crop SHA;
- crop function không nhận target, sex hoặc prediction;
- test ảnh ngang/dọc, tối, sát biên, hai tay và mask rỗng;
- kiểm tra bounds, coverage, fallback và single-resampling;
- exact fold ID coverage và no overlap;
- một batch forward/backward và interrupt/resume.

Sau train:

- OOF integrity và paired bootstrap;
- fold consistency và subgroup gates;
- Grad-CAM/occlusion trên tập mẫu khóa trước chỉ dùng làm qualitative explanation, không dùng chọn model;
- báo cáo rõ segmentation supervision, compute, fallback rate và negative results.

## 12. Tuyên bố được phép

Nếu C-ROI đạt gate, tuyên bố hợp lệ là: "An anatomy-aware local/global ensemble improved paired five-fold OOF MAE on the RSNA development set under a locked protocol."

Không tuyên bố confirmatory rằng mô hình vượt Bram 3,68 chỉ từ RSNA test, vì bộ test 200 ảnh đã được truy cập ở P8 và không còn là holdout hoàn toàn mới. Một tuyên bố xác nhận cần external holdout chưa được dùng để thiết kế.

## 13. Decision log

| Quyết định | Phương án khác | Lý do |
|---|---|---|
| Giữ A trước, C làm fallback | Bỏ A và chạy C ngay | A đã sẵn sàng và có screening ensemble dương; C còn engineering risk |
| Chọn C-ROI | C-Stage/C-Arch | Khớp lỗi thực tế và tạo field-of-view diversity |
| Giữ E1 global, không train lại | End-to-end global-local mới | Tiết kiệm compute và tách attribution |
| Một ROI rộng trong v1 | Hai/ba ROI hoặc learned gate | Giảm VRAM, hyperparameter và collapse risk |
| ConvNeXt-Tiny độc lập | Warm-start E1 | Bảo toàn diversity và so sánh sạch |
| Mask chỉ dùng hình học | Nhân mask vào pixel | P3 masking không cải thiện; tránh thay intensity distribution |
| Chấp nhận segmentation supervision có disclosure | Zero-external tuyệt đối | ROI geometry tin cậy hơn mà không thêm age/keypoint labels |
| Full-hand fallback, không loại ảnh | Exclude ca crop lỗi | Giữ protocol và tránh selection bias |
| Fixed 0,5/0,5 | Tune ensemble weight | Tránh overfit official validation/test |
| Hai endpoint phân cấp raw/TTA | Chọn endpoint tốt nhất hậu nghiệm | Tách hiệu quả kiến trúc khỏi utility của pipeline hiện tại |

## 14. Tài liệu tham khảo chính

- Wang et al., *Improve bone age assessment by learning from anatomical local regions*: https://arxiv.org/abs/2005.13452
- Chen et al., *Attention-Guided Discriminative Region Localization and Label Distribution Learning for Bone Age Assessment*: https://doi.org/10.1109/JBHI.2021.3095128
- Kitamura and Pan, *Artificial Intelligence Class Activation Mapping of Bone Age*: https://doi.org/10.1148/radiol.211790
- Bram et al., *Determination of Skeletal Age From Hand Radiographs Using Deep Learning*: https://doi.org/10.1177/03635465251359618

## 15. Implementation handoff

Chưa triển khai. Khi được cho phép, thứ tự là:

1. Audit/restore mask source và dung lượng.
2. Viết test ROI trước implementation.
3. Tạo/audit ROI cache và contact sheet training-only.
4. Tạo năm config từ fold P7.
5. Smoke/resume rồi train Fold 1 operational pilot.
6. Chạy Fold 2-5 nếu operational gate PASS.
7. Aggregate raw/TTA OOF và áp dụng statistical gate đã khóa.

