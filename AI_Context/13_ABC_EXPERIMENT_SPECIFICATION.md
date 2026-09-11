# Đặc tả chi tiết phương án A–B–C

> Ngày khóa: 2026-08-23  
> Trạng thái: thiết kế; chưa train phương án mới  
> Quy tắc: không dùng RSNA test để chọn model, crop, TTA hoặc ensemble weight

## 1. Mục tiêu và thứ tự

Mục tiêu là giảm clean pooled OOF MAE trên đúng 14.036 ảnh development bằng giả thuyết có thể tái lập và bảo vệ học thuật. Không lọc ca khó, đổi split, cân bằng cực đoan hoặc tối ưu theo test.

Thứ tự khuyến nghị:

1. Chạy A/D3-OOF trước vì registry, config, test và preflight đã sẵn sàng.
2. Nếu A đạt gate, giữ A làm ứng viên chính.
3. Nếu A thất bại, đóng A như kết quả âm tính và kích hoạt C.
4. B giữ vai trò robustness phụ/backup, không tự động thay A chỉ vì synthetic-artifact MAE giảm.

## 2. Baseline và quy tắc chung

| Baseline | Vai trò | MAE tháng |
|---|---|---:|
| E1/P7 raw | global OOF reference | 6,316691 |
| E1-TTA/P9 | current inference reference | 6,210446 |
| E1 official validation | screening reference | 6,184792 |

- Dùng 5 fold P7 đã khóa.
- Mọi OOF phải khớp image ID, target, sex, fold và SHA với E1.
- Mỗi run lưu manifest SHA, config/code hash, seed, GPU, checkpoint, warning log và prediction schema.
- Checkpoint phải atomic, lưu optimizer/scheduler/RNG và resume được.
- Không push remote nếu chưa được cho phép.

## 3. Phương án A — D3-OOF + E1 ensemble

### Giả thuyết

Label-distribution learning có thể biểu diễn sự mơ hồ giữa các tuổi xương lân cận và tạo lỗi bổ sung cho E1 direct regression.

### Thiết kế

- Train năm fold D3 độc lập từ ImageNet initialization, seed 42.
- Giữ target mean/std theo từng fold P7.
- D3 gồm regression head và label-distribution head.
- Lưu cả fused prediction và regression-only prediction.
- Primary ensemble cố định: 0,5 × E1 raw + 0,5 × D3 raw.
- Chỉ chạy D3-TTA sau khi raw OOF qua gate.

### Gate A

- Pooled OOF MAE giảm ít nhất 0,10 tháng.
- Paired bootstrap 95% CI của delta hoàn toàn dưới 0.
- Ít nhất 4/5 fold cải thiện.
- Không subgroup collapse theo sex hoặc age bin.
- Không duplicate/missing/non-finite ID.

Nếu A fail, không đổi LDL weight, sigma, fold, checkpoint hoặc ensemble weight hậu nghiệm. Chuyển sang C.

## 4. Phương án B — Artifact Consistency Training

### Vai trò

B kiểm tra khả năng ổn định trước biến đổi ảnh hợp lý. Đây là hướng robustness, không phải cam kết giảm clean MAE.

### Thiết kế

- E1 ConvNeXt-Tiny + sex embedding giữ nguyên.
- Mỗi ảnh tạo clean view và artifact view nhẹ: blur, gamma, contrast, noise.
- Mức artifact chỉ lấy từ quality metrics của training split.
- Không occlusion trung tâm và không phá cấu trúc xương.
- Loss gồm clean regression, artifact regression và consistency.
- Warm-up 3 epoch clean-only; ramp artifact/consistency ở epoch 4–8.
- Clean MAE là endpoint chính; synthetic stress MAE là endpoint phụ.

### Rủi ro

Synthetic artifact không đại diện hoàn toàn ảnh lâm sàng. Nếu B qua screening thì chạy control có augmentation nhưng consistency weight bằng 0 để tách tác động. Không claim clinical robustness nếu không có external artifact set.

## 5. Phương án C — C-ROI local/global ensemble

### Giả thuyết

E1 resize toàn bàn tay về 512 nên có thể mất chi tiết nhỏ. Local specialist nhìn vùng carpal–metacarpal–MCP ở trường nhìn hẹp hơn có thể sửa lỗi bổ sung, nhất là 60–132 tháng.

### Kiến trúc

Ảnh toàn bàn tay đi vào E1 đã khóa. ROI giải phẫu đi vào C-ROI ConvNeXt-Tiny độc lập. Prediction chính là mean cố định 0,5/0,5.

- C-ROI khởi tạo độc lập từ ImageNet; không warm-start E1.
- Giữ direct regression, sex embedding, SmoothL1 và recipe E1.
- V1 chỉ dùng một ROI rộng; không learned gate, age routing hoặc nhiều ROI.
- C-ROI standalone vẫn báo cáo nhưng không bắt buộc thắng E1.

### ROI protocol

1. Khôi phục/tái sinh đúng mask segmentation đã audit và khóa source SHA.
2. Chọn hand component lớn nhất, tìm trục dài bằng PCA.
3. Xác định đầu ngón/cổ tay bằng mask profile, không dùng target hoặc prediction.
4. Crop trực tiếp từ ảnh gốc bằng một affine/resize.
5. Lấy vùng từ khoảng 28% chiều dài tính từ đầu ngón đến hết cổ tay, phủ toàn chiều ngang, margin 8%.
6. Không nhân mask vào pixel; giữ intensity gốc, pad vuông và resize 512.

Tọa độ chỉ được kiểm tra bằng anatomical coverage trên training QC đã khóa, không chọn lại theo MAE.

### Fallback

- Mask thiếu/rỗng/sai kích thước: full-hand fallback.
- Hai component lớn hoặc orientation không chắc: fallback, không loại ảnh.
- Mask missing và tổng fallback hình học đều phải ≤1%; vượt ngưỡng thì dừng trước train.
- Cache lưu image SHA, mask SHA, coordinates, angle, fallback reason và output SHA.
- Phải công khai segmentation supervision bên ngoài; C không gọi là annotation-free tuyệt đối.

### Train C

- Đúng 5 fold P7, seed 42, output riêng.
- Local micro-batch 6, accumulation 6, effective batch 36.
- Colab T4 có thể dùng batch 12, accumulation 3 sau khi kiểm tra ổn định.
- Fold 1 chỉ là operational pilot: preflight, forward/backward và interrupt/resume.
- Không kết luận khoa học từ Fold 1.
- Nếu không có NaN/Inf, OOM lặp lại, collapse, resume lỗi hoặc fallback >1%, chạy bốn fold còn lại.
- Ước tính 3,5–4,5 giờ/fold trên RTX 3050 Ti; 2,5–4 giờ/fold trên T4 tùy tải và I/O.

## 6. Statistical gate C

Endpoint 1: so sánh 0,5 × E1 raw + 0,5 × C-ROI raw với E1 raw 6,316691.

Endpoint 2: chạy đúng TTA P9 cho C-ROI rồi so 0,5 × E1-TTA + 0,5 × C-ROI-TTA với E1-TTA 6,210446.

C chỉ thay pipeline chính khi đồng thời:

- Endpoint 1 giảm ít nhất 0,10 tháng.
- Paired bootstrap 95% CI endpoint 1 hoàn toàn dưới 0.
- Ít nhất 4/5 fold cải thiện.
- Endpoint 2 giảm ít nhất 0,10 tháng so với E1-TTA.
- MAE nam/nữ không tăng quá 0,15 tháng.
- Age bin lớn không tăng quá 0,20 tháng.
- Sex × age cell có ít nhất 200 ảnh không tăng quá 0,30 tháng.
- Không prediction collapse hoặc mean shrinkage.

Nếu raw tốt nhưng TTA không đạt, C là ablation dương tính nhưng không thay E1-TTA. Không tune weight sau OOF.

## 7. Kiểm thử và failure matrix

Trước train:

- Cùng input SHA phải sinh cùng crop SHA.
- Crop function không nhận target, sex hoặc prediction.
- Test ảnh ngang/dọc, tối, sát biên, hai tay, mask rỗng và mask sai kích thước.
- Kiểm tra bounds, coverage, fallback rate và single-resampling.
- Xác minh no-overlap giữa train/validation.
- Chạy một batch forward/backward và interrupt/resume.

Sau train:

- Đủ 14.036 OOF unique ID, target/sex/fold/SHA khớp E1.
- Paired bootstrap, fold consistency và subgroup metrics.
- Grad-CAM/occlusion chỉ dùng giải thích, không chọn model.
- Báo cáo negative result và segmentation supervision.

| Failure | Hành động |
|---|---|
| Mask SHA/count sai | Dừng và khôi phục đúng artifact |
| ROI sai/cắt anatomy | Fallback; trên 1% thì không train |
| Affine làm mờ | Sửa single-resampling, không đổi MAE gate |
| Model collapse/overfit | Early stop; không đổi loss hậu nghiệm |
| Prediction quá giống E1 | Không tune weight; kết luận thiếu diversity |
| TTA cắt ROI | Giữ raw protocol, không chọn view theo MAE |
| Mất runtime/hết ổ | Resume checkpoint, không ghi đè run hoàn tất |

## 8. Claim policy và trạng thái

Claim hợp lệ nếu đạt gate:

> An anatomy-aware local/global ensemble improved paired five-fold OOF MAE on the RSNA development set under a locked protocol.

Không claim confirmatory vượt Bram 3,68 từ test 200 ảnh vì test đã được mở ở P8. Muốn claim xác nhận cần external holdout mới.

Tài liệu liên quan:

- [13_GROUP_COMBINED_TECHNIQUES_RESULTS_2026_08_24.md](13_GROUP_COMBINED_TECHNIQUES_RESULTS_2026_08_24.md)
- [19_REPORT_PLAN_A_D3_FINAL.md](19_REPORT_PLAN_A_D3_FINAL.md)
- [11_C_ROI_FALLBACK_PLAN.md](11_C_ROI_FALLBACK_PLAN.md)

Trạng thái hiện tại: A có registry/config đã chuẩn bị; B và C mới ở mức thiết kế; chưa train, chưa sửa dữ liệu và chưa push remote.


