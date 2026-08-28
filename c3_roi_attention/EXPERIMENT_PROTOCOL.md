# C3-ROI + Spatial Attention — protocol v1

## Câu hỏi nghiên cứu

Sau khi ROI đã loại phần lớn nền ngoài bàn tay, một spatial gate nhẹ trên feature
map cuối của ConvNeXt-Tiny có cải thiện dự đoán tuổi xương so với C3-ROI không?

Attention không thay thế ROI. ROI quyết định vùng ảnh đầu vào; attention học cách
phân bổ trọng số bên trong ROI, ví dụ giữa vùng cổ tay/carpal, metacarpal và
phalanges.

## Thay đổi duy nhất

- Baseline: `C3_ROI_V1`, kiến trúc `convnext_tiny`.
- Candidate: `C3_ROI_ATTN_V1`, kiến trúc
  `convnext_tiny_spatial_attention`.
- Spatial gate: convolution `1x1`, một kênh, áp dụng trên feature map cuối.
- Gate: `2 * sigmoid(logit)`, khởi tạo logit bằng 0 nên gate ban đầu bằng 1.
- Sau gate vẫn dùng đúng pooling, normalization, sex embedding và regression
  head của recipe C3.

Mỗi fold giữ nguyên manifest, split hash, seed, target normalization,
augmentation, optimizer, scheduler, batch, epoch và early stopping của C3. Năm
config candidate chỉ khác `run_id`, `output_root` và `architecture`.

## Đánh giá đã khóa

1. Train đủ 5 fold và tạo prediction OOF cho đúng 14.036 development IDs.
2. Ghép prediction theo `image_id`; kiểm tra target và sex khớp C3.
3. Primary metric: OOF MAE tháng.
4. Báo cáo thêm RMSE, median AE, accuracy trong 6/12 tháng, từng fold và giới.
5. Paired bootstrap 95% CI cho `MAE_attention - MAE_C3`; số âm có lợi cho
   attention.
6. Không chọn fold đẹp, không đổi recipe sau khi xem kết quả, không mở test 200
   để tune.

Candidate chỉ được coi là cải thiện nếu OOF MAE tốt hơn C3 và hiệu ứng không do
một fold đơn lẻ. Nếu CI cắt 0, kết luận là chưa có bằng chứng rõ ràng; không tuyên
bố attention thắng.

## Phạm vi kết luận

Đây là ablation về độ chính xác trên ROI, tách biệt với hướng artifact robustness
paired/uncertainty. Kết quả tốt không tự động chứng minh robustness trước marker,
kim loại, dây/ống hoặc che khuất.
