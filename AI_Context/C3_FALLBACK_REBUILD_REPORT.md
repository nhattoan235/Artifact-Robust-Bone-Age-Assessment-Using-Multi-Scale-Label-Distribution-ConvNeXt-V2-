# Báo cáo dễ hiểu: fallback của C3 ROI

**Ngày:** 2026-09-05  
**Phạm vi:** chỉ xử lý fallback phân đoạn bàn tay và tạo ROI cache  
**Không thực hiện:** đổi preprocessing của model, train lại hoặc đánh giá MAE

## 1. Fallback là gì?

Ảnh X-quang bàn tay cần được tìm vùng bàn tay trước khi đưa vào model. Chương trình segmentation cố gắng tạo ra một **mask** và một **bbox**:

- **Mask:** bản đồ pixel cho biết vùng nào là bàn tay.
- **BBox:** hình chữ nhật bao quanh bàn tay, có dạng `x,y,width,height`.
- **ROI:** phần ảnh được crop từ bbox để đưa vào pipeline C3.

Fallback là phương án dự phòng khi cách tìm bàn tay tiêu chuẩn không tìm được vùng nào đủ đáng tin cậy. Nó xảy ra **trước khi train**, không phải là fallback của model dự đoán tuổi xương.

Nếu không có fallback, ảnh lỗi có thể bị loại khỏi dataset. Cách hiện tại giữ ảnh lại bằng cách dùng toàn bộ ảnh làm ROI, an toàn hơn việc crop nhầm vào cassette, frame hoặc nhãn dán.

## 2. Luồng xử lý hiện tại

Mỗi ảnh đi qua tối đa hai lần segmentation:

```text
Ảnh X-quang
    |
    v
Lần 1: strict, max_borders=2
    |
    +-- tìm được bàn tay --> dùng bbox + margin 12% --> ROI hợp lệ
    |
    `-- không tìm được
            |
            v
        Lần 2: rescue, max_borders=3
            |
            +-- tìm được bàn tay --> dùng bbox + margin 12% --> ROI hợp lệ
            |
            `-- vẫn thất bại --> dùng toàn bộ ảnh --> global fallback
```

`max_borders` là số cạnh ảnh mà vùng được phép chạm vào. Một số ảnh hợp lệ có cẳng tay chạm mép ảnh, nên strict `=2` có thể bỏ qua chúng. Rescue `=3` giúp thu hồi các trường hợp này.

Không dùng `max_borders=3` ngay từ đầu vì vùng cassette hoặc nền phơi sáng cũng có thể chạm nhiều cạnh. Tách rescue thành bước thứ hai giúp giữ strict pass an toàn cho phần lớn ảnh, chỉ nới lỏng điều kiện khi thật sự cần.

## 3. Ý nghĩa từng trạng thái trong manifest

### `ok`

Lần strict tìm được mask và bbox hợp lệ. Đây là trường hợp bình thường, không phải fallback.

### `border_rescue`

Lần strict thất bại, nhưng lần rescue với `max_borders=3` tìm được mask và bbox hợp lệ. Ảnh vẫn được crop theo bàn tay, nên đây là **rescue thành công**, không phải hard failure.

### `segment_hand_failed`

Cả strict và rescue đều thất bại. Chương trình không đoán bừa vùng bàn tay:

- mask được ghi là toàn màu đen;
- ROI được đặt là toàn bộ ảnh;
- ảnh vẫn được giữ trong dataset;
- manifest ghi rõ đây là hard fallback.

### `invalid_bbox`

Segmentation báo thành công nhưng bbox bị rỗng hoặc sai định dạng. Trường hợp này cũng chuyển về toàn ảnh để tránh crash hoặc crop sai.

## 4. Kết quả rebuild trên 14,036 ảnh

| Trạng thái | Số ảnh | Tỷ lệ | Giải thích |
|---|---:|---:|---|
| Strict `ok` | 11,441 | 81.51% | Thành công ngay lần đầu |
| `border_rescue` | 621 | 4.43% | Được cứu ở lần nới lỏng |
| Hard fallback | 1,974 | 14.06% | Cả hai lần đều thất bại |
| ROI bbox hợp lệ | 12,062 | 85.94% | `ok` + `border_rescue` |

### So sánh trước và sau rescue

Nếu chỉ dùng strict `max_borders=2` thì có:

```text
2595 ảnh fail / 14036 ảnh = 18.49%
```

Rescue đã cứu được 621 ảnh:

```text
2595 - 621 = 1974 ảnh fail
1974 / 14036 = 14.06%
```

Nói cách khác, rescue đã giảm số hard failure khoảng 24% so với strict-only. Tuy nhiên, nó chưa đủ để đưa tỷ lệ xuống dưới 10%.

### Theo split

| Split | Tổng | Strict `ok` | Rescue | Hard fallback | Tỷ lệ hard fallback |
|---|---:|---:|---:|---:|---:|
| Train | 12,611 | 10,311 | 542 | 1,758 | 13.94% |
| Official validation | 1,425 | 1,130 | 79 | 216 | 15.16% |

Validation có tỷ lệ fail cao hơn train một chút. Đây là dấu hiệu cần kiểm tra thêm bằng ảnh mẫu, nhưng chưa đủ để kết luận model sẽ giảm MAE hay tăng MAE.

## 5. Timeline 33% → 18.49% → hiện tại

Đây là phần cần đọc cẩn thận: các mốc phần trăm không dùng cùng một tập ảnh.

| Mốc | Tập ảnh | Kết quả | Ý nghĩa |
|---|---|---:|---|
| Ban đầu | Test khóa, 200 ảnh | 66/200 = **33.00%** | Cache test cũ, strict-only; 66 ảnh dùng global fallback |
| Sau cải thiện strict | Train + official validation, 14,036 ảnh | 2,595/14,036 = **18.49%** | Cache backup `margin8`, vẫn strict `max_borders=2` |
| Sau rescue R2 | Train + official validation, 14,036 ảnh | 1,974/14,036 = **14.06%** | Thêm rescue `max_borders=3`, cứu được 621 ảnh |
| Test R2 mới | Test khóa, 200 ảnh | 56/200 = **28.00%** | Rescue cứu thêm 10 ảnh; 144 ảnh có ROI bbox |

Vì vậy không nên viết rằng cùng một test set đã giảm trực tiếp từ 33% xuống 18.49%. Cách diễn đạt chính xác là:

- trên test 200 ảnh: **33.00% → 28.00%** sau fallback R2;
- trên development 14,036 ảnh: **18.49% → 14.06%** sau rescue;
- mốc 18.49% là bằng chứng trung gian của pipeline strict, không phải kết quả test cuối.

Mốc 33% ban đầu vẫn quan trọng vì nó cho thấy fallback cũ trên test còn cao. Mốc 18.49% cho biết sau khi cải thiện segmentation strict và đánh giá trên development thì tỷ lệ thấp hơn, nhưng do khác tập nên chỉ dùng để theo dõi tiến trình, không dùng để tuyên bố benchmark công bằng.

## 6. Vì sao chưa dưới 10%?

Mục tiêu 10% nghĩa là hard fallback phải nhỏ hơn:

```text
14036 x 10% = 1403.6
```

Vì số ảnh là số nguyên, cần còn tối đa 1,403 ảnh fail. Hiện có 1,974 ảnh, nên cần cứu thêm ít nhất:

```text
1974 - 1403 = 571 ảnh
```

Để cứu thêm 571 ảnh, cần cải thiện thuật toán segmentation, ví dụ:

- thêm một nhánh threshold/preprocessing thay thế;
- thử morphology khác cho ảnh nền sáng hoặc tương phản thấp;
- xử lý riêng nhóm ảnh chạm frame/cassette;
- kiểm tra các ảnh fail bằng montage để biết chúng fail vì lý do nào;
- chỉ sau khi kiểm tra hình ảnh mới cân nhắc rescue rộng hơn.

Không nên chỉ đổi thẳng thành `max_borders=4`, vì như vậy có nguy cơ nhận cassette hoặc vùng nền làm bàn tay. Khi đó tỷ lệ fallback có thể giảm trên giấy nhưng ROI sai sẽ tăng, ảnh hưởng trực tiếp đến model.

## 7. Train lại có làm giảm fallback không?

Không. Tỷ lệ fallback được tính trong bước segmentation/cache, trước khi model được train. Train ConvNeXt không thể làm cho segmentation tìm được bbox nhiều hơn.

Train lại chỉ trả lời một câu hỏi khác:

> Với cache ROI mới, model dự đoán tuổi xương có tốt hơn không?

Vì vậy thứ tự đúng là:

1. Cải thiện fallback nếu muốn giảm tỷ lệ dưới 10%.
2. Chốt cache và kiểm tra ảnh/mask.
3. Train lại model trên cache đã chốt.
4. Đánh giá MAE trên validation/test.

Không được dùng MAE của lần train cũ để tuyên bố cache fallback mới tốt hơn, vì model cũ chưa được train trên cache mới.

## 8. Cache đã tạo

- Mask manifest: `c3_roi/cache/C3_MASK_REUSE_FALLBACK_R2/mask_manifest.csv`
- ROI manifest: `c3_roi/cache/C3_ROI_FALLBACK_R2/roi_manifest.csv`
- Test ROI manifest: `c3_roi/cache/C3_ROI_FALLBACK_R2_TEST/test_roi_manifest.csv`
- Train: 12,611 ảnh.
- Official validation: 1,425 ảnh.
- Test khóa: 200 ảnh.
- Mask files có mặt: 14,036/14,036.
- ROI files có mặt: 14,036/14,036.
- Test ROI files có mặt: 200/200.
- Tất cả ROI được đánh dấu readable.

Cache cũ không bị ghi đè.

## 9. Kiểm tra và kết luận

Đã xác minh:

- 11 test fallback/ROI pass.
- Các module fallback/cache compile thành công.
- `git diff --check` pass.
- Bốn ảnh chạm biên được kiểm tra riêng đều đi vào nhánh `border_rescue` với bbox hợp lệ.
- 14,036 dòng của mask manifest và ROI manifest khớp nhau.
- Test R2 có 200/200 dòng; 144 `mask_bbox` và 56 `global_fallback`.

### Kết luận ngắn

Fallback hiện tại đã an toàn và rõ ràng hơn: strict trước, rescue sau, fail thì dùng toàn ảnh thay vì loại mẫu hoặc crop bừa. Trên test khóa, tỷ lệ đã giảm **33.00% → 28.00%**; trên development, tỷ lệ đã giảm **18.49% → 14.06%**. Cache đã sẵn sàng cho bước train sau này. Tuy nhiên hard fallback vẫn trên 10%, nên nếu mục tiêu bắt buộc là dưới 10% thì cần thêm một vòng cải thiện segmentation; train lại một mình sẽ không giải quyết được tỷ lệ này.
