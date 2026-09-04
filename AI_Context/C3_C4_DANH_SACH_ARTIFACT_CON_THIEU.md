# Danh sách artifact còn thiếu để kiểm tra C3-ROI, P15 7-view và C4

> Ngày tổng hợp: 2026-09-04
> Mục tiêu: thu thập đủ bằng chứng để audit tính đúng đắn, khả năng tái lập và độ bền vững của phương pháp.
> Phạm vi: C3-ROI, bộ dữ liệu P15 gồm một ảnh toàn bàn tay và sáu ROI, cùng mô hình C4. Không kiểm tra hoặc thay đổi P14.

## 1. Các điểm đã xác nhận

- C3-ROI nhận một ảnh ROI rộng của toàn bàn tay, không nhận trực tiếp bảy ảnh.
- C4 là mô hình `global + six ROI`, mỗi ca gồm một ảnh toàn bàn tay và sáu ROI giải phẫu.
- Bộ development 7-view đã chốt 14.024/14.036 ca, tương đương coverage 99,9145%; 12 ca không đủ điều kiện giải phẫu đã được loại.
- Báo cáo C4 hiện có ghi OOF MAE 6,639306 tháng trên 14.024 ca và `test_used=false`.
- E1, C3-ROI và ensemble E1+C3 cũ được đánh giá trên 14.036 ca; khi giới hạn prediction của chúng về cùng 14.024 ID, MAE lần lượt là 6,305843; 6,425632 và 6,164757 tháng.
- CSV test tổng hợp C3 hiện ghi đè cột prediction C3 bằng prediction ensemble. JSON metric vẫn đúng, nhưng CSV cần được tái tạo từ prediction từng fold trước khi dùng cho báo cáo chính thức.

## 2. Mức ưu tiên A — bắt buộc tìm trước

### A1. Prediction OOF gốc của C4 V2

Trạng thái: **chưa thấy trong repository hiện tại**.

Tìm trong Google Drive/Colab, đặc biệt quanh:

```text
MyDrive/data/c4_multi_roi_runs/
C4_MULTI_ROI_V2/
C4_MULTI_ROI_V2_FOLD_1 ... C4_MULTI_ROI_V2_FOLD_5
```

Artifact cần tìm cho mỗi fold:

- [ ] `val_predictions_best.csv`
- [ ] `config_resolved.json` hoặc `config_resolved.yaml`
- [ ] `run_state.json`
- [ ] `metrics.jsonl`
- [ ] `environment.json`
- [ ] `train.log`
- [ ] `best_mae.ckpt` hoặc ít nhất SHA-256 của checkpoint
- [ ] `last.ckpt` hoặc ít nhất SHA-256 của checkpoint

Artifact tổng hợp cần tìm:

- [ ] File prediction OOF đủ 14.024 dòng
- [ ] Script tổng hợp năm fold
- [ ] Báo cáo per-fold
- [ ] Báo cáo paired-bootstrap
- [ ] Manifest SHA-256 `d56634f084163bd360a3b43847958869fdc5fa3dcf19d1aae1836f441e612b92`

Mục đích kiểm tra:

1. Đủ năm fold và mỗi ID xuất hiện đúng một lần trong OOF.
2. Không thiếu/trùng ID, không có prediction NaN/Inf.
3. Tái tính được MAE 6,639306 và RMSE 8,815673.
4. Xác minh checkpoint được chọn hoàn toàn bằng validation.
5. Tính paired CI với E1/C3 trên cùng 14.024 ID.

### A2. Config và provenance chính xác của C4 V2

Trạng thái: **đang có bất nhất V1/V2**.

Hiện tại:

- `protocol_v1.json` khóa 14.036 ID.
- Năm config Colab hiện có cũng yêu cầu 14.036 dòng và mang tên V1.
- Báo cáo kết quả lại mang tên C4 V2 và dùng 14.024 dòng.

Cần tìm:

- [ ] Protocol C4 V2 được khóa trước khi train
- [ ] Năm config C4 V2 thực tế đã chạy
- [ ] Code commit/hash dùng khi train V2
- [ ] Manifest path và manifest hash thực tế
- [ ] Ngày chạy, GPU, Python, PyTorch, torchvision và CUDA
- [ ] Seed và trạng thái deterministic
- [ ] Bằng chứng resume không làm thay đổi fold/config

### A3. Manifest và audit gốc của development 7-view

Đường dẫn được ghi trong báo cáo:

```text
E:\Data_200test_cut\p15_roi_7views\outputs\development_7views_v1
```

Cần tìm:

- [ ] `roi_manifest.csv`
- [ ] `roi_manifest_long.csv`
- [ ] `dataset_audit.json`
- [ ] `split_layout_audit.json`
- [ ] `manual_roi_reviewed_v1.json`
- [ ] `manual_overlay_report.json`
- [ ] `raw_detections.jsonl`
- [ ] `exclusion_manifest.csv`
- [ ] File ghi hash của toàn bộ manifest/dataset nếu có

Tiêu chí cần xác minh:

- 14.024 ID duy nhất.
- Đúng bảy view cho mỗi ID, tổng cộng 98.168 PNG.
- Train 12.601 ID và validation 1.423 ID.
- Không có ID giao nhau giữa split.
- Không có ảnh thiếu, hỏng, trùng nội dung hoặc gắn sai ID.
- Sáu ROI đều được sinh từ đúng ảnh global của cùng ca.
- Manual ROI ghi đè detector đúng theo hồ sơ review.

### A4. Manifest và audit của bộ test 200 ca × 7 ảnh

Trạng thái: **người dùng xác nhận bộ ảnh tồn tại, nhưng chưa thấy manifest, audit hoặc metric trong artifact hiện tại**.

Cần tìm:

- [ ] Đường dẫn chính xác của dataset
- [ ] `test_manifest.csv` hoặc tên tương đương
- [ ] `test_roi_manifest.csv` hoặc manifest long theo view
- [ ] `test_audit.json` hoặc `dataset_audit.json`
- [ ] Báo cáo số folder và số file
- [ ] Source-image SHA-256 và mapping từ ảnh gốc sang sáu ROI
- [ ] QC status của từng ROI
- [ ] Danh sách ca bị loại hoặc thiếu view, nếu có
- [ ] Prediction và report nếu C4 đã infer trên bộ này

Tiêu chí bắt buộc:

- Đúng 200 `image_id` của RSNA test gốc.
- Mỗi ID có đúng bảy file:

```text
00_whole.png
01_wrist_carpal.png
02_mcp_thumb.png
03_mcp_index.png
04_mcp_middle.png
05_mcp_ring.png
06_mcp_little.png
```

- Tổng cộng đúng 1.400 ảnh nếu không có ca bị loại.
- Sex và ground truth ghép đúng ID.
- Ground truth test không được dùng trong quá trình phát hiện/cắt ROI.
- Mỗi ca chỉ được tính là một đơn vị thống kê; không xem 1.400 ảnh là 1.400 mẫu độc lập.

## 3. Mức ưu tiên B — cần cho audit C3-ROI

### B1. Manifest ROI/fallback của C3 development

Cần tìm:

```text
c3_roi/_drive_upload/C3_ROI_V1/cpu_artifacts/audit_summary.json
c3_roi/_drive_upload/C3_ROI_V1/cpu_artifacts/roi_manifest.csv
```

Hoặc artifact tương đương chứa:

- [ ] `image_id`
- [ ] `roi_mode`
- [ ] `fallback_reason`
- [ ] Tọa độ crop/bounding box
- [ ] Source-image SHA-256
- [ ] ROI-image SHA-256
- [ ] Fold, sex và target

Mục tiêu: phân tích riêng 11.441 ca bbox ROI và 2.595 ca full-image fallback.

### B2. Manifest ROI/fallback của C3 test

Cần tìm quanh:

```text
c3_roi/cache/C3_ROI_V1_TEST/
```

Artifact cần có:

- [ ] `roi_manifest.csv`
- [ ] `audit_summary.json`
- [ ] Danh sách 134 ca bbox ROI
- [ ] Danh sách 66 ca fallback
- [ ] Crop coordinates và fallback reason
- [ ] Hash nguồn và hash ROI

Mục tiêu:

1. So sánh MAE bbox với MAE fallback.
2. Kiểm tra fallback có tập trung theo tuổi/giới hay không.
3. Kiểm tra cleaned cải thiện nhóm bbox hay fallback.
4. Xác định domain shift khiến fallback tăng từ 18,49% lên 33,00%.

### B3. Tái tạo đúng CSV C3 test

Nguồn hiện còn:

```text
c3_roi/outputs/C3_ROI_V1_TEST/C3_fold_1_predictions.csv
...
c3_roi/outputs/C3_ROI_V1_TEST/C3_fold_5_predictions.csv
p8_test_ensemble/outputs/P8_ensemble_predictions.csv
```

CSV đúng cần có ba cột độc lập:

- [ ] `c3_prediction_months`
- [ ] `e1_prediction_months`
- [ ] `ensemble_50_50_prediction_months`

Sau khi tái tạo cần:

- [ ] Đối chiếu lại MAE C3 4,337267
- [ ] Đối chiếu lại MAE E1 4,730321
- [ ] Đối chiếu lại MAE ensemble 4,454661
- [ ] Ghi script version/hash và không thay đổi model/weight

## 4. Mức ưu tiên C — hồ sơ eligibility của 12 ca bị loại

Người dùng xác nhận 12 ca bị loại vì không đủ cấu trúc giải phẫu như thiếu ngón hoặc không thể xác đủ sáu ROI.

Cần tìm:

- [ ] Ảnh review/overlay của 12 ca
- [ ] Phiếu hoặc JSON quyết định review ban đầu
- [ ] Thời điểm quyết định loại
- [ ] Quy tắc eligibility áp dụng thống nhất
- [ ] Xác nhận quyết định được đưa ra trước khi xem prediction/MAE C4
- [ ] Lý do chuẩn hóa cuối cùng cho từng ID

Các điểm cần thống nhất trong hồ sơ:

- `scope` hiện ghi `c3_roi_primary`, trong khi dataset 7-view được dùng cho P15/C4.
- Lý do hiện ghi hỗn hợp: foot, missing MCP, severe deformity, hardware và wrist không quan sát được.
- Cần tách rõ lỗi dữ liệu (`wrong anatomy/foot`) với ca bàn tay bất thường nhưng có ý nghĩa robustness (`missing finger`, `hardware`, `deformity`).

Chính sách báo cáo đề xuất:

- Primary cohort: X-quang bàn tay chuẩn, đủ cấu trúc, n=14.024.
- Coverage: 99,9145%; abstention/exclusion: 0,0855%.
- Giữ 12 ca thành failure/OOD set, không xóa artifact.
- Không tuyên bố mô hình áp dụng được cho bàn tay thiếu ngón, biến dạng nặng hoặc hardware nếu chưa đánh giá riêng.

## 5. Các phân tích cần thực hiện sau khi thu thập đủ artifact

### 5.1. Tính toàn vẹn dữ liệu

- [ ] ID duy nhất và khớp giữa image, manifest, prediction và label
- [ ] Không leakage giữa train/validation/test
- [ ] Không duplicate SHA giữa split
- [ ] Đúng thứ tự sáu ROI
- [ ] Không sai sex/target/fold
- [ ] Không có ảnh hoặc prediction non-finite/hỏng

### 5.2. Thống kê mô hình

Cho E1, C3, E1+C3 và C4 trên cùng cohort:

- [ ] MAE
- [ ] RMSE
- [ ] Median absolute error
- [ ] Accuracy ±6/±12/±18 tháng
- [ ] Signed bias
- [ ] Metric từng fold
- [ ] Metric theo giới
- [ ] Metric theo nhóm tuổi
- [ ] Paired-bootstrap 95% CI
- [ ] Tỷ lệ ca từng mô hình cải thiện/xấu đi
- [ ] Kiểm tra outlier và prediction collapse

### 5.3. Robustness

- [ ] Original so với cleaned theo cùng ID
- [ ] ROI bbox so với full-image fallback
- [ ] Detector ROI so với manual ROI
- [ ] Độ nhạy với rotation, brightness, contrast và gamma
- [ ] Độ nhạy với crop jitter
- [ ] Độ nhạy khi thiếu một ROI
- [ ] Prediction shift giữa các điều kiện ảnh
- [ ] TTA disagreement và quan hệ với absolute error
- [ ] Failure/OOD coverage và quy tắc abstention

### 5.4. Khả năng tái lập

- [ ] Config/model/data/code hash
- [ ] Seed và trạng thái RNG
- [ ] Environment đầy đủ
- [ ] Checkpoint/resume audit
- [ ] Script aggregate độc lập
- [ ] Tái tính metric từ CSV không phụ thuộc report JSON
- [ ] Xác nhận hai lần build cho cùng kết quả/hash

## 6. Các câu hỏi cần trả lời trước khi chốt phương pháp

1. C4 V2 đã train bằng config nào và artifact năm fold đang ở đâu?
2. Bộ test 200 × 7 ảnh mới chỉ được tạo hay C4 đã infer trên đó?
3. Nếu đã infer, prediction cấp ca được fusion như thế nào?
4. C4 có đủ một prediction cho mỗi ca hay có ca thiếu view/bị loại?
5. C3 đạt MAE tốt trên test chủ yếu ở bbox ROI hay fallback?
6. Kết quả cleaned của C3 đến từ đa số ca hay một nhóm ít ca cải thiện mạnh?
7. Có thể so sánh E1/C3/C4 trên đúng cùng 14.024 ID không?
8. Có đủ bằng chứng để tái tạo MAE C4 từ raw prediction không?
9. Phạm vi ứng dụng cuối cùng có loại trừ bàn tay thiếu ngón/hardware/deformity không?
10. Khi một hoặc nhiều ROI không hợp lệ, hệ thống sẽ từ chối hay dùng masked fusion?

## 7. Thứ tự thu thập khuyến nghị

1. Năm file `val_predictions_best.csv` của C4 V2.
2. Config V2, manifest V2 và hash tương ứng.
3. `roi_manifest.csv`, `roi_manifest_long.csv` và audit JSON của development 7-view.
4. Manifest/audit của bộ test 200 × 7 ảnh.
5. Prediction/report test C4 nếu đã chạy.
6. Manifest C3 development/test có `roi_mode` và `fallback_reason`.
7. Hồ sơ review của 12 ca bị loại.
8. Checkpoint và log đầy đủ sau khi metadata/prediction đã khớp.

## 8. Mẫu ghi khi tìm được artifact

Điền thêm một dòng cho mỗi artifact tìm được:

| Hạng mục | Đường dẫn thực tế | Kích thước | SHA-256 | Trạng thái | Ghi chú |
|---|---|---:|---|---|---|
| Ví dụ: C4 Fold 1 prediction |  |  |  | Chưa tìm thấy |  |
| C4 Fold 1 config |  |  |  | Chưa tìm thấy |  |
| C4 V2 manifest |  |  |  | Chưa tìm thấy |  |
| Development 7-view audit |  |  |  | Chưa tìm thấy |  |
| Test 7-view manifest |  |  |  | Chưa tìm thấy |  |
| Test 7-view audit |  |  |  | Chưa tìm thấy |  |
| C3 development ROI manifest |  |  |  | Chưa tìm thấy |  |
| C3 test ROI manifest |  |  |  | Chưa tìm thấy |  |

## 9. Điều kiện tối thiểu để chốt báo cáo khoa học

Chỉ nên chốt phương pháp sau khi đạt đủ:

- [ ] C4 V2 có config, manifest và raw OOF prediction tái lập được.
- [ ] E1/C3/C4 được so sánh trên cùng ID hoặc giới hạn so sánh được ghi rõ.
- [ ] C3 fallback được phân tích riêng.
- [ ] Bộ test 7-view được audit đủ 200 ID × 7 view.
- [ ] Mọi metric chính có paired CI khi phù hợp.
- [ ] 12 ca loại có tiêu chí label-blind, hồ sơ rõ và coverage được báo cáo.
- [ ] Không gọi ba điều kiện ảnh là ba cohort test độc lập.
- [ ] Không dùng kết quả test để điều chỉnh model, TTA hoặc ensemble weight.
- [ ] CSV prediction C3 test được sửa schema và kiểm tra lại số học.
- [ ] Claim cuối phù hợp với phạm vi bàn tay đủ cấu trúc giải phẫu.
