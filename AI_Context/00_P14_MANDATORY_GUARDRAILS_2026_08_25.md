# PHỤ LỤC BẮT BUỘC — GUARDRAIL THỐNG KÊ VÀ TÁI LẬP P14

> Đọc ngay sau 00_CURRENT_DECISIONS_P14_HANDOFF_2026_08_25.md.
> Nếu có cách hiểu mơ hồ, phụ lục này được ưu tiên cho các vấn đề thống kê, provenance và claim.

## 1. Trạng thái chính xác của baseline 6,075568541

MAE 6,075568541 là kết quả outer-5-fold cross-fitted của ba nguồn E1-TTA, EXP006-TTA và C3.

Trọng số không âm, tổng bằng 1 được học trên bốn outer fold và áp dụng lên fold còn lại. Giá trị trung bình xấp xỉ 0,430/0,321/0,249 chỉ dùng để mô tả.

Không được dùng fixed weight 0,43/0,32/0,25 trên toàn OOF rồi gọi metric tương ứng là cross-fitted. Fixed weight đó chịu ảnh hưởng của toàn OOF và có thể lạc quan.

Khi tái tính phải ghi:

- solver và phiên bản thư viện;
- objective MAE;
- initialization;
- bounds không âm;
- constraint tổng bằng 1;
- tolerance và max iteration;
- outer-fold definition;
- join one-to-one theo image_id;
- row count 14.036;
- equality của target, sex và source ID;
- prediction/source SHA-256.

## 2. Cách diễn giải paired bootstrap

Delta được định nghĩa là:

MAE của candidate trừ MAE của baseline trên cùng image_id.

Khi candidate tốt hơn, delta âm. Gate thống kê đúng là upper bound của paired 95% CI nhỏ hơn 0.

CI xấp xỉ [−0,1603; −0,1092] đã nêu:

- resample prediction pairs theo image_id;
- 10.000 bootstrap;
- seed 2026;
- conditional trên các OOF prediction đã cố định.

CI này không bao phủ:

- stochastic training uncertainty;
- lựa chọn split;
- lựa chọn model sau nhiều thử nghiệm;
- lựa chọn preprocessing;
- multiple testing;
- việc đã xem các kết quả OOF trước đó.

Vì vậy đây là bằng chứng kỹ thuật mạnh cho prediction set hiện có, nhưng không được gọi là một confirmatory CI bao phủ toàn bộ quy trình adaptive research.

## 3. Comparator chính của P14

Primary comparator:

- nested OOF 6,075568541.

Secondary comparator:

- E1-TTA 6,210446.

P14 không đạt mục tiêu chỉ vì tốt hơn E1-TTA. Nó phải tốt hơn baseline nested hiện tại theo protocol không lạc quan.

Mốc OOF ≤5,95 là internal engineering gate. Mốc này không có nghĩa là đã vượt Deeplasia 3,87 hoặc Bram 3,68 vì hai mốc bài báo được tính trên RSNA test 200.

Mốc khoảng 5,83 chỉ là strong engineering target, tương ứng cải thiện xấp xỉ 0,25 tháng so với 6,075569; không phải ngưỡng thống kê do bài báo quy định.

## 4. Nhãn provenance bắt buộc cho kết quả lịch sử

### D3

D3-TTA 6,246542 và E1-TTA + D3-TTA 6,101345 hiện phải ghi:

**Historically reported / report-verified, chưa independently recomputable theo từng dòng trong audit hiện tại.**

Không đưa D3 vào row-level P14 fusion cho tới khi:

- có OOF prediction CSV;
- đủ 14.036 ID duy nhất;
- target/sex/fold khớp;
- prediction finite;
- metric được recompute;
- source checkpoint/code/config và CSV có SHA-256.

### P7

P7 6,316691 được xem là metric-level verified. Training provenance chưa hoàn chỉnh vì Fold 1 thiếu best-epoch metrics block trong audit lịch sử.

Không gọi P7 là full training-reproducible cho tới khi provenance Fold 1 được bổ sung hoặc giải thích bằng evidence có hash. Không sửa report gốc.

## 5. Cách diễn giải Deeplasia

Official online commit b010cff8693f64712e65dfba2c817438f2da09f5 có:

- data/parameters.yml;
- data/annotation.csv;
- data/rsna_test.csv.

Chỉ local clone từng audit là thiếu các file này. Không được viết rằng toàn bộ các file trên không được công khai.

Bit-exact reproduction vẫn bị chặn chủ yếu bởi:

- final checkpoints;
- đúng FastSurferCNN/FSCNN weight và final masks;
- split artifact độc lập được code/README tham chiếu;
- golden per-image predictions;
- đầy đủ environment, seed và execution provenance.

Cách gọi đúng:

**Protocol/code-faithful reproduction có thể xây dựng sau khi tái tạo segmentation và split; không kỳ vọng numerical identity.**

## 6. Protocol selection–confirmation cho P14

Để tiết kiệm GPU và hạn chế adaptive overfit:

1. Chỉ dùng P7 Fold 1 cho screening các candidate G1, L1 và L2.
2. Fold 1 chỉ là selection/promotion evidence, không phải kết luận khoa học.
3. Chọn tối đa hai candidate và khóa preprocessing, architecture, loss, augmentation, seed, budget và checkpoint rule.
4. Chạy candidate đã khóa trên Fold 2–5 mà không thay recipe theo metric trung gian.
5. Pooled Fold 2–5 là confirmation subset chính.
6. Full five-fold OOF được báo cáo bổ sung và phải ghi Fold 1 đã tham gia model selection.
7. Nếu tiếp tục chọn lại model sau khi xem Fold 2–5 thì confirmation status bị mất; phải ghi là adaptive development.

Mỗi ablation G0 → G1 → L1 → L2 phải one-factor-at-a-time trong vòng đầu:

- cùng backbone;
- cùng pretrained source;
- cùng optimizer/loss;
- cùng augmentation;
- cùng budget;
- cùng seed;
- chỉ khác input view/preprocessing đã khai báo.

Sau khi khóa ROI tốt nhất mới mở đúng một heterogeneous backbone. Phải chọn trước EfficientNetV2-S hoặc EfficientNet-B3; không chạy cả hai rồi chỉ giữ kết quả tốt hơn mà không hiệu chỉnh multiple selection.

## 7. Quy tắc inner-fit, outer-apply

Mọi thành phần học từ prediction phải tuân theo:

- calibration fit trên inner/training folds, apply trên held-out outer fold;
- ensemble weight fit trên inner/training folds, apply trên held-out outer fold;
- gating fit trên inner/training folds, apply trên held-out outer fold;
- threshold fit trên inner/training folds, apply trên held-out outer fold.

Không có held-out outer row nào được tham gia fit tham số áp dụng lên chính nó.

Sau khi nested evaluation hoàn tất và model đã khóa, có thể fit deployment weight trên toàn development OOF để áp dụng cho dữ liệu mới. Không dùng metric self-evaluated của deployment weight thay cho nested OOF metric.

## 8. QC phải fail-closed

Các ngưỡng localization áp dụng trên toàn bộ 14.036 development ID:

- fallback rate ≤1% nghĩa là số ảnh fallback chia 14.036 không vượt 0,01;
- foreground clipped ≤0,5% nghĩa là phần diện tích hand mask nguồn nằm ngoài crop chia tổng diện tích mask không vượt 0,005 cho mỗi ảnh, trừ khi protocol định nghĩa và khóa một aggregation khác;
- missing hoặc empty ROI phải bằng 0;
- toàn bộ per-ID QC phải được lưu, không chỉ summary.

Nếu gate fail:

- dừng trước long training;
- ghi reason theo từng ID;
- sửa preprocessing version;
- chạy lại full audit;
- không chuyển âm thầm sang full-image fallback;
- không loại ca khó hậu nghiệm để đạt ngưỡng.

## 9. Manifest tối thiểu bổ sung

Ngoài các trường trong handoff chính, phải khóa:

- dataset root và dataset manifest SHA-256;
- label unit là tháng và numeric precision;
- image_id schema, dedup rule và sort order;
- sex encoding;
- normalization mean/std và channel mapping;
- crop, mask, interpolation, fill và resize order;
- exact augmentation order/probability/range;
- exact TTA views và aggregation;
- optimizer/scheduler/warmup/precision/gradient settings;
- checkpoint selection and tie-break rule;
- raw prediction CSV schema;
- calibration/blend outer-fold mapping;
- CUDA, cuDNN và driver nếu có thể ghi nhận;
- seed cho Python, NumPy, PyTorch, dataloader và augmentation.

## 10. Claim cuối

RSNA test đã bị chạm. MAE 3,70 chỉ là aspiration thăm dò.

Muốn nói vượt Deeplasia phải tối thiểu:

- cùng cohort và inclusion criteria;
- cùng đơn vị/label precision;
- cùng metric implementation;
- frozen method;
- không chọn bằng test;
- CI phù hợp;
- nếu có prediction đối chứng, paired comparison.

Point estimate thấp hơn 3,87 không tự động chứng minh superiority có ý nghĩa thống kê.
