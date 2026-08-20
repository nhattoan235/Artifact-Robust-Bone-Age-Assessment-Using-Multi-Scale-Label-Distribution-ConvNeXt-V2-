# P5 – Giao thức xác nhận nhiều seed

**Khóa trước khi chạy seed mới:** 2026-08-15  
**Test set:** không sử dụng.

## Mục tiêu

Xác nhận liệu auxiliary label-distribution learning (D3) có cải thiện ổn định so với baseline D0 hay kết quả seed 42 chỉ là dao động ngẫu nhiên.

## Cấu hình và seed

- Seed khóa: `17`, `42`, `123` cho cả D0 và D3.
- D0 seed 42 tái sử dụng `P2_A2_LIGHT_FLIP_CONVNEXT_TINY_SEED42`.
- D3 seed 42 tái sử dụng `P4_D3_CONVNEXT_TINY_LDL_SIGMA2_LAMBDA02_SEED42`.
- Chỉ train bốn run còn thiếu: D0-17, D3-17, D0-123, D3-123.
- Mọi run giữ nguyên official validation 1.425 ảnh, preprocessing `none`, augmentation A2, input 512, effective batch 36, optimizer, scheduler, loss và early stopping.

## Endpoint khóa

1. **Primary:** D3 fused prediction `0,5 regression + 0,5 distribution expectation`, giữ đúng primary của P4.
2. **Secondary đã khai báo trước seed mới:** regression-only từ cùng checkpoint D3. Endpoint này bắt nguồn từ phân tích exploratory seed 42 và phải được ghi rõ là adaptive; không được trình bày như primary P4.

Checkpoint D3 tiếp tục được chọn bằng validation MAE fused để seed 42 có thể tái sử dụng và mọi seed dùng cùng một quy tắc.

## Thống kê và quyết định

- Tính MAE/RMSE/subgroup cho từng seed và từng config.
- Tính paired delta trên cùng ảnh: `MAE_D3 - MAE_D0`; số âm có lợi cho D3.
- Báo cáo mean, SD và range của MAE/delta qua ba seed.
- Cluster bootstrap 10.000 lần trên image ID; mỗi lần lấy mẫu ảnh rồi trung bình delta qua ba seed.
- Giữ LDL trong pipeline chính nếu primary fused cải thiện trung bình ít nhất `0,10` tháng, có lợi ở ít nhất 2/3 seed và bootstrap/nhóm con không cho tín hiệu bất ổn rõ rệt.
- Nếu primary không đạt nhưng regression-only đạt đầy đủ cùng tiêu chí, có thể đưa regression-only sang phase tiếp theo như một quyết định adaptive đã công bố; fused vẫn phải được báo cáo.
- Nếu cả hai không đạt, chọn D0 đơn giản hơn.

## Thứ tự chạy

1. `P5_D0_CONVNEXT_TINY_SEED17`
2. `P5_D3_CONVNEXT_TINY_LDL_SEED17`
3. `P5_D0_CONVNEXT_TINY_SEED123`
4. `P5_D3_CONVNEXT_TINY_LDL_SEED123`

Không khởi chạy run kế tiếp trước khi run hiện tại early stop/kết thúc, checkpoint hợp lệ và không có lỗi đỏ.

## Trạng thái thực thi

### D0 seed 17 – hoàn thành

- Run: `P5_D0_CONVNEXT_TINY_SEED17`.
- Early stop epoch 19; best epoch 11, validation MAE `6,31950` tháng.
- Chỉ có warning cam prediction range/overfit; không lỗi đỏ, collapse, NaN/Inf/OOM hoặc skipped batch.
- Config hash: `d369d6d7c411c09c23c12cb67bdc22e92080610e9754e4556210ddd8969f47d9`.
- Log: `p1_baseline/P5_D0_SEED17_launcher.stdout.log`.

### D3 seed 17 – hoàn thành

- Run: `P5_D3_CONVNEXT_TINY_LDL_SEED17`.
- Config hash: `9ef0879ebe153f88f248c7af4c1cb3e52169d31ddb872d34635a7267cacf744e`.
- Early stop epoch 19; best epoch 11, fused MAE `6,24066` tháng.
- Fused delta so với D0-17: `-0,07884`, paired bootstrap 95% CI `[-0,21105; +0,05364]`.
- Regression-only adaptive MAE `6,29524`, delta `-0,02426`, CI `[-0,16177; +0,11447]`.
- Không lỗi đỏ, collapse, NaN/Inf/OOM hoặc skipped batch.
- Log: `p1_baseline/P5_D3_SEED17_launcher.stdout.log`.

### D0 seed 123 – hoàn thành

- Run: `P5_D0_CONVNEXT_TINY_SEED123`.
- Config hash: `7209e3e6dada60d3ff91e258947185c5e1efe41f3fcf1d7f1c3946aba6de0a02`.
- Preflight PASS 7/7; smoke/resume PASS.
- Early stop epoch 27; best epoch 19, validation MAE `6,28259` tháng.
- Không lỗi đỏ, collapse, NaN/Inf/OOM hoặc skipped batch; warning cuối phản ánh overfitting sau checkpoint tốt nhất.
- Log: `p1_baseline/P5_D0_SEED123_launcher.stdout.log`.

### D3 seed 123 – hoàn thành

- Run: `P5_D3_CONVNEXT_TINY_LDL_SEED123`.
- Config hash: `22c003a8f0e01bc76f7ae3249f813d0e2f93d7e832575020f0d46e889d997777`.
- Preflight PASS 7/7; khởi chạy lúc 23:06 ngày 2026-08-15.
- Log đầu epoch 1 ổn định, không NaN/Inf/OOM hoặc skipped batch.
- Log: `p1_baseline/P5_D3_SEED123_launcher.stdout.log`.

## Kết luận P5

- D3 seed 123 early stop epoch 25; best epoch 17, fused MAE `6,28917` tháng.
- D3 fused cải thiện trung bình `0,03721` tháng qua ba seed, bootstrap 95% CI `[-0,13521; +0,05863]`, có lợi ở 2/3 seed.
- Regression-only adaptive cải thiện trung bình `0,01932` tháng, bootstrap 95% CI `[-0,11270; +0,07069]`, có lợi ở 2/3 seed.
- Cả hai không đạt ngưỡng `0,10` tháng và CI đều chứa 0; chọn D0 đơn giản hơn cho P6.
- Chi tiết: `p5_seed_confirmation/P5_HANDOFF.md` và `p5_seed_confirmation/P5_AGGREGATE.json`.
