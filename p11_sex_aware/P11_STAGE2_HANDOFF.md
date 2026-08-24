# P11 Giai đoạn 2 – sex-aware screening handoff

> **Ngày hoàn tất:** 2026-08-22  
> **Trạng thái:** PASS vận hành; E2 không đạt gate khoa học  
> **Phạm vi:** official train 12.611 / validation 1.425, seed 42  
> **Test policy:** không đọc hoặc dùng RSNA test để chọn mô hình

## 1. Protocol đã khóa

- E0 image-only; E1 sex embedding control; E2 shared backbone với hai output M/F.
- Giữ cố định split, seed, augmentation, loss, optimizer, scheduler, batch và
  early stopping; paired bootstrap 10.000 lần trên đúng 1.425 ID.
- Gate E2: overall cải thiện ≥0,10 tháng; hoặc female cải thiện ≥0,20 tháng
  và overall không xấu quá 0,05 tháng.

## 2. Tính toàn vẹn vận hành

- E0/E2 qua GPU interrupt/resume; split/config/code hash và optimizer state khớp.
- Hai run early-stop bình thường; không NaN/Inf/OOM; peak VRAM 4.402 MiB.
- E0 config hash: `eb6c43a67bcad42623dc9d3b8ef7408392b2610c5057a814420032c1457b72ce`.
- E2 config hash: `60e44d0710c59129ebe1e5de5f76fda0da40694ea30267d30bbf9bf6bd7955a8`.

## 3. Kết quả checkpoint tốt nhất

| Model | Best/stop epoch | MAE | RMSE | Median AE | Female | Male |
|---|---:|---:|---:|---:|---:|---:|
| E0 image-only | 22/30 | 7,4717 | 9,9279 | 6,0 | 7,7686 | 7,2213 |
| E1 sex embedding | 12/20 | 6,1848 | 8,4865 | 5,0 | 6,3891 | 6,0125 |
| E2 dual-output | 15/23 | 6,1571 | 8,3678 | 4,5 | 6,3839 | 5,9657 |

`best_epoch` trong `run_state.json` là zero-based; bảng dùng epoch one-based.

## 4. Phân tích paired và quyết định

- E0−E1: **+1,2869 tháng**, CI [+1,0114; +1,5667].
- E0−E1 nữ/nam: +1,3795 / +1,2088; cả hai CI hoàn toàn >0.
- E2−E1: **−0,0277 tháng**, CI [−0,1831; +0,1254].
- E2−E1 nữ: −0,0052, CI [−0,2214; +0,2118].
- E2−E1 nam: −0,0467, CI [−0,2549; +0,1645].

E2 không đạt gate: cải thiện overall 0,0277 <0,10 và nữ 0,0052 <0,20.
Giữ E1; không chạy seed bổ sung, OOF hoặc E3 cho E2; không dùng test để cứu
candidate.

> Trong recipe ConvNeXt đã khóa, thông tin giới tính có giá trị dự đoán rõ ràng,
> nhưng hai output head không tạo lợi ích ổn định so với sex embedding.

## 5. Artifact và bước tiếp theo

- `p11_sex_aware/analysis/e0_vs_e1_seed42.json`
- `p11_sex_aware/analysis/e2_vs_e1_seed42.json`
- Runtime run directories nằm dưới `p11_sex_aware/runs/` và không đưa vào Git.

Không mở E3. Chuyển sang P12 phân tích TTA disagreement–error trên 14.036 OOF,
phân tầng sex × age và không truy cập RSNA test.
