# Pilot C Full 5-Fold Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Mở rộng Pilot C từ Fold 1 sang đủ 5 folds, tạo OOF clean/artifact có thể so sánh ghép cặp với C3-ROI V2 gốc mà không dùng tập test để lựa chọn mô hình.

**Architecture:** Tổng quát hóa runner và evaluator B/C hiện tại để nhận `--fold`, nhưng chỉ đóng gói và chạy Pilot C ở Fold 2–5; Fold 1 dùng checkpoint và prediction đã hoàn tất. Mỗi fold có config, run ID, checkpoint mirror và báo cáo riêng trên Google Drive. Sau khi đủ năm fold, một aggregator ghép đúng 14.036 ID, kiểm tra mỗi ID xuất hiện đúng một lần, rồi tính MAE, subgroup, disagreement và paired bootstrap so với baseline trên cùng clean/artifact view.

**Tech Stack:** Python 3.13, PyTorch, timm ConvNeXt-Tiny, pandas, NumPy, TOML, Google Colab T4, Google Drive.

---

## Protocol đã khóa trước khi chạy

- Chỉ mở rộng **Pilot C**: mild artifact augmentation + consistency loss `0.30`.
- Giữ nguyên seed `42`, split, ConvNeXt-Tiny, ảnh `512`, sex embedding, optimizer, scheduler, augmentation, warmup `3`, ramp `5`, early stopping và toàn bộ hyperparameter Fold 1.
- Chỉ thay `train_manifest`, `val_manifest`, hash/count tương ứng, `run_id` và fold.
- Early stopping tiếp tục dựa trên clean validation MAE.
- Không đổi tham số sau khi nhìn kết quả Fold 2–5.
- Không đọc tập RSNA test 200 ảnh trong quá trình train, chọn checkpoint hoặc tổng hợp OOF.
- Không loại một fold chỉ vì kết quả kém; tất cả fold hợp lệ đều phải vào OOF.
- Pilot B giữ làm ablation Fold 1. Không tốn GPU train B thêm ở giai đoạn này và không tuyên bố consistency chắc chắn hơn B trên toàn bộ dữ liệu.

## Tiêu chí quyết định đã khóa

So sánh `Pilot C - C3-V2 baseline`, vì vậy delta âm là tốt hơn.

1. **Clean non-inferiority:** delta clean OOF không quá `+0.10 tháng`, đồng thời cận trên CI 95% không vượt `+0.10 tháng`.
2. **Artifact superiority:** artifact OOF cải thiện ít nhất `0.30 tháng` và cận trên CI 95% của delta nhỏ hơn `0`.
3. **Stability:** mean clean–artifact disagreement giảm ít nhất `40%` so với baseline.
4. **Subgroup:** cả nam và nữ đều cải thiện artifact; không nhóm tuổi nào giảm clean MAE quá `0.20 tháng` nếu có đủ mẫu.
5. Nếu đạt 1–4, khóa model và mới chạy test TTA một lần. Nếu không đạt, báo cáo Pilot C như một kết quả pilot/ablation và không tối ưu tiếp dựa trên test.

### Task 1: Tổng quát hóa config Pilot C cho Fold 2–5

**Files:**
- Create: `c3_roi/pilot_configs/C3_Z26_C3_ROI_V2_PILOT_C_FOLD_2_SEED_42.toml`
- Create: `c3_roi/pilot_configs/C3_Z26_C3_ROI_V2_PILOT_C_FOLD_3_SEED_42.toml`
- Create: `c3_roi/pilot_configs/C3_Z26_C3_ROI_V2_PILOT_C_FOLD_4_SEED_42.toml`
- Create: `c3_roi/pilot_configs/C3_Z26_C3_ROI_V2_PILOT_C_FOLD_5_SEED_42.toml`
- Modify: `c3_roi/test_build_pilot_bc_bundle.py`

**Step 1: Viết test kiểm tra mỗi config dùng đúng manifest, hash, count và run ID**

Giá trị phải lấy nguyên từ config C3-ROI V2 gốc:

| Fold | Train count | Validation count | Train hash prefix | Validation hash prefix |
|---|---:|---:|---|---|
| 2 | 11.229 | 2.807 | `f4270f3983ba` | `26124f6279aa` |
| 3 | 11.229 | 2.807 | `896d4fd4efb4` | `c894202bb22f` |
| 4 | 11.229 | 2.807 | `f82483c670fc` | `60440af4a7bb` |
| 5 | 11.229 | 2.807 | `413153e8bf86` | `23724152ada9` |

Test cũng phải khẳng định mọi trường khoa học khác giống hệt config Pilot C Fold 1.

**Step 2: Chạy test và xác nhận nó fail vì thiếu bốn config**

Run: `python -m unittest c3_roi.test_build_pilot_bc_bundle -v`

Expected: FAIL tại kiểm tra config Fold 2–5.

**Step 3: Tạo bốn config**

Run ID phải theo mẫu:

```text
C3_Z26_C3_ROI_V2_PILOT_C_V2_FOLD_<fold>_SEED_42
```

Drive mirror giữ nguyên:

```text
/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs
```

**Step 4: Chạy lại test**

Expected: PASS; hash/count của từng fold khớp config baseline gốc.

### Task 2: Cho runner nhận fold và resume độc lập

**Files:**
- Modify: `c3_roi/pilot_bc_runner.py`
- Modify: `c3_roi/test_pilot_bc_runner.py`

**Step 1: Viết test cho `--fold 1..5` và đường dẫn config**

Test các trường hợp:

- `pilot_config_path("C", 2)` trả config Fold 2.
- Fold ngoài `1..5` bị từ chối.
- B chỉ được phép Fold 1 trong bundle mới.
- C Fold 2–5 chọn checkpoint dựa trên đúng config hash và manifest hash của chính fold đó.
- Run đã `completed` hoặc `early_stopped` được skip; checkpoint không tương thích làm runner dừng.

**Step 2: Chạy test để thấy fail với runner khóa Fold 1**

Run: `python -m unittest c3_roi.test_pilot_bc_runner -v`

**Step 3: Thêm đối số `--fold`**

Runner phải:

- mặc định Fold 1 để tương thích notebook cũ;
- preflight đúng manifest fold được chọn;
- resume từ local hoặc Drive mirror;
- không ghi đè thư mục Fold 1;
- báo rõ `FRESH`, `RESUME`, `SKIP` hoặc lỗi checkpoint không tương thích.

**Step 4: Chạy lại test**

Expected: PASS toàn bộ test runner.

### Task 3: Tổng quát hóa evaluator clean/artifact theo fold

**Files:**
- Modify: `c3_roi/evaluate_pilot_bc.py`
- Modify: `c3_roi/test_evaluate_pilot_bc.py`

**Step 1: Viết test evaluator nhận `--fold`**

Mỗi báo cáo phải chứa `fold`, split hash, checkpoint hash, count, clean metrics, artifact metrics, disagreement, subgroup và CI artifact-minus-clean.

**Step 2: Bỏ hằng số Fold 1 khỏi protocol**

Không dùng `BASELINE_FOLD1_MAE` làm kết luận cho Fold 2–5. So sánh baseline ghép cặp sẽ được thực hiện ở aggregator.

**Step 3: Đặt tên output giữ nguyên trong run directory riêng**

```text
artifact_robustness_predictions.csv
artifact_robustness_report.json
```

**Step 4: Chạy test evaluator**

Run: `python -m unittest c3_roi.test_evaluate_pilot_bc -v`

Expected: PASS.

### Task 4: Tạo aggregator OOF baseline/C

**Files:**
- Create: `c3_roi/evaluate_pilot_c_5fold.py`
- Create: `c3_roi/test_evaluate_pilot_c_5fold.py`

**Step 1: Viết test integrity OOF**

Test phải fail khi:

- thiếu một fold;
- trùng `image_id` giữa các fold;
- tổng count khác `14.036`;
- target/sex giữa baseline và C không khớp;
- artifact view không cùng seed/recipe;
- prediction có NaN/Inf.

**Step 2: Viết test metric và bootstrap trên dữ liệu nhỏ cố định**

Kiểm tra dấu delta là `candidate - baseline`, CI tái lập theo seed, và phép tính disagreement reduction.

**Step 3: Implement aggregator**

Aggregator phải inference baseline và C bằng cùng evaluator view nếu baseline artifact prediction chưa được lưu. Nó lưu:

```text
C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_OOF_predictions.csv
C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_report.json
C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_subgroups.csv
```

Báo cáo gồm per-fold và pooled OOF cho clean/artifact, paired bootstrap 20.000 lần, disagreement, sex/age-bin và kết quả từng gate đã khóa.

**Step 4: Chạy test aggregator**

Run: `python -m unittest c3_roi.test_evaluate_pilot_c_5fold -v`

Expected: PASS.

### Task 5: Tạo bundle và notebook Colab riêng cho Fold 2–5

**Files:**
- Create: `c3_roi/build_pilot_c_5fold_bundle.py`
- Modify: `c3_roi/test_build_pilot_bc_bundle.py`
- Create: `C3_Z26_C3_ROI_V2_PILOT_C_5FOLD.ipynb`
- Create: `C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_CODE_V1.zip`
- Create: `C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_CODE_V1.sha256.txt`

**Step 1: Notebook dùng một biến `FOLD`**

```python
FOLD = 2  # lần lượt đổi 2, 3, 4, 5
assert FOLD in {2, 3, 4, 5}
```

Không tự loop cả bốn fold trong một runtime vì một lần Colab ngắt sẽ khó theo dõi và dễ hết quota. Mỗi runtime chỉ chạy một fold; runner tự resume nếu ngắt.

**Step 2: Setup kiểm tra ba ZIP trên Drive**

```text
C3_Z26_C3_ROI_T4_CODE_V2.zip
C3_Z26_COMBO_V2_FINAL.zip
C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_CODE_V1.zip
```

**Step 3: Cell train và evaluator truyền đúng fold**

```text
python -u pilot_bc_runner.py --pilot C --fold <FOLD>
python -u evaluate_pilot_bc.py --pilot C --fold <FOLD>
```

**Step 4: Thêm cell kiểm tra file Drive sau mỗi fold**

Bắt buộc có:

- `best_mae.ckpt`
- `last.ckpt`
- `run_state.json`
- `train.log`
- `metrics.jsonl`
- `artifact_robustness_predictions.csv`
- `artifact_robustness_report.json`

**Step 5: Build và kiểm tra ZIP**

Run: `python c3_roi/build_pilot_bc_bundle.py --output C3_Z26_C3_ROI_V2_PILOT_C_5FOLD_CODE_V1.zip`

Expected: ZIP test pass, SHA-256 được in và ghi vào file checksum.

### Task 6: Chạy kiểm thử trước khi upload

**Files:**
- Test: `c3_roi/test_pilot_bc_runner.py`
- Test: `c3_roi/test_evaluate_pilot_bc.py`
- Test: `c3_roi/test_evaluate_pilot_c_5fold.py`
- Test: `c3_roi/test_build_pilot_bc_bundle.py`
- Test: `p1_baseline/test_pilot_artifacts.py`

**Step 1: Chạy bộ test mục tiêu**

```text
python -m unittest \
  c3_roi.test_pilot_bc_runner \
  c3_roi.test_evaluate_pilot_bc \
  c3_roi.test_evaluate_pilot_c_5fold \
  c3_roi.test_build_pilot_bc_bundle \
  p1_baseline.test_pilot_artifacts -v
```

Expected: tất cả PASS.

**Step 2: Kiểm tra checksum và nội dung bundle**

Expected: đủ config Fold 1–5, runner, evaluator, aggregator, notebook và dependencies; không chứa checkpoint/dataset.

### Task 7: Lịch chạy Colab Fold 2–5

**Files:**
- Runtime output: Google Drive `RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs/`

Mỗi fold dự kiến khoảng **4–5 giờ train** trên T4 và thêm khoảng **5–10 phút evaluator**. Tổng còn lại khoảng **16–20 giờ GPU**.

| Phiên | FOLD | Hành động | Điều kiện kết thúc |
|---|---:|---|---|
| 1 | 2 | Setup, train/resume, evaluator | report và predictions đã có trên Drive |
| 2 | 3 | Runtime mới, đổi `FOLD = 3` | report và predictions đã có trên Drive |
| 3 | 4 | Runtime mới, đổi `FOLD = 4` | report và predictions đã có trên Drive |
| 4 | 5 | Runtime mới, đổi `FOLD = 5` | report và predictions đã có trên Drive |

Không chạy đồng thời hai fold trên một T4. Có thể disconnect sau khi evaluator của fold hiện tại in đủ hai dòng `SAVED`.

Theo dõi terminal bằng run ID tương ứng:

```text
tail -n 40 -F /content/C3_Z26_C3_ROI_V2_PILOTS/runs/C3_Z26_C3_ROI_V2_PILOT_C_V2_FOLD_<FOLD>_SEED_42/train.log
```

Nếu local runtime mất log, dùng cùng đường dẫn dưới Drive mirror.

### Task 8: Tổng hợp OOF và ra quyết định

**Files:**
- Output: Google Drive `RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/OOF/`

**Step 1: Xác minh đủ năm checkpoint/report/prediction**

Expected count: Fold 1 có `2.808`, Fold 2–5 mỗi fold `2.807`, tổng `14.036`.

**Step 2: Chạy aggregator với T4**

GPU cần thiết nếu phải inference lại baseline/C; bootstrap sau đó chạy CPU.

**Step 3: Đọc gate trong report JSON**

- Nếu đạt cả bốn tiêu chí: chọn C làm model robustness chính.
- Nếu clean đạt nhưng artifact không đạt: giữ baseline làm model chính, C thành negative/partial result.
- Nếu artifact đạt nhưng clean non-inferiority không đạt: thử ensemble baseline+C, nhưng weight chỉ được chọn bằng OOF.

**Step 4: Khóa mọi lựa chọn trước test**

Ghi model, checkpoint hash, TTA recipe, ensemble weight (nếu có) và preprocessing hash vào report khóa.

### Task 9: Test xác nhận cuối cùng

Chỉ thực hiện sau Task 8 nếu gate đạt.

- Chạy đúng manifest test 200 ảnh và TTA10 đã dùng cho C3-V2.
- Báo riêng raw 5-fold ensemble và TTA ensemble.
- So sánh với C3-V2 test TTA đã khóa `4.251692` và mốc mục tiêu `4.20`.
- Không thay weight, checkpoint hoặc TTA sau khi thấy test.
- Lưu prediction từng fold/view, đúng 200 ID, config hash, preprocessing hash và checkpoint SHA-256.

## Dung lượng và sao lưu

- Một run hiện giữ khoảng bốn checkpoint xấp xỉ `321 MB`, tức khoảng `1.28 GB/fold`.
- Bốn fold mới cần khoảng `5.1 GB`; toàn bộ năm fold khoảng `6.4 GB`, chưa tính CSV/log.
- Giữ `best_mae.ckpt`, `last.ckpt` và periodic cho đến khi OOF hoàn tất và backup đã kiểm tra.
- Sau khi xác minh backup, có thể xóa periodic/last và giữ `best_mae.ckpt` cùng toàn bộ report/prediction/hash. Không xóa tự động trong notebook.
