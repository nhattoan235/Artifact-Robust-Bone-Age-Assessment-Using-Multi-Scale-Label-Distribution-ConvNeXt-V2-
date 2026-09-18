# B/C trên RSNA test 200 ảnh — cùng recipe TTA 10-view

Đây là **inference**, không train lại. Recipe lấy nguyên từ C3-V2 đã khóa:
rotation `-10, -5, 0, +5, +10` độ, mỗi rotation chạy cả không lật và lật ngang;
trung bình đều 10 dự đoán mỗi fold. Code tiền xử lý và hàm inference dùng lại
`c3_roi/tta_test.py`. Checkpoint được chọn sẵn bằng clean validation MAE.

So sánh hợp lệ:

1. **C 5-fold** với **baseline 5-fold** trên cùng 200 ảnh, cùng 10-view.
2. **B Fold 1+5** với **baseline Fold 1+5**; thêm **C Fold 1+5** làm đối chứng.

Không so B hai-fold trực tiếp với baseline năm-fold. B chưa có Fold 2–4. Cả hai
kết quả test ở đây là **exploratory**: test 200 ảnh từng được xem trong các vòng
nghiên cứu trước, không dùng nó để đổi trọng số, view TTA, hay chọn checkpoint.

## Trước khi chạy trên Colab đang kết nối T4

Cần có trên Drive:

- `RSNA_DATA/C3_Z26_C3_ROI_V2_TTA_TEST/C3_Z26_C3_ROI_V2_TTA_TEST_predictions.csv`
  (baseline đã chạy, chứa 200 ID, giới tính, nhãn và dự đoán từng fold).
- `RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs/` với
  `best_mae.ckpt` và `config_resolved.yaml` cho C Fold 1–5, B Fold 1 và 5.

Trong runtime cần `/content/p1_baseline/model.py` và
`/content/C3_Z26_COMBO_V2/images/test/` có đúng 200 PNG. Nếu đã train B/C trong
runtime hiện tại, thường hai thư mục này đã có. Nếu không, dùng lại phần setup
code và data cũ trước; không cần train.

Upload gói nhỏ `C3_Z26_C3_ROI_V2_PILOT_BC_TEST_TTA_10VIEW_CODE_V1.zip` lên
`MyDrive/RSNA_DATA/`, rồi chạy cell này:

```python
from google.colab import drive
from pathlib import Path
import subprocess, sys, zipfile

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/RSNA_DATA')
pack = root / 'C3_Z26_C3_ROI_V2_PILOT_BC_TEST_TTA_10VIEW_CODE_V1.zip'
assert pack.is_file(), pack
with zipfile.ZipFile(pack) as zf:
    zf.extractall('/content')

assert Path('/content/p1_baseline/model.py').is_file()
assert len(list(Path('/content/C3_Z26_COMBO_V2/images/test').glob('*.png'))) == 200
print('SETUP PASS')
```

Sau đó chạy cell inference:

```python
subprocess.run([
    sys.executable, '-u',
    '/content/c3_roi/evaluate_pilot_bc_test_tta.py',
    '--batch-size', '16',
], check=True)
```

Tiến độ in ra `pilot=C fold=... view=.../10 done`. Sau mỗi fold script ghi cache
vào Drive; nếu Colab ngắt, chạy lại cell và nó sẽ dùng cache đã xác minh theo
hash ảnh, metadata, checkpoint và config. Kết quả cuối ở:

```text
MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/TEST_TTA_10VIEW_EXPLORATORY/
├── pilot_bc_test_tta_predictions.csv
├── pilot_bc_test_tta_report.json
└── fold_cache/
```

Để xem nhanh kết quả:

```python
import json
from pathlib import Path

report = json.loads((root / 'C3_Z26_C3_ROI_V2_PILOTS/TEST_TTA_10VIEW_EXPLORATORY/pilot_bc_test_tta_report.json').read_text())
for name, result in report['comparisons'].items():
    print(name)
    print('  Reference MAE:', result['reference']['mae_months'])
    print('  Candidate MAE:', result['candidate']['mae_months'])
    print('  Delta:', result['paired_candidate_minus_reference']['estimate_months'])
    print('  CI 95%:', result['paired_candidate_minus_reference']['bootstrap_95_ci'])
```

Chỉ disconnect khi report và CSV cuối đều tồn tại; nếu chưa xong vẫn có thể
reconnect và dùng cache để chạy tiếp.
