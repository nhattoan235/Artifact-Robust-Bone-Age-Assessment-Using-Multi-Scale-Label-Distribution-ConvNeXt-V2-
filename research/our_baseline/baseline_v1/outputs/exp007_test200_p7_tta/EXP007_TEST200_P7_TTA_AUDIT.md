# EXP007 — EXP006 P7 five-fold raw/TTA on exploratory test-200

## Protocol

- Checkpoints: `EXP006_P7_CONTROL_FOLD_1..5/best_mae.ckpt`
- Device: CUDA with FP16 autocast
- Test images: 200
- TTA transforms: rotations `-10, -5, 0, 5, 10` degrees × horizontal flip/no-flip
- Raw prediction: five-fold ensemble using `rot_0_no_flip`
- TTA prediction: average of 5 folds × 5 rotations × 2 flip states
- Test labels: read only because this exploratory evaluation was explicitly requested

## Results

| Method | MAE | RMSE | Median AE | Accuracy ±6 | Accuracy ±12 |
|---|---:|---:|---:|---:|---:|
| EXP006 raw five-fold | 4.6041 | 5.8059 | 3.9740 | 70.0% | 95.0% |
| EXP006 TTA five-fold | **4.4669** | **5.7472** | **3.6317** | **71.5%** | **95.5%** |

## TTA effect

- MAE improvement: `0.1372` months, approximately `2.98%` relative to raw five-fold.
- RMSE improvement: `0.0587` months.
- Accuracy within 6 months: `+1.5` percentage points.
- Accuracy within 12 months: `+0.5` percentage points.

## Comparison with previous exploratory results

- Friend P7 ensemble: MAE `4.7309` months.
- EXP005 50/50 blend: MAE `4.9334` months.
- Previous single P7 reference model: MAE `5.1133` months.
- EXP004 retrained model: MAE `5.2146` months.

On this exploratory 200-image set, EXP007 TTA is lower than the previous friend P7 result by `0.2640` months. This is an encouraging diagnostic, but it is not a leakage-safe scientific comparison because the labeled 200-image set has been inspected before.

## Error analysis

The most difficult age group remains `60–119` months with MAE `6.4075` months and positive bias `+3.8540` months. The overall signed bias is `+0.7604` months. This suggests mild overprediction, especially in the 60–119-month group.

## Decision

- Keep TTA as the inference candidate because it improves both EXP006 OOF (`6.3236 → 6.2962`) and this exploratory test-200 evaluation (`4.6041 → 4.4669`).
- Do not use this test-200 result to tune weights or claim the final generalization score.
- The target MAE `<3.5` months has not yet been reached; the next improvement must come from a new OOF-validated model or a leakage-safe blend, not from further tuning on these 200 labels.
