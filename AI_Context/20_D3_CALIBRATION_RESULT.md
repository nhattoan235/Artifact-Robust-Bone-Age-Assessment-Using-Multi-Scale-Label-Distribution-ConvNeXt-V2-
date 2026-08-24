# D3-TTA OOF-only calibration result

## Protocol

- Calibration was evaluated only on the 14,036-image development OOF set.
- For each held-out fold, a global linear mapping was fitted on the other four folds and applied to the held-out fold.
- The RSNA test set was not accessed.
- The candidate before calibration was the locked `0.5 * E1-TTA + 0.5 * D3-TTA` ensemble.

## Result

| Candidate | Cross-fitted OOF MAE |
|---|---:|
| Fixed E1-TTA + D3-TTA 50/50 | 6.10134 |
| With global linear calibration | 6.11641 |

Calibration changed MAE by **+0.01506 months**, so it was rejected. It is not applied to the test predictions.

## Decision

The final Plan A protocol remains unchanged. D3-TTA is retained as the best standalone test model with MAE 4.50846 months, and the fixed 50/50 ensemble remains the pre-registered ensemble ablation. This negative result is retained to document that post-hoc OOF calibration did not improve the locked candidate.

Machine-readable result: `d3_oof/outputs/D3_TTA_OOF_V1/D3_TTA_calibration_report.json`.
