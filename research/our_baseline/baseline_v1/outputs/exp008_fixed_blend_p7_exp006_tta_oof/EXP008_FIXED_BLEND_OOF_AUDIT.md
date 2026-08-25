# EXP008 — Fixed 50/50 P7 + EXP006 TTA OOF blend

## Protocol

- Rows: 14036
- Test labels used: **No**
- Weight: P7 friend OOF `0.50` + EXP006 TTA OOF `0.50`
- Target labels matched exactly across both sources.

## Metrics

| Method | MAE | RMSE | Median AE | Accuracy ±6 | Accuracy ±12 |
|---|---:|---:|---:|---:|---:|
| P7 friend OOF | 6.3167 | 8.5194 | 4.8750 | 59.0125% | 86.7341% |
| EXP006 TTA OOF | 6.2962 | 8.5059 | 4.7609 | 59.4828% | 86.3066% |
| Fixed 50/50 blend | **6.1204** | **8.2726** | **4.6243** | **60.5372%** | **87.3254%** |

## Decision

Keep the fixed blend as the current OOF candidate. The next model, LDL, must produce row-aligned OOF predictions on the same 14,036 images and beat this `MAE=6.1204` before being added to a three-model blend.

The labeled 200-image test set remains exploratory and is not used for selecting the blend.
