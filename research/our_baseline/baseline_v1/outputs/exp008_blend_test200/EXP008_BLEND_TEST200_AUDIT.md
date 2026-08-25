# EXP008 — Fixed 50/50 blend on exploratory test-200

The 50/50 weight was fixed from the OOF audit before this test report was read.

| Method | MAE | RMSE | Median AE | Accuracy ±6 | Accuracy ±12 |
|---|---:|---:|---:|---:|---:|
| Friend P7 ensemble | 4.7309 | 6.0292 | 4.1342 | 70.0% | 94.0% |
| EXP006 P7-TTA 5-fold | **4.4669** | **5.7472** | **3.6317** | **71.5%** | **95.5%** |
| Fixed 50/50 blend | 4.5530 | 5.7857 | 3.9215 | 70.5% | **95.5%** |

## Interpretation

The blend improves over the friend P7 model but is worse than EXP006 TTA on this 200-image diagnostic set by `0.0861` months. This does not invalidate the OOF blend result: the 200-image set is small, has been inspected previously, and must not be used to retune the weight.

For the scientific selection gate, retain the fixed 50/50 blend as the OOF candidate (`MAE 6.1204`) and continue with the planned LDL five-fold experiment. The test-200 result is recorded only as exploratory evidence.
