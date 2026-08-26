# Results summary

All ages are measured in months. Lower MAE is better. The 200-image results are
exploratory and must not be used as a model-selection claim.

## Verified comparison

| Experiment / method | Evaluation protocol | N | MAE | RMSE | Status |
|---|---|---:|---:|---:|---|
| Own P7 reference baseline | Official validation | 1,425 | 6.471143 | 8.878041 | Baseline |
| Own A2 light flip | Official validation | 1,425 | 6.706371 | 9.025617 | Rejected; worse than baseline |
| Friend P7 ensemble | Five-fold OOF | 14,036 | 6.316691 | 8.519420 | Shared reference |
| EXP-004 P7 retrain | Fresh holdout | 1,260 | 6.213796 | 8.450559 | Completed |
| EXP-006 P7 control | Five-fold pooled OOF | 14,036 | 6.323629 | 8.529249 | Completed |
| EXP-006 P7 + TTA | Five-fold pooled OOF | 14,036 | 6.296203 | 8.505894 | Retained inference improvement |
| EXP-008 fixed 50/50 blend | Five-fold pooled OOF | 14,036 | **6.120448** | **8.272590** | Current OOF selection gate |
| Friend P7 ensemble | Exploratory test-200 | 200 | 4.730865 | 6.029187 | Best group reference on test-200 |
| EXP-006 P7 + TTA | Exploratory test-200 | 200 | **4.466886** | 5.747240 | Exploratory only |
| EXP-008 fixed 50/50 blend | Exploratory test-200 | 200 | 4.553021 | 5.785748 | Exploratory only |
| All 15 packaged models, raw | Cleaned test-200 | 200 | 4.766358 | 5.974673 | Exploratory only |
| All 15 packaged models + TTA | Cleaned test-200 | 200 | **4.675289** | **5.928851** | Exploratory only |

## EXP-001: blend on official validation

The prediction rule was:

```text
blend = (1 - w) × own_prediction + w × friend_prediction
```

On the 1,425-row development validation, the 50/50 blend reached MAE 6.177155,
compared with 6.471143 for the own baseline. Because the weight was searched on
that same validation set, this was treated as a development signal rather than an
unbiased final result.

## EXP-002: internal holdout confirmation

With the fixed 50/50 weight, the 391-row internal holdout reached MAE 5.761539,
compared with 6.070253 for the baseline. This supported the blend hypothesis but
was not an entirely independent holdout because the parent validation set had
already been inspected during EXP-001.

## EXP-003/004: inference verification and fresh holdout

The first manual reconstruction of the shared P7 inference pipeline was rejected
because it omitted square padding. Re-running through the original data/model code
reduced the maximum difference from the stored predictions to 0.054260 months on
the verification sample. A fresh-holdout retrain then achieved MAE 6.213796.

## EXP-006: five-fold P7 control and TTA

The development pool contained 14,036 images. The folds were stratified by sex and
age bins `[0, 60, 120, 180, 229]`, with 2,807–2,808 validation rows per fold.
The control pooled OOF MAE was 6.323629. TTA averaged five rotations and two flip
states and reduced OOF MAE to 6.296203, an improvement of 0.027426 months.

## EXP-008: fixed OOF blend

The P7 friend OOF and EXP-006 TTA OOF were row-aligned on all 14,036 development
rows, with matching targets. A fixed 50/50 average achieved MAE 6.120448,
improving over the individual sources without reading the test labels.

## Exploratory test-200 measurements

The labeled 200-image set was evaluated only after the method choices were fixed.
EXP-006 P7 + TTA obtained MAE 4.466886 and the fixed blend obtained 4.553021.
These values are useful for comparison with the group reference but are not a
selection protocol and do not establish generalization by themselves.

## EXP-010: all packaged models on cleaned test-200

All 15 available checkpoints were evaluated on the manually cleaned 200-image
set. The raw score is the native deterministic inference of each checkpoint.
The TTA score averages five rotations (-10, -5, 0, 5, 10 degrees), each with
and without horizontal flip. The equal-weight ensemble improved from MAE
4.766358 to 4.675289 months. This is an exploratory test-200 audit only and
must not be used to select weights or hyperparameters. The full per-model table,
predictions, and provenance are in
`baseline_v1/outputs/evaluate_raw_test200_20260826/RAW_TEST200_ALL_MODELS_AUDIT.md`.

## EXP-009: LDL roadmap

EXP-009 prepared five folds for `convnext_tiny_ldl` with sigma 2.0, auxiliary label
distribution weight 0.2, and regression/distribution inference weight 0.5. It uses
the same split as EXP-006. The required gate is pooled OOF MAE below 6.120448.
Until that gate is measured, EXP-009 is a planned/ongoing experiment rather than
an achieved result.

## Lessons and rejected directions

- Light horizontal flip in the original baseline did not improve the official
  validation result.
- Multi-scale and ConvNeXtV2 variants in the shared exploratory work did not beat
  the control reliably.
- Resolution increases and mask/foreground variants added cost without a strong
  OOF improvement in the recorded trials.
- TTA and heterogeneous blending had the clearest evidence so far.
- More training epochs alone did not solve the validation-overfitting pattern.
