# Experiment registry

| ID | Method | Data/protocol | Result | Decision |
|---|---|---|---|---|
| EXP-001 | Own P7 + friend P7 prediction blend | Official validation, 1,425 rows | 50/50 MAE 6.177155 | Keep as development signal; do not call final |
| EXP-002 | Fixed 50/50 blend | Internal 391-row holdout | MAE 5.761539 | Supports blend; independence limited |
| EXP-003 | Reconstruct friend P7 inference | 10 validation images | max difference 0.054260 after correct padding | Verification passed |
| EXP-004 | Fresh P7 retrain | 1,260-row fresh holdout | MAE 6.213796 | Completed; below friend test reference not established |
| EXP-005 | Compare retrain/friend/blends | Labeled test-200 | friend P7 MAE 4.730865 | Exploratory only |
| EXP-006 | P7 control, five folds | 14,036-row pooled OOF | MAE 6.323629 | Completed |
| EXP-007-TTA | Five-fold P7 TTA | OOF and test-200 | OOF MAE 6.296203; test MAE 4.466886 | Retain TTA; test value exploratory |
| EXP-008 | Fixed 50/50 P7 + TTA OOF blend | 14,036-row pooled OOF | MAE 6.120448 | Current OOF gate |
| EXP-009 | ConvNeXt-Tiny LDL fused | Same five-fold split | No final result in this bundle | Run only if resources available; must beat 6.120448 |

## Evidence locations

The detailed files are under `baseline_v1/` in this directory. The main audit and
metric artifacts are in:

- `baseline_v1/outputs/exp001_friend_blend/`
- `baseline_v1/outputs/exp002_blend_holdout/`
- `baseline_v1/outputs/exp004_friend_holdout/`
- `baseline_v1/outputs/exp006_p7_tta/`
- `baseline_v1/outputs/exp007_test200_p7_tta/`
- `baseline_v1/outputs/exp008_fixed_blend_p7_exp006_tta_oof/`
- `baseline_v1/outputs/exp008_blend_test200/`
- `baseline_v1/outputs/exp009_ldl_roadmap/`

The original chronological notes are preserved in
`baseline_v1/EXPERIMENT_LOG.md` and `baseline_v1/TONG_HOP_TOAN_BO_TIEN_TRINH.md`.
