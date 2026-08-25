# Final model packages

Mỗi thư mục con là một package độc lập theo cấu trúc:

```text
model/
config/
code/
evidence/
README_MODEL.md
manifest.json
```

## Danh sách package

| Package | Nhóm | MAE ghi nhận | Ghi chú |
|---|---|---:|---|
| `P7_E1_FOLD1_CONVNEXT_TINY_SEX_A2` | P7 FINAL V3 | 6.2987 | Fold 1 OOF |
| `P7_E1_FOLD2_CONVNEXT_TINY_SEX_A2` | P7 FINAL V3 | 6.1968 | Fold 2 OOF |
| `P7_E1_FOLD3_CONVNEXT_TINY_SEX_A2` | P7 FINAL V3 | 6.4115 | Fold 3 OOF |
| `P7_E1_FOLD4_CONVNEXT_TINY_SEX_A2` | P7 FINAL V3 | 6.3488 | Fold 4 OOF |
| `P7_E1_FOLD5_CONVNEXT_TINY_SEX_A2` | P7 FINAL V3 | 6.3277 | Fold 5 OOF |
| `EXP006_P7_CONTROL_FOLD1_CONVNEXT_TINY_SEX_LDL` | EXP006 | 6.2400 | Fold contribution to pooled OOF |
| `EXP006_P7_CONTROL_FOLD2_CONVNEXT_TINY_SEX_LDL` | EXP006 | 6.2573 | Fold contribution to pooled OOF |
| `EXP006_P7_CONTROL_FOLD3_CONVNEXT_TINY_SEX_LDL` | EXP006 | 6.3443 | Fold contribution to pooled OOF |
| `EXP006_P7_CONTROL_FOLD4_CONVNEXT_TINY_SEX_LDL` | EXP006 | 6.4665 | Fold contribution to pooled OOF |
| `EXP006_P7_CONTROL_FOLD5_CONVNEXT_TINY_SEX_LDL` | EXP006 | 6.3100 | Fold contribution to pooled OOF |
| `EXP004_FRIEND_P7_FRESH_HOLDOUT_CONVNEXT_TINY_SEX` | EXP004 | 6.2138 | Fresh holdout; protocol differs from OOF |
| `OFFICIAL_A2_LIGHT_FLIP_CONVNEXT_TINY_SEX` | Official baseline | 6.7064 | 1,425-image validation |
| `OFFICIAL_P7_REFERENCE_CONVNEXT_TINY_SEX` | Official baseline | 6.4711 | 1,425-image validation |

## Important comparison note

The P7 and EXP006 values are OOF/fold-based, while EXP004 is a fresh holdout and the two official baselines use the original 1,425-image validation split. They should not be ranked as if they were the same evaluation protocol.

The labeled 200-image test set is exploratory only and was not used to select these packages.
