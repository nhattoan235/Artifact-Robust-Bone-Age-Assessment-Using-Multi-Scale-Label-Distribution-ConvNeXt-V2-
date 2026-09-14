# Pilot C failure analysis and Fold-5 follow-up — 2026-09-14

## Scope and protocol

This analysis uses the locked 14,036-image five-fold validation OOF file. It
does not read the 200-image RSNA test set and does not select a new blend
weight. The evaluated blend remains fixed at 27% baseline + 73% Pilot C.

## Step 1 — OOF failure analysis (complete)

Pilot C remains a strong artifact-robustness model, but its clean behavior is
not uniform across folds and age groups.

| Slice | Baseline clean MAE | Pilot C clean MAE | Fixed-blend clean MAE | Blend delta vs baseline | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| Overall | 6.3300 | 6.3707 | 6.2725 | -0.0575 | [-0.0925, -0.0229] |
| Fold 5 | 6.1823 | 6.4305 | 6.2598 | +0.0775 | [-0.0113, +0.1640] |
| Fold 5, 60–119 months | 7.4161 | 8.2461 | 7.9140 | +0.4980 | [+0.3183, +0.6784] |

The Fold-5 effect is broad rather than caused by only a few extreme images:

- 51.3% of Fold-5 images become worse under the fixed blend;
- after trimming the 1% largest absolute changes, the mean delta is still
  +0.0690 months;
- after trimming 5%, the mean delta is still +0.0496 months;
- Pilot C nevertheless improves Fold-5 artifact MAE by 0.5084 months, and the
  fixed blend improves it by 0.6579 months.

The clearest failure is the 60–119-month group in Fold 5. Its clean prediction
bias changes from +1.5142 months for baseline to +2.9385 for Pilot C and +2.5539
for the blend. This indicates systematic overestimation in this slice, rather
than a small number of corrupt validation samples.

Generated evidence:

- `results/pilot_bc_20260914/failure_analysis/failure_analysis_report.json`
- `results/pilot_bc_20260914/failure_analysis/fold_age_failure_summary.csv`
- `results/pilot_bc_20260914/failure_analysis/error_threshold_transitions.csv`
- `results/pilot_bc_20260914/failure_analysis/worst_200_clean_regressions.csv`

## Step 2 — artifact-type matrix (prepared; requires T4 inference)

Run the prepared Fold-5 matrix before another training experiment. It evaluates
the existing baseline and Pilot C checkpoints on clean, exposure, contrast,
gamma, blur, noise, edge-band and the locked composite `mild_v1` views. Severity
1.0 is the maximum of the existing training range; no extrapolated corruption
is used.

The evaluator must reproduce the locked clean and composite OOF predictions
within 0.02 months. It stops with an error if this integrity check fails. It
then saves overall and age-group results with paired bootstrap intervals.

## Step 3 — Pilot B Fold 5 control (prepared; requires T4 training)

Pilot B Fold 5 is preregistered as the next training control. Its split, seed,
architecture, augmentation, optimizer and schedule are identical to Pilot C
Fold 5. The only scientific change is:

| Model | Artifact augmentation | Consistency weight |
|---|---|---:|
| Pilot B Fold 5 | `mild_v1`, probability 1.0 | 0.00 |
| Pilot C Fold 5 | `mild_v1`, probability 1.0 | 0.30 |

This comparison identifies whether Fold-5 clean degradation mainly comes from
artifact augmentation shared by B/C or from the added consistency penalty in C.

## Step 4 — decision rule after B Fold 5

- If B preserves clean performance and C does not, while C has no significant
  artifact advantage over B, reduce or remove the 0.30 consistency term.
- If both B and C degrade the same clean age slice, modify the artifact sampling
  or add age-aware sampling; changing consistency alone is unlikely to fix it.
- If C gives a significant artifact gain over B with acceptable clean cost,
  retain consistency and test a smaller weight such as 0.10 in one locked fold.
- Do not rerun the 200-image test set while choosing among these alternatives.

## Colab package

Use the generated package in
`c3_roi/outputs/C3_Z26_C3_ROI_V2_FOLLOWUP_FOLD5_V1/`. The notebook executes the
artifact matrix, Pilot B Fold-5 training, Pilot B evaluation, and paired
B/C/baseline comparison in that order. Every result is written under
`MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/` before the notebook reports that it
is safe to disconnect.
