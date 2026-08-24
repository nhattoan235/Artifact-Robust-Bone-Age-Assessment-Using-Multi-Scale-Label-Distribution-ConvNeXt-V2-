# Bản đồ file quan trọng

## Hồ sơ phase

- `p0_audit/P0_HANDOFF.md` – audit split, leakage, fingerprint.
- `p1_baseline/P1_HANDOFF.md` – hạ tầng baseline/checkpoint/AMP.
- `p1_baseline/P2_HANDOFF.md` – kết quả augmentation A0–A2.
- `p3_preprocessing/P3_HANDOFF.md` – mask B1 và quyết định loại.
- `p4_architecture/P4_HANDOFF.md` – D0–D3 và paired tests.
- `p5_seed_confirmation/P5_HANDOFF.md` – xác nhận 3 seed.
- `p5_seed_confirmation/P5_AGGREGATE.json` – số liệu seed máy đọc.
- `p6_resolution/P6_768_vs_512_paired.json` – quyết định 512.

## P7 final

- `p7_final_v3/P7_OOF_report.json` – báo cáo OOF chính thức.
- `p7_final_v3/P7_5FOLD_AUDIT.txt` – audit 5 fold.
- `p7_results.zip` – best model/results của 5 fold.
- `p7_oof_final.zip` – OOF CSV/report.

## P8 test

- `p8_test_ensemble/audit_inputs.py` – audit input.
- `p8_test_ensemble/infer_ensemble.py` – inference equal-weight 5 fold.
- `p8_test_ensemble/outputs/P8_test_ensemble_report.json` – kết quả và benchmark.
- `p8_test_ensemble/outputs/P8_ensemble_predictions.csv` – dự đoán 200 test.

## Thí nghiệm C / C3-ROI

- `AI_Context/24_EXPERIMENT_C_C3_ROI_FINAL_REPORT.md` – báo cáo phương pháp,
  OOF, test thăm dò, sai lệch protocol và claim policy.
- `c3_roi/configs/fold_1.toml` … `fold_5.toml` – config 5 fold đã chạy.
- `c3_roi/runs/C3_ROI_V1/C3_ROI_V1_FOLD_X/` – checkpoint/log từng fold.
- `c3_roi/outputs/C3_ROI_V1_OOF/C3_ROI_V1_OOF_report.json` – báo cáo OOF.
- `c3_roi/outputs/C3_ROI_V1_OOF/C3_ROI_V1_E1_ensemble_OOF_predictions.csv` –
  dự đoán paired E1/C3/ensemble.
- `c3_roi/outputs/C3_ROI_V1_TEST/C3_E1_50_50_test_report.json` – test thăm dò.
- `c3_roi/_drive_upload/C3_ROI_V1/cpu_artifacts/audit_summary.json` – audit
  14.036 ROI và tỷ lệ fallback.

## Bối cảnh và lịch sử

- `AI_Context/00_START_HERE.md` – file phải đọc đầu tiên.
- `AI_Context/CHANGELOG.md` – changelog chính thức của handoff này.

## C1 curated data

- `c1_curated/C1_HANDOFF.md` – trạng thái audit, config, smoke/resume và gate.
- `c1_curated/outputs/C1_MANIFEST_V1/C1_manifest_report.json` – báo cáo audit/hash/weight.
- `c1_curated/outputs/C1_MANIFEST_V1/C1_BALANCED_train_manifest.csv` – train manifest có weight.
- `p1_baseline/configs/c1_balanced_seed42.toml` – config primary/Colab.
- `p1_baseline/configs/c1_balanced_local_seed42.toml` – config local RTX 3050 Ti.
- `PROJECT_CONTEXT.md`, root `CHANGELOG.md` – tài liệu cũ; chỉ dùng để đối chiếu lịch sử.

