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

## Bối cảnh và lịch sử

- `AI_Context/00_START_HERE.md` – file phải đọc đầu tiên.
- `AI_Context/CHANGELOG.md` – changelog chính thức của handoff này.
- `PROJECT_CONTEXT.md`, root `CHANGELOG.md` – tài liệu cũ; chỉ dùng để đối chiếu lịch sử.

