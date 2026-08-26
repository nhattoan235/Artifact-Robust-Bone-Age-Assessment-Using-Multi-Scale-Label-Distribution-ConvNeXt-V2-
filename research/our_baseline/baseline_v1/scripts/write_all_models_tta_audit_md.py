"""Write the consolidated Markdown audit from the completed TTA JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))

    lines = [
        "# Đánh giá toàn bộ model trên tập test 200 ảnh đã làm sạch",
        "",
        "## Protocol",
        "",
        f"- Ảnh sạch: `{report['image_root']}`",
        f"- Nhãn: `{report['labels']}`",
        f"- Số lượng: {report['image_count']} ảnh, ID 4360–4559.",
        "- Đã đánh giá 15 checkpoint hiện có.",
        "- Model thuần (`raw`): preprocessing xác định của từng model, không xoay/lật.",
        "- TTA: trung bình 10 views gồm xoay -10°, -5°, 0°, 5°, 10° với ảnh nguyên bản và lật ngang.",
        "- Precision: AMP FP16 trên GPU RTX 3050; checkpoint không bị sửa.",
        "- Đây là đánh giá thăm dò; nhãn test không được dùng để chọn hyperparameter hoặc trọng số.",
        "",
        "## Kết quả theo nhóm",
        "",
        "| Nhóm | Số model | MAE raw | MAE + TTA | Δ MAE | RMSE raw | RMSE + TTA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, data in report["groups"].items():
        lines.append(
            f"| {group} | {data['model_count']} | {f(data['raw']['mae_months'])} | "
            f"{f(data['tta']['mae_months'])} | {f(data['delta_tta_minus_raw']['mae_months'])} | "
            f"{f(data['raw']['rmse_months'])} | {f(data['tta']['rmse_months'])} |"
        )

    lines += [
        "",
        "## Kết quả từng model",
        "",
        "| Model | MAE raw | MAE + TTA | Δ MAE | RMSE raw | RMSE + TTA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, data in sorted(report["models"].items(), key=lambda item: item[1]["tta"]["mae_months"]):
        lines.append(
            f"| {name} | {f(data['raw']['mae_months'])} | {f(data['tta']['mae_months'])} | "
            f"{f(data['delta_tta_minus_raw']['mae_months'])} | {f(data['raw']['rmse_months'])} | "
            f"{f(data['tta']['rmse_months'])} |"
        )

    all15 = report["groups"]["all_15_models"]
    lines += [
        "",
        "## Nhận xét chính",
        "",
        f"- Ensemble 15 model thuần đạt MAE **{f(all15['raw']['mae_months'])}**.",
        f"- Ensemble 15 model + TTA đạt MAE **{f(all15['tta']['mae_months'])}**.",
        f"- TTA thay đổi MAE **{f(all15['delta_tta_minus_raw']['mae_months'])}** tháng; giá trị âm là cải thiện.",
        "- TTA cải thiện ensemble EXP006 5-fold, nhưng mức cải thiện nhỏ hơn nhóm P7 standard.",
        "- TTA làm xấu nhóm EXP009 LDL 2-fold trong phép thử này; không nên mặc định áp dụng TTA cho mọi nhóm.",
        "- Mốc EXP006 TTA 5-fold MAE 4.4669 trước đây thuộc một lần đánh giá khác; không trộn với kết quả ảnh sạch hiện tại.",
        "- Không dùng kết quả test 200 này để chọn trọng số hoặc hyperparameter; quyết định chính thức cần dựa trên OOF/holdout độc lập.",
        "",
        "## Kiểm tra dữ liệu sạch",
        "",
    ]
    qc = report.get("qc_summary", {})
    if qc:
        lines += [
            f"- QC rows: {qc.get('rows')}",
            f"- Pixel preservation pass: {qc.get('pixel_preservation_pass_count')}/{qc.get('rows')}",
            f"- Tỷ lệ pixel thay đổi trung bình: {qc.get('changed_pct_mean'):.4f}%",
            f"- Trạng thái: `{qc.get('status_counts')}`",
        ]
    lines += [
        "",
        "## File kết quả",
        "",
        f"- [Prediction CSV]({Path(report['prediction_file']).as_posix()})",
        f"- [Report JSON]({args.report.resolve().as_posix()})",
        "- [Script đánh giá](D:/do_an_tot_nghiep/project/baseline_v1/scripts/evaluate_all_models_tta_cleaned_test200.py)",
        "- [Script tạo báo cáo](D:/do_an_tot_nghiep/project/baseline_v1/scripts/write_all_models_tta_audit_md.py)",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
