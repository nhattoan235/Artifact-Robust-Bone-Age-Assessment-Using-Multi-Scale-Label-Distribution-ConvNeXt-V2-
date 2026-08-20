from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from run_open_boneage_benchmark import summarize_predictions


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "outputs" / "boneage_benchmark_all_200"
OUT = ROOT / "outputs" / "technical_report_assets"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "original": "#4C78A8",
    "ours": "#2CA25F",
    "author_mean": "#E45756",
    "author_1": "#F28E8B",
    "author_2": "#E45756",
    "author_3": "#B73A3A",
}
LABELS = {
    "original": "Ảnh gốc",
    "ours": "Phương pháp đề xuất",
    "author_mean": "Tác giả (TB 3 lần sinh)",
    "author_1": "Tác giả - lần 1",
    "author_2": "Tác giả - lần 2",
    "author_3": "Tác giả - lần 3",
}
PROTOCOL_LABELS = {
    "fullframe": "Toàn khung",
    "histmatch": "Khớp histogram",
    "fixed_crop": "Crop cố định",
    "fixed_crop_histmatch": "Crop + histogram",
}


def add_author_mean(frame: pd.DataFrame) -> pd.DataFrame:
    authors = frame[frame["Group"].isin(["author_1", "author_2", "author_3"])]
    mean_rows = (
        authors.groupby(
            ["Case_ID", "Sex", "Ground_Truth_Months", "Protocol"], as_index=False
        )["Prediction_Months"]
        .mean()
        .assign(Group="author_mean", Source_Path="mean(author_1,author_2,author_3)")
    )
    mean_rows["Error_Months"] = (
        mean_rows["Prediction_Months"] - mean_rows["Ground_Truth_Months"]
    )
    mean_rows["Absolute_Error_Months"] = mean_rows["Error_Months"].abs()
    columns = list(frame.columns)
    return pd.concat([frame, mean_rows[columns]], ignore_index=True)


def save_three_way_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = frame[frame["Group"].isin(["original", "ours", "author_mean"])].copy()
    summary, paired, pairwise, cases = summarize_predictions(
        selected, bootstrap_repetitions=10_000, seed=20260729
    )
    selected.to_csv(OUT / "predictions_three_way.csv", index=False)
    summary.to_csv(OUT / "summary_three_way.csv", index=False)
    paired.to_csv(OUT / "paired_three_way.csv", index=False)
    pairwise.to_csv(OUT / "pairwise_three_way.csv", index=False)
    cases.to_csv(OUT / "case_level_three_way.csv", index=False)
    return summary, paired


def plot_mae(summary: pd.DataFrame) -> None:
    protocols = list(PROTOCOL_LABELS)
    groups = ["original", "ours", "author_mean"]
    x = np.arange(len(protocols))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, group in enumerate(groups):
        part = summary.set_index(["Protocol", "Group"])
        means = [part.loc[(p, group), "MAE_Months"] for p in protocols]
        lows = [part.loc[(p, group), "MAE_CI95_Low"] for p in protocols]
        highs = [part.loc[(p, group), "MAE_CI95_High"] for p in protocols]
        yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
        bars = ax.bar(
            x + (j - 1) * width,
            means,
            width,
            yerr=yerr,
            capsize=3,
            color=COLORS[group],
            label=LABELS[group],
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylabel("MAE (tháng)")
    ax.set_xlabel("Protocol tiền xử lý")
    ax.set_xticks(x, [PROTOCOL_LABELS[p] for p in protocols])
    ax.set_title("Sai số tuổi xương trên cùng 200 ca và cùng model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mae_three_way.png", dpi=220)
    plt.close(fig)


def plot_paired_drift(paired: pd.DataFrame) -> None:
    protocols = list(PROTOCOL_LABELS)
    groups = ["ours", "author_mean"]
    x = np.arange(len(protocols))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for j, group in enumerate(groups):
        part = paired.set_index(["Protocol", "Group"])
        drift = [part.loc[(p, group), "Mean_Absolute_Prediction_Drift"] for p in protocols]
        delta = [part.loc[(p, group), "Mean_Delta_AE_vs_Original"] for p in protocols]
        axes[0].bar(
            x + (j - 0.5) * width,
            drift,
            width,
            color=COLORS[group],
            label=LABELS[group],
        )
        axes[1].bar(
            x + (j - 0.5) * width,
            delta,
            width,
            color=COLORS[group],
            label=LABELS[group],
        )
    axes[0].set_title("Độ trôi dự đoán tuyệt đối so với ảnh gốc")
    axes[0].set_ylabel("|Dự đoán xử lý - dự đoán gốc| (tháng)")
    axes[1].set_title("Thay đổi sai số tuyệt đối so với ảnh gốc")
    axes[1].set_ylabel("ΔAE = AE xử lý - AE gốc (tháng)")
    axes[1].axhline(0, color="black", linewidth=1)
    for ax in axes:
        ax.set_xticks(x, [PROTOCOL_LABELS[p] for p in protocols], rotation=15)
        ax.set_xlabel("Protocol tiền xử lý")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "paired_drift_three_way.png", dpi=220)
    plt.close(fig)


def plot_protocol_detail(frame: pd.DataFrame, protocol: str) -> None:
    sub = frame[
        (frame["Protocol"] == protocol)
        & frame["Group"].isin(["original", "ours", "author_mean"])
    ]
    groups = ["original", "ours", "author_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for group in groups:
        g = sub[sub["Group"] == group]
        axes[0].scatter(
            g["Ground_Truth_Months"],
            g["Prediction_Months"],
            s=18,
            alpha=0.62,
            color=COLORS[group],
            label=LABELS[group],
        )
    low = min(sub["Ground_Truth_Months"].min(), sub["Prediction_Months"].min())
    high = max(sub["Ground_Truth_Months"].max(), sub["Prediction_Months"].max())
    axes[0].plot([low, high], [low, high], "k--", linewidth=1.2)
    axes[0].set_xlabel("Tuổi xương ground truth (tháng)")
    axes[0].set_ylabel("Tuổi xương model dự đoán (tháng)")
    axes[0].set_title("Ground truth so với dự đoán")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)
    values = [
        sub.loc[sub["Group"] == group, "Absolute_Error_Months"].to_numpy()
        for group in groups
    ]
    box = axes[1].boxplot(
        values, tick_labels=[LABELS[g] for g in groups], showfliers=False, patch_artist=True
    )
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.75)
    axes[1].set_ylabel("Sai số tuyệt đối (tháng)")
    axes[1].set_title("Phân bố sai số tuyệt đối")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].tick_params(axis="x", rotation=10)
    fig.suptitle(f"Phân tích protocol: {PROTOCOL_LABELS[protocol]}", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / f"{protocol}_three_way.png", dpi=220)
    plt.close(fig)


def plot_bland_altman(frame: pd.DataFrame, protocol: str) -> None:
    sub = frame[frame["Protocol"] == protocol]
    original = sub[sub["Group"] == "original"][
        ["Case_ID", "Prediction_Months"]
    ].rename(columns={"Prediction_Months": "Original"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, group in zip(axes, ["ours", "author_mean"]):
        current = sub[sub["Group"] == group][
            ["Case_ID", "Prediction_Months"]
        ].rename(columns={"Prediction_Months": "Processed"})
        paired = original.merge(current, on="Case_ID")
        mean = (paired["Original"] + paired["Processed"]) / 2
        diff = paired["Processed"] - paired["Original"]
        bias = diff.mean()
        sd = diff.std(ddof=1)
        ax.scatter(mean, diff, s=18, alpha=0.65, color=COLORS[group])
        ax.axhline(bias, color="#1F4E79", label=f"Bias = {bias:.2f}")
        ax.axhline(bias + 1.96 * sd, color="#C00000", linestyle="--")
        ax.axhline(bias - 1.96 * sd, color="#C00000", linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Trung bình hai dự đoán (tháng)")
        ax.set_ylabel("Dự đoán xử lý - dự đoán gốc (tháng)")
        ax.set_title(f"{LABELS[group]} so với ảnh gốc")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(f"Bland-Altman - {PROTOCOL_LABELS[protocol]}", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / f"{protocol}_bland_altman_three_way.png", dpi=220)
    plt.close(fig)


def plot_author_variability(frame: pd.DataFrame) -> None:
    authors = frame[
        (frame["Protocol"] == "fullframe")
        & frame["Group"].isin(["author_1", "author_2", "author_3"])
    ]
    stats = (
        authors.groupby(["Case_ID", "Ground_Truth_Months"])["Prediction_Months"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    stats.to_csv(OUT / "author_generation_variability_fullframe.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(stats["std"], bins=24, color=COLORS["author_mean"], alpha=0.85)
    axes[0].axvline(stats["std"].mean(), color="black", linestyle="--",
                    label=f"Trung bình = {stats['std'].mean():.2f} tháng")
    axes[0].set_xlabel("Độ lệch chuẩn dự đoán trong 3 lần sinh (tháng)")
    axes[0].set_ylabel("Số bệnh nhân")
    axes[0].set_title("Mức bất ổn giữa ba ảnh sinh")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].scatter(
        stats["Ground_Truth_Months"], stats["std"], s=22, alpha=0.7,
        color=COLORS["author_mean"]
    )
    axes[1].set_xlabel("Tuổi xương ground truth (tháng)")
    axes[1].set_ylabel("Độ lệch chuẩn dự đoán (tháng)")
    axes[1].set_title("Bất ổn theo tuổi xương")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "author_variability.png", dpi=220)
    plt.close(fig)


def plot_qc() -> None:
    old = pd.read_csv(
        Path(
            r"D:\Hoctap\Doan_totnghiep\DinhTanPhuong_docs"
            r"\segment_and_composite_pipeline\segment_and_composite_pipeline"
            r"\sample_output_200images.csv"
        )
    )
    final = pd.read_csv(
        ROOT
        / "outputs"
        / "artifact_only_200_manual_v3"
        / "artifact_only_qc.csv"
    )
    labels = [
        "Có đầu ra",
        "Bảo toàn vùng giải phẫu",
        "Không đổi ngoài mask",
        "Không bị loại khỏi cohort",
    ]
    old_values = [
        (old["Status"] == "SUCCESS").mean() * 100,
        100.0,
        np.nan,
        (old["Status"] == "SUCCESS").mean() * 100,
    ]
    final_values = [
        100.0,
        final["changed_inside_protected"].eq(0).mean() * 100,
        final["changed_outside_artifact"].eq(0).mean() * 100,
        100.0,
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, old_values, width, label="Segment-and-composite cũ",
           color="#9E9E9E")
    bars = ax.bar(x + width / 2, final_values, width, label="Artifact-only cuối",
                  color=COLORS["ours"])
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Tỷ lệ ca đạt (%)")
    ax.set_xticks(x, labels)
    ax.set_title("Chuyển từ ghép toàn nền sang chỉnh sửa cục bộ có bất biến pixel")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "qc_progress.png", dpi=220)
    plt.close(fig)


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    items = [
        ("Ảnh X-quang\ngốc", "#DCE6F1"),
        ("Mask bảo vệ\ngiải phẫu", "#C6E0B4"),
        ("Mask dị vật\nngoài giải phẫu", "#F8CBAD"),
        ("Tái tạo nền cục bộ\nplane + grain", "#FFF2CC"),
        ("Ghép chỉ trong mask\n+ feather hẹp", "#DDEBF7"),
        ("QC pixel + seam\n+ tuổi xương", "#D9EAD3"),
    ]
    xs = np.linspace(0.45, 12.25, len(items))
    for i, ((label, color), x) in enumerate(zip(items, xs)):
        patch = FancyBboxPatch(
            (x, 1.35), 1.7, 1.2,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=color, edgecolor="#2F5597", linewidth=1.4
        )
        ax.add_patch(patch)
        ax.text(x + 0.85, 1.95, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        if i < len(items) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + 1.72, 1.95), (xs[i + 1] - 0.08, 1.95),
                arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
                color="#2F5597"
            ))
    ax.text(
        7, 3.35,
        "Tư tưởng cốt lõi: xác định vùng được phép thay đổi, không tái sinh toàn ảnh",
        ha="center", va="center", fontsize=14, color="#1F4E79", fontweight="bold"
    )
    ax.text(
        7, 0.55,
        "Bất biến kiểm chứng: pixel trong vùng giải phẫu và ngoài artifact mask phải giống ảnh gốc tuyệt đối",
        ha="center", va="center", fontsize=10.5, color="#555555"
    )
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_artifact_only.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def trouble_montage() -> None:
    old_root = Path(
        r"D:\Hoctap\Doan_totnghiep\DinhTanPhuong_docs"
        r"\segment_and_composite_pipeline\segment_and_composite_pipeline"
    )
    paths = [
        old_root / "morphology_comparison.png",
        ROOT / "outputs" / "masked_hand_200_v1" / "images" / "4370.png",
        old_root / "outputs_v7" / "composites" / "4370_comp_A.png",
        ROOT / "outputs" / "artifact_only_200_manual_v3" / "review" / "4371.jpg",
    ]
    titles = [
        "Morphology làm mất ngón mảnh",
        "Ghép toàn nền: nền phẳng, biên răng cưa",
        "Mask sai: mất da / vùng khuyết",
        "Kết quả cuối: chỉ vùng đỏ được sửa",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, path, title in zip(axes.flat, paths, titles):
        ax.imshow(read_rgb(path))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "trouble_and_fix_montage.png", dpi=220)
    plt.close(fig)


def reconstruction_trials() -> None:
    entries = [
        ("Telea", ROOT / "outputs" / "diagnostic_telea" / "4371.png"),
        ("Feather rộng", ROOT / "outputs" / "diagnostic_feather_v3" / "4371.jpg"),
        ("Gradient plane", ROOT / "outputs" / "diagnostic_gradient_v4" / "4371.jpg"),
        ("Clamp", ROOT / "outputs" / "diagnostic_clamped_v5" / "4371.jpg"),
        ("Tangent", ROOT / "outputs" / "diagnostic_tangent_v6" / "4371.jpg"),
        ("Seam-match cuối", ROOT / "outputs" / "diagnostic_seammatch_v9" / "4371.jpg"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (title, path) in zip(axes.flat, entries):
        ax.imshow(read_rgb(path))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Các thử nghiệm tái tạo nền cục bộ trên cùng ca 4371", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "reconstruction_trials.png", dpi=220)
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(BENCH / "predictions_long.csv", dtype={"Case_ID": str})
    frame = add_author_mean(frame)
    summary, paired = save_three_way_analysis(frame)
    plot_mae(summary)
    plot_paired_drift(paired)
    for protocol in PROTOCOL_LABELS:
        plot_protocol_detail(frame, protocol)
        plot_bland_altman(frame, protocol)
    plot_author_variability(frame)
    plot_qc()
    pipeline_diagram()
    trouble_montage()
    reconstruction_trials()
    print(f"Saved report assets to {OUT}")


if __name__ == "__main__":
    main()
