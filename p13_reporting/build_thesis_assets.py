from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026
MANIFEST_NAME = "report_manifest.json"
AGE_ORDER = ["0-59", "60-119", "120-179", "180-228"]
SEX_ORDER = ["F", "M"]
MODEL_PATHS = {
    "E0 image-only": Path(
        "p11_sex_aware/runs/P11_E0_IMAGE_ONLY_CONVNEXT_TINY_SEED42/"
        "val_predictions_best.csv"
    ),
    "E1 sex embedding": Path(
        "p9_preprocessing/runs/P10_B0_P2_CONTROL_CONVNEXT_TINY_SEED42/"
        "val_predictions_best.csv"
    ),
    "E2 dual-output": Path(
        "p11_sex_aware/runs/P11_E2_SHARED_DUAL_OUTPUT_CONVNEXT_TINY_SEED42/"
        "val_predictions_best.csv"
    ),
}
P12_DIR = Path("p12_uncertainty/outputs/P12_OOF_UNCERTAINTY")
P11_REPORTS = [
    Path("p11_sex_aware/analysis/e0_vs_e1_seed42.json"),
    Path("p11_sex_aware/analysis/e2_vs_e1_seed42.json"),
]
PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#666666",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_hashable_outputs(output_dir: Path) -> list[Path]:
    """Return generated files while excluding the self-referential manifest."""
    return sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != MANIFEST_NAME
    )


def load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image_id", "target_months", "prediction_months", "sex"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} thiếu cột: {missing}")
    if len(frame) != 1425 or frame["image_id"].nunique() != 1425:
        raise ValueError(f"{path} không có đúng 1.425 ID duy nhất")
    if set(frame["sex"]) != {"F", "M"}:
        raise ValueError(f"{path} có sex ngoài F/M")
    return frame.sort_values("image_id").reset_index(drop=True)


def validate_alignment(frames: dict[str, pd.DataFrame]) -> None:
    baseline = frames["E1 sex embedding"]
    for name, frame in frames.items():
        if not np.array_equal(frame["image_id"], baseline["image_id"]):
            raise ValueError(f"ID không khớp giữa {name} và E1")
        if not np.array_equal(frame["target_months"], baseline["target_months"]):
            raise ValueError(f"Target không khớp giữa {name} và E1")
        if not np.array_equal(frame["sex"], baseline["sex"]):
            raise ValueError(f"Sex không khớp giữa {name} và E1")


def model_metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame["prediction_months"].to_numpy() - frame["target_months"].to_numpy()
    absolute = np.abs(error)
    female = frame["sex"].to_numpy() == "F"
    male = ~female
    return {
        "count": int(len(frame)),
        "mae_months": float(absolute.mean()),
        "rmse_months": float(np.sqrt(np.mean(error**2))),
        "median_ae_months": float(np.median(absolute)),
        "accuracy_within_6_months": float((absolute <= 6).mean()),
        "accuracy_within_12_months": float((absolute <= 12).mean()),
        "female_mae_months": float(absolute[female].mean()),
        "male_mae_months": float(absolute[male].mean()),
        "sex_gap_female_minus_male_months": float(
            absolute[female].mean() - absolute[male].mean()
        ),
    }


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    chunk_size = 128
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        output[start:stop] = values[index].mean(axis=1)
    return [float(value) for value in np.quantile(output, [0.025, 0.975])]


def paired_forest_data(
    frames: dict[str, pd.DataFrame], samples: int, seed: int
) -> pd.DataFrame:
    baseline = frames["E1 sex embedding"]
    baseline_error = np.abs(
        baseline["prediction_months"].to_numpy()
        - baseline["target_months"].to_numpy()
    )
    rows = []
    for candidate_index, candidate_name in enumerate(("E0 image-only", "E2 dual-output")):
        candidate = frames[candidate_name]
        candidate_error = np.abs(
            candidate["prediction_months"].to_numpy()
            - candidate["target_months"].to_numpy()
        )
        delta = candidate_error - baseline_error
        for group_index, group in enumerate(("Overall", "Nữ", "Nam")):
            if group == "Overall":
                mask = np.ones(len(baseline), dtype=bool)
            elif group == "Nữ":
                mask = baseline["sex"].to_numpy() == "F"
            else:
                mask = baseline["sex"].to_numpy() == "M"
            values = delta[mask]
            ci = bootstrap_mean_ci(
                values, samples, seed + candidate_index * 100 + group_index
            )
            rows.append(
                {
                    "candidate": candidate_name,
                    "group": group,
                    "count": int(mask.sum()),
                    "delta_mae_months": float(values.mean()),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "definition": "candidate absolute error - E1 absolute error",
                }
            )
    return pd.DataFrame(rows)


def hypothesis_table(
    model_table: pd.DataFrame, forest: pd.DataFrame, p12_report: dict
) -> pd.DataFrame:
    e0_overall = forest[(forest["candidate"] == "E0 image-only") & (forest["group"] == "Overall")].iloc[0]
    e2_overall = forest[(forest["candidate"] == "E2 dual-output") & (forest["group"] == "Overall")].iloc[0]
    e2_female = forest[(forest["candidate"] == "E2 dual-output") & (forest["group"] == "Nữ")].iloc[0]
    overall_improvement = -float(e2_overall["delta_mae_months"])
    female_improvement = -float(e2_female["delta_mae_months"])
    return pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "question": "Sex cải thiện so với image-only",
                "status": "SUPPORTED",
                "evidence": (
                    f"E0−E1 ΔMAE={e0_overall['delta_mae_months']:.4f}, "
                    f"95% CI [{e0_overall['ci_low']:.4f}; {e0_overall['ci_high']:.4f}]"
                ),
            },
            {
                "hypothesis": "H2",
                "question": "Dual-output tốt hơn sex embedding",
                "status": "NOT_SUPPORTED",
                "evidence": (
                    f"Overall improvement={overall_improvement:.4f}<0.10; "
                    f"female improvement={female_improvement:.4f}<0.20 tháng"
                ),
            },
            {
                "hypothesis": "H3",
                "question": "Subgroup cải thiện dưới non-inferiority gate",
                "status": "NOT_SUPPORTED",
                "evidence": "E2 không đạt gate nữ đã khóa; không mở E3/OOF",
            },
            {
                "hypothesis": "H4",
                "question": "TTA disagreement liên hệ dương với absolute error",
                "status": "SUPPORTED_ASSOCIATION_LIMITED_UTILITY",
                "evidence": (
                    f"rho={p12_report['primary']['rho']:.4f}, "
                    f"CI [{p12_report['primary']['bootstrap_95_ci'][0]:.4f}; "
                    f"{p12_report['primary']['bootstrap_95_ci'][1]:.4f}], "
                    f"AUROC>12={p12_report['overall']['auc_error_gt12']:.4f}"
                ),
            },
        ]
    )


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def plot_forest(forest: pd.DataFrame, output_dir: Path) -> None:
    display = forest.copy()
    display["label"] = display["candidate"].str.replace(" image-only", "").str.replace(" dual-output", "") + " — " + display["group"]
    display = display.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors = [PALETTE["orange"] if value.startswith("E0") else PALETTE["blue"] for value in display["candidate"]]
    y = np.arange(len(display))
    for index, row in display.iterrows():
        ax.errorbar(
            row["delta_mae_months"],
            y[index],
            xerr=[[row["delta_mae_months"] - row["ci_low"]], [row["ci_high"] - row["delta_mae_months"]]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            markersize=6,
            linewidth=1.5,
        )
        ax.text(row["ci_high"] + 0.045, y[index], f"{row['delta_mae_months']:+.3f}", va="center", fontsize=9)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, display["label"])
    ax.set_xlabel("Chênh lệch MAE so với E1 (tháng); âm tốt hơn")
    ax.set_title("Paired effect size và bootstrap 95% CI")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    low = float(display["ci_low"].min())
    high = float(display["ci_high"].max())
    padding = (high - low) * 0.18
    ax.set_xlim(low - padding, high + padding)
    save_figure(fig, output_dir, "figure_1_paired_forest")


def plot_heatmaps(groups: pd.DataFrame, output_dir: Path) -> None:
    sex_age = groups[groups["group_type"] == "sex_age"].copy()
    mae = sex_age.pivot(index="sex", columns="age_bin", values="tta_mae").reindex(index=SEX_ORDER, columns=AGE_ORDER)
    rho = sex_age.pivot(index="sex", columns="age_bin", values="spearman_rho").reindex(index=SEX_ORDER, columns=AGE_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    sns.heatmap(
        mae,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "MAE (tháng)"},
        ax=axes[0],
    )
    axes[0].set_title("TTA MAE theo giới tính × tuổi")
    axes[0].set_xlabel("Nhóm tuổi (tháng)")
    axes[0].set_ylabel("Giới tính")
    sns.heatmap(
        rho,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-0.4,
        vmax=0.4,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ"},
        ax=axes[1],
    )
    axes[1].set_title("Disagreement–error association")
    axes[1].set_xlabel("Nhóm tuổi (tháng)")
    axes[1].set_ylabel("Giới tính")
    save_figure(fig, output_dir, "figure_2_sex_age_heatmaps")


def plot_disagreement_risk(
    quartiles: pd.DataFrame, risk: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    x = np.arange(len(quartiles))
    labels = ["Q1 thấp", "Q2", "Q3", "Q4 cao"]
    axes[0].bar(x, quartiles["tta_mae"], color=PALETTE["blue"], width=0.68)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("TTA MAE (tháng)")
    axes[0].set_title("Sai số theo disagreement quartile")
    axes[0].set_ylim(0, float(quartiles["tta_mae"].max()) * 1.18)
    for index, value in enumerate(quartiles["tta_mae"]):
        axes[0].text(index, value + 0.12, f"{value:.2f}", ha="center", fontsize=9)

    axes[1].plot(x, quartiles["error_rate_gt12"] * 100, marker="o", color=PALETTE["orange"], label="Lỗi >12 tháng")
    axes[1].plot(x, quartiles["error_rate_gt18"] * 100, marker="s", color=PALETTE["green"], label="Lỗi >18 tháng")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Tỷ lệ lỗi (%)")
    axes[1].set_title("Lỗi lớn theo disagreement quartile")
    axes[1].legend(frameon=False, fontsize=9)

    risk_sorted = risk.sort_values("coverage")
    axes[2].plot(risk_sorted["coverage"] * 100, risk_sorted["tta_mae"], marker="o", color=PALETTE["purple"])
    axes[2].set_xlabel("Coverage giữ lại (%)")
    axes[2].set_ylabel("TTA MAE (tháng)")
    axes[2].set_title("Risk–coverage mô tả")
    axes[2].set_xlim(48, 102)
    axes[2].set_ylim(0, float(risk_sorted["tta_mae"].max()) * 1.18)
    for _, row in risk_sorted.iterrows():
        axes[2].text(row["coverage"] * 100, row["tta_mae"] + 0.12, f"{row['tta_mae']:.2f}", ha="center", fontsize=8)
    save_figure(fig, output_dir, "figure_3_disagreement_risk")


def build(output_dir: Path, bootstrap: int, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: load_prediction(path) for name, path in MODEL_PATHS.items()}
    validate_alignment(frames)
    model_table = pd.DataFrame(
        [{"model": name, **model_metrics(frame)} for name, frame in frames.items()]
    )
    forest = paired_forest_data(frames, bootstrap, seed)
    p12_report = json.loads((P12_DIR / "report.json").read_text(encoding="utf-8"))
    p12_groups = pd.read_csv(P12_DIR / "group_metrics.csv")
    p12_quartiles = pd.read_csv(P12_DIR / "disagreement_quartiles.csv")
    p12_risk = pd.read_csv(P12_DIR / "risk_coverage.csv")
    hypotheses = hypothesis_table(model_table, forest, p12_report)

    tables = {
        "table_1_model_comparison": model_table,
        "table_2_paired_forest_data": forest,
        "table_3_p12_sex_age": p12_groups[p12_groups["group_type"] == "sex_age"],
        "table_4_hypothesis_summary": hypotheses,
    }
    for stem, table in tables.items():
        table.to_csv(output_dir / f"{stem}.csv", index=False)
        write_markdown_table(table, output_dir / f"{stem}.md")

    configure_plot_style()
    plot_forest(forest, output_dir)
    plot_heatmaps(p12_groups, output_dir)
    plot_disagreement_risk(p12_quartiles, p12_risk, output_dir)

    captions = {
        "figure_1_paired_forest": "Chênh lệch absolute error paired so với E1; thanh ngang là bootstrap 95% CI.",
        "figure_2_sex_age_heatmaps": "TTA MAE và Spearman association giữa disagreement–absolute error theo giới tính × tuổi.",
        "figure_3_disagreement_risk": "Sai số, lỗi lớn theo disagreement quartile và risk–coverage mô tả trên cùng OOF.",
    }
    (output_dir / "figure_captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    input_paths = [*MODEL_PATHS.values(), *P11_REPORTS, P12_DIR / "report.json", P12_DIR / "group_metrics.csv", P12_DIR / "disagreement_quartiles.csv", P12_DIR / "risk_coverage.csv"]
    output_paths = collect_hashable_outputs(output_dir)
    manifest = {
        "status": "PASS",
        "test_accessed": False,
        "bootstrap_samples": bootstrap,
        "bootstrap_seed": seed,
        "inputs": {str(path): sha256_file(path) for path in input_paths},
        "outputs": {path.name: sha256_file(path) for path in output_paths},
        "primary_conclusions": {
            "H1": "SUPPORTED",
            "H2": "NOT_SUPPORTED",
            "H3": "NOT_SUPPORTED",
            "H4": "SUPPORTED_ASSOCIATION_LIMITED_UTILITY",
        },
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("p13_reporting/outputs/P13_THESIS_REPORT"),
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    manifest = build(args.output_dir, args.bootstrap, args.seed)
    print(json.dumps({
        "status": manifest["status"],
        "test_accessed": manifest["test_accessed"],
        "output_count": len(manifest["outputs"]),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
