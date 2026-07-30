from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy import stats
from skimage.exposure import match_histograms
from transformers import AutoModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PROTOCOLS = ("fullframe", "histmatch", "fixed_crop", "fixed_crop_histmatch")


@dataclass(frozen=True)
class Case:
    case_id: str
    sex: str
    female: int
    truth: float


def find_column(frame: pd.DataFrame, names: Iterable[str]) -> str:
    normalized = {
        str(column).strip().lower().replace("_", " "): str(column)
        for column in frame.columns
    }
    for name in names:
        key = name.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Missing one of {tuple(names)}; columns={list(frame.columns)}")


def normalize_case_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalize_sex(value: object, column: str) -> tuple[str, int]:
    text = str(value).strip().upper()
    if column.strip().lower() == "male":
        if text in {"1", "TRUE", "T", "YES"}:
            return "M", 0
        if text in {"0", "FALSE", "F", "NO"}:
            return "F", 1
    if text in {"M", "MALE"}:
        return "M", 0
    if text in {"F", "FEMALE"}:
        return "F", 1
    raise ValueError(f"Unsupported sex value: {value!r}")


def load_cases(path: Path, limit: int | None = None) -> list[Case]:
    frame = pd.read_csv(path)
    id_col = find_column(frame, ("patient_ID", "Case ID", "Case_ID", "id"))
    sex_col = find_column(frame, ("sex", "gender", "male"))
    age_col = find_column(
        frame,
        (
            "bone_age",
            "Bone Age",
            "Ground truth bone age (months)",
            "bone age months",
            "age",
        ),
    )
    cases: list[Case] = []
    for _, row in frame.iterrows():
        sex, female = normalize_sex(row[sex_col], sex_col)
        cases.append(
            Case(
                case_id=normalize_case_id(row[id_col]),
                sex=sex,
                female=female,
                truth=float(row[age_col]),
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Metadata contains duplicate Case IDs")
    return cases[:limit] if limit else cases


def parse_groups(values: list[str]) -> dict[str, Path]:
    groups: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Group must use NAME=DIR syntax, got {value!r}")
        name, directory = value.split("=", 1)
        name = name.strip()
        if not name or name in groups:
            raise ValueError(f"Invalid or duplicate group name: {name!r}")
        groups[name] = Path(directory).resolve()
    if "original" not in groups:
        raise ValueError("A group named 'original' is required as the paired reference")
    return groups


def parse_group_templates(values: list[str] | None) -> dict[str, str]:
    templates: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Group template must use NAME=RELATIVE_TEMPLATE syntax, got {value!r}"
            )
        name, template = value.split("=", 1)
        name = name.strip()
        template = template.strip()
        if not name or not template or name in templates:
            raise ValueError(f"Invalid or duplicate group template: {value!r}")
        if "{case_id}" not in template:
            raise ValueError(
                f"Group template for {name!r} must contain {{case_id}}"
            )
        templates[name] = template
    return templates


def image_path(
    directory: Path, case_id: str, template: str | None = None
) -> Path:
    if template is not None:
        candidate = directory / template.format(case_id=case_id)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"No image for Case ID {case_id} at templated path {candidate}"
        )
    for suffix in SUPPORTED_SUFFIXES:
        candidate = directory / f"{case_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image for Case ID {case_id} in {directory}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        low, high = np.percentile(image.astype(np.float32), (0.1, 99.9))
        if high <= low:
            return np.zeros(image.shape, dtype=np.uint8)
        image = np.clip((image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)
        image = image.astype(np.uint8)
    return image


def fixed_crop_box(mask_path: Path, margin_fraction: float) -> tuple[int, int, int, int]:
    mask = read_gray(mask_path) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"Empty crop mask: {mask_path}")
    height, width = mask.shape
    margin = int(round(max(height, width) * margin_fraction))
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(width, int(xs.max()) + 1 + margin)
    y1 = min(height, int(ys.max()) + 1 + margin)
    return x0, y0, x1, y1


def scale_box(
    box: tuple[int, int, int, int],
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    from_h, from_w = from_shape
    to_h, to_w = to_shape
    return (
        int(round(x0 * to_w / from_w)),
        int(round(y0 * to_h / from_h)),
        int(round(x1 * to_w / from_w)),
        int(round(y1 * to_h / from_h)),
    )


def prepare_image(
    image: np.ndarray,
    protocol: str,
    model: torch.nn.Module,
    reference: np.ndarray | None,
    crop_box: tuple[int, int, int, int] | None,
    crop_mask_shape: tuple[int, int] | None,
) -> np.ndarray:
    output = image
    if protocol.startswith("fixed_crop"):
        if crop_box is None or crop_mask_shape is None:
            raise ValueError(f"{protocol} requires --crop-mask-dir")
        box = scale_box(crop_box, crop_mask_shape, image.shape)
        x0, y0, x1, y1 = box
        output = image[y0:y1, x0:x1]
        if output.size == 0:
            raise ValueError(f"Invalid scaled crop box: {box} for shape {image.shape}")
    if protocol.endswith("histmatch") or protocol == "histmatch":
        if reference is None:
            raise ValueError(f"{protocol} requires the model reference image")
        output = match_histograms(output, reference)
        output = np.clip(output, 0, 255).astype(np.float32)
    return np.asarray(model.preprocess(output), dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, repetitions: int
) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_predictions(
    predictions: pd.DataFrame, bootstrap_repetitions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    case_pair_rows: list[dict[str, object]] = []
    for protocol, protocol_frame in predictions.groupby("Protocol", sort=False):
        for group, frame in protocol_frame.groupby("Group", sort=False):
            error = frame["Prediction_Months"].to_numpy(float) - frame[
                "Ground_Truth_Months"
            ].to_numpy(float)
            absolute = np.abs(error)
            mae_low, mae_high = bootstrap_mean_ci(
                absolute, rng, bootstrap_repetitions
            )
            summary_rows.append(
                {
                    "Protocol": protocol,
                    "Group": group,
                    "N": len(frame),
                    "MAE_Months": absolute.mean(),
                    "MAE_CI95_Low": mae_low,
                    "MAE_CI95_High": mae_high,
                    "RMSE_Months": np.sqrt(np.mean(error**2)),
                    "Median_AE_Months": np.median(absolute),
                    "Bias_Months": error.mean(),
                    "Within_6_Months": np.mean(absolute <= 6),
                    "Within_12_Months": np.mean(absolute <= 12),
                    "Within_24_Months": np.mean(absolute <= 24),
                    "Pearson_vs_Truth": stats.pearsonr(
                        frame["Ground_Truth_Months"], frame["Prediction_Months"]
                    ).statistic,
                }
            )

        original = protocol_frame.loc[
            protocol_frame["Group"] == "original",
            [
                "Case_ID",
                "Ground_Truth_Months",
                "Prediction_Months",
                "Absolute_Error_Months",
            ],
        ].rename(
            columns={
                "Prediction_Months": "Original_Prediction",
                "Absolute_Error_Months": "Original_AE",
            }
        )
        for group in protocol_frame["Group"].drop_duplicates():
            if group == "original":
                continue
            current = protocol_frame.loc[
                protocol_frame["Group"] == group,
                ["Case_ID", "Prediction_Months", "Absolute_Error_Months"],
            ].rename(
                columns={
                    "Prediction_Months": "Processed_Prediction",
                    "Absolute_Error_Months": "Processed_AE",
                }
            )
            paired = original.merge(current, on="Case_ID", validate="one_to_one")
            signed_drift = (
                paired["Processed_Prediction"] - paired["Original_Prediction"]
            ).to_numpy(float)
            absolute_drift = np.abs(signed_drift)
            delta_ae = (paired["Processed_AE"] - paired["Original_AE"]).to_numpy(
                float
            )
            for row in paired.itertuples(index=False):
                case_pair_rows.append(
                    {
                        "Protocol": protocol,
                        "Group": group,
                        "Case_ID": row.Case_ID,
                        "Ground_Truth_Months": row.Ground_Truth_Months,
                        "Original_Prediction_Months": row.Original_Prediction,
                        "Processed_Prediction_Months": row.Processed_Prediction,
                        "Signed_Prediction_Drift": (
                            row.Processed_Prediction - row.Original_Prediction
                        ),
                        "Absolute_Prediction_Drift": abs(
                            row.Processed_Prediction - row.Original_Prediction
                        ),
                        "Original_Absolute_Error": row.Original_AE,
                        "Processed_Absolute_Error": row.Processed_AE,
                        "Delta_AE_vs_Original": row.Processed_AE - row.Original_AE,
                    }
                )
            drift_low, drift_high = bootstrap_mean_ci(
                absolute_drift, rng, bootstrap_repetitions
            )
            delta_low, delta_high = bootstrap_mean_ci(
                delta_ae, rng, bootstrap_repetitions
            )
            try:
                wilcoxon = stats.wilcoxon(
                    paired["Processed_AE"],
                    paired["Original_AE"],
                    alternative="two-sided",
                )
                wilcoxon_stat = float(wilcoxon.statistic)
                wilcoxon_p = float(wilcoxon.pvalue)
            except ValueError:
                wilcoxon_stat, wilcoxon_p = math.nan, math.nan
            paired_rows.append(
                {
                    "Protocol": protocol,
                    "Group": group,
                    "N": len(paired),
                    "Mean_Absolute_Prediction_Drift": absolute_drift.mean(),
                    "Drift_CI95_Low": drift_low,
                    "Drift_CI95_High": drift_high,
                    "Mean_Signed_Prediction_Drift": signed_drift.mean(),
                    "Mean_Delta_AE_vs_Original": delta_ae.mean(),
                    "Delta_AE_CI95_Low": delta_low,
                    "Delta_AE_CI95_High": delta_high,
                    "Pearson_vs_Original": stats.pearsonr(
                        paired["Original_Prediction"],
                        paired["Processed_Prediction"],
                    ).statistic,
                    "Wilcoxon_Statistic": wilcoxon_stat,
                    "Wilcoxon_P": wilcoxon_p,
                }
            )

        group_names = list(protocol_frame["Group"].drop_duplicates())
        for group_a, group_b in combinations(group_names, 2):
            columns = ["Case_ID", "Prediction_Months", "Absolute_Error_Months"]
            frame_a = protocol_frame.loc[
                protocol_frame["Group"] == group_a, columns
            ].rename(
                columns={
                    "Prediction_Months": "Prediction_A",
                    "Absolute_Error_Months": "AE_A",
                }
            )
            frame_b = protocol_frame.loc[
                protocol_frame["Group"] == group_b, columns
            ].rename(
                columns={
                    "Prediction_Months": "Prediction_B",
                    "Absolute_Error_Months": "AE_B",
                }
            )
            paired = frame_a.merge(frame_b, on="Case_ID", validate="one_to_one")
            prediction_difference = (
                paired["Prediction_B"] - paired["Prediction_A"]
            ).to_numpy(float)
            ae_difference = (paired["AE_B"] - paired["AE_A"]).to_numpy(float)
            ae_low, ae_high = bootstrap_mean_ci(
                ae_difference, rng, bootstrap_repetitions
            )
            try:
                wilcoxon = stats.wilcoxon(
                    paired["AE_B"], paired["AE_A"], alternative="two-sided"
                )
                wilcoxon_stat = float(wilcoxon.statistic)
                wilcoxon_p = float(wilcoxon.pvalue)
            except ValueError:
                wilcoxon_stat, wilcoxon_p = math.nan, math.nan
            pairwise_rows.append(
                {
                    "Protocol": protocol,
                    "Group_A": group_a,
                    "Group_B": group_b,
                    "N": len(paired),
                    "Mean_Absolute_Prediction_Difference": np.mean(
                        np.abs(prediction_difference)
                    ),
                    "Mean_Signed_Prediction_B_Minus_A": prediction_difference.mean(),
                    "Mean_AE_B_Minus_A": ae_difference.mean(),
                    "AE_Difference_CI95_Low": ae_low,
                    "AE_Difference_CI95_High": ae_high,
                    "Wilcoxon_Statistic": wilcoxon_stat,
                    "Wilcoxon_P": wilcoxon_p,
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(case_pair_rows),
    )


def save_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    for protocol, frame in predictions.groupby("Protocol", sort=False):
        groups = list(frame["Group"].drop_duplicates())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for group in groups:
            subset = frame[frame["Group"] == group]
            axes[0].scatter(
                subset["Ground_Truth_Months"],
                subset["Prediction_Months"],
                s=12,
                alpha=0.65,
                label=group,
            )
        bounds = [
            min(frame["Ground_Truth_Months"].min(), frame["Prediction_Months"].min()),
            max(frame["Ground_Truth_Months"].max(), frame["Prediction_Months"].max()),
        ]
        axes[0].plot(bounds, bounds, "k--", linewidth=1)
        axes[0].set(
            title=f"Truth vs prediction — {protocol}",
            xlabel="Ground truth (months)",
            ylabel="Prediction (months)",
        )
        axes[0].legend()
        axes[0].grid(alpha=0.2)

        box_values = [
            frame.loc[frame["Group"] == group, "Absolute_Error_Months"].to_numpy()
            for group in groups
        ]
        axes[1].boxplot(box_values, tick_labels=groups, showfliers=False)
        axes[1].set(
            title=f"Absolute error — {protocol}",
            ylabel="Absolute error (months)",
        )
        axes[1].grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{protocol}_accuracy.png", dpi=180)
        plt.close(fig)

        original = frame[frame["Group"] == "original"][
            ["Case_ID", "Prediction_Months"]
        ].rename(columns={"Prediction_Months": "Original"})
        processed_groups = [group for group in groups if group != "original"]
        if not processed_groups:
            continue
        fig, axes = plt.subplots(
            1, len(processed_groups), figsize=(6 * len(processed_groups), 5), squeeze=False
        )
        for axis, group in zip(axes[0], processed_groups):
            processed = frame[frame["Group"] == group][
                ["Case_ID", "Prediction_Months"]
            ].rename(columns={"Prediction_Months": "Processed"})
            paired = original.merge(processed, on="Case_ID")
            mean_prediction = (paired["Original"] + paired["Processed"]) / 2
            difference = paired["Processed"] - paired["Original"]
            bias = difference.mean()
            sd = difference.std(ddof=1)
            axis.scatter(mean_prediction, difference, s=15, alpha=0.7)
            axis.axhline(bias, color="tab:blue", label=f"bias={bias:.2f}")
            axis.axhline(bias + 1.96 * sd, color="tab:red", linestyle="--")
            axis.axhline(bias - 1.96 * sd, color="tab:red", linestyle="--")
            axis.set(
                title=f"{group} vs original — {protocol}",
                xlabel="Mean prediction (months)",
                ylabel="Processed − original (months)",
            )
            axis.legend()
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{protocol}_bland_altman.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bone-age benchmark for original, author, and proposed images."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Repeat NAME=DIR. A group named original is mandatory.",
    )
    parser.add_argument(
        "--group-image-template",
        action="append",
        help=(
            "Optional NAME=RELATIVE_TEMPLATE for nested/variant images; "
            "the template must contain {case_id}."
        ),
    )
    parser.add_argument(
        "--protocol",
        action="append",
        choices=PROTOCOLS,
        help="Repeat to run multiple protocols. Default: fullframe.",
    )
    parser.add_argument("--crop-mask-dir", type=Path)
    parser.add_argument("--crop-margin-fraction", type=float, default=0.03)
    parser.add_argument("--model-id", default="ianpan/bone-age")
    parser.add_argument(
        "--revision", default="2ab81275b84e9f518f04584177221a1d8c1dc1a5"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("external/hf_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    groups = parse_groups(args.group)
    group_templates = parse_group_templates(args.group_image_template)
    unknown_template_groups = sorted(set(group_templates) - set(groups))
    if unknown_template_groups:
        raise SystemExit(
            "Templates supplied for unknown groups: "
            + ", ".join(unknown_template_groups)
        )
    protocols = args.protocol or ["fullframe"]
    cases = load_cases(args.metadata_csv.resolve(), args.limit)
    for name, directory in groups.items():
        if not directory.is_dir():
            raise SystemExit(f"Missing group directory {name}: {directory}")
        missing = [
            case.case_id
            for case in cases
            if (
                (
                    group_templates.get(name) is not None
                    and not (
                        directory
                        / group_templates[name].format(case_id=case.case_id)
                    ).exists()
                )
                or (
                    group_templates.get(name) is None
                    and not any(
                        (directory / f"{case.case_id}{suffix}").exists()
                        for suffix in SUPPORTED_SUFFIXES
                    )
                )
            )
        ]
        if missing:
            raise SystemExit(
                f"Group {name} is missing {len(missing)} images; first IDs: {missing[:10]}"
            )
    if any(protocol.startswith("fixed_crop") for protocol in protocols):
        if args.crop_mask_dir is None or not args.crop_mask_dir.is_dir():
            raise SystemExit("Fixed-crop protocols require a valid --crop-mask-dir")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id}@{args.revision} on {device} ...", flush=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir.resolve()),
    )
    model = model.eval().to(device)

    reference: np.ndarray | None = None
    reference_path: Path | None = None
    if any("histmatch" in protocol for protocol in protocols):
        reference_path = Path(
            hf_hub_download(
                repo_id=args.model_id,
                filename="ref_img.png",
                revision=args.revision,
                cache_dir=str(args.cache_dir.resolve()),
            )
        )
        reference = read_gray(reference_path)

    rows: list[dict[str, object]] = []
    pending: list[tuple[Case, str, str, Path, np.ndarray]] = []

    def flush_batch() -> None:
        if not pending:
            return
        tensor = torch.from_numpy(
            np.stack([item[4] for item in pending], axis=0)
        ).unsqueeze(1)
        female = torch.tensor([item[0].female for item in pending], dtype=torch.long)
        with torch.inference_mode():
            values = model(tensor.float().to(device), female.to(device))
        predictions = values.detach().cpu().numpy().reshape(-1)
        for (case, protocol, group, path, _), prediction in zip(
            pending, predictions
        ):
            prediction = float(prediction)
            error = prediction - case.truth
            rows.append(
                {
                    "Case_ID": case.case_id,
                    "Sex": case.sex,
                    "Ground_Truth_Months": case.truth,
                    "Protocol": protocol,
                    "Group": group,
                    "Prediction_Months": prediction,
                    "Error_Months": error,
                    "Absolute_Error_Months": abs(error),
                    "Source_Path": str(path),
                }
            )
        pending.clear()

    total = len(cases) * len(protocols) * len(groups)
    completed = 0
    for protocol in protocols:
        for case in cases:
            crop_box = None
            crop_mask_shape = None
            if protocol.startswith("fixed_crop"):
                mask_path = image_path(args.crop_mask_dir.resolve(), case.case_id)
                mask = read_gray(mask_path)
                crop_mask_shape = mask.shape
                crop_box = fixed_crop_box(mask_path, args.crop_margin_fraction)
            for group, directory in groups.items():
                path = image_path(
                    directory, case.case_id, group_templates.get(group)
                )
                prepared = prepare_image(
                    read_gray(path),
                    protocol,
                    model,
                    reference,
                    crop_box,
                    crop_mask_shape,
                )
                pending.append((case, protocol, group, path, prepared))
                if len(pending) >= args.batch_size:
                    flush_batch()
                completed += 1
                if completed % max(1, len(groups) * 10) == 0 or completed == total:
                    print(f"[{completed}/{total}] prepared/inferred", flush=True)
    flush_batch()

    predictions = pd.DataFrame(rows).sort_values(
        ["Protocol", "Case_ID", "Group"]
    )
    summary, paired, pairwise, case_pairs = summarize_predictions(
        predictions, args.bootstrap_repetitions, args.seed
    )
    predictions.to_csv(args.output_dir / "predictions_long.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_original.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_group_comparisons.csv", index=False)
    case_pairs.to_csv(args.output_dir / "case_level_vs_original.csv", index=False)
    save_plots(predictions, args.output_dir)

    manifest = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "groups": {name: str(path) for name, path in groups.items()},
        "group_image_templates": group_templates,
        "protocols": protocols,
        "metadata_csv": str(args.metadata_csv.resolve()),
        "metadata_sha256": sha256(args.metadata_csv.resolve()),
        "reference_image": str(reference_path) if reference_path else None,
        "crop_mask_dir": str(args.crop_mask_dir.resolve()) if args.crop_mask_dir else None,
        "crop_margin_fraction": args.crop_margin_fraction,
        "case_count": len(cases),
        "seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nSummary:", flush=True)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPaired vs original:", flush=True)
    if paired.empty:
        print("(no processed groups)")
    else:
        print(paired.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAll pairwise group comparisons:", flush=True)
    print(pairwise.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved benchmark to: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
