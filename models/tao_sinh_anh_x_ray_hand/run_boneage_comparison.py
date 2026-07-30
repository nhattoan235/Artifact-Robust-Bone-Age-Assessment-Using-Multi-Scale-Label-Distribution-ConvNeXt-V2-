from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _find_column(frame: pd.DataFrame, names: tuple[str, ...], required: bool = True) -> str | None:
    normalized = {
        str(column).strip().lower().replace("_", " "): column
        for column in frame.columns
    }
    for name in names:
        key = name.strip().lower().replace("_", " ")
        if key in normalized:
            return str(normalized[key])
    if required:
        raise ValueError(
            f"Missing one of columns {names}; available columns: {list(frame.columns)}"
        )
    return None


def _normalize_sex(value: object, column_name: str) -> str:
    text = str(value).strip().upper()
    if column_name.strip().lower() == "male":
        if text in {"1", "TRUE", "T", "YES"}:
            return "M"
        if text in {"0", "FALSE", "F", "NO"}:
            return "F"
    if text in {"M", "MALE", "1", "TRUE"}:
        return "M"
    if text in {"F", "FEMALE", "0", "FALSE"}:
        return "F"
    raise ValueError(f"Unsupported sex value: {value!r}")


def _image_path(directory: Path, case_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = directory / f"{case_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for Case_ID {case_id} in {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the author's sex-specific bone-age ensemble on original and cleaned images."
    )
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--cleaned-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--author-code-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "external" / "author_bone_age_repo",
    )
    parser.add_argument("--expected-folds", type=int, default=5)
    args = parser.parse_args()

    for directory in (
        args.original_dir,
        args.cleaned_dir,
        args.ensemble_dir,
        args.author_code_dir,
    ):
        if not directory.exists():
            raise SystemExit(f"Missing directory: {directory}")
    if not args.metadata_csv.exists():
        raise SystemExit(f"Missing metadata CSV: {args.metadata_csv}")

    male_weights = sorted(args.ensemble_dir.glob("male_resnet50_fold*_best_score.pth"))
    female_weights = sorted(args.ensemble_dir.glob("female_resnet50_fold*_best_score.pth"))
    if len(male_weights) != args.expected_folds or len(female_weights) != args.expected_folds:
        raise SystemExit(
            "Checkpoint mismatch: expected "
            f"{args.expected_folds} male + {args.expected_folds} female files, found "
            f"{len(male_weights)} male + {len(female_weights)} female in {args.ensemble_dir}"
        )

    sys.path.insert(0, str(args.author_code_dir.resolve()))
    try:
        import torch
        from app import device, load_model_ensemble, preprocess_image
    except ImportError as error:
        raise SystemExit(
            f"Missing inference dependency: {error}. "
            "Install requirements_boneage.txt first."
        ) from error

    metadata = pd.read_csv(args.metadata_csv)
    case_column = _find_column(metadata, ("Case ID", "Case_ID", "id"))
    sex_column = _find_column(metadata, ("Sex", "gender", "male"))
    age_column = _find_column(
        metadata,
        ("Bone Age", "boneage", "bone age months", "age"),
        required=False,
    )

    print(f"Loading {args.expected_folds} male and {args.expected_folds} female models...")
    ensembles = {
        "M": load_model_ensemble(0, str(args.ensemble_dir)),
        "F": load_model_ensemble(1, str(args.ensemble_dir)),
    }

    def predict_one(path: Path, sex: str) -> float:
        image = np.asarray(Image.open(path).convert("L"))
        tensor = preprocess_image(image)
        with torch.no_grad():
            values = [
                float(model(tensor.unsqueeze(0)).detach().cpu().reshape(-1)[0])
                for model in ensembles[sex]
            ]
        return round(float(np.mean(values) * 200.0), 2)

    results: list[dict[str, object]] = []
    for index, row in metadata.iterrows():
        case_id = str(row[case_column]).strip()
        sex = _normalize_sex(row[sex_column], sex_column)
        original_prediction = predict_one(_image_path(args.original_dir, case_id), sex)
        cleaned_prediction = predict_one(_image_path(args.cleaned_dir, case_id), sex)
        result: dict[str, object] = {
            "Case_ID": case_id,
            "Sex": sex,
            "Original_Predicted_Months": original_prediction,
            "Cleaned_Predicted_Months": cleaned_prediction,
            "Delta_Cleaned_Minus_Original": round(
                cleaned_prediction - original_prediction, 2
            ),
            "Absolute_Paired_Delta": round(
                abs(cleaned_prediction - original_prediction), 2
            ),
        }
        if age_column is not None:
            truth = float(row[age_column])
            result["Ground_Truth_Months"] = truth
            result["Original_Absolute_Error"] = abs(original_prediction - truth)
            result["Cleaned_Absolute_Error"] = abs(cleaned_prediction - truth)
        results.append(result)
        print(f"[{index + 1:03d}/{len(metadata):03d}] {case_id}")

    output = pd.DataFrame(results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    paired_delta = output["Absolute_Paired_Delta"].to_numpy(float)
    print(f"Mean absolute paired delta: {paired_delta.mean():.3f} months")
    if age_column is not None:
        truth = output["Ground_Truth_Months"].to_numpy(float)
        original = output["Original_Predicted_Months"].to_numpy(float)
        cleaned = output["Cleaned_Predicted_Months"].to_numpy(float)
        for name, prediction in (("original", original), ("cleaned", cleaned)):
            error = prediction - truth
            print(
                f"{name}: MAE={np.mean(np.abs(error)):.3f}, "
                f"RMSE={np.sqrt(np.mean(error ** 2)):.3f}, "
                f"bias={np.mean(error):+.3f} months"
            )
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
