"""Evaluate the retrained P7 and blend variants on the labeled 200-image test.

This is explicitly exploratory. The test labels are read only because the
user requested an evaluation; no method selection should be based on this
report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "n": int(len(target)),
        "mae_months": float(absolute.mean()),
        "rmse_months": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_months": float(np.median(absolute)),
        "accuracy_pm6": float(np.mean(absolute <= 6)),
        "accuracy_pm12": float(np.mean(absolute <= 12)),
        "accuracy_pm18": float(np.mean(absolute <= 18)),
    }


def load_model(repo_root: Path, checkpoint: Path, target_mean: float, target_std: float, device: torch.device):
    from p1_baseline.model import build_model

    model = build_model("convnext_tiny", False, 16, 256, 0.2, 229, sex_mode="embedding").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model, target_mean, target_std


def predict(repo_root: Path, frame: pd.DataFrame, checkpoint: Path, target_mean: float, target_std: float, device: torch.device, batch_size: int) -> np.ndarray:
    from p1_baseline.data import BoneAgeDataset

    rows = frame.to_dict("records")
    dataset = BoneAgeDataset(
        rows, image_size=512, target_mean=target_mean, target_std=target_std,
        train=False, image_normalization="imagenet",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model, target_mean, target_std = load_model(repo_root, checkpoint, target_mean, target_std, device)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            raw = model(batch["image"].to(device), batch["sex"].to(device))
            predictions.extend((raw * target_std + target_mean).float().cpu().numpy())
    return np.asarray(predictions, dtype=np.float64)


def make_frame(data_root: Path, test_csv: Path) -> pd.DataFrame:
    source = pd.read_csv(test_csv)
    id_col = "id" if "id" in source.columns else "Image ID"
    target_col = "boneage" if "boneage" in source.columns else "Bone Age (months)"
    sex_col = "male" if "male" in source.columns else "sex"
    frame = pd.DataFrame({"image_id": source[id_col].astype(str).str.replace(r"\.0$", "", regex=True)})
    frame["target_months"] = pd.to_numeric(source[target_col], errors="raise")
    sex = source[sex_col]
    frame["sex"] = sex.astype(str).str.strip().str.lower().isin(["true", "1", "male", "m"]).map({True: "M", False: "F"})
    path_map: dict[str, Path] = {}
    for path in data_root.rglob("*.png"):
        if path.stem in path_map:
            raise ValueError(f"Duplicate image ID: {path.stem}")
        path_map[path.stem] = path.resolve()
    frame["image_path"] = frame.image_id.map(lambda value: str(path_map[value]))
    if frame.image_path.isna().any():
        raise FileNotFoundError("Missing test image in data_root")
    frame["split"] = "test_exploratory"
    frame["bone_age_months"] = frame["target_months"]
    frame["sex_text"] = frame["sex"]
    return frame[["split", "image_id", "bone_age_months", "sex", "image_path", "target_months"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--retrain-checkpoint", type=Path, required=True)
    parser.add_argument("--friend-root", type=Path, required=True)
    parser.add_argument("--friend-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = make_frame(args.data_root, args.test_csv)
    target = frame.target_months.to_numpy(dtype=np.float64)

    # EXP-004 retrained model: statistics are recorded in its generated config/audit.
    retrain_mean = 127.30549467752036
    retrain_std = 41.309203642993275
    retrain_pred = predict(args.repo_root, frame, args.retrain_checkpoint, retrain_mean, retrain_std, device, args.batch_size)

    # Friend P7 OOF ensemble: use the five released best_model.pt checkpoints.
    friend_mean = 127.23833273957962
    friend_std = 41.248974358112605
    friend_files = []
    if args.friend_predictions is not None:
        stored = pd.read_csv(args.friend_predictions)
        stored["image_id"] = stored["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        stored = stored.set_index("image_id").loc[frame.image_id].reset_index()
        friend_pred = stored["prediction_months"].to_numpy(dtype=np.float64)
    else:
        friend_predictions = []
        for checkpoint in sorted(args.friend_root.glob("P7_FINAL_V3_FOLD_*/best_model.pt")):
            friend_files.append(str(checkpoint))
            friend_predictions.append(predict(args.repo_root, frame, checkpoint, friend_mean, friend_std, device, args.batch_size))
        if len(friend_predictions) != 5:
            raise FileNotFoundError(f"Expected 5 friend P7 checkpoints, found {len(friend_predictions)}")
        friend_pred = np.mean(np.stack(friend_predictions), axis=0)
    blend50 = 0.50 * retrain_pred + 0.50 * friend_pred
    blend25_friend = 0.75 * retrain_pred + 0.25 * friend_pred

    result = frame[["image_id", "target_months", "sex"]].copy()
    result["prediction_retrain"] = retrain_pred
    result["prediction_friend_p7_ensemble"] = friend_pred
    result["prediction_blend_50_50"] = blend50
    result["prediction_blend_75_retrain_25_friend"] = blend25_friend
    result.to_csv(args.output_dir / "test200_method_predictions.csv", index=False)

    report = {
        "evaluation": "exploratory_test_200",
        "test_labels_used": True,
        "selection_warning": "Do not use this test report to select weights or hyperparameters; the 200-image test has been inspected previously.",
        "device": str(device),
        "n": int(len(frame)),
        "retrain_checkpoint": str(args.retrain_checkpoint),
        "friend_checkpoints": friend_files,
        "friend_prediction_file": str(args.friend_predictions) if args.friend_predictions else None,
        "methods": {
            "retrain": metrics(target, retrain_pred),
            "friend_p7_ensemble": metrics(target, friend_pred),
            "blend_50_50": metrics(target, blend50),
            "blend_75_retrain_25_friend": metrics(target, blend25_friend),
        },
        "prediction_file": str(args.output_dir / "test200_method_predictions.csv"),
    }
    (args.output_dir / "test200_method_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
