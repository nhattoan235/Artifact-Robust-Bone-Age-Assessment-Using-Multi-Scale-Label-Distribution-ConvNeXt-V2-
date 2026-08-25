"""Verify the friend's checkpoint with the friend's own model/data code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--stored-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root))
    from p1_baseline.data import BoneAgeDataset
    from p1_baseline.model import build_model

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    stored = pd.read_csv(args.stored_predictions).head(args.max_samples)
    source = pd.read_csv(args.source_manifest, dtype={"image_id": str})
    source["image_id"] = source["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    local_paths = {path.stem: str(path.resolve()) for path in args.data_root.rglob("*.png")}
    source["image_path"] = source["image_id"].map(local_paths)
    if source["image_path"].isna().any():
        raise FileNotFoundError("A validation image is missing from --data-root")
    source = source[source.image_id.isin(stored.image_id.astype(str))].copy()
    source = source.set_index("image_id").loc[stored.image_id.astype(str)].reset_index()
    rows = source.to_dict("records")
    dataset = BoneAgeDataset(
        rows, image_size=512, target_mean=127.23833273957962, target_std=41.248974358112605,
        train=False, image_normalization="imagenet",
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = build_model("convnext_tiny", False, 16, 256, 0.2, 229, sex_mode="embedding").to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            prediction_norm = model(batch["image"].to(device), batch["sex"].to(device))
            predictions.extend((prediction_norm * 41.248974358112605 + 127.23833273957962).cpu().tolist())
    result = stored.copy()
    result["reproduced_prediction_months"] = predictions
    result["absolute_difference"] = (result["prediction_months"] - result["reproduced_prediction_months"]).abs()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result[["image_id", "prediction_months", "reproduced_prediction_months", "absolute_difference"]].to_string(index=False))
    print(f"device={device} n={len(result)} max_abs_difference={result.absolute_difference.max():.9f}")


if __name__ == "__main__":
    main()
