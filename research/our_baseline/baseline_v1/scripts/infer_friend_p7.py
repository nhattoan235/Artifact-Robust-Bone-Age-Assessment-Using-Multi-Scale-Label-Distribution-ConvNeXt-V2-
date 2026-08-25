"""Reproduce inference for the friend's P7 ConvNeXt-Tiny checkpoints.

The script is used only to create an independent 200-image holdout prediction
for the pre-registered 50/50 blend audit. It does not read labels unless the
caller supplies a labeled CSV for verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


TARGET_MEAN = 127.23833273957962
TARGET_STD = 41.248974358112605
IMAGE_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FriendP7Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        feature_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        self.sex_embedding = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim + 16, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(image)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x)
        x = self.backbone.classifier[1](x)
        s = self.sex_embedding(sex)
        return self.regressor(torch.cat([x, s], dim=1)).squeeze(1)


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, path_map: dict[str, Path]) -> None:
        self.frame = frame.reset_index(drop=True)
        self.path_map = path_map
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.path_map[str(row.image_id)]) as source:
            image = source.convert("L").convert("RGB")
            image = self.transform(image)
        sex = torch.tensor([float(row.sex)], dtype=torch.float32)
        return image, sex, str(row.image_id)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def make_path_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*.png"):
        if path.stem in result:
            raise ValueError(f"Duplicate image ID {path.stem}: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def load_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "image_id" if "image_id" in frame.columns else "id"
    sex_col = "sex" if "sex" in frame.columns else "male"
    if id_col not in frame or sex_col not in frame:
        raise ValueError(f"Input must contain image/id and sex/male columns: {list(frame.columns)}")
    result = pd.DataFrame({"image_id": normalize_id(frame[id_col])})
    sex = frame[sex_col]
    if sex.dtype == bool:
        result["sex"] = sex.astype(int)
    else:
        result["sex"] = sex.astype(str).str.strip().str.lower().isin(["true", "1", "male", "m"]).astype(int)
    if result.image_id.duplicated().any():
        raise ValueError("Input contains duplicate image IDs")
    return result


def predict(checkpoint: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    model = FriendP7Model().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    outputs = []
    with torch.no_grad():
        for images, sexes, _ in loader:
            raw = model(images.to(device), sexes.to(device))
            outputs.append((raw * TARGET_STD + TARGET_MEAN).float().cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    frame = load_input(args.input_csv)
    if args.max_samples:
        frame = frame.head(args.max_samples).copy()
    path_map = make_path_map(args.image_root)
    missing = sorted(set(frame.image_id) - set(path_map))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images, examples: {missing[:5]}")
    loader = DataLoader(ImageDataset(frame, path_map), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    checkpoints = sorted(args.checkpoints_root.glob("P7_FINAL_V3_FOLD_*/best_model.pt"))
    if len(checkpoints) != 5:
        raise FileNotFoundError(f"Expected 5 P7 checkpoints, found {len(checkpoints)} under {args.checkpoints_root}")
    result = frame.copy()
    predictions = []
    for checkpoint in checkpoints:
        name = checkpoint.parent.name.lower()
        pred = predict(checkpoint, loader, device)
        result[f"pred_{name}"] = pred
        predictions.append(pred)
    result["friend_p7_prediction_months"] = np.mean(np.stack(predictions), axis=0)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"device={device} n={len(result)} checkpoints={len(checkpoints)} output={args.output_csv}")


if __name__ == "__main__":
    main()
