from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .schema import ROI_NAMES, VIEW_NAMES


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EpochPermutationSampler(Sampler[int]):
    def __init__(self, length: int, seed: int, epoch: int, start_index: int = 0):
        generator = torch.Generator().manual_seed(seed + epoch)
        self.indices = torch.randperm(length, generator=generator).tolist()[start_index:]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def split_records(rows: list[dict[str, str]], *, validation_fold: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ids = [str(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source image_id in master manifest")
    train = [row for row in rows if int(row["fold"]) != validation_fold]
    validation = [row for row in rows if int(row["fold"]) == validation_fold]
    if not train or not validation:
        raise ValueError(f"empty train/validation split for fold {validation_fold}")
    if {row["image_id"] for row in train} & {row["image_id"] for row in validation}:
        raise RuntimeError("source ID leakage between train and validation")
    return train, validation


def _view_names(mode: str) -> tuple[str, ...]:
    if mode == "global_only":
        return ("global",)
    if mode == "six_roi_only":
        return ROI_NAMES
    if mode == "global_plus_six":
        return VIEW_NAMES
    raise ValueError(f"unsupported view_mode: {mode}")


def _pad_square(image: Image.Image) -> Image.Image:
    side = max(image.size)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


class MultiViewBoneAgeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        image_root: str | Path,
        image_size: int,
        target_mean: float,
        target_std: float,
        view_mode: str = "global_plus_six",
        train: bool = False,
        epoch: int = 0,
        seed: int = 42,
        augmentation: str = "none",
        horizontal_flip_probability: float = 0.5,
        rotation_degrees: float = 7.0,
        translation_fraction: float = 0.03,
        scale_min: float = 0.95,
        scale_max: float = 1.05,
        brightness_delta: float = 0.10,
        contrast_delta: float = 0.10,
        gamma_min: float = 0.90,
        gamma_max: float = 1.10,
    ):
        self.rows = rows
        self.image_root = Path(image_root)
        self.image_size = int(image_size)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.view_names = _view_names(view_mode)
        self.train = train
        self.epoch = epoch
        self.seed = seed
        self.augmentation = augmentation
        self.horizontal_flip_probability = horizontal_flip_probability
        self.rotation_degrees = rotation_degrees
        self.translation_fraction = translation_fraction
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, row: dict[str, str], name: str) -> Path:
        value = row["global_path"] if name == "global" else row[f"roi_{name}_path"]
        path = Path(value)
        return path if path.is_absolute() else self.image_root / path

    def _rng(self, image_id: str) -> random.Random:
        token = f"{self.seed}|{self.epoch}|{image_id}|{self.augmentation}|C4"
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        return random.Random(seed)

    def _parameters(self, image_id: str) -> dict:
        rng = self._rng(image_id)
        return {
            "flip": rng.random() < self.horizontal_flip_probability,
            "angle": rng.uniform(-self.rotation_degrees, self.rotation_degrees),
            "translate": [
                int(round(rng.uniform(-self.translation_fraction, self.translation_fraction) * self.image_size)),
                int(round(rng.uniform(-self.translation_fraction, self.translation_fraction) * self.image_size)),
            ],
            "scale": rng.uniform(self.scale_min, self.scale_max),
            "brightness": rng.uniform(1.0 - self.brightness_delta, 1.0 + self.brightness_delta),
            "contrast": rng.uniform(1.0 - self.contrast_delta, 1.0 + self.contrast_delta),
            "gamma": rng.uniform(self.gamma_min, self.gamma_max),
        }

    def _prepare(self, image: Image.Image, parameters: dict | None) -> torch.Tensor:
        image = _pad_square(image.convert("L"))
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if parameters is not None:
            if parameters["flip"]:
                image = TF.hflip(image)
            image = TF.affine(
                image,
                angle=parameters["angle"],
                translate=parameters["translate"],
                scale=parameters["scale"],
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            image = TF.adjust_brightness(image, parameters["brightness"])
            image = TF.adjust_contrast(image, parameters["contrast"])
            image = TF.adjust_gamma(image, parameters["gamma"])
        tensor = TF.pil_to_tensor(image).float().div_(255.0).repeat(3, 1, 1)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        parameters = self._parameters(row["image_id"]) if self.train and self.augmentation != "none" else None
        tensors = []
        for name in self.view_names:
            path = self._path(row, name)
            try:
                with Image.open(path) as handle:
                    tensors.append(self._prepare(handle, parameters))
            except Exception as exc:
                raise RuntimeError(f"cannot read C4 view {name} for ID={row['image_id']}: {path}: {exc}") from exc
        age = float(row["bone_age_months"])
        sex_value = 1.0 if row["sex"] == "M" else 0.0
        return {
            "views": torch.stack(tensors),
            "view_names": self.view_names,
            "sex": torch.tensor([sex_value], dtype=torch.float32),
            "target_norm": torch.tensor((age - self.target_mean) / self.target_std, dtype=torch.float32),
            "target_months": torch.tensor(age, dtype=torch.float32),
            "image_id": row["image_id"],
            "sex_text": row["sex"],
            "fold": row["fold"],
        }
