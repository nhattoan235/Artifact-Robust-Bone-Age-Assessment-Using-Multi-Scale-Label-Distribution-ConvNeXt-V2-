from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_manifest(path: str | Path, expected_split: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "image_id", "bone_age_months", "sex", "image_path", "sha256", "readable"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest thiếu cột bắt buộc: {path}")
    if any(row["split"] != expected_split for row in rows):
        raise ValueError(f"Manifest chứa split khác {expected_split}: {path}")
    if any(not row["bone_age_months"] for row in rows):
        raise ValueError(f"Manifest phát triển có nhãn tuổi trống: {path}")
    return rows


def manifest_hash(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    keys = ["split", "image_id", "bone_age_months", "sex", "sha256"]
    curated_keys = [
        "age_bin", "sex_age_stratum", "sample_weight", "audit_status", "audit_reason",
    ]
    # Giữ fingerprint P0 bit-exact cho manifest cũ. Chỉ manifest C1 có các cột
    # curated mới đưa chúng vào hash để đổi weight không thể resume nhầm checkpoint.
    keys.extend(key for key in curated_keys if any(key in row for row in rows))
    # Phải giữ đúng thuật toán khóa ở P0: ID đã chuẩn hóa nhưng sắp xếp lexical.
    # Không đổi sang numeric sort vì sẽ làm fingerprint khác dù dữ liệu giống hệt.
    for row in sorted(rows, key=lambda item: str(int(float(item["image_id"])))):
        digest.update(("\t".join(row.get(key, "") for key in keys) + "\n").encode("utf-8"))
    return digest.hexdigest()


def pad_square(image: Image.Image, fill: int = 0) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


class BoneAgeDataset(Dataset):
    def __init__(
        self, rows: list[dict[str, str]], image_size: int, target_mean: float,
        target_std: float, train: bool = False, epoch: int = 0, seed: int = 42,
        augmentation: str = "none", horizontal_flip_probability: float = 0.0,
        rotation_degrees: float = 0.0, translation_fraction: float = 0.0,
        scale_min: float = 1.0, scale_max: float = 1.0,
        brightness_delta: float = 0.0, contrast_delta: float = 0.0,
        gamma_min: float = 1.0, gamma_max: float = 1.0,
        shear_degrees: float = 0.0, clahe_probability: float = 0.0,
        sharpen_probability: float = 0.0,
        preprocessing: str = "none", preprocessed_root: str = "",
        image_root: str = "", image_normalization: str = "imagenet",
    ):
        self.rows = rows
        self.image_size = image_size
        self.target_mean = target_mean
        self.target_std = target_std
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
        self.shear_degrees = shear_degrees
        self.clahe_probability = clahe_probability
        self.sharpen_probability = sharpen_probability
        self.preprocessing = preprocessing
        self.preprocessed_root = Path(preprocessed_root) if preprocessed_root else None
        self.image_root = Path(image_root) if image_root else None
        self.image_normalization = image_normalization

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        path = Path(row["image_path"])
        if self.image_root is not None and not path.is_absolute():
            path = self.image_root / path
        if self.preprocessing != "none":
            if self.preprocessed_root is None:
                raise RuntimeError(f"{self.preprocessing} thiếu preprocessed_root")
            path = self.preprocessed_root / row["split"] / f"{row['image_id']}.png"
            if not path.is_file():
                raise RuntimeError(f"Thiếu cache preprocessing cho ID={row['image_id']}: {path}")
        try:
            with Image.open(path) as source:
                image = source.convert("L")
                image = pad_square(image)
                image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
                if self.train and self.augmentation != "none":
                    image = self._augment(image, row["image_id"])
                tensor = TF.pil_to_tensor(image).float().div_(255.0).repeat(3, 1, 1)
                if self.image_normalization == "imagenet":
                    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
                elif self.image_normalization == "per_image_zscore":
                    tensor = (tensor - tensor.mean()) / tensor.std().clamp_min(1e-6)
        except Exception as exc:
            raise RuntimeError(f"Không đọc được ảnh {path}: {exc}") from exc
        age = float(row["bone_age_months"])
        sex = 1.0 if row["sex"] == "M" else 0.0
        return {
            "image": tensor,
            "sex": torch.tensor([sex], dtype=torch.float32),
            "target_norm": torch.tensor((age - self.target_mean) / self.target_std, dtype=torch.float32),
            "target_months": torch.tensor(age, dtype=torch.float32),
            "image_id": row["image_id"],
            "sex_text": row["sex"],
        }

    def _rng(self, image_id: str) -> random.Random:
        token = f"{self.seed}|{self.epoch}|{image_id}|{self.augmentation}"
        value = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        return random.Random(value)

    def _augment(self, image: Image.Image, image_id: str) -> Image.Image:
        """Augmentation xác định bởi seed+epoch+ID, nên resume không đổi ảnh."""
        rng = self._rng(image_id)
        if rng.random() < self.horizontal_flip_probability:
            image = TF.hflip(image)
        if self.augmentation == "flip":
            return image
        angle = rng.uniform(-self.rotation_degrees, self.rotation_degrees)
        max_shift = self.translation_fraction * self.image_size
        translate = [int(round(rng.uniform(-max_shift, max_shift))), int(round(rng.uniform(-max_shift, max_shift)))]
        scale = rng.uniform(self.scale_min, self.scale_max)
        shear = rng.uniform(-self.shear_degrees, self.shear_degrees) if self.augmentation == "deeplasia_fancy" else 0.0
        image = TF.affine(
            image, angle=angle, translate=translate, scale=scale, shear=[shear, 0.0],
            interpolation=InterpolationMode.BILINEAR, fill=0,
        )
        if self.augmentation == "deeplasia_fancy":
            if rng.random() < self.sharpen_probability:
                image = TF.adjust_sharpness(image, sharpness_factor=rng.uniform(1.5, 2.0))
            # Deeplasia dùng OneOf(CLAHE, RandomGamma); chọn tối đa một
            # phép biến đổi cường độ để không vô tình áp dụng cả hai.
            choose_clahe = rng.random() < 0.5
            if choose_clahe and rng.random() < self.clahe_probability:
                array = np.asarray(image, dtype=np.uint8)
                image = Image.fromarray(
                    cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(array),
                    mode="L",
                )
            elif not choose_clahe and (self.gamma_min != 1.0 or self.gamma_max != 1.0):
                image = TF.adjust_gamma(image, rng.uniform(self.gamma_min, self.gamma_max))
        else:
            if self.brightness_delta > 0:
                image = TF.adjust_brightness(image, rng.uniform(1.0 - self.brightness_delta, 1.0 + self.brightness_delta))
            if self.contrast_delta > 0:
                image = TF.adjust_contrast(image, rng.uniform(1.0 - self.contrast_delta, 1.0 + self.contrast_delta))
            if self.gamma_min != 1.0 or self.gamma_max != 1.0:
                image = TF.adjust_gamma(image, rng.uniform(self.gamma_min, self.gamma_max))
        return image


class EpochPermutationSampler(Sampler[int]):
    """Permutation tái lập được; start_index hỗ trợ resume giữa epoch."""
    def __init__(self, length: int, seed: int, epoch: int, start_index: int = 0):
        generator = torch.Generator().manual_seed(seed + epoch)
        self.indices = torch.randperm(length, generator=generator).tolist()[start_index:]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class EpochWeightedSampler(Sampler[int]):
    """Weighted epoch sequence deterministic by seed+epoch, resumable by suffix."""

    def __init__(
        self, weights: list[float], num_samples: int, seed: int, epoch: int,
        start_index: int = 0,
    ):
        tensor = torch.as_tensor(weights, dtype=torch.double)
        if tensor.ndim != 1 or len(tensor) == 0:
            raise ValueError("sample weights phải là vector không rỗng")
        if not torch.isfinite(tensor).all() or (tensor <= 0).any():
            raise ValueError("sample weights phải hữu hạn và > 0")
        if num_samples <= 0 or not 0 <= start_index <= num_samples:
            raise ValueError("num_samples/start_index không hợp lệ")
        generator = torch.Generator().manual_seed(seed + epoch)
        full = torch.multinomial(
            tensor, num_samples=num_samples, replacement=True, generator=generator,
        ).tolist()
        self.indices = full[start_index:]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def build_train_sampler(
    rows: list[dict[str, str]], strategy: str, seed: int, epoch: int,
    start_index: int = 0,
) -> Sampler[int]:
    if strategy == "permutation":
        return EpochPermutationSampler(len(rows), seed, epoch, start_index)
    if strategy != "manifest_weighted":
        raise ValueError(f"sampling_strategy không được hỗ trợ: {strategy}")
    try:
        weights = [float(row["sample_weight"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest weighted thiếu sample_weight hợp lệ") from exc
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Manifest weighted có sample_weight không hữu hạn hoặc <= 0")
    return EpochWeightedSampler(weights, len(rows), seed, epoch, start_index)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
