"""
RSNA Bone Age Dataset — tiền xử lý khớp mô tả baseline:
- foreground cropping tự động (nonzero-pixel bounding box)
- resize về image_size x image_size
- RGB hoá qua channel duplication
- intensity normalization (min-max -> ImageNet mean/std)
- augmentation: affine, brightness/contrast, sharpen (chỉ áp dụng khi train=True)

Yêu cầu cấu trúc thư mục:
    data_root/
        train_csv        (cột: id, boneage, male)
        train_img_dir/{id}.png
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def foreground_crop(img: np.ndarray, thresh: int = 10) -> np.ndarray:
    """Crop theo bounding box của vùng pixel khác nền (nonzero-pixel bbox),
    tương đương ý tưởng foreground cropping của MONAI CropForeground."""
    mask = img > thresh
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1]


def build_transforms(cfg, train: bool) -> A.Compose:
    ops = [A.Resize(cfg.image_size, cfg.image_size)]
    if train:
        ops += [
            A.Affine(
                rotate=(-cfg.aug_rotate_deg, cfg.aug_rotate_deg),
                translate_percent=cfg.aug_translate_pct,
                scale=cfg.aug_scale_range,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.aug_brightness_contrast,
                contrast_limit=cfg.aug_brightness_contrast,
                p=0.5,
            ),
            A.Sharpen(p=cfg.aug_sharpen_prob),
        ]
    ops += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(ops)


class RSNABoneAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, cfg, train: bool = True):
        """
        df: bắt buộc có cột ['id', 'boneage', 'male'] (male: 1 nam / 0 nữ)
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = self._resolve_img_dir(img_dir, df)
        self.cfg = cfg
        self.train = train
        self.transform = build_transforms(cfg, train)

    @staticmethod
    def _resolve_img_dir(img_dir: str, df: pd.DataFrame) -> str:
        """RSNA tải từ Kaggle/nguồn gốc thường có cấu trúc lồng:
        boneage-training-dataset/boneage-training-dataset/{id}.png
        thay vì phẳng boneage-training-dataset/{id}.png.
        Tự dò để tránh FileNotFoundError hàng loạt."""
        sample_id = int(df.iloc[0]["id"])
        flat_path = os.path.join(img_dir, f"{sample_id}.png")
        if os.path.exists(flat_path):
            return img_dir
        nested = os.path.join(img_dir, os.path.basename(img_dir.rstrip("/")))
        nested_path = os.path.join(nested, f"{sample_id}.png")
        if os.path.exists(nested_path):
            print(f"[dataset] Phát hiện cấu trúc thư mục lồng, dùng: {nested}")
            return nested
        raise FileNotFoundError(
            f"Không tìm thấy ảnh mẫu {sample_id}.png ở '{flat_path}' lẫn '{nested_path}'. "
            f"Kiểm tra lại cấu trúc thư mục dataset trên Drive."
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{int(row['id'])}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

        img = foreground_crop(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # channel duplication -> RGB

        augmented = self.transform(image=img)
        image = augmented["image"].float()

        gender = torch.tensor([float(row["male"])], dtype=torch.float32)
        boneage = torch.tensor(float(row["boneage"]), dtype=torch.float32)

        return {"image": image, "gender": gender, "boneage": boneage, "id": int(row["id"])}


def load_split_csv(cfg):
    """Đọc CSV train, chuẩn hoá tên cột về ['id','boneage','male']."""
    df = pd.read_csv(os.path.join(cfg.data_root, cfg.train_csv))
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("id", "image_id", "imageid"):
            rename_map[c] = "id"
        elif cl in ("boneage", "bone_age"):
            rename_map[c] = "boneage"
        elif cl in ("male", "sex", "gender"):
            rename_map[c] = "male"
    df = df.rename(columns=rename_map)
    if df["male"].dtype == object:
        df["male"] = df["male"].map({True: 1, False: 0, "True": 1, "False": 0,
                                      "M": 1, "F": 0}).fillna(df["male"]).astype(int)
    else:
        df["male"] = df["male"].astype(int)
    return df
