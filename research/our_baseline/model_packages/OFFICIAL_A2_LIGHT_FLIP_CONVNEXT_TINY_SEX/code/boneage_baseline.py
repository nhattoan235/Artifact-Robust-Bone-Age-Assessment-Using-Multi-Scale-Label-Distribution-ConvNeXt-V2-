"""Reproducible RSNA bone-age baseline.

This file intentionally avoids the RSNA test labels.  It supports an official
train/validation run and a 5-fold development OOF run.  The default model is
ConvNeXt-Tiny with a sex embedding and direct month regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


@dataclass
class Config:
    recipe: str = "p7_reference"
    seed: int = 42
    img_size: int = 512
    batch_size: int = 8
    epochs: int = 100
    patience: int = 15
    min_delta: float = 0.01
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    accum_steps: int = 1
    folds: int = 5
    workers: int = 2
    pretrained: bool = True
    beta: float = 1.0
    amp: str = "auto"
    device: str = "auto"
    max_train_samples: int = 0
    max_val_samples: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "official", "oof", "infer_test"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("project/baseline_v1/outputs"))
    p.add_argument("--run-name", default="run")
    p.add_argument(
        "--recipe",
        choices=["p7_reference", "a2_light_flip", "bram_lite"],
        default="p7_reference",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument(
        "--amp",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="CUDA precision: auto, fp16, bf16, or fp32",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Thiết bị chạy: auto tự chọn CUDA nếu có; cuda bắt buộc GPU; cpu bắt buộc CPU",
    )
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> torch.device:
    """Resolve the runtime device and fail loudly when CUDA was requested but unavailable."""
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "Đã yêu cầu --device cuda nhưng PyTorch không nhìn thấy GPU. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"cuda_available={cuda_available}. Hãy bật GPU runtime rồi khởi động lại session."
        )
    if requested == "cuda" or (requested == "auto" and cuda_available):
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    print(
        f"device={device} torch={torch.__version__} "
        f"torch_cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}"
    )
    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / (1024 ** 3)
        print(
            f"gpu_index={index} gpu_name={props.name} gpu_memory={total_gb:.1f}GB "
            f"compute_capability={props.major}.{props.minor}"
        )
        # These settings improve ConvNeXt throughput on Colab NVIDIA GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_frame(df: pd.DataFrame, has_target: bool = True) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"id", "image id", "case id"}:
            mapping[col] = "id"
        elif key in {"boneage", "bone age", "bone age (months)"}:
            mapping[col] = "boneage"
        elif key in {"male", "sex", "gender"}:
            mapping[col] = "male"
    out = df.rename(columns=mapping).copy()
    required = {"id", "male"} | ({"boneage"} if has_target else set())
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột {sorted(missing)}; cột hiện có={list(df.columns)}")
    out["id"] = out["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["male"] = out["male"].map(parse_male)
    if has_target:
        out["boneage"] = pd.to_numeric(out["boneage"], errors="raise").astype(float)
    return out[[c for c in ["id", "boneage", "male"] if c in out.columns]]


def parse_male(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "male", "m", "nam"}


def find_dirs_for_ids(root: Path, ids: Iterable[str]) -> list[Path]:
    wanted = set(map(str, ids))
    found: list[Path] = []
    for p in sorted((x for x in root.rglob("*") if x.is_dir()), key=lambda x: (len(x.parts), str(x))):
        stems = {x.stem for x in p.glob("*.png")}
        if stems & wanted:
            found.append(p)
    if not found:
        raise FileNotFoundError(f"Không tìm được thư mục PNG cho ID dưới {root}")
    return found


def discover_data(data_root: Path) -> dict:
    train_csv = data_root / "boneage-training-dataset.csv"
    val_csv = data_root / "boneage-validation-dataset" / "Validation Dataset.csv"
    test_csv = data_root / "boneage-test-dataset.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy train/validation CSV dưới {data_root}")

    train_df = normalize_frame(pd.read_csv(train_csv), has_target=True)
    val_df = normalize_frame(pd.read_csv(val_csv), has_target=True)
    train_img = find_dirs_for_ids(data_root / "boneage-training-dataset", train_df.id)
    val_img = find_dirs_for_ids(data_root / "boneage-validation-dataset", val_df.id)
    result = {
        "train_csv": train_csv,
        "val_csv": val_csv,
        "train_img": train_img,
        "val_img": val_img,
        "test_csv": test_csv if test_csv.exists() else None,
        "test_img": None,
        "train_df": train_df,
        "val_df": val_df,
    }
    if test_csv.exists():
        test_df = normalize_frame(pd.read_csv(test_csv), has_target=False)
        result["test_df"] = test_df
        result["test_img"] = find_dirs_for_ids(data_root / "boneage-test-dataset", test_df.id)
    return result


def make_path_map(img_dir: Path | list[Path]) -> dict[str, Path]:
    paths = {}
    dirs = [img_dir] if isinstance(img_dir, Path) else img_dir
    for directory in dirs:
        for p in directory.glob("*.png"):
            if p.stem in paths:
                raise ValueError(f"Trùng image id {p.stem} giữa các thư mục {paths[p.stem]} và {p}")
            paths[p.stem] = p
    return paths


def foreground_crop(img: Image.Image, threshold: int = 10) -> Image.Image:
    gray = np.asarray(img.convert("L"))
    mask = gray > threshold
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def standardize_image(img: Image.Image, recipe: str, size: int) -> Image.Image:
    if recipe == "bram_lite":
        img = foreground_crop(img)
        img = ImageOps.autocontrast(img.convert("L"))
        return ImageOps.pad(img, (size, size), method=Image.Resampling.BILINEAR, color=0)
    return img.resize((size, size), Image.Resampling.BILINEAR).convert("L")


def build_transforms(cfg: Config, train: bool):
    ops = []
    if train:
        if cfg.recipe == "a2_light_flip":
            # A deliberately minimal augmentation candidate: only horizontal
            # flip, so its effect can be evaluated independently.
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            ops += [
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03),
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
    ops += [transforms.ToTensor(), transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)]
    return transforms.Compose(ops)


class BoneAgeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, img_dir: Path | list[Path], cfg: Config, train: bool):
        self.frame = frame.reset_index(drop=True).copy()
        self.path_map = make_path_map(img_dir)
        self.cfg = cfg
        self.train = train
        self.transform = build_transforms(cfg, train)
        missing = [x for x in self.frame.id if x not in self.path_map]
        if missing:
            raise FileNotFoundError(f"Thiếu {len(missing)} ảnh; ví dụ {missing[:5]} trong {img_dir}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        with Image.open(self.path_map[row.id]) as source:
            img = standardize_image(source.convert("L"), self.cfg.recipe, self.cfg.img_size)
        img = img.convert("RGB")
        img = self.transform(img)
        sex = torch.tensor([float(row.male)], dtype=torch.float32)
        target = torch.tensor(float(row.boneage), dtype=torch.float32) if "boneage" in row else torch.tensor(float("nan"))
        return img, sex, target, row.id


class ConvNeXtSex(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        feature_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        self.sex_embedding = nn.Sequential(nn.Linear(1, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim + 32),
            nn.Linear(feature_dim + 32, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(image)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x)
        x = self.backbone.classifier[1](x)
        s = self.sex_embedding(sex)
        return self.head(torch.cat([x, s], dim=1)).squeeze(1)


def make_strata(df: pd.DataFrame) -> np.ndarray:
    # Sex + 12 age bands gives balanced folds without changing targets.
    bins = pd.cut(df["boneage"], bins=np.linspace(0, 228, 13), labels=False, include_lowest=True)
    return (df["male"].astype(int).to_numpy() * 100 + bins.fillna(0).astype(int).to_numpy())


def split_folds(df: pd.DataFrame, n_splits: int, seed: int):
    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(df)), make_strata(df)))
    except Exception as exc:
        print(f"[WARN] Không dùng được StratifiedKFold ({exc}); dùng KFold.")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(df))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ae = np.abs(y - p)
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "median_ae": float(np.median(ae)),
        "acc_pm6": float((ae <= 6).mean()),
        "acc_pm12": float((ae <= 12).mean()),
        "acc_pm18": float((ae <= 18).mean()),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, seed: int = 42, n_boot: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values[i] = np.abs(y[idx] - p[idx]).mean()
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def subgroup_report(frame: pd.DataFrame) -> dict:
    report = {}
    for name, group in [("female", frame[frame.male == False]), ("male", frame[frame.male == True])]:
        if len(group):
            report[name] = metrics(group.boneage.to_numpy(), group.pred.to_numpy())
    frame = frame.copy()
    frame["age_band"] = pd.cut(frame.boneage, bins=[-1, 83, 143, 228], labels=["0-83", "84-143", "144-228"])
    for name, group in frame.groupby("age_band", observed=True):
        report[f"age_{name}"] = metrics(group.boneage.to_numpy(), group.pred.to_numpy())
    return report


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(model, optimizer, scheduler, path: Path, device: torch.device):
    if not path.exists():
        return 0, float("inf"), 0
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("epoch", 0)), float(state.get("best_mae", float("inf"))), int(state.get("bad_epochs", 0))


def make_scheduler(optimizer, cfg: Config):
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return float(epoch + 1) / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, criterion, device, cfg: Config, train: bool):
    model.train(train)
    total_loss = 0.0
    y_all, p_all = [], []
    if device.type != "cuda" or cfg.amp == "fp32":
        amp_enabled, amp_dtype = False, torch.float32
    elif cfg.amp == "fp16":
        amp_enabled, amp_dtype = True, torch.float16
    elif cfg.amp == "bf16":
        amp_enabled, amp_dtype = True, torch.bfloat16
    else:
        major, _ = torch.cuda.get_device_capability(device)
        amp_enabled = True
        amp_dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16
    optimizer.zero_grad(set_to_none=True) if train else None
    for step, (images, sexes, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        sexes = sexes.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            preds = model(images, sexes)
            loss = criterion(preds, targets)
            loss_for_backward = loss / cfg.accum_steps
        if train:
            loss_for_backward.backward()
            if (step + 1) % cfg.accum_steps == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * len(targets)
        y_all.append(targets.detach().cpu().numpy())
        p_all.append(preds.detach().float().cpu().numpy())
    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    return total_loss / len(loader.dataset), metrics(y, p), y, p


def write_manifest(data: dict, out_dir: Path, cfg: Config, mode: str) -> None:
    def frame_digest(frame: pd.DataFrame) -> str:
        cols = [c for c in ["id", "boneage", "male"] if c in frame]
        payload = frame[cols].sort_values("id").to_csv(index=False).encode()
        return hashlib.sha256(payload).hexdigest()
    manifest = {
        "mode": mode,
        "config": asdict(cfg),
        "train_csv": str(data["train_csv"]),
        "val_csv": str(data["val_csv"]),
        "train_csv_sha256": sha256_file(data["train_csv"]),
        "val_csv_sha256": sha256_file(data["val_csv"]),
        "train_count": len(data["train_df"]),
        "val_count": len(data["val_df"]),
        "train_frame_sha256": frame_digest(data["train_df"]),
        "val_frame_sha256": frame_digest(data["val_df"]),
        "test_labels_loaded": False,
    }
    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def make_loaders(train_df, val_df, data, cfg: Config):
    # OOF validation folds chứa ID từ cả official train và official validation.
    # Gộp hai index ảnh để không bỏ sót fold nào.
    all_img_dirs = list(data["train_img"]) + list(data["val_img"])
    train_ds = BoneAgeDataset(train_df, all_img_dirs, cfg, train=True)
    val_ds = BoneAgeDataset(val_df, all_img_dirs, cfg, train=False)
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds, shuffle=True, drop_last=False, **common), DataLoader(val_ds, shuffle=False, drop_last=False, **common)


def train_one_split(train_df, val_df, data, cfg: Config, out_dir: Path, label: str, device: torch.device):
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.max_train_samples:
        train_df = train_df.iloc[: cfg.max_train_samples].copy()
    if cfg.max_val_samples:
        val_df = val_df.iloc[: cfg.max_val_samples].copy()
    train_loader, val_loader = make_loaders(train_df, val_df, data, cfg)
    model = ConvNeXtSex(pretrained=cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)
    criterion = nn.SmoothL1Loss(beta=cfg.beta)
    best_mae = float("inf")
    bad_epochs = 0
    log_rows = []
    best_path = out_dir / "best.pt"
    resume_path = out_dir / "resume.pt"
    start_epoch = 0
    if resume_path.exists():
        start_epoch, best_mae, bad_epochs = load_checkpoint(model, optimizer, scheduler, resume_path, device)
        print(f"[{label}] resume từ epoch {start_epoch}, best MAE={best_mae:.4f}")
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_loss, train_m, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, cfg, train=True)
        val_loss, val_m, y, p = run_epoch(model, val_loader, optimizer, criterion, device, cfg, train=False)
        scheduler.step()
        improved = val_m["mae"] < best_mae - cfg.min_delta
        if improved:
            best_mae = val_m["mae"]
            bad_epochs = 0
            atomic_torch_save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch + 1, "best_mae": best_mae}, best_path)
        else:
            bad_epochs += 1
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "best_mae": best_mae, "bad_epochs": bad_epochs,
        }
        atomic_torch_save(state, resume_path)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_mae": train_m["mae"], "val_mae": val_m["mae"], "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
        print(f"[{label}] ep {epoch+1:03d}/{cfg.epochs} train={train_m['mae']:.3f} val={val_m['mae']:.3f} best={best_mae:.3f} bad={bad_epochs}/{cfg.patience}")
        if bad_epochs >= cfg.patience:
            break
    if not best_path.exists():
        raise RuntimeError(f"{label}: chưa tạo được best checkpoint")
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state["model"])
    _, final_m, y, p = run_epoch(model, val_loader, optimizer, criterion, device, cfg, train=False)
    ids = val_df.iloc[:len(y)]["id"].to_numpy()
    pred_df = val_df.iloc[:len(y)].copy()
    pred_df["pred"] = p
    pred_df["fold_or_split"] = label
    pred_df.to_csv(out_dir / "val_predictions.csv", index=False)
    report = {"label": label, "best_mae": best_mae, "final": final_m, "bootstrap_ci_mae": bootstrap_ci(y, p, cfg.seed), "n": len(y)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return pred_df, report


def save_oof_report(oof: pd.DataFrame, out_dir: Path, cfg: Config):
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    y, p = oof.boneage.to_numpy(), oof.pred.to_numpy()
    report = {"pooled": metrics(y, p), "bootstrap_ci_mae": bootstrap_ci(y, p, cfg.seed), "subgroups": subgroup_report(oof), "folds": {}}
    for fold, group in oof.groupby("fold_or_split"):
        report["folds"][str(fold)] = metrics(group.boneage.to_numpy(), group.pred.to_numpy())
    (out_dir / "oof_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["pooled"], indent=2))


def run_smoke(data, cfg: Config, out_dir: Path, device: torch.device):
    smoke_cfg = Config(**asdict(cfg))
    smoke_cfg.img_size = min(cfg.img_size, 64)
    smoke_cfg.batch_size = min(cfg.batch_size, 2)
    smoke_cfg.pretrained = False
    train_df = data["train_df"].head(4)
    val_df = data["val_df"].head(4)
    train_ds = BoneAgeDataset(train_df, data["train_img"], smoke_cfg, True)
    val_ds = BoneAgeDataset(val_df, data["val_img"], smoke_cfg, False)
    smoke_loader = DataLoader(train_ds, batch_size=2, pin_memory=device.type == "cuda")
    x, s, y, ids = next(iter(smoke_loader))
    model = ConvNeXtSex(pretrained=False).to(device)
    with torch.no_grad():
        pred = model(
            x.to(device, non_blocking=True),
            s.to(device, non_blocking=True),
        )
    report = {"device": str(device), "train_count": len(data["train_df"]), "val_count": len(data["val_df"]), "batch_shape": list(x.shape), "pred_shape": list(pred.shape), "ids": list(ids), "finite": bool(torch.isfinite(pred).all())}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_official(data, cfg: Config, out_dir: Path, device: torch.device):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(data, out_dir, cfg, "official")
    train_df = data["train_df"]
    val_df = data["val_df"]
    _, report = train_one_split(train_df, val_df, data, cfg, out_dir / "official", "official", device)
    (out_dir / "official_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_oof(data, cfg: Config, out_dir: Path, device: torch.device):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(data, out_dir, cfg, "oof")
    dev = pd.concat([
        data["train_df"].assign(source_split="official_train"),
        data["val_df"].assign(source_split="official_validation"),
    ], ignore_index=True)
    folds = split_folds(dev, cfg.folds, cfg.seed)
    all_preds = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        fold_dir = out_dir / f"fold_{fold}"
        train_df, val_df = dev.iloc[tr_idx].copy(), dev.iloc[va_idx].copy()
        pred_df, _ = train_one_split(train_df, val_df, data, cfg, fold_dir, f"fold_{fold}", device)
        pred_df["fold_or_split"] = fold
        all_preds.append(pred_df)
    save_oof_report(pd.concat(all_preds, ignore_index=True), out_dir, cfg)


def run_test_inference(data, cfg: Config, out_dir: Path, checkpoint_dir: Path, device: torch.device):
    if "test_df" not in data or data["test_img"] is None:
        raise FileNotFoundError("Không tìm thấy test CSV/ảnh")
    test_ds = BoneAgeDataset(data["test_df"].assign(boneage=0.0), data["test_img"], cfg, train=False)
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    preds = []
    checkpoints = sorted(checkpoint_dir.glob("fold_*/best.pt")) or sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"Không tìm thấy checkpoint dưới {checkpoint_dir}")
    for ckpt in checkpoints:
        model = ConvNeXtSex(pretrained=False).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for images, sexes, _, _ in loader:
                fold_preds.append(model(images.to(device), sexes.to(device)).cpu().numpy())
        preds.append(np.concatenate(fold_preds))
    out = data["test_df"].copy()
    out["pred"] = np.mean(np.stack(preds), axis=0)
    out["n_checkpoints"] = len(checkpoints)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "test_predictions_no_labels.csv", index=False)
    print(f"Đã lưu test predictions không nhãn: {out_dir / 'test_predictions_no_labels.csv'}")


def main():
    args = parse_args()
    cfg = Config(recipe=args.recipe, seed=args.seed, img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience, min_delta=args.min_delta, lr=args.lr, weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs, accum_steps=args.accum_steps, folds=args.folds, workers=args.workers, beta=args.beta, amp=args.amp, device=args.device, pretrained=args.pretrained, max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples)
    seed_everything(cfg.seed)
    data = discover_data(args.data_root)
    out_dir = args.output_root / args.run_name
    device = resolve_device(args.device)
    print_device_info(device)
    print(f"recipe={cfg.recipe} train={len(data['train_df'])} val={len(data['val_df'])}")
    if args.mode == "smoke":
        run_smoke(data, cfg, out_dir, device)
    elif args.mode == "official":
        run_official(data, cfg, out_dir, device)
    elif args.mode == "oof":
        run_oof(data, cfg, out_dir, device)
    else:
        if args.checkpoint_dir is None:
            raise ValueError("infer_test cần --checkpoint-dir")
        run_test_inference(data, cfg, out_dir, args.checkpoint_dir, device)


if __name__ == "__main__":
    main()
