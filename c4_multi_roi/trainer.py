from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cache_utils import sha256_file
from .config import C4Config, scientific_config_hash
from .data import (
    EpochPermutationSampler,
    MultiViewBoneAgeDataset,
    load_manifest,
    split_records,
    worker_seed,
)
from .model import build_c4_model


def set_seed(seed: int, deterministic: bool, allow_tf32: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def _atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    torch.load(temporary, map_location="cpu", weights_only=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def current_code_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verified_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.copying")
    shutil.copyfile(source, temporary)
    if sha256_file(source) != sha256_file(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"mirror temporary verification failed: {destination}")
    # Google Drive FUSE has previously failed atomic rename of temporary files.
    # A second verified copy avoids relying on rename while retaining a complete
    # temporary source for the duration of the destination write.
    shutil.copyfile(temporary, destination)
    if sha256_file(source) != sha256_file(destination):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"mirror destination verification failed: {destination}")
    temporary.unlink(missing_ok=True)


class C4Trainer:
    def __init__(self, config: C4Config, *, model: nn.Module | None = None):
        self.config = config
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._logger()
        set_seed(config.seed, config.deterministic, config.allow_tf32)
        self.config_hash = scientific_config_hash(config)
        self.code_hash = current_code_hash()
        self.manifest_path = Path(config.manifest)
        self.manifest_hash = sha256_file(self.manifest_path)
        if config.expected_manifest_sha256 and self.manifest_hash != config.expected_manifest_sha256:
            raise RuntimeError("C4 manifest SHA-256 does not match config")
        rows = load_manifest(self.manifest_path)
        if len(rows) != config.expected_rows:
            raise RuntimeError(f"C4 manifest row count {len(rows)} != {config.expected_rows}")
        self.train_rows, self.val_rows = split_records(rows, validation_fold=config.validation_fold)
        self.device = self._device()
        self.model = model or build_c4_model(
            pretrained=config.pretrained,
            view_mode=config.view_mode,
            sex_embedding_dim=config.sex_embedding_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.min_learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss(beta=config.smooth_l1_beta_months / config.target_std)
        self.amp_enabled = config.amp and self.device.type == "cuda"
        use_bfloat16 = (
            self.amp_enabled
            and config.amp_dtype in {"auto", "bfloat16"}
            and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16,
            init_scale=config.amp_init_scale,
        )
        self.epoch = 0
        self.samples_seen_epoch = 0
        self.global_step = 0
        self.best_mae = math.inf
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.last_checkpoint_clock = time.perf_counter()
        self._write_metadata()

    def _logger(self) -> logging.Logger:
        logger = logging.getLogger(str(self.run_dir.resolve()))
        logger.setLevel(logging.INFO)
        for existing in logger.handlers:
            existing.close()
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(self.run_dir / "train.log", encoding="utf-8"),
        ):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    def _device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _write_metadata(self) -> None:
        _atomic_json(self.run_dir / "config_resolved.json", asdict(self.config))
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp_enabled": self.amp_enabled,
            "amp_dtype_resolved": str(self.amp_dtype),
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "manifest_hash": self.manifest_hash,
        }
        _atomic_json(self.run_dir / "environment.json", environment)
        self._mirror_small_files()

    def _mirror_path(self, source: Path) -> Path | None:
        return self.config.mirror_dir / source.name if self.config.mirror_dir else None

    def _mirror_file(self, source: Path) -> None:
        destination = self._mirror_path(source)
        if destination is None or not source.is_file():
            return
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                verified_copy(source, destination)
                return
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "mirror attempt %d/3 failed for %s: %s", attempt, destination, exc
                )
                time.sleep(attempt)
        # Keep training and the local checkpoint alive instead of crashing a
        # multi-hour run because Drive FUSE is briefly unavailable.
        self.logger.error("mirror deferred for %s after retries: %s", destination, last_error)

    def _mirror_small_files(self) -> None:
        for name in ("config_resolved.json", "environment.json", "run_state.json", "metrics.jsonl", "val_predictions_best.csv", "train.log"):
            self._mirror_file(self.run_dir / name)

    def _checkpoint_payload(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "samples_seen_epoch": self.samples_seen_epoch,
            "global_step": self.global_step,
            "best_mae": self.best_mae,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "manifest_hash": self.manifest_hash,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    def save_checkpoint(self, name: str) -> Path:
        path = self.run_dir / name
        _atomic_torch_save(path, self._checkpoint_payload())
        self._mirror_file(path)
        self.last_checkpoint_clock = time.perf_counter()
        return path

    def resume(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        for key, expected in (
            ("config_hash", self.config_hash),
            ("code_hash", self.code_hash),
            ("manifest_hash", self.manifest_hash),
        ):
            if state.get(key) != expected:
                raise RuntimeError(f"resume rejected: {key} mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch = int(state["epoch"])
        self.samples_seen_epoch = int(state["samples_seen_epoch"])
        self.global_step = int(state["global_step"])
        self.best_mae = float(state["best_mae"])
        self.best_epoch = int(state["best_epoch"])
        self.epochs_without_improvement = int(state["epochs_without_improvement"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if state["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in state["rng"]["cuda"]])
        self.logger.info(
            "RESUME epoch=%d samples_seen=%d global_step=%d best_mae=%.6f",
            self.epoch, self.samples_seen_epoch, self.global_step, self.best_mae,
        )

    def _dataset(self, rows: list[dict[str, str]], *, train: bool) -> MultiViewBoneAgeDataset:
        return MultiViewBoneAgeDataset(
            rows,
            image_root=self.config.image_root,
            image_size=self.config.image_size,
            target_mean=self.config.target_mean,
            target_std=self.config.target_std,
            view_mode=self.config.view_mode,
            train=train,
            epoch=self.epoch,
            seed=self.config.seed,
            augmentation=self.config.augmentation if train else "none",
        )

    def _train_loader(self) -> DataLoader:
        dataset = self._dataset(self.train_rows, train=True)
        sampler = EpochPermutationSampler(
            len(self.train_rows), self.config.seed, self.epoch, self.samples_seen_epoch
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=worker_seed,
            persistent_workers=self.config.num_workers > 0,
        )

    def train_epoch(self, *, interrupt_after_global_step: int | None = None) -> tuple[float, bool]:
        self.model.train()
        loader = self._train_loader()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        loss_count = 0
        accumulated = 0
        for batch in loader:
            views = batch["views"].to(self.device, non_blocking=True)
            sex = batch["sex"].to(self.device, non_blocking=True)
            target = batch["target_norm"].to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                prediction = self.model(views, sex)
                raw_loss = self.loss_fn(prediction, target)
                loss = raw_loss / self.config.grad_accum_steps
            if not torch.isfinite(raw_loss):
                raise RuntimeError("non-finite C4 training loss")
            self.scaler.scale(loss).backward()
            total_loss += float(raw_loss.detach())
            loss_count += 1
            accumulated += 1
            self.samples_seen_epoch += int(views.shape[0])
            is_last = self.samples_seen_epoch >= len(self.train_rows)
            if accumulated == self.config.grad_accum_steps or is_last:
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                self.global_step += 1
                if self.global_step % self.config.log_every_steps == 0:
                    self.logger.info(
                        "epoch=%d global_step=%d samples=%d/%d loss=%.5f grad_norm=%.4f",
                        self.epoch + 1, self.global_step, self.samples_seen_epoch,
                        len(self.train_rows), float(raw_loss.detach()), grad_norm,
                    )
                checkpoint_due = self.global_step % self.config.checkpoint_every_steps == 0
                checkpoint_due = checkpoint_due or (
                    time.perf_counter() - self.last_checkpoint_clock >= self.config.checkpoint_every_minutes * 60
                )
                if checkpoint_due:
                    self.save_checkpoint("last.ckpt")
                if interrupt_after_global_step is not None and self.global_step >= interrupt_after_global_step:
                    self.save_checkpoint("last.ckpt")
                    return total_loss / max(loss_count, 1), True
        return total_loss / max(loss_count, 1), False

    def validate(self) -> tuple[dict, list[dict]]:
        self.model.eval()
        loader = DataLoader(
            self._dataset(self.val_rows, train=False),
            batch_size=max(1, self.config.batch_size),
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=worker_seed,
        )
        records: list[dict] = []
        with torch.inference_mode():
            for batch in loader:
                views = batch["views"].to(self.device, non_blocking=True)
                sex = batch["sex"].to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    prediction_norm = self.model(views, sex)
                predictions = prediction_norm.float().cpu() * self.config.target_std + self.config.target_mean
                for image_id, sex_text, target, prediction in zip(
                    batch["image_id"], batch["sex_text"], batch["target_months"], predictions
                ):
                    records.append({
                        "image_id": str(image_id),
                        "sex": str(sex_text),
                        "target_months": float(target),
                        "prediction_months": float(prediction),
                        "absolute_error": abs(float(prediction) - float(target)),
                        "fold": self.config.validation_fold,
                    })
        errors = np.asarray([row["prediction_months"] - row["target_months"] for row in records])
        absolute = np.abs(errors)
        metrics = {
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "median_ae": float(np.median(absolute)),
            "count": len(records),
        }
        return metrics, records

    def _write_predictions(self, path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def _append_metrics(self, value: dict) -> None:
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    def _write_state(self, status: str, metrics: dict | None = None) -> None:
        state = {
            "status": status,
            "run_id": self.config.run_id,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "best_mae": self.best_mae,
            "last_metrics": metrics,
            "last_checkpoint": str(self.run_dir / "last.ckpt"),
            "best_checkpoint": str(self.run_dir / "best_mae.ckpt"),
        }
        _atomic_json(self.run_dir / "run_state.json", state)
        self._mirror_small_files()

    def fit(
        self,
        *,
        resume: str | Path | None = None,
        interrupt_after_global_step: int | None = None,
    ) -> str:
        if resume is not None:
            self.resume(resume)
        self.logger.info(
            "Run=%s device=%s fold=%d view_mode=%s config_hash=%s",
            self.config.run_id, self.device, self.config.validation_fold,
            self.config.view_mode, self.config_hash,
        )
        while self.epoch < self.config.epochs:
            train_loss, interrupted = self.train_epoch(interrupt_after_global_step=interrupt_after_global_step)
            if interrupted:
                self._write_state("interrupted")
                return "interrupted"
            metrics, records = self.validate()
            current_epoch = self.epoch + 1
            self.scheduler.step()
            improved = metrics["mae"] < self.best_mae - self.config.min_delta_mae
            if improved:
                self.best_mae = metrics["mae"]
                self.best_epoch = current_epoch
                self.epochs_without_improvement = 0
                self._write_predictions(self.run_dir / "val_predictions_best.csv", records)
            else:
                self.epochs_without_improvement += 1
            self.epoch = current_epoch
            self.samples_seen_epoch = 0
            event = {
                "epoch": current_epoch,
                "global_step": self.global_step,
                "train_loss": train_loss,
                **metrics,
                "best_mae": self.best_mae,
                "best_epoch": self.best_epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            self._append_metrics(event)
            self.save_checkpoint("last.ckpt")
            if improved:
                self.save_checkpoint("best_mae.ckpt")
            self._write_state("running", metrics)
            self.logger.info(
                "epoch=%d train_loss=%.5f val_mae=%.5f best_mae=%.5f",
                current_epoch, train_loss, metrics["mae"], self.best_mae,
            )
            if self.epochs_without_improvement >= self.config.patience:
                self._write_state("early_stopped", metrics)
                return "early_stopped"
        self._write_state("completed")
        return "completed"
