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
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config, save_resolved_yaml, scientific_config_hash
from .data import BoneAgeDataset, build_train_sampler, load_manifest, manifest_hash, worker_seed
from .metrics import compute_metrics
from .model import build_model


CHECKPOINT_KEYS = {
    "model", "optimizer", "scheduler", "scaler", "epoch", "batch_in_epoch",
    "samples_seen_in_epoch", "epoch_loss_sum", "epoch_loss_count", "global_step", "best_mae", "best_epoch",
    "epochs_without_improvement", "rng_state", "seed", "train_manifest_hash",
    "val_manifest_hash", "config_hash", "code_version",
}


def gaussian_label_distribution(
    target_months: torch.Tensor, age_class_count: int, sigma: float,
) -> torch.Tensor:
    ages = torch.arange(age_class_count, device=target_months.device, dtype=torch.float32)
    distances = ages.unsqueeze(0) - target_months.float().unsqueeze(1)
    distribution = torch.exp(-0.5 * (distances / sigma) ** 2)
    return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(1e-12)


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    missing = CHECKPOINT_KEYS - set(loaded)
    if missing:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint tạm thiếu key: {sorted(missing)}")
    os.replace(temporary, path)


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location=cuda) cũng chuyển tensor RNG CPU sang CUDA;
    # set_rng_state bắt buộc nhận ByteTensor trên CPU.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


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


class RunLogger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        # Windows PowerShell có thể mặc định cp1252; ép UTF-8 để log tiếng Việt
        # không làm phát sinh Logging error trong các run dài.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        self.logger = logging.getLogger(str(run_dir.resolve()))
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.warning_path = run_dir / "warnings.log"
        self.warning_path.touch(exist_ok=True)
        self.current_warnings: list[str] = []

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, level: str, message: str) -> None:
        line = f"STABILITY WARNING [{level}] {message}"
        self.current_warnings.append(f"{level}: {message}")
        self.logger.warning(line)
        with self.warning_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def reset_epoch_warnings(self) -> None:
        self.current_warnings.clear()

    @property
    def has_red_warning(self) -> bool:
        return any(item.startswith("ĐỎ:") for item in self.current_warnings)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.logger = RunLogger(self.run_dir)
        set_seed(cfg.seed, cfg.deterministic, cfg.allow_tf32)
        self.config_hash = scientific_config_hash(cfg)
        self.code_version = self._code_version()

        self.train_rows = load_manifest(cfg.train_manifest, "train")
        self.val_rows = load_manifest(cfg.val_manifest, "validation_official")
        if len(self.train_rows) != cfg.expected_train_count or len(self.val_rows) != cfg.expected_val_count:
            raise RuntimeError(
                "Số ảnh train/validation không khớp cấu hình: "
                f"{len(self.train_rows)}/{len(self.val_rows)} != "
                f"{cfg.expected_train_count}/{cfg.expected_val_count}"
            )
        if cfg.max_train_samples:
            self.train_rows = self.train_rows[: cfg.max_train_samples]
        if cfg.max_val_samples:
            self.val_rows = self.val_rows[: cfg.max_val_samples]
        self.train_hash = manifest_hash(load_manifest(cfg.train_manifest, "train"))
        self.val_hash = manifest_hash(load_manifest(cfg.val_manifest, "validation_official"))
        if self.train_hash != cfg.expected_train_hash or self.val_hash != cfg.expected_val_hash:
            raise RuntimeError("Fingerprint train/validation không khớp P0; từ chối train")

        self.device = self._device()
        self.model = build_model(
            cfg.architecture, cfg.pretrained, cfg.sex_embedding_dim,
            cfg.head_hidden_dim, cfg.dropout, cfg.age_class_count,
        ).to(self.device)
        optimizer_class = torch.optim.Adam if cfg.optimizer_name == "adam" else torch.optim.AdamW
        self.optimizer = optimizer_class(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        if cfg.scheduler_name == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=cfg.scheduler_factor,
                patience=cfg.scheduler_patience,
                cooldown=cfg.scheduler_cooldown,
                min_lr=cfg.scheduler_min_lr,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=cfg.min_learning_rate
            )
        self.amp_enabled = cfg.amp and self.device.type == "cuda"
        if cfg.amp_dtype not in {"auto", "float16", "bfloat16"}:
            raise ValueError("amp_dtype phải là auto, float16 hoặc bfloat16")
        bf16_conv_ok = (
            self.amp_enabled
            and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        if cfg.amp_dtype == "bfloat16" and not bf16_conv_ok:
            raise RuntimeError("GPU hiện tại không hỗ trợ BF16 convolution an toàn; hãy dùng float16")
        use_bfloat16 = self.amp_enabled and (
            cfg.amp_dtype == "bfloat16" or (cfg.amp_dtype == "auto" and bf16_conv_ok)
        )
        self.amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16,
            init_scale=cfg.amp_init_scale,
        )
        if cfg.regression_loss == "mae":
            self.loss_fn = nn.L1Loss()
        elif cfg.regression_loss == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.SmoothL1Loss(
                beta=cfg.smooth_l1_beta_months / cfg.target_std
            )
        self.last_distribution_loss = float("nan")

        self.epoch = 0
        self.batch_in_epoch = 0
        self.samples_seen_in_epoch = 0
        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0
        self.global_step = 0
        self.best_mae = math.inf
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.train_loss_history: list[float] = []
        self.val_mae_history: list[float] = []
        self.gradient_warning_streak = 0
        self.last_grad_norm = float("nan")
        self.last_checkpoint_clock = time.perf_counter()
        self._prepare_run_files()

    def _device(self) -> torch.device:
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.cfg.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Cấu hình yêu cầu CUDA nhưng PyTorch không thấy GPU")
        return device

    @staticmethod
    def _code_version() -> str:
        digest = hashlib.sha256()
        for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _prepare_run_files(self) -> None:
        save_resolved_yaml(self.cfg, self.run_dir / "config_resolved.yaml")
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp_enabled": self.amp_enabled,
            "amp_dtype_resolved": str(self.amp_dtype) if self.amp_enabled else "disabled",
            "config_hash": self.config_hash,
            "code_version": self.code_version,
            "train_manifest_hash": self.train_hash,
            "val_manifest_hash": self.val_hash,
        }
        (self.run_dir / "environment.txt").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(f"Run={self.cfg.run_id} device={self.device} config_hash={self.config_hash}")
        self.logger.info(f"Split hash train={self.train_hash} validation={self.val_hash}")

    def _loader(self, train: bool, start_index: int = 0) -> DataLoader:
        rows = self.train_rows if train else self.val_rows
        dataset = BoneAgeDataset(
            rows, self.cfg.image_size, self.cfg.target_mean, self.cfg.target_std,
            train=train, epoch=self.epoch, seed=self.cfg.seed,
            augmentation=self.cfg.augmentation if train else "none",
            horizontal_flip_probability=self.cfg.horizontal_flip_probability if train else 0.0,
            rotation_degrees=self.cfg.rotation_degrees,
            translation_fraction=self.cfg.translation_fraction,
            scale_min=self.cfg.scale_min, scale_max=self.cfg.scale_max,
            brightness_delta=self.cfg.brightness_delta,
            contrast_delta=self.cfg.contrast_delta,
            gamma_min=self.cfg.gamma_min, gamma_max=self.cfg.gamma_max,
            shear_degrees=self.cfg.shear_degrees, clahe_probability=self.cfg.clahe_probability,
            sharpen_probability=self.cfg.sharpen_probability,
            preprocessing=self.cfg.preprocessing, preprocessed_root=self.cfg.preprocessed_root,
            image_root=self.cfg.image_root, image_normalization=self.cfg.image_normalization,
        )
        sampler = (
            build_train_sampler(
                rows, self.cfg.sampling_strategy, self.cfg.seed, self.epoch, start_index
            )
            if train else None
        )
        generator = torch.Generator().manual_seed(self.cfg.seed + self.epoch)
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size, sampler=sampler, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=self.device.type == "cuda",
            persistent_workers=self.cfg.num_workers > 0, worker_init_fn=worker_seed,
            generator=generator,
        )

    def _checkpoint_payload(self) -> dict:
        return {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
            "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
            "samples_seen_in_epoch": self.samples_seen_in_epoch,
            "epoch_loss_sum": self.epoch_loss_sum, "epoch_loss_count": self.epoch_loss_count,
            "global_step": self.global_step,
            "best_mae": self.best_mae, "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "rng_state": rng_state(), "seed": self.cfg.seed,
            "train_manifest_hash": self.train_hash, "val_manifest_hash": self.val_hash,
            "config_hash": self.config_hash, "code_version": self.code_version,
        }

    def save_checkpoint(self, path: Path) -> None:
        atomic_torch_save(path, self._checkpoint_payload())
        self.last_checkpoint_clock = time.perf_counter()
        self._write_state("running")
        self._mirror_file(path)
        self._mirror_small_artifacts()

    @property
    def mirror_dir(self) -> Path | None:
        if not self.cfg.checkpoint_mirror_root:
            return None
        return Path(self.cfg.checkpoint_mirror_root) / self.cfg.run_id

    def _mirror_file(self, source: Path, relative: Path | None = None) -> None:
        mirror_dir = self.mirror_dir
        if mirror_dir is None or not source.is_file():
            return
        destination = mirror_dir / (relative or source.relative_to(self.run_dir))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
                return
            except OSError as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt < 3:
                    time.sleep(2 * attempt)
        raise RuntimeError(f"Không đồng bộ được artifact sang mirror: {destination}: {last_error}")

    def _mirror_small_artifacts(self) -> None:
        if self.mirror_dir is None:
            return
        for name in (
            "run_state.json", "config_resolved.yaml", "environment.txt", "train.log",
            "warnings.log", "metrics.jsonl", "val_predictions_best.csv",
        ):
            self._mirror_file(self.run_dir / name)

    def _write_state(self, status: str) -> None:
        state_path = self.run_dir / "run_state.json"
        atomic_json(state_path, {
            "run_id": self.cfg.run_id, "status": status, "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch, "samples_seen_in_epoch": self.samples_seen_in_epoch,
            "global_step": self.global_step, "best_mae": self.best_mae,
            "best_epoch": self.best_epoch, "epochs_without_improvement": self.epochs_without_improvement,
            "last_checkpoint": str(self.run_dir / "last.ckpt"),
            "best_checkpoint": str(self.run_dir / "best_mae.ckpt"),
        })
        self._mirror_file(state_path)

    def resume(self, path: Path) -> None:
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            self.logger.warning("ĐỎ", f"Checkpoint không đọc được: {exc}")
            raise
        missing = CHECKPOINT_KEYS - set(state)
        if missing:
            raise RuntimeError(f"Checkpoint thiếu key: {sorted(missing)}")
        split_ok = state["train_manifest_hash"] == self.train_hash and state["val_manifest_hash"] == self.val_hash
        config_ok = state["config_hash"] == self.config_hash
        code_ok = state["code_version"] == self.code_version
        print("RESUME CHECK")
        print(f"Run ID: {self.cfg.run_id}")
        print(f"Checkpoint path: {path}")
        print(f"Epoch/global step tiếp tục: {state['epoch']}/{state['global_step']}")
        print(f"Best validation MAE trước đó: {state['best_mae']}")
        print(f"Split hash khớp: {'CÓ' if split_ok else 'KHÔNG'}")
        print(f"Config hash khớp: {'CÓ' if config_ok else 'KHÔNG'}")
        print(f"Code version khớp: {'CÓ' if code_ok else 'KHÔNG'}")
        if not split_ok or not config_ok or not code_ok:
            self.logger.warning("ĐỎ", "Split/config/code hash không khớp; từ chối resume")
            raise RuntimeError("Resume bị từ chối do hash không khớp")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        for key in ("epoch", "batch_in_epoch", "samples_seen_in_epoch", "epoch_loss_sum", "epoch_loss_count", "global_step", "best_mae", "best_epoch", "epochs_without_improvement"):
            setattr(self, key, state[key])
        restore_rng_state(state["rng_state"])
        print("Optimizer/scheduler/scaler đã phục hồi: CÓ")

    def _months(self, prediction_norm: torch.Tensor) -> torch.Tensor:
        return prediction_norm * self.cfg.target_std + self.cfg.target_mean

    def train_epoch(self, interrupt_after_global_step: int | None = None) -> float:
        self.model.train()
        loader = self._loader(True, self.samples_seen_in_epoch)
        self.optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        start = time.perf_counter()
        for local_batch, batch in enumerate(loader):
            self.batch_in_epoch += 1
            images = batch["image"].to(self.device, non_blocking=True)
            sex = batch["sex"].to(self.device, non_blocking=True)
            target = batch["target_norm"].to(self.device, non_blocking=True)
            try:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    output = self.model(images, sex)
                    if isinstance(output, dict):
                        prediction = output["regression"]
                        raw_loss = self.loss_fn(prediction, target)
                        soft_target = gaussian_label_distribution(
                            batch["target_months"].to(self.device, non_blocking=True),
                            self.cfg.age_class_count,
                            self.cfg.label_distribution_sigma,
                        )
                        log_probability = torch.log_softmax(
                            output["distribution_logits"].float(), dim=1
                        )
                        distribution_loss = -(soft_target * log_probability).sum(dim=1).mean()
                        combined_loss = raw_loss + self.cfg.label_distribution_weight * distribution_loss
                        self.last_distribution_loss = float(distribution_loss.detach().cpu())
                    else:
                        prediction = output
                        raw_loss = self.loss_fn(prediction, target)
                        combined_loss = raw_loss
                        self.last_distribution_loss = float("nan")
                    loss = combined_loss / self.cfg.grad_accum_steps
                if not torch.isfinite(raw_loss):
                    self.logger.warning("ĐỎ", f"NaN/Inf loss tại epoch={self.epoch} batch={self.batch_in_epoch}")
                    self.save_checkpoint(self.run_dir / "last.ckpt")
                    raise FloatingPointError("Non-finite loss")
                self.scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                self.logger.warning("ĐỎ", f"GPU OOM epoch={self.epoch} batch={self.batch_in_epoch}")
                self.save_checkpoint(self.run_dir / "last.ckpt")
                raise
            loss_months = float(raw_loss.detach().cpu()) * self.cfg.target_std
            self.epoch_loss_sum += loss_months
            self.epoch_loss_count += 1
            accumulated += 1
            self.samples_seen_in_epoch += len(images)
            if accumulated == self.cfg.grad_accum_steps or self.samples_seen_in_epoch >= len(self.train_rows):
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_norm))
                self.last_grad_norm = grad_norm
                if not math.isfinite(grad_norm):
                    self.logger.warning("ĐỎ", f"Gradient NaN/Inf tại global_step={self.global_step}")
                    self.save_checkpoint(self.run_dir / "last.ckpt")
                    raise FloatingPointError("Non-finite gradient")
                self.gradient_warning_streak = self.gradient_warning_streak + 1 if grad_norm > self.cfg.warning_gradient_norm else 0
                if self.gradient_warning_streak >= self.cfg.warning_gradient_consecutive:
                    self.logger.warning("CAM", f"Gradient norm cao liên tiếp: {grad_norm:.3f}")
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                self.global_step += 1
                checkpoint_due = self.global_step % self.cfg.checkpoint_every_steps == 0
                checkpoint_due = checkpoint_due or (time.perf_counter() - self.last_checkpoint_clock >= self.cfg.checkpoint_every_minutes * 60)
                if checkpoint_due:
                    self.save_checkpoint(self.run_dir / "last.ckpt")
                if interrupt_after_global_step is not None and self.global_step >= interrupt_after_global_step:
                    self.save_checkpoint(self.run_dir / "last.ckpt")
                    self.logger.info("Dừng có chủ ý sau checkpoint để kiểm thử resume")
                    return self.epoch_loss_sum / self.epoch_loss_count
            if self.batch_in_epoch % self.cfg.log_every_steps == 0:
                elapsed = max(time.perf_counter() - start, 1e-6)
                throughput = self.samples_seen_in_epoch / elapsed
                allocated = torch.cuda.memory_allocated() / 2**20 if self.device.type == "cuda" else 0.0
                reserved = torch.cuda.memory_reserved() / 2**20 if self.device.type == "cuda" else 0.0
                remaining_images = max(0, len(self.train_rows) - self.samples_seen_in_epoch)
                eta_seconds = remaining_images / max(throughput, 1e-6)
                ldl_text = f" loss_ldl={self.last_distribution_loss:.4f}" if math.isfinite(self.last_distribution_loss) else ""
                self.logger.info(f"epoch={self.epoch + 1} batch={self.batch_in_epoch} global_step={self.global_step} loss_reg_months={self.epoch_loss_sum / self.epoch_loss_count:.4f}{ldl_text} lr={self.optimizer.param_groups[0]['lr']:.8f} grad_norm={self.last_grad_norm:.4f} throughput={throughput:.2f} img/s eta_seconds={eta_seconds:.1f} gpu_alloc={allocated:.1f}MiB gpu_reserved={reserved:.1f}MiB skipped_batches=0")
        return self.epoch_loss_sum / self.epoch_loss_count

    @torch.inference_mode()
    def validate(self) -> tuple[float, dict, list[dict]]:
        self.model.eval()
        records = []
        losses = []
        for batch in self._loader(False):
            images = batch["image"].to(self.device, non_blocking=True)
            sex = batch["sex"].to(self.device, non_blocking=True)
            target_norm = batch["target_norm"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                output = self.model(images, sex)
                prediction_norm = output["regression"] if isinstance(output, dict) else output
                loss = self.loss_fn(prediction_norm, target_norm)
            regression_months = self._months(prediction_norm).float()
            if isinstance(output, dict):
                probabilities = torch.softmax(output["distribution_logits"].float(), dim=1)
                age_axis = torch.arange(
                    self.cfg.age_class_count, device=self.device, dtype=torch.float32
                )
                distribution_months = probabilities @ age_axis
                predictions_tensor = (
                    self.cfg.regression_inference_weight * regression_months
                    + (1.0 - self.cfg.regression_inference_weight) * distribution_months
                )
            else:
                distribution_months = None
                predictions_tensor = regression_months
            predictions = predictions_tensor.cpu().tolist()
            regression_values = regression_months.cpu().tolist()
            distribution_values = distribution_months.cpu().tolist() if distribution_months is not None else None
            targets = batch["target_months"].tolist()
            losses.append(float(loss.cpu()) * self.cfg.target_std)
            for item_index, (image_id, sex_text, target, prediction) in enumerate(zip(batch["image_id"], batch["sex_text"], targets, predictions)):
                record = {"image_id": image_id, "target_months": float(target), "prediction_months": float(prediction), "absolute_error": abs(float(prediction) - float(target)), "sex": sex_text}
                if distribution_values is not None:
                    record["regression_months"] = float(regression_values[item_index])
                    record["distribution_months"] = float(distribution_values[item_index])
                records.append(record)
        metrics = compute_metrics(records, self.cfg.age_bins)
        metrics["loss_months"] = statistics.fmean(losses)
        if not all(math.isfinite(float(v)) for v in (metrics["mae"], metrics["rmse"], metrics["prediction_std"])):
            self.logger.warning("ĐỎ", "Validation metric chứa NaN/Inf")
            raise FloatingPointError("Non-finite validation metric")
        if metrics["prediction_std"] < self.cfg.warning_prediction_std_min:
            self.logger.warning("ĐỎ", f"Prediction gần hằng số: std={metrics['prediction_std']:.3f}")
        for sex_name, sex_std in metrics["prediction_std_by_sex"].items():
            if sex_std < self.cfg.warning_prediction_std_min:
                self.logger.warning(
                    "ĐỎ", f"Prediction collapse trong nhóm giới {sex_name}: std={sex_std:.3f}"
                )
        prediction_min = metrics["prediction_min"]
        prediction_max = metrics["prediction_max"]
        if prediction_min < self.cfg.warning_prediction_hard_min or prediction_max > self.cfg.warning_prediction_hard_max:
            self.logger.warning("ĐỎ", f"Prediction lệch nghiêm trọng: [{prediction_min:.2f}, {prediction_max:.2f}]")
        elif prediction_min < self.cfg.warning_prediction_soft_min or prediction_max > self.cfg.warning_prediction_soft_max:
            self.logger.warning("CAM", f"Prediction vượt nhẹ miền tuổi 0–228: [{prediction_min:.2f}, {prediction_max:.2f}]")
        sex_values = metrics["mae_by_sex"]
        if len(sex_values) == 2 and abs(sex_values["M"] - sex_values["F"]) > self.cfg.warning_sex_mae_gap:
            self.logger.warning("CAM", f"Khoảng cách MAE nam-nữ={abs(sex_values['M'] - sex_values['F']):.3f} tháng")
        return metrics["loss_months"], metrics, records

    def _write_predictions(self, records: list[dict]) -> None:
        path = self.run_dir / "val_predictions_best.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def _append_metrics(self, row: dict) -> None:
        path = self.run_dir / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _snapshot(self, train_loss: float, metrics: dict, warnings: str) -> None:
        peak = torch.cuda.max_memory_reserved() / 2**20 if self.device.type == "cuda" else 0.0
        suggestion = "TẠM DỪNG KIỂM TRA" if "ĐỎ" in warnings else ("THEO DÕI" if warnings else "TIẾP TỤC")
        block = f"""TRAINING REVIEW SNAPSHOT
Run ID / config hash / split hash: {self.cfg.run_id} / {self.config_hash[:12]} / {self.train_hash[:12]}+{self.val_hash[:12]}
Epoch / global step: {self.epoch + 1} / {self.global_step}
Train loss: {train_loss:.4f}
Validation MAE / RMSE / median AE: {metrics['mae']:.4f} / {metrics['rmse']:.4f} / {metrics['median_ae']:.4f}
MAE nam / nữ / age bins: {metrics['mae_by_sex']} / {metrics['mae_by_age_bin']}
Learning rate / peak VRAM: {self.optimizer.param_groups[0]['lr']:.8f} / {peak:.1f} MiB
Best epoch / best MAE / epochs không cải thiện: {self.best_epoch + 1} / {self.best_mae:.4f} / {self.epochs_without_improvement}
Cảnh báo hiện tại: {warnings or 'KHÔNG'}
Checkpoint gần nhất: {self.run_dir / 'last.ckpt'}
Checkpoint tốt nhất: {self.run_dir / 'best_mae.ckpt'}
Đề xuất tự động: {suggestion}"""
        self.logger.info("\n" + block)

    def fit(self, resume_path: Path | None = None, interrupt_after_global_step: int | None = None) -> str:
        if resume_path:
            self.resume(resume_path)
        status = "completed"
        while self.epoch < self.cfg.epochs:
            self.logger.reset_epoch_warnings()
            epoch_wall_start = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            train_loss = self.train_epoch(interrupt_after_global_step)
            if interrupt_after_global_step is not None and self.global_step >= interrupt_after_global_step and self.samples_seen_in_epoch < len(self.train_rows):
                self._write_state("interrupted_for_resume_test")
                return "interrupted_for_resume_test"
            _, metrics, records = self.validate()
            peak_vram = torch.cuda.max_memory_reserved() / 2**20 if self.device.type == "cuda" else 0.0
            row = {"epoch": self.epoch + 1, "global_step": self.global_step, "train_loss_months": train_loss, "learning_rate": self.optimizer.param_groups[0]["lr"], "last_gradient_norm": self.last_grad_norm, "epoch_seconds": time.perf_counter() - epoch_wall_start, "peak_vram_mib": peak_vram, "skipped_batches": 0, "best_mae": self.best_mae, "best_epoch": self.best_epoch + 1, "epochs_without_improvement": self.epochs_without_improvement, **metrics}
            if self.logger.has_red_warning:
                self._append_metrics(row)
                self.save_checkpoint(self.run_dir / "last.ckpt")
                self._snapshot(train_loss, metrics, " | ".join(self.logger.current_warnings))
                self._write_state("stopped_instability")
                return "stopped_instability"
            improved = metrics["mae"] < self.best_mae - self.cfg.min_delta_mae
            if improved:
                self.best_mae, self.best_epoch, self.epochs_without_improvement = metrics["mae"], self.epoch, 0
                self._write_predictions(records)
                self.save_checkpoint(self.run_dir / "best_mae.ckpt")
                best_copy = self.run_dir / "best" / f"epoch_{self.epoch + 1:03d}_mae_{self.best_mae:.4f}.ckpt"
                best_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.run_dir / "best_mae.ckpt", best_copy)
                old = sorted(best_copy.parent.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)[self.cfg.best_keep:]
                for item in old: item.unlink()
            else:
                self.epochs_without_improvement += 1
            row.update({"best_mae": self.best_mae, "best_epoch": self.best_epoch + 1, "epochs_without_improvement": self.epochs_without_improvement})
            self._append_metrics(row)
            if self.epochs_without_improvement >= self.cfg.warning_overfit_epochs and len(self.train_loss_history) >= self.cfg.warning_overfit_epochs:
                if train_loss < min(self.train_loss_history[-self.cfg.warning_overfit_epochs:]) and metrics["mae"] > min(self.val_mae_history[-self.cfg.warning_overfit_epochs:] or [math.inf]):
                    self.logger.warning("CAM", "Validation MAE xấu đi trong khi train loss giảm")
            self._snapshot(train_loss, metrics, " | ".join(self.logger.current_warnings))
            self.train_loss_history.append(train_loss)
            self.val_mae_history.append(metrics["mae"])
            if self.cfg.scheduler_name == "reduce_on_plateau":
                self.scheduler.step(metrics["mae"])
            else:
                self.scheduler.step()
            self.epoch += 1
            self.batch_in_epoch = 0
            self.samples_seen_in_epoch = 0
            self.epoch_loss_sum = 0.0
            self.epoch_loss_count = 0
            self.save_checkpoint(self.run_dir / "last.ckpt")
            periodic = self.run_dir / "periodic" / f"epoch_{self.epoch:03d}.ckpt"
            periodic.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.run_dir / "last.ckpt", periodic)
            for item in sorted(periodic.parent.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)[self.cfg.periodic_keep:]: item.unlink()
            self._mirror_file(periodic)
            if self.mirror_dir is not None:
                mirror_periodic = self.mirror_dir / "periodic"
                for item in sorted(mirror_periodic.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)[self.cfg.periodic_keep:]:
                    item.unlink()
                self._mirror_small_artifacts()
            if self.epochs_without_improvement >= self.cfg.patience:
                self.logger.warning("CAM", f"Early stopping patience={self.cfg.patience}")
                status = "early_stopped"
                break
        self._write_state(status)
        self._mirror_small_artifacts()
        return status
