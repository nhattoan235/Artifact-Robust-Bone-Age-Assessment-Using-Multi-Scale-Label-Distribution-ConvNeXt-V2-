from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config
from .data import BoneAgeDataset, load_manifest
from .model import build_model


def attempt(cfg, rows, batch_size: int, steps: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg.architecture, cfg.pretrained, cfg.sex_embedding_dim, cfg.head_hidden_dim, cfg.dropout).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    dataset = BoneAgeDataset(
        rows[: batch_size * steps], cfg.image_size, cfg.target_mean, cfg.target_std,
        preprocessing=cfg.preprocessing, preprocessed_root=cfg.preprocessed_root,
        image_root=cfg.image_root,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    bf16_conv_ok = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(0)[0] >= 8
    if cfg.amp_dtype == "bfloat16" and not bf16_conv_ok:
        raise RuntimeError("GPU hiện tại không hỗ trợ BF16 convolution an toàn")
    use_bfloat16 = cfg.amp_dtype == "bfloat16" or (cfg.amp_dtype == "auto" and bf16_conv_ok)
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16, init_scale=cfg.amp_init_scale)
    loss_fn = nn.SmoothL1Loss(beta=cfg.smooth_l1_beta_months / cfg.target_std)
    start = time.perf_counter()
    seen = 0
    try:
        for batch in loader:
            image = batch["image"].cuda(non_blocking=True)
            sex = batch["sex"].cuda(non_blocking=True)
            target = batch["target_norm"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = loss_fn(model(image, sex), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            seen += len(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {
            "batch_size": batch_size, "status": "ok", "steps": steps,
            "images_per_second": seen / elapsed,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "amp_dtype": str(amp_dtype),
        }
    except torch.cuda.OutOfMemoryError:
        return {"batch_size": batch_size, "status": "oom"}
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="p1_baseline/configs/p1_a0.toml")
    parser.add_argument("--batches", default="8,12,16")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA không khả dụng")
    cfg = load_config(args.config)
    rows = load_manifest(cfg.train_manifest, "train")
    results = [attempt(cfg, rows, int(value), args.steps) for value in args.batches.split(",")]
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
