"""Small synthetic 512x512 GPU capability check for P0 (no research data)."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import timm
import torch


def benchmark(model_name: str, batch_size: int, image_size: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = {
        "model": model_name,
        "batch_size": batch_size,
        "image_size": image_size,
        "status": "unknown",
    }
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=1).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        inputs = torch.randn(batch_size, 3, image_size, image_size, device="cuda")
        targets = torch.randn(batch_size, 1, device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            predictions = model(inputs)
            loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        result.update(
            {
                "status": "ok",
                "loss": float(loss.detach().cpu()),
                "step_seconds": time.perf_counter() - started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
        del inputs, targets, predictions, loss, optimizer, scaler, model
    except torch.cuda.OutOfMemoryError as exc:
        result.update(
            {
                "status": "oom",
                "error": str(exc),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
    except Exception as exc:
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    candidates = ["convnext_tiny", "convnextv2_tiny.fcmae_ft_in22k_in1k"]
    available = set(timm.list_models())
    resolved = []
    for model in candidates:
        if model in available:
            resolved.append(model)
        elif model.startswith("convnextv2_tiny"):
            alternatives = timm.list_models("convnextv2_tiny*")
            if alternatives:
                resolved.append(alternatives[0])

    report = {
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        },
        "results": [benchmark(name, args.batch_size, args.image_size) for name in resolved],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["results"] and all(item["status"] == "ok" for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
