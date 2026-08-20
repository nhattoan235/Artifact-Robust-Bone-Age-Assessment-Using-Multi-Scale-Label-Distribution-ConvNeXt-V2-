"""Guarded code-hash migration for the reviewed CPU RNG restore fix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config, scientific_config_hash
from .trainer import Trainer, atomic_torch_save


EXPECTED_OLD_CODE = "e52ebd6dfbda67cbb4d419dc355e4837037f273ffb83269edc7f1fe1cc097d5d"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "epoch": 3, "batch_in_epoch": 0, "samples_seen_in_epoch": 0,
        "epoch_loss_count": 0, "global_step": 1053,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise SystemExit("Checkpoint không đúng trạng thái sau recovery epoch 3")
    if state.get("code_version") != EXPECTED_OLD_CODE:
        raise SystemExit("Code hash cũ không khớp migration đã duyệt")
    if state.get("config_hash") != scientific_config_hash(cfg):
        raise SystemExit("Config hash không khớp")
    if not all(torch.isfinite(value).all().item() for value in state["model"].values()):
        raise SystemExit("Model chứa tensor không hữu hạn")
    old = state["code_version"]
    state["code_version"] = Trainer._code_version()
    atomic_torch_save(args.checkpoint, state)
    report = {"status": "PASS", "old_code_version": old, "new_code_version": state["code_version"], "fix": "CPU RNG tensor restored on CPU"}
    report_path = args.checkpoint.parent / "recovery_evidence" / "rng_restore_code_migration.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
