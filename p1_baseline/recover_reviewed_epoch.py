"""One-time, guarded recovery for an epoch stopped after validation review.

The stopped checkpoint already contains the fully trained epoch-3 weights, but the
trainer intentionally stopped before scheduler/epoch bookkeeping. This utility
finishes only that bookkeeping, updates reviewed config/code hashes and preserves
the original checkpoint as evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch

from .config import load_config, scientific_config_hash
from .model import build_model
from .trainer import Trainer, atomic_json, atomic_torch_save


OLD_CONFIG_HASH = "be7d7bdf2eccc578a92f357f10e995ce34a86763b2857a2d7a92b7d074f0de71"
EXPECTED = {
    "epoch": 2,
    "batch_in_epoch": 1051,
    "samples_seen_in_epoch": 12611,
    "epoch_loss_count": 1051,
    "global_step": 1053,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint_path = args.run_dir / "last.ckpt"
    state_path = args.run_dir / "run_state.json"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_state = json.loads(state_path.read_text(encoding="utf-8"))

    mismatches = {key: (state.get(key), value) for key, value in EXPECTED.items() if state.get(key) != value}
    if mismatches or state.get("config_hash") != OLD_CONFIG_HASH:
        raise SystemExit(f"Checkpoint không đúng trạng thái recovery đã duyệt: {mismatches}")
    if run_state.get("status") != "stopped_instability":
        raise SystemExit("Run không ở trạng thái stopped_instability")
    metrics = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(metrics) != 3 or metrics[-1]["epoch"] != 3 or not math.isclose(metrics[-1]["mae"], 7.950877192982456):
        raise SystemExit("Metric epoch 3 không khớp bằng chứng đã review")
    if metrics[-1]["mae"] <= state["best_mae"]:
        raise SystemExit("Epoch dừng lại là best mới; không được recovery tự động")

    backup_dir = args.run_dir / "recovery_evidence"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / "last_before_warning_policy_migration.ckpt"
    if backup.exists():
        raise SystemExit(f"Backup đã tồn tại, từ chối chạy recovery lần hai: {backup}")
    shutil.copy2(checkpoint_path, backup)

    # Tái tạo optimizer/scheduler đúng recipe, load state cuối epoch 3 rồi thực hiện
    # scheduler.step() duy nhất vốn chưa chạy vì dừng sau validation.
    model = build_model(cfg.architecture, False, cfg.sex_embedding_dim, cfg.head_hidden_dim, cfg.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_learning_rate)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scheduler.step()

    state["optimizer"] = optimizer.state_dict()
    state["scheduler"] = scheduler.state_dict()
    state["epoch"] = 3
    state["batch_in_epoch"] = 0
    state["samples_seen_in_epoch"] = 0
    state["epoch_loss_sum"] = 0.0
    state["epoch_loss_count"] = 0
    state["epochs_without_improvement"] = 1
    state["config_hash"] = scientific_config_hash(cfg)
    state["code_version"] = Trainer._code_version()
    atomic_torch_save(checkpoint_path, state)

    atomic_json(state_path, {
        "run_id": cfg.run_id,
        "status": "reviewed_ready_to_resume",
        "epoch": 3,
        "batch_in_epoch": 0,
        "samples_seen_in_epoch": 0,
        "global_step": state["global_step"],
        "best_mae": state["best_mae"],
        "best_epoch": state["best_epoch"],
        "epochs_without_improvement": 1,
        "last_checkpoint": str(checkpoint_path),
        "best_checkpoint": str(args.run_dir / "best_mae.ckpt"),
    })
    report = {
        "status": "PASS",
        "backup": str(backup),
        "resume_epoch_zero_based": state["epoch"],
        "resume_human_epoch": state["epoch"] + 1,
        "global_step": state["global_step"],
        "scheduler_last_epoch": state["scheduler"]["last_epoch"],
        "next_learning_rate": state["optimizer"]["param_groups"][0]["lr"],
        "new_config_hash": state["config_hash"],
        "new_code_version": state["code_version"],
    }
    (backup_dir / "recovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
