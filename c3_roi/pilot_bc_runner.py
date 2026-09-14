"""Safe Colab runner for the locked Fold-1 artifact pilots B and C."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1_baseline.config import Config, load_config, scientific_config_hash  # noqa: E402
from p1_baseline.data import load_manifest, manifest_hash  # noqa: E402
from p1_baseline.trainer import Trainer  # noqa: E402


DEFAULT_PILOT_ROOT = Path("/content/C3_Z26_C3_ROI_V2_PILOTS")
TERMINAL_STATUSES = {"completed", "early_stopped"}


def pilot_config_path(
    pilot: str, pilot_root: Path = DEFAULT_PILOT_ROOT, fold: int = 1,
) -> Path:
    name = pilot.upper()
    if name not in {"B", "C"}:
        raise ValueError("pilot phải là B hoặc C")
    if fold not in {1, 2, 3, 4, 5}:
        raise ValueError("fold phải nằm trong 1..5")
    if name == "B" and fold != 1:
        raise ValueError("Pilot B chỉ được khóa ở Fold 1; Pilot C mới chạy mở rộng 5-fold")
    return pilot_root / "configs" / f"C3_Z26_C3_ROI_V2_PILOT_{name}_FOLD_{fold}_SEED_42.toml"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def terminal_run_status(run_dir: Path) -> str | None:
    state = _read_json(run_dir / "run_state.json")
    if state is None:
        return None
    status = state.get("status")
    if status in TERMINAL_STATUSES:
        return str(status)
    if status == "stopped_instability":
        raise RuntimeError(f"Run đã dừng do bất ổn; cần kiểm tra thủ công: {run_dir}")
    return None


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _checkpoint_candidates(cfg: Config) -> list[Path]:
    directories = [cfg.run_dir]
    if cfg.checkpoint_mirror_root:
        directories.append(Path(cfg.checkpoint_mirror_root) / cfg.run_id)
    candidates: list[Path] = []
    for directory in directories:
        candidates.append(directory / "last.ckpt")
        candidates.extend(sorted((directory / "periodic").glob("*.ckpt"), reverse=True))
    return candidates


def _checkpoint_compatible(
    state: dict[str, Any], cfg: Config, *, train_hash: str, val_hash: str,
    code_version: str,
) -> bool:
    return (
        state.get("config_hash") == scientific_config_hash(cfg)
        and state.get("train_manifest_hash") == train_hash
        and state.get("val_manifest_hash") == val_hash
        and state.get("code_version") == code_version
    )


def select_resume_checkpoint(
    cfg: Config, *, train_hash: str, val_hash: str, code_version: str,
) -> Path | None:
    compatible: list[tuple[tuple[int, int, int], Path]] = []
    for path in _checkpoint_candidates(cfg):
        if not path.is_file():
            continue
        state = _load_checkpoint(path)
        if state is None or not _checkpoint_compatible(
            state, cfg, train_hash=train_hash, val_hash=val_hash,
            code_version=code_version,
        ):
            continue
        progress = (
            int(state.get("epoch", 0)),
            int(state.get("samples_seen_in_epoch", 0)),
            int(state.get("global_step", 0)),
        )
        compatible.append((progress, path))
    return max(compatible, default=(None, None), key=lambda item: item[0])[1]


def _validate_manifests(cfg: Config) -> tuple[str, str]:
    train_rows = load_manifest(cfg.train_manifest, "train")
    val_rows = load_manifest(cfg.val_manifest, "validation_official")
    train_hash = manifest_hash(train_rows)
    val_hash = manifest_hash(val_rows)
    checks = {
        "train_count": len(train_rows) == cfg.expected_train_count,
        "val_count": len(val_rows) == cfg.expected_val_count,
        "train_hash": train_hash == cfg.expected_train_hash,
        "val_hash": val_hash == cfg.expected_val_hash,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Manifest preflight FAIL cho {cfg.run_id}: {checks}")
    return train_hash, val_hash


def _restore_small_artifacts(cfg: Config) -> None:
    if not cfg.checkpoint_mirror_root:
        return
    source = Path(cfg.checkpoint_mirror_root) / cfg.run_id
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_state.json", "config_resolved.yaml", "environment.txt", "train.log",
        "warnings.log", "metrics.jsonl", "val_predictions_best.csv",
    ):
        path = source / name
        if path.is_file() and not (cfg.run_dir / name).exists():
            shutil.copy2(path, cfg.run_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pilot B or C safely on a locked fold")
    parser.add_argument("--pilot", required=True, choices=("B", "C", "b", "c"))
    parser.add_argument("--fold", type=int, default=1, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    args = parser.parse_args()

    config_path = pilot_config_path(args.pilot, args.pilot_root, args.fold)
    if not config_path.is_file():
        raise FileNotFoundError(f"Thiếu config: {config_path}")
    cfg = load_config(config_path)
    mirror_dir = Path(cfg.checkpoint_mirror_root) / cfg.run_id
    status = terminal_run_status(mirror_dir)
    if status:
        print(f"SKIP {cfg.run_id}: trạng thái cuối {status}", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU chưa bật. Chọn Runtime > Change runtime type > T4 GPU.")

    train_hash, val_hash = _validate_manifests(cfg)
    code_version = Trainer._code_version()
    resume = select_resume_checkpoint(
        cfg, train_hash=train_hash, val_hash=val_hash, code_version=code_version
    )
    existing = [path for path in _checkpoint_candidates(cfg) if path.is_file()]
    if resume is None and existing:
        raise RuntimeError(
            "Có checkpoint nhưng không có bản tương thích với config/code V2; "
            "runner từ chối chạy fresh để tránh ghi đè: "
            + ", ".join(str(path) for path in existing)
        )

    subprocess.run(
        [sys.executable, "-m", "p1_baseline.preflight", "--config", str(config_path),
         "--no-pretrained"],
        cwd=ROOT, check=True,
    )
    _restore_small_artifacts(cfg)
    command = [
        sys.executable, "-u", "-m", "p1_baseline.train", "--config", str(config_path)
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
        print(f"RESUME {cfg.run_id}: {resume}", flush=True)
    else:
        print(f"FRESH {cfg.run_id}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

    status = terminal_run_status(mirror_dir)
    if status is None:
        raise RuntimeError(f"Train kết thúc nhưng mirror chưa có trạng thái cuối: {mirror_dir}")
    print(f"DONE {cfg.run_id}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
