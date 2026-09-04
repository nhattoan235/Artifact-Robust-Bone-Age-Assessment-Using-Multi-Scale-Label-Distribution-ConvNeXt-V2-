from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .cache_utils import sha256_file
from .config import C4Config, load_config, scientific_config_hash
from .preflight import preflight_report
from .trainer import current_code_hash


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_compatible(
    path: Path,
    config: C4Config,
    *,
    manifest_hash: str,
    code_hash: str,
) -> bool:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        state.get("config_hash") == scientific_config_hash(config)
        and state.get("manifest_hash") == manifest_hash
        and state.get("code_hash") == code_hash
    )


def select_resume_checkpoint(
    config: C4Config,
    *,
    manifest_hash: str,
    code_hash: str,
) -> Path | None:
    candidates = [config.run_dir / "last.ckpt"]
    if config.mirror_dir is not None:
        candidates.append(config.mirror_dir / "last.ckpt")
    for path in candidates:
        if path.is_file() and checkpoint_compatible(
            path, config, manifest_hash=manifest_hash, code_hash=code_hash
        ):
            return path
    return None


def archive_incompatible_checkpoints(
    config: C4Config,
    *,
    manifest_hash: str,
    code_hash: str,
) -> list[Path]:
    archived: list[Path] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directories = [config.run_dir]
    if config.mirror_dir is not None:
        directories.append(config.mirror_dir)
    for directory in directories:
        for name in ("last.ckpt", "best_mae.ckpt"):
            path = directory / name
            if path.is_file() and not checkpoint_compatible(
                path, config, manifest_hash=manifest_hash, code_hash=code_hash
            ):
                destination = path.with_name(f"{path.stem}.incompatible.{stamp}{path.suffix}")
                path.replace(destination)
                archived.append(destination)
    return archived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--config-root", type=Path, default=ROOT / "c4_multi_roi/configs")
    # Kept for compatibility with older Colab cells. Paths are intentionally
    # resolved from the locked config, so this argument does not alter them.
    parser.add_argument("--drive-data-root", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config_root / f"fold_{args.fold}_colab.toml"
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU chưa được bật. Chọn Runtime > Change runtime type > T4 GPU.")
    report = preflight_report(config, check_all_files=True)
    if report["status"] != "PASS":
        raise RuntimeError(f"C4 preflight FAIL: {report['errors']}")
    manifest_hash = sha256_file(Path(config.manifest))
    code_hash = current_code_hash()
    resume = select_resume_checkpoint(
        config, manifest_hash=manifest_hash, code_hash=code_hash
    )
    if resume is None:
        archived = archive_incompatible_checkpoints(
            config, manifest_hash=manifest_hash, code_hash=code_hash
        )
        if archived:
            print("Đã bảo toàn checkpoint không tương thích:")
            print("\n".join(str(path) for path in archived))
        print(f"FOLD {args.fold}: chạy fresh")
    else:
        print(f"FOLD {args.fold}: resume từ {resume}")
    command = [
        sys.executable,
        "-u",
        "-m",
        "c4_multi_roi.train",
        "--config",
        str(config_path),
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
