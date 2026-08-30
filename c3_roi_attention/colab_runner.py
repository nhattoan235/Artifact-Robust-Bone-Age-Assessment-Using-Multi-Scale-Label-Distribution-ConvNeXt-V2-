"""One-command Colab preflight, fresh train, and safe resume for one fold."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .prepare_configs import ROOT, generate_colab_config


def build_train_command(
    python: str, config: Path, resume: Path | None,
) -> list[str]:
    command = [python, "-m", "p1_baseline.train", "--config", str(config)]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--drive-data-root", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    config = generate_colab_config(args.fold, args.drive_data_root)
    config_relative = config.relative_to(ROOT)
    if not args.skip_preflight:
        subprocess.run(
            [sys.executable, "-m", "p1_baseline.preflight", "--config", str(config_relative)],
            cwd=ROOT,
            check=True,
        )
    if args.preflight_only:
        print(f"PREFLIGHT_ONLY_PASS fold={args.fold} config={config_relative.as_posix()}")
        return 0

    run_id = f"C3_ROI_ATTN_V1_FOLD_{args.fold}"
    checkpoint = Path(args.drive_data_root) / "c3_attention_runs" / run_id / "last.ckpt"
    resume = checkpoint if checkpoint.is_file() else None
    print("RESUME" if resume else "FRESH", run_id, flush=True)
    subprocess.run(
        build_train_command(sys.executable, config_relative, resume), cwd=ROOT, check=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
