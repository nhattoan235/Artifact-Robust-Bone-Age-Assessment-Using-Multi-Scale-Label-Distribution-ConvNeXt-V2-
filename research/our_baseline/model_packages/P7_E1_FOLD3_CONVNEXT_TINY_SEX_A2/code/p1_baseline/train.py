from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 ConvNeXt-Tiny bone-age baseline")
    parser.add_argument("--config", default="p1_baseline/configs/p1_a0.toml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--interrupt-after-global-step", type=int)
    args = parser.parse_args()
    trainer = Trainer(load_config(args.config))
    status = trainer.fit(args.resume, args.interrupt_after_global_step)
    print(f"RUN STATUS: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
