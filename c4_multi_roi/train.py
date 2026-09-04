from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .trainer import C4Trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--interrupt-after-global-step", type=int, default=None)
    args = parser.parse_args()
    trainer = C4Trainer(load_config(args.config))
    try:
        status = trainer.fit(
            resume=args.resume,
            interrupt_after_global_step=args.interrupt_after_global_step,
        )
        print(f"RUN STATUS: {status}")
        return 0
    finally:
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
