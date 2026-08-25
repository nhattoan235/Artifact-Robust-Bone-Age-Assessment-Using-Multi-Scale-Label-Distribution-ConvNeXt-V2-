"""Train a fresh baseline without touching a deterministic holdout.

The holdout is sampled only from the original official training split. The
original official validation split is added to the new training frame. This
keeps the new holdout separate from the validation used in EXP-001/EXP-002 and
produces a prediction file that can be blended with the friend's P7 OOF file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import boneage_baseline as baseline  # noqa: E402


def holdout_score(image_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--recipe", choices=["p7_reference", "a2_light_flip", "bram_lite"], default="p7_reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--amp", choices=["auto", "fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.01 <= args.holdout_fraction <= 0.40:
        raise ValueError("holdout-fraction must be between 0.01 and 0.40")
    baseline.seed_everything(args.seed)
    device = baseline.resolve_device(args.device)
    baseline.print_device_info(device)
    data = baseline.discover_data(args.data_root)

    official_train = data["train_df"].copy()
    score = official_train["id"].map(lambda value: holdout_score(str(value), args.seed))
    holdout_mask = score < args.holdout_fraction
    holdout_df = official_train.loc[holdout_mask].copy().sort_values("id").reset_index(drop=True)
    remaining_train = official_train.loc[~holdout_mask].copy()
    new_train = pd.concat([remaining_train, data["val_df"].copy()], ignore_index=True)
    new_train = new_train.sort_values("id").reset_index(drop=True)

    if holdout_df.empty or len(new_train) == 0:
        raise RuntimeError("Fresh split produced an empty train or holdout frame")
    if set(new_train.id).intersection(set(holdout_df.id)):
        raise RuntimeError("Holdout leakage detected")

    cfg = baseline.Config(
        recipe=args.recipe,
        seed=args.seed,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        accum_steps=args.accum_steps,
        workers=args.workers,
        beta=args.beta,
        amp=args.amp,
        device=args.device,
        pretrained=args.pretrained,
    )
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(out_dir / "holdout.csv", index=False)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "EXP-004 fresh baseline holdout",
        "data_root": str(args.data_root),
        "holdout_source": "original official_train only",
        "official_train_count": int(len(official_train)),
        "original_official_validation_count_added_to_train": int(len(data["val_df"])),
        "new_train_count": int(len(new_train)),
        "holdout_count": int(len(holdout_df)),
        "holdout_fraction_requested": args.holdout_fraction,
        "holdout_rule": "sha256(seed:image_id) / 0xffffffff < holdout_fraction",
        "seed": args.seed,
        "train_ids_disjoint_from_holdout": True,
        "test_labels_loaded": False,
        "config": asdict(cfg),
        "holdout_file": str(out_dir / "holdout.csv"),
        "next_step": "Blend holdout predictions with the friend's P7 OOF predictions by image_id after training completes.",
    }
    (out_dir / "fresh_holdout_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["new_train_count", "holdout_count", "test_labels_loaded", "config"]}, indent=2))

    # Reuse the audited, tested baseline training implementation.
    _, report = baseline.train_one_split(new_train, holdout_df, data, cfg, out_dir / "holdout", "fresh_holdout", device)
    audit["training_report"] = report
    (out_dir / "fresh_holdout_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
