from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("resumed", type=Path)
    parser.add_argument("uninterrupted", type=Path)
    args = parser.parse_args()
    a = torch.load(args.resumed, map_location="cpu", weights_only=False)
    b = torch.load(args.uninterrupted, map_location="cpu", weights_only=False)
    model_equal = set(a["model"]) == set(b["model"]) and all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    state_equal = all(a[key] == b[key] for key in ["epoch", "batch_in_epoch", "samples_seen_in_epoch", "global_step", "best_mae", "best_epoch", "epochs_without_improvement", "config_hash"])
    report = {"status": "PASS" if model_equal and state_equal else "FAIL", "model_tensors_bitwise_equal": model_equal, "training_state_equal": state_equal, "resumed_global_step": a["global_step"], "uninterrupted_global_step": b["global_step"]}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
