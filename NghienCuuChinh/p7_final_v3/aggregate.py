from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .storage import MountedDriveStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--local-runs", type=Path, default=Path("/content/p7_v3_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/p7_v3_oof"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.local_runs.exists():
        shutil.rmtree(args.local_runs)
    args.local_runs.mkdir(parents=True)
    for fold in range(1, 6):
        run_id = f"P7_FINAL_V3_FOLD_{fold}"
        store = MountedDriveStore(args.storage_root, run_id, 10)
        if not store.result_complete():
            raise RuntimeError(f"Fold {fold} chưa có result hoàn chỉnh")
        shutil.copytree(store.result_root, args.local_runs / run_id)
    subprocess.run([
        sys.executable, "-u", "-m", "p7_final.aggregate_oof",
        "--runs-root", str(args.local_runs), "--output-dir", str(args.output_dir),
        "--run-prefix", "P7_FINAL_V3_FOLD_", "--bootstrap", str(args.bootstrap),
    ], check=True)
    destination = args.storage_root / "oof_final"
    if destination.exists():
        raise RuntimeError(f"OOF output đã tồn tại, không ghi đè: {destination}")
    shutil.copytree(args.output_dir, destination)
    print(f"OOF FINAL PASS: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
