from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import MountedDriveStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for fold in range(1, 6):
        run_id = f"P7_FINAL_V3_FOLD_{fold}"
        store = MountedDriveStore(args.storage_root, run_id, 10)
        snapshots = store.snapshots()
        state_path = store.run_root / "status" / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        rows.append({
            "fold": fold, "run_id": run_id, "status": state.get("status", "not_started"),
            "epoch": state.get("epoch"), "global_step": state.get("global_step"),
            "best_mae": state.get("best_mae"), "persistent_snapshots": len(snapshots),
            "persistent_best_models": len(store.best_models()),
            "latest_snapshot_step": snapshots[0].global_step if snapshots else None,
            "result_verified": store.result_complete(),
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
