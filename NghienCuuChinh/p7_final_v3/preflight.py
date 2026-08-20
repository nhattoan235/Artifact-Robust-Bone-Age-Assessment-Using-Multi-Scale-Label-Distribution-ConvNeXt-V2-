from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .storage import MountedDriveStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--max-snapshots", type=int, default=10)
    parser.add_argument("--local-root", type=Path, default=Path("/content"))
    args = parser.parse_args()
    usage = shutil.disk_usage(args.local_root)
    if usage.free < 25 * 1024**3:
        raise RuntimeError(f"Ổ local còn dưới 25 GiB: {usage.free / 1024**3:.2f} GiB")
    store = MountedDriveStore(args.storage_root, f"P7_FINAL_V3_FOLD_{args.fold}", args.max_snapshots)
    report = store.preflight()
    report["local_free_gib"] = round(usage.free / 1024**3, 2)
    report["estimated_checkpoint_mib"] = "321-334"
    report["max_new_checkpoint_gib"] = round(
        max(0, args.max_snapshots - report["snapshots"]) * 334 / 1024, 2
    )
    if report["incomplete_uploads"]:
        raise RuntimeError(
            "Có upload dở từ session trước. Không tự xóa để tránh mất dữ liệu: "
            + ", ".join(report["incomplete_uploads"])
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P7 V3 PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
