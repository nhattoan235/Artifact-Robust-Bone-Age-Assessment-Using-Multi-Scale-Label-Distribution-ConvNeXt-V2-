from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from p1_baseline.config import load_config

from .storage import FINAL_STATUSES, MountedDriveStore, SnapshotLimitReached


def fingerprint(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="P7 V3: train/resume qua một Drive mount")
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--remote-minutes", type=float, default=60.0)
    parser.add_argument("--max-snapshots", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--interrupt-after-global-step", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    expected_run_id = f"P7_FINAL_V3_FOLD_{args.fold}"
    if cfg.run_id != expected_run_id:
        raise RuntimeError(f"Run ID sai: {cfg.run_id} != {expected_run_id}")
    store = MountedDriveStore(args.storage_root, cfg.run_id, args.max_snapshots)
    print(json.dumps(store.preflight(), ensure_ascii=False, indent=2), flush=True)
    if store.result_complete():
        print(json.dumps({"status": "SKIP_ALREADY_FINISHED", "fold": args.fold}, ensure_ascii=False))
        return 0

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    resume = store.restore_latest(run_dir / "last.ckpt")
    command = [sys.executable, "-u", "-m", "p1_baseline.train", "--config", str(args.config)]
    if resume:
        command += ["--resume", str(resume)]
    if args.interrupt_after_global_step is not None:
        command += ["--interrupt-after-global-step", str(args.interrupt_after_global_step)]
    print("TRAIN CHILD:", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=Path.cwd(), env=os.environ.copy())
    last_uploaded_at = time.monotonic()
    last_seen = fingerprint(run_dir / "last.ckpt") if resume else None
    last_best_seen = fingerprint(run_dir / "best_mae.ckpt")
    storage_failures = 0
    limit_reached = False
    try:
        while process.poll() is None:
            time.sleep(args.poll_seconds)
            checkpoint = run_dir / "last.ckpt"
            current = fingerprint(checkpoint)
            best = run_dir / "best_mae.ckpt"
            current_best = fingerprint(best)
            if current_best is not None and current_best != last_best_seen:
                try:
                    saved_best = store.save_best_model(best)
                    if saved_best:
                        print(f"PERSISTENT BEST MODEL PASS epoch={saved_best.epoch + 1} file={saved_best.checkpoint.name}", flush=True)
                        # Prediction được trainer ghi ngay trước best checkpoint;
                        # đồng bộ cùng transaction để reset session không làm mất OOF.
                        store.sync_small(run_dir)
                    last_best_seen = current_best
                    if saved_best and len(store.best_models()) >= store.max_best_models:
                        print("Đã lưu best-model cuối trong giới hạn; dừng fold an toàn để dọn bản cũ.", flush=True)
                        stop_child(process)
                        limit_reached = True
                        break
                except SnapshotLimitReached as exc:
                    print(f"STORAGE ROTATION REQUIRED: {exc}", flush=True)
                    stop_child(process)
                    limit_reached = True
                    break
            due = current is not None and current != last_seen and time.monotonic() - last_uploaded_at >= args.remote_minutes * 60
            if not due:
                continue
            try:
                snapshot = store.save_snapshot(checkpoint)
                if snapshot:
                    print(f"PERSISTENT CHECKPOINT PASS step={snapshot.global_step} file={snapshot.checkpoint.name}", flush=True)
                    last_uploaded_at = time.monotonic()
                    last_seen = current
                store.sync_small(run_dir)
                storage_failures = 0
                if snapshot and len(store.snapshots()) >= args.max_snapshots:
                    print("Đã lưu bản persistent cuối trong giới hạn; dừng fold an toàn để dọn checkpoint cũ.", flush=True)
                    stop_child(process)
                    limit_reached = True
                    break
            except SnapshotLimitReached as exc:
                print(f"STORAGE ROTATION REQUIRED: {exc}", flush=True)
                stop_child(process)
                limit_reached = True
                break
            except Exception as exc:
                storage_failures += 1
                print(f"STORAGE WARNING {storage_failures}/3: {type(exc).__name__}: {exc}", flush=True)
                if storage_failures >= 3:
                    stop_child(process)
                    raise RuntimeError("Dừng train vì lưu persistent thất bại 3 lần") from exc
    except KeyboardInterrupt:
        print("Đang dừng và lưu checkpoint gần nhất lên Drive...", flush=True)
        stop_child(process)
        store.save_snapshot(run_dir / "last.ckpt", force=True)
        store.sync_small(run_dir)
        return 130

    checkpoint = run_dir / "last.ckpt"
    if checkpoint.is_file() and not limit_reached:
        snapshot = store.save_snapshot(checkpoint, force=True)
        if snapshot:
            print(f"FINAL SESSION CHECKPOINT PASS step={snapshot.global_step}", flush=True)
    best = run_dir / "best_mae.ckpt"
    if best.is_file() and not limit_reached:
        try:
            saved_best = store.save_best_model(best)
            if saved_best:
                print(f"FINAL SESSION BEST MODEL PASS epoch={saved_best.epoch + 1}", flush=True)
        except SnapshotLimitReached as exc:
            print(f"BEST MODEL LIMIT: {exc}", flush=True)
    store.sync_small(run_dir)
    if limit_reached:
        return 75
    state_path = run_dir / "run_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Train kết thúc code={process.returncode} nhưng thiếu run_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    status = state.get("status")
    if status in FINAL_STATUSES:
        result = store.finalize(run_dir)
        print(json.dumps({"FINAL RESULT PASS": result}, ensure_ascii=False, indent=2))
    smoke_ok = args.interrupt_after_global_step is not None and status == "interrupted_for_resume_test"
    return 0 if status in FINAL_STATUSES or smoke_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
