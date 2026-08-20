from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from p1_baseline.trainer import CHECKPOINT_KEYS


SMALL_ARTIFACTS = (
    "run_state.json", "config_resolved.yaml", "environment.txt", "train.log",
    "warnings.log", "metrics.jsonl", "val_predictions_best.csv",
)
FINAL_STATUSES = {"early_stopped", "completed"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing = CHECKPOINT_KEYS - set(state)
    if missing:
        raise RuntimeError(f"Checkpoint thiếu key: {sorted(missing)}")
    return state


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Snapshot:
    checkpoint: Path
    metadata: Path
    global_step: int
    epoch: int
    sha256: str
    size: int


class SnapshotLimitReached(RuntimeError):
    pass


class MountedDriveStore:
    """Kho checkpoint append-only trên Drive đã mount.

    Checkpoint không bao giờ bị ghi đè hoặc tự xóa. Cách này tránh revision ẩn và
    tránh đưa checkpoint lớn vào Thùng rác ngoài ý muốn.
    """

    def __init__(self, root: Path, run_id: str, max_snapshots: int = 10, max_best_models: int = 15):
        self.root = root.resolve()
        self.run_id = run_id
        self.max_snapshots = max_snapshots
        self.max_best_models = max_best_models
        self.run_root = self.root / "checkpoints" / run_id
        self.snapshot_root = self.run_root / "snapshots"
        self.best_root = self.run_root / "best_models"
        self.result_root = self.root / "results" / run_id
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.best_root.mkdir(parents=True, exist_ok=True)
        if max_snapshots < 2:
            raise ValueError("max_snapshots phải >= 2")

    def preflight(self) -> dict:
        text = str(self.root).replace("\\", "/")
        if not text.startswith("/content/drive/"):
            raise RuntimeError(f"STORAGE_ROOT phải nằm trên Drive đã mount: {self.root}")
        marker = self.root / "P7_STORAGE_V3.marker"
        if not marker.is_file():
            raise RuntimeError(f"Thiếu marker: {marker}")
        token = f"p7-v3-probe-{time.time_ns()}"
        probe = self.root / f".{token}.tmp"
        probe.write_text(token, encoding="utf-8")
        if probe.read_text(encoding="utf-8") != token:
            raise RuntimeError("Drive mount ghi/đọc probe không khớp")
        probe.unlink()
        incomplete = list(self.snapshot_root.glob("*.uploading"))
        return {
            "storage_root": str(self.root),
            "marker": "PASS", "read_write_probe": "PASS",
            "snapshots": len(self.snapshots()),
            "best_models": len(self.best_models()),
            "incomplete_uploads": [path.name for path in incomplete],
            "max_snapshots": self.max_snapshots,
        }

    def snapshots(self) -> list[Snapshot]:
        output: list[Snapshot] = []
        for metadata in self.snapshot_root.glob("resume_*.json"):
            try:
                value = json.loads(metadata.read_text(encoding="utf-8"))
                checkpoint = self.snapshot_root / value["checkpoint"]
                output.append(Snapshot(
                    checkpoint=checkpoint, metadata=metadata,
                    global_step=int(value["global_step"]), epoch=int(value["epoch"]),
                    sha256=str(value["sha256"]), size=int(value["bytes"]),
                ))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        return sorted(output, key=lambda item: (item.global_step, item.checkpoint.name), reverse=True)

    def best_models(self) -> list[Snapshot]:
        output: list[Snapshot] = []
        for metadata in self.best_root.glob("best_*.json"):
            try:
                value = json.loads(metadata.read_text(encoding="utf-8"))
                model = self.best_root / value["checkpoint"]
                output.append(Snapshot(
                    checkpoint=model, metadata=metadata,
                    global_step=int(value["global_step"]), epoch=int(value["epoch"]),
                    sha256=str(value["sha256"]), size=int(value["bytes"]),
                ))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        return sorted(output, key=lambda item: (item.epoch, item.global_step, item.checkpoint.name), reverse=True)

    def latest_step(self) -> int:
        values = self.snapshots()
        return values[0].global_step if values else -1

    def save_snapshot(self, source: Path, force: bool = False) -> Snapshot | None:
        state = load_checkpoint(source)
        step = int(state["global_step"])
        epoch = int(state["epoch"])
        # Không bao giờ tạo hai bản cho cùng một global_step, kể cả lúc ép sync.
        if step <= self.latest_step():
            return None
        existing = self.snapshots()
        if len(existing) >= self.max_snapshots:
            raise SnapshotLimitReached(
                f"Đã đạt {self.max_snapshots} checkpoint persistent cho {self.run_id}. "
                "Fold được dừng an toàn; giữ hai resume mới nhất rồi dọn các bản cũ và Thùng rác."
            )
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        base = f"resume_s{step:08d}_e{epoch:03d}_{stamp}"
        destination = self.snapshot_root / f"{base}.ckpt"
        temporary = self.snapshot_root / f"{base}.ckpt.uploading"
        shutil.copyfile(source, temporary)
        expected_size = source.stat().st_size
        if temporary.stat().st_size != expected_size:
            raise RuntimeError("Checkpoint upload không đủ byte")
        expected_sha = sha256_file(source)
        actual_sha = sha256_file(temporary)
        if actual_sha != expected_sha:
            raise RuntimeError("SHA-256 checkpoint trên Drive không khớp")
        os.replace(temporary, destination)
        metadata = destination.with_suffix(".json")
        atomic_json(metadata, {
            "schema_version": 3, "checkpoint": destination.name,
            "run_id": self.run_id, "global_step": step, "epoch": epoch,
            "batch_in_epoch": int(state["batch_in_epoch"]),
            "best_mae": float(state["best_mae"]), "bytes": expected_size,
            "sha256": expected_sha, "created_utc": stamp,
        })
        return Snapshot(destination, metadata, step, epoch, expected_sha, expected_size)

    def restore_latest(self, destination: Path) -> Path | None:
        failures: list[str] = []
        for item in self.snapshots():
            try:
                if not item.checkpoint.is_file() or item.checkpoint.stat().st_size != item.size:
                    raise RuntimeError("thiếu file hoặc sai kích thước")
                if sha256_file(item.checkpoint) != item.sha256:
                    raise RuntimeError("SHA-256 sai")
                load_checkpoint(item.checkpoint)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".restore")
                shutil.copyfile(item.checkpoint, temporary)
                load_checkpoint(temporary)
                os.replace(temporary, destination)
                return destination
            except Exception as exc:
                failures.append(f"{item.checkpoint.name}: {exc}")
        if failures:
            raise RuntimeError("Không checkpoint nào phục hồi được:\n" + "\n".join(failures))
        return None

    def save_best_model(self, source: Path) -> Snapshot | None:
        state = load_checkpoint(source)
        epoch = int(state["best_epoch"])
        step = int(state["global_step"])
        mae = float(state["best_mae"])
        existing = self.best_models()
        if any(item.epoch == epoch for item in existing):
            return None
        if len(existing) >= self.max_best_models:
            raise SnapshotLimitReached(
                f"Đã đạt {self.max_best_models} best-model persistent cho {self.run_id}; "
                "dừng an toàn để giữ bản mới nhất và dọn các bản cũ."
            )
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        base = f"best_e{epoch + 1:03d}_mae{mae:.4f}_s{step:08d}_{stamp}"
        local_reduced = source.parent / f".{base}.pt"
        torch.save({
            "model": state["model"], "best_mae": mae, "best_epoch": epoch,
            "config_hash": state["config_hash"], "code_version": state["code_version"],
            "train_manifest_hash": state["train_manifest_hash"],
            "val_manifest_hash": state["val_manifest_hash"],
        }, local_reduced)
        destination = self.best_root / f"{base}.pt"
        temporary = self.best_root / f"{base}.pt.uploading"
        try:
            shutil.copyfile(local_reduced, temporary)
            expected_size = local_reduced.stat().st_size
            expected_sha = sha256_file(local_reduced)
            if temporary.stat().st_size != expected_size or sha256_file(temporary) != expected_sha:
                raise RuntimeError("Best-model upload sai kích thước hoặc SHA-256")
            os.replace(temporary, destination)
        finally:
            local_reduced.unlink(missing_ok=True)
        metadata = destination.with_suffix(".json")
        atomic_json(metadata, {
            "schema_version": 3, "checkpoint": destination.name, "run_id": self.run_id,
            "global_step": step, "epoch": epoch, "best_mae": mae,
            "bytes": expected_size, "sha256": expected_sha, "created_utc": stamp,
        })
        return Snapshot(destination, metadata, step, epoch, expected_sha, expected_size)

    def verified_latest_best(self) -> Path:
        failures = []
        for item in self.best_models():
            try:
                if not item.checkpoint.is_file() or item.checkpoint.stat().st_size != item.size:
                    raise RuntimeError("thiếu file hoặc sai kích thước")
                if sha256_file(item.checkpoint) != item.sha256:
                    raise RuntimeError("SHA-256 sai")
                value = torch.load(item.checkpoint, map_location="cpu", weights_only=False)
                if "model" not in value or "best_mae" not in value:
                    raise RuntimeError("thiếu model/best_mae")
                return item.checkpoint
            except Exception as exc:
                failures.append(f"{item.checkpoint.name}: {exc}")
        raise RuntimeError("Không có best-model persistent hợp lệ:\n" + "\n".join(failures))

    def sync_small(self, run_dir: Path) -> None:
        destination = self.run_root / "status"
        destination.mkdir(parents=True, exist_ok=True)
        for name in SMALL_ARTIFACTS:
            source = run_dir / name
            if not source.is_file():
                continue
            temporary = destination / f"{name}.tmp"
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination / name)

    def finalize(self, run_dir: Path) -> dict:
        state_path = run_dir / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") not in FINAL_STATUSES:
            raise RuntimeError(f"Không finalize trạng thái {state.get('status')!r}")
        best = run_dir / "best_mae.ckpt"
        local_best_state = load_checkpoint(best) if best.is_file() else None
        if best.is_file():
            try:
                self.save_best_model(best)
            except SnapshotLimitReached:
                # Fold đã kết thúc: result gọn bên dưới vẫn bảo toàn đúng local best,
                # không cần thêm một bản best-model trung gian.
                pass
        staging = self.root / "results" / f".{self.run_id}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        if local_best_state is not None:
            torch.save({
                "model": local_best_state["model"],
                "best_mae": local_best_state["best_mae"],
                "best_epoch": local_best_state["best_epoch"],
                "config_hash": local_best_state["config_hash"],
                "code_version": local_best_state["code_version"],
                "train_manifest_hash": local_best_state["train_manifest_hash"],
                "val_manifest_hash": local_best_state["val_manifest_hash"],
            }, staging / "best_model.pt")
        else:
            shutil.copyfile(self.verified_latest_best(), staging / "best_model.pt")
        for name in SMALL_ARTIFACTS:
            source = run_dir / name
            # Sau khi đổi session, prediction của best epoch có thể chỉ còn ở
            # status persistent dù best-model đã được bảo toàn riêng.
            if name == "val_predictions_best.csv" and not source.is_file():
                persistent = self.run_root / "status" / name
                source = persistent if persistent.is_file() else source
            if source.is_file():
                shutil.copyfile(source, staging / name)
        if not (staging / "val_predictions_best.csv").is_file():
            raise RuntimeError(
                "Thiếu val_predictions_best.csv; từ chối finalize vì chưa thể tổng hợp OOF"
            )
        manifest = {}
        for path in sorted(staging.iterdir()):
            if path.is_file():
                manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        atomic_json(staging / "result_manifest.json", {
            "schema_version": 3, "run_id": self.run_id, "status": state["status"],
            "best_mae": state["best_mae"], "files": manifest,
        })
        if self.result_root.exists():
            raise RuntimeError(f"Result đã tồn tại, không ghi đè: {self.result_root}")
        os.replace(staging, self.result_root)
        return json.loads((self.result_root / "result_manifest.json").read_text(encoding="utf-8"))

    def result_complete(self) -> bool:
        manifest = self.result_root / "result_manifest.json"
        if not manifest.is_file():
            return False
        value = json.loads(manifest.read_text(encoding="utf-8"))
        for name, expected in value.get("files", {}).items():
            path = self.result_root / name
            if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
                return False
            if sha256_file(path) != expected["sha256"]:
                return False
        return True
