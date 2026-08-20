from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np


class AnnotationServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler, config: dict[str, Path]):
        super().__init__(address, handler)
        self.config = config

    @property
    def status_path(self) -> Path:
        return self.config["manual_dir"] / "review_status.json"

    def load_statuses(self) -> dict[str, str]:
        if not self.status_path.exists():
            return {}
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def save_statuses(self, statuses: dict[str, str]) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(
            json.dumps(statuses, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class Handler(BaseHTTPRequestHandler):
    server: AnnotationServer

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def resolve_case_file(self, directory_key: str, case_id: str, suffix: str) -> Path:
        if not case_id.isdigit():
            raise ValueError("Invalid Case_ID")
        return self.server.config[directory_key] / f"{case_id}{suffix}"

    def case_records(self) -> list[dict[str, object]]:
        qc_path = self.server.config["qc"]
        qc: dict[str, dict[str, str]] = {}
        if qc_path.exists():
            with qc_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    qc[row["Case_ID"]] = row
        statuses = self.server.load_statuses()
        records: list[dict[str, object]] = []
        for image_path in sorted(
            self.server.config["input_dir"].glob("*.png"), key=lambda p: p.stem
        ):
            case_id = image_path.stem
            row = qc.get(case_id, {})
            records.append(
                {
                    "case_id": case_id,
                    "pipeline_status": row.get("Status", "UNKNOWN"),
                    "review_reason": row.get("Review_Reason", ""),
                    "artifact_area_pct": row.get("artifact_area_pct", ""),
                    "manual_mask": (
                        self.server.config["manual_dir"] / f"{case_id}.png"
                    ).exists(),
                    "review_status": statuses.get(case_id, "pending"),
                }
            )
        return records

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/index.html"}:
            self.send_file(self.server.config["ui_dir"] / "index.html")
            return
        if path == "/api/cases":
            self.send_json(self.case_records())
            return

        routes = {
            "/image/": ("input_dir", ".png"),
            "/auto-mask/": ("auto_mask_dir", ".png"),
            "/manual-mask/": ("manual_dir", ".png"),
            "/cleaned/": ("cleaned_dir", ".png"),
            "/review/": ("review_dir", ".jpg"),
        }
        for prefix, (key, suffix) in routes.items():
            if path.startswith(prefix):
                try:
                    self.send_file(
                        self.resolve_case_file(key, path[len(prefix) :], suffix)
                    )
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        if path.startswith("/api/mask/"):
            case_id = path.removeprefix("/api/mask/")
            if not case_id.isdigit():
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            encoded = np.frombuffer(body, dtype=np.uint8)
            mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            original = cv2.imread(
                str(self.server.config["input_dir"] / f"{case_id}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if mask is None or original is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid PNG or Case_ID")
                return
            if mask.shape != original.shape:
                self.send_error(HTTPStatus.BAD_REQUEST, "Mask shape mismatch")
                return
            mask = (mask >= 128).astype(np.uint8) * 255
            destination = self.server.config["manual_dir"] / f"{case_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), mask)
            statuses = self.server.load_statuses()
            statuses[case_id] = "annotated"
            self.server.save_statuses(statuses)
            self.send_json({"ok": True, "case_id": case_id})
            return

        if path.startswith("/api/status/"):
            case_id = path.removeprefix("/api/status/")
            if not case_id.isdigit():
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = json.loads(body.decode("utf-8"))
                status = payload["status"]
            except (ValueError, KeyError):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if status not in {"pending", "approved", "annotated", "skipped"}:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            statuses = self.server.load_statuses()
            statuses[case_id] = status
            self.server.save_statuses(statuses)
            self.send_json({"ok": True, "case_id": case_id, "status": status})
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[annotation] {self.address_string()} {fmt % args}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Local manual artifact-mask editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--input-dir", type=Path, default=root / "data/original_200")
    parser.add_argument(
        "--output-root", type=Path, default=root / "outputs/artifact_only_200_v2"
    )
    parser.add_argument(
        "--manual-dir", type=Path, default=root / "annotations/manual_artifacts"
    )
    args = parser.parse_args()

    config = {
        "ui_dir": root / "annotation_ui",
        "input_dir": args.input_dir.resolve(),
        "auto_mask_dir": (args.output_root / "artifact_masks").resolve(),
        "cleaned_dir": (args.output_root / "cleaned").resolve(),
        "review_dir": (args.output_root / "review").resolve(),
        "manual_dir": args.manual_dir.resolve(),
        "qc": (args.output_root / "artifact_only_qc.csv").resolve(),
    }
    config["manual_dir"].mkdir(parents=True, exist_ok=True)
    server = AnnotationServer((args.host, args.port), Handler, config)
    print(f"Mask editor: http://{args.host}:{args.port}")
    print(f"Manual masks: {config['manual_dir']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
