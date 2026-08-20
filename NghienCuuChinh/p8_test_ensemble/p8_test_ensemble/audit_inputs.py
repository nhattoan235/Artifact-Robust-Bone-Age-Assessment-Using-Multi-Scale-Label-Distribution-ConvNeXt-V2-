from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import torch


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def csv_rows(value: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(value.decode("utf-8-sig"))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument("--oof-zip", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--test-sex", type=Path, required=True)
    parser.add_argument("--test-ground-truth", type=Path, required=True)
    parser.add_argument("--ground-truth-copy", type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {"status": "PASS", "folds": {}}

    with zipfile.ZipFile(args.results_zip) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Results ZIP CRC lỗi: {bad}")
        names = set(archive.namelist())
        for fold in range(1, 6):
            prefix = f"results/P7_FINAL_V3_FOLD_{fold}/"
            manifest_name = prefix + "result_manifest.json"
            state_name = prefix + "run_state.json"
            if manifest_name not in names or state_name not in names:
                raise RuntimeError(f"Fold {fold} thiếu manifest/state")
            manifest = json.loads(archive.read(manifest_name))
            state = json.loads(archive.read(state_name))
            required = {"best_model.pt", "val_predictions_best.csv", "metrics.jsonl", "run_state.json"}
            if not required.issubset(manifest.get("files", {})):
                raise RuntimeError(f"Fold {fold} manifest thiếu {sorted(required - set(manifest.get('files', {})))}")
            for filename, expected in manifest["files"].items():
                member = prefix + filename
                if member not in names:
                    raise RuntimeError(f"Fold {fold} thiếu {filename}")
                value = archive.read(member)
                if len(value) != int(expected["bytes"]) or sha256_bytes(value) != expected["sha256"]:
                    raise RuntimeError(f"Fold {fold} sai size/SHA: {filename}")
            model = torch.load(io.BytesIO(archive.read(prefix + "best_model.pt")), map_location="cpu", weights_only=False)
            if not math.isclose(float(model["best_mae"]), float(state["best_mae"]), abs_tol=1e-9):
                raise RuntimeError(f"Fold {fold} model MAE không khớp state")
            if int(model["best_epoch"]) != int(state["best_epoch"]):
                raise RuntimeError(f"Fold {fold} model epoch không khớp state")
            report["folds"][str(fold)] = {
                "status": state["status"], "best_epoch": int(state["best_epoch"]) + 1,
                "best_mae": float(state["best_mae"]), "model_tensors": len(model["model"]),
                "manifest_sha_pass": True,
            }
            del model

    with zipfile.ZipFile(args.oof_zip) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"OOF ZIP CRC lỗi: {bad}")
        oof_report = json.loads(archive.read("oof_final/P7_OOF_report.json"))
        oof_rows = csv_rows(archive.read("oof_final/P7_OOF_predictions.csv"))
        ids = [row["image_id"] for row in oof_rows]
        if oof_report.get("status") != "PASS" or len(ids) != 14036 or len(set(ids)) != 14036:
            raise RuntimeError("OOF không PASS/đủ/unique 14.036")
        report["oof"] = {
            "rows": len(ids), "unique_ids": len(set(ids)),
            "mae": oof_report["pooled_oof_metrics"]["mae"],
            "report_status": oof_report["status"],
        }

    sex_rows = list(csv.DictReader(args.test_sex.open("r", encoding="utf-8-sig", newline="")))
    gt_rows = list(csv.DictReader(args.test_ground_truth.open("r", encoding="utf-8-sig", newline="")))
    sex_map = {str(int(row["Case ID"])): row["Sex"].strip().upper() for row in sex_rows}
    gt_map = {str(int(row["patient_ID"])): row for row in gt_rows}
    image_map = {path.stem: path for path in args.test_images.glob("*.png")}
    if not (len(sex_map) == len(gt_map) == len(image_map) == 200):
        raise RuntimeError(f"Test count sai: sex={len(sex_map)} gt={len(gt_map)} images={len(image_map)}")
    if set(sex_map) != set(gt_map) or set(gt_map) != set(image_map):
        raise RuntimeError("ID test images/sex/ground truth không khớp")
    if any(sex_map[key] != gt_map[key]["sex"].strip().upper() for key in gt_map):
        raise RuntimeError("Sex giữa hai annotation không khớp")
    ages = [float(row["bone_age"]) for row in gt_rows]
    if not all(math.isfinite(value) and 0 <= value <= 228 for value in ages):
        raise RuntimeError("Ground truth tuổi ngoài miền hoặc không hữu hạn")
    gt_sha = hashlib.sha256(args.test_ground_truth.read_bytes()).hexdigest()
    if args.ground_truth_copy:
        copy_rows = list(csv.DictReader(args.ground_truth_copy.open("r", encoding="utf-8-sig", newline="")))
        canonical = {
            str(int(row["patient_ID"])): (row["sex"].strip().upper(), float(row["bone_age"]))
            for row in gt_rows
        }
        canonical_copy = {
            str(int(row["patient_ID"])): (row["sex"].strip().upper(), float(row["bone_age"]))
            for row in copy_rows
        }
        if canonical_copy != canonical:
            raise RuntimeError("Hai bản ground truth Deeplasia khác nội dung ID/sex/bone_age")
    report["rsna_test"] = {
        "images": len(image_map), "sex_rows": len(sex_map), "ground_truth_rows": len(gt_map),
        "id_sex_alignment": True, "age_min": min(ages), "age_max": max(ages),
        "ground_truth_sha256": gt_sha, "ground_truth_semantic_copy_match": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
