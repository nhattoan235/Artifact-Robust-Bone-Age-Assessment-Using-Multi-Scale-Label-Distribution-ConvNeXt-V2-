"""Evaluate baseline and Pilot C by synthetic artifact type on one locked fold.

The default diagnostic is Fold 5 because the fixed blend's clean regression is
largest there.  It reuses trained checkpoints and validation manifests; no
training and no 200-image test access occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1_baseline.artifacts import ARTIFACT_PROFILES, apply_artifact_profile  # noqa: E402
from p1_baseline.config import Config, load_config  # noqa: E402
from p1_baseline.data import BoneAgeDataset, load_manifest, pad_square  # noqa: E402
from p1_baseline.model import build_model  # noqa: E402


DEFAULT_BASELINE_CODE_ROOT = Path("/content/C3_Z26_C3_ROI_V2")
DEFAULT_PILOT_ROOT = Path("/content/C3_Z26_C3_ROI_V2_PILOTS")
DEFAULT_BASELINE_RUN_ROOT = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2/runs"
)
DEFAULT_PILOT_RUN_ROOT = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs"
)
DEFAULT_OOF = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/OOF/"
    "C3_Z26_C3_ROI_V2_PILOT_C_OOF_predictions.csv"
)
DEFAULT_OUTPUT = DEFAULT_OOF.parent / "artifact_matrix_fold5"
DEFAULT_PROFILES = tuple(profile for profile in ARTIFACT_PROFILES if profile != "clean")
AGE_EDGES = (0, 60, 120, 180, 229)
AGE_LABELS = ("0-59", "60-119", "120-179", "180-228")


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def build_cases(profiles: list[str], severities: list[float]) -> list[tuple[str, float, str]]:
    unknown = set(profiles) - set(DEFAULT_PROFILES)
    if unknown:
        raise ValueError(f"Profile khong hop le: {sorted(unknown)}")
    if any(not 0.0 < value <= 1.0 for value in severities):
        raise ValueError("Severity phai nam trong (0, 1]")
    cases = [("clean", 0.0, "clean")]
    for profile in profiles:
        for severity in severities:
            cases.append((profile, float(severity), f"{profile}_s{severity:.2f}"))
    return cases


class ArtifactMatrixDataset(Dataset):
    def __init__(
        self, rows: list[dict[str, str]], cfg: Config,
        cases: list[tuple[str, float, str]], *, fold: int,
        artifact_seed: int,
    ) -> None:
        self.rows = rows
        self.cfg = cfg
        self.cases = cases
        self.fold = int(fold)
        self.artifact_seed = int(artifact_seed)
        self.tensorizer = BoneAgeDataset(
            [], cfg.image_size, cfg.target_mean, cfg.target_std,
            image_normalization=cfg.image_normalization,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, row: dict[str, str]) -> Path:
        path = Path(row["image_path"])
        if self.cfg.image_root and not path.is_absolute():
            path = Path(self.cfg.image_root) / path
        if self.cfg.preprocessing != "none":
            path = (
                Path(self.cfg.preprocessed_root) / row["split"]
                / f"{row['image_id']}.png"
            )
        return path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self._path(row)
        try:
            with Image.open(path) as source:
                image = pad_square(source.convert("L"))
                image = TF.resize(
                    image, [self.cfg.image_size, self.cfg.image_size],
                    interpolation=InterpolationMode.BICUBIC, antialias=True,
                )
                tensors = []
                for profile, severity, _ in self.cases:
                    if profile == "composite_mild":
                        seed = _stable_seed(
                            f"artifact|{self.artifact_seed}|0|"
                            f"{row['image_id']}|mild_v1"
                        )
                    else:
                        seed = _stable_seed(
                            f"artifact-profile-v1|{self.artifact_seed}|{self.fold}|"
                            f"{row['image_id']}|{profile}|{severity:.8f}"
                        )
                    view = apply_artifact_profile(
                        image, profile, seed=seed, severity=severity,
                    )
                    tensors.append(self.tensorizer._to_tensor(view))
        except Exception as exc:
            raise RuntimeError(f"Khong doc duoc anh {path}: {exc}") from exc
        return {
            "images": torch.stack(tensors),
            "sex": torch.tensor(
                [1.0 if row["sex"] == "M" else 0.0], dtype=torch.float32,
            ),
            "image_id": str(row["image_id"]),
            "sex_text": str(row["sex"]),
            "target_months": torch.tensor(float(row["bone_age_months"])),
        }


def _load_model(cfg: Config, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(
        cfg.architecture, False, cfg.sex_embedding_dim, cfg.head_hidden_dim,
        cfg.dropout, cfg.age_class_count, sex_mode=cfg.sex_mode,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def infer_matrix(
    cfg: Config, checkpoint: Path, loader: DataLoader,
    *, case_count: int, device: torch.device,
) -> np.ndarray:
    model = _load_model(cfg, checkpoint, device)
    parts: list[np.ndarray] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        batch_size = images.shape[0]
        flat_images = images.flatten(0, 1)
        sex = batch["sex"].to(device, non_blocking=True)
        flat_sex = sex[:, None, :].expand(-1, case_count, -1).flatten(0, 1)
        output = model(flat_images, flat_sex)
        if isinstance(output, dict):
            output = output["regression"]
        months = output.reshape(batch_size, case_count)
        months = months * cfg.target_std + cfg.target_mean
        parts.append(months.cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(parts, axis=0)


def _paired_ci(delta: np.ndarray, repetitions: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        values[start:start + size] = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def summarize_matrix(
    predictions: pd.DataFrame, *, weight_c: float = 0.73,
    repetitions: int = 20000, seed: int = 20260914,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "image_id", "sex", "target_months", "profile", "severity",
        "case", "base_prediction", "c_prediction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction thieu cot: {sorted(missing)}")
    work = predictions.copy()
    work["blend_prediction"] = (
        (1.0 - weight_c) * work["base_prediction"]
        + weight_c * work["c_prediction"]
    )
    work["age_bin"] = pd.cut(
        work["target_months"], AGE_EDGES, right=False, labels=AGE_LABELS,
    ).astype(str)
    clean = work[work["case"] == "clean"][
        ["image_id", "base_prediction", "c_prediction", "blend_prediction"]
    ].rename(columns={
        "base_prediction": "base_clean",
        "c_prediction": "c_clean",
        "blend_prediction": "blend_clean",
    })
    work = work.merge(clean, on="image_id", validate="many_to_one")
    summary_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    for offset, (case, group) in enumerate(work.groupby("case", sort=False)):
        y = group["target_months"].to_numpy(dtype=np.float64)
        base_error = np.abs(group["base_prediction"].to_numpy() - y)
        c_error = np.abs(group["c_prediction"].to_numpy() - y)
        blend_error = np.abs(group["blend_prediction"].to_numpy() - y)
        c_delta = c_error - base_error
        blend_delta = blend_error - base_error
        c_interval = _paired_ci(c_delta, repetitions, seed + offset)
        blend_interval = _paired_ci(
            blend_delta, repetitions, seed + 100 + offset,
        )
        summary_rows.append({
            "case": case,
            "profile": str(group["profile"].iloc[0]),
            "severity": float(group["severity"].iloc[0]),
            "count": int(len(group)),
            "base_mae": float(base_error.mean()),
            "c_mae": float(c_error.mean()),
            "blend_mae": float(blend_error.mean()),
            "c_delta": float(c_delta.mean()),
            "c_delta_ci_low": c_interval[0],
            "c_delta_ci_high": c_interval[1],
            "blend_delta": float(blend_delta.mean()),
            "blend_delta_ci_low": blend_interval[0],
            "blend_delta_ci_high": blend_interval[1],
            "base_disagreement": float(np.abs(
                group["base_prediction"] - group["base_clean"]
            ).mean()),
            "c_disagreement": float(np.abs(
                group["c_prediction"] - group["c_clean"]
            ).mean()),
            "blend_disagreement": float(np.abs(
                group["blend_prediction"] - group["blend_clean"]
            ).mean()),
        })
        for age, age_group in group.groupby("age_bin", observed=True):
            age_y = age_group["target_months"].to_numpy(dtype=np.float64)
            age_base = np.abs(age_group["base_prediction"].to_numpy() - age_y)
            age_c = np.abs(age_group["c_prediction"].to_numpy() - age_y)
            age_blend = np.abs(age_group["blend_prediction"].to_numpy() - age_y)
            subgroup_rows.append({
                "case": case,
                "profile": str(group["profile"].iloc[0]),
                "severity": float(group["severity"].iloc[0]),
                "age_bin": str(age),
                "count": int(len(age_group)),
                "base_mae": float(age_base.mean()),
                "c_mae": float(age_c.mean()),
                "blend_mae": float(age_blend.mean()),
                "c_delta": float((age_c - age_base).mean()),
                "blend_delta": float((age_blend - age_base).mean()),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(subgroup_rows)


def _config_and_checkpoint(
    fold: int, *, candidate: bool, baseline_code_root: Path,
    pilot_root: Path, baseline_run_root: Path, pilot_run_root: Path,
) -> tuple[Config, Path]:
    if candidate:
        config = pilot_root / "configs" / (
            f"C3_Z26_C3_ROI_V2_PILOT_C_FOLD_{fold}_SEED_42.toml"
        )
        checkpoint = pilot_run_root / (
            f"C3_Z26_C3_ROI_V2_PILOT_C_V2_FOLD_{fold}_SEED_42"
        ) / "best_mae.ckpt"
    else:
        config = baseline_code_root / "configs_t4_b36" / f"fold_{fold}.toml"
        checkpoint = baseline_run_root / (
            f"C3_Z26_C3_ROI_V2_FOLD_{fold}"
        ) / "best_mae.ckpt"
    if not config.is_file():
        raise FileNotFoundError(f"Thieu config: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Thieu checkpoint: {checkpoint}")
    return load_config(config), checkpoint


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Can GPU cho artifact-matrix inference")
    cases = build_cases(args.profiles, args.severities)
    base_cfg, base_checkpoint = _config_and_checkpoint(
        args.fold, candidate=False, baseline_code_root=args.baseline_code_root,
        pilot_root=args.pilot_root, baseline_run_root=args.baseline_run_root,
        pilot_run_root=args.pilot_run_root,
    )
    c_cfg, c_checkpoint = _config_and_checkpoint(
        args.fold, candidate=True, baseline_code_root=args.baseline_code_root,
        pilot_root=args.pilot_root, baseline_run_root=args.baseline_run_root,
        pilot_run_root=args.pilot_run_root,
    )
    if base_cfg.val_manifest != c_cfg.val_manifest:
        raise ValueError("Baseline va Pilot C khong dung cung val manifest")
    base_cfg = replace(base_cfg, batch_size=args.image_batch_size, num_workers=args.num_workers)
    c_cfg = replace(c_cfg, batch_size=args.image_batch_size, num_workers=args.num_workers)
    rows = load_manifest(base_cfg.val_manifest, "validation_official")
    dataset = ArtifactMatrixDataset(
        rows, base_cfg, cases, fold=args.fold, artifact_seed=args.artifact_seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.image_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    device = torch.device("cuda")
    base_matrix = infer_matrix(
        base_cfg, base_checkpoint, loader, case_count=len(cases), device=device,
    )
    c_matrix = infer_matrix(
        c_cfg, c_checkpoint, loader, case_count=len(cases), device=device,
    )
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for case_index, (profile, severity, case) in enumerate(cases):
            records.append({
                "image_id": str(row["image_id"]),
                "fold": args.fold,
                "sex": str(row["sex"]),
                "target_months": float(row["bone_age_months"]),
                "profile": profile,
                "severity": severity,
                "case": case,
                "base_prediction": float(base_matrix[row_index, case_index]),
                "c_prediction": float(c_matrix[row_index, case_index]),
            })
    predictions = pd.DataFrame(records)
    locked = pd.read_csv(args.oof)
    locked = locked[locked["fold"] == args.fold].copy()
    clean = predictions[predictions["case"] == "clean"]
    check = clean.merge(
        locked[["image_id", "base_clean", "c_clean"]].assign(
            image_id=lambda x: x["image_id"].astype(str)
        ), on="image_id", validate="one_to_one",
    )
    clean_error = max(
        float(np.abs(check["base_prediction"] - check["base_clean"]).max()),
        float(np.abs(check["c_prediction"] - check["c_clean"]).max()),
    )
    if clean_error > args.integrity_tolerance:
        raise RuntimeError(
            f"Clean integrity FAIL: max prediction mismatch={clean_error:.6f}"
        )
    composite_error: float | None = None
    if "composite_mild_s1.00" in set(predictions["case"]):
        composite = predictions[predictions["case"] == "composite_mild_s1.00"]
        check = composite.merge(
            locked[["image_id", "base_artifact", "c_artifact"]].assign(
                image_id=lambda x: x["image_id"].astype(str)
            ), on="image_id", validate="one_to_one",
        )
        composite_error = max(
            float(np.abs(check["base_prediction"] - check["base_artifact"]).max()),
            float(np.abs(check["c_prediction"] - check["c_artifact"]).max()),
        )
        if composite_error > args.integrity_tolerance:
            raise RuntimeError(
                f"Composite integrity FAIL: max prediction mismatch={composite_error:.6f}"
            )
    summary, subgroups = summarize_matrix(
        predictions, weight_c=args.weight_c,
        repetitions=args.bootstrap_repetitions,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "artifact_matrix_predictions.csv"
    summary_path = args.output_dir / "artifact_matrix_summary.csv"
    subgroup_path = args.output_dir / "artifact_matrix_age_subgroups.csv"
    predictions.to_csv(prediction_path, index=False)
    summary.to_csv(summary_path, index=False)
    subgroups.to_csv(subgroup_path, index=False)
    report = {
        "protocol": (
            f"Fold {args.fold} validation only; fixed diagnostic artifact types; "
            "no training; no 200-image test access"
        ),
        "test_accessed": False,
        "fold": args.fold,
        "count": len(rows),
        "artifact_seed": args.artifact_seed,
        "weight_c": args.weight_c,
        "cases": [case for _, _, case in cases],
        "integrity": {
            "tolerance_months": args.integrity_tolerance,
            "clean_max_prediction_mismatch_months": clean_error,
            "composite_max_prediction_mismatch_months": composite_error,
            "pass": True,
        },
        "files": {
            "predictions": str(prediction_path),
            "summary": str(summary_path),
            "age_subgroups": str(subgroup_path),
        },
    }
    report_path = args.output_dir / "artifact_matrix_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"SAVED: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Artifact type matrix on a locked fold")
    parser.add_argument("--fold", type=int, default=5, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--severities", nargs="+", type=float, default=[1.0])
    parser.add_argument("--artifact-seed", type=int, default=20260912)
    parser.add_argument("--weight-c", type=float, default=0.73)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--integrity-tolerance", type=float, default=0.02)
    parser.add_argument("--baseline-code-root", type=Path, default=DEFAULT_BASELINE_CODE_ROOT)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--baseline-run-root", type=Path, default=DEFAULT_BASELINE_RUN_ROOT)
    parser.add_argument("--pilot-run-root", type=Path, default=DEFAULT_PILOT_RUN_ROOT)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
