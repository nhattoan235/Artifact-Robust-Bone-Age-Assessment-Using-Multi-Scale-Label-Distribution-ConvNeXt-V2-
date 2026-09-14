"""Evaluate the locked 27/73 baseline/Pilot-C blend on multiple artifact seeds.

Seed 20260912 is reused from the existing OOF file.  Additional seeds run
artifact-only inference for the locked baseline and Pilot C checkpoints.  The
script never reads the 200-image RSNA test set and writes resumable per-fold
caches to Google Drive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONTENT_ROOT = Path("/content")
PILOT_CODE_ROOT = CONTENT_ROOT / "C3_Z26_C3_ROI_V2_PILOTS"
BASELINE_CODE_ROOT = CONTENT_ROOT / "C3_Z26_C3_ROI_V2"
for import_root in (CONTENT_ROOT, PILOT_CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DEFAULT_OOF = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/OOF/"
    "C3_Z26_C3_ROI_V2_PILOT_C_OOF_predictions.csv"
)
DEFAULT_OUTPUT = DEFAULT_OOF.parent / "multiseed_artifact"
BASELINE_RUN_ROOT = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2/runs"
)
PILOT_RUN_ROOT = Path(
    "/content/drive/MyDrive/RSNA_DATA/C3_Z26_C3_ROI_V2_PILOTS/runs"
)
LOCKED_EXISTING_SEED = 20260912
DEFAULT_SEEDS = (20260912, 20260913, 20260914)
AGE_EDGES = (0, 60, 120, 180, 229)
AGE_LABELS = ("0-59", "60-119", "120-179", "180-228")
OOF_REQUIRED = {
    "image_id", "fold", "sex", "target_months",
    "base_clean", "c_clean", "base_artifact", "c_artifact",
}


def _validate_oof(frame: pd.DataFrame) -> None:
    missing = OOF_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"OOF thiếu cột: {sorted(missing)}")
    if frame.empty or frame["image_id"].astype(str).duplicated().any():
        raise ValueError("OOF rỗng hoặc có image_id trùng")
    for column in OOF_REQUIRED - {"image_id", "sex"}:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"OOF cột {column} có NaN/Inf")


def _prediction_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"image_id", "artifact_prediction_months"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} thiếu cột: {sorted(missing)}")
    result = frame[["image_id", "artifact_prediction_months"]].copy()
    result["image_id"] = result["image_id"].astype(str)
    if result["image_id"].duplicated().any():
        raise ValueError(f"{source} có image_id trùng")
    values = pd.to_numeric(
        result["artifact_prediction_months"], errors="coerce"
    ).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{source} có prediction NaN/Inf")
    return result


def combine_seed_predictions(
    oof: pd.DataFrame, baseline: pd.DataFrame, pilot_c: pd.DataFrame,
    *, seed: int, weight_c: float,
) -> pd.DataFrame:
    """Align one artifact draw with locked clean OOF predictions."""
    _validate_oof(oof)
    if not 0.0 <= float(weight_c) <= 1.0:
        raise ValueError("weight_c phải nằm trong [0, 1]")
    base = _prediction_frame(baseline, "baseline").rename(
        columns={"artifact_prediction_months": "base_artifact_seed"}
    )
    pilot = _prediction_frame(pilot_c, "pilot_c").rename(
        columns={"artifact_prediction_months": "c_artifact_seed"}
    )
    ordered = oof.copy()
    ordered["image_id"] = ordered["image_id"].astype(str)
    if set(ordered["image_id"]) != set(base["image_id"]):
        raise ValueError("Baseline seed image_id không khớp OOF")
    if set(ordered["image_id"]) != set(pilot["image_id"]):
        raise ValueError("Pilot C seed image_id không khớp OOF")
    merged = ordered.merge(base, on="image_id", validate="one_to_one")
    merged = merged.merge(pilot, on="image_id", validate="one_to_one")
    weight_c = float(weight_c)
    merged["artifact_seed"] = int(seed)
    merged["blend_weight_c"] = weight_c
    merged["blend_clean"] = (
        (1.0 - weight_c) * merged["base_clean"]
        + weight_c * merged["c_clean"]
    )
    merged["blend_artifact"] = (
        (1.0 - weight_c) * merged["base_artifact_seed"]
        + weight_c * merged["c_artifact_seed"]
    )
    return merged


def _paired_ci(delta: np.ndarray, repetitions: int, seed: int) -> list[float]:
    if repetitions <= 0 or len(delta) == 0:
        raise ValueError("Bootstrap cần dữ liệu và repetitions > 0")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        values[start:start + size] = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _age_bins(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame["target_months"], bins=AGE_EDGES, right=False,
        labels=AGE_LABELS,
    )


def summarize_seed(
    frame: pd.DataFrame, *, repetitions: int = 10000,
    bootstrap_seed: int = 20260918,
) -> dict:
    y = frame["target_months"].to_numpy(dtype=np.float64)
    base_artifact = frame["base_artifact_seed"].to_numpy(dtype=np.float64)
    c_artifact = frame["c_artifact_seed"].to_numpy(dtype=np.float64)
    blend_artifact = frame["blend_artifact"].to_numpy(dtype=np.float64)
    base_clean = frame["base_clean"].to_numpy(dtype=np.float64)
    blend_clean = frame["blend_clean"].to_numpy(dtype=np.float64)
    baseline_error = np.abs(base_artifact - y)
    blend_error = np.abs(blend_artifact - y)
    delta = blend_error - baseline_error
    baseline_disagreement = np.abs(base_artifact - base_clean)
    blend_disagreement = np.abs(blend_artifact - blend_clean)

    work = frame.copy()
    work["age_bin"] = _age_bins(work)
    age_groups: dict[str, dict] = {}
    for offset, label in enumerate(AGE_LABELS):
        group = work[work["age_bin"].astype(str) == label]
        if group.empty:
            continue
        target = group["target_months"].to_numpy(dtype=np.float64)
        group_delta = (
            np.abs(group["blend_artifact"].to_numpy(dtype=np.float64) - target)
            - np.abs(group["base_artifact_seed"].to_numpy(dtype=np.float64) - target)
        )
        age_groups[label] = {
            "count": int(len(group)),
            "blend_artifact_delta": float(group_delta.mean()),
            "blend_artifact_delta_ci_95": _paired_ci(
                group_delta, repetitions, bootstrap_seed + 10 + offset
            ),
        }

    return {
        "artifact_seed": int(frame["artifact_seed"].iloc[0]),
        "count": int(len(frame)),
        "baseline_artifact_mae": float(baseline_error.mean()),
        "pilot_c_artifact_mae": float(np.abs(c_artifact - y).mean()),
        "blend_artifact_mae": float(blend_error.mean()),
        "blend_artifact_delta": float(delta.mean()),
        "blend_artifact_gain": float(-delta.mean()),
        "blend_delta_ci_95": _paired_ci(delta, repetitions, bootstrap_seed),
        "baseline_disagreement_mean": float(baseline_disagreement.mean()),
        "blend_disagreement_mean": float(blend_disagreement.mean()),
        "disagreement_reduction": float(
            1.0
            - blend_disagreement.mean()
            / max(baseline_disagreement.mean(), 1e-12)
        ),
        "age_bins": age_groups,
    }


def aggregate_multiseed(
    seed_frames: Iterable[pd.DataFrame], *, repetitions: int = 20000,
    bootstrap_seed: int = 20260919,
) -> dict:
    frames = list(seed_frames)
    if not frames:
        raise ValueError("Không có artifact seed để tổng hợp")
    reference_ids = frames[0]["image_id"].astype(str).tolist()
    for frame in frames:
        if frame["image_id"].astype(str).tolist() != reference_ids:
            raise ValueError("Thứ tự/image_id giữa các seed không khớp")
    seeds = sorted(int(frame["artifact_seed"].iloc[0]) for frame in frames)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Artifact seed bị trùng")

    y = frames[0]["target_months"].to_numpy(dtype=np.float64)
    base_errors = np.stack([
        np.abs(frame["base_artifact_seed"].to_numpy(dtype=np.float64) - y)
        for frame in frames
    ])
    blend_errors = np.stack([
        np.abs(frame["blend_artifact"].to_numpy(dtype=np.float64) - y)
        for frame in frames
    ])
    mean_base = base_errors.mean(axis=0)
    mean_blend = blend_errors.mean(axis=0)
    worst_base = base_errors.max(axis=0)
    worst_blend = blend_errors.max(axis=0)
    mean_delta = mean_blend - mean_base
    worst_delta = worst_blend - worst_base
    return {
        "image_count": int(len(y)),
        "artifact_seeds": seeds,
        "mean_seed_baseline_mae": float(mean_base.mean()),
        "mean_seed_blend_mae": float(mean_blend.mean()),
        "mean_seed_blend_gain": float(-mean_delta.mean()),
        "mean_seed_delta_ci_95": _paired_ci(
            mean_delta, repetitions, bootstrap_seed
        ),
        "worst_seed_baseline_mae": float(worst_base.mean()),
        "worst_seed_blend_mae": float(worst_blend.mean()),
        "worst_seed_blend_gain": float(-worst_delta.mean()),
        "worst_seed_delta_ci_95": _paired_ci(
            worst_delta, repetitions, bootstrap_seed + 1
        ),
    }


def _checkpoint(fold: int, model_name: str) -> Path:
    if model_name == "baseline":
        return (
            BASELINE_RUN_ROOT
            / f"C3_Z26_C3_ROI_V2_FOLD_{fold}"
            / "best_mae.ckpt"
        )
    return (
        PILOT_RUN_ROOT
        / f"C3_Z26_C3_ROI_V2_PILOT_C_V2_FOLD_{fold}_SEED_42"
        / "best_mae.ckpt"
    )


def _config(fold: int, model_name: str) -> Path:
    if model_name == "baseline":
        return BASELINE_CODE_ROOT / "configs_t4_b36" / f"fold_{fold}.toml"
    return (
        PILOT_CODE_ROOT / "configs"
        / f"C3_Z26_C3_ROI_V2_PILOT_C_FOLD_{fold}_SEED_42.toml"
    )


def _cache_path(output_dir: Path, seed: int, fold: int, model_name: str) -> Path:
    return output_dir / f"seed_{seed}" / f"fold_{fold}_{model_name}.csv"


def _read_cache(path: Path, expected_count: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    result = _prediction_frame(frame, str(path))
    if len(result) != expected_count:
        raise ValueError(
            f"Cache {path} cần {expected_count} dòng, có {len(result)}"
        )
    return result


def _infer_artifact(
    config_path: Path, checkpoint: Path, artifact_seed: int,
    batch_size: int,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    from p1_baseline.config import load_config
    from p1_baseline.data import BoneAgeDataset, load_manifest
    from p1_baseline.model import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("Cần GPU để inference artifact seed mới")
    cfg = load_config(config_path)
    rows = load_manifest(cfg.val_manifest, "validation_official")
    dataset = BoneAgeDataset(
        rows, cfg.image_size, cfg.target_mean, cfg.target_std,
        train=True, epoch=0, seed=int(artifact_seed), augmentation="none",
        preprocessing=cfg.preprocessing,
        preprocessed_root=cfg.preprocessed_root, image_root=cfg.image_root,
        image_normalization=cfg.image_normalization,
        artifact_augmentation="mild_v1", artifact_probability=1.0,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    device = torch.device("cuda")
    model = build_model(
        cfg.architecture, False, cfg.sex_embedding_dim,
        cfg.head_hidden_dim, cfg.dropout, cfg.age_class_count,
        sex_mode=cfg.sex_mode,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    records: list[dict] = []
    with torch.inference_mode():
        for batch in loader:
            artifact = batch["artifact_image"].to(device, non_blocking=True)
            sex = batch["sex"].to(device, non_blocking=True)
            output = model(artifact, sex)
            if isinstance(output, dict):
                output = output["regression"]
            months = (
                output * cfg.target_std + cfg.target_mean
            ).cpu().tolist()
            for image_id, value in zip(batch["image_id"], months):
                records.append({
                    "image_id": str(image_id),
                    "artifact_prediction_months": float(value),
                })
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(records)


def _load_or_infer(
    *, output_dir: Path, seed: int, fold: int, model_name: str,
    expected_count: int, batch_size: int,
) -> pd.DataFrame:
    cache = _cache_path(output_dir, seed, fold, model_name)
    if cache.is_file():
        print(f"CACHE seed={seed} fold={fold} model={model_name}", flush=True)
        return _read_cache(cache, expected_count)
    config_path = _config(fold, model_name)
    checkpoint = _checkpoint(fold, model_name)
    if not config_path.is_file():
        raise FileNotFoundError(f"Thiếu config: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint: {checkpoint}")
    print(f"START seed={seed} fold={fold} model={model_name}", flush=True)
    frame = _infer_artifact(config_path, checkpoint, seed, batch_size)
    if len(frame) != expected_count:
        raise ValueError(
            f"Inference seed={seed} fold={fold} model={model_name}: "
            f"cần {expected_count}, có {len(frame)}"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    print(f"DONE seed={seed} fold={fold} model={model_name}", flush=True)
    return frame


def run_multiseed(
    *, oof_path: Path = DEFAULT_OOF, output_dir: Path = DEFAULT_OUTPUT,
    seeds: Iterable[int] = DEFAULT_SEEDS, weight_c: float = 0.73,
    batch_size: int = 36, repetitions: int = 20000,
) -> dict:
    oof = pd.read_csv(oof_path)
    _validate_oof(oof)
    oof["image_id"] = oof["image_id"].astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_frames: list[pd.DataFrame] = []
    seed_reports: dict[str, dict] = {}

    for seed_index, seed in enumerate(tuple(int(value) for value in seeds)):
        if seed == LOCKED_EXISTING_SEED:
            baseline = oof[["image_id", "base_artifact"]].rename(
                columns={"base_artifact": "artifact_prediction_months"}
            )
            pilot = oof[["image_id", "c_artifact"]].rename(
                columns={"c_artifact": "artifact_prediction_months"}
            )
            print(f"REUSE locked OOF artifact seed={seed}", flush=True)
        else:
            baseline_parts: list[pd.DataFrame] = []
            pilot_parts: list[pd.DataFrame] = []
            for fold in range(1, 6):
                fold_count = int((oof["fold"] == fold).sum())
                baseline_parts.append(_load_or_infer(
                    output_dir=output_dir, seed=seed, fold=fold,
                    model_name="baseline", expected_count=fold_count,
                    batch_size=batch_size,
                ))
                pilot_parts.append(_load_or_infer(
                    output_dir=output_dir, seed=seed, fold=fold,
                    model_name="pilot_c", expected_count=fold_count,
                    batch_size=batch_size,
                ))
            baseline = pd.concat(baseline_parts, ignore_index=True)
            pilot = pd.concat(pilot_parts, ignore_index=True)
        combined = combine_seed_predictions(
            oof, baseline, pilot, seed=seed, weight_c=weight_c
        )
        combined_path = output_dir / f"seed_{seed}" / "combined_predictions.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        seed_frames.append(combined)
        seed_reports[str(seed)] = summarize_seed(
            combined, repetitions=repetitions,
            bootstrap_seed=20260920 + seed_index * 20,
        )
        current = seed_reports[str(seed)]
        print(
            f"SEED {seed}: baseline={current['baseline_artifact_mae']:.4f} "
            f"blend={current['blend_artifact_mae']:.4f} "
            f"gain={current['blend_artifact_gain']:.4f}",
            flush=True,
        )

    aggregate = aggregate_multiseed(
        seed_frames, repetitions=repetitions, bootstrap_seed=20261001
    )
    long_predictions = pd.concat(seed_frames, ignore_index=True)
    long_path = output_dir / "multiseed_artifact_predictions.csv"
    long_predictions.to_csv(long_path, index=False)
    report = {
        "protocol": (
            "Five-fold OOF only; fixed 27% baseline + 73% Pilot C; "
            "same mild_v1 artifact draw paired between models; three seeds"
        ),
        "test_accessed": False,
        "weight_baseline": 1.0 - float(weight_c),
        "weight_pilot_c": float(weight_c),
        "per_seed": seed_reports,
        "aggregate": aggregate,
        "limitations": [
            "Artifacts are synthetic mild_v1, not external hospital data.",
            "Three seeds characterize draw sensitivity but do not span every artifact type.",
            "Weight 0.73 was selected from the same development OOF predictions.",
        ],
        "predictions_csv": str(long_path),
    }
    report_path = output_dir / "multiseed_artifact_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"SAVED: {long_path}", flush=True)
    print(f"SAVED: {report_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-seed artifact OOF evaluation for fixed 27/73 blend"
    )
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--weight-c", type=float, default=0.73)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()
    run_multiseed(
        oof_path=args.oof, output_dir=args.output_dir,
        seeds=args.seeds, weight_c=args.weight_c,
        batch_size=args.batch_size, repetitions=args.bootstrap_repetitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
