from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from artifact_only_pipeline import ArtifactOnlyPipeline


def _optional_mask(directory: Path | None, case_id: str) -> Path | None:
    if directory is None:
        return None
    candidates = (
        directory / f"{case_id}.png",
        directory / f"{case_id}_mask.png",
        directory / f"{case_id}.npy",
        directory / f"{case_id}_mask.npy",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def run(
    input_dir: Path,
    output_dir: Path,
    seed_protection_dir: Path | None = None,
    manual_artifact_dir: Path | None = None,
) -> Path:
    images = sorted(input_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not images:
        raise SystemExit(f"No PNG images found in {input_dir}")

    cleaned_dir = output_dir / "cleaned"
    protected_dir = output_dir / "protected_masks"
    artifact_dir = output_dir / "artifact_masks"
    review_dir = output_dir / "review"
    for directory in (cleaned_dir, protected_dir, artifact_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pipeline = ArtifactOnlyPipeline()
    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(images, 1):
        case_id = image_path.stem
        result = pipeline.process(
            image_path,
            seed_protection_path=_optional_mask(seed_protection_dir, case_id),
            manual_artifact_path=_optional_mask(manual_artifact_dir, case_id),
        )
        cv2.imwrite(str(cleaned_dir / f"{case_id}.png"), result.cleaned)
        cv2.imwrite(str(protected_dir / f"{case_id}.png"), result.protected_mask)
        cv2.imwrite(str(artifact_dir / f"{case_id}.png"), result.artifact_mask)
        cv2.imwrite(str(review_dir / f"{case_id}.jpg"), result.review_overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(result.metrics)
        print(
            f"[{index:03d}/{len(images):03d}] {case_id}: {result.metrics['Status']} "
            f"edit={result.metrics['artifact_area_pct']:.3f}% "
            f"protected={result.metrics['protection_area_pct']:.1f}%"
        )

    csv_path = output_dir / "artifact_only_qc.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_count = len(list(cleaned_dir.glob("*.png")))
    if output_count != len(images):
        raise RuntimeError(f"Output count mismatch: expected {len(images)}, got {output_count}")
    if not all(bool(row["pixel_preservation_pass"]) for row in rows):
        raise RuntimeError("Pixel-preservation invariant failed; inspect QC CSV")

    print(f"\nCompleted {output_count}/{len(images)} images.")
    print(f"QC report: {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove only non-anatomical labels while locking all protected anatomy pixels."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-protection-dir", type=Path)
    parser.add_argument("--manual-artifact-dir", type=Path)
    args = parser.parse_args()
    run(
        args.input_dir,
        args.output_dir,
        seed_protection_dir=args.seed_protection_dir,
        manual_artifact_dir=args.manual_artifact_dir,
    )


if __name__ == "__main__":
    main()
