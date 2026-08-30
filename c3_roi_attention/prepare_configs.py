"""Create locked C3-ROI + attention configs from the C3-ROI baseline."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_CONFIG_ROOT = ROOT / "c3_roi" / "configs"
OUTPUT_CONFIG_ROOT = ROOT / "c3_roi_attention" / "configs"


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def make_attention_config(source: str, fold: int) -> str:
    """Change only run identity, output location, and model architecture."""
    source = _replace_once(
        source,
        f'run_id = "C3_ROI_V1_FOLD_{fold}"',
        f'run_id = "C3_ROI_ATTN_V1_FOLD_{fold}"',
    )
    source = _replace_once(
        source,
        'output_root = "c3_roi/runs/C3_ROI_V1"',
        'output_root = "c3_roi_attention/runs/C3_ROI_ATTN_V1"',
    )
    return _replace_once(
        source,
        'architecture = "convnext_tiny"',
        'architecture = "convnext_tiny_spatial_attention"',
    )


def make_colab_config(source: str, drive_data_root: str) -> str:
    """Add an operational Drive checkpoint mirror without changing science keys."""
    mirror = Path(drive_data_root).as_posix().rstrip("/") + "/c3_attention_runs"
    marker = 'output_root = "c3_roi_attention/runs/C3_ROI_ATTN_V1"'
    replacement = marker + f'\ncheckpoint_mirror_root = "{mirror}"'
    return _replace_once(source, marker, replacement)


def generate_base_configs() -> list[Path]:
    OUTPUT_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fold in range(1, 6):
        source = (SOURCE_CONFIG_ROOT / f"fold_{fold}.toml").read_text(encoding="utf-8")
        destination = OUTPUT_CONFIG_ROOT / f"fold_{fold}.toml"
        destination.write_text(make_attention_config(source, fold), encoding="utf-8")
        outputs.append(destination)
    return outputs


def generate_colab_config(fold: int, drive_data_root: str) -> Path:
    base_path = OUTPUT_CONFIG_ROOT / f"fold_{fold}.toml"
    if not base_path.is_file():
        generate_base_configs()
    destination = OUTPUT_CONFIG_ROOT / f"fold_{fold}_colab.toml"
    destination.write_text(
        make_colab_config(base_path.read_text(encoding="utf-8"), drive_data_root),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--drive-data-root")
    args = parser.parse_args()
    base_paths = generate_base_configs()
    for path in base_paths:
        print(path.relative_to(ROOT).as_posix())
    if args.fold is not None:
        if not args.drive_data_root:
            parser.error("--drive-data-root is required with --fold")
        print(generate_colab_config(args.fold, args.drive_data_root).relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
