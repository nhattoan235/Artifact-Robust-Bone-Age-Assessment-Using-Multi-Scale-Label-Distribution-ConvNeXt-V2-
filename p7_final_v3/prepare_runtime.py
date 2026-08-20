from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-data", type=Path, default=Path("/content/p7_data"))
    parser.add_argument("--local-runs", type=Path, default=Path("/content/p7_runs"))
    args = parser.parse_args()
    # Dùng đúng khóa khoa học của P7 V2: FP16 cố định, checkpoint local 5 phút.
    source = Path(f"p7_final_v2/configs/fold_{args.fold}.toml")
    text = source.read_text(encoding="utf-8")
    text = text.replace(f'run_id = "P7_FINAL_V2_FOLD_{args.fold}"', f'run_id = "P7_FINAL_V3_FOLD_{args.fold}"')
    text = text.replace('image_root = "/content/p7/data"', f'image_root = "{args.local_data.as_posix()}"')
    text = text.replace('output_root = "/content/p7/runs"', f'output_root = "{args.local_runs.as_posix()}"')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
