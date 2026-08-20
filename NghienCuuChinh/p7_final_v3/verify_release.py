from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    bundle = here / "P7_COLAB_BUNDLE_V3.zip"
    notebook = here / "P7_COLAB_V3.ipynb"
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip():
            raise RuntimeError("ZIP CRC FAIL")
        names = set(archive.namelist())
        required = {
            "p7_final_v3/run_fold.py", "p7_final_v3/storage.py",
            "p7_final_v3/prepare_runtime.py", "p7_final_v3/source_manifest.json",
            "p1_baseline/trainer.py", "p7_final/stage_images.py",
            "p7_final_v2/configs/fold_1.toml", "p7_final/manifests/development_portable.csv",
        }
        missing = required - names
        if missing:
            raise RuntimeError(f"Bundle thiếu: {sorted(missing)}")
        with tempfile.TemporaryDirectory(prefix="p7_v3_release_") as temporary:
            archive.extractall(temporary)
            root = Path(temporary)
            subprocess.run(
                [sys.executable, "-m", "unittest", "-v", "p7_final_v3.test_storage"],
                cwd=root, check=True,
            )
            registry = json.loads((root / "p7_final_v2/P7_V2_FOLD_REGISTRY.json").read_text(encoding="utf-8"))
            sys.path.insert(0, str(root))
            from p1_baseline.config import load_config, scientific_config_hash
            for fold in range(1, 6):
                output = root / f"runtime_fold_{fold}.toml"
                subprocess.run([
                    sys.executable, "-m", "p7_final_v3.prepare_runtime", "--fold", str(fold),
                    "--output", str(output), "--local-data", "/content/p7_v3/data",
                    "--local-runs", "/content/p7_v3/runs",
                ], cwd=root, check=True)
                cfg = load_config(output)
                assert cfg.run_id == f"P7_FINAL_V3_FOLD_{fold}"
                assert cfg.amp_dtype == "float16" and cfg.image_size == 512
                assert not cfg.checkpoint_mirror_root
                assert scientific_config_hash(cfg) == registry["folds"][str(fold)]["config_hash"]
    value = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(value["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook_cell_{index}", "exec")
    print(json.dumps({
        "release": "P7 V3", "bundle_crc": "PASS", "bundle_required_files": "PASS",
        "bundle_tests": "PASS", "five_scientific_hashes": "PASS",
        "notebook_syntax": "PASS", "notebook_cells": len(value["cells"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
