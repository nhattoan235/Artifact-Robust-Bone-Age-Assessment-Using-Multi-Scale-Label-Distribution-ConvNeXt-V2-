from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "p7_final_v3/P7_COLAB_BUNDLE_V3.zip"


def wanted(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if "__pycache__" in relative.parts or path.suffix == ".pyc":
        return False
    if path == OUTPUT or path.name == "P7_COLAB_BUNDLE_V3_SHA256.txt":
        return False
    top = relative.parts[0]
    if top == "p1_baseline":
        return path.suffix == ".py" or path.name == "requirements.txt"
    if top == "p7_final":
        return path.suffix == ".py" or "manifests" in relative.parts or path.name == "P7_FOLD_REGISTRY.json"
    if top == "p7_final_v2":
        return "configs" in relative.parts or path.name in {"P7_V2_FOLD_REGISTRY.json", "validate_setup.py", "__init__.py"}
    if top == "p7_final_v3":
        return path.is_file()
    return False


files = sorted(
    (path for folder in ("p1_baseline", "p7_final", "p7_final_v2", "p7_final_v3")
     for path in (ROOT / folder).rglob("*") if wanted(path)),
    key=lambda path: path.relative_to(ROOT).as_posix(),
)
with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        archive.write(path, path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(OUTPUT) as archive:
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f"Bundle ZIP hỏng tại {bad}")
digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
(ROOT / "p7_final_v3/P7_COLAB_BUNDLE_V3_SHA256.txt").write_text(
    f"sha256={digest}\nfiles={len(files)}\nbytes={OUTPUT.stat().st_size}\n", encoding="utf-8"
)
print(f"Created {OUTPUT} files={len(files)} bytes={OUTPUT.stat().st_size} sha256={digest}")
