from __future__ import annotations

import re
from collections.abc import Mapping


ROI_NAMES = (
    "carpal",
    "mcp_thumb",
    "mcp_index",
    "mcp_middle",
    "mcp_ring",
    "mcp_little",
)
VIEW_NAMES = ("global",) + ROI_NAMES

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_mapping(record: Mapping, field: str) -> Mapping:
    value = record.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    missing = set(ROI_NAMES) - set(value)
    extra = set(value) - set(ROI_NAMES)
    if missing or extra:
        raise ValueError(
            f"{field} must contain exactly {ROI_NAMES}; missing={sorted(missing)} extra={sorted(extra)}"
        )
    return value


def validate_record(record: Mapping, *, development_only: bool = True) -> None:
    required = {
        "image_id",
        "split",
        "fold",
        "global_path",
        "source_sha256",
        "sex",
        "bone_age_months",
        "roi_paths",
        "roi_boxes",
        "roi_sha256",
        "quality_flags",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"record missing fields: {sorted(missing)}")
    if not str(record["image_id"]).strip():
        raise ValueError("image_id is empty")
    if development_only and str(record["split"]).lower() == "test":
        raise ValueError("test records are forbidden in development manifests")
    if not _SHA256.fullmatch(str(record["source_sha256"])):
        raise ValueError("source_sha256 must be 64 lowercase hex characters")
    for field in ("roi_paths", "roi_boxes", "roi_sha256", "quality_flags"):
        mapping = _require_mapping(record, field)
        if any(not str(mapping[name]).strip() for name in ROI_NAMES):
            raise ValueError(f"{field} contains an empty value")
    hashes = record["roi_sha256"]
    for name in ROI_NAMES:
        if not _SHA256.fullmatch(str(hashes[name])):
            raise ValueError(f"roi_sha256[{name}] must be 64 lowercase hex characters")

