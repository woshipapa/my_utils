from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NsysVersionInfo:
    exporter_version: str = ""
    export_schema_version: str = ""
    adapter_family: str = "generic"
    known_version: bool = False
    parsed_version: List[int] = field(default_factory=list)


def _pick_first(meta: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        if key in meta and meta[key]:
            return str(meta[key])
    return ""


def _parse_version(version_text: str) -> List[int]:
    matches = re.findall(r"\d+", version_text or "")
    return [int(token) for token in matches[:4]]


def _resolve_family(version_nums: List[int]) -> str:
    if not version_nums:
        return "generic"
    year = version_nums[0]
    if year >= 2024:
        return "nsys_2024_plus"
    if year == 2023:
        return "nsys_2023"
    if year == 2022:
        return "nsys_2022"
    return "generic"


def detect_nsys_version(meta: Dict[str, str]) -> NsysVersionInfo:
    exporter_version = _pick_first(
        meta,
        [
            "NSIGHT_SYSTEMS_VERSION",
            "EXPORTER_VERSION",
            "EXPORT_VERSION",
            "TOOL_VERSION",
        ],
    )
    export_schema_version = _pick_first(
        meta,
        [
            "EXPORT_SCHEMA_VERSION",
            "EXPORT_SCHEMA",
        ],
    )
    parsed = _parse_version(exporter_version)
    family = _resolve_family(parsed)
    return NsysVersionInfo(
        exporter_version=exporter_version,
        export_schema_version=export_schema_version,
        adapter_family=family,
        known_version=bool(parsed),
        parsed_version=parsed,
    )

