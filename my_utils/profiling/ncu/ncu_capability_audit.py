# SPDX-License-Identifier: Apache-2.0
"""Version-aware audit for Nsight Compute collection and report support.

The metric catalog deliberately follows the locally installed ``.section``
files.  That keeps collection architecture-correct, but it is not proof that
the installation keeps pace with NVIDIA's current documentation.  This module
makes the distinction explicit: it records the current documentation baseline,
states which newer feature surface needs a newer report, and audits whether a
collection sidecar contains enough provenance to reproduce an A/B comparison.

It intentionally does *not* claim a feature works merely because a version is
new enough.  Report/API features remain ``awaiting_report_validation`` until a
report from that version has been parsed by this package.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CURRENT_NCU_DOCUMENTATION_VERSION",
    "parse_ncu_version",
    "format_ncu_version",
    "probe_ncu_version",
    "audit_ncu_capabilities",
    "audit_collection_provenance",
]


CURRENT_NCU_DOCUMENTATION_VERSION = "2026.2.1"

# New surfaces introduced after the locally audited 2026.1.1 installation.
# ``collection_options`` are exact CLI names, without leading ``--``.  A
# feature with no option is a report/UI surface and needs a real report for
# verification rather than an invented metric spelling.
_CURRENT_FEATURES: Tuple[Dict[str, Any], ...] = (
    {
        "key": "sass_instruction_size",
        "introduced": "2026.2",
        "kind": "metric",
        "summary": "SASS instruction-size metrics",
        "collection_options": (),
    },
    {
        "key": "function_statistics_line_time_range",
        "introduced": "2026.2",
        "kind": "report_api",
        "summary": "Function Statistics per high-level source line and represented time range",
        "collection_options": (),
    },
    {
        "key": "instruction_stats_hw_warp_id",
        "introduced": "2026.2",
        "kind": "report_api",
        "summary": "Instruction Statistics warp-can't-issue samples per HW warp-ID slot",
        "collection_options": (),
    },
    {
        "key": "cuda_injection_api",
        "introduced": "2026.2.1",
        "kind": "collection_api",
        "summary": "In-process dynamic CUDA injection API workflow",
        "collection_options": (),
    },
    {
        "key": "attach_process_id",
        "introduced": "2026.2.1",
        "kind": "cli",
        "summary": "Attach profiling to a specific process",
        "collection_options": ("process-id",),
    },
    {
        "key": "injection_path_listing",
        "introduced": "2026.2.1",
        "kind": "cli",
        "summary": "List 64-bit and 32-bit CUDA injection library paths",
        "collection_options": ("list-injection-path-64", "list-injection-path-32"),
    },
)

_VERSION_RE = re.compile(r"(?<!\d)(20\d{2})\.(\d+)(?:\.(\d+))?")


def parse_ncu_version(value: object) -> Optional[Tuple[int, int, int]]:
    """Parse an NCU version from CLI/bundle text without assuming its format."""
    match = _VERSION_RE.search(str(value or ""))
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def format_ncu_version(value: Optional[Sequence[int]]) -> str:
    """Render a parsed version using NCU's release-style spelling."""
    if not value or len(value) < 2:
        return ""
    major, minor = int(value[0]), int(value[1])
    patch = int(value[2]) if len(value) > 2 else 0
    return f"{major}.{minor}" + (f".{patch}" if patch else "")


def _normalise_options(options: Iterable[object]) -> set[str]:
    out = set()
    for option in options:
        name = str(option or "").strip()
        if not name:
            continue
        if name.startswith("--"):
            name = name[2:]
        out.add(name.split("=", 1)[0].replace("_", "-"))
    return out


def probe_ncu_version(executable: str = "ncu") -> Dict[str, Any]:
    """Run ``ncu --version`` without profiling, returning structured evidence."""
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "version": "",
            "parsed": None,
            "source": "ncu --version",
            "reason": str(exc),
        }
    output = (completed.stdout or completed.stderr or "").strip()
    parsed = parse_ncu_version(output)
    return {
        "available": bool(parsed),
        "version": format_ncu_version(parsed),
        "parsed": list(parsed) if parsed else None,
        "source": "ncu --version",
        "returncode": completed.returncode,
        "raw": output,
        "reason": "" if parsed else "could not parse an Nsight Compute version",
    }


def audit_ncu_capabilities(
    version: object = "",
    *,
    configured_options: Optional[Iterable[object]] = None,
    validated_feature_keys: Iterable[object] = (),
) -> Dict[str, Any]:
    """Compare an installed/configured NCU against the current doc baseline.

    ``configured_options`` should contain the option names that a collection
    wrapper exposes, not merely one invocation's choices.  This distinguishes
    “the installed profiler cannot do it” from “the wrapper can do it but this
    experiment did not request it”.  Pass an empty iterable only when that
    wrapper surface has been inspected and is known to expose no options.  A
    ``None`` value means the wrapper surface is unknown, so the audit must not
    call an unconfigured invocation an implementation gap.

    ``validated_feature_keys`` is intentionally explicit and normally comes
    from a controlled report-validation test.
    """
    installed = parse_ncu_version(version)
    current = parse_ncu_version(CURRENT_NCU_DOCUMENTATION_VERSION)
    options_known = configured_options is not None
    options = _normalise_options(configured_options or ())
    validated = {str(item or "").strip() for item in validated_feature_keys}

    features = []
    for feature in _CURRENT_FEATURES:
        introduced = parse_ncu_version(feature["introduced"])
        supported = installed is not None and introduced is not None and installed >= introduced
        required = tuple(feature.get("collection_options") or ())
        missing_options = [name for name in required if name not in options]
        if not supported:
            status = "unavailable_in_installed_version" if installed else "installed_version_unknown"
        elif feature["key"] in validated:
            status = "validated"
        elif options_known and missing_options:
            status = "wrapper_option_missing"
        else:
            status = "awaiting_report_validation"
        features.append(
            {
                "key": feature["key"],
                "introduced": feature["introduced"],
                "kind": feature["kind"],
                "summary": feature["summary"],
                "required_options": list(required),
                "missing_wrapper_options": missing_options if options_known else [],
                "wrapper_option_coverage": "known" if options_known else "unknown",
                "status": status,
            }
        )

    version_status = (
        "unknown"
        if installed is None
        else "current"
        if installed >= current
        else "upgrade_required"
    )
    return {
        "documentation_baseline": CURRENT_NCU_DOCUMENTATION_VERSION,
        "installed_version": format_ncu_version(installed),
        "installed_version_parsed": list(installed) if installed else None,
        "version_status": version_status,
        "configured_option_count": len(options),
        "configured_options_known": options_known,
        "features": features,
        "summary": (
            "Installed NCU is current against the documentation baseline; newer "
            "report surfaces still need controlled report validation."
            if version_status == "current"
            else "Installed NCU predates the documentation baseline; collect a current "
            "report before claiming full latest-version coverage."
            if version_status == "upgrade_required"
            else "Nsight Compute version was not available; version-dependent feature "
            "coverage cannot be claimed."
        ),
    }


def audit_collection_provenance(collection: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """State whether a collection sidecar is sufficient for reproducible A/B work."""
    values = dict(collection or {})
    required_groups = {
        "tool_and_driver": ("ncu_version", "driver_version"),
        "device_identity": ("gpu_identities", "cuda_visible_devices"),
        "replay_and_cache": ("replay_mode", "cache_control", "clock_control"),
        "pipeline_and_sampling": (
            "pipeline_boost_state",
            "pm_sampling_interval",
            "warp_sampling_interval",
        ),
        "kernel_selection": ("kernel_name", "kernel_id", "launch_count", "launch_skip"),
    }
    groups = []
    for name, fields in required_groups.items():
        missing = [field for field in fields if values.get(field) in (None, "", [])]
        groups.append(
            {
                "group": name,
                "fields": list(fields),
                "missing": missing,
                "complete": not missing,
            }
        )

    all_missing = sorted({field for group in groups for field in group["missing"]})
    return {
        "available": bool(values),
        "complete": bool(values) and not all_missing,
        "groups": groups,
        "missing_fields": all_missing,
        "summary": (
            "Collection provenance includes the tool, device, replay, sampling, and "
            "kernel-selection context needed for a reproducible comparison."
            if values and not all_missing
            else "Collection provenance is incomplete; missing fields are unknown, not "
            "defaults, so a timing difference may be a collection difference."
        ),
    }
