# SPDX-License-Identifier: Apache-2.0
"""Tests for the version-aware Nsight Compute audit.

The audit is deliberately pure Python: CI can prove that a local 2026.1.1
install is not mistaken for current documentation support without installing a
GPU driver or executing a profiler.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "my_utils"
        / "profiling"
        / "ncu"
        / "ncu_capability_audit.py"
    )
    spec = importlib.util.spec_from_file_location("ncu_capability_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def test_version_parser_handles_bundle_and_cli_text() -> None:
    assert audit.parse_ncu_version("2026.1.1.0 (build 37693799)") == (2026, 1, 1)
    assert audit.parse_ncu_version("Nsight Compute version 2026.2") == (2026, 2, 0)
    assert audit.parse_ncu_version("not installed") is None


def test_2026_1_1_is_not_claimed_current() -> None:
    result = audit.audit_ncu_capabilities("2026.1.1")
    assert result["version_status"] == "upgrade_required"
    statuses = {row["key"]: row["status"] for row in result["features"]}
    assert statuses["sass_instruction_size"] == "unavailable_in_installed_version"
    assert statuses["attach_process_id"] == "unavailable_in_installed_version"


def test_current_version_requires_real_report_validation() -> None:
    result = audit.audit_ncu_capabilities(
        "2026.2.1",
        configured_options=(
            "process-id",
            "list-injection-path-64",
            "list-injection-path-32",
        ),
    )
    assert result["version_status"] == "current"
    statuses = {row["key"]: row["status"] for row in result["features"]}
    assert statuses["attach_process_id"] == "awaiting_report_validation"
    assert statuses["sass_instruction_size"] == "awaiting_report_validation"


def test_wrapper_option_gap_is_not_hidden_by_a_current_tool() -> None:
    result = audit.audit_ncu_capabilities("2026.2.1", configured_options=())
    row = next(item for item in result["features"] if item["key"] == "attach_process_id")
    assert row["status"] == "wrapper_option_missing"
    assert row["missing_wrapper_options"] == ["process-id"]


def test_audit_does_not_assume_wrapper_gaps_without_option_inventory() -> None:
    result = audit.audit_ncu_capabilities("2026.2.1")

    row = next(item for item in result["features"] if item["key"] == "attach_process_id")
    assert result["configured_options_known"] is False
    assert row["status"] == "awaiting_report_validation"
    assert row["wrapper_option_coverage"] == "unknown"


def test_collection_provenance_marks_missing_values_unknown() -> None:
    result = audit.audit_collection_provenance(
        {
            "ncu_version": "2026.2.1",
            "driver_version": "580.1",
            "gpu_identities": ["6, GPU-abc, H100, 580.1, 0000:00:00.0"],
            "cuda_visible_devices": "6",
            "replay_mode": "kernel",
            "cache_control": "all",
            "clock_control": "boost",
            "pipeline_boost_state": "stable",
            "pm_sampling_interval": "1000",
            "warp_sampling_interval": "4",
            "kernel_name": "regex:foo",
            "kernel_id": "::1",
            "launch_count": "1",
            "launch_skip": "0",
        }
    )
    assert result["complete"] is True

    incomplete = audit.audit_collection_provenance({"replay_mode": "kernel"})
    assert incomplete["complete"] is False
    assert "ncu_version" in incomplete["missing_fields"]
