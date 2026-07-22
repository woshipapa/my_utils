# SPDX-License-Identifier: Apache-2.0
"""Deprecation-shim tests (P1.7).

Covers the 0.1.x deprecation policy (docs/API_STABILITY.md):
- plain `import my_utils.profiling` and modern public API emit zero warnings;
- legacy flat-module aliases under my_utils.profiling warn on access;
- _LegacyNsysSqliteMetricsProvider warns on instantiation;
- legacy NsysLaunchConfig fields warn when set;
- behavior is preserved under the warning (aliases resolve to the real objects).
"""

import importlib
import subprocess
import sys
import warnings

import pytest

import my_utils.profiling as profiling
from my_utils.profiling.metrics.metrics_provider import BaseMetricsProvider
from my_utils.profiling.metrics.metrics_providers import (
    _LegacyNsysSqliteMetricsProvider,
)
from my_utils.profiling.runtime import capture_controller as real_capture_controller
from my_utils.profiling.runtime.config import NsysLaunchConfig


def _fresh(name):
    """Resolve a module at test time.

    Other tests in the suite purge my_utils.* from sys.modules and re-import,
    so module objects captured at this file's import time can go stale;
    identity checks must compare against the current sys.modules generation.
    """
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# (a) Importing the modern public API emits zero DeprecationWarnings.
# ---------------------------------------------------------------------------


def test_import_profiling_is_warning_free():
    """`python -W error::DeprecationWarning -c "import my_utils.profiling"` is clean.

    Run in a subprocess so the import is fresh (this test module has already
    imported my_utils.profiling in-process).
    """
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import my_utils.profiling",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import my_utils.profiling raised under -W error::DeprecationWarning:\n"
        f"{result.stderr}"
    )


def test_modern_public_api_access_is_warning_free():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Representative modern names: classes, functions, constants, config.
        assert profiling.CaptureController is real_capture_controller.CaptureController
        _ = profiling.MetricEvent
        _ = profiling.compute_mfu_single
        _ = profiling.NsysSqliteMetricsProvider
        _ = profiling.PROFILE_SCHEMA_VERSION
        NsysLaunchConfig()  # default/modern construction must not warn
        NsysLaunchConfig(gpu_metrics_devices="all", nic_metrics_mode="lf")


# ---------------------------------------------------------------------------
# (b) Legacy flat-module aliases fire DeprecationWarning on access.
# ---------------------------------------------------------------------------


def test_legacy_alias_nsys_mfu_warns_on_attribute_access():
    prof = _fresh("my_utils.profiling")
    with pytest.warns(
        DeprecationWarning, match=r"my_utils\.profiling\.nsys_mfu.*0\.3\.0"
    ):
        module = prof.nsys_mfu
    # (d) still resolves to the real relocated module.
    assert module is _fresh("my_utils.profiling.sources.nsys_mfu")
    assert module.compute_mfu_single is prof.compute_mfu_single


def test_legacy_alias_capture_controller_warns_on_attribute_access():
    prof = _fresh("my_utils.profiling")
    with pytest.warns(
        DeprecationWarning, match=r"my_utils\.profiling\.capture_controller.*0\.3\.0"
    ):
        module = prof.capture_controller
    assert module is _fresh("my_utils.profiling.runtime.capture_controller")
    assert module.CaptureController is prof.CaptureController


def test_legacy_alias_statement_form_import_warns_and_resolves():
    # Statement-form imports (`import my_utils.profiling.metrics_types`) go
    # through the meta-path finder rather than module __getattr__.
    prof = _fresh("my_utils.profiling")
    legacy_name = "my_utils.profiling.metrics_types"
    sys.modules.pop(legacy_name, None)
    with pytest.warns(
        DeprecationWarning, match=r"my_utils\.profiling\.metrics_types.*0\.3\.0"
    ):
        module = importlib.import_module(legacy_name)
    assert module is _fresh("my_utils.profiling.metrics.metrics_types")
    assert module.MetricEvent is prof.MetricEvent


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = profiling.definitely_not_a_real_name


# ---------------------------------------------------------------------------
# (c) _LegacyNsysSqliteMetricsProvider warns on instantiation.
# ---------------------------------------------------------------------------


def test_legacy_nsys_sqlite_provider_warns_on_init(tmp_path):
    with pytest.warns(
        DeprecationWarning, match=r"_LegacyNsysSqliteMetricsProvider.*0\.3\.0"
    ):
        provider = _LegacyNsysSqliteMetricsProvider(str(tmp_path / "trace.sqlite"))
    # (d) still fully functional under the warning.
    assert isinstance(provider, BaseMetricsProvider)
    assert provider.provider_id == "nsys_sqlite"
    assert provider.get_metrics() == []  # nonexistent file -> empty, no crash


# ---------------------------------------------------------------------------
# Legacy NsysLaunchConfig fields warn only when actually used.
# ---------------------------------------------------------------------------


def test_nsys_launch_config_legacy_gpu_metrics_device_warns():
    with pytest.warns(
        DeprecationWarning, match=r"gpu_metrics_device.*gpu_metrics_devices"
    ):
        cfg = NsysLaunchConfig(gpu_metrics_device="all")
    # (d) legacy field still round-trips.
    assert cfg.gpu_metrics_device == "all"


def test_nsys_launch_config_legacy_nic_metrics_warns():
    with pytest.warns(DeprecationWarning, match=r"nic_metrics.*nic_metrics_mode"):
        cfg = NsysLaunchConfig(nic_metrics=True)
    assert cfg.nic_metrics is True
