# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``NCU_PATH`` / ``NSIGHT_COMPUTE_HOME`` environment overrides.

The env-var logic lives in one place (``nsight_paths``) and is consumed by
``section_index.find_sections_dir``, ``metric_catalog.verify_catalog_coverage``
and ``ncu_report_tools.find_ncu_report_dir``.  These tests are pure path
logic against ``tmp_path`` fixtures — no real Nsight Compute install is
required — and they also pin the legacy fallback lists byte-for-byte, so the
overrides can never silently change behaviour when the variables are unset.
"""

from __future__ import annotations

import glob as _glob
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

_NCU_DIR = Path(__file__).resolve().parents[2] / "my_utils" / "profiling" / "ncu"


def _load(name: str, filename: str):
    """Load one module standalone, under a test-private sys.modules name."""
    full = f"_nsight_paths_test_{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _NCU_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


nsight_paths = _load("nsight_paths", "nsight_paths.py")
section_index = _load("section_index", "section_index.py")
metric_catalog = _load("metric_catalog", "metric_catalog.py")
ncu_report_tools = _load("ncu_report_tools", "ncu_report_tools.py")

_ENV_VARS = ("NCU_PATH", "NSIGHT_COMPUTE_HOME", "NCU_PYTHON_DIR")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with no Nsight-related env vars set."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Fixture install trees
# ---------------------------------------------------------------------------

_SECTION_TEXT = """Identifier: "SpeedOfLight"
DisplayName: "GPU Speed Of Light Throughput"
Sets {
  Identifier: "full"
}
Metrics {
  Metrics {
    Name: "sm__throughput.avg.pct_of_peak_sustained_elapsed"
  }
}
"""


def _linux_install(root: Path, name: str = "nsight-compute-2026.1.1") -> Path:
    """A Linux-layout install: <install>/{ncu,sections,extras/python}."""
    install = root / name
    (install / "sections").mkdir(parents=True)
    (install / "sections" / "SpeedOfLight.section").write_text(_SECTION_TEXT)
    python_dir = install / "extras" / "python"
    python_dir.mkdir(parents=True)
    (python_dir / "ncu_report.py").write_text("# stub\n")
    (install / "ncu").write_text("#!/bin/sh\n")
    return install


def _mac_install(root: Path) -> Path:
    """A macOS app-bundle install with Contents/{Resources,MacOS} subtrees."""
    app = root / "NVIDIA Nsight Compute.app"
    (app / "Contents" / "Resources" / "sections").mkdir(parents=True)
    (app / "Contents" / "Resources" / "sections" / "SpeedOfLight.section").write_text(
        _SECTION_TEXT
    )
    python_dir = app / "Contents" / "MacOS" / "python"
    python_dir.mkdir(parents=True)
    (python_dir / "ncu_report.py").write_text("# stub\n")
    (app / "Contents" / "MacOS" / "ncu").write_text("binary")
    return app


# ---------------------------------------------------------------------------
# nsight_paths: install-dir normalization and precedence
# ---------------------------------------------------------------------------


def test_unset_env_vars_yield_no_candidates():
    assert nsight_paths.nsight_install_dirs({}) == []
    assert nsight_paths.sections_dir_candidates({}) == []
    assert nsight_paths.python_dir_candidates({}) == []


def test_ncu_path_accepts_an_install_dir(tmp_path):
    install = _linux_install(tmp_path)
    dirs = nsight_paths.nsight_install_dirs({"NCU_PATH": str(install)})
    assert dirs == [install]


def test_ncu_path_accepts_the_binary(tmp_path):
    install = _linux_install(tmp_path)
    dirs = nsight_paths.nsight_install_dirs({"NCU_PATH": str(install / "ncu")})
    assert dirs == [install]


def test_ncu_path_follows_a_symlinked_binary(tmp_path):
    install = _linux_install(tmp_path)
    link = tmp_path / "bin" / "ncu"
    link.parent.mkdir()
    link.symlink_to(install / "ncu")
    dirs = nsight_paths.nsight_install_dirs({"NCU_PATH": str(link)})
    assert dirs == [install]


def test_ncu_path_at_a_mac_bundle_binary_normalizes_to_the_app_root(tmp_path):
    app = _mac_install(tmp_path)
    binary = app / "Contents" / "MacOS" / "ncu"
    dirs = nsight_paths.nsight_install_dirs({"NCU_PATH": str(binary)})
    assert dirs == [app]


def test_ncu_path_wins_over_nsight_compute_home(tmp_path):
    first = _linux_install(tmp_path, "first")
    second = _linux_install(tmp_path, "second")
    dirs = nsight_paths.nsight_install_dirs(
        {"NCU_PATH": str(first), "NSIGHT_COMPUTE_HOME": str(second)}
    )
    assert dirs == [first, second]


def test_candidates_cover_both_install_layouts(tmp_path):
    install = tmp_path / "anywhere"
    env = {"NSIGHT_COMPUTE_HOME": str(install)}
    assert nsight_paths.sections_dir_candidates(env) == [
        str(install / "sections"),
        str(install / "Contents" / "Resources" / "sections"),
    ]
    assert nsight_paths.python_dir_candidates(env) == [
        str(install / "extras" / "python"),
        str(install / "Contents" / "MacOS" / "python"),
    ]


def test_find_ncu_binary_precedence(tmp_path):
    install = _linux_install(tmp_path)
    app = _mac_install(tmp_path)

    # NCU_PATH pointing straight at the binary wins.
    direct = nsight_paths.find_ncu_binary({"NCU_PATH": str(install / "ncu")})
    assert direct == install / "ncu"

    # An install dir is searched in both layouts.
    assert nsight_paths.find_ncu_binary({"NCU_PATH": str(install)}) == install / "ncu"
    assert (
        nsight_paths.find_ncu_binary({"NSIGHT_COMPUTE_HOME": str(app)})
        == app / "Contents" / "MacOS" / "ncu"
    )

    # Unset: falls through to a $PATH lookup, whatever this machine has.
    which = shutil.which("ncu")
    expected = Path(which) if which else None
    assert nsight_paths.find_ncu_binary({}) == expected


# ---------------------------------------------------------------------------
# section_index.find_sections_dir
# ---------------------------------------------------------------------------

# The historical fallback list, byte-for-byte. If this assertion ever fails,
# the "unset env vars keep legacy behaviour" guarantee has been broken.
_LEGACY_SECTION_DIRS = (
    "/Applications/NVIDIA Nsight Compute.app/Contents/Resources/sections",
    "/opt/nvidia/nsight-compute/*/sections",
    "/usr/local/cuda/nsight-compute-*/sections",
    "/usr/local/NVIDIA-Nsight-Compute*/sections",
)

_LEGACY_PYTHON_DIRS = (
    "/opt/nvidia/nsight-compute/*/extras/python",
    "/usr/local/cuda*/nsight-compute-*/extras/python",
    "/usr/local/NVIDIA-Nsight-Compute*/extras/python",
    "/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python",
    str(Path.home() / "nsight-compute" / "*" / "extras" / "python"),
)


def test_legacy_section_fallbacks_are_unchanged():
    assert section_index._DEFAULT_SECTION_DIRS == _LEGACY_SECTION_DIRS


def _legacy_find_sections_dir():
    """Reimplementation of the pre-override lookup, for equivalence checks."""
    for candidate in _LEGACY_SECTION_DIRS:
        if "*" in candidate:
            matches = sorted(_glob.glob(candidate))
            if matches:
                return Path(matches[-1])
        elif Path(candidate).is_dir():
            return Path(candidate)
    return None


def _legacy_find_ncu_report_dir():
    """Reimplementation of the pre-override lookup, for equivalence checks."""
    for candidate in _LEGACY_PYTHON_DIRS:
        for path in sorted(_glob.glob(candidate), reverse=True) or [candidate]:
            if (Path(path) / "ncu_report.py").exists():
                return Path(path)
    return None


def test_unset_env_vars_reproduce_legacy_sections_lookup():
    assert section_index.find_sections_dir() == _legacy_find_sections_dir()


def test_ncu_path_install_dir_locates_sections(tmp_path, monkeypatch):
    install = _linux_install(tmp_path)
    monkeypatch.setenv("NCU_PATH", str(install))
    assert section_index.find_sections_dir() == install / "sections"


def test_ncu_path_binary_locates_sections(tmp_path, monkeypatch):
    install = _linux_install(tmp_path)
    monkeypatch.setenv("NCU_PATH", str(install / "ncu"))
    assert section_index.find_sections_dir() == install / "sections"


def test_ncu_path_mac_bundle_binary_locates_sections(tmp_path, monkeypatch):
    app = _mac_install(tmp_path)
    monkeypatch.setenv("NCU_PATH", str(app / "Contents" / "MacOS" / "ncu"))
    expected = app / "Contents" / "Resources" / "sections"
    assert section_index.find_sections_dir() == expected


def test_nsight_compute_home_locates_sections(tmp_path, monkeypatch):
    install = _linux_install(tmp_path)
    monkeypatch.setenv("NSIGHT_COMPUTE_HOME", str(install))
    assert section_index.find_sections_dir() == install / "sections"


def test_ncu_path_beats_nsight_compute_home_for_sections(tmp_path, monkeypatch):
    winner = _linux_install(tmp_path, "winner")
    loser = _linux_install(tmp_path, "loser")
    monkeypatch.setenv("NCU_PATH", str(winner))
    monkeypatch.setenv("NSIGHT_COMPUTE_HOME", str(loser))
    assert section_index.find_sections_dir() == winner / "sections"


def test_explicit_argument_beats_env_vars(tmp_path, monkeypatch):
    env_install = _linux_install(tmp_path, "from-env")
    explicit = tmp_path / "explicit-sections"
    explicit.mkdir()
    monkeypatch.setenv("NCU_PATH", str(env_install))
    assert section_index.find_sections_dir(str(explicit)) == explicit


def test_bogus_env_vars_fall_through_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("NCU_PATH", str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("NSIGHT_COMPUTE_HOME", str(tmp_path / "also-missing"))
    assert section_index.find_sections_dir() == _legacy_find_sections_dir()


# ---------------------------------------------------------------------------
# ncu_report_tools.find_ncu_report_dir
# ---------------------------------------------------------------------------


def test_unset_env_vars_reproduce_legacy_python_lookup():
    assert ncu_report_tools.find_ncu_report_dir() == _legacy_find_ncu_report_dir()


def test_ncu_path_locates_the_python_dir(tmp_path, monkeypatch):
    install = _linux_install(tmp_path)
    monkeypatch.setenv("NCU_PATH", str(install))
    assert ncu_report_tools.find_ncu_report_dir() == install / "extras" / "python"


def test_ncu_path_mac_bundle_locates_the_python_dir(tmp_path, monkeypatch):
    app = _mac_install(tmp_path)
    monkeypatch.setenv("NCU_PATH", str(app / "Contents" / "MacOS" / "ncu"))
    expected = app / "Contents" / "MacOS" / "python"
    assert ncu_report_tools.find_ncu_report_dir() == expected


def test_nsight_compute_home_locates_the_python_dir(tmp_path, monkeypatch):
    install = _linux_install(tmp_path)
    monkeypatch.setenv("NSIGHT_COMPUTE_HOME", str(install))
    assert ncu_report_tools.find_ncu_report_dir() == install / "extras" / "python"


def test_ncu_python_dir_stays_the_most_specific_override(tmp_path, monkeypatch):
    """NCU_PYTHON_DIR names the python dir itself, so it outranks NCU_PATH."""
    install = _linux_install(tmp_path)
    specific = tmp_path / "elsewhere"
    specific.mkdir()
    (specific / "ncu_report.py").write_text("# stub\n")
    monkeypatch.setenv("NCU_PATH", str(install))
    monkeypatch.setenv("NCU_PYTHON_DIR", str(specific))
    assert ncu_report_tools.find_ncu_report_dir() == specific


# ---------------------------------------------------------------------------
# metric_catalog.verify_catalog_coverage
# ---------------------------------------------------------------------------


def test_verify_catalog_coverage_honours_ncu_path(tmp_path, monkeypatch):
    install = _linux_install(tmp_path)
    monkeypatch.setenv("NCU_PATH", str(install))
    result = metric_catalog.verify_catalog_coverage()
    assert result["available"] is True
    assert result["sections_dir"] == str(install / "sections")
    assert result["sections_in_full_set"] == 1


def test_verify_catalog_coverage_explicit_dir_beats_env(tmp_path, monkeypatch):
    env_install = _linux_install(tmp_path, "from-env")
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "Other.section").write_text(_SECTION_TEXT)
    monkeypatch.setenv("NCU_PATH", str(env_install))
    result = metric_catalog.verify_catalog_coverage(str(explicit))
    assert result["available"] is True
    assert result["sections_dir"] == str(explicit)
