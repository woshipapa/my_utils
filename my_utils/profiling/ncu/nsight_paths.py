# SPDX-License-Identifier: Apache-2.0
"""Locate a Nsight Compute installation from environment variables.

Single source of truth for the env-var override logic shared by
``ncu_report_tools`` (which needs the bundled ``ncu_report`` Python module),
``section_index`` and ``metric_catalog`` (which need the ``.section`` files).
Two variables are honoured, most specific first:

* ``NCU_PATH`` — path to the ``ncu`` binary *or* to the installation
  directory.  Both spellings work: a binary path is normalized to the install
  directory containing it (following symlinks, and stepping out of a macOS
  app bundle's ``Contents/MacOS``).
* ``NSIGHT_COMPUTE_HOME`` — the installation directory.

These only *prepend* candidates: every caller keeps its historical platform
fallbacks after them, so behaviour with the variables unset is unchanged.

Nsight Compute ships two on-disk layouts.  Both are derived for every install
dir - callers check existence, so only the layout that is actually present is
ever picked up:

* Linux / CUDA-bundled: ``<install>/sections``, ``<install>/extras/python``,
  ``<install>/ncu``
* macOS app bundle: ``<app>/Contents/Resources/sections``,
  ``<app>/Contents/MacOS/python``, ``<app>/Contents/MacOS/ncu``
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

__all__ = [
    "NCU_PATH_ENV",
    "NSIGHT_COMPUTE_HOME_ENV",
    "nsight_install_dirs",
    "sections_dir_candidates",
    "python_dir_candidates",
    "find_ncu_binary",
]

NCU_PATH_ENV = "NCU_PATH"
NSIGHT_COMPUTE_HOME_ENV = "NSIGHT_COMPUTE_HOME"

# Relative locations of each resource inside the two install layouts.
_SECTIONS_LAYOUTS: Tuple[Tuple[str, ...], ...] = (
    ("sections",),  # Linux / CUDA-bundled
    ("Contents", "Resources", "sections"),  # macOS app bundle
)
_PYTHON_LAYOUTS: Tuple[Tuple[str, ...], ...] = (
    ("extras", "python"),
    ("Contents", "MacOS", "python"),
)
_BINARY_LAYOUTS: Tuple[Tuple[str, ...], ...] = (
    ("ncu",),
    ("Contents", "MacOS", "ncu"),
)


def _env(environ: Optional[Mapping[str, str]], name: str) -> str:
    source = os.environ if environ is None else environ
    return str(source.get(name, "") or "").strip()


def _install_dir_from_ncu_path(raw: str) -> Path:
    """Normalize ``NCU_PATH`` — binary or install dir — to the install dir."""
    path = Path(raw).expanduser()
    if not path.is_dir() and (path.is_file() or path.name in ("ncu", "ncu.exe")):
        # Points at the binary. Follow symlinks (e.g. /usr/local/bin/ncu) to
        # the real install before taking the parent directory.
        if path.is_file():
            try:
                path = path.resolve()
            except OSError:
                pass
        path = path.parent
    # A macOS app-bundle binary lives in <app>/Contents/MacOS; normalize to
    # the .app root so both Resources/ and MacOS/ subtrees are derivable.
    if path.name == "MacOS" and path.parent.name == "Contents":
        path = path.parent.parent
    return path


def nsight_install_dirs(environ: Optional[Mapping[str, str]] = None) -> List[Path]:
    """Install dirs named by the env vars, most-preferred first.

    ``NCU_PATH`` wins over ``NSIGHT_COMPUTE_HOME``.  The returned dirs are not
    checked for existence — callers filter — and the list is empty when
    neither variable is set, which is what keeps legacy fallback behaviour
    untouched.
    """
    dirs: List[Path] = []
    ncu_path = _env(environ, NCU_PATH_ENV)
    if ncu_path:
        dirs.append(_install_dir_from_ncu_path(ncu_path))
    home = _env(environ, NSIGHT_COMPUTE_HOME_ENV)
    if home:
        dirs.append(Path(home).expanduser())
    deduped: List[Path] = []
    seen = set()
    for d in dirs:
        if str(d) not in seen:
            seen.add(str(d))
            deduped.append(d)
    return deduped


def _candidates(
    environ: Optional[Mapping[str, str]],
    layouts: Tuple[Tuple[str, ...], ...],
) -> List[str]:
    out: List[str] = []
    seen = set()
    for install in nsight_install_dirs(environ):
        # Only ordering, not correctness: an .app bundle tries the bundle
        # layout first; either way only the layout that exists is usable.
        ordered = tuple(reversed(layouts)) if install.name.endswith(".app") else layouts
        for parts in ordered:
            candidate = str(install.joinpath(*parts))
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def sections_dir_candidates(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Env-var derived candidates for the ``sections`` directory, in
    precedence order.  Empty when neither env var is set."""
    return _candidates(environ, _SECTIONS_LAYOUTS)


def python_dir_candidates(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Env-var derived candidates for the directory holding ``ncu_report.py``,
    in precedence order.  Empty when neither env var is set."""
    return _candidates(environ, _PYTHON_LAYOUTS)


def find_ncu_binary(environ: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Path to the ``ncu`` executable, or ``None``.

    ``NCU_PATH`` pointing directly at the binary wins; otherwise the binary is
    looked for inside the env-named install dirs; finally ``$PATH``.
    """
    raw = _env(environ, NCU_PATH_ENV)
    if raw:
        direct = Path(raw).expanduser()
        if direct.is_file():
            return direct
    for install in nsight_install_dirs(environ):
        for parts in _BINARY_LAYOUTS:
            candidate = install.joinpath(*parts)
            if candidate.is_file():
                return candidate
    which = shutil.which("ncu")
    return Path(which) if which else None
