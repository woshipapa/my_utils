# SPDX-License-Identifier: Apache-2.0
"""Regression guard: importing my_utils must not import torch.

``my_utils/__init__.py`` re-exports the profiling package eagerly and defers
every torch-dependent subpackage behind a module-level ``__getattr__``
(PEP 562).  The check runs in a subprocess so it stays valid even when the
pytest process itself already has torch loaded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_my_utils_does_not_import_torch() -> None:
    code = (
        "import my_utils; "
        "import my_utils.profiling; "
        "import sys; "
        "assert 'torch' not in sys.modules, 'torch was imported eagerly'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        # `python -c` puts the cwd on sys.path, so the subprocess imports the
        # repo checkout regardless of where pytest was invoked from.
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
