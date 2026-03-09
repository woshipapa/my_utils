from __future__ import annotations

from pathlib import Path


def get_profiling_templates_dir() -> Path:
    """Return the directory that stores reusable profiling shell templates."""
    return Path(__file__).resolve().parents[1] / "templates"


def get_profiling_template_path(filename: str) -> Path:
    """Return an absolute path to a profiling template file."""
    path = get_profiling_templates_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Profiling template not found: {path}")
    return path
