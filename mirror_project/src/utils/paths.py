"""Filesystem helpers."""
from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path
