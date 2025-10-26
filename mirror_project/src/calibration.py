"""Calibration helpers for aligning camera and mirror references."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CalibrationData:
    """Represents intrinsic and extrinsic calibration matrices."""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    extrinsics: Optional[np.ndarray] = None


def load_calibration(path: Path) -> CalibrationData:
    """Load calibration matrices from a NumPy .npz archive."""

    archive = np.load(path)
    return CalibrationData(
        camera_matrix=archive["camera_matrix"],
        distortion_coeffs=archive["distortion_coeffs"],
        extrinsics=archive.get("extrinsics"),
    )


def save_calibration(data: CalibrationData, path: Path) -> None:
    """Persist calibration matrices to a NumPy .npz archive."""

    np.savez(
        path,
        camera_matrix=data.camera_matrix,
        distortion_coeffs=data.distortion_coeffs,
        extrinsics=data.extrinsics,
    )
