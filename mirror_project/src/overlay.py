"""Overlay utilities for rendering pose results onto mirror displays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np

from .body_tracking import PoseResult


@dataclass
class OverlayColors:
    skeleton: Tuple[int, int, int] = (0, 255, 0)
    joints: Tuple[int, int, int] = (0, 128, 255)


class OverlayRenderer:
    """Project pose landmarks onto mirror video frames."""

    def __init__(self, colors: OverlayColors | None = None) -> None:
        self._colors = colors or OverlayColors()
        self._pairs = self._default_pairs()

    def render(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Return frame annotated with pose skeleton."""

        height, width = pose.image_height, pose.image_width
        points = pose.as_ndarray()
        coords = np.column_stack((points[:, 0] * width, points[:, 1] * height)).astype(int)

        annotated = frame.copy()
        for start, end in self._pairs:
            if start >= len(coords) or end >= len(coords):  # pragma: no cover - safety
                continue
            cv2.line(annotated, tuple(coords[start]), tuple(coords[end]), self._colors.skeleton, 2)
        for x, y in coords:
            cv2.circle(annotated, (x, y), 4, self._colors.joints, -1)
        return annotated

    def render_sequence(
        self, frames: Iterable[np.ndarray], poses: Iterable[PoseResult | None]
    ) -> Iterable[np.ndarray]:
        """Yield annotated frames for synchronized pose results."""

        for frame, pose in zip(frames, poses):
            if pose is None:
                yield frame
            else:
                yield self.render(frame, pose)

    @staticmethod
    def _default_pairs() -> Tuple[Tuple[int, int], ...]:
        # Simplified skeleton connection map
        return (
            (11, 12),  # shoulders
            (12, 14),  # right upper arm
            (14, 16),  # right lower arm
            (11, 13),  # left upper arm
            (13, 15),  # left lower arm
            (11, 23),  # left hip
            (12, 24),  # right hip
            (23, 25),  # left thigh
            (25, 27),  # left calf
            (24, 26),  # right thigh
            (26, 28),  # right calf
        )
