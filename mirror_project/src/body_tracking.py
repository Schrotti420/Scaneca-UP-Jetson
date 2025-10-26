"""Body tracking abstractions using MediaPipe as the default backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "mediapipe is required for body tracking. Install it with 'pip install mediapipe'."
    ) from exc


@dataclass
class PoseLandmark:
    """Container for a single pose landmark in normalized coordinates."""

    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseResult:
    """Bundle of detected pose landmarks for a single frame."""

    landmarks: List[PoseLandmark]
    image_height: int
    image_width: int

    def as_ndarray(self) -> np.ndarray:
        """Return landmarks as an array of shape (N, 4)."""

        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in self.landmarks])


class PoseEstimator:
    """MediaPipe-based pose estimation wrapper."""

    def __init__(self, model_complexity: int = 1) -> None:
        self._mp_pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            enable_segmentation=False,
        )

    def estimate(self, frame: np.ndarray) -> Optional[PoseResult]:
        """Run pose estimation on a BGR frame."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None
        landmarks = [
            PoseLandmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )
            for lm in results.pose_landmarks.landmark
        ]
        height, width, _ = frame.shape
        return PoseResult(landmarks=landmarks, image_height=height, image_width=width)

    def batch_estimate(
        self, frames: Iterable[np.ndarray]
    ) -> Iterable[Optional[PoseResult]]:
        """Stream pose results for an iterable of frames."""

        for frame in frames:
            yield self.estimate(frame)

    def close(self) -> None:  # pragma: no cover - resource cleanup
        self._mp_pose.close()
