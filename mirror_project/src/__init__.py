"""Source package for the smart mirror project."""

from .capture import CaptureSession, CaptureConfig
from .body_tracking import PoseEstimator
from .overlay import OverlayRenderer

__all__ = [
    "CaptureSession",
    "CaptureConfig",
    "PoseEstimator",
    "OverlayRenderer",
]
