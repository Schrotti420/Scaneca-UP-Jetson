"""Capture utilities for live RealSense streaming and recorded playback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - hardware dependent
    raise RuntimeError(
        "pyrealsense2 is required for capture operations. "
        "Install the RealSense SDK and Python bindings before using this module."
    ) from exc

import cv2
import numpy as np


@dataclass
class StreamProfile:
    """Stream resolution and framerate configuration."""

    width: int = 848
    height: int = 480
    fps: int = 30


@dataclass
class CaptureConfig:
    """Configuration for RealSense capture sessions."""

    color_stream: StreamProfile = StreamProfile()
    depth_stream: StreamProfile = StreamProfile()
    align_to_color: bool = True
    playback_file: Optional[Path] = None


class CaptureSession:
    """Context manager handling RealSense pipelines for live and playback."""

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._pipeline = rs.pipeline()
        self._align = rs.align(rs.stream.color) if config.align_to_color else None
        self._started = False

    def __enter__(self) -> "CaptureSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - resource cleanup
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.color,
            self._config.color_stream.width,
            self._config.color_stream.height,
            rs.format.bgr8,
            self._config.color_stream.fps,
        )
        cfg.enable_stream(
            rs.stream.depth,
            self._config.depth_stream.width,
            self._config.depth_stream.height,
            rs.format.z16,
            self._config.depth_stream.fps,
        )
        if self._config.playback_file:
            cfg.enable_device_from_file(str(self._config.playback_file), repeat_playback=True)
        self._pipeline.start(cfg)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._pipeline.stop()
        self._started = False

    def frames(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield synchronized color and depth frames as NumPy arrays."""

        if not self._started:
            raise RuntimeError("CaptureSession must be started before requesting frames")

        while True:
            frames = self._pipeline.wait_for_frames()
            if self._align is not None:
                frames = self._align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:  # pragma: no cover - defensive
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            yield color, depth

    def show_preview(self) -> None:  # pragma: no cover - requires GUI
        """Preview the stream in a window until ESC is pressed."""

        for color, depth in self.frames():
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            combined = np.hstack((color, depth_colormap))
            cv2.imshow("Color | Depth", combined)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()


def playback(file_path: Path) -> CaptureSession:
    """Convenience helper to build a playback session from a .bag recording."""

    return CaptureSession(CaptureConfig(playback_file=file_path))
