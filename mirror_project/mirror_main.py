"""Command line entry point for the smart mirror pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.capture import CaptureSession, CaptureConfig
from src.body_tracking import PoseEstimator
from src.overlay import OverlayRenderer
from src.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart mirror capture and overlay demo")
    parser.add_argument(
        "--playback",
        type=Path,
        help="Optional path to a .bag recording for playback",
    )
    parser.add_argument(
        "--record-out",
        type=Path,
        help="Optional path to store annotated output video",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - interactive demo
    args = parse_args()
    renderer = OverlayRenderer()
    estimator = PoseEstimator()

    config = CaptureConfig(playback_file=args.playback)
    output_writer = None

    if args.record_out:
        ensure_directory(args.record_out.parent)

    with CaptureSession(config) as session:
        try:
            for color, depth in session.frames():
                pose = estimator.estimate(color)
                annotated = renderer.render(color, pose) if pose else color

                if args.record_out:
                    if output_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        height, width, _ = annotated.shape
                        output_writer = cv2.VideoWriter(
                            str(args.record_out), fourcc, config.color_stream.fps, (width, height)
                        )
                    output_writer.write(annotated)

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03),
                    cv2.COLORMAP_JET,
                )
                combined = cv2.hconcat([annotated, depth_colormap])
                cv2.imshow("Smart Mirror", combined)
                if cv2.waitKey(1) == 27:
                    break
        finally:
            if output_writer is not None:
                output_writer.release()
            cv2.destroyAllWindows()
            estimator.close()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
