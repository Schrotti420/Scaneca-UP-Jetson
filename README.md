# Scaneca-UP-Jetson

This repository hosts the capture and estimation software for the Scaneca smart mirror
prototype. The code base is organised to separate data acquisition, calibration, body
tracking, and visual overlay concerns so that each piece can be iterated on
independently.

## Project layout

```
mirror_project/
 ├─ data/
 │   └─ recordings/           # store your .bag recordings here
 ├─ src/
 │   ├─ capture.py            # live capture and playback helpers
 │   ├─ calibration.py        # mirror/camera calibration storage helpers
 │   ├─ body_tracking.py      # MediaPipe-based pose estimation
 │   ├─ overlay.py            # rendering of pose overlays
 │   └─ utils/
 ├─ tests/
 └─ mirror_main.py            # interactive demo entry point
```

The `mirror_main.py` script currently demonstrates playback from an Intel RealSense `.bag`
recording while running body tracking and rendering pose overlays.

## Setup

Follow these steps on your Jetson device to prepare the environment.

### 1. Install RealSense SDK

```bash
sudo apt update
sudo apt install -y git cmake build-essential libssl-dev libusb-1.0-0-dev \
                    pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense && mkdir build && cd build
cmake .. -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS=ON
make -j4
sudo make install
```

### 2. Python environment

```bash
sudo apt install python3-pip
pip install numpy opencv-python pyrealsense2 mediapipe
```

Optional virtual environment:

```bash
python3 -m venv ~/mirror-env
source ~/mirror-env/bin/activate
```

## Recording test data

1. Launch the RealSense Viewer:
   ```bash
   realsense-viewer
   ```
2. Enable the streams:
   - Color: 848x480 @ 30 FPS
   - Depth: 848x480 @ 30 FPS (aligned to color)
3. Record a session (`person_test.bag`) with the subject roughly 2 m from the camera in
   normal indoor lighting. Encourage light movement (rotation, arm motion) to capture a
   range of poses.

Place the resulting `.bag` file in `mirror_project/data/recordings/`.

## Minimal playback test

Create a file similar to the following or run the included `mirror_main.py` script to
validate playback:

```python
import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("person_test.bag", repeat_playback=True)
pipe.start(cfg)

align = rs.align(rs.stream.color)

while True:
    frames = pipe.wait_for_frames()
    frames = align.process(frames)
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth = np.asanyarray(frames.get_depth_frame().get_data())

    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=0.03),
        cv2.COLORMAP_JET
    )
    combined = np.hstack((color, depth_colormap))
    cv2.imshow("Color | Depth", combined)

    if cv2.waitKey(1) == 27:
        break

pipe.stop()
cv2.destroyAllWindows()
```

## Validation checklist

| Test | Ziel |
| ---- | ---- |
| `realsense-viewer` erkennt D415 | Treiber & USB ok |
| `.bag`-Aufnahme möglich | Farbtiefe synchron |
| Python-Skript zeigt Color+Depth | SDK & Bindings ok |
| FPS stabil (>20fps) | USB3 ok |
| Auf Jetson läuft ohne Absturz | Performance ausreichend |

## Next steps

The modular structure allows iterative improvements:

1. Start with playback before integrating live streaming.
2. Begin with 2D keypoint tracking (MediaPipe) before expanding to 3D mapping.
3. Introduce advanced mirror reflection logic after verifying overlay behaviour.
