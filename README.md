# squidly-hardware-multi-camera-mapping
A project to map 2 camera view

The viewer now shows:

- Camera 0 with its detected bounding box and the projected box transferred from Camera 1.
- Camera 1 with its detected bounding box.
- A dedicated projection panel that overlays Camera 1's transferred box onto Camera 0 for easier comparison.

Projection settings live in [bbox_transfer.py](/Users/poyaohuang/dev/squidly/squidly-hardware-multi-camera-mapping/bbox_transfer.py). Adjust the baseline, vertical offset, operating depth, and intrinsics there to match your hardware.

The horizontal offset in [bbox_transfer.py](/Users/poyaohuang/dev/squidly/squidly-hardware-multi-camera-mapping/bbox_transfer.py) is defined from your point of view while you face the cameras. For the current `Camera 1 -> Camera 0` transfer, keep `CAM0_RELATIVE_TO_CAM1_VIEWER_X_MM` negative when Camera 0 is physically to your left of Camera 1.

## Run Guide

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

### Install dependencies and run

```sh
uv run main.py
```

This will automatically create a virtual environment, install dependencies (`mediapipe`, `opencv-python`), and run the program.

Press **q** to quit the application.
