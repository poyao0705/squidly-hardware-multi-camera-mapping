# pyright: reportAttributeAccessIssue=false

import cv2
import numpy as np


# =============================================================================
# SETUP PARAMETERS - adjust these to match your physical setup
# =============================================================================

# Distance from the source camera to subject along optical axis (mm)
OPERATING_DEPTH_MM = 400.0

# Horizontal offset of Camera 0 relative to Camera 1 (mm), measured while YOU
# face the cameras.
#
# Negative = Camera 0 is to your LEFT of Camera 1.
# Positive = Camera 0 is to your RIGHT of Camera 1.
#
# For your setup, Camera 1 is on the right-hand side and Camera 0 is on the
# left-hand side, so this should stay NEGATIVE.
CAM0_RELATIVE_TO_CAM1_VIEWER_X_MM = -100.0

# Vertical offset of Camera 0 relative to Camera 1 (mm).
# Positive = Camera 0 appears LOWER in the image than Camera 1.
CAM0_RELATIVE_TO_CAM1_VIEWER_Y_MM = 0.0


# =============================================================================
# LOGITECH C270 CAMERA PARAMETERS (approximate - replace with calibrated values)
# =============================================================================

C270_FX = 600.0
C270_FY = 600.0

C270_CX = 320.0
C270_CY = 240.0

C270_DIST = np.array([0.1, -0.25, 0.0, 0.0, 0.1], dtype=np.float64)

DEFAULT_K1 = np.array(
    [[C270_FX, 0, C270_CX], [0, C270_FY, C270_CY], [0, 0, 1]],
    dtype=np.float64,
)
DEFAULT_K2 = DEFAULT_K1.copy()

DEFAULT_D1 = C270_DIST.copy()
DEFAULT_D2 = C270_DIST.copy()


# =============================================================================
# EXTRINSIC PARAMETERS (Camera 1 -> Camera 0)
# =============================================================================

def make_translation_from_viewer_offsets(horizontal_offset_mm, vertical_offset_mm):
    """
    Build a source-to-target translation vector for parallel cameras.

    Offsets are specified while the user faces the cameras.

    horizontal_offset_mm:
        Positive if the target camera is to the viewer's right of the source.
        Negative if the target camera is to the viewer's left of the source.

    vertical_offset_mm:
        Positive if the target camera is lower in the image than the source.
    """
    return np.array(
        [[horizontal_offset_mm], [-vertical_offset_mm], [0.0]], dtype=np.float64
    )


DEFAULT_R = np.eye(3, dtype=np.float64)
DEFAULT_T = make_translation_from_viewer_offsets(
    CAM0_RELATIVE_TO_CAM1_VIEWER_X_MM,
    CAM0_RELATIVE_TO_CAM1_VIEWER_Y_MM,
)


def xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def clamp_bbox_xyxy(bbox, image_shape):
    if bbox is None:
        return None

    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return int(x1), int(y1), int(x2), int(y2)


def project_bbox_to_cam2(
    bbox_cam1,
    K1=None,
    D1=None,
    K2=None,
    D2=None,
    R=None,
    T=None,
    operating_depth_mm=OPERATING_DEPTH_MM,
    image_shape_cam2=None,
):
    """
    Project a bounding box from a source camera into a target camera.

    Args:
        bbox_cam1: Bounding box in the source camera as (x1, y1, x2, y2) pixels.
        K1: 3x3 intrinsic matrix of the source camera.
        D1: Distortion coefficients of the source camera.
        K2: 3x3 intrinsic matrix of the target camera.
        D2: Distortion coefficients of the target camera.
        R: 3x3 rotation matrix from source-camera coordinates to target-camera coordinates.
        T: 3x1 translation vector from source-camera coordinates to target-camera coordinates.
        operating_depth_mm: Assumed subject depth from the source camera in mm.
        image_shape_cam2: Optional target frame shape for clamping.

    Returns:
        Bounding box in camera 2 as (x1, y1, x2, y2), or None if invalid.
    """
    if bbox_cam1 is None:
        return None

    K1 = DEFAULT_K1 if K1 is None else K1
    D1 = DEFAULT_D1 if D1 is None else D1
    K2 = DEFAULT_K2 if K2 is None else K2
    D2 = DEFAULT_D2 if D2 is None else D2
    R = DEFAULT_R if R is None else R
    T = DEFAULT_T if T is None else T

    x, y, w, h = xyxy_to_xywh(bbox_cam1)
    if w <= 0 or h <= 0:
        return None

    corners_cam1 = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64
    )

    corners_undist = cv2.undistortPoints(
        src=corners_cam1.reshape(-1, 1, 2), cameraMatrix=K1, distCoeffs=D1, P=K1
    ).reshape(-1, 2)

    depth = float(operating_depth_mm)
    fx1, fy1 = K1[0, 0], K1[1, 1]
    cx1, cy1 = K1[0, 2], K1[1, 2]

    points_3d = np.array(
        [
            [(u - cx1) * depth / fx1, (v - cy1) * depth / fy1, depth]
            for u, v in corners_undist
        ],
        dtype=np.float64,
    )

    r_vec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(
        objectPoints=points_3d,
        rvec=r_vec,
        tvec=T,
        cameraMatrix=K2,
        distCoeffs=D2,
    )
    projected = projected.reshape(-1, 2)

    x2, y2, w2, h2 = cv2.boundingRect(projected.astype(np.float32))
    bbox_cam2 = xywh_to_xyxy((x2, y2, w2, h2))

    if image_shape_cam2 is not None:
        return clamp_bbox_xyxy(bbox_cam2, image_shape_cam2)

    return tuple(int(value) for value in bbox_cam2)