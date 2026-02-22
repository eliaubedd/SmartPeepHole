"""
Face Preprocessing and Detection Utilities

This module implements the face preprocessing stage of the SmartPeephole
pipeline. It provides:

1) Lightweight structured data containers used internally to describe
   detection results in a model-agnostic way.
2) A face alignment routine (rotation + crop + resize) used to normalize
   faces before embedding extraction.
3) A detection helper that extracts the highest-confidence face bounding
   box and eye landmarks from a frame.

The goal of this module is to:
    - Isolate face alignment logic from recognition logic
    - Provide consistent, normalized face crops to downstream models
    - Keep the detection interface independent from any specific framework
"""

from typing import NamedTuple

import cv2
import numpy as np
from mediapipe.python.solutions import face_detection



# Structured detection containers (pipeline-level abstractions)
class BoundingBox(NamedTuple):
    """
    Axis-aligned bounding box in normalized image coordinates.

    Attributes:
        xmin (float): Left coordinate relative to image width [0, 1].
        ymin (float): Top coordinate relative to image height [0, 1].
        width (float): Box width relative to image width.
        height (float): Box height relative to image height.
    """
    xmin: float
    ymin: float
    width: float
    height: float


class KeyPoint(NamedTuple):
    """
    Normalized 2D keypoint.

    Attributes:
        x (float): Horizontal coordinate relative to image width [0, 1].
        y (float): Vertical coordinate relative to image height [0, 1].
    """
    x: float
    y: float


class LocationData(NamedTuple):
    """
    Spatial information associated with a detection.

    Attributes:
        relative_bounding_box (BoundingBox): Face bounding box.
        relative_keypoints (list[KeyPoint]): Key landmarks used for alignment.
    """
    relative_bounding_box: BoundingBox
    relative_keypoints: list[KeyPoint]


class Detection(NamedTuple):
    """
    Face detection representation used by this pipeline.

    Attributes:
        label_id (int): Identifier of the detected class.
        score (list[float]): Confidence score(s) associated with detection.
        location_data (LocationData): Spatial metadata.
    """
    label_id: int
    score: list[float]
    location_data: LocationData


class DetectionResults(NamedTuple):
    """
    Container for multiple detection objects.

    Attributes:
        detections (list[Detection]): Detected face instances.
    """
    detections: list[Detection]



# Face Alignment
def rotate_and_crop(frame_bgr, bbox, left_eye, right_eye, out_size=112, margin=0.25):
    """
    Align and normalize a detected face region.

    This function performs geometric normalization of a detected face by:
        1. Expanding the bounding box by a configurable margin
        2. Cropping the region of interest
        3. Computing in-plane rotation based on eye coordinates
        4. Rotating the crop to horizontally align the eyes
        5. Resizing to a fixed square resolution
        6. Applying a final flip transformation (as defined by project design)

    Args:
        frame_bgr (np.ndarray):
            Original input frame in BGR format.
        bbox (tuple[int, int, int, int]):
            Face bounding box in pixel coordinates (x, y, width, height).
        left_eye (tuple[int, int]):
            Pixel coordinates of the left eye.
        right_eye (tuple[int, int]):
            Pixel coordinates of the right eye.
        out_size (int, optional):
            Output image size (square). Defaults to 112.
        margin (float, optional):
            Fractional expansion applied to bounding box size.
            Defaults to 0.25.

    Returns:
        np.ndarray | None:
            Aligned face image of shape (out_size, out_size, 3),
            or None if cropping fails (e.g., invalid region).

    Purpose in Pipeline:
        Ensures all faces passed to embedding models share:
            - Consistent scale
            - Consistent orientation
            - Reduced background variance
    """
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox

    cx, cy = x + bw/2, y + bh/2
    size = max(bw, bh) * (1.0 + margin)

    x1 = int(max(0, cx - size/2))
    y1 = int(max(0, cy - size/2))
    x2 = int(min(w, cx + size/2))
    y2 = int(min(h, cy + size/2))

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    lx, ly = left_eye
    rx, ry = right_eye
    angle = np.degrees(np.arctan2((ry - ly), (rx - lx)))

    ch, cw = crop.shape[:2]
    M = cv2.getRotationMatrix2D((cw/2, ch/2), angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR)

    return cv2.flip(
        cv2.flip(
            cv2.resize(rotated, (out_size, out_size), interpolation=cv2.INTER_LINEAR),
            0
        ),
        1
    )



# Face Detection Wrapper
def detect_bbox_and_eyes(
    frame_bgr: cv2.typing.MatLike,
    detector: face_detection.FaceDetection,
    min_conf=0.6
):
    """
    Detect the most confident face in a frame and extract alignment landmarks.

    This function:
        - Converts the frame to RGB
        - Runs the configured face detector
        - Selects the highest-confidence detection
        - Converts normalized coordinates to pixel space
        - Returns bounding box + eye keypoints

    Args:
        frame_bgr (cv2.typing.MatLike):
            Input image in BGR format.
        detector (face_detection.FaceDetection):
            Initialized face detection model instance.
        min_conf (float, optional):
            Minimum acceptable detection confidence. Defaults to 0.6.

    Returns:
        tuple | None:
            ((x, y, bw, bh), left_eye, right_eye, score)
            or None if:
                - No face is detected
                - Confidence is below threshold
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res: DetectionResults = DetectionResults(
        detections=detector.process(rgb).detections
    )

    if not res.detections:
        return None

    best = max(res.detections, key=lambda d: d.score[0])
    if best.score[0] < min_conf:
        return None

    bb = best.location_data.relative_bounding_box
    x = int(bb.xmin * w)
    y = int(bb.ymin * h)
    bw = int(bb.width * w)
    bh = int(bb.height * h)

    kps = best.location_data.relative_keypoints
    right_eye = (int(kps[0].x * w), int(kps[0].y * h))
    left_eye  = (int(kps[1].x * w), int(kps[1].y * h))

    return (x, y, bw, bh), left_eye, right_eye, best.score[0]
