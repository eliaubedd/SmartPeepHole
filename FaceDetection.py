import time
from typing import NamedTuple, Any
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_detection
import wespeakerruntime as wespeaker
from os import listdir, path
import pyaudio
import wave
from io import BytesIO

class BoundingBox(NamedTuple):
    xmin: float
    ymin: float
    width: float
    height: float

class KeyPoint(NamedTuple):
    x: float
    y: float

class LocationData(NamedTuple):
    relative_bounding_box: BoundingBox
    relative_keypoints: list[KeyPoint]

class Detection(NamedTuple):
    label_id: int
    score: list[float]
    location_data: LocationData

class DetectionResults(NamedTuple):
    detections: list[Detection]

def rotate_and_crop(frame_bgr, bbox, left_eye, right_eye, out_size=112, margin=0.25):
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

    return cv2.flip(cv2.flip(cv2.resize(rotated, (out_size, out_size), interpolation=cv2.INTER_LINEAR), 0), 1)



def detect_bbox_and_eyes(frame_bgr: cv2.typing.MatLike, detector: face_detection.FaceDetection, min_conf=0.6):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res: DetectionResults = DetectionResults(detections=detector.process(rgb).detections)

    if not res.detections:
        return None

    best = max(res.detections, key=lambda d: d.score[0])
    if best.score[0] < min_conf:
        return None

    #print(list(map(lambda d: (d.label_id, d.score), res.detections)))

    bb = best.location_data.relative_bounding_box
    x = int(bb.xmin * w)
    y = int(bb.ymin * h)
    bw = int(bb.width * w)
    bh = int(bb.height * h)

    # keypoints: right_eye, left_eye, nose, mouth, right_ear, left_ear
    kps = best.location_data.relative_keypoints
    right_eye = (int(kps[0].x * w), int(kps[0].y * h))
    left_eye  = (int(kps[1].x * w), int(kps[1].y * h))

    return (x, y, bw, bh), left_eye, right_eye, best.score[0]
        