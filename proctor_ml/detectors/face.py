"""
Face detectors: MediaPipe (primary), YuNet (secondary), Haar cascade (fallback).

All detectors return a uniform list of dicts:
    {x, y, w, h, score, landmarks: {right_eye, left_eye, nose, right_mouth, left_mouth}}
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from proctor_ml.config import (
    FACE_EDGE_MARGIN_FRAC,
    FACE_IOU_DEDUP_THRESHOLD,
    FACE_MIN_AREA_FRAC,
    FACE_SCORE_MIN,
)
from proctor_ml.models.loader import Models


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------
def detect_faces_mediapipe(image: np.ndarray, models: Models) -> List[dict]:
    if models.mediapipe_face is None:
        return []

    # MediaPipe accepts BGR directly — no conversion needed
    results = models.mediapipe_face.process(image)
    if not results.detections:
        return []

    h_img, w_img = image.shape[:2]
    faces = []
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w_img)
        y = int(bbox.ymin * h_img)
        w = int(bbox.width * w_img)
        h = int(bbox.height * h_img)
        score = float(det.score[0]) if det.score else 1.0
        if score < FACE_SCORE_MIN:
            continue

        landmarks: dict = {}
        kps = det.location_data.relative_keypoints
        if kps and len(kps) >= 4:
            landmarks["right_eye"]   = (int(kps[0].x * w_img), int(kps[0].y * h_img))
            landmarks["left_eye"]    = (int(kps[1].x * w_img), int(kps[1].y * h_img))
            landmarks["nose"]        = (int(kps[2].x * w_img), int(kps[2].y * h_img))
            # MediaPipe only has one mouth keypoint — duplicate for format compat
            mouth = (int(kps[3].x * w_img), int(kps[3].y * h_img))
            landmarks["right_mouth"] = mouth
            landmarks["left_mouth"]  = mouth

        faces.append({"x": x, "y": y, "w": w, "h": h, "score": score, "landmarks": landmarks})
    return faces


# ---------------------------------------------------------------------------
# YuNet
# ---------------------------------------------------------------------------
def detect_faces_yunet(image: np.ndarray, models: Models) -> List[dict]:
    if models.yunet is None:
        return []

    h, w = image.shape[:2]
    models.yunet.setInputSize((w, h))
    _, raw = models.yunet.detect(image)
    if raw is None:
        return []

    faces = []
    for face in raw:
        x, y, w_b, h_b, x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm, score = face
        if float(score) < FACE_SCORE_MIN:
            continue
        faces.append({
            "x": max(0, int(x)), "y": max(0, int(y)),
            "w": int(w_b), "h": int(h_b),
            "score": float(score),
            "landmarks": {
                "right_eye":   (int(x_re), int(y_re)),
                "left_eye":    (int(x_le), int(y_le)),
                "nose":        (int(x_n),  int(y_n)),
                "right_mouth": (int(x_rm), int(y_rm)),
                "left_mouth":  (int(x_lm), int(y_lm)),
            },
        })
    return faces


# ---------------------------------------------------------------------------
# Haar cascade fallback
# ---------------------------------------------------------------------------
def detect_faces_haar(image: np.ndarray, models: Models) -> List[dict]:
    if models.haar_face is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = models.haar_face.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return [
        {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": 1.0, "landmarks": {}}
        for (x, y, w, h) in detections
    ]


# ---------------------------------------------------------------------------
# Post-detection filtering  (false-positive reduction)
# ---------------------------------------------------------------------------
def _iou(a: dict, b: dict) -> float:
    """Intersection-over-Union for two {x, y, w, h} dicts."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def filter_faces(faces: List[dict], image_shape: tuple) -> List[dict]:
    """Remove spurious detections that are likely reflections or duplicates.

    Three passes (all thresholds are env-var tunable via config):

    1. **Area filter** — faces whose box area is less than
       ``FACE_MIN_AREA_FRAC × frame_area`` are discarded.  Genuine candidate
       faces occupy a meaningful fraction of the frame; tiny blobs are
       reflections or background noise.

    2. **Edge-margin filter** — faces that touch within
       ``FACE_EDGE_MARGIN_FRAC`` of any image border are discarded.  Monitor
       reflections typically appear near frame edges; the real candidate is
       near-centre.

    3. **IoU deduplication** — when two boxes overlap above
       ``FACE_IOU_DEDUP_THRESHOLD``, the lower-score one is removed.  This
       collapses double-detections of a single real face from overlapping
       anchor windows (common with MediaPipe at short ranges).
    """
    if not faces:
        return faces

    ih, iw = image_shape[:2]
    frame_area = max(1, ih * iw)
    margin_x = iw * FACE_EDGE_MARGIN_FRAC
    margin_y = ih * FACE_EDGE_MARGIN_FRAC
    min_area = frame_area * FACE_MIN_AREA_FRAC

    # Pass 1 & 2 — area and edge-margin
    kept: List[dict] = []
    for f in faces:
        if f["w"] * f["h"] < min_area:
            continue
        # Removed edge margin rejection: models are robust to cropped faces,
        # and sitting close to the webcam often crops the chin.
        kept.append(f)

    # Pass 3 — IoU deduplication (greedy, highest-score first)
    kept.sort(key=lambda f: f["score"], reverse=True)
    deduped: List[dict] = []
    for candidate in kept:
        if all(_iou(candidate, accepted) < FACE_IOU_DEDUP_THRESHOLD
               for accepted in deduped):
            deduped.append(candidate)

    return deduped


# ---------------------------------------------------------------------------
# SCRFD (InsightFace — highest accuracy, opt-in)
# ---------------------------------------------------------------------------
def detect_faces_scrfd(image: np.ndarray, models: Models) -> List[dict]:
    """Run SCRFD face detection via the SCRFDDetector ONNX wrapper.

    SCRFD-10GF achieves 83.05% WIDER FACE Hard AP.  Only active when
    ``SCRFD_ENABLED=1`` is set in the environment and the model loaded
    successfully (models.scrfd is not None).
    """
    if models.scrfd is None:
        return []
    try:
        return models.scrfd.detect(image)
    except Exception as exc:
        print(f"[Face] SCRFD inference error: {exc}")
        return []


def detect_faces(image: np.ndarray, models: Models) -> tuple[List[dict], str]:
    """Run face detection using the best available detector.

    Priority ladder (highest accuracy first):
      1. SCRFD      — InsightFace ONNX, 83.05 % WIDER FACE Hard AP (opt-in via SCRFD_ENABLED=1)
      2. MediaPipe  — fast, good for near-frontal webcam faces
      3. YuNet      — 81.1 % Hard AP, CPU-optimised via cv2.FaceDetectorYN
      4. Haar       — legacy fallback

    Returns (faces, detector_name).  All raw detections are passed through
    ``filter_faces`` to remove reflection artefacts and duplicate boxes.
    """
    if models.scrfd is not None:
        raw, name = detect_faces_scrfd(image, models), "scrfd"
    elif models.mediapipe_face is not None:
        raw, name = detect_faces_mediapipe(image, models), "mediapipe"
    elif models.yunet is not None:
        raw, name = detect_faces_yunet(image, models), "yunet"
    else:
        raw, name = detect_faces_haar(image, models), "haar"
    return filter_faces(raw, image.shape), name
