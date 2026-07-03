"""
Face detectors: MediaPipe (primary), YuNet (secondary), Haar cascade (fallback).

All detectors return a uniform list of dicts:
    {x, y, w, h, score, landmarks: {right_eye, left_eye, nose, right_mouth, left_mouth}}
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from proctor_ml.config import FACE_SCORE_MIN
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
# Unified entry point
# ---------------------------------------------------------------------------
def detect_faces(image: np.ndarray, models: Models) -> tuple[List[dict], str]:
    """Run face detection using the best available detector.

    Returns (faces, detector_name).
    """
    if models.mediapipe_face is not None:
        return detect_faces_mediapipe(image, models), "mediapipe"
    if models.yunet is not None:
        return detect_faces_yunet(image, models), "yunet"
    return detect_faces_haar(image, models), "haar"
