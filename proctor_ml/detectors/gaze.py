"""
Head-pose / gaze estimation.

Primary:  6DRepNet  (neural, accurate)
Fallback: geometric (eye-nose-mouth triangle, fast, no extra model)
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from proctor_ml.config import (
    FACE_COVERED_SCORE_MIN,
    FACE_COVERED_STD_MIN,
    FACE_COVERED_NOSE_MOUTH_RATIO,
    GAZE_YAW_THRESHOLD,
    GAZE_PITCH_THRESHOLD,
)
from proctor_ml.models.loader import Models


def estimate_head_pose_6drepnet(face_crop: np.ndarray, models: Models) -> Optional[dict]:
    """6DRepNet head pose. Returns {pitch, yaw, roll} or None."""
    if models.head_pose is None:
        return None
    try:
        pitch, yaw, roll = models.head_pose.predict(face_crop)
        return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}
    except Exception as exc:
        print(f"[Gaze] 6DRepNet error: {exc}")
        return None


def estimate_head_pose_geometric(face_crop: np.ndarray, landmarks: dict) -> Optional[dict]:
    """Geometric fallback: estimate pose from the eye–nose triangle."""
    if not landmarks:
        return None

    le = landmarks.get("left_eye")
    re = landmarks.get("right_eye")
    nose = landmarks.get("nose")
    if not (le and re and nose):
        return None

    h, w = face_crop.shape[:2]
    if w == 0 or h == 0:
        return None

    eye_cx = (le[0] + re[0]) / 2
    eye_cy = (le[1] + re[1]) / 2  # noqa: F841

    dx = le[0] - re[0]
    dy = le[1] - re[1]
    roll = np.degrees(np.arctan2(dy, dx))

    ied = max(1.0, float(np.sqrt(dx * dx + dy * dy)))
    nose_dx = (nose[0] - eye_cx) / ied
    nose_dy = (nose[1] - (le[1] + re[1]) / 2) / ied

    yaw = nose_dx * 60
    pitch = -nose_dy * 50

    return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


def estimate_head_pose(face_crop: np.ndarray, landmarks: dict, models: Models) -> tuple[Optional[dict], str]:
    """Return (pose_dict, model_used_label)."""
    pose = estimate_head_pose_6drepnet(face_crop, models)
    if pose is not None:
        return pose, "6drepnet"
    pose = estimate_head_pose_geometric(face_crop, landmarks)
    return pose, "geometric"


def is_looking_away(smoothed_yaw: float, smoothed_pitch: float) -> bool:
    return abs(smoothed_yaw) > GAZE_YAW_THRESHOLD or abs(smoothed_pitch) > GAZE_PITCH_THRESHOLD


def check_face_covered(face_crop: np.ndarray, landmarks: dict, score: float) -> bool:
    """Heuristic: return True if the face appears partially obscured."""
    if score < FACE_COVERED_SCORE_MIN:
        return True

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < FACE_COVERED_STD_MIN:
        return True

    if landmarks:
        le_x, le_y = landmarks.get("left_eye", (0, 0))
        re_x, re_y = landmarks.get("right_eye", (0, 0))
        dx = le_x - re_x
        dy = le_y - re_y
        iod = max(1.0, float(np.sqrt(dx * dx + dy * dy)))

        nose_y  = landmarks.get("nose", (0, 0))[1]
        rm_y    = landmarks.get("right_mouth", (0, 0))[1]
        lm_y    = landmarks.get("left_mouth", (0, 0))[1]
        mouth_y = (rm_y + lm_y) / 2

        if (mouth_y - nose_y) < iod * FACE_COVERED_NOSE_MOUTH_RATIO:
            return True

    return False
