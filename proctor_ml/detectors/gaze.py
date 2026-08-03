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
    GAZE_GLARE_EXTRA_TOLERANCE,
    GAZE_CONFIRMATION_REQUIRED,
    GAZE_CONFIRMATION_MARGIN,
    GLASS_EYE_ROI_HALF,
    GLASSES_GLARE_SATURATION_MAX,
    GLASSES_GLARE_VALUE_MIN,
)
from proctor_ml.models.loader import Models


def estimate_head_pose_6drepnet(face_crop: np.ndarray, models: Models) -> Optional[dict]:
    """6DRepNet head pose. Returns {pitch, yaw, roll} or None.

    6DRepNet (Hempel et al., ICIP 2022) regresses full 3D head rotation from
    the face image. It does NOT use eye/iris landmarks, making it robust to
    eyeglass glare. MAE: 3.97 deg on AFLW2000, 3.54 deg on BIWI.
    """
    if models.head_pose is None:
        return None
    try:
        pitch, yaw, roll = models.head_pose.predict(face_crop)
        return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}
    except Exception as exc:
        print(f"[Gaze] 6DRepNet error: {exc}")
        return None


def estimate_gaze_l2cs(face_crop: np.ndarray, models: Models) -> Optional[dict]:
    """L2CS-Net appearance-based gaze estimation. Returns {pitch, yaw} or None.

    L2CS-Net (Abdelrahman et al., arXiv:2203.03339) is a ResNet-50 CNN trained
    on Gaze360 that regresses gaze from the full face crop.  It does not depend
    on iris/pupil landmarks so it is robust to eyeglass glare.

    Reported accuracy:
      - 3.92 deg mean angular error on MPIIGaze
      - 10.41 deg mean angular error on Gaze360
    """
    if models.l2cs_gaze is None:
        return None
    if face_crop is None or face_crop.size == 0:
        return None
    try:
        result = models.l2cs_gaze.predict(face_crop)
        if result is None:
            return None
        pitch_deg, yaw_deg = result
        return {"pitch": float(pitch_deg), "yaw": float(yaw_deg)}
    except Exception as exc:
        print(f"[Gaze] L2CS-Net error: {exc}")
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
    """Return (pose_dict, model_used_label).

    If eyeglass glare is detected on the landmark positions and the neural
    head-pose model (6DRepNet) is unavailable, the geometric fallback is
    suppressed — its eye-nose triangle computation yields unreliable angles
    when the pupil centre is hidden by a specular reflection.  The caller
    receives (None, "geometric_suppressed_glare") in that case.
    """
    pose = estimate_head_pose_6drepnet(face_crop, models)
    if pose is not None:
        return pose, "6drepnet"
    # Geometric fallback: only use if landmarks are not corrupted by glare
    if detect_glasses_glare(face_crop, landmarks):
        return None, "geometric_suppressed_glare"
    pose = estimate_head_pose_geometric(face_crop, landmarks)
    return pose, "geometric"


def detect_glasses_glare(face_crop: np.ndarray, landmarks: dict) -> bool:
    """Heuristic: detect specular reflections (glare) on eyeglass lenses.

    Samples a small HSV ROI centred on each eye landmark.  A glare hotspot
    has very high Value (bright) and very low Saturation (washed-out / white),
    which is characteristic of specular reflection on a glass surface.

    Returns True when at least one eye shows probable glare.
    """
    if not landmarks:
        return False

    h_crop, w_crop = face_crop.shape[:2]
    if h_crop == 0 or w_crop == 0:
        return False

    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    half = GLASS_EYE_ROI_HALF

    for eye_key in ("left_eye", "right_eye"):
        pt = landmarks.get(eye_key)
        if not pt:
            continue
        ex, ey = int(pt[0]), int(pt[1])
        x1 = max(0, ex - half)
        y1 = max(0, ey - half)
        x2 = min(w_crop, ex + half)
        y2 = min(h_crop, ey + half)
        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        mean_s = float(np.mean(roi[:, :, 1]))  # Saturation channel
        mean_v = float(np.mean(roi[:, :, 2]))  # Value channel
        if mean_s < GLASSES_GLARE_SATURATION_MAX and mean_v > GLASSES_GLARE_VALUE_MIN:
            return True

    return False


def is_looking_away(
    smoothed_yaw: float,
    smoothed_pitch: float,
    glare: bool = False,
    l2cs_gaze: Optional[dict] = None,
) -> bool:
    """Return True when the smoothed gaze angles indicate the candidate is
    looking away from the screen.

    Decision logic:
    1. Apply glare tolerance buffer to thresholds if eyeglass glare detected.
    2. Check whether head pose (smoothed 6DRepNet output) exceeds thresholds.
       If it doesn't, immediately return False — head pose is glasses-immune
       and is the primary, most reliable signal.
    3. If GAZE_CONFIRMATION_REQUIRED=True and L2CS gaze is available, require
       the L2CS gaze also to exceed (threshold - GAZE_CONFIRMATION_MARGIN).
       This dual-signal gate eliminates false positives where head pose briefly
       tips over the threshold but the candidate's actual eye gaze is still
       directed at the screen.

    Args:
        smoothed_yaw:   EMA-smoothed yaw from 6DRepNet (degrees).
        smoothed_pitch: EMA-smoothed pitch from 6DRepNet (degrees).
        glare:          True when eyeglass glare was detected on this frame.
        l2cs_gaze:      Optional dict {pitch, yaw} from L2CS-Net (degrees).

    Returns:
        True if the candidate is considered to be looking away.
    """
    extra = GAZE_GLARE_EXTRA_TOLERANCE if glare else 0.0
    yaw_thr   = GAZE_YAW_THRESHOLD   + extra
    pitch_thr = GAZE_PITCH_THRESHOLD + extra

    # --- Primary check: head pose ---
    head_pose_away = (
        abs(smoothed_yaw)   > yaw_thr
        or abs(smoothed_pitch) > pitch_thr
    )
    if not head_pose_away:
        return False  # Head pose says facing screen — trust it

    # --- Dual-signal confirmation via L2CS-Net ---
    if GAZE_CONFIRMATION_REQUIRED and l2cs_gaze is not None:
        # L2CS must agree that gaze is away by at least (threshold - margin)
        confirm_yaw_thr   = yaw_thr   - GAZE_CONFIRMATION_MARGIN
        confirm_pitch_thr = pitch_thr - GAZE_CONFIRMATION_MARGIN
        gaze_confirms = (
            abs(l2cs_gaze["yaw"])   > confirm_yaw_thr
            or abs(l2cs_gaze["pitch"]) > confirm_pitch_thr
        )
        if not gaze_confirms:
            return False  # Head pose tipped over but gaze says screen — suppress FP

    return True


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
