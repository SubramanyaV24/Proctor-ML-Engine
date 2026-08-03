"""
Object and body detection using YOLO (primary) and Haar cascades (fallback).

YOLO inference is run once per frame and the results are split into
object detections (phone, laptop, ...) and body/person detections.

Two-stage mobile phone verification pipeline
============================================

    Camera
          |
        YOLO11
          |
    phone candidate  (label == "cell phone", YOLO conf >= 0.25 pass-through)
          |
    geometry filter  (_is_plausible_phone — area + aspect ratio sanity check)
          |
    confidence gate  (PHONE_DETECTION_THRESHOLD = 0.60)
          |
    PhoneVerifier    (MobileNetV3-Small, PHONE / NOT_PHONE, <10 ms on CPU)
          |
    Boolean smoother
          |
    API response  ->  Frontend alert

Only verified phones reach the Boolean smoother and are emitted as
``phoneDetected=true`` in the API response.

"remote" class
--------------
The ``"remote"`` class is NO LONGER accepted as a phone shortcut.  YOLO
frequently confuses rectangular objects (chargers, wall sockets, remotes) with
the remote class, which inflated false positives.

To re-enable (not recommended): set ``PHONE_VERIFIER_REMOTE_ENABLED=True``.
Remote candidates are then gated at ``PHONE_VERIFIER_REMOTE_CONF`` (default
0.85) AND must pass PhoneVerifier before being emitted as ``"cell phone"``.

Logging
-------
Enable DEBUG-level logging on this logger to see per-candidate logs::

    YOLO candidate: label=cell phone confidence=0.67 box=48x92 area%=1.4%
    PhoneVerifier: verified=False confidence=0.18 reason=not_phone

Debug logging is off by default (INFO level) to keep prod logs clean.

See Also
--------
proctor_ml/verifiers/phone_verifier.py — PhoneVerifier implementation.
proctor_ml/config.py                   — PHONE_* configuration constants.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from proctor_ml.config import (
    OBJECT_CONFIDENCE_THRESHOLDS,
    PHONE_MIN_AREA_FRAC,
    PHONE_MAX_ASPECT,
    PHONE_DETECTION_THRESHOLD,
    PHONE_VERIFIER_ENABLED,
    YOLO_PERSON_CONFIDENCE,
    YOLO_SECONDARY_PERSON_CONFIDENCE,
    YOLO_SECONDARY_IOU_MAX,
    YOLO_NMS_IOU,
)
from proctor_ml.models.loader import Models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote-as-phone path (opt-in, disabled by default to reduce false positives)
# ---------------------------------------------------------------------------
_REMOTE_ENABLED: bool = os.getenv("PHONE_VERIFIER_REMOTE_ENABLED", "False").lower() == "true"
_REMOTE_CONF: float   = float(os.getenv("PHONE_VERIFIER_REMOTE_CONF", "0.85"))


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------
def _is_plausible_phone(x1: float, y1: float, x2: float, y2: float,
                         img_h: int, img_w: int) -> bool:
    """Sanity-check a phone bounding box (area fraction + aspect ratio).

    Returns False for:
    * Boxes whose area is smaller than PHONE_MIN_AREA_FRAC of the frame
      (catches spurious detections in the distance or on tiny reflections).
    * Boxes with extreme aspect ratios (wider than PHONE_MAX_ASPECT:1 or
      taller than 1:PHONE_MAX_ASPECT) — phones are roughly portrait/landscape
      rectangles, not thin slivers.
    """
    box_w = x2 - x1
    box_h = y2 - y1
    if (box_w * box_h) / max(1, img_h * img_w) < PHONE_MIN_AREA_FRAC:
        return False
    aspect = box_w / max(1.0, box_h)
    if aspect > PHONE_MAX_ASPECT or aspect < (1.0 / PHONE_MAX_ASPECT):
        return False
    return True


# ---------------------------------------------------------------------------
# Second-stage verifier helper
# ---------------------------------------------------------------------------
def _apply_phone_verifier(
    image: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    img_h: int, img_w: int,
    yolo_conf: float,
    label: str,
    verifier,                 # proctor_ml.verifiers.phone_verifier.PhoneVerifier | None
) -> bool:
    """Crop the YOLO box, run PhoneVerifier, emit structured debug logs.

    Returns True if the crop is accepted (or if the verifier is unavailable).
    Only called when PHONE_VERIFIER_ENABLED=True and verifier is loaded.
    """
    if verifier is None:
        return True

    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(img_w, int(x2)), min(img_h, int(y2))
    crop = image[iy1:iy2, ix1:ix2]

    v = verifier.verify(crop)

    # Phase 7 logging: structured per-candidate debug output
    logger.debug(
        "[Object] YOLO candidate\n"
        "  label=%s\n"
        "  confidence=%.2f\n"
        "[PhoneVerifier]\n"
        "  verified=%s\n"
        "  confidence=%.2f\n"
        "  reason=%s",
        label, yolo_conf,
        v.verified, v.confidence, v.reason,
    )

    return v.verified


# ---------------------------------------------------------------------------
# Core YOLO runner
# ---------------------------------------------------------------------------
def run_yolo_once(image: np.ndarray, models: Models) -> Tuple[List[dict], List[dict], List[dict]]:
    """Run YOLO inference once and split results into objects / persons.

    Splitting the single inference result avoids calling YOLO twice per frame,
    which was the primary cause of high latency in earlier versions.

    Phone detection pipeline (executed inside this function):

        YOLO (low 0.25 pass-through conf)
            -> geometry filter     _is_plausible_phone()
            -> confidence gate     PHONE_DETECTION_THRESHOLD (0.60)
            -> PhoneVerifier       MobileNetV3-Small crop classifier
            -> append to objects   (reaches Boolean smoother -> API)

    Returns
    -------
    objects:
        Suspicious-object dicts: {label, confidence, x, y, w, h}.
        Only verified phone detections are included.
    bodies:
        High-confidence person dicts (conf >= YOLO_PERSON_CONFIDENCE).
    secondary_bodies:
        Lower-confidence partial-person dicts (spatially distinct from primary).
    """
    if models.yolo is None:
        return [], [], []

    try:
        # Low pass-through threshold so Python can apply per-label gates.
        # YOLO does not filter here; our loop enforces OBJECT_CONFIDENCE_THRESHOLDS.
        yolo_results = models.yolo(
            image,
            verbose=False,
            conf=0.25,
            iou=YOLO_NMS_IOU,
            imgsz=320,
        )
    except Exception as exc:
        logger.error("[Object] YOLO inference error: %s", exc)
        return [], [], []

    img_h, img_w = image.shape[:2]
    detected_objects: List[dict]  = []
    raw_primary_boxes: List[tuple]    = []   # conf >= YOLO_PERSON_CONFIDENCE
    raw_secondary_boxes: List[tuple]  = []   # YOLO_SECONDARY_PERSON_CONFIDENCE <= conf < primary

    # PhoneVerifier is loaded by the model loader and stored in models.phone_verifier.
    # It is None when PHONE_VERIFIER_ENABLED=False or when loading failed.
    verifier = getattr(models, "phone_verifier", None)

    for result in yolo_results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label  = models.yolo.names[cls_id]
            conf   = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # ----------------------------------------------------------------
            # Suspicious object path
            # ----------------------------------------------------------------
            if label in OBJECT_CONFIDENCE_THRESHOLDS:

                # -- "remote" class: disabled by default ---------------------
                if label == "remote":
                    if not _REMOTE_ENABLED:
                        # Remote-as-phone path is OFF.  Skip entirely.
                        # Allowing "remote" without high confidence + verifier
                        # significantly increases false positives.
                        continue

                    # Remote path (opt-in): very high YOLO conf + verifier required
                    if conf < _REMOTE_CONF:
                        logger.debug(
                            "[Object] Remote discarded: conf=%.2f < threshold=%.2f",
                            conf, _REMOTE_CONF,
                        )
                        continue
                    if not _is_plausible_phone(x1, y1, x2, y2, img_h, img_w):
                        logger.debug(
                            "[Object] Remote rejected by geometry: conf=%.2f box=%dx%d area%%=%.1f%%",
                            conf, int(x2 - x1), int(y2 - y1),
                            100 * (x2 - x1) * (y2 - y1) / (img_h * img_w),
                        )
                        continue
                    if not _apply_phone_verifier(
                        image, x1, y1, x2, y2, img_h, img_w, conf, label, verifier
                    ):
                        continue
                    # Accepted remote-as-phone: emit as "cell phone"
                    detected_objects.append({
                        "label": "cell phone",
                        "confidence": round(conf, 2),
                        "x": int(x1), "y": int(y1),
                        "w": int(x2 - x1), "h": int(y2 - y1),
                    })
                    continue

                # -- cell phone: full three-stage pipeline -------------------
                if label == "cell phone":
                    logger.debug(
                        "[Object] YOLO candidate: label=cell phone confidence=%.2f "
                        "box=%dx%d area%%=%.1f%%",
                        conf, int(x2 - x1), int(y2 - y1),
                        100 * (x2 - x1) * (y2 - y1) / (img_h * img_w),
                    )

                    # Stage 1 -- geometry filter
                    if not _is_plausible_phone(x1, y1, x2, y2, img_h, img_w):
                        logger.debug(
                            "[Object] Phone rejected by geometry: conf=%.2f box=%dx%d area%%=%.1f%%",
                            conf, int(x2 - x1), int(y2 - y1),
                            100 * (x2 - x1) * (y2 - y1) / (img_h * img_w),
                        )
                        continue

                    # Stage 2 -- raised confidence threshold (0.60 vs old ~0.40-0.45)
                    if conf < PHONE_DETECTION_THRESHOLD:
                        logger.debug(
                            "[Object] Phone below confidence threshold: %.2f < %.2f",
                            conf, PHONE_DETECTION_THRESHOLD,
                        )
                        continue

                    # Stage 3 -- PhoneVerifier (second-stage MobileNetV3 classifier)
                    if PHONE_VERIFIER_ENABLED and verifier is not None:
                        if not _apply_phone_verifier(
                            image, x1, y1, x2, y2, img_h, img_w, conf, label, verifier
                        ):
                            continue

                    detected_objects.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "x": int(x1), "y": int(y1),
                        "w": int(x2 - x1), "h": int(y2 - y1),
                    })
                    continue

                # -- all other labels (laptop, book, ...) --------------------
                min_conf = OBJECT_CONFIDENCE_THRESHOLDS[label]
                if conf >= min_conf:
                    detected_objects.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "x": int(x1), "y": int(y1),
                        "w": int(x2 - x1), "h": int(y2 - y1),
                    })

            # ----------------------------------------------------------------
            # Person path (unchanged from original)
            # ----------------------------------------------------------------
            if label == "person":
                if conf >= YOLO_PERSON_CONFIDENCE:
                    raw_primary_boxes.append((conf, int(x1), int(y1), int(x2), int(y2)))
                elif conf >= YOLO_SECONDARY_PERSON_CONFIDENCE:
                    raw_secondary_boxes.append((conf, int(x1), int(y1), int(x2), int(y2)))

    # -----------------------------------------------------------------------
    # Greedy IoU-NMS for person boxes
    # -----------------------------------------------------------------------
    def _nms(boxes: List[tuple]) -> List[tuple]:
        """Greedy IoU-NMS (highest confidence first)."""
        boxes_sorted = sorted(boxes, key=lambda b: b[0], reverse=True)
        kept: List[tuple] = []
        for cand in boxes_sorted:
            _, cx1, cy1, cx2, cy2 = cand
            suppress = False
            for _, kx1, ky1, kx2, ky2 in kept:
                ix1 = max(cx1, kx1); iy1 = max(cy1, ky1)
                ix2 = min(cx2, kx2); iy2 = min(cy2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (cx2 - cx1) * (cy2 - cy1) + (kx2 - kx1) * (ky2 - ky1) - inter
                if union > 0 and inter / union > YOLO_NMS_IOU:
                    suppress = True
                    break
            if not suppress:
                kept.append(cand)
        return kept

    primary_kept = _nms(raw_primary_boxes)
    bodies = [
        {"type": "yolo_person", "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "confidence": round(conf, 2)}
        for conf, x1, y1, x2, y2 in primary_kept
    ]

    secondary_bodies: List[dict] = []
    for conf, sx1, sy1, sx2, sy2 in raw_secondary_boxes:
        distinct = True
        for _, px1, py1, px2, py2 in primary_kept:
            ix1 = max(sx1, px1); iy1 = max(sy1, py1)
            ix2 = min(sx2, px2); iy2 = min(sy2, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (sx2 - sx1) * (sy2 - sy1) + (px2 - px1) * (py2 - py1) - inter
            if union > 0 and inter / union > YOLO_SECONDARY_IOU_MAX:
                distinct = False
                break
        if distinct:
            secondary_bodies.append({
                "type": "yolo_person_partial",
                "x": sx1, "y": sy1, "w": sx2 - sx1, "h": sy2 - sy1,
                "confidence": round(conf, 2),
            })

    return detected_objects, bodies, secondary_bodies


def detect_objects(image: np.ndarray, models: Models) -> List[dict]:
    """Detect suspicious objects (cell phone, laptop, book).

    Prefer calling run_yolo_once() from the pipeline to avoid double inference.
    Returns list of dicts: {label, confidence, x, y, w, h}.
    """
    objects, _, _sec = run_yolo_once(image, models)
    return objects


# ---------------------------------------------------------------------------
# Body / person detection
# ---------------------------------------------------------------------------
def detect_bodies(image: np.ndarray, models: Models) -> List[dict]:
    """Detect persons.  YOLO primary; Haar cascade fallback.

    Prefer calling run_yolo_once() from the pipeline to avoid double inference.
    Returns list of dicts: {type, x, y, w, h, confidence?}.
    """
    _, bodies, secondary = run_yolo_once(image, models)
    all_bodies = bodies + secondary

    # Haar cascade fallback — only when YOLO is entirely unavailable
    if models.yolo is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for cascade, body_type in [
            (models.haar_upper, "upper_body"),
            (models.haar_full,  "full_body"),
        ]:
            if cascade is None:
                continue
            detections = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
            )
            for (x, y, w, h) in detections:
                all_bodies.append({"type": body_type, "x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return all_bodies
