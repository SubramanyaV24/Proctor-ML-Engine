"""
Object and body detection using YOLO (primary) and Haar cascades (fallback).

YOLO inference is run once per frame and the results are split into
object detections (phone, laptop, …) and body/person detections.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from proctor_ml.config import (
    OBJECT_CONFIDENCE_THRESHOLDS,
    PHONE_MIN_AREA_FRAC,
    PHONE_MAX_ASPECT,
    YOLO_PERSON_CONFIDENCE,
    YOLO_SECONDARY_PERSON_CONFIDENCE,
    YOLO_SECONDARY_IOU_MAX,
    YOLO_NMS_IOU,
)
from proctor_ml.models.loader import Models


# ---------------------------------------------------------------------------
# Suspicious object detection
# ---------------------------------------------------------------------------
def _is_plausible_phone(x1: float, y1: float, x2: float, y2: float,
                         img_h: int, img_w: int) -> bool:
    """Sanity-check a phone bounding box (area + aspect ratio)."""
    box_w = x2 - x1
    box_h = y2 - y1
    if (box_w * box_h) / max(1, img_h * img_w) < PHONE_MIN_AREA_FRAC:
        return False
    aspect = box_w / max(1.0, box_h)
    if aspect > PHONE_MAX_ASPECT or aspect < (1.0 / PHONE_MAX_ASPECT):
        return False
    return True


def run_yolo_once(image: np.ndarray, models: Models) -> Tuple[List[dict], List[dict], List[dict]]:
    """Run YOLO inference once and return (objects, bodies, secondary_bodies).

    Splitting the single inference result avoids calling YOLO twice per frame,
    which was the primary cause of high latency.

    Returns:
        objects:          List of suspicious-object dicts {label, confidence, x, y, w, h}
        bodies:           High-confidence person detections (>= YOLO_PERSON_CONFIDENCE)
        secondary_bodies: Lower-confidence person detections (YOLO_SECONDARY_PERSON_CONFIDENCE
                          .. YOLO_PERSON_CONFIDENCE) that are spatially distinct from all
                          primary bodies.  Used to catch partial/occluded background people.
    """
    if models.yolo is None:
        return [], [], []

    try:
        # Pass a low threshold (0.25) so YOLO outputs all candidates (phones, books).
        # Our Python loop below will strictly filter them using the config thresholds.
        yolo_results = models.yolo(
            image,
            verbose=False,
            conf=0.25,
            iou=YOLO_NMS_IOU,
            imgsz=320,
        )
    except Exception as exc:
        print(f"[Object] YOLO inference error: {exc}")
        return [], [], []

    img_h, img_w = image.shape[:2]
    detected_objects: List[dict] = []
    raw_primary_boxes: List[tuple] = []   # conf >= YOLO_PERSON_CONFIDENCE
    raw_secondary_boxes: List[tuple] = [] # YOLO_SECONDARY_PERSON_CONFIDENCE <= conf < YOLO_PERSON_CONFIDENCE

    for result in yolo_results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = models.yolo.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # --- suspicious objects (only at normal confidence) ---
            if label in OBJECT_CONFIDENCE_THRESHOLDS:
                if conf >= OBJECT_CONFIDENCE_THRESHOLDS[label]:
                    if label == "cell phone" and not _is_plausible_phone(x1, y1, x2, y2, img_h, img_w):
                        print(
                            f"[Object] Rejected implausible phone (conf={conf:.2f}, "
                            f"box={int(x2-x1)}x{int(y2-y1)}, "
                            f"area%={100*(x2-x1)*(y2-y1)/(img_h*img_w):.1f}%)"
                        )
                        continue
                    detected_objects.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "x": int(x1), "y": int(y1),
                        "w": int(x2 - x1), "h": int(y2 - y1),
                    })

            # --- persons ---
            if label == "person":
                if conf >= YOLO_PERSON_CONFIDENCE:
                    raw_primary_boxes.append((conf, int(x1), int(y1), int(x2), int(y2)))
                elif conf >= YOLO_SECONDARY_PERSON_CONFIDENCE:
                    raw_secondary_boxes.append((conf, int(x1), int(y1), int(x2), int(y2)))

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

    # Secondary: keep only boxes that don't heavily overlap any primary body
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
    """Detect suspicious objects (cell phone, laptop, book, tv).

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

    # Haar fallback (only when YOLO is unavailable)
    if models.yolo is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for cascade, body_type in [
            (models.haar_upper, "upper_body"),
            (models.haar_full, "full_body"),
        ]:
            if cascade is None:
                continue
            detections = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
            )
            for (x, y, w, h) in detections:
                all_bodies.append({"type": body_type, "x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return all_bodies
