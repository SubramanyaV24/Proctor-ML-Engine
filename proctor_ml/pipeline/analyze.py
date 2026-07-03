"""
Main inference pipeline — orchestrates detectors into a single result dict.

This module has zero knowledge of sessions, trust scores, Frappe, or any
application-level concept. It takes raw media, runs models, and returns
structured detection results.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

from proctor_ml.detectors.audio import detect_speech
from proctor_ml.detectors.face import detect_faces
from proctor_ml.detectors.gaze import (
    check_face_covered,
    estimate_head_pose,
    is_looking_away,
)
from proctor_ml.detectors.object import detect_bodies, detect_objects, run_yolo_once
from proctor_ml.models.loader import Models
from proctor_ml.pipeline.smoother import SESSION_STATES
from proctor_ml.preprocess.audio import decode_base64_audio
from proctor_ml.preprocess.image import decode_base64_image
from proctor_ml.utils.logger import get_session_logger


def analyze_image(image_b64: str, session_id: str, models: Models) -> Dict:
    """Run the full vision pipeline on a single base64-encoded frame.

    Args:
        image_b64:  Base64-encoded image (data-URI or raw).
        session_id: Opaque string used to look up per-session smoothers.
        models:     Loaded model container from `load_all_models()`.

    Returns:
        A dict of detection results (see inline keys below).
    """
    state = SESSION_STATES[session_id]
    logger = get_session_logger(session_id)
    t0 = time.time()

    result: Dict = {
        "faceDetected": False,
        "faceCount": 0,
        "multipleFaces": False,
        "multiplePeople": False,
        "faceCovered": False,
        "lookingAway": False,
        "headPose": None,
        "detector_used": None,
        "model": "none",
        "bodyDetected": False,
        "bodyCount": 0,
        "bodyBox": None,
        "body_detector": "none",
        "multipleBodies": False,
        "detectedObjects": [],
        "phoneDetected": False,
        "secondaryDeviceDetected": False,
        "bookDetected": False,
        "processingTimeMs": 0,
        "error": None,
    }

    image = decode_base64_image(image_b64)
    if image is None:
        result["error"] = "Failed to decode image"
        result["processingTimeMs"] = _elapsed(t0)
        logger.error("Failed to decode image")
        return result

    # Cache image dimensions (used multiple times below)
    ih, iw = image.shape[:2]

    # ------------------------------------------------------------------ faces
    faces, detector_name = detect_faces(image, models)
    face_count = len(faces)
    result["faceDetected"] = face_count >= 1
    result["faceCount"] = face_count
    result["detector_used"] = detector_name

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"detector={detector_name} face_count={face_count}")
        for i, f in enumerate(faces):
            logger.debug(f"face[{i}]={f}")

    # ----------------------------------------------------------------- bodies + objects (single YOLO pass)
    detected_objects, bodies = run_yolo_once(image, models)
    body_count = len(bodies)
    result["bodyCount"] = body_count
    result["multipleBodies"] = body_count > 1

    face_present = face_count >= 1
    raw_multi_faces  = face_present and face_count > 1
    raw_multi_people = face_present and (face_count > 1 or body_count > 1)

    result["multipleFaces"]  = state.multiple_faces.update(raw_multi_faces,  logger=logger)
    result["multiplePeople"] = state.multiple_people.update(raw_multi_people, logger=logger)
    result["bodyDetected"]   = body_count >= 1 or face_count >= 1

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"body_count={body_count} multi_faces={result['multipleFaces']} "
                     f"multi_people={result['multiplePeople']}")

    # Cache primary face (used twice below)
    primary_face = max(faces, key=lambda f: f["score"]) if faces else None

    if bodies:
        pb = max(bodies, key=lambda b: b["w"] * b["h"])
        result["bodyBox"] = {**pb, "img_w": iw, "img_h": ih}
        result["body_detector"] = "yolo" if pb["type"] == "yolo_person" else "cascade"
    elif primary_face:
        fx, fy, fw, fh = primary_face["x"], primary_face["y"], primary_face["w"], primary_face["h"]
        bx = int(max(0, fx - fw))
        by = int(min(ih - 1, fy + fh * 0.8))
        bw = int(min(iw - bx, fw * 3.5))
        bh = int(min(ih - by, fh * 4.5))
        result["bodyBox"] = {"x": bx, "y": by, "w": bw, "h": bh,
                              "type": "inferred_body", "img_w": iw, "img_h": ih}
        result["body_detector"] = "inferred"

    # ------------------------------------------------------------ objects (already computed above)
    result["detectedObjects"] = detected_objects

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"objects={len(detected_objects)}")
        for i, obj in enumerate(detected_objects):
            logger.debug(f"object[{i}]={obj}")

    raw_phone = any(o["label"] == "cell phone" for o in detected_objects)
    result["phoneDetected"]          = state.phone_detected.update(raw_phone, logger=logger)
    result["secondaryDeviceDetected"] = False  # hardware/API detection only
    result["bookDetected"]           = any(o["label"] == "book" for o in detected_objects)

    if not faces and not bodies and not detected_objects:
        result["processingTimeMs"] = _elapsed(t0)
        return result

    # ------------------------------------------------------------ primary face
    if primary_face:
        x, y, w, h = primary_face["x"], primary_face["y"], primary_face["w"], primary_face["h"]
        result["faceBox"] = {"x": x, "y": y, "w": w, "h": h, "img_w": iw, "img_h": ih}

        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad);   y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
        face_crop = image[y1:y2, x1:x2]

        pose, model_label = estimate_head_pose(face_crop, primary_face.get("landmarks", {}), models)
        result["model"] = f"{detector_name}+{model_label}"

        if pose:
            yaw_s   = state.gaze_yaw.update(float(pose["yaw"]))
            pitch_s = state.gaze_pitch.update(float(pose["pitch"]))
            result["headPose"] = {**pose, "smoothed_yaw": yaw_s, "smoothed_pitch": pitch_s}
            result["lookingAway"] = is_looking_away(yaw_s, pitch_s)
        else:
            state.gaze_yaw.update(None)
            state.gaze_pitch.update(None)

        result["faceCovered"] = check_face_covered(face_crop, primary_face.get("landmarks", {}), primary_face["score"])
    else:
        state.gaze_yaw.update(None)
        state.gaze_pitch.update(None)

    result["processingTimeMs"] = _elapsed(t0)
    return result


def analyze_audio(audio_b64: str, session_id: str, models: Models) -> Dict:
    """Run VAD on a base64-encoded PCM audio clip.

    Args:
        audio_b64:  Base64-encoded 16-bit mono PCM at 16 kHz.
        session_id: Used for temporal smoothing.
        models:     Loaded model container.

    Returns:
        {"multi_speaker": bool, "speech_confidence": float}
    """
    state = SESSION_STATES[session_id]
    logger = get_session_logger(session_id)

    audio_bytes = decode_base64_audio(audio_b64)
    if audio_bytes is None:
        return {"multi_speaker": False, "speech_confidence": 0.0, "error": "decode_failed"}

    multi, confidence = detect_speech(audio_bytes, models)
    smoothed_multi = state.multi_speaker.update(multi, logger=logger)
    smoothed_conf  = state.speech_confidence.update(confidence)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"audio raw multi={multi} conf={confidence:.2f} "
                     f"→ smoothed multi={smoothed_multi} conf={smoothed_conf:.2f}")

    return {"multi_speaker": smoothed_multi, "speech_confidence": smoothed_conf}


# ---------------------------------------------------------------------------
def _elapsed(t0: float) -> float:
    return round((time.time() - t0) * 1000, 2)
