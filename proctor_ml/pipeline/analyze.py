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
    detect_glasses_glare,
    estimate_gaze_l2cs,
    estimate_head_pose,
    is_looking_away,
)
from proctor_ml.detectors.object import detect_bodies, detect_objects, run_yolo_once
from proctor_ml.models.loader import Models
from proctor_ml.pipeline.smoother import SESSION_STATES
from proctor_ml.preprocess.audio import decode_base64_audio
from proctor_ml.preprocess.image import decode_base64_image
from proctor_ml.utils.logger import get_session_logger


def analyze_image(image_data: str | bytes, session_id: str, models: Models) -> Dict:
    """Run the full vision pipeline on a single base64-encoded or raw binary frame.

    Args:
        image_data: Base64-encoded image (data-URI or raw) or raw bytes.
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
        "glassesGlareDetected": False,
        "headPose": None,
        "gazeEstimate": None,           # L2CS-Net pitch/yaw (degrees)
        "headPoseConfirmedByGaze": None, # True/False when L2CS available
        "faceDetector": None,           # scrfd / mediapipe / yunet / haar
        "detector_used": None,          # kept for back-compat
        "model": "none",
        "bodyDetected": False,
        "bodyCount": 0,
        "bodyBox": None,
        "body_detector": "none",
        "multipleBodies": False,
        "partialBodyDetected": False,
        "detectedObjects": [],
        "phoneDetected": False,
        "secondaryDeviceDetected": False,
        "bookDetected": False,
        "multipleFacesDurationMs": 0,
        "multiplePeopleDurationMs": 0,
        "faceMissingDurationMs": 0,
        "lookAwayDurationMs": 0,
        "phoneDurationMs": 0,
        "processingTimeMs": 0,
        "error": None,
    }

    if isinstance(image_data, bytes):
        from proctor_ml.preprocess.image import decode_binary_image
        image = decode_binary_image(image_data)
    else:
        image = decode_base64_image(image_data)

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
    result["faceDetector"] = detector_name

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"detector={detector_name} face_count={face_count}")
        for i, f in enumerate(faces):
            logger.debug(f"face[{i}]={f}")

    # ----------------------------------------------------------------- bodies + objects (single YOLO pass)
    detected_objects, bodies, secondary_bodies = run_yolo_once(image, models)
    body_count = len(bodies)
    # Include partial/lower-confidence bodies for multi-people detection
    total_body_count = body_count + len(secondary_bodies)
    result["bodyCount"] = body_count
    result["multipleBodies"] = body_count > 1

    face_present = face_count >= 1
    raw_multi_faces  = face_present and face_count > 1
    raw_multi_people = face_present and (face_count > 1 or total_body_count > 1)

    result["multipleFaces"]  = state.multiple_faces.update(raw_multi_faces,  logger=logger)
    result["multiplePeople"] = state.multiple_people.update(raw_multi_people, logger=logger)
    result["bodyDetected"]   = body_count >= 1 or face_count >= 1
    result["partialBodyDetected"] = len(secondary_bodies) > 0
    
    result["multipleFacesDurationMs"] = state.multiple_faces_duration.update(result["multipleFaces"])
    result["multiplePeopleDurationMs"] = state.multiple_people_duration.update(result["multiplePeople"])
    result["faceMissingDurationMs"] = state.face_missing_duration.update(face_count == 0)

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
    result["phoneDurationMs"]        = state.phone_duration.update(result["phoneDetected"])
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

        # --- Detect eyeglass glare on the face crop (uses landmarks) ---
        glare = detect_glasses_glare(face_crop, primary_face.get("landmarks", {}))
        result["glassesGlareDetected"] = glare

        # --- L2CS-Net appearance-based gaze (glasses-robust confirmation) ---
        l2cs_gaze = estimate_gaze_l2cs(face_crop, models)
        if l2cs_gaze is not None:
            result["gazeEstimate"] = l2cs_gaze

        if pose:
            yaw_s   = state.gaze_yaw.update(float(pose["yaw"]))
            pitch_s = state.gaze_pitch.update(float(pose["pitch"]))
            result["headPose"] = {**pose, "smoothed_yaw": yaw_s, "smoothed_pitch": pitch_s}

            looking_away = is_looking_away(
                yaw_s, pitch_s,
                glare=glare,
                l2cs_gaze=l2cs_gaze,
            )
            result["lookingAway"] = looking_away
            result["lookAwayDurationMs"] = state.look_away_duration.update(looking_away)

            # Record whether L2CS confirmed or suppressed the head-pose signal
            if l2cs_gaze is not None:
                from proctor_ml.config import (
                    GAZE_YAW_THRESHOLD, GAZE_PITCH_THRESHOLD,
                    GAZE_GLARE_EXTRA_TOLERANCE, GAZE_CONFIRMATION_MARGIN,
                )
                extra = GAZE_GLARE_EXTRA_TOLERANCE if glare else 0.0
                yaw_thr = GAZE_YAW_THRESHOLD + extra
                pitch_thr = GAZE_PITCH_THRESHOLD + extra
                gaze_confirmed = (
                    abs(l2cs_gaze["yaw"])   > yaw_thr - GAZE_CONFIRMATION_MARGIN
                    or abs(l2cs_gaze["pitch"]) > pitch_thr - GAZE_CONFIRMATION_MARGIN
                )
                # Only expose as true if lookingAway is actually true to prevent frontend false positives
                result["headPoseConfirmedByGaze"] = gaze_confirmed and looking_away
        else:
            state.gaze_yaw.update(None)
            state.gaze_pitch.update(None)

        result["faceCovered"] = check_face_covered(face_crop, primary_face.get("landmarks", {}), primary_face["score"])
    else:
        state.gaze_yaw.update(None)
        state.gaze_pitch.update(None)
        result["lookAwayDurationMs"] = state.look_away_duration.update(False)

    result["processingTimeMs"] = _elapsed(t0)

    # Add snake_case aliases for frontend compatibility
    result["face_detected"] = result["faceDetected"]
    result["face_count"] = result["faceCount"]
    result["multiple_faces"] = result["multipleFaces"]
    result["multiple_people"] = result["multiplePeople"]
    result["face_covered"] = result["faceCovered"]
    result["looking_away"] = result["lookingAway"]
    result["glasses_glare_detected"] = result["glassesGlareDetected"]
    result["head_pose"] = result["headPose"]
    result["gaze_estimate"] = result["gazeEstimate"]
    result["head_pose_confirmed_by_gaze"] = result["headPoseConfirmedByGaze"]
    result["face_detector"] = result["faceDetector"]
    result["body_detected"] = result["bodyDetected"]
    result["body_count"] = result["bodyCount"]
    result["body_box"] = result["bodyBox"]
    result["multiple_bodies"] = result["multipleBodies"]
    result["partial_body_detected"] = result["partialBodyDetected"]
    result["detected_objects"] = result["detectedObjects"]
    result["phone_detected"] = result["phoneDetected"]
    result["secondary_device_detected"] = result["secondaryDeviceDetected"]
    result["book_detected"] = result["bookDetected"]
    result["multiple_faces_duration_ms"] = result["multipleFacesDurationMs"]
    result["multiple_people_duration_ms"] = result["multiplePeopleDurationMs"]
    result["face_missing_duration_ms"] = result["faceMissingDurationMs"]
    result["look_away_duration_ms"] = result["lookAwayDurationMs"]
    result["phone_duration_ms"] = result["phoneDurationMs"]

    return result


def analyze_audio(audio_data: str | bytes, session_id: str, models: Models) -> Dict:
    """Run VAD on a base64-encoded or raw PCM audio clip.

    Args:
        audio_data: Base64-encoded or raw bytes 16-bit mono PCM at 16 kHz.
        session_id: Used for temporal smoothing.
        models:     Loaded model container.

    Returns:
        {"multi_speaker": bool, "speech_confidence": float}
    """
    state = SESSION_STATES[session_id]
    logger = get_session_logger(session_id)

    if isinstance(audio_data, bytes):
        audio_bytes = audio_data
    else:
        audio_bytes = decode_base64_audio(audio_data)

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
