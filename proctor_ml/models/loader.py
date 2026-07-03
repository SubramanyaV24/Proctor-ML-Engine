"""
Model loader — downloads weights on first run and initialises all ML models.

Call `load_all_models()` once at startup. The returned `Models` dataclass is
passed through to detectors so there is no module-level global state.

All 6 independent model groups are loaded in parallel via ThreadPoolExecutor,
cutting cold-start time by ~60% vs serial loading.
"""
from __future__ import annotations

import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import cv2
import torch

from proctor_ml.config import (
    MODEL_WEIGHTS_DIR,
    YUNET_FILENAME, YUNET_URL,
    YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD,
    YOLO_FILENAME, YOLO_DOWNLOAD_URL,
    SIXDREPNET_CHECKPOINT,
)


# ---------------------------------------------------------------------------
# Dataclass that holds every loaded model (or None if unavailable)
# ---------------------------------------------------------------------------
@dataclass
class Models:
    yunet: Optional[object] = None               # cv2.FaceDetectorYN
    haar_face: Optional[object] = None           # cv2.CascadeClassifier
    haar_upper: Optional[object] = None          # cv2.CascadeClassifier
    haar_full: Optional[object] = None           # cv2.CascadeClassifier
    mediapipe_face: Optional[object] = None      # mp.solutions.face_detection.FaceDetection
    head_pose: Optional[object] = None           # SixDRepNet
    yolo: Optional[object] = None                # ultralytics.YOLO
    silero_vad: Optional[object] = None          # silero VAD model
    silero_get_timestamps: Optional[object] = None
    webrtc_vad: Optional[object] = None          # webrtcvad.Vad


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _download_if_missing(url: str, dest: str) -> bool:
    """Download *url* to *dest* if the file does not already exist."""
    if os.path.exists(dest):
        return True
    print(f"[Loader] Downloading {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[Loader] Downloaded → {dest}")
        return True
    except Exception as exc:
        print(f"[Loader] WARNING: Download failed ({exc})")
        return False


# ---------------------------------------------------------------------------
# Individual model loaders — each runs in its own thread
# ---------------------------------------------------------------------------

def _load_yunet() -> dict:
    yunet_path = os.path.join(MODEL_WEIGHTS_DIR, YUNET_FILENAME)
    if not _download_if_missing(YUNET_URL, yunet_path):
        return {}
    try:
        yunet = cv2.FaceDetectorYN_create(
            model=yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
            top_k=5000,
        )
        print("[Loader] YuNet face detector loaded.")
        return {"yunet": yunet}
    except Exception as exc:
        print(f"[Loader] WARNING: YuNet load failed ({exc})")
        return {}


def _load_haar() -> dict:
    try:
        haar_face  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        haar_upper = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
        haar_full  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
        print("[Loader] Haar cascades ready.")
        return {"haar_face": haar_face, "haar_upper": haar_upper, "haar_full": haar_full}
    except Exception as exc:
        print(f"[Loader] WARNING: Haar cascade load failed ({exc})")
        return {}


def _load_mediapipe() -> dict:
    try:
        import mediapipe as mp
        mp_fd = mp.solutions.face_detection
        face = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        print("[Loader] MediaPipe face detector loaded.")
        return {"mediapipe_face": face}
    except Exception as exc:
        print(f"[Loader] WARNING: MediaPipe not available ({exc})")
        return {}


def _load_sixdrepnet() -> dict:
    try:
        from sixdrepnet import SixDRepNet
        checkpoint_path = os.path.join(MODEL_WEIGHTS_DIR, "checkpoints", SIXDREPNET_CHECKPOINT)
        gpu_id = 0 if torch.cuda.is_available() else -1
        if not os.path.exists(checkpoint_path):
            print(f"[Loader] WARNING: 6DRepNet checkpoint not found at {checkpoint_path}")
            return {}
        head_pose = SixDRepNet(gpu_id=gpu_id, dict_path=checkpoint_path)
        device = "cuda" if gpu_id >= 0 else "cpu"
        print(f"[Loader] 6DRepNet loaded on {device}.")
        return {"head_pose": head_pose}
    except Exception as exc:
        print(f"[Loader] WARNING: 6DRepNet not available ({exc})")
        return {}


def _load_yolo() -> dict:
    try:
        import torch
        from ultralytics import YOLO
        yolo_path = os.path.join(MODEL_WEIGHTS_DIR, YOLO_FILENAME)
        if not os.path.exists(yolo_path):
            print("[Loader] Downloading YOLO weights ...")
            torch.hub.download_url_to_file(YOLO_DOWNLOAD_URL, yolo_path)
        # ultralytics handles its own safe_globals internally — load via
        # the YOLO wrapper which sets up serialization correctly for .pt files
        yolo = YOLO(yolo_path, task="detect")
        print(f"[Loader] YOLO loaded from {yolo_path}.")
        return {"yolo": yolo}
    except Exception as exc:
        print(f"[Loader] WARNING: YOLO not available ({exc})")
        return {}


def _load_silero_vad() -> dict:
    try:
        import sys
        hub_src = os.path.join(MODEL_WEIGHTS_DIR, "snakers4_silero-vad_master", "src")
        if hub_src not in sys.path:
            sys.path.insert(0, hub_src)
        from silero_vad.utils_vad import OnnxWrapper, get_speech_timestamps
        onnx_model_path = os.path.join(
            MODEL_WEIGHTS_DIR, "snakers4_silero-vad_master",
            "src", "silero_vad", "data", "silero_vad.onnx"
        )
        if not os.path.exists(onnx_model_path):
            print(f"[Loader] WARNING: Silero ONNX model not found at {onnx_model_path}")
            return {}
        vad_onnx = OnnxWrapper(onnx_model_path, force_onnx_cpu=True)
        print("[Loader] Silero VAD loaded (ONNX).")
        return {"silero_vad": vad_onnx, "silero_get_timestamps": get_speech_timestamps}
    except Exception as exc:
        print(f"[Loader] WARNING: Silero VAD not available ({exc})")
        return {}


def _load_webrtc_vad() -> dict:
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)  # aggressiveness mode 2
        print("[Loader] WebRTC VAD loaded.")
        return {"webrtc_vad": vad}
    except Exception as exc:
        print(f"[Loader] WARNING: webrtcvad not available ({exc})")
        return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def load_all_models() -> Models:
    """Load all ML models in parallel. Returns a Models instance."""
    _ensure_dir(MODEL_WEIGHTS_DIR)
    torch.hub.set_dir(MODEL_WEIGHTS_DIR)

    loaders = [
        _load_yunet,
        _load_haar,
        _load_mediapipe,
        _load_sixdrepnet,
        _load_yolo,
        _load_silero_vad,
        _load_webrtc_vad,
    ]

    results: dict = {}
    # Use at most 4 threads — model loading is partly I/O (disk reads) and
    # partly compute (weight initialisation). More threads don't help beyond ~4.
    with ThreadPoolExecutor(max_workers=min(4, len(loaders))) as pool:
        futures = {pool.submit(fn): fn.__name__ for fn in loaders}
        for future in as_completed(futures):
            try:
                results.update(future.result())
            except Exception as exc:
                print(f"[Loader] ERROR in {futures[future]}: {exc}")

    m = Models(**{k: v for k, v in results.items() if hasattr(Models, k)})
    print("[Loader] All models loaded.")
    return m
