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
    SCRFD_ENABLED, SCRFD_MODEL_NAME, SCRFD_HF_REPO,
    SCRFD_SCORE_THRESHOLD, SCRFD_NMS_THRESHOLD, SCRFD_INPUT_SIZE,
    L2CS_ENABLED, L2CS_CHECKPOINT, L2CS_HF_REPO, L2CS_HF_FILENAME,
    WHISPER_ENABLED, WHISPER_MODEL,
)


# ---------------------------------------------------------------------------
# Dataclass that holds every loaded model (or None if unavailable)
# ---------------------------------------------------------------------------
@dataclass
class Models:
    yunet: Optional[object] = None               # cv2.FaceDetectorYN
    scrfd: Optional[object] = None               # SCRFDDetector (onnxruntime wrapper)
    haar_face: Optional[object] = None           # cv2.CascadeClassifier
    haar_upper: Optional[object] = None          # cv2.CascadeClassifier
    haar_full: Optional[object] = None           # cv2.CascadeClassifier
    mediapipe_face: Optional[object] = None      # mp.solutions.face_detection.FaceDetection
    head_pose: Optional[object] = None           # SixDRepNet
    l2cs_gaze: Optional[object] = None           # L2CS-Net gaze model
    yolo: Optional[object] = None                # ultralytics.YOLO
    silero_vad: Optional[object] = None          # silero VAD model
    silero_get_timestamps: Optional[object] = None
    webrtc_vad: Optional[object] = None          # webrtcvad.Vad
    whisper: Optional[object] = None             # Faster Whisper model


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


def _load_scrfd() -> dict:
    """Load SCRFD face detector from a local ONNX file or download from HuggingFace Hub.

    Only runs when SCRFD_ENABLED=1. The ONNX weights are licensed for
    non-commercial research use (InsightFace); check before enabling in production.
    """
    if not SCRFD_ENABLED:
        return {}
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(f"[Loader] WARNING: SCRFD unavailable — missing dependency ({exc})")
        return {}

    local_path = os.path.join(MODEL_WEIGHTS_DIR, SCRFD_MODEL_NAME)
    if not os.path.exists(local_path):
        print(f"[Loader] Downloading {SCRFD_MODEL_NAME} from {SCRFD_HF_REPO} ...")
        try:
            local_path = hf_hub_download(
                repo_id=SCRFD_HF_REPO,
                filename=SCRFD_MODEL_NAME,
                local_dir=MODEL_WEIGHTS_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"[Loader] SCRFD downloaded → {local_path}")
        except Exception as exc:
            print(f"[Loader] WARNING: SCRFD download failed ({exc})")
            return {}

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 2
        sess_opts.intra_op_num_threads = 2
        session = ort.InferenceSession(
            local_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # Wrap in a small helper to keep the detector interface uniform
        from proctor_ml.detectors._scrfd import SCRFDDetector
        detector = SCRFDDetector(
            session=session,
            input_size=SCRFD_INPUT_SIZE,
            score_threshold=SCRFD_SCORE_THRESHOLD,
            nms_threshold=SCRFD_NMS_THRESHOLD,
        )
        print(f"[Loader] SCRFD ({SCRFD_MODEL_NAME}) loaded.")
        return {"scrfd": detector}
    except Exception as exc:
        print(f"[Loader] WARNING: SCRFD load failed ({exc})")
        return {}


def _load_l2cs() -> dict:
    """Load L2CS-Net gaze model (ResNet-50, Gaze360-trained).

    Downloads L2CSNet_gaze360.pkl from HuggingFace Hub on first run (~100 MB).
    Set L2CS_ENABLED=0 to disable; service falls back to 6DRepNet head-pose only.
    """
    if not L2CS_ENABLED:
        return {}
    try:
        import torch
        from l2cs import Pipeline as L2CSPipeline  # noqa: F401 — existence check
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(f"[Loader] WARNING: L2CS-Net unavailable — missing dependency ({exc})")
        return {}

    local_path = os.path.join(MODEL_WEIGHTS_DIR, L2CS_CHECKPOINT)
    if not os.path.exists(local_path):
        print(f"[Loader] Downloading {L2CS_HF_FILENAME} from {L2CS_HF_REPO} ...")
        try:
            local_path = hf_hub_download(
                repo_id=L2CS_HF_REPO,
                filename=L2CS_HF_FILENAME,
                local_dir=MODEL_WEIGHTS_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"[Loader] L2CS-Net weights downloaded → {local_path}")
        except Exception as exc:
            print(f"[Loader] WARNING: L2CS-Net download failed ({exc})")
            return {}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from proctor_ml.detectors._l2cs import L2CSGazeModel
        gaze_model = L2CSGazeModel(checkpoint_path=local_path, device=device)
        print(f"[Loader] L2CS-Net gaze model loaded on {device}.")
        return {"l2cs_gaze": gaze_model}
    except Exception as exc:
        print(f"[Loader] WARNING: L2CS-Net load failed ({exc})")
        return {}


def _load_yolo() -> dict:
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        
        yolo_path = os.path.join(MODEL_WEIGHTS_DIR, YOLO_FILENAME)
        if not os.path.exists(yolo_path):
            print("[Loader] Downloading YOLO weights ...")
            torch.hub.download_url_to_file(YOLO_DOWNLOAD_URL, yolo_path)

        # PyTorch 2.6 defaults to weights_only=True which breaks YOLO.
        # Ultralytics internally uses torch.load so we patch it temporarily.
        _original_load = torch.load
        def _safe_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)
        
        try:
            torch.load = _safe_load
            yolo = YOLO(yolo_path, task="detect")
        finally:
            torch.load = _original_load
            
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


def _load_whisper() -> dict:
    if not WHISPER_ENABLED:
        return {}
    try:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type, download_root=MODEL_WEIGHTS_DIR)
        print(f"[Loader] Faster Whisper ({WHISPER_MODEL}) loaded on {device} ({compute_type}).")
        return {"whisper": model}
    except Exception as exc:
        print(f"[Loader] WARNING: Faster Whisper not available ({exc})")
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
        _load_scrfd,
        _load_l2cs,
        _load_yolo,
        _load_silero_vad,
        _load_webrtc_vad,
        _load_whisper,
    ]

    results: dict = {}
    # Use 1 thread to avoid OOM spikes on constrained hardware (SIGKILL 137).
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(fn): fn.__name__ for fn in loaders}
        for future in as_completed(futures):
            try:
                results.update(future.result())
            except Exception as exc:
                print(f"[Loader] ERROR in {futures[future]}: {exc}")

    m = Models(**{k: v for k, v in results.items() if hasattr(Models, k)})
    print("[Loader] All models loaded.")
    return m
