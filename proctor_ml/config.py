"""
Central configuration for the Proctor-ML-Engine.

All values are read from environment variables so the service stays
12-factor-app compliant and free of hard-coded credentials.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model weights are stored here; the service downloads missing files on startup.
MODEL_WEIGHTS_DIR = os.getenv(
    "MODEL_WEIGHTS_DIR",
    os.path.join(BASE_DIR, "model_weights"),
)

LOGS_DIR = os.getenv(
    "LOGS_DIR",
    os.path.join(BASE_DIR, "logs"),
)

# ---------------------------------------------------------------------------
# Model weight filenames / download URLs
# ---------------------------------------------------------------------------
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

YOLO_FILENAME = "yolo11m.pt"
YOLO_DOWNLOAD_URL = (
    "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolo11m.pt"
)

SIXDREPNET_CHECKPOINT = "6DRepNet_300W_LP_AFLW2000.pth"

# ---------------------------------------------------------------------------
# Inference thresholds
# ---------------------------------------------------------------------------
YUNET_SCORE_THRESHOLD: float = float(os.getenv("YUNET_SCORE_THRESHOLD", "0.3"))
YUNET_NMS_THRESHOLD: float = float(os.getenv("YUNET_NMS_THRESHOLD", "0.3"))
FACE_SCORE_MIN: float = float(os.getenv("FACE_SCORE_MIN", "0.35"))
MEDIAPIPE_MIN_CONFIDENCE: float = float(os.getenv("MEDIAPIPE_MIN_CONFIDENCE", "0.5"))

YOLO_PERSON_CONFIDENCE: float = float(os.getenv("YOLO_PERSON_CONFIDENCE", "0.65"))
YOLO_NMS_IOU: float = float(os.getenv("YOLO_NMS_IOU", "0.45"))

# Per-label confidence thresholds for suspicious object detection
OBJECT_CONFIDENCE_THRESHOLDS: dict = {
    "cell phone": float(os.getenv("CONF_CELL_PHONE", "0.70")),
    "laptop":     float(os.getenv("CONF_LAPTOP",     "0.70")),
    "book":       float(os.getenv("CONF_BOOK",       "0.70")),
    "tv":         float(os.getenv("CONF_TV",         "0.70")),
}

# Minimum phone bounding-box area as a fraction of the frame
PHONE_MIN_AREA_FRAC: float = float(os.getenv("PHONE_MIN_AREA_FRAC", "0.005"))
PHONE_MAX_ASPECT: float = float(os.getenv("PHONE_MAX_ASPECT", "3.0"))

# Gaze thresholds (degrees)
GAZE_YAW_THRESHOLD: float = float(os.getenv("GAZE_YAW_THRESHOLD", "25.0"))
GAZE_PITCH_THRESHOLD: float = float(os.getenv("GAZE_PITCH_THRESHOLD", "20.0"))

# Face-covered heuristic
FACE_COVERED_SCORE_MIN: float = float(os.getenv("FACE_COVERED_SCORE_MIN", "0.3"))
FACE_COVERED_STD_MIN: float = float(os.getenv("FACE_COVERED_STD_MIN", "15.0"))
FACE_COVERED_NOSE_MOUTH_RATIO: float = float(os.getenv("FACE_COVERED_NOSE_MOUTH_RATIO", "0.08"))

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
MAX_PAYLOAD_BYTES: int = int(os.getenv("MAX_PAYLOAD_BYTES", str(5 * 1024 * 1024)))  # 5 MB
CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
