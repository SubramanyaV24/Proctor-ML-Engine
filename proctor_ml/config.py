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
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
)

SIXDREPNET_CHECKPOINT = "6DRepNet_300W_LP_AFLW2000.pth"

# SCRFD — opt-in (non-commercial InsightFace weights).
# Set SCRFD_ENABLED=0 in the environment to disable.
# Model files are downloaded from HuggingFace Hub on first start.
SCRFD_ENABLED: bool = os.getenv("SCRFD_ENABLED", "1") == "1"
# SCRFD-10GF is the recommended server-side balance (83.05 % hard AP, ~4.9 ms on 2080Ti).
# Use scrfd_2.5g_bnkps.onnx for lighter CPU deployments.
SCRFD_MODEL_NAME: str = os.getenv("SCRFD_MODEL_NAME", "scrfd_10g_bnkps.onnx")
SCRFD_HF_REPO: str = os.getenv("SCRFD_HF_REPO", "DIAMONIK7777/antelopev2")
SCRFD_SCORE_THRESHOLD: float = float(os.getenv("SCRFD_SCORE_THRESHOLD", "0.5"))
SCRFD_NMS_THRESHOLD: float = float(os.getenv("SCRFD_NMS_THRESHOLD", "0.4"))
SCRFD_INPUT_SIZE: int = int(os.getenv("SCRFD_INPUT_SIZE", "640"))  # px (height=width)

# L2CS-Net — appearance-based gaze (ResNet-50, Gaze360-trained).
# More robust to eyeglass glare than iris/pupil geometry methods.
# Set L2CS_ENABLED=0 to disable.
L2CS_ENABLED: bool = os.getenv("L2CS_ENABLED", "1") == "1"
L2CS_CHECKPOINT: str = os.getenv("L2CS_CHECKPOINT", "L2CSNet_gaze360.pkl")
L2CS_HF_REPO: str = os.getenv("L2CS_HF_REPO", "tianfxc/l2cs")
L2CS_HF_FILENAME: str = os.getenv("L2CS_HF_FILENAME", "L2CSNet_gaze360.pkl")
L2CS_IMAGE_SIZE: int = int(os.getenv("L2CS_IMAGE_SIZE", "448"))  # input resolution

# Dual-signal gaze confirmation.
# When True: 'looking away' fires only if head pose AND L2CS gaze agree.
# When False: head pose alone is sufficient (legacy behaviour).
GAZE_CONFIRMATION_REQUIRED: bool = os.getenv("GAZE_CONFIRMATION_REQUIRED", "0") == "1"
# How close the L2CS gaze must be to the threshold to count as confirming.
# E.g. with yaw threshold=30 and margin=8: L2CS gaze must exceed 22° to confirm.
GAZE_CONFIRMATION_MARGIN: float = float(os.getenv("GAZE_CONFIRMATION_MARGIN", "8.0"))

# ---------------------------------------------------------------------------
# Inference thresholds
# ---------------------------------------------------------------------------
YUNET_SCORE_THRESHOLD: float = float(os.getenv("YUNET_SCORE_THRESHOLD", "0.3"))
YUNET_NMS_THRESHOLD: float = float(os.getenv("YUNET_NMS_THRESHOLD", "0.3"))
FACE_SCORE_MIN: float = float(os.getenv("FACE_SCORE_MIN", "0.35"))
MEDIAPIPE_MIN_CONFIDENCE: float = float(os.getenv("MEDIAPIPE_MIN_CONFIDENCE", "0.5"))

YOLO_PERSON_CONFIDENCE: float = float(os.getenv("YOLO_PERSON_CONFIDENCE", "0.75"))
# Lower confidence threshold used ONLY to detect additional/partial people in
# the background (e.g. a hand or torso partially entering the frame).  The
# primary candidate is still required at YOLO_PERSON_CONFIDENCE.
YOLO_SECONDARY_PERSON_CONFIDENCE: float = float(os.getenv("YOLO_SECONDARY_PERSON_CONFIDENCE", "0.70"))
# Minimum IoU below which a secondary person box is considered spatially distinct
# from the primary candidate's body (avoids double-counting the same person)
YOLO_SECONDARY_IOU_MAX: float = float(os.getenv("YOLO_SECONDARY_IOU_MAX", "0.30"))
YOLO_NMS_IOU: float = float(os.getenv("YOLO_NMS_IOU", "0.45"))

# Per-label confidence thresholds for suspicious object detection
OBJECT_CONFIDENCE_THRESHOLDS: dict = {
    "cell phone": float(os.getenv("CONF_CELL_PHONE", "0.45")),
    "remote":     float(os.getenv("CONF_REMOTE",     "0.45")),
    "laptop":     float(os.getenv("CONF_LAPTOP",     "0.50")),
    "book":       float(os.getenv("CONF_BOOK",       "0.50")),
}

# Minimum phone bounding-box area as a fraction of the frame
PHONE_MIN_AREA_FRAC: float = float(os.getenv("PHONE_MIN_AREA_FRAC", "0.001"))
PHONE_MAX_ASPECT: float = float(os.getenv("PHONE_MAX_ASPECT", "10.0"))

# Gaze thresholds (degrees)
GAZE_YAW_THRESHOLD: float = float(os.getenv("GAZE_YAW_THRESHOLD", "20.0"))
GAZE_PITCH_THRESHOLD: float = float(os.getenv("GAZE_PITCH_THRESHOLD", "20.0"))
# Extra tolerance (degrees) added on top of the above when glasses glare is detected
GAZE_GLARE_EXTRA_TOLERANCE: float = float(os.getenv("GAZE_GLARE_EXTRA_TOLERANCE", "5.0"))

# Face-covered heuristic
FACE_COVERED_SCORE_MIN: float = float(os.getenv("FACE_COVERED_SCORE_MIN", "0.3"))
FACE_COVERED_STD_MIN: float = float(os.getenv("FACE_COVERED_STD_MIN", "15.0"))
FACE_COVERED_NOSE_MOUTH_RATIO: float = float(os.getenv("FACE_COVERED_NOSE_MOUTH_RATIO", "0.08"))

# ---------------------------------------------------------------------------
# Face deduplication / filtering (false-positive reduction)
# ---------------------------------------------------------------------------
# Discard faces smaller than this fraction of the total frame area
FACE_MIN_AREA_FRAC: float = float(os.getenv("FACE_MIN_AREA_FRAC", "0.005"))
# Discard faces whose box touches within this fraction of any image edge
# (catches partial reflections near screen borders)
FACE_EDGE_MARGIN_FRAC: float = float(os.getenv("FACE_EDGE_MARGIN_FRAC", "0.04"))
# IoU above which two detections are considered duplicates; keep highest score
FACE_IOU_DEDUP_THRESHOLD: float = float(os.getenv("FACE_IOU_DEDUP_THRESHOLD", "0.40"))

# Multiple-faces smoother tuning
# Window raised 3→5; threshold raised 2→4 (≈80 % majority)
MULTI_FACE_WINDOW: int = int(os.getenv("MULTI_FACE_WINDOW", "5"))
MULTI_FACE_THRESHOLD: int = int(os.getenv("MULTI_FACE_THRESHOLD", "4"))
# Multiple-people smoother — slightly more permissive than multi-face
# (3-of-5): partial bodies can flicker so we use a lower threshold
MULTI_PEOPLE_WINDOW: int = int(os.getenv("MULTI_PEOPLE_WINDOW", "5"))
MULTI_PEOPLE_THRESHOLD: int = int(os.getenv("MULTI_PEOPLE_THRESHOLD", "3"))

# ---------------------------------------------------------------------------
# Eyeglass glare detection (HSV on eye ROIs)
# ---------------------------------------------------------------------------
# Eye ROI half-size in pixels around the landmark point
GLASS_EYE_ROI_HALF: int = int(os.getenv("GLASS_EYE_ROI_HALF", "12"))
# HSV Saturation ceiling — low saturation implies washed-out / glare region
GLASSES_GLARE_SATURATION_MAX: int = int(os.getenv("GLASSES_GLARE_SATURATION_MAX", "40"))
# HSV Value floor — high brightness confirms glare rather than dark occlusion
GLASSES_GLARE_VALUE_MIN: int = int(os.getenv("GLASSES_GLARE_VALUE_MIN", "200"))

# ---------------------------------------------------------------------------
# Audio / VAD Confirmation
# ---------------------------------------------------------------------------
WHISPER_ENABLED: bool = os.getenv("WHISPER_ENABLED", "1") == "1"
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny.en")

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
MAX_PAYLOAD_BYTES: int = int(os.getenv("MAX_PAYLOAD_BYTES", str(5 * 1024 * 1024)))  # 5 MB
CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
