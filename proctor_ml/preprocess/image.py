"""
Image preprocessing utilities.
"""
from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np


def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """Decode a base64-encoded image (data-URI or raw) to a BGR NumPy array.

    The image is flipped horizontally so bounding boxes align with a
    CSS-mirrored front-end camera preview.

    Returns None on any decoding error.
    """
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        # Decode directly with OpenCV (native BGR) — eliminates PIL + RGB→BGR conversion
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Mirror to match the frontend's CSS-mirrored camera preview
        return cv2.flip(img, 1)
    except Exception as exc:
        print(f"[Preprocess] Image decode error: {exc}")
        return None


def decode_binary_image(img_bytes: bytes) -> Optional[np.ndarray]:
    """Decode a raw binary image (JPEG/WebP) to a BGR NumPy array.

    The image is flipped horizontally so bounding boxes align with a
    CSS-mirrored front-end camera preview.

    Returns None on any decoding error.
    """
    try:
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Mirror to match the frontend's CSS-mirrored camera preview
        return cv2.flip(img, 1)
    except Exception as exc:
        print(f"[Preprocess] Binary image decode error: {exc}")
        return None
