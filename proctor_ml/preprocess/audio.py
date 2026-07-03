"""
Audio preprocessing utilities.
"""
from __future__ import annotations

import base64
from typing import Optional


def decode_base64_audio(b64_string: str) -> Optional[bytes]:
    """Decode a base64-encoded audio string (data-URI or raw) to raw PCM bytes.

    Expects 16-bit mono PCM at 16 kHz. Returns None on any error.
    """
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        return base64.b64decode(b64_string)
    except Exception as exc:
        print(f"[Preprocess] Audio decode error: {exc}")
        return None
